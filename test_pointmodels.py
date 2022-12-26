import argparse
from datetime import datetime, timedelta, time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from models import PointFormer, PointRNN, PointLSTM
import numpy as np
import pandas as pd
import pathlib
import torch
from torch.utils.data import DataLoader
root_path = pathlib.Path("")
from utils.datasets import make_datasets
from utils.losses import Window_Loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--point_num", default=60, type=int)
parser.add_argument("--future_seconds", default=60, type=int)
parser.add_argument("--load_name", required=True, type=str)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--start_dates", required=True, type=str, nargs="*")
parser.add_argument("--end_dates", required=True, type=str, nargs="*")
parser.add_argument("--symbols", required=True, type=str, nargs="*")
parser.add_argument("--total_loss_save_name", required=True, type=str)
parser.add_argument("--loss_list_save_name", required=True, type=str)
parser.add_argument("--start_datetime", required=True, type=str)
parser.add_argument("--sub_start_datetime", required=True, type=str)
parser.add_argument("--sub_end_datetime", required=True, type=str)
parser.add_argument("--pred_times_name", required=True, type=str)
args = parser.parse_args()

def test(model, test_dataset, future_seconds, window_size, batch_size, num_workers, 
         device, total_loss_save_path, loss_list_save_path, cut_time_num=3):
    test_n = len(test_dataset)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)
    model.eval()
    model.to(device)
    len_data = future_seconds // cut_time_num
    criterion_all = Window_Loss(window_size, future_seconds, device)
    criterion = Window_Loss(window_size, len_data, device)
    total_loss = 0
    loss_list = [0] * cut_time_num
    for batch in test_dataloader:
        with torch.no_grad():
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion_all(outputs, targets)
            total_loss += float(loss) / test_n
            for i in range(cut_time_num):
                target = targets[:,:,i*cut_time_num:i*cut_time_num+len_data]
                output = outputs[:,:,i*cut_time_num:i*cut_time_num+len_data]
                loss = criterion(output, target)
                loss_list[i] += float(loss) / test_n
            inputs = inputs.cpu()
            targets = targets.cpu()
    total_loss = np.sqrt(total_loss)
    with open(str(total_loss_save_path), "w") as f:
        f.write(str(total_loss))
    loss_list = [np.sqrt(loss) for loss in loss_list]
    with open(str(loss_list_save_path), "w") as f:
        for loss in loss_list:
            f.write(str(loss))
            f.write("\n")
    return 

def make_prediction(model, test_pointprocess_dfs, 
                    start_datetime, start_sec, end_sec,
                    sub_start_datetime, sub_end_datetime,
                    symbols, point_num, feature_num, future_seconds,
                    result_path, pred_times_name, save_imgs_path,
                    device):
    coin_num = len(symbols)
    times_idxes = [start_datetime + timedelta(seconds=sec) \
                   for sec in range(int(end_sec-start_sec))]
    target_dfs = []
    if not result_path.exists():
        result_path.mkdir(parents=True)
    for c in range(coin_num):
        coin_name = symbols[c]
        test_pp_df = test_pointprocess_dfs[c]
        test_pp_df = test_pp_df[
            str(start_datetime+timedelta(seconds=future_seconds)):\
            str(start_datetime+timedelta(seconds=end_sec-start_sec-future_seconds*2))
        ]
        target_df = test_pp_df["volume"]
        target_df.index = pd.to_datetime(target_df.index)
        target_df.rename("volume")
        target_csv_path = result_path / f"target_df_{coin_name}.csv"
        target_df.to_csv(str(target_csv_path), index=True)
        target_dfs.append(target_df)
    pred_times = np.zeros([int(end_sec-start_sec),coin_num])
    model.to(device)
    model.eval()
    for sec in range(int(end_sec-start_sec-future_seconds)):
        input_arr = np.zeros([coin_num, point_num, feature_num])
        for i, df in enumerate(test_pointprocess_dfs):
            past_arr = df[df["seconds"]<=start_sec+sec].values[-point_num:,:]
            past_arr[:,0] -= (start_sec+sec)
            input_arr[i,:,:] = past_arr
        input_tensor = torch.from_numpy(input_arr.astype(np.float32))
        input_tensor = input_tensor.unsqueeze(0).to(device)
        outputs = model(input_tensor)
        outputs = outputs.detach().cpu().numpy()[0] # (coin_num, future_seconds)
        for c in range(coin_num):
            output = outputs[c,:]
            pred_times[sec:sec+future_seconds,c] += output / future_seconds
    pred_times = pred_times[future_seconds:-future_seconds,:]
    times_idxes = times_idxes[future_seconds:-future_seconds]
    pred_times = pd.DataFrame(pred_times, columns=symbols, index=times_idxes)
    pred_times_csv_path = result_path / pred_times_name
    pred_times.to_csv(str(pred_times_csv_path), index=True)
    xfmt = mdates.DateFormatter("%m/%d\n%H:%M")
    xloc = mdates.DayLocator()
    fig = plt.figure(figsize=(8,20), dpi=100, facecolor="w")
    for i in range(coin_num):
        ax1 = fig.add_subplot(3,1,i+1)
        ax2 = ax1.twinx()
        target_df = target_dfs[i]
        symbol = symbols[i]
        ax1.bar(target_df.index, target_df, color="black", width=0.002)
        ax2.plot(pred_times.index, pred_times[symbol], color="blue",
                 linewidth=0.5, label="intensity")
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.xaxis.set_major_locator(xloc)
        ax1.xaxis.set_major_formatter(xfmt)   
        ax1.set_ylabel("Volume (USD)")
        ax1.set_title(symbol)
        ax2.set_ylim(np.min(pred_times[symbol]), 
                     np.max(pred_times[symbol]))
        ax2.legend()
        ax2.set_ylabel("intensity")
    prediction_img_path = save_imgs_path / pred_times_name
    plt.savefig(str(prediction_img_path))
    plt.close(fig)
    xloc = mdates.HourLocator()
    sub_pred_times = pred_times[(sub_start_datetime < pred_times.index) & \
                                (pred_times.index < sub_end_datetime)]
    fig = plt.figure(figsize=(8,20), dpi=100, facecolor="w")
    for i in range(coin_num):
        ax1 = fig.add_subplot(3,1,i+1)
        ax2 = ax1.twinx()
        target_df = target_dfs[i]
        sub_target_df = target_df[(sub_start_datetime < target_df.index) & \
                                  (target_df.index <  sub_end_datetime)]
        symbol = symbols[i]
        ax1.bar(sub_target_df.index, sub_target_df, color="black", width=0.001)
        ax2.plot(sub_pred_times.index, sub_pred_times[symbol], color="blue",
                 linewidth=0.5, label="intensity")
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.xaxis.set_major_locator(xloc)
        ax1.xaxis.set_major_formatter(xfmt)   
        ax1.set_ylabel("Volume (USD)")
        ax1.set_title(symbol)
        ax2.set_ylim(np.min(sub_pred_times[symbol]), 
                     np.max(sub_pred_times[symbol]))
        ax2.legend()
        ax2.set_ylabel("intensity")
    sub_prediction_img_path = save_imgs_path / f"sub_{pred_times_name}"
    plt.savefig(str(sub_prediction_img_path))
    plt.close(fig)
    return 

if __name__ == "__main__":
    start_dates = args.start_dates
    start_dates = [datetime.strptime(x, "%Y/%m/%d").date() for x in start_dates]
    end_dates = args.end_dates
    end_dates = [datetime.strptime(x, "%Y/%m/%d").date() for x in end_dates]
    symbols = args.symbols
    coin_num = len(symbols)
    point_num = args.point_num
    future_seconds = args.future_seconds
    datasets, all_dfs = make_datasets(symbols, start_dates, end_dates,
                                      point_num, future_seconds)
    test_dataset = datasets[0]
    test_pointprocess_dfs = all_dfs[0]
    model_name = args.model
    if model_name == "PointFormer":
        model = PointFormer(coin_num=coin_num,
                            feature_num=4, 
                            point_num=point_num,
                            d_model=4, d_ff=12, d_ff2=12, 
                            future_seconds=future_seconds,
                            nhead1=4, device=device)
    elif model_name == "PointRNN":
        model = PointRNN(coin_num=coin_num,
                         feature_num=4, 
                         point_num=point_num, future_seconds=future_seconds,
                         hidden_size=12, device=device)
    elif model_name =="PointLSTM":
        pointlstm = PointLSTM(coin_num=coin_num,
                              feature_num=4, point_num=point_num, future_seconds=future_seconds,
                              hidden_size=12, device=device)
    else:
        try:
            raise ValueError("model name must be PointFormer, PointRNN or PointLSTM.")
        except ValueError as e:
            print(e)
    checkpoints_path = root_path / "checkpoints"
    load_name = args.load_name
    load_path = checkpoints_path / load_name
    checkpoint = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    window_size = args.window_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    total_loss_save_name = args.total_loss_save_name
    loss_list_save_name = args.loss_list_save_name
    result_path = root_path / "result"
    if not result_path.exists():
        result_path.mkdir(parents=True)
    total_loss_save_path = result_path / total_loss_save_name
    loss_list_save_path = result_path / loss_list_save_name
    test(model, test_dataset, future_seconds,
         window_size, batch_size, num_workers, device,
         total_loss_save_path, loss_list_save_path)
    start_datetime = args.start_datetime
    start_datetime = datetime.strptime(start_datetime, "%Y/%m/%d/%H:%M:%S")
    start_datetime -= timedelta(seconds=future_seconds)
    start_sec = (start_datetime - datetime.combine(start_dates[0], time())).total_seconds()
    end_sec = start_sec + 24*60*60 + future_seconds
    sub_start_datetime = args.sub_start_datetime
    sub_start_datetime = datetime.strptime(sub_start_datetime, 
                                           "%Y/%m/%d/%H:%M:%S")
    sub_end_datetime = args.sub_end_datetime
    sub_end_datetime = datetime.strptime(sub_end_datetime, 
                                         "%Y/%m/%d/%H:%M:%S")
    save_imgs_path = root_path / "images"
    pred_times_name = args.pred_times_name
    make_prediction(model, test_pointprocess_dfs, 
                    start_datetime, start_sec, end_sec,
                    sub_start_datetime, sub_end_datetime,
                    symbols, point_num, 4, future_seconds,
                    result_path, pred_times_name, save_imgs_path,
                    device)