import argparse
from datetime import datetime
from models import PointFormer, PointRNN, PointLSTM
import numpy as np
import pathlib
import time as t
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import torch_np_fix_seed, make_datasets
from utils.losses import Window_Loss
root_path = pathlib.Path("")
torch_np_fix_seed(1111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--point_num", default=60, type=int)
parser.add_argument("--future_seconds", default=60, type=int)
parser.add_argument("--load_name", type=str)
parser.add_argument("--best_save_name", required=True, type=str)
parser.add_argument("--last_save_name", required=True, type=str)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--num_epoch", default=100, type=int)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--start_dates", required=True, type=str, nargs="*")
parser.add_argument("--end_dates", required=True, type=str, nargs="*")
parser.add_argument("--symbols", required=True, type=str, nargs="*")
args = parser.parse_args()

def train(model, future_seconds, 
          load_path, best_save_path, last_save_path, learning_rate,
          train_dataset, valid_dataset, 
          batch_size, num_epoch, num_workers, window_size,
          device):
    print(f"device: {device}")
    torch.backends.cudnn.benchmark = True
    if load_path is not None:
        checkpoint = torch.load(load_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        START_EPOCH = checkpoint["current_epoch"]
        END_EPOCH = START_EPOCH + num_epoch
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
        best_loss = checkpoint["best_loss"]
    else:
        END_EPOCH = num_epoch
        START_EPOCH = 0
        train_losses = []
        valid_losses = []
        best_loss = 1e+10
    train_n = len(train_dataset)
    valid_n = len(valid_dataset)
    params = model.parameters()
    criterion = Window_Loss(window_size, future_seconds, device)
    optimizer = optim.AdamW(params, lr=learning_rate)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)
    model.to(device)
    for epoch in range(START_EPOCH, END_EPOCH):
        train_loss = 0
        train_time_start = t.time()
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss) / train_n
            if i % 100 == 99:
                print(float(loss))
            inputs = inputs.to("cpu")
            targets = targets.to("cpu")
            outputs = outputs.to("cpu")
        train_loss = np.sqrt(train_loss)
        train_losses.append(train_loss)
        train_time_end = t.time()
        valid_loss = 0
        valid_time_start = t.time()
        model.eval()
        with torch.no_grad():
            for j, batch in enumerate(valid_dataloader):
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if j % 100 == 99:
                    print(float(loss))
                valid_loss += float(loss) / valid_n
                inputs = inputs.to("cpu")
                targets = targets.to("cpu")
                outputs = outputs.to("cpu")
        valid_loss = np.sqrt(valid_loss)
        valid_losses.append(valid_loss)
        valid_time_end = t.time()
        train_time_total = train_time_end - train_time_start
        valid_time_total = valid_time_end - valid_time_start
        total_time = train_time_total + valid_time_total
        print(f"epoch:{epoch+1}/{END_EPOCH}" + 
              f" [loss]tra:{train_loss:.6f} val:{valid_loss:.6f}"
              f" [time]total{total_time:.2f}sec" +
              f" tra{train_time_total:.2f}sec val{valid_time_total:.2f}sec")
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_path = best_save_path
        else:
            save_path = last_save_path
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "current_epoch": epoch+1,
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "best_loss": best_loss
            },
            str(save_path)
        )
        print(f"model saved to >> {str(save_path)}")
        print()
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
    datasets, _ = make_datasets(symbols, start_dates, end_dates,
                                point_num, future_seconds)
    train_dataset = datasets[0]
    valid_dataset = datasets[1]
    model_name = args.model
    if model_name == "PointFormer":
        model = PointFormer(coin_num=coin_num,
                            feature_num=4, 
                            point_num=point_num,
                            d_model=4, d_ff=2, d_ff2=2, 
                            future_seconds=future_seconds,
                            nhead1=4, device=device)
    elif model_name == "PointRNN":
        model = PointRNN(coin_num=coin_num,
                         feature_num=4, 
                         point_num=point_num, future_seconds=future_seconds,
                         hidden_size=12, device=device)
    elif model_name =="PointLSTM":
        model = PointLSTM(coin_num=coin_num,
                          feature_num=4, point_num=point_num, future_seconds=future_seconds,
                          hidden_size=12, device=device)
    else:
        try:
            raise ValueError("model name must be PointFormer or PointRNN or PointLSTM.")
        except ValueError as e:
            print(e)
    checkpoints_path = root_path / "checkpoints"
    load_name = args.load_name
    if load_name is not None:
        load_path = checkpoints_path / load_name
    else:
        load_path = None
    best_save_name = args.best_save_name
    best_save_path = checkpoints_path / best_save_name
    last_save_name = args.last_save_name
    last_save_path = checkpoints_path / last_save_name
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    num_workers = args.num_workers
    window_size = args.window_size
    train(model, future_seconds, 
          load_path, best_save_path, last_save_path, learning_rate,
          train_dataset, valid_dataset, 
          batch_size, num_epoch, num_workers, window_size,
          device)
