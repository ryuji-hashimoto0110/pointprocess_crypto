import argparse
from datetime import datetime, timedelta, time
import gzip
import numpy as np
import pandas as pd
import pathlib
from urllib import request
import time as t
import torch
from torch.utils.data import Dataset
root_path = pathlib.Path("")
parser = argparse.ArgumentParser()
parser.add_argument("--start_dates", required=True, type=str, args="*")
parser.add_argument("--end_dates", required=True, type=str, args="*")
parser.add_argument("--symbols", required=True, type=str, nargs="*")
parser.add_argument("--v_name", default="size", type=str)
parser.add_argument("--v_thresholds", required=True, type=int, nargs="*")
args = parser.parse_args()
def torch_np_fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
torch_np_fix_seed(1111)

def download(url, filepath):
    request.urlretrieve(url, filepath)
    with gzip.open(filepath, "rt") as f:
        df = pd.read_csv(f)
    filepath.unlink()
    return df

def bybit_load_contract_data(start_date, end_date, symbol):
    baseurl = f"https://public.bybit.com/trading/{symbol}/"
    contract_df = pd.DataFrame()
    days_num = (end_date - start_date).days
    for i in range(days_num+1):
        date = start_date + timedelta(i)
        date_str = date.strftime("%Y-%m-%d")
        filepath = pathlib.Path(f"{date_str}.csv")
        dlurl = baseurl + f"{symbol}{date_str}.csv.gz"
        df2 = download(dlurl, filepath)
        df2.sort_values(by="timestamp", ascending=True, inplace=True) 
        contract_df = pd.concat([contract_df, df2])
        t.sleep(0.1)
    contract_df["timestamp"] = pd.to_datetime(contract_df["timestamp"], 
                                              unit="s")
    contract_df.set_index("timestamp", inplace=True)
    return contract_df

def make_pointprocess_from_contract_data(contract_df, 
                                         start_datetime, end_datetime,
                                         v_name, v_threshold):
    contract_df_selected = contract_df[contract_df[v_name]>v_threshold]
    cols = ["timestamp", "seconds", "hour", "volume", "price"]
    pointprocess_df = pd.DataFrame(index=[], columns=cols)
    pod = (end_datetime - start_datetime).total_seconds() # period of time
    t_ = 0
    for i in range(len(contract_df_selected)):
        dt_now = contract_df_selected.index[i]
        t = (dt_now - start_datetime).total_seconds()
        h = dt_now.hour + dt_now.minute/60 + dt_now.second/2600
        if t < 0:
            continue
        elif t > pod:
            break
        v = contract_df_selected[v_name][i]
        p = contract_df_selected["price"][i]
        record = pd.Series([dt_now, t,h,v,p], index=cols)
        pointprocess_df = pointprocess_df.append(record, ignore_index=True)
    pointprocess_df.set_index("timestamp", inplace=True)
    return pointprocess_df

def save_pointprocess_dfs(start_dates, end_dates,
                          symbols, v_name, v_threshold_dic,
                          save_csvs_path):
    for symbol in symbols:
        for i in range(len(start_dates)):
            start_date = start_dates[i]
            end_date = end_dates[i]
            contract_df = bybit_load_contract_data(start_date, end_date, 
                                                   symbol)
            start_datetime = datetime.combine(start_date, time())
            end_datetime = datetime.combine(end_date+timedelta(days=1), time())
            pointprocess_df = make_pointprocess_from_contract_data(contract_df, 
                                                        start_datetime,
                                                        end_datetime,
                                                        v_name,
                                                        v_threshold_dic[symbol])
            contract_csv_name = f"contract_{symbol}_{str(start_date)}_{str(end_date)}.csv"                                          
            pointprocess_csv_name = f"pointprocess_{symbol}_{str(start_date)}_{str(end_date)}.csv"
            if not save_csvs_path.exists():
                save_csvs_path.mkdir(parents=True)
            contract_csv_path = save_csvs_path / contract_csv_name  
            pointprocess_csv_path = save_csvs_path / pointprocess_csv_name  
            contract_df.to_csv(str(contract_csv_path), index=True)
            pointprocess_df.to_csv(str(pointprocess_csv_path), index=True)
            print(f"Data saved to >> {str(contract_csv_path)}")
            print(f"Data saved to >> {str(pointprocess_csv_path)}")
            print()
    return

class Pointprocess_Dataset(Dataset):
    def __init__(self, 
                 pointprocess_dfs, 
                 point_num=60, 
                 future_seconds=60):
        self.pointprocess_dfs = pointprocess_dfs
        self.coin_num = len(pointprocess_dfs)
        self.feature_num = len(pointprocess_dfs[0].columns)
        self.point_num = point_num
        self.future_seconds = future_seconds
        first_time = 0
        self.ts = np.array([])
        for df in pointprocess_dfs:
            first_time_ = df["seconds"].values[point_num] + point_num
            if first_time < first_time_:
                first_time = first_time_
        for df in pointprocess_dfs:
            times = df[first_time<df["seconds"]]["seconds"].values[point_num:-future_seconds]
            self.ts = np.append(self.ts, times)
        self.ts = np.sort(np.unique(self.ts))

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        time = self.ts[index] + int(np.random.randint(-self.point_num,
                                                      self.point_num, (1,)))
        input_arr = np.zeros([self.coin_num, self.point_num, self.feature_num])
        target_arr = np.zeros([self.coin_num, self.future_seconds]) # (coin_num, future_seconds)
        for i, df in enumerate(self.pointprocess_dfs):
            past_arr = df[df["seconds"]<=time].values[-self.point_num:,:] # (point_num, feature_num)
            future_arr = df[(time<df["seconds"]) & \
                            (df["seconds"]<time+self.future_seconds)].values # (-1, feature_num)
            past_arr[:,0] -= time    # (point_num, feature_num)
            future_arr[:,0] -= (time+1)  # (-1, feature_num)
            input_arr[i,:,:] = past_arr
            for j in range(len(future_arr)):
                future_time = int(future_arr[j,0])
                target_arr[i,future_time] = 1.0
        input_tensor = torch.from_numpy(input_arr.astype(np.float32))   # (coin_num, point_num, feature_num)
        target_tensor = torch.from_numpy(target_arr.astype(np.float32)) # (coin_num, future_seconds)
        return input_tensor, target_tensor

if __name__ == '__main__':
    start_dates = args.start_dates
    start_dates = [datetime.strptime(x, "%Y/%m/%d").date() for x in start_dates]
    end_dates = args.end_dates
    end_dates = [datetime.strptime(x, "%Y/%m/%d").date() for x in end_dates]
    symbols = args.symbols
    v_name = args.v_name
    v_thresholds = args.v_thresholds
    v_threshold_dic = {}
    for i, symbol in enumerate(symbols):
        v_threshold_dic[symbol] = v_thresholds[i]
    save_csvs_path = root_path / "dataframes"
    save_pointprocess_dfs(start_dates, end_dates, symbols, 
                          v_name, v_threshold_dic, save_csvs_path)
