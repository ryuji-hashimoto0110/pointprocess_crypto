import argparse
from datetime import datetime, timedelta, time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pathlib
from utils.datasets import bybit_load_contract_data, make_pointprocess_from_contract_data
root_path = pathlib.Path("")
parser = argparse.ArgumentParser()
parser.add_argument("--start_date", required=True, type=str)
parser.add_argument("--end_date", required=True, type=str)
parser.add_argument("--symbol", required=True, type=str)
parser.add_argument("--v_name", default="size", type=str)
parser.add_argument("--v_threshold", required=True, type=int)
args = parser.parse_args()

def np_sigmoid(x):
    return np.exp(x) / np.sum(np.exp(x))

def calc_band_width(x): # Silvermann
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    std = np.std(x)
    band_width = 0.9 * np.min([iqr, std]) / (len(x) ** 1/5)
    return band_width

def epsilon_greedy(eps, x):
    if len(x) == 1:
        return 0
    maximize = np.random.binomial(1, eps)
    if maximize:
        i = np.argmax(x)
    else:
        i_arr = np.random.multinomial(1, x)
        i = int(np.where(i_arr==1)[0])
    return i

def prior_P(background_prior, len_data):
    prior_P = np.triu(np.ones((len_data, len_data)), k=1) * (1-background_prior)
    prior_P[:,1:] /= np.arange(1,len_data)
    prior_P += np.identity(len_data) * background_prior
    return prior_P

class kernel_estimate:
    def __init__(self, 
                 sampled_data, # (len_sample)
                 band_width):
        self.sampled_data = sampled_data
        self.len_data = len(sampled_data)
        self.t_sampled = sampled_data[np.newaxis,:] # (1,len_sample)
        self.band_width = band_width

    def calc_density(self, t_vec): # (len_t, 1)
        kernel_arr = 1 / np.sqrt(2*np.pi*self.band_width) * \
                     np.exp(- (t_vec-self.t_sampled)**2 / (2*self.band_width**2)) # (len_t, len_sampled)
        return np.sum(kernel_arr, axis=1) / self.len_data # (len_t,)

def stochastic_declustering(save_imgs_path,
                            save_arrs_path,
                            start_date, end_date, symbol,
                            v_name="size", v_threshold=150000,
                            pointprocess_df=None,
                            contract_df=None, 
                            iter_num=2,
                            background_prior=0.5):
    start_datetime = datetime.combine(start_date, time())
    end_datetime = datetime.combine(end_date+timedelta(days=1), time())
    if contract_df is None:
        contract_df = bybit_load_contract_data(start_date, end_date, symbol)
    if pointprocess_df is None:
        pointprocess_df = make_pointprocess_from_contract_data(
            contract_df, start_datetime, end_datetime, v_name, v_threshold
        )
    len_data = len(pointprocess_df)
    print(f"data num: {len_data}")
    pointprocess_arr = pointprocess_df["seconds"].values[:,np.newaxis] # (len_data, 1)
    background_P = prior_P(background_prior, len_data)
    P = np.zeros_like(background_P) + background_P
    P_err_list = []
    for n in range(iter_num):
        background_list = []
        aftershock_list = []
        for j in range(len_data):
            i = epsilon_greedy(0, P[:j+1,j])
            j_data = pointprocess_arr[j,0]
            i_data = pointprocess_arr[i,0]
            if i == j:
                background_list.append(j_data)
            else:
                aftershock_list.append(j_data - i_data)
        background_arr = np.array(background_list)
        aftershock_arr = np.array(aftershock_list)
        band_width_mu  = calc_band_width(background_arr)
        band_width_g   = calc_band_width(aftershock_arr)
        mu = kernel_estimate(background_arr, 
                             band_width=calc_band_width(background_arr))
        g = kernel_estimate(aftershock_arr, 
                            band_width=calc_band_width(aftershock_arr))
        distances = - (pointprocess_arr - pointprocess_arr.T)
        distances = distances[np.triu_indices(len_data,1)][:,np.newaxis]
        mu_vec = mu.calc_density(pointprocess_arr)
        g_vec = g.calc_density(distances)
        P_ = np.zeros_like(background_P) + background_P
        P_[~np.tri(len_data, dtype=bool, k=0)] *= g_vec
        P_[np.diag_indices(len_data)] *= mu_vec
        lam = np.sum(P_, axis=0)
        P_ /= lam
        P_err = np.sum((P_-P)**2 / len_data)
        P_err_list.append(P_err)
        print(f"iteration{n+1}:[P_err]{P_err:.4f} " +
              f"[Aftershock num]{len(aftershock_arr)} " +
              f"[band_width](mu){band_width_mu:.2f}, (g){band_width_g:.2f}")
        P = P_
    if not save_arrs_path.exists():
        save_arrs_path.mkdir(parents=True)
    mu_save_path = save_arrs_path / "mu"
    np.save(str(mu_save_path), mu)
    g_save_path = save_arrs_path / "g"
    np.save(str(g_save_path), g)
    P_save_path = save_arrs_path / "P"
    np.save(str(P_save_path), P)
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(P_err_list,
            marker="", markersize=6, color="black", label="P error")
    ax.legend(loc="upper right")
    ax.set(xlabel="epoch", ylabel="error")
    if not save_imgs_path.exists():
        save_imgs_path.mkdir(parents=True)
    p_err_img_path = save_imgs_path / "p_err.png"
    plt.savefig(str(p_err_img_path))
    plt.close(fig)
    xfmt = mdates.DateFormatter("%m/%d")
    xloc = mdates.DayLocator()
    fig = plt.figure(figsize=(8,11), dpi=100, facecolor="w")
    # ax1 contract_df bar 
    ax1 = fig.add_subplot(3,1,1)
    contract_df_selected = contract_df[contract_df[v_name]>v_threshold]
    ax1.bar(contract_df_selected.index, contract_df_selected[v_name],
            width=0.002, color="black")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(xloc)
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.set_ylabel(f"Volume (USD)")
    ax1.set_title(f"Transaction records of {symbol} on Bybit (2022)")
    # ax2 mu
    dates_list = [start_datetime + timedelta(seconds=seconds) for seconds in \
                  range(int((end_datetime - start_datetime).total_seconds()))]
    t_vec_mu = np.arange(len(dates_list))[:,np.newaxis]
    mu_arr = mu.calc_density(t_vec_mu)
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(dates_list, mu_arr, color="black")
    ax2.xaxis.set_major_locator(xloc)
    ax2.xaxis.set_major_formatter(xfmt)
    ax2.set_ylabel("Density")
    ax2.set_title("Estimated mu")
    # ax3 g
    t_vec_g = np.arange(60*30)[:,np.newaxis]
    g_arr = g.calc_density(t_vec_g)
    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(t_vec_g, g_arr, color="black")
    ax3.set_xticks(np.arange(0, 60*31, 60*10))
    ax3.set_xticklabels(["0", "10", "20", "30"])
    ax3.set_xlabel("minute")
    ax3.set_ylabel("Density")
    ax3.set_title("Estimated g")
    mug_estimate_img_path = save_imgs_path / "mug_estimate.png"
    plt.savefig(str(mug_estimate_img_path))
    plt.close(fig) 
    return 

if __name__ == "__main__":
    start_date = datetime.strptime(args.start_date, "%Y/%m/%d").date()
    end_date = datetime.strptime(args.end_date, "%Y/%m/%d").date()
    symbol = args.symbol
    v_name = args.v_name
    v_threshold = args.v_threshold
    save_imgs_path = root_path / "images"
    save_arrs_path = root_path / "checkpoints"
    stochastic_declustering(save_imgs_path,
                            save_arrs_path,
                            start_date, end_date, symbol,
                            v_name, v_threshold)
