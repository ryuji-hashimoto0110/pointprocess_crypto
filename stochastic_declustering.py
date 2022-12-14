from datetime import datetime, timedelta, date, time
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pathlib

root_path = pathlib.Path("")

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

def stochastic_declustering(contract_df, 
                            pointprocess_df, 
                            save_imgs_path,
                            iter_num=20,
                            background_prior=0.5):
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
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(P_err_list,
            marker="", markersize=6, color="black", label="P error")
    ax.legend(loc="upper right")
    ax.set(xlabel="epoch", ylabel="error")
    p_err_img_path = save_imgs_path / "p_err.png"
    plt.savefig(str(p_err_img_path))
    plt.close(fig)
    
    return mu, g, P