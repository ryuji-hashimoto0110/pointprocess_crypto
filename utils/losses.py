import numpy as np
import torch
from torch import nn

class Window_Loss(nn.Module):
    def __init__(self, window_size, future_seconds, device):
        super(Window_Loss, self).__init__()
        self.window_size = window_size
        window = np.ones((future_seconds-window_size+1,
                          future_seconds))
        window -= np.triu(window, k=window_size)
        window -= np.tril(window, k=-1)
        self.window = torch.from_numpy(window.astype(np.float32)).to(device)
        self.device = device
        self.mseloss = nn.MSELoss(reduction="none")
    
    def forward(self, x, y): # (b, coin_num, future_seconds)
        b, coin_num, future_seconds = x.shape
        x_ma = torch.einsum('wT,bcT -> bcw', self.window, x) \
               / self.window_size # (b, coin_num, future_seconds-window_size+1)
        y_ma = torch.einsum('wT,bcT -> bcw', self.window, y) \
               / self.window_size
        x_ma = x_ma.view(b*coin_num,-1)
        y_ma = y_ma.view(b*coin_num,-1)
        y_ma_ = torch.softmax(y_ma, dim=1)
        window_loss = torch.sum(self.mseloss(x_ma,y_ma) * y_ma_)\
                      / (future_seconds - self.window_size + 1)
        return window_loss