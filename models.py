import math
import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, feature_num, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(0.1)
        self.div_term = torch.exp(torch.arange(0, feature_num, 2) * \
                                  -(math.log(10000.0) / feature_num)).to(device)

    def forward(self, x): # (b, point_num, feature_num)
        b, _, feature_num = x.shape
        pe = torch.zeros_like(x).float().to(self.device)
        x_seconds = x[:,:,0].unsqueeze(2).repeat(1,1,feature_num) # (b, point_num, feature_num)
        pe[:,:,0::2] += torch.sin(x_seconds[:,:,0::2] * self.div_term)
        pe[:,:,1::2] += torch.cos(x_seconds[:,:,1::2] * self.div_term)
        x = x + pe
        return self.dropout(x)

class FFN(nn.Module):
    def __init__(self, d_input, d_ff, d_output):
        super().__init__()
        self.linear1 = nn.Linear(d_input, d_ff)
        self.linear2 = nn.Linear(d_ff, d_output)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class Self_Exciting_Transformer(nn.Module):
    def __init__(self, coin_num, d_ff, d_model, feature_num, point_num, nhead, device):
        super(Self_Exciting_Transformer, self).__init__()
        self.coin_num = coin_num
        ffn = FFN(feature_num, d_ff, d_model)
        self.set_module_list = nn.ModuleList([])
        for _ in range(coin_num):
            pos = PositionalEncoding(point_num, feature_num, device)
            encoder_layer = nn.TransformerEncoderLayer(d_model=feature_num, 
                                                       nhead=nhead)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                        num_layers=1)
            ffn = FFN(feature_num, d_ff, d_model)
            set_module = nn.Sequential(
                pos, transformer_encoder, ffn
            )
            self.set_module_list.append(set_module)
    
    def forward(self, x): # (b, coin_num, point_num, feature_num)
        coin_list = []
        for i in range(self.coin_num):
            coin_feature = self.set_module_list[i](x[:,i,:,:]) # (b, point_num, d_model)
            coin_feature = coin_feature.unsqueeze(dim=1) # (b, 1, point_num, d_model)       
            coin_list.append(coin_feature)
        out = torch.cat(coin_list, dim=1) # (b, coin_num, point_num, d_model)
        d_model2 = out.shape[2] * out.shape[3]    
        out = out.view(-1, self.coin_num, d_model2) # (b, coin_num, point_num*d_model)
        return out

class Relation_Transformer(nn.Module):
    def __init__(self, coin_num, d_model2, d_ff2, feature_num,
                 future_seconds, nhead, device):
        super(Relation_Transformer, self).__init__()
        self.coin_num = coin_num
        self.future_seconds = future_seconds
        self.feature_num = feature_num
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model2, 
                                                   nhead=nhead)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                    num_layers=1)
        #ffn = FFN(d_model2, d_ff2, future_seconds*feature_num)
        self.rt_module = transformer_encoder

    def forward(self, x): # (b, coin_num, d_model2=point_num*d_model)
        out = self.rt_module(x) # (b, coin_num, d_model2)
        return out

class PointFormer(nn.Module):
    def __init__(self, 
                 coin_num, feature_num, point_num, d_model, d_ff, d_ff2, 
                 future_seconds, nhead1, nhead2, device):
        super(PointFormer, self).__init__()
        self.coin_num = coin_num
        self.setm = Self_Exciting_Transformer(coin_num, d_ff, d_model, 
                                              feature_num, point_num,
                                              nhead1, device)
        d_model2 = point_num * d_model
        #self.rtm = Relation_Transformer(coin_num, d_model2, d_ff2, 
        #                                feature_num, future_seconds, nhead2,
        #                                device)
        self.ffn_hour_list = nn.ModuleList([])
        self.ffn_time_list = nn.ModuleList([])
        for _ in range(coin_num):
            self.ffn_hour_list.append(FFN(point_num, d_ff2, future_seconds))
            self.ffn_time_list.append(FFN(d_model2, d_ff2, future_seconds))
        
    def forward(self, x): # (b, coin_num, point_num, feature_num)
        b, coin_num, _, _ = x.shape
        hours = x[:,:,:,1] # (b, coin_num, point_num)
        out = self.setm(x)  # (b, coin_num, point_num*d_model))
        #out = self.rtm(out) # (b, coin_num, d_model2)
        out_list = []
        for i in range(self.coin_num):
            hour_feature = self.ffn_hour_list[i](hours[:,i,:]) # (b, future_seconds)
            time_feature = self.ffn_time_list[i](out[:,i,:]) # (b, future_seconds)
            out_ = hour_feature + time_feature
            out_ = out_.unsqueeze(dim=1)
            out_list.append(out_)
        out = torch.cat(out_list, dim=1)
        out = torch.sigmoid(out)
        return out

class PointRNN(nn.Module):
    def __init__(self, 
                 coin_num, feature_num, point_num,
                 future_seconds, hidden_size, device):
        super(PointRNN, self).__init__()
        self.device = device
        self.coin_num = coin_num
        self.point_num = point_num
        self.feature_num = feature_num
        self.future_seconds = future_seconds
        self.hidden_size = hidden_size
        self.rnn_list = nn.ModuleList([])
        for _ in range(coin_num):
            rnn = nn.RNN(input_size=feature_num,
                         hidden_size=hidden_size,
                         num_layers=2, dropout=0.1, batch_first=True)
            self.rnn_list.append(rnn)
        self.ffn_time = FFN(hidden_size, hidden_size, future_seconds)

    def forward(self, x): # (b, coin_num, point_num, feature_num)
        b, _, _, _ = x.shape
        h_0 = torch.zeros(2,b,self.hidden_size).to(self.device)
        coin_list = []
        for i in range(self.coin_num):
            coin_feature, _ = self.rnn_list[i](x[:,i,:,:], h_0) # (b,point_num,hidden_size)
            coin_feature = coin_feature[:,-1,:].unsqueeze(dim=1)
            coin_list.append(coin_feature)
        out = torch.cat(coin_list, dim=1) # (b, coin_num, hidden_size)
        out = torch.sigmoid(self.ffn_time(out))
        return out