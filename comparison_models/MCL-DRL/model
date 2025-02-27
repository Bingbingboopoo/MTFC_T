""""
MCL-DRL
Chen H, Meng W, Li Y, et al. An anti-noise fault diagnosis approach for rolling bearings
based on multiscale CNN-LSTM and a deep residual learning model[J]. Measurement Science
and Technology, 2023, 34(4): 045013.
"""

import torch
import torch.nn as nn
from torchinfo import summary


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution_block = nn.Sequential(
            nn.Conv1d(1, 64, 64, 7),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(3,1)
        )

    def forward(self, x):

        return self.convolution_block(x)    # 1, 64, 136


class CNN_LSTM_block(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        padding = (2*136 + kernel-3) // 2 + 1
        self.cnn = nn.Sequential(
            nn.Conv1d(64, 32, kernel, 3, padding),
            nn.BatchNorm1d(32),
        )
        self.lstm = nn.LSTM(32,32,batch_first=True)

    def forward(self, x):

        output_c = self.cnn(x)
        output_c = output_c.transpose(1,2)
        output_l, _ = self.lstm(output_c)
        output_l = output_l.transpose(1,2)
        return output_l

class Resdual_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool1d(3,1)
        self.Res_block = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.BatchNorm1d(32)
        )
        self.lstm = nn.LSTM(32, 32, batch_first=True)
    def forward(self, x):
        x = self.maxpool(x)
        output_ = self.Res_block(x).transpose(1,2)
        output_rb, _ = self.lstm(output_)

        return x + output_rb.transpose(1,2)


class MCNN_LSTM_DR(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.Conv_block = ConvBlock()
        self.CNN_LSTM_Block_1 = CNN_LSTM_block(32)
        self.CNN_LSTM_Block_2 = CNN_LSTM_block(16)
        self.CNN_LSTM_Block_3 = CNN_LSTM_block(8)
        self.Res_block = Resdual_Block()
        self.Class_layer = nn.Sequential(
            nn.Linear(4288,64),
            nn.Linear(64, class_num)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        y_0 = self.Conv_block(x)
        y_1_1 = self.CNN_LSTM_Block_1(y_0)
        y_1_2 = self.CNN_LSTM_Block_2(y_0)
        y_1_3 = self.CNN_LSTM_Block_3(y_0)
        y_1 = y_1_1 + y_1_2 + y_1_3
        y_2 = self.Res_block(y_1)
        y_2 = y_2.reshape(x.shape[0], -1)
        output = self.Class_layer(y_2)
        return output
