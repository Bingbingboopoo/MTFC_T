"""
MRA-CNN
Jia L, Chow T W S, Wang Y, et al. Multiscale residual attention convolutional neural network
for bearing fault diagnosis[J]. IEEE Transactions on Instrumentation and Measurement, 2022, 71: 1-13.
"""
import torch
import torch.nn as nn
from argparse import Namespace


class BaseCNN(nn.Module):
    def __init__(self, basekernel, i_channel, o_channel):
        super().__init__()
        padding = int((basekernel - 1) // 2)
        self.BaseC = nn.Conv1d(in_channels=i_channel, out_channels=o_channel, kernel_size=basekernel, stride=2,
                               padding=padding)

    def forward(self, x):
        return self.BaseC(x)


class CR_NN(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.CR_NN = nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=5, stride=1, padding="same")

    def forward(self, x):
        output = self.CR_NN(x)

        return output


class Muti_scale_CR(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.scale = 4
        self.width = int(in_channel / self.scale)
        self.muti_scale_cr = nn.ModuleList()
        for i in range(self.scale):
            self.muti_scale_cr.append(CR_NN(self.width))
        self.BNlayer = nn.BatchNorm1d(in_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        split_x = torch.split(x, split_size_or_sections=self.width, dim=1)
        for i in range(self.scale):
            if i == 0:
                out_x = self.muti_scale_cr[i](split_x[i])
                out_y = out_x
            else:
                input_x = split_x[i] + out_x
                out_x = self.muti_scale_cr[i](input_x)
                out_y = torch.concat((out_y, out_x), dim=1)
        out_BN = self.BNlayer(out_y)
        output = self.relu(out_BN)

        return output


class RAMod_block(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channel, 1, 1, 1)
        self.GAPool = nn.AdaptiveAvgPool1d(1)
        self.BatchNormal = nn.BatchNorm1d(in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s1 = self.conv1d(x)
        c1 = self.GAPool(x)

        sc1 = torch.matmul(c1, s1)
        BN = self.BatchNormal(sc1)
        sigmoid = self.sigmoid(BN)
        output = x * sigmoid + x + sigmoid
        return output


class MR_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_num = len(config.in_channels) - 1
        self.MR_cnn = nn.ModuleList()

        for i in range(self.layer_num):
            self.MR_cnn.append(BaseCNN(config.base_kernel[i], i_channel=config.in_channels[i], o_channel=config.in_channels[i + 1]))
            self.MR_cnn.append(Muti_scale_CR(in_channel=config.in_channels[i + 1]))
            self.MR_cnn.append(RAMod_block(config.in_channels[i + 1]))
        self.MR_cnn.append(nn.Sequential(
            nn.AdaptiveAvgPool1d(1)
        ))
        self.Liner = nn.Linear(config.in_channels[-1], 7)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        for layer in self.MR_cnn:
            output = layer(x)
            x = output

        x = x.view(x.shape[0], -1)
        output = self.Liner(x)
        return output
