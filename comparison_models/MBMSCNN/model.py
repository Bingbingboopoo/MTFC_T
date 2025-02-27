"""
MBMSCNN
Peng D, Wang H, Liu Z, et al. Multibranch and multiscale CNN for fault diagnosis of wheelset bearings
under strong noise and variable load condition[J]. IEEE Transactions on Industrial Informatics, 2020, 16(7): 4949-4960.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.polys.agca.modules import Module
from torch.nn.functional import batch_norm


def normalization_processing(data):
    data_mean = data.mean()
    data_var = data.var()
    return (data - data_mean) / data_var


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x) * np.sqrt(npower))


def add_noise(data, snr_num):
    rand_data = wgn(data, snr_num)
    data = data + rand_data
    return data


def moving_average(x, w=5, padding='same'):
    return np.convolve(x, np.ones(x), padding) / w


def guassian_func(x):
    delta = 1
    return 1 / (delta * np.sqrt(2 * np.pi)) * np.exp(-x * x / (2 * delta * delta))


def guassian_filtering(x, padding='same'):
    w = 5
    w_j = np.arange(5) - 2
    guassian_coef = [guassian_func(i) for i in w_j]
    return np.convolve(x, guassian_coef, padding) / sum(guassian_coef)


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MultiscaleModule(nn.Module):
    def __init__(self, in_channels, h=4, K=6, C=16, S=2, D=0.5):
        super(MultiscaleModule, self).__init__()
        # out_channels : C 注意：最终输出维度为C
        self.h = h
        kernel_sizes = [(2 ** i) * K for i in range(h)]
        for i in range(h):
            conv_block = Conv1DBlock(in_channels, int(C / h), kernel_sizes[i], stride=S, dropout_rate=D)
            setattr(self, f"conv_block_{i}", conv_block)

    def forward(self, x):
        all_scales = []
        for i in range(self.h):
            conv_block = getattr(self, f"conv_block_{i}")
            x0 = conv_block(x)
            all_scales.append(x0)

        y = torch.cat(all_scales, dim=1)
        return y


class ConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, drs):
        super(ConvBranch, self).__init__()
        self.n_layers = len(out_channels)
        in_c = in_channels
        for i, (out_c, k_size, stride, dr) in enumerate(
                zip(out_channels, kernel_sizes, strides, drs)):
            conv1d_block = Conv1DBlock(in_c, out_c, kernel_size=k_size, stride=stride, dropout_rate=dr)
            in_c = out_c
            setattr(self, f"conv1d_block_{i}", conv1d_block)

    def forward(self, x):
        for i in range(self.n_layers):
            conv1d_block = getattr(self, f"conv1d_block_{i}")
            x = conv1d_block(x)
        return x


class MultiscaleBranch(nn.Module):
    def __init__(self, in_channels, h_s, k_s, c_s, s_s, d_s):
        super(MultiscaleBranch, self).__init__()
        self.n_layers = len(h_s)
        in_c = in_channels
        for i, (h, k, c, s, d) in enumerate(zip(h_s, k_s, c_s, s_s, d_s)):
            ms_module = MultiscaleModule(in_c, h, K=k, C=c, S=s, D=d)
            in_c = c
            setattr(self, f"ms_module_{i}", ms_module)

    def forward(self, x):
        for i in range(self.n_layers):
            ms_module = getattr(self, f"ms_module_{i}")
            x = ms_module(x)
        return x


class MBSCNN(nn.Module):
    def __init__(self, raw_in_channels=1, lof_in_channels=1, den_in_channels=1, classes_num=7):
        super(MBSCNN, self).__init__()
        # in_channels = x.shape[1]
        self.raw_ms_branch = MultiscaleBranch(
            raw_in_channels,
            h_s=[4] * 5,
            k_s=[i for i in range(6, 1, -1)],
            c_s=[2 ** (i + 4) for i in range(5)],
            s_s=[4, 2, 2, 2, 2],
            d_s=[i / 10 for i in range(5, 0, -1)]
        )
        self.lof_conv_branch = ConvBranch(
            in_channels=lof_in_channels,
            out_channels=[2 ** (i+4) for i in range(5)],
            kernel_sizes=[i for i in range(6, 1, -1)],
            strides=[4, 2, 2, 2, 2],
            drs=[i / 10 for i in range(5, 0, -1)]
        )
        self.den_branch = ConvBranch(
            in_channels=den_in_channels,
            out_channels=[2 ** (i + 4) for i in range(5)],
            kernel_sizes=[i for i in range(6, 1, -1)],
            strides=[4, 2, 2, 2, 2],
            drs=[i / 10 for i in range(5, 0, -1)]
        )

        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(768, classes_num)

    def forward(self, raw_signal, lof_signal, den_signal):
        raw_signal = raw_signal.unsqueeze(1)
        lof_signal = lof_signal.unsqueeze(1)
        den_signal = den_signal.unsqueeze(1)
        y1 = self.raw_ms_branch(raw_signal)
        y2 = self.lof_conv_branch(lof_signal)
        y3 = self.den_branch(den_signal)
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.pool1d(y).squeeze(-1)
        y = self.linear(y)
        return F.softmax(y, dim=1)
