"""
QCNN
Liao J X, Dong H C, Sun Z Q, et al. Attention-embedded quadratic network (qttention) for effective
and interpretable bearing fault diagnosis[J]. IEEE Transactions on Instrumentation and Measurement, 2023, 72: 1-13.
"""
import torch.nn as nn
import torch.nn.functional as F
from QCNN.ConvQuadraticOperation import ConvQuadraticOperation

class QCNN(nn.Module):
    """
    QCNN builder
    """

    def __init__(self, ) -> object:
        super(QCNN, self).__init__()
        self.cnn = nn.Sequential()
        # self.cnn1 = nn.Sequential()
        self.cnn.add_module('Conv1D_1', ConvQuadraticOperation(1, 16, 64, 8, 28))
        self.cnn.add_module('BN_1', nn.BatchNorm1d(16))
        self.cnn.add_module('Relu_1', nn.ReLU())
        self.cnn.add_module('MAXPool_1', nn.MaxPool1d(2, 2))
        self.__make_layerq(16, 32, 1, 2)
        self.__make_layerq(32, 64, 1, 3)
        self.__make_layerq(64, 64, 1, 4)
        self.__make_layerq(64, 64, 1, 5)
        self.__make_layerq(64, 64, 0, 6)

        self.fc1 = nn.Linear(64, 100)
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 7)



    def __make_layerq(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), ConvQuadraticOperation(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        # self.cnn.add_module('DP_%d' %(nb_patch), nn.Dropout(0.5))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def __make_layerc(self, in_channels, out_channels, padding, nb_patch):
        self.cnn1.add_module('Conv1D_%d' % (nb_patch), nn.Conv1d(in_channels, out_channels, 3, 1, padding))
        self.cnn1.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        # self.cnn.add_module('DP_%d' %(nb_patch), nn.Dropout(0.5))
        self.cnn1.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn1.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        out1 = self.cnn(x)
        out = self.fc1(out1.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)
