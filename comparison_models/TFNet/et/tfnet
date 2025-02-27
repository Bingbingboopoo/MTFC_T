"""
TFNet
Chen Q, Dong X, Tu G, et al. TFN: An interpretable neural network with time-frequency transform embedded
 for intelligent fault diagnosis[J]. Mechanical Systems and Signal Processing, 2024, 207: 110952.
"""
from TFNet.TFconvlayer import *
from TFNet.BackboneCNN import CNN


class Base_FUNC_CNN(CNN):
    """
    the base class of TFN
    """
    FuncConv1d = BaseFuncConv1d
    funckernel_size = 21

    def __init__(self, in_channels=1, out_channels=10, kernel_size=15, clamp_flag=True, mid_channel=16):
        super().__init__(in_channels, out_channels, kernel_size)
        # Reinitialize the first layer by changing the in_channels
        args = {x: getattr(self.layer1[0], x) for x in
                ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias']}
        args['bias'] = None if (args['bias'] is None) else True
        args['in_channels'] = mid_channel
        self.layer1[0] = nn.Conv1d(**args)
        # use the TFconvlayer as the first preprocessing layer
        self.funconv = self.FuncConv1d(in_channels, mid_channel, self.funckernel_size,
                                       padding=self.funckernel_size // 2,
                                       bias=False, clamp_flag=clamp_flag)
        self.superparams = self.funconv.superparams

    def forward(self, x):
        x = self.funconv(x)
        return super().forward(x)

    def getweight(self):
        """
        get the weight and superparams of the first preprocessing layer (for recording)
        """
        weight = self.funconv.weight.cpu().detach().numpy()
        superparams = self.funconv.superparams.cpu().detach().numpy()
        return weight, superparams


class TFN_STTF(Base_FUNC_CNN):
    """
    TFN with TFconv-STTF as the first preprocessing layer
    FuncConv1d = TFconv_STTF
    kernel_size = mid_channel * 2 - 1
    """
    FuncConv1d = TFconv_STTF
    def __init__(self, mid_channel=16, **kwargs):
        self.funckernel_size = mid_channel * 2 - 1
        super().__init__(mid_channel=mid_channel, **kwargs)
