from typing import Tuple, List, Union, Type
import torch.nn
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

import numpy as np

class ChannelMaxPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        match dim:
            case 2:
                self.maxpool = nn.AdaptiveMaxPool2d(1)
            case 3:
                self.maxpool = nn.AdaptiveMaxPool3d(1)

    def forward(self, input):
        out = self.maxpool(input)
        out = torch.reshape(out, (out.size()[0], out.size()[1]))
        return out

class ChannelAvgPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        match dim:
            case 2:
                self.avgpool = nn.AdaptiveAvgPool2d(1)
            case 3:
                self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, input):
        out = self.avgpool(input)
        out = torch.reshape(out, (out.size()[0], out.size()[1]))
        return out

class SpatialModule(nn.Module):
    def __init__(self, dim, nonlin=nn.Sigmoid, nonlin_kwargs={},kernel_size=7):
        super().__init__()
        self.dim = dim
        match self.dim:
            case 2:
                self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=3)
            case 3:
                self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=3)
        self.nonlin = nonlin(**nonlin_kwargs)

    def forward(self, input):
        out1, _ = torch.max(input, dim=1)
        out2 = torch.mean(input, dim=1)
        print(out1.shape, out2.shape)
        out1 = out1.unsqueeze(1)
        out2 = out2.unsqueeze(1)
        print(out1.shape, out2.shape)
        out = torch.cat((out1, out2), 1)
        print(out.shape)
        out = self.nonlin(self.conv(out))
        print(out.shape)
        return out

class ChannelModule(nn.Module):
    def __init__(self, dim, in_size, reduction_ratio, nonlin, nonlin_kwargs):
        super().__init__()
        self.dim = dim
        self.maxChannelPool = ChannelMaxPooling(self.dim)
        self.avgChannelPool = ChannelAvgPooling(self.dim)
        self.mlp = MLP(in_size, reduction_ratio, nonlin, nonlin_kwargs)

    def forward(self, input):
        out1 = self.maxChannelPool(input)
        out1 = self.mlp(out1)
        out2 = self.avgChannelPool(input)
        out2 = self.mlp(out2)
        out = out1 + out2
        out = torch.sigmoid(out)
        # out = out.expand_as(input)
        for _ in range(self.dim):
            out.unsqueeze_(-1)
        return out


class MLP(nn.Module):
    def __init__(self, in_size, reduction_ratio, nonlin, nonlin_kwargs):
        super().__init__()
        hidden_size = in_size//reduction_ratio
        self.mlp = nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            nonlin(**nonlin_kwargs),
            torch.nn.Linear(hidden_size, in_size)
        )

    def forward(self, input):
        return self.mlp(input)

class CBAM(nn.Module):
    def __init__(self, nonlin=nn.ReLU, nonlin_kwargs={"inplace": True}, dim=3, reduction_ratio=16, in_size=20, kernel_size=7):
        super().__init__()
        assert dim in {2,3}, "dim is the dimension of the input : dim in {2,3}"
        self.channelModule = ChannelModule(dim=dim, in_size=in_size, reduction_ratio=reduction_ratio, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
        self.spatialModule = SpatialModule(dim, kernel_size=kernel_size)


    def forward(self, input):
        print(input.shape)
        out = self.channelModule(input)*input
        print(out.shape)
        out = self.spatialModule(out)*out
        return out

if __name__ == "__main__":
    y = torch.rand(3,7,32,32,32)
    test = CBAM(dim=3,reduction_ratio=2, in_size=7)
    print(test.forward(y))
