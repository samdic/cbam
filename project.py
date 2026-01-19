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
        self.dim = dim

    def forward(self, input : Tensor):
        match self.dim:
            case 2:
                out = F.max_pool2d(input, kernel_size=input.size()[2:])
                out = torch.reshape(out, (out.size()[0], out.size()[1]))
            case 3:
                out = F.max_pool3d(input, kernel_size=input.size()[2:])
                out = torch.reshape(out, (out.size()[0], out.size()[1]))
                return out

class ChannelAvgPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input : Tensor):
        match self.dim:
            case 2:
                out = F.avg_pool2d(input, kernel_size=input.size()[2:])
                out = torch.reshape(out, (out.size()[0], out.size()[1]))
                return out
            case 3:
                out = F.avg_pool3d(input, kernel_size=input.size()[2:])
                out = torch.reshape(out, (out.size()[0], out.size()[1]))
                return out

class SpatialMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input : Tensor):
        out = torch.max(input, dim=1)[0]
        return out

class SpatialAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input : Tensor):
        out = torch.mean(input, dim=1)
        return out

class SpatialModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input : Tensor):
        spatialMaxPool = SpatialMaxPooling()
        spatialAvgPool = SpatialAvgPooling()
        out1 = spatialMaxPool(input)
        out2 = spatialAvgPool(input)
        out = torch.cat((out1, out2), 0)
        match self.dim:
            case 2:
                conv = nn.Conv2d(2, 1, kernel_size=1)
            case 3:
                conv = nn.Conv3d(2, 1, kernel_size=1)
        out = conv(out)
        out = torch.sigmoid(out)
        return out

class ChannelModule(nn.Module):
    def __init__(self, dim, r):
        super().__init__()
        self.dim = dim
        self.r = r

    def forward(self, input : Tensor):
        maxChannelPool = ChannelMaxPooling(self.dim)
        avgChannelPool = ChannelAvgPooling(self.dim)
        mlp = MLP(self.r)
        out1 = maxChannelPool(input)
        out1 = mlp(out1)
        out2 = avgChannelPool(input)
        out2 = mlp(out2)
        out = out1+out2
        out = torch.sigmoid(out)
        for i in range(self.dim):
            out = out.unsqueeze(-1)
        return out


class MLP(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r
    
    def forward(self, input : Tensor):
        mlp = []
        mlp.append(torch.nn.Linear(input.size()[1], input.size()[1]//self.r))
        mlp.append(torch.nn.Linear(input.size()[1]//self.r,input.size()[1]))
        mlp = nn.Sequential(*mlp)
        return mlp(input)

class CBAM(nn.Module):
    def __init__(self, activation, activation_kwargs, norm, norm_kwargs , dim, r):
        super().__init__()
        assert dim in {2,3}, "dim is the dimension of the input : dim in {2,3}"
        self.dim = dim
        self.r = r


    def forward(self, input : Tensor):
        channelModule = ChannelModule(self.dim, self.r)
        channelOutput = channelModule(input)
        out = input*channelOutput
        spatialModule = SpatialModule(self.dim)
        spatialOutput = spatialModule(out)
        out = out*spatialOutput
        return out

if __name__ == "__main__":
    x = torch.tensor(np.array([[[[[1,1,1], [2,2,2], [3,3,3]], [[4,4,4], [5,5,5], [6,6,6]], [[7,7,7], [8,8,8], [9,9,9]]]   ,    [[[1,1,1], [2,2,2], [3,3,3]], [[4,4,4], [5,5,5], [6,6,6]], [[7,7,7], [8,8,8], [9,9,9]]]  ]]), dtype=torch.float32)
    test = CBAM(1,1,1,1,3,5)
    print(test.forward(x))
