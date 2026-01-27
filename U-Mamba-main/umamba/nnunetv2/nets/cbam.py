import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for 3D data
    Focuses on WHAT is meaningful in the feature maps
    """
    def __init__(
        self, 
        dim,
        channels,
        reduction_ratio=16,
        nonlin=None,
        nonlin_kwargs=None
    ):
       
        super(ChannelAttention, self).__init__()
        
        # Default configurations
        if nonlin is None:
            nonlin = nn.ReLU
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        
        # Hidden dimension
        hidden_channels = max(channels // reduction_ratio, 1)
        
        if dim==2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        elif dim==3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.mlp = nn.Sequential(nn.Linear(channels, hidden_channels, bias=True),
                                 nonlin(**nonlin_kwargs),
                                 nn.Linear(hidden_channels, channels, bias=True))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            Channel attention weighted features of shape (B, C, D, H, W)
        """
        batch_size, channels = x.size()[:2]
        
        out1 = self.avg_pool(x).view(batch_size, channels)
        out1 = self.mlp(out1)

        out2 = self.max_pool(x).view(batch_size, channels)
        out2 = self.mlp(out2)
        out = out1 + out2   
        scale = torch.sigmoid(out).view(batch_size, channels, *([1] * (x.dim() - 2)))
        return x * scale


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for 3D data
    Focuses on WHERE is meaningful in the feature maps
    """
    def __init__(
        self,
        dim=3,
        kernel_size=7
    ):
        """
        Args:
            dim: data dimension (2 or 3)
            kernel_size: Size of convolutional kernel (default: 7, must be odd)
        """
        super(SpatialAttention, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        # Default configurations
        if dim==2:
            conv_op = nn.Conv2d
        elif dim==3:
            conv_op = nn.Conv3d

        # Convolution layer
        self.conv = conv_op(
                in_channels=2,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            Spatially attended features of shape (B, C, D, H, W)
        """
        # Average pooling along channel: (B, C, D, H, W) -> (B, 1, D, H, W)
        out1 = torch.mean(x, dim=1, keepdim=True)
        # Max pooling along channel: (B, C, D, H, W) -> (B, 1, D, H, W)
        out2, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate pooled features: (B, num_pool_types, D, H, W)
        out = torch.cat([out1, out2], dim=1)
        
        # Apply sigmoid
        scale = torch.sigmoid(self.conv(out))
        
        # Element-wise multiplication
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module for 2D and 3D data
    Combines Channel Attention and Spatial Attention sequentially
    Fully parameterizable with configuration dictionaries
    """
    def __init__(
        self,
        dim,
        channels,
        reduction_ratio=16,
        spatial_kernel_size=7,
        nonlin=None,
        nonlin_kwargs=None,
        # Options
        no_spatial=False,
        no_channel=False
    ):
        """
        Args:
            channels: Number of input channels
        """
        super(CBAM, self).__init__()
        
        assert not (no_spatial and no_channel), "Cannot disable both channel and spatial attention!"
        
        self.no_spatial = no_spatial
        self.no_channel = no_channel
        
        # Channel Attention Module
        if not no_channel:
            self.channel_attention = ChannelAttention(
                dim=dim,
                channels=channels,
                reduction_ratio=reduction_ratio,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs
            )
        
        # Spatial Attention Module
        if not no_spatial:
            self.spatial_attention = SpatialAttention(
                dim=dim,
                kernel_size=spatial_kernel_size
            )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            Attended features of shape (B, C, D, H, W)
        """
        # Apply channel attention
        if not self.no_channel:
            x = self.channel_attention(x)
        
        # Apply spatial attention
        if not self.no_spatial:
            x = self.spatial_attention(x)
        
        return x
