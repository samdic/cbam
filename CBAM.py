import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention3D(nn.Module):
    """
    Channel Attention Module for 3D data
    Focuses on WHAT is meaningful in the feature maps
    """
    def __init__(
        self, 
        channels,
        reduction_ratio=16,
        pool_types=['avg', 'max'],
        norm_op=None,
        norm_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None
    ):
       
       
        super(ChannelAttention3D, self).__init__()
        
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.pool_types = pool_types
        
        # Default configurations
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin is None:
            nonlin = nn.ReLU
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        
        # Hidden dimension
        hidden_channels = max(channels // reduction_ratio, 1)
        
        # Shared MLP with optional normalization
        mlp_layers = []
        
        # First FC layer
        mlp_layers.append(nn.Linear(channels, hidden_channels, bias=True))
        
        # Optional normalization after first FC
        if norm_op is not None:
            mlp_layers.append(norm_op(hidden_channels, **norm_op_kwargs))
        
        # Non-linearity
        mlp_layers.append(nonlin(**nonlin_kwargs))
        
        # Second FC layer
        mlp_layers.append(nn.Linear(hidden_channels, channels, bias=True))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            Channel attention weighted features of shape (B, C, D, H, W)
        """
        batch_size, channels, depth, height, width = x.size()
        
        # Aggregate spatial information using different pooling strategies
        channel_att_sum = None
        
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                
                pooled = F.adaptive_avg_pool3d(x, 1).view(batch_size, channels)
            elif pool_type == 'max':
               
                pooled = F.adaptive_max_pool3d(x, 1).view(batch_size, channels)
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}. Use 'avg' or 'max'.")
            
            
            channel_att_raw = self.mlp(pooled)
            
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        # Apply sigmoid and reshape to (B, C, 1, 1, 1)
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        # Element-wise multiplication
        return x * scale


class SpatialAttention3D(nn.Module):
    """
    Spatial Attention Module for 3D data
    Focuses on WHERE is meaningful in the feature maps
    """
    def __init__(
        self,
        kernel_size=7,
        pool_types=['avg', 'max'],
        conv_op=None,
        norm_op=None,
        norm_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None
    ):
        """
        Args:
            kernel_size: Size of convolutional kernel (default: 7, must be odd)
            pool_types: List of pooling types ['avg', 'max'] (default: ['avg', 'max'])
            conv_op: Convolution operation (default: nn.Conv3d)
            norm_op: Normalization operation (e.g., nn.BatchNorm3d, nn.InstanceNorm3d, None)
            norm_op_kwargs: Dictionary with normalization parameters
            nonlin: Non-linearity operation (e.g., nn.ReLU, nn.LeakyReLU, None for Sigmoid only)
            nonlin_kwargs: Dictionary with non-linearity parameters
        """
        super(SpatialAttention3D, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.pool_types = pool_types
        num_channels = len(pool_types)  
        
        # Default configurations
        if conv_op is None:
            conv_op = nn.Conv3d
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        
        padding = kernel_size // 2
        
        # Build spatial attention layers
        layers = []
        
        # Convolution layer
        layers.append(
            conv_op(
                in_channels=num_channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
                bias=True if norm_op is None else False
            )
        )
        
        
        if norm_op is not None:
            layers.append(norm_op(1, **norm_op_kwargs))
        
        
        if nonlin is not None:
            layers.append(nonlin(**nonlin_kwargs))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            Spatially attended features of shape (B, C, D, H, W)
        """
        # Aggregate channel information using pooling
        pooled_features = []
        
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                # Average pooling along channel: (B, C, D, H, W) -> (B, 1, D, H, W)
                pooled = torch.mean(x, dim=1, keepdim=True)
            elif pool_type == 'max':
                # Max pooling along channel: (B, C, D, H, W) -> (B, 1, D, H, W)
                pooled, _ = torch.max(x, dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}. Use 'avg' or 'max'.")
            
            pooled_features.append(pooled)
        
        # Concatenate pooled features: (B, num_pool_types, D, H, W)
        concat = torch.cat(pooled_features, dim=1)
        
        # Apply convolution (and optional norm/nonlin)
        attention_map = self.conv(concat)
        
        # Apply sigmoid
        scale = torch.sigmoid(attention_map)
        
        # Element-wise multiplication
        return x * scale


class CBAM3D(nn.Module):
    """
    Convolutional Block Attention Module for 3D data
    Combines Channel Attention and Spatial Attention sequentially
    Fully parameterizable with configuration dictionaries
    """
    def __init__(
        self,
        channels,
        # Channel attention params
        reduction_ratio=16,
        channel_pool_types=['avg', 'max'],
        # Spatial attention params
        spatial_kernel_size=7,
        spatial_pool_types=['avg', 'max'],
        # Shared params
        conv_op=None,
        norm_op=None,
        norm_op_kwargs=None,
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
        super(CBAM3D, self).__init__()
        
        assert not (no_spatial and no_channel), "Cannot disable both channel and spatial attention!"
        
        self.no_spatial = no_spatial
        self.no_channel = no_channel
        
        # Default configurations
        if conv_op is None:
            conv_op = nn.Conv3d
        
        # Channel Attention Module
        if not no_channel:
            self.channel_attention = ChannelAttention3D(
                channels=channels,
                reduction_ratio=reduction_ratio,
                pool_types=channel_pool_types,
                norm_op=norm_op,  # Can be None
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs
            )
        
        # Spatial Attention Module
        if not no_spatial:
            self.spatial_attention = SpatialAttention3D(
                kernel_size=spatial_kernel_size,
                pool_types=spatial_pool_types,
                conv_op=conv_op,
                norm_op=norm_op,  
                norm_op_kwargs=norm_op_kwargs,
                nonlin=None,  
                nonlin_kwargs=None
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