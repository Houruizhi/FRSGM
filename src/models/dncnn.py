import torch
import torch.nn as nn
from .weight import weights_init_kaiming
from .norm_layer import get_norm_layer

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_layers, num_features, kernel_size=3, residual=True, bias=False, norm_type='bn'):
        super(DnCNN, self).__init__()
        self.residual = residual
        if self.residual:
            assert in_channels == out_channels
        norm_layer = get_norm_layer(norm_type)
        kernel_size = kernel_size
        padding = kernel_size//2
        features = num_features
        in_channels = in_channels
        groups = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=features, padding=padding,
            kernel_size=kernel_size, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, padding=padding,
                kernel_size=kernel_size, bias=bias, groups=groups))
            layers.append(norm_layer(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=out_channels, padding=padding,
            kernel_size=kernel_size, bias=bias))
        self.dncnn = nn.Sequential(*layers)
        self.dncnn.apply(weights_init_kaiming)
    def forward(self, inputs, sigma=None):
        x = self.dncnn(inputs)
        if self.residual:
            return inputs - x
        else:
            return x