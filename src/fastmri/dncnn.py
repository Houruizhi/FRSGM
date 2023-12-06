import torch
import torch.nn as nn
from .weight import weights_init_kaiming

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_layers, num_features, norm_layer='bn', kernel_size=3, bias=False, if_residual=False):
        super(DnCNN, self).__init__()
        kernel_size = kernel_size
        padding = kernel_size//2
        features = num_features
        in_channels = in_channels
        groups = 1
        layers = []
        layers.append(nn.ReflectionPad2d((padding,padding,padding,padding)))
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=features, 
            kernel_size=kernel_size, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.ReflectionPad2d((padding,padding,padding,padding)))
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, 
                kernel_size=kernel_size, bias=bias, groups=groups))
            if norm_layer == 'in':
                layers.append(nn.InstanceNorm2d(features))
            elif norm_layer == 'bn':
                layers.append(nn.BatchNorm2d(features))
            elif norm_layer == None:
                pass
            else:
                raise NotImplementedError(f'The normalization {norm_layer} is not implemented')
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ReflectionPad2d((padding,padding,padding,padding)))
        layers.append(nn.Conv2d(in_channels=features, out_channels=out_channels, 
            kernel_size=kernel_size, bias=bias))
        self.dncnn = nn.Sequential(*layers)
        self.if_residual = if_residual
        # self.dncnn.apply(weights_init_kaiming)
    def forward(self, inputs):
        outputs = self.dncnn(inputs)
        if self.if_residual:
            return inputs + outputs
        else:
            return outputs