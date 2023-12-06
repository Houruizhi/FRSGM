import torch.nn as nn
def get_norm_layer(norm_type):
    if norm_type == 'in':
        norm_layer = nn.InstanceNorm2d
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d
    elif norm_type == None:
        norm_layer = nn.Identity
    else:
        raise NotImplementedError(f'{norm_type} is not implemented')
    return norm_layer