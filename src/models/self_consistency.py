import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfSimilar(nn.Module):
    def __init__(self, kernel_size, c, h, w):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight1 = nn.Parameter(torch.ones(1,1,self.kernel_size**2,h,w)/self.kernel_size**2)
        self.conv1_11 = nn.Conv2d(c,c,1)
        self.register_buffer('mean_filter', self.get_blur_kernel(5, 5, 1/3.).repeat(2,1,1,1))

    def forward(self, x):
        return self.get_Gx(x)
    
    def get_Gx(self, inputs):
        '''
        x: c,h,w,2
        '''
        x = inputs.permute(3,0,1,2)
        x = self.conv_pixelwise(x, self.weight1)
        x = self.conv1_11(x)
        return x.permute(1,2,3,0)
    
    def get_GTx(self, inputs):
        '''
        x: c,h,w,2
        '''
        x = inputs.permute(3,0,1,2)
        x = F.conv2d(x, self.conv1_11.weight.flip(dims=(-1,-2)).permute(1,0,2,3))
        x = self.get_adaptive_T_x(x, self.weight1)
        return x.permute(1,2,3,0)
    
    def get_adaptive_T_x(self, x, weight):
        h,w = x.shape[-2:]
        p = self.kernel_size//2
        pad_h, pad_w = self.kernel_size - h % self.kernel_size, self.kernel_size - w % self.kernel_size
        weight_t = F.pad(weight, (0,pad_w,0,pad_h), mode='constant', value=0.0)
        weight_t = weight_t.reshape(1,1,self.kernel_size**2,(h+pad_h)//self.kernel_size,self.kernel_size,(w+pad_w)//self.kernel_size,self.kernel_size)    
        weight_t = weight_t.flip(dims=(2,4,6))
        weight_t = weight_t.reshape(1,1,self.kernel_size**2,h+pad_h,w+pad_w)
        weight_t = weight_t[...,:h,:w]
        x = self.conv_pixelwise(x, weight_t)
        return x
    
    def conv_sep(self, x, kernel):
        '''
        x: 2,c,h,w
        '''
        x_real = F.conv2d(x[0].unsqueeze(0), kernel, padding=kernel.shape[-1]//2)
        x_imag = F.conv2d(x[1].unsqueeze(0), kernel, padding=kernel.shape[-1]//2)
        return torch.cat([x_real, x_imag], dim=0)
    
    def conv_mean_filter(self, x):
        '''
        x: c,h,w,2
        '''
        x = x.permute(0,3,1,2)
        x = F.conv2d(x, self.mean_filter, padding=self.mean_filter.shape[-1]//2, groups=2)
        return x.permute(0,2,3,1)
    
    def conv_pixelwise(self, x, kernel):
        '''
        x: 2,c,h,w
        '''
        p = self.kernel_size//2
        n,c,h,w = x.shape
        x = F.pad(x, (p,p,p,p), mode='constant', value=0.0)
        x = F.unfold(x, kernel_size=self.kernel_size).reshape(n, c, self.kernel_size**2, h, w)
        x = x * kernel
        x = x.sum(2)
        return x
    
    def get_blur_kernel(self, h, w, s=1.):
        X = torch.linspace(-1, 1, h)
        Y = torch.linspace(-1, 1, w)
        x, y = torch.meshgrid(X, Y)
        gauss_1 = 1-torch.exp(- (x ** 2 + y ** 2) / s)
        gauss_1 = gauss_1 / gauss_1.sum()
        return gauss_1.unsqueeze(0).unsqueeze(0).float()
    
class SelfSimilarSC(nn.Module):
    def __init__(self, kernel_size, c):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight1 = nn.Parameter(torch.ones(1,1,self.kernel_size,self.kernel_size)/self.kernel_size**2)
        self.conv1_11 = nn.Conv2d(c,c,1)

    def forward(self, x):
        return self.get_Gx(x)
    
    def get_Gx(self, inputs):
        '''
        x: c,h,w,2
        '''
        x = inputs.permute(3,0,1,2)
        x = self.conv_sep(x, self.weight1)
        x = self.conv1_11(x)
        return x.permute(1,2,3,0)
    
    def get_GTx(self, inputs):
        '''
        x: c,h,w,2
        '''
        x = inputs.permute(3,0,1,2)
        x = F.conv2d(x, self.conv1_11.weight.flip(dims=(-1,-2)).permute(1,0,2,3))
        x = x = self.conv_sep(x, self.weight1.flip(dims=(-1,-2)))
        return x.permute(1,2,3,0)
    
    def conv_sep(self, x, kernel):
        '''
        x: 2,c,h,w
        '''
        x_real = F.conv2d(x[0].unsqueeze(1), kernel, padding=kernel.shape[-1]//2).permute(1,0,2,3)
        x_imag = F.conv2d(x[1].unsqueeze(1), kernel, padding=kernel.shape[-1]//2).permute(1,0,2,3)
        return torch.cat([x_real, x_imag], dim=0)