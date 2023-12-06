import torch
from .tv_op import nablat_x, nablat_y, nabla_x, nabla_y
from ..fastmri.fft import fft2, ifft2, fft2c, ifft2c, fftshift

def shrink2_torch(x,y,soft_thre):
    s = torch.sqrt(x.pow(2) + y.pow(2))
    r = (s-soft_thre).clip(0.)
    return r*x, r*y

def ROF_torch(f, lam=1, gamma=1e-3, nInner=10):
    u = torch.zeros_like(f)
    x = torch.zeros_like(f)
    y = torch.zeros_like(f)
    bx = torch.zeros_like(f)
    by = torch.zeros_like(f)
    
    uker = torch.zeros_like(f).permute(0,2,3,1)
    uker[:,0,0,0] = 4
    uker[:,0,1,0] = -1
    uker[:,1,0,0] = -1
    uker[:,-1,0,0] = -1
    uker[:,0,-1,0] = -1
    uker = 1 + (1e-3*gamma)*fftshift(fft2(uker))
    for _ in range(nInner):
        rhs = f + (1e-3*gamma)*(nablat_x(x-bx) + nablat_y(y-by))
        u = ifft2c(fft2c(rhs.permute(0,2,3,1))/uker).permute(0,3,1,2)
        udx = nabla_x(u)
        udy = nabla_y(u)
        x, y = shrink2_torch(udx+bx, udy+by, gamma/lam)
        bx = bx+udx-x
        by = by+udy-y
    return u