import torch.nn.functional as F
def nabla_x(x):
    '''
    x: (n,2,h,w)
    '''
    x = F.pad(x, (1,0), mode='constant', value=0.0)
    return x[...,1:]-x[...,0:-1]
    
def nabla_y(x):
    '''
    x: (n,2,h,w)
    '''
    x = F.pad(x, (0,0,1,0), mode='constant', value=0.0)
    return x[...,1:,:]-x[...,0:-1,:]

def nablat_x(x):
    '''
    x: (n,2,h,w)
    '''
    x = F.pad(x, (0,1), mode='constant', value=0.0)
    return x[...,0:-1]-x[...,1:]

def nablat_y(x):
    '''
    x: (n,2,h,w)
    '''
    x = F.pad(x, (0,0,0,1), mode='constant', value=0.0)
    return x[...,0:-1,:]-x[...,1:,:]