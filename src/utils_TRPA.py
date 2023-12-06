
import os
import time
import math
import torch
import numpy as np

from .utils import projection_tensor, compare_psnr_ssim_tensor, iFT, rss_complex

COIL_DIM = 0

def denoiser_TRPA(scorenet, sigma, x, step_lr=1):
    x = x.contiguous()
    sigma = torch.tensor(sigma).to(x.device)
    sigma = sigma.view(1,1,1,1)
    noise = torch.randn_like(x) * sigma
    inputs = x + noise
    logp = (step_lr*sigma**2)*scorenet(inputs, sigma)
    return x + logp

def recon_TRPA(
    scorenet, 
    data_kspace, 
    mask, 
    sense_map=None,
    target_rss=None,
    gamma=1.07,
    lam=3e-4,
    rho=0.0033,
    max_iter=120,
    eps=7e-5,
    step_lr=1,
    repeat=1,
    normlize_input=False,
    verbose=False,
    crop_win=320):
    
    '''
    inputs:
        data_kspace: (c,h,w,2)
        sense_map: (c,h,w,2)
        target_rss: (h,w)

    '''
    v = iFT(data_kspace, sense_map, coil_dim=COIL_DIM)
    
    if normlize_input:
        max_i = rss_complex(v, dim=COIL_DIM).max()
        data_kspace = data_kspace/max_i
        v = v/max_i

    x = v.clone()
    
    u = torch.zeros_like(v)
    delta = 100
    
    psnrs = []
    rho_k = rho
    time1 = time.time()
    for idx in range(max_iter):
        x_old = x.clone()
        v_old = v.clone()
        u_old = u.clone()
        
        #-----------------------------------------------
        # denoising step
        #-----------------------------------------------
        sigma = math.sqrt(lam/rho_k)
        inputs = (x+u).permute(0,3,1,2)
        v = torch.zeros_like(inputs)
        for _ in range(repeat):
            with torch.no_grad():
                v = v + denoiser_TRPA(scorenet, sigma, inputs, step_lr=step_lr)
        v = v.permute(0,2,3,1) / repeat
        #-----------------------------------------------
        # projection step
        #-----------------------------------------------
        v_sub_u = v - u
        x = projection_tensor(v_sub_u,data_kspace,mask,rho=1e-4*rho_k,sense_map=sense_map,coil_dim=COIL_DIM)

        #-----------------------------------------------
        # multiplier update step
        #-----------------------------------------------
        u = x - v_sub_u

        if target_rss is not None:
            PSNR, SSIM, _ = compare_psnr_ssim_tensor(x, target_rss, crop_win, coil_dim=COIL_DIM)
            psnrs.append((PSNR, SSIM, time.time()-time1))
        
        delta = (v_old-v).pow(2).mean() + (u_old-u).pow(2).mean() + (x_old-x).pow(2).mean()
        rho_k = gamma*rho_k
        
        if verbose and (idx%10 == 0):
            print(f'iter: {idx}, delta: {delta}, sigma: {int(sigma*255)}, PSNR: {PSNR}, SSIM: {SSIM}, TIME: {time.time()-time1}')
        
        if delta < eps:
            break
        
    return x, psnrs