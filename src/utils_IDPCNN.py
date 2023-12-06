
import os
import time
import math
import torch
import numpy as np

from .utils import projection_tensor, compare_psnr_ssim_tensor, iFT, rss_complex
from .models.dncnn import DnCNN

COIL_DIM = 0

class Denoisers(torch.nn.Module):
    def __init__(self, pretrained_path, *args, **kargs):
        super().__init__()
        nets_path = os.listdir(pretrained_path)
        self.models = torch.nn.ModuleDict()
        for file_name in nets_path:
            model = DnCNN(*args, **kargs)
            noise_level = file_name.split('_')[-1]
            noise_level = float(noise_level)
            if noise_level < 1:
                noise_level = '%.1f'%(noise_level)
            else:
                noise_level = str(int(noise_level))
            model_path = os.path.join(pretrained_path, file_name, 'net.pth')
            model.load_state_dict(torch.load(os.path.join(model_path))['weights'])
            self.models[str(noise_level)] = model.cuda()
        
        self.keys = list(self.models.keys())
        self.dncnn_sigmas = np.array([float(i) for i in list(self.keys)])/255.
    def forward(self, x, sigma):
        dis = np.abs(self.dncnn_sigmas - sigma)
        sigma_ = self.keys[dis.argmin()]
        return self.models[str(sigma_)](x.contiguous())
        
def recon_IDPCNN(
        models, 
        data_kspace, 
        mask, 
        sense_map=None,
        target_rss=None,
        gamma=1.2,
        lam=6e-5,
        rho=0.0041,
        eta=1 + 1e-4,
        max_iter=100,
        eps=1e-6,
        normlize_input=False,
        verbose=False,
        crop_win=256):

    image_recon = iFT(data_kspace, sense_map)
    if normlize_input:
        max_i = rss_complex(image_recon, COIL_DIM).max()
        data_kspace = data_kspace/max_i
        image_recon = image_recon/max_i
    
    image_recon = image_recon

    psnrs = []
    
    rho_k = rho
    delta = 0
    count_iter = 0

    time1 = time.time()
    for kk in range(max_iter):
        delta_old = delta
        image_recon_old = image_recon.clone()

        sigma = np.sqrt(lam/rho_k)
        with torch.no_grad():
            image_recon = image_recon.permute(0,3,1,2)
            image_recon = models(image_recon, sigma).permute(0,2,3,1)

            image_recon = image_recon
            image_recon = projection_tensor(
                image_recon,
                data_kspace,
                mask,rho=1/80.,
                sense_map=sense_map,
                coil_dim=COIL_DIM
                )

        delta = torch.mean((image_recon_old-image_recon)**2)

        if delta > eta*delta_old:
            rho_k = gamma*rho_k

        if delta < eps:
            break

        if target_rss is not None:
            PSNR, SSIM, _ = compare_psnr_ssim_tensor(image_recon, target_rss, crop_win, coil_dim=COIL_DIM)
            psnrs.append((PSNR, SSIM, time.time()-time1))

        count_iter += 1 
        if verbose and (kk % 10 == 0):
            print(f'iter: {count_iter}, sigma: {sigma}, PSNR: {PSNR}')
        
    return image_recon, psnrs