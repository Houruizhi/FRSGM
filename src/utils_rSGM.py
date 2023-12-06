import time
import math
import torch

COIL_DIM = 0

from .models.self_consistency import SelfSimilar, SelfSimilarSC
from .utils import ifft2c, FT, iFT, rss_complex
from .utils import (
    compare_psnr_ssim_tensor, 
    projection_tensor,
    normalize_tensor
)

class RSGM(torch.nn.Module):
    def __init__(self, 
        scorenet=None, 
        cnn_denoisers=None,
        gamma=1.06,
        rho=0.2,
        lam=2e-3,
        delta=1,
        reg_diffusion=True,
        reg_cnn=False,
        reg_similar=False,
        reg_similar_mode='SASC',
        reg_kernel_size=3,
        K_step=1e-3,
        self_learn_K_iter=5,
        crop_win=320
    ):
        super().__init__()
        
        if reg_cnn:
            assert cnn_denoisers is not None
        if reg_diffusion:
            assert scorenet is not None
            
        self.scorenet = scorenet
        self.cnn_denoisers = cnn_denoisers
        
        self.reg_diffusion = reg_diffusion
        self.gamma = gamma
        self.lam = lam
        self.rho = rho
        self.delta = delta

        self.crop_win = crop_win

        self.reg_cnn = reg_cnn

        self.reg_similar = reg_similar
        self.reg_similar_mode = reg_similar_mode
        self.reg_kernel_size = reg_kernel_size
        self.K_step = K_step
        self.self_learn_K_iter = self_learn_K_iter

    def forward(self, data_kspace, mask, sense_map=None, target_rss=None, normlize_input=False, K_gd_iter=5, max_iter=100, verbose=False):
        '''
        data_kspace: torch.Tensor, (c,h,w,2)
        sense_map: torch.Tensor, (c,h,w,2)
        target_rss: np.array, (h,w)
        '''
        
        c,h,w,_ = data_kspace.shape
        if self.reg_similar:
            if self.reg_similar_mode=='SASC':
                self.K = SelfSimilar(3,c,h,w).cuda()
            elif self.reg_similar_mode=='SC':
                self.K = SelfSimilarSC(3,c).cuda()
            else:
                raise NotImplementedError
            self.optimizer_K = torch.optim.Adam(self.K.parameters(), lr=1e-2)
        
        v = iFT(data_kspace, sense_map)
        if normlize_input:
            max_i = rss_complex(ifft2c(data_kspace), dim=COIL_DIM).max()
            data_kspace = data_kspace/max_i
            v = v/max_i

        u = torch.zeros_like(v)
        x = v.clone()
    
        psnrs = []
        rho_k = self.rho
        lam = self.lam
        delta = self.delta

        time1 = time.time()
        for idx in range(max_iter):
            #-----------------------------------------------
            # proximal of the regularization term
            #-----------------------------------------------
            sigma = math.sqrt(lam/rho_k)
            s = sigma
            with torch.no_grad():
                #-----------------------------------------------
                # generative prior
                #-----------------------------------------------
                inputs = x + u
                if self.reg_diffusion:
                    v = self.generative_step(sigma, inputs.permute(0,3,1,2), step_lr=delta).permute(0,2,3,1)
                else:
                    v = inputs
                #-----------------------------------------------
                # deep denoiser prior
                #-----------------------------------------------
                if self.reg_cnn:
                    if self.reg_diffusion:
                        s = (inputs - v).std().item() / (2*math.sqrt(2))
                    v = v.permute(0,3,1,2)
                    v = self.cnn_denoisers(v, s)
                    v = v.permute(0,2,3,1)


                v_sub_u = v - u
                #-----------------------------------------------
                # data consistency
                #-----------------------------------------------
                if self.reg_similar:
                    x = self.data_consistancy_with_K(K_gd_iter,x,v_sub_u,data_kspace,mask,1e-4*rho_k,sense_map)
                else:
                    x = projection_tensor(v_sub_u,data_kspace,mask,rho=1e-4*rho_k,sense_map=sense_map,coil_dim=COIL_DIM)

            if self.reg_similar:
                #-----------------------------------------------
                # self training of the self similar kernel
                #-----------------------------------------------
                self.learn_K(FT(x, sense_map).detach())

            #-----------------------------------------------
            # update the Lagrange multiplier
            #-----------------------------------------------
            u = x - v_sub_u
            
            if target_rss is not None:
                PSNR, SSIM, _ = compare_psnr_ssim_tensor(x, target_rss, crop_win=self.crop_win, coil_dim=COIL_DIM)
                psnrs.append((PSNR, SSIM, time.time()-time1))

            rho_k = self.gamma*rho_k

            if verbose and (target_rss is not None) and (idx%10 == 0):
                print(f'iter: {idx}, sigma: {int(sigma*255), int(s*255)}, PSNR: {PSNR}, SSIM: {SSIM}, TIME: {time.time()-time1}')

        return x, psnrs

    def data_consistancy_with_K(self, K_gd_iter, x, v_sub_u, data_kspace, mask, rho_k, sense_map):
        self.K.eval()
        x = v_sub_u
        for _ in range(K_gd_iter):
            x = FT(x, sense_map)
            grad = self.K(x) - x
            grad_ss = self.K.get_GTx(grad) - grad
            x = (rho_k * (x - self.K_step * grad_ss) + data_kspace)/(rho_k + mask)
            x = iFT(x, sense_map, COIL_DIM)
        return x

    def learn_K(self, kspace):
        n,c,h,w = kspace.shape
        for _ in range(self.self_learn_K_iter):
            self.K.train()
            self.K.zero_grad()
            self.optimizer_K.zero_grad()
            grad = self.K(kspace)- kspace
            loss = 1e-2*grad.pow(2).sum(0).sum(-1).mean().sqrt()
            loss.backward()
            self.optimizer_K.step()
            
    def generative_step(self, sigma, x, step_lr=1):
        x = x.contiguous()
        sigma = torch.tensor(sigma).to(x.device).view(1,1,1,1)
        noise = torch.randn_like(x) * sigma
        x_max = x.pow(2).sum(1).sqrt().max()
        inputs = x/x_max + noise
        logp = self.scorenet(inputs, sigma)
        res = x + (step_lr*sigma**2)*logp
        return res