import os
import cv2
import torch
import numpy as np
import random
import shutil

from .metrics import psnr, ssim
from .fastmri.transforms import to_tensor
from .fastmri.fft import fft2c, ifft2c
from .fastmri.coil_combine import rss, rss_complex, sense_reduce, sense_expand
from .fastmri.math import complex_mul, complex_conj

def process_MRI_data(target_np, sense_map=None, coil_dim=0):
    '''
    Args:
        target_np: (c,h,w)
        sense_map: (c,h,w)
    Returns:
        target_tensor: (1,c,h,w,2)
        target_rss: (h,w)
        sense_map: (1,c,h,w,2)
    '''
    if len(target_np.shape) == 3:
        target_rss = np.sqrt(np.sum(np.abs(target_np)**2, coil_dim))
    else:
        target_rss = np.abs(target_np)

    data_range = target_rss.max()
    target_rss = target_rss / data_range
    target_tensor = to_tensor(target_np).float()
    target_tensor = target_tensor / data_range
    if len(target_tensor.shape) == 3:
        target_tensor = target_tensor.unsqueeze(0) # coil dim
    
    target_tensor = target_tensor.unsqueeze(0) # batch dim
    if sense_map is None:
        return target_tensor, target_rss
    else:
        sense_map = to_tensor(sense_map).float().unsqueeze(0)
        return target_tensor, target_rss, sense_map

def tensor2image(img):
    return img.detach().cpu().numpy()
    
def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / (std + 1e-11), mean, std

def normalize_tensor(img):
    '''
    img: (n,c,h,w)
    '''
    mean = img.mean(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
    std = img.std(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
    return (img - mean) / (std + 1e-11), mean, std

def iFT(kspace, sense_map=None, coil_dim=0):
    x = ifft2c(kspace)
    if sense_map is not None:
        x = sense_reduce(x, sense_map, coil_dim)
    return x

def FT(x, sense_map=None):
    if sense_map is not None:
        x = sense_expand(x, sense_map)
    return fft2c(x)

def projection_tensor(image_denoised,data_kspace,mask,rho=2e-2,sense_map=None,coil_dim=0):
    '''
    input shape: (n,c,h,w,2)
    output shape: (n,c,h,w,2)
    '''
    image_projection = rho*FT(image_denoised, sense_map) + data_kspace*mask
    return iFT(image_projection/(rho+mask), sense_map, coil_dim)

def compare_psnr_ssim_tensor(image_, target_, crop_win=0, coil_dim=0):
    image_ = rss_complex(image_, coil_dim).squeeze().cpu().numpy()
    if crop_win > 0:
        image_ = crop_image(image_, crop_win)
        target_ = crop_image(target_, crop_win)
    return compare_psnr_ssim(image_, target_)

def compare_psnr_ssim(image_, target_):
    _, mean_rss, std_rss = normalize(target_)
    image_, _, _ = normalize(image_)
    image_ = (image_*std_rss) + mean_rss
    return psnr(image_, target_), ssim(image_, target_), image_

def crop_image(image, win = 320):
    h, w = image.shape[:2]
    pad_size = (max((win-h)//2,0), max((win-w)//2,0))
    padding_array = [(pad_size[0],max(win-h-pad_size[0],0)), (pad_size[1], max(win-w-pad_size[1],0))]
    padding_array += [(0,0)] * (len(image.shape)-2)
    image = np.pad(image, padding_array, mode='constant')
    h, w = image.shape[:2]
    return image[h//2-win//2:h//2+win//2,w//2-win//2:w//2+win//2,...]

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def get_files(root, ext = ['.jpg','.bmp','.png']):
    files = []
    for file_ in os.listdir(root):
        file_path = os.path.join(root, file_)
        if os.path.isdir(file_path):
            files += get_files(file_path, ext)
        else:
            if ('.' + file_path.split('.')[-1]) in ext:
                files.append(file_path)
    return files

def initial_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)