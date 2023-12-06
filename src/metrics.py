import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def mse(image_target, image):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((image_target - image) ** 2)

def nmse(image_target, image):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(image_target - image) ** 2 / np.linalg.norm(image_target) ** 2

def psnr(image_target, image, data_range=1.):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(image_target, image, data_range=data_range)

def ssim(image_target, image, data_range=1.):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        image_target, image, multichannel=False, data_range=data_range
    )