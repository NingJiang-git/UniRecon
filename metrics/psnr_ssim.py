import cv2
import numpy as np

import skimage.metrics
import torch
import torchmetrics
from torch.nn import functional as F
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor
from torchmetrics.utilities.distributed import reduce
from torch.autograd import Variable
from math import exp
from torchmetrics.regression import MeanSquaredError

def calculate_psnr(img1,
                   img2):
    psnr = torchmetrics.PeakSignalNoiseRatio(data_range=2)
    return psnr(img1,img2)

def calculate_ssim(img1,
                   img2):
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=2)
    return ssim(img1,img2)

def calculate_rmse(img1,
                   img2):
    mse = MeanSquaredError()
    return torch.sqrt(mse(img1,img2))




def gaussian(window_size, sigma):
    # print('window_size', window_size)
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window



def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = (0.01 * 2)**2
    C2 = (0.03 * 2)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size = 11, size_average = False):
    channel = 1
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)