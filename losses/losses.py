import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from .loss_util import weighted_loss
from metrics.psnr_ssim import ssim



_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, X, Y, Z). Predicted tensor.
            target (Tensor): of shape (N, C, X, Y, Z). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, X, Y, Z). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, X, Y, Z). Predicted tensor.
            target (Tensor): of shape (N, C, X, Y, Z). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, X, Y, Z). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, X, Y, Z). Predicted tensor.
            target (Tensor): of shape (N, C, X, Y, Z). Ground truth tensor.
        """
        # assert len(pred.size()) == 4
        assert len(pred.size()) == 5

        # NOTE : Keep self.toY = false. It's unchanged from RGB image-type. Do not use it !!!
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        # assert len(pred.size()) == 4
        assert len(pred.size()) == 5

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3, 4)) + 1e-8).mean()


class SSIMLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(SSIMLoss, self).__init__()

        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)

    def forward(self, img1, img2):
        """
        Args:
            img1 (Tensor): of shape (N, C, X, Y). Predicted tensor.
            img2 (Tensor): of shape (N, C, X, Y). Ground truth tensor.
        """
        return (1 - ssim(img1, img2).mean())


class TVLoss(nn.Module):
    """ TV loss.

    Args:
        loss_weight (float): Loss weight for TV loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(TVLoss, self).__init__()

        self.loss_weight = loss_weight

    def forward(self, pred, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, X, Y, Z). Predicted tensor.
            weight (Tensor, optional): of shape (N, C, X, Y, Z). Element-wise
                weights. Default: None.
        """
        # N, C, X, Y, Z = pred.size()
        N, C, X, Y = pred.size()

        tv_x = torch.abs(pred[:,:,1:,:] - pred[:,:,:-1,:]).sum()
        tv_y = torch.abs(pred[:,:,:,1:] - pred[:,:,:,:-1]).sum()
        # tv_z = torch.abs(pred[:,:,:,:,1:] - pred[:,:,:,:,:-1]).sum()

        return self.loss_weight * (tv_x + tv_y) / (N * C * X * Y)

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x
        x0_v = F.conv2d(x0, self.weight_v, padding=2)
        x0_h = F.conv2d(x0, self.weight_h, padding=2)
        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        return x0
    