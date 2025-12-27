"""Gaussian Process regression for epistemic uncertainty in autoencoder latent space.

Reference:
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
    MIT Press. Available at http://www.gaussianprocess.org/gpml/
"""
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DoubleConv
from model_aleatoric import AE_aleatoric, RNN, Hybrid_MD_RNN_AE_aleatoric
from model_aleatoric_lightweight import AE_aleatoric_lightweight
from utils import get_AE_loaders
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_latent_features(model, loader):
    model.eval()
    latents = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            channel_data = data[:, 0:1, :, :, :].float().to(device)
            channel_data = torch.add(channel_data, 1.0).float().to(device)
            latent, _ = model(channel_data, y='get_bottleneck')
            latents.append(latent.cpu().numpy())
            targets.append(target[:, 0, 6, 6, 6].cpu().numpy())
    latents = np.concatenate([l.reshape(l.shape[0], -1) for l in latents], axis=0)
    targets = np.concatenate(targets)
    return latents, targets

def gp_predict(gp, latents):
    y_pred, y_std = gp.predict(latents, return_std=True)
    return y_pred, y_std

def tensor_FIFO_pipe(tensor, x, device):
    """The tensor_FIFO_pipe function acts as a first-in-first-out updater and
    is required for the hybrid models. It takes a tensor 'tensor' containing
    information from previous timesteps and concatenates new information 'x'
    to the front fo 'tensor'. In a FIFO manner, it returns all but the last
    element of 'tensor'.

    Args:
        tensor:
          Object of PyTorch-type tensor containing information from previous
          timesteps.
        x:
          Object of PyTorch-type tensor containing information from current
          timestep.
    Return:
        result:
          Object of PyTorch-type tensor upated in a FIFO manner.
    """
    result = torch.cat((tensor[1:].to(device), x.to(device)))
    return result



def resetPipeline(model):
    """The resetPipeline function resets the model pipeline by setting all
    gradients to zero and setting the model to evaluation mode.

    Args:
        model:
          Object of PyTorch Module class, i.e. the model to be reset.

    Returns:
        NONE:
          This function does not have a return value.
    """
    model.zero_grad()
    model.eval()


def test_forward_overloading():
    """The test_forward_overloading function tests the forward method
    overloading functionality of the AE model.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Testing forward method overloading...')
    _model = AE_aleatoric(device=device, in_channels=1, out_channels=2, features=[4, 8, 16])
    _x = torch.randn(1, 1, 24, 24, 24)
    _y = _model(_x)
    print('Standard forward pass shape:', _y.shape)
    _bottleneck, _skip = _model(_x, y='get_bottleneck')
    print('Bottleneck shape:', _bottleneck.shape)
    _md_output = _model(_bottleneck, y='get_MD_output')
    print('MD output shape:', _md_output.shape)
    print('Forward method overloading test completed.')


if __name__ == "__main__":
    test_forward_overloading() 
