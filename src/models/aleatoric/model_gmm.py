"""
GMM-based Aleatoric Uncertainty Model

This module implements a Gaussian Mixture Model (GMM) for aleatoric uncertainty prediction.
The model outputs parameters for a 2-component Gaussian mixture and uses a negative
log-likelihood loss for training.

Architecture:
- Input: 3D volume [B, C, D, H, W]
- Output: 6 channels [mu1, log_var1, mu2, log_var2, w1, w2]
- Loss: Negative log-likelihood of GMM

The GMM provides more flexible uncertainty modeling compared to a single Gaussian,
allowing for multi-modal distributions and better handling of heteroscedastic noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DoubleConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AE_gmm(nn.Module):
    """
    Autoencoder with Gaussian Mixture Model for aleatoric uncertainty prediction.
    
    Outputs parameters for a 2-component Gaussian mixture:
    - mu1, mu2: means of the two components
    - log_var1, log_var2: log variances of the two components  
    - w1, w2: mixture weights (softmax normalized)
    
    The model uses a negative log-likelihood loss that accounts for the mixture
    of Gaussians, providing more flexible uncertainty modeling.
    """
    
    def __init__(self, device, in_channels=1, out_channels=6, features=[4, 8, 16], 
                 n_components=2, activation=nn.ReLU(inplace=True)):
        super(AE_gmm, self).__init__()
        self.device = device
        self.n_components = n_components
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Static sizing like in model.py to avoid channel mismatch
        self.helper_down = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.helper_up_1 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0, bias=False)
        self.helper_up_2 = nn.Conv3d(in_channels=4, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Add a projection layer to handle weight mismatch
        self.final_projection = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Down part of AE
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, activation))
            in_channels = feature

        # Up part of AE
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature, feature, activation))

        # This is the "deepest" part.
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, activation)

    def forward(self, x, y=0, skip_connections=0):
        """Forward pass of the GMM autoencoder.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            y: Mode flag (0: standard, 'get_bottleneck': return bottleneck, 'get_MD_output': decode from bottleneck)
            skip_connections: Unused (for compatibility)
            
        Returns:
            If y=0: GMM parameters [B, 6, D, H, W] with [mu1, log_var1, mu2, log_var2, w1, w2]
            If y='get_bottleneck': Bottleneck representation
            If y='get_MD_output': Decoded output from bottleneck
        """
        # Monitor input
        if torch.isnan(x).any():
            print(f"WARNING: NaN in input - shape: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}]")
        
        if y == 0 or y == 'get_bottleneck':
            # The following for-loop describes the entire (left) contracting side,
            for down in self.downs:
                x = down(x)
                x = self.pool(x)

            # This is the bottleneck
            x = self.helper_down(x)
            x = self.activation(x)
            x = self.bottleneck(x)
            x = self.activation(x)

            if y == 'get_bottleneck':
                return x, skip_connections

            x = self.helper_up_1(x)
            x = self.activation(x)

            # The following for-loop describes the entire (right) expanding side.
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                x = self.ups[idx+1](x)

            x = self.helper_up_2(x)
            
            # Apply final projection to handle weight mismatch
            x = self.final_projection(x)
            
            # Monitor final output
            if torch.isnan(x).any():
                print(f"WARNING: NaN in final output - shape: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}]")

            return x
            
        if y == 'get_MD_output':
            x = self.helper_up_1(x)
            x = self.activation(x)
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                x = self.ups[idx+1](x)
            x = self.helper_up_2(x)
            return x

    @staticmethod
    def gmm_parameters_from_output(output):
        """Extract GMM parameters from network output.
        
        Args:
            output: Network output [B, 6, D, H, W] with [mu1, log_var1, mu2, log_var2, w1, w2]
            
        Returns:
            mu1, log_var1, mu2, log_var2, w1, w2: GMM parameters
        """
        mu1 = output[:, 0:1]  # First component mean
        log_var1 = output[:, 1:2]  # First component log variance
        mu2 = output[:, 2:3]  # Second component mean
        log_var2 = output[:, 3:4]  # Second component log variance
        
        # Mixture weights (apply softmax to ensure they sum to 1)
        w_raw = output[:, 4:6]  # Raw weights
        w_softmax = F.softmax(w_raw, dim=1)
        w1 = w_softmax[:, 0:1]  # First component weight
        w2 = w_softmax[:, 1:2]  # Second component weight
        
        return mu1, log_var1, mu2, log_var2, w1, w2

    @staticmethod
    def gmm_uncertainty_from_params(mu1, log_var1, mu2, log_var2, w1, w2):
        """Compute aleatoric uncertainty from GMM parameters.
        
        Args:
            mu1, log_var1, mu2, log_var2, w1, w2: GMM parameters
            
        Returns:
            mean_pred: Mixture mean
            aleatoric_std: Mixture standard deviation
        """
        # Compute mixture mean
        mean_pred = w1 * mu1 + w2 * mu2
        
        # Compute mixture variance
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        
        # Mixture variance formula: E[X^2] - E[X]^2
        # E[X^2] = w1 * (var1 + mu1^2) + w2 * (var2 + mu2^2)
        # E[X]^2 = (w1 * mu1 + w2 * mu2)^2
        mixture_var = w1 * (var1 + mu1**2) + w2 * (var2 + mu2**2) - mean_pred**2
        
        # Ensure positive variance
        mixture_var = torch.clamp(mixture_var, min=1e-10)
        aleatoric_std = torch.sqrt(mixture_var)
        
        return mean_pred, aleatoric_std


class GMMLoss(nn.Module):
    """
    Negative log-likelihood loss for Gaussian Mixture Model with component diversity regularization.
    
    The loss function computes the negative log-likelihood of the target data
    under the predicted GMM distribution, providing a principled way to train
    the model for both prediction and uncertainty estimation.
    
    Added component diversity regularization to prevent single-component convergence
    that was causing flat predictions.
    """
    
    def __init__(self, eps=1e-8, diversity_weight=0.1):
        super(GMMLoss, self).__init__()
        self.eps = eps
        self.diversity_weight = diversity_weight  # Weight for component diversity regularization
        
    def forward(self, output, target):
        """
        Compute GMM negative log-likelihood loss.
        
        Args:
            output: Network output [B, 6, D, H, W] with GMM parameters
            target: Target data [B, 1, D, H, W]
            
        Returns:
            loss: Negative log-likelihood loss
        """
        # Extract GMM parameters
        mu1, log_var1, mu2, log_var2, w1, w2 = AE_gmm.gmm_parameters_from_output(output)
        
        # Compute variances
        var1 = torch.exp(log_var1) + self.eps
        var2 = torch.exp(log_var2) + self.eps
        
        # Compute log-likelihoods for each component
        # log N(x|mu, var) = -0.5 * log(2*pi*var) - 0.5 * (x-mu)^2 / var
        log_likelihood1 = -0.5 * (torch.log(2 * np.pi * var1) + (target - mu1)**2 / var1)
        log_likelihood2 = -0.5 * (torch.log(2 * np.pi * var2) + (target - mu2)**2 / var2)
        
        # Compute mixture log-likelihood
        # log p(x) = log(w1 * p1(x) + w2 * p2(x))
        # Use log-sum-exp trick for numerical stability
        max_log_likelihood = torch.max(log_likelihood1, log_likelihood2)
        mixture_log_likelihood = max_log_likelihood + torch.log(
            w1 * torch.exp(log_likelihood1 - max_log_likelihood) + 
            w2 * torch.exp(log_likelihood2 - max_log_likelihood) + self.eps
        )
        
        # Return negative log-likelihood (loss to minimize)
        nll_loss = -mixture_log_likelihood.mean()
        
        # Component diversity regularization
        # Encourage the two components to be different by penalizing similarity
        mean_diff = torch.abs(mu1 - mu2).mean()
        var_diff = torch.abs(torch.exp(log_var1) - torch.exp(log_var2)).mean()
        
        # Weight diversity penalty (inverse of difference)
        diversity_loss = 1.0 / (mean_diff + var_diff + self.eps)
        
        # Combined loss
        total_loss = nll_loss + self.diversity_weight * diversity_loss
        
        return total_loss


if __name__ == "__main__":
    # Test the GMM model
    print("Testing AE_gmm model...")
    model = AE_gmm(device=device, in_channels=1, out_channels=6, features=[4, 8, 16], n_components=2).to(device)
    
    # Create a dummy input
    dummy_input = torch.randn(2, 1, 24, 24, 24).to(device)
    dummy_target = torch.randn(2, 1, 24, 24, 24).to(device)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Extract GMM parameters
    mu1, log_var1, mu2, log_var2, w1, w2 = AE_gmm.gmm_parameters_from_output(output)
    print(f"GMM parameters shapes:")
    print(f"  mu1: {mu1.shape}, log_var1: {log_var1.shape}")
    print(f"  mu2: {mu2.shape}, log_var2: {log_var2.shape}")
    print(f"  w1: {w1.shape}, w2: {w2.shape}")
    
    # Compute uncertainty
    mean_pred, aleatoric_std = AE_gmm.gmm_uncertainty_from_params(mu1, log_var1, mu2, log_var2, w1, w2)
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Aleatoric std shape: {aleatoric_std.shape}")
    
    # Test loss function
    criterion = GMMLoss()
    loss = criterion(output, dummy_target)
    print(f"GMM Loss: {loss.item():.6f}")
    
    # Test get_bottleneck
    bottleneck, _ = model(dummy_input, y='get_bottleneck')
    print(f"Bottleneck shape: {bottleneck.shape}")
    
    # Test get_MD_output
    md_output = model(bottleneck, y='get_MD_output')
    print(f"MD output shape: {md_output.shape}")
    
    print("AE_gmm model test completed successfully!")
