"""
Prediction Interval (PI) Model for Aleatoric Uncertainty

This module implements a Prediction Interval model based on the PIVEN approach
(Simhayev et al., 2020) for aleatoric uncertainty prediction.

The model outputs three values per voxel:
- Point prediction (mean)
- Lower prediction interval bound
- Upper prediction interval bound

The loss function combines three objectives:
1. Coverage: PI should contain true values ~(1-alpha) of the time
2. Width: PI should not be unnecessarily wide
3. Accuracy: Point prediction should be accurate

Reference:
Simhayev, Eli, Gilad Katz, and Lior Rokach. "PIVEN: A Deep Neural Network for 
Prediction Intervals with Specific Value Prediction." arXiv preprint arXiv:2006.05139 (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import DoubleConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AE_pi(nn.Module):
    """
    Autoencoder with Prediction Interval output for aleatoric uncertainty prediction.
    
    Outputs three channels per voxel:
    - mu: point prediction (mean)
    - lower: lower prediction interval bound
    - upper: upper prediction interval bound
    
    The model uses a PIVEN-style loss that balances coverage, width, and accuracy.
    """
    
    def __init__(self, device, in_channels=1, out_channels=3, features=[4, 8, 16], 
                 activation=nn.ReLU(inplace=True)):
        super(AE_pi, self).__init__()
        self.device = device
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
        """Forward pass of the PI autoencoder.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            y: Mode flag (0: standard, 'get_bottleneck': return bottleneck, 'get_MD_output': decode from bottleneck)
            skip_connections: Unused (for compatibility)
            
        Returns:
            If y=0: PI parameters [B, 3, D, H, W] with [mu, lower, upper]
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
    def pi_parameters_from_output(output):
        """Extract PI parameters from network output.
        
        Args:
            output: Network output [B, 3, D, H, W] with [mu, lower, upper]
            
        Returns:
            mu, lower, upper: PI parameters
        """
        mu = output[:, 0:1]  # Point prediction
        lower = output[:, 1:2]  # Lower bound
        upper = output[:, 2:3]  # Upper bound
        
        return mu, lower, upper

    @staticmethod
    def pi_uncertainty_from_params(mu, lower, upper):
        """Compute aleatoric uncertainty from PI parameters.
        
        Args:
            mu, lower, upper: PI parameters
            
        Returns:
            mean_pred: Point prediction
            aleatoric_std: Half the PI width as uncertainty measure
        """
        mean_pred = mu
        # Use half the PI width as uncertainty measure
        aleatoric_std = 0.5 * (upper - lower)
        
        return mean_pred, aleatoric_std


class PILoss(nn.Module):
    """
    Prediction Interval loss function based on PIVEN approach.
    
    The loss combines three objectives:
    1. Coverage loss: Penalize when true values fall outside PI
    2. Width loss: Penalize unnecessarily wide PIs
    3. Accuracy loss: MSE between point prediction and target
    
    Reference: Simhayev et al. (2020) - PIVEN paper
    """
    
    def __init__(self, alpha=0.05, lambda_=2.0, soften=160.0, eps=1e-8):
        """
        Initialize PI loss.
        
        Args:
            alpha: Desired significance level (1-alpha coverage)
            lambda_: Weight for width penalty vs coverage (updated from 25.0 to 2.0)
            soften: Softening parameter for numerical stability
            eps: Small constant for numerical stability
            
        Note: Previous lambda_=25.0 was too high, causing flat predictions at mean value
        instead of following temporal dynamics. Reduced to 2.0 for better balance.
        """
        super(PILoss, self).__init__()
        self.alpha = alpha
        self.lambda_ = lambda_
        self.soften = soften
        self.eps = eps
        
    def forward(self, output, target):
        """
        Compute PI loss.
        
        Args:
            output: Network output [B, 3, D, H, W] with [mu, lower, upper]
            target: Target data [B, 1, D, H, W]
            
        Returns:
            loss: Combined PI loss
        """
        # Extract PI parameters
        mu, lower, upper = AE_pi.pi_parameters_from_output(output)
        
        # Ensure lower <= upper (soft constraint)
        width = upper - lower
        width_penalty = F.relu(-width)  # Penalty when lower > upper
        
        # 1. Coverage loss: penalize when target falls outside PI
        # Use soft indicator function for differentiability
        outside_lower = F.relu(lower - target)  # Positive when target < lower
        outside_upper = F.relu(target - upper)  # Positive when target > upper
        outside_pi = outside_lower + outside_upper
        
        # Soft coverage loss using sigmoid
        coverage_loss = torch.sigmoid(self.soften * outside_pi).mean()
        
        # 2. Width loss: penalize wide PIs
        width_loss = width.mean()
        
        # 3. Accuracy loss: MSE between point prediction and target
        accuracy_loss = F.mse_loss(mu, target)
        
        # Combined loss
        total_loss = accuracy_loss + self.lambda_ * coverage_loss + width_loss + width_penalty.mean()
        
        return total_loss


class QuantileLoss(nn.Module):
    """Quantile loss for comparison (from existing code)."""
    def __init__(self, quantiles=[0.05, 0.95]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
    def forward(self, pred, target):
        # pred: (batch, 2, ...), target: (batch, 1, ...)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, i:i+1, ...]
            losses.append(torch.max((q-1)*errors, q*errors).mean())
        return sum(losses)


if __name__ == "__main__":
    # Test the PI model
    print("Testing AE_pi model...")
    model = AE_pi(device=device, in_channels=1, out_channels=3, features=[4, 8, 16]).to(device)
    
    # Create a dummy input
    dummy_input = torch.randn(2, 1, 24, 24, 24).to(device)
    dummy_target = torch.randn(2, 1, 24, 24, 24).to(device)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Extract PI parameters
    mu, lower, upper = AE_pi.pi_parameters_from_output(output)
    print(f"PI parameters shapes:")
    print(f"  mu: {mu.shape}, lower: {lower.shape}, upper: {upper.shape}")
    
    # Compute uncertainty
    mean_pred, aleatoric_std = AE_pi.pi_uncertainty_from_params(mu, lower, upper)
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Aleatoric std shape: {aleatoric_std.shape}")
    
    # Test loss function
    criterion = PILoss(alpha=0.05, lambda_=25.0)
    loss = criterion(output, dummy_target)
    print(f"PI Loss: {loss.item():.6f}")
    
    # Test get_bottleneck
    bottleneck, _ = model(dummy_input, y='get_bottleneck')
    print(f"Bottleneck shape: {bottleneck.shape}")
    
    # Test get_MD_output
    md_output = model(bottleneck, y='get_MD_output')
    print(f"MD output shape: {md_output.shape}")
    
    print("AE_pi model test completed successfully!")
