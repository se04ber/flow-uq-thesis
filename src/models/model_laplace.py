"""model_laplace2

This script contains AE models with true sparse Laplace approximation uncertainty quantification.
The Laplace approximation estimates the posterior distribution of model parameters
by fitting a Gaussian distribution to the posterior at the MAP estimate using
diagonal Hessian approximation for computational efficiency.
Strongly based on Daxberger et al. (2021). (arXiv:2106.14806. https://github.com/aleximmer/Laplace)
"""
import torch.nn as nn
import torch
import numpy as np
from torch.distributions import MultivariateNormal
from model import AE as AE_laplace2
from model import DoubleConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SparseLaplaceApproximation_AE:
    """Sparse Laplace approximation using diagonal
    Hessian approximation for computational efficiency. It estimates the posterior
    distribution of model parameters with a Gaussian distribution.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.is_fitted = False
        self.prior_precision = 1e-4  # Prior precision (lambda)
        self.hessian_diag = None
        self.posterior_var_diag = None
        self.map_params = None
        
    def fit(self, train_loader, criterion, num_samples=1000):
        """Fit the sparse Laplace approximation to the model.
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            num_samples: Number of samples for Hessian estimation
        """
        print("Fitting sparse Laplace approximation for AE")
        # Get MAP parameters
        self.map_params = self._get_parameters()
        # Estimate diagonal Hessian using finite differences
        self.hessian_diag = self._estimate_diagonal_hessian(train_loader, criterion, num_samples)
        # Compute posterior variance (diagonal)
        self.posterior_var_diag = self._compute_posterior_variance_diagonal()
        self.is_fitted = True
        print("Sparse Laplace approximation fitted successfully for AE.")
        
    def _get_parameters(self):
        """Get current model parameters as a flat vector."""
        params = []
        for param in self.model.parameters():
            if param.requires_grad:
                params.append(param.view(-1))
        return torch.cat(params)
    
    def _estimate_diagonal_hessian(self, train_loader, criterion, num_samples):
        """Estimate diagonal of Hessian matrix using finite differences."""
        print("Estimating diagonal Hessian for AE...")
        # Get current parameters
        original_params = self._get_parameters()
        n_params = len(original_params)
        # Initialize diagonal Hessian
        hessian_diag = torch.zeros(n_params, device=self.device)
        # Collect data samples
        data_samples = []
        target_samples = []
        sample_count = 0
        
        for batch_data, batch_targets in train_loader:
            if sample_count >= num_samples:
                break
            data_samples.append(batch_data.to(self.device))
            target_samples.append(batch_targets.to(self.device))
            sample_count += len(batch_data)
        
        if not data_samples:
            raise ValueError("No data samples available for Hessian estimation")
        
        data = torch.cat(data_samples, dim=0)[:num_samples]
        targets = torch.cat(target_samples, dim=0)[:num_samples]
        
        # Estimate diagonal elements using finite differences
        epsilon = 1e-4
        for i in range(n_params):
            if i % 1000 == 0:
                print(f"Computing Hessian diagonal: {i}/{n_params}") 
            # Create temporary parameters for perturbation
            temp_params_plus = original_params.clone()
            temp_params_plus[i] += epsilon 
            temp_params_minus = original_params.clone()
            temp_params_minus[i] -= epsilon 
            # Forward difference
            self._set_parameters(temp_params_plus)
            loss_plus = self._compute_loss(data, targets, criterion) 
            # Backward difference
            self._set_parameters(temp_params_minus)
            loss_minus = self._compute_loss(data, targets, criterion) 
            # Central difference (original parameters)
            self._set_parameters(original_params)
            loss_center = self._compute_loss(data, targets, criterion) 
            # Second derivative approximation
            hessian_diag[i] = (loss_plus - 2 * loss_center + loss_minus) / (epsilon ** 2)
        # Add prior precision to diagonal
        hessian_diag += self.prior_precision
        return hessian_diag
    
    def _set_parameters(self, params):
        """Set all model parameters from a flat tensor."""
        param_idx = 0
        for param in self.model.parameters():
            if param.requires_grad:
                param_size = param.numel()
                param_data = params[param_idx:param_idx + param_size].view(param.shape)
                param.data.copy_(param_data)
                param_idx += param_size
    
    def _compute_loss(self, data, targets, criterion):
        """Compute loss for given data and targets."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            loss = criterion(outputs, targets)
        return loss.item()
    
    def _compute_posterior_variance_diagonal(self):
        """Compute diagonal posterior variance."""
        print("Computing posterior variance (diagonal) for AE...")
        
        # Posterior variance is inverse of Hessian diagonal
        posterior_var_diag = 1.0 / self.hessian_diag
        
        return posterior_var_diag
    
    def predict_with_uncertainty(self, x):
        """Make predictions with uncertainty estimates using sparse Laplace.
        
        Args:
            x: Input data
            
        Returns:
            mean_pred: Mean prediction
            epistemic_std: Epistemic uncertainty (standard deviation)
        """
        if not self.is_fitted:
            raise RuntimeError("Sparse Laplace approximation not fitted. Call fit() first.")
        
        self.model.eval()
        # Get mean prediction
        with torch.no_grad():
            mean_pred = self.model(x)
        # Estimate epistemic uncertainty using sparse Laplace
        epistemic_std = self._estimate_epistemic_uncertainty_sparse(x)
        return mean_pred, epistemic_std
    
    def _estimate_epistemic_uncertainty_sparse(self, x):
        """Estimate epistemic uncertainty using true sparse diagonal Laplace approximation.
        
        This computes the proper parameter-space uncertainty around MAP estimate
        using the diagonal posterior covariance matrix.
        """
        self.model.eval()
        
        # Compute Jacobian of model output with respect to model parameters
        param_jacobian = self._compute_parameter_jacobian(x)
        
        # True sparse Laplace uncertainty propagation:
        # epistemic_var = diag(J_θ @ diag(Σ_θ) @ J_θ^T)
        # where J_θ is parameter Jacobian and Σ_θ is diagonal posterior covariance
        
        # For each output dimension, compute uncertainty
        epistemic_var = torch.zeros(x.shape[0], param_jacobian.shape[1], device=self.device)
        
        for i in range(param_jacobian.shape[1]):  # For each output dimension
            # Get parameter Jacobian for this output dimension: (batch_size, n_params)
            j_theta_i = param_jacobian[:, i, :]
            
            # Compute epistemic variance using diagonal posterior covariance
            # Var[f(x)] = Σ_j (∂f/∂θ_j)² * Var[θ_j]
            epistemic_var[:, i] = torch.sum(
                j_theta_i.pow(2) * self.posterior_var_diag.unsqueeze(0), 
                dim=1
            )
        epistemic_std = torch.sqrt(epistemic_var)
        return epistemic_std
    
    def _compute_parameter_jacobian(self, x):
        """Compute Jacobian of model output with respect to model parameters.
        Test for post-hoc approximation.

        Computes ∂f(x)/∂θ for each parameter θ using efficient autograd.
        Args:
            x: Input data (batch_size, channels, height, width, depth)
            
        Returns:
            jacobian: Jacobian matrix (batch_size, output_channels, n_params)
        """
        batch_size = x.shape[0]
        n_params = len(self.map_params)
        # Get model output to determine output dimensions
        with torch.no_grad():
            output = self.model(x)
        output_channels = output.shape[1]  # AE has channel dimension
        # Initialize Jacobian matrix
        jacobian = torch.zeros(batch_size, output_channels, n_params, device=self.device)
      
        for param in self.model.parameters():
            param.requires_grad_(True)
        # Compute Jacobian using autograd for each output channel
        for out_idx in range(output_channels):
            # Forward pass
            output = self.model(x)
            # Get gradients w.r.t. parameters for this output channel
            grad_outputs = torch.zeros_like(output)
            grad_outputs[:, out_idx:out_idx+1] = 1.0
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=output,
                inputs=list(self.model.parameters()),
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            # Flatten and store gradients
            param_idx = 0
            for grad in grads:
                if grad is not None:
                    grad_flat = grad.flatten()
                    total_elements = grad_flat.numel()
                    if total_elements == 0:
                        continue
                    if total_elements >= batch_size and total_elements % batch_size == 0:
                        param_size = total_elements // batch_size
                        grad_reshaped = grad_flat.view(batch_size, param_size)
                    else:
                        param_size = total_elements
                        grad_reshaped = grad_flat.unsqueeze(0).expand(batch_size, param_size) 
                    jacobian[:, out_idx, param_idx:param_idx + param_size] = grad_reshaped
                    param_idx += param_size
        return jacobian

def jacobian_norm_uncertainty_laplace2(model, data, t_max=100):
    """Compute Jacobian norm uncertainty for Laplace2 method."""
    norms = []
    means = []
    for i in range(min(t_max, data.shape[0])):
        x = data[i:i+1, 0:1].clone().detach().requires_grad_(True)
        y = model(x)
        y = torch.add(y, -1.0)
        mu = y[:, 0:1]
        means.append(mu.detach().cpu().numpy()[0,0])
        grad_outputs = torch.ones_like(mu)
        grads = torch.autograd.grad(outputs=mu, inputs=x, grad_outputs=grad_outputs, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grads is None:
            gnorm = 0.0
        else:
            gnorm = grads.pow(2).mean().sqrt().item()
        norms.append(gnorm)
    return np.array(means), np.array(norms)


def main():
    # Testing
    model = AE_laplace2(device=device, in_channels=1, out_channels=2, features=[4, 8, 16])
    # Test forward pass
    x = torch.randn(1, 1, 24, 24, 24).to(device)
    output = model(x)
    print(f"Forward Output shape: {output.shape}")
    bottleneck, skip = model(x, y='get_bottleneck')
    print(f"Bottleneck shape: {bottleneck.shape}")
    md_output = model(bottleneck, y='get_MD_output')
    print(f"MD output shape: {md_output.shape}")
    
main()