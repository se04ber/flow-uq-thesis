import torch
import torch.nn as nn
import numpy as np
from model import DoubleConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AE_evidential(nn.Module):
	"""Autoencoder head with evidential regression (Normal-Inverse-Gamma).
	Outputs 4 channels per voxel: mu, log_lambda, log_alpha, log_beta.

	Interpretation (Amini et al., 2020):
	- mu: predictive mean
	- lambda > 0: evidence precision (via softplus on log_lambda)
	- alpha > 1, beta > 0: Normal-Inverse-Gamma hyper-parameters (via softplus on logs)

	Uncertainty decomposition per voxel and timestep:
	- Aleatoric variance:  beta / (alpha - 1)
	- Epistemic variance:  beta / (lambda * (alpha - 1))

	We provide simple helpers to map network outputs to parameters and to compute
	aleatoric/epistemic maps without prescribing a specific training loss here.
	"""
	def __init__(self, device, in_channels=1, out_channels=4, features=[4, 8, 16], activation=nn.ReLU(inplace=True)):
		super(AE_evidential, self).__init__()
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
		# self.final_projection = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

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
		
	def load_weights_selectively(self, state_dict):
		"""Load weights selectively, handling mismatched final layer"""
		model_dict = self.state_dict()
		
		# Filter out incompatible keys (final layer)
		compatible_dict = {k: v for k, v in state_dict.items() 
						  if k in model_dict and model_dict[k].shape == v.shape}
		
		print(f"Loading {len(compatible_dict)}/{len(state_dict)} compatible layers")
		print(f"Skipped layers: {[k for k in state_dict.keys() if k not in compatible_dict]}")
		
		model_dict.update(compatible_dict)
		self.load_state_dict(model_dict)
		
		# Initialize missing layers with default weights
		missing_layers = [k for k in model_dict.keys() if k not in compatible_dict]
		if missing_layers:
			print(f"Initializing missing layers: {missing_layers}")
			# final_projection will be initialized with default PyTorch weights
		
		# Initialize final layer with reasonable values for evidential output
		#with torch.no_grad():
			# Initialize final projection layer
			# if hasattr(self, 'final_projection'):
			#     # Initialize as identity-like transformation
			#     nn.init.eye_(self.final_projection.weight)
			#     if self.final_projection.bias is not None:
			#         nn.init.zeros_(self.final_projection.bias)
			#     print("Initialized final_projection layer as identity transformation")
			
		# Initialize helper_up_2 if it was skipped
		if 'helper_up_2.weight' not in compatible_dict:
			print("Initializing helper_up_2 with evidential-friendly weights")
			with torch.no_grad():
				nn.init.normal_(self.helper_up_2.weight, mean=0.0, std=0.01)

	def monitor_weights(self, epoch=None):
		"""Monitor weight health during training"""
		nan_count = 0
		inf_count = 0
		total_params = 0
		
		for name, param in self.named_parameters():
			total_params += param.numel()
			if torch.isnan(param).any():
				nan_count += param.numel()
			if torch.isinf(param).any():
				inf_count += param.numel()
		
		if nan_count > 0 or inf_count > 0:
			print(f"Epoch {epoch}: Weight health - NaN: {nan_count}/{total_params}, Inf: {inf_count}/{total_params}")
		
		return nan_count == 0 and inf_count == 0

	def monitor_uncertainties(self, pred, target, epoch=None):
		"""Monitor uncertainty calibration during training"""
		mu, v, alpha, beta = self.output_to_nig_params(pred)
		ale, epi = self.nig_uncertainties_from_params(mu, v, alpha, beta)
		
		# Calculate calibration metrics
		error = torch.abs(target - mu)
		ale_mean = ale.mean().item()
		epi_mean = epi.mean().item()
		error_mean = error.mean().item()
		
		if epoch and epoch % 20 == 0:
			print(f"Epoch {epoch}: Aleatoric={ale_mean:.4f}, Epistemic={epi_mean:.4f}, Error={error_mean:.4f}")
		
		return ale_mean, epi_mean, error_mean

	def forward(self, x, y=0, skip_connections=0):
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
				print(f"Evidential bottleneck shape: {x.shape}")
				return x, skip_connections

			x = self.helper_up_1(x)
			x = self.activation(x)

			# The following for-loop describes the entire (right) expanding side.
			for idx in range(0, len(self.ups), 2):
				x = self.ups[idx](x)
				x = self.ups[idx+1](x)

			x = self.helper_up_2(x)
			
			# Apply final projection to handle weight mismatch
			# x = self.final_projection(x)
			
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
	def output_to_nig_params(y: torch.Tensor, v_offset=1e-3, alpha_offset=1e-3, beta_offset=1e-3):
		"""Map raw network outputs to NIG parameters.
		Args:
			y: tensor [..., 4, D, H, W] with channels [mu, logv, logalpha, logbeta]
			v_offset: Offset added to v after softplus (default: 1e-3)
			alpha_offset: Offset added to alpha after softplus (default: 1e-3)
			beta_offset: Offset added to beta after softplus (default: 1e-3)
		Returns:
			mu, v, alpha, beta
		"""
		mu = y[:, 0]
		logv = y[:, 1]
		logalpha = y[:, 2]
		logbeta = y[:, 3]
	
		# Add numerical stability with configurable offsets
		v = torch.nn.functional.softplus(logv) + v_offset
		alpha = torch.nn.functional.softplus(logalpha) + 1.0 + alpha_offset
		beta = torch.nn.functional.softplus(logbeta) + beta_offset
	
		return mu, v, alpha, beta

	@staticmethod
	def nig_uncertainties_from_params(mu: torch.Tensor, v: torch.Tensor, 
								   alpha: torch.Tensor, beta: torch.Tensor):
		"""Compute aleatoric and epistemic uncertainties.

		Returns:
			aleatoric_std, epistemic_std
		"""
		# Aleatoric (data) uncertainty
		aleatoric_var = beta / (alpha - 1.0)

		# Epistemic (model) uncertainty  
		epistemic_var = beta / (v * (alpha - 1.0))

		return (torch.sqrt(torch.clamp(aleatoric_var, min=1e-10)),
				torch.sqrt(torch.clamp(epistemic_var, min=1e-10)))


#Amini et. al 2020)
class EvidentialNIGLoss(nn.Module):
	"""Normal-Inverse-Gamma evidential regression loss (Amini et al., NeurIPS 2020).

	Matches the official implementation at:
	github.com/aamini/evidential-deep-learning
	"""
	def __init__(self, coef_reg: float = 1e-2, v_offset=1e-6, alpha_offset=1e-6, beta_offset=1e-6):
		super().__init__()
		self.coef_reg = coef_reg
		self.v_offset = v_offset
		self.alpha_offset = alpha_offset
		self.beta_offset = beta_offset

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		# pred shape: [B, 4, D, H, W]; target: [B, 1, D, H, W]
		gamma = pred[:, 0:1]  # mean
		logv = pred[:, 1:2]   # log(nu) - virtual observations
		logalpha = pred[:, 2:3]  # log(alpha)
		logbeta = pred[:, 3:4]	 # log(beta)

		# Apply softplus to ensure positivity with numerical stability (configurable offsets)
		v = torch.nn.functional.softplus(logv) + self.v_offset
		alpha = torch.nn.functional.softplus(logalpha) + 1.0 + self.alpha_offset
		beta = torch.nn.functional.softplus(logbeta) + self.beta_offset

		y = target

		# NLL term (exactly as in official implementation)
		twoBlambda = 2.0 * beta * (1.0 + v)

		# Add numerical stability to log operations
		log_pi_v = torch.log(torch.tensor(np.pi, device=pred.device) / v)
		log_twoBlambda = torch.log(twoBlambda + 1e-8)
		log_v_error = torch.log(v * (y - gamma)**2 + twoBlambda + 1e-8)
		
		nll = 0.5 * log_pi_v \
			- alpha * log_twoBlambda \
			+ (alpha + 0.5) * log_v_error \
			+ torch.lgamma(alpha) \
			- torch.lgamma(alpha + 0.5)

		nll = nll.mean()

		# Regularizer (exactly as in official implementation)
		error = torch.abs(y - gamma)
		evi = 2.0 * v + alpha  # evidence
		reg = (error * evi).mean()

		return nll + self.coef_reg * reg


def np_log_pi():
	# Avoid importing numpy to keep this module lightweight
	return torch.tensor(1.1447298858494002, device=device)		# log(pi)


def log_two():
	return torch.tensor(0.6931471805599453, device=device)		# log(2)
