"""
Training script for the convolutional autoencoder with evidential regression
(Normal-Inverse-Gamma). Closely mirrors existing training scripts to enable
comparability. The training and model functionality is based on work from Amini et al. (2020).
(arXiv preprint arXiv:2006.10562.)
"""


import torch
import random
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import numpy as np
import platform
import time
import datetime
import matplotlib.pyplot as plt

from model_evidential import AE_evidential, EvidentialNIGLoss
from utils import get_AE_loaders


torch.manual_seed(10)
random.seed(10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1
#try:
#	mp.set_start_method('spawn')
#except RuntimeError:
#	pass
#LOAD_MODEL = False

#Copy here for having full main training functionality in one script
def train_epoch(loader, model, optimizer, criterion):
	model.train()
	epoch_loss = 0.0
	count = 0
	for _batch_idx, (_data_0, _targ_0) in enumerate(loader):
		# _data_0: [T, C=1, D, H, W]
		t, c, h, d, w = _data_0.shape
		_data = _data_0.flatten(start_dim=0, end_dim=1).float().to(device)
		_data = _data.view(-1, 1, h, d, w).float().to(device)
		_targ = _targ_0.flatten(start_dim=0, end_dim=1).float().to(device)
		_targ = _targ.view(-1, 1, h, d, w).float().to(device)
		pred = model(_data).float().to(device)
		pred = torch.add(pred, -1.0).float().to(device)
		loss = criterion(pred, _targ)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
		optimizer.step()
		optimizer.zero_grad()
		epoch_loss += loss.item()
		count += 1
	return epoch_loss / max(1, count)

def valid_epoch(loader, model, criterion):
	model.eval()
	epoch_loss = 0.0
	count = 0
	with torch.no_grad():
		for _batch_idx, (_data_0, _targ_0) in enumerate(loader):
			t, c, h, d, w = _data_0.shape
			_data = _data_0.flatten(start_dim=0, end_dim=1).float().to(device)
			_data = _data.view(-1, 1, h, d, w).float().to(device)
			_targ = _targ_0.flatten(start_dim=0, end_dim=1).float().to(device)
			_targ = _targ.view(-1, 1, h, d, w).float().to(device)
			pred = model(_data).float().to(device)
			pred = torch.add(pred, -1.0).float().to(device)
			loss = criterion(pred, _targ)
			epoch_loss += loss.item()
			count += 1
	return epoch_loss / max(1, count)



def compute_evidential_uncertainties(pred_4c: torch.Tensor, beta_offset=0.1):
	"""From network outputs [mu, log_lambda, log_alpha, log_beta] compute
	aleatoric and epistemic std maps using AE_evidential helpers.
	Returns numpy arrays: mean, ale_std, epi_std with shapes [B, D, H, W].
	
	Args:
		pred_4c: Network output tensor [B, 4, D, H, W]
		beta_offset: Softplus offset for beta parameter (default: 0.1, based on finetuning results)
	"""
	mu, v, alpha, beta = AE_evidential.output_to_nig_params(pred_4c, beta_offset=beta_offset)
	ale, epi = AE_evidential.nig_uncertainties_from_params(mu, v, alpha, beta)
	return (
		mu.detach().cpu().numpy(),
		ale.detach().cpu().numpy(),
		epi.detach().cpu().numpy(),
	)

def trial_1_AE_evidential(alpha, alpha_string, train_loaders, valid_loaders, num_epochs=50, beta_offset=0.1):
    """
    Train evidential AE model.
    
    Args:
        alpha: Learning rate
        alpha_string: Learning rate as string for model identifier
        train_loaders: Training data loaders
        valid_loaders: Validation data loaders
        num_epochs: Number of training epochs
        beta_offset: Softplus offset for beta parameter (default: 0.1, based on finetuning results)
    """
    # Use beta_offset in loss function for consistency (changed from default 1e-6 to match inference)
    criterion = EvidentialNIGLoss(coef_reg=1e-4, beta_offset=beta_offset).to(device)  # Reduced regularization
    
    # Create beta_offset string for model identifier (e.g., 0.1 -> "beta0_1", 1e-3 -> "beta1e-3")
    if beta_offset >= 1e-2:
        # For values >= 0.01, use decimal format (e.g., 0.1 -> "beta0_1")
        beta_string = f"beta{str(beta_offset).replace('.', '_')}"
    else:
        # For smaller values, use scientific notation (e.g., 1e-3 -> "beta1e-3")
        # Convert to string and replace 'e-' with 'e-' (no change needed) or handle differently
        beta_str = f"{beta_offset:.0e}" if beta_offset < 1e-2 else str(beta_offset)
        beta_string = f"beta{beta_str.replace('e-', 'e-')}"
    
    model_identifier = f'EVI_LR{alpha_string}_{beta_string}_i'
    model = AE_evidential(
            device=device,
            in_channels=1,
            out_channels=4,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
            ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=alpha, weight_decay=1e-6)  # Added weight decay
    train_losses = []
    valid_losses = []
    print('Beginning evidential training.')
    for epoch in range(num_epochs):
        print('Hardware: ', platform.processor())
        start = time.time()
        avg_train = 0.0
        for train_loader in train_loaders:
            avg_train += train_epoch(train_loader, model, optimizer, criterion)
        avg_train /= len(train_loaders)
        end = time.time()
        print('Duration of one ML Calculation = 1 Coupling Cycle: ', end - start)
        avg_valid = 0.0
        for valid_loader in valid_loaders:
            avg_valid += valid_epoch(valid_loader, model, criterion)
        avg_valid /= len(valid_loaders)
        print('------------------------------------------------------------')
        print(f'[{model_identifier}] Training Epoch: {epoch+1}')
        print(f'[{model_identifier}] -> Avg evidential train loss {avg_train:.3f}')
        print(f'[{model_identifier}] -> Avg evidential valid loss {avg_valid:.3f}')
        # Monitor weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.monitor_weights(epoch + 1)
        #Additionally: Monitor uncertainties every 20 epochs
        if (epoch + 1) % 20 == 0:
            # Get a sample from validation set for uncertainty monitoring
            model.eval()
            with torch.no_grad():
                for data, target in valid_loaders[0]:
                    t, c, h, d, w = data.shape
                    _data = data.flatten(start_dim=0, end_dim=1).float().to(device)
                    _data = _data.view(-1, 1, h, d, w).float().to(device)
                    _targ = target.flatten(start_dim=0, end_dim=1).float().to(device)
                    _targ = _targ.view(-1, 1, h, d, w).float().to(device)
                    
                    pred = model(_data).float().to(device)
                    pred = torch.add(pred, -1.0).float().to(device)
                    model.monitor_uncertainties(pred, _targ, epoch + 1)
                    break  # Only use first batch for monitoring
            model.train()
        
        train_losses.append(avg_train)
        valid_losses.append(avg_valid)
    # Save artifacts
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'Model_AE_evidential_LR{alpha_string}_{beta_string}_seed0_{timestamp}.pth'
    torch.save(model.state_dict(), model_name)
    np.save('train_losses_evidential.npy', np.array(train_losses))
    np.save('valid_losses_evidential.npy', np.array(valid_losses))
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.legend()
    plt.title('Evidential Loss Curves')
    plt.tight_layout()
    plt.savefig('evidential_loss_curves.png')
    #plt.close()
    # Export example mean/aleatoric/epistemic on validation set for comparison

    means = []
    tales = []
    epis = []
    model.eval()
    with torch.no_grad():
        for data, target in valid_loaders[0]:
            data_x = torch.add(data[:, 0:1], 1.0).float().to(device)
            y = model(data_x)
            y = torch.add(y, -1.0)
            m, a, e = compute_evidential_uncertainties(y, beta_offset=beta_offset)
            means.append(m)
            tales.append(a)
            epis.append(e)
    means = np.concatenate(means, axis=0)
    tales = np.concatenate(tales, axis=0)
    epis = np.concatenate(epis, axis=0)
    np.save('evidential_mean.npy', means)
    np.save('evidential_aleatoric.npy', tales)
    np.save('evidential_epistemic.npy', epis)
    return 


def main():
    train_loaders, valid_loaders = get_AE_loaders(
            data_distribution='get_KVS',
            batch_size=2, #1 #32
            shuffle=True
	)
    alpha = 1e-4
    alpha_string = '0_0001'
    beta_offset = 0.1 # Softplus beta offset: 0.1 based on finetuning results (changed from default 1e-3)
    trial_1_AE_evidential(alpha, alpha_string, train_loaders, valid_loaders, num_epochs=50, beta_offset=beta_offset) 
    

main()