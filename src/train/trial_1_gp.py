"""Training convolutional autoencoder with Gaussian Process for epistemic uncertainty estimation.

Reference:
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
    MIT Press. Available at http://www.gaussianprocess.org/gpml/
"""

import torch
import random
import copy
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import numpy as np
import platform
import time
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#from model_aleatoric import AE_aleatoric
from model_MCDropout import AE_dropout
from model_aleatoric import AE_aleatoric
import matplotlib.pyplot as plt
from utils import get_AE_loaders, get_RNN_loaders, dataset2csv, mlready2dataset
from plotting import plot_flow_profile, plotPredVsTargKVS, plot_flow_profile_std

torch.manual_seed(10)
random.seed(10)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.05, 0.95]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
    def forward(self, pred, target):
        # pred: (batch, 2, ...), target: (batch, 1, ...)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, i:i+1, ...]
            losses.append(torch.max((q-1)*errors, q*errors).mean())
        return sum(losses) #losses.sum()

class AleatoricLoss(nn.Module):
    """Loss function that includes aleatoric uncertainty estimation.
    This loss function implements the negative log-likelihood loss for
    heteroscedastic regression.
    Loss: 0.5 * exp(-log_var) * (target - mean)^2 + 0.5 * log_var
    """
    def __init__(self):
        super(AleatoricLoss, self).__init__()
    def forward(self, pred, target):
        """
        Args:
            pred: tensor of shape (batch, 2, height, depth, width) 
                 where pred[:, 0, ...] is the mean and pred[:, 1, ...] is log_var
            target: tensor of shape (batch, 1, height, depth, width)
        """
        mean = pred[:, 0:1, :, :, :]  # Extract mean prediction
        log_var = pred[:, 1:2, :, :, :]  # Extract log variance  
        # Ensure log_var doesn't get too negative (numerical stability)
        log_var = torch.clamp(log_var, min=-10, max=10)    
        #NLL loss for heteroscedastic noise
        loss = 0.5 * torch.exp(-log_var) * (target - mean)**2 + 0.5 * log_var
        return loss.mean()



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


def train_AE_aleatoric(loader, model_i, optimizer_i, model_identifier_i, criterion, scaler, current_epoch):
    """The train_AE_aleatoric function trains the aleatoric uncertainty-aware
    single channel model and computes the average loss on the training set.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the model to be trained.
        optimizer:
          The optimization algorithm applied during training.
        criterion:
          The loss function applied to quantify the error.
        scaler:
          Object of torch.cuda.amp.GradScaler to conveniently help perform the
          steps of gradient scaling.
        model_identifier:
          A unique string to identify the model. Here, the learning rate is
          used to identify which model is being trained.
        current_epoch:
          A string containing the current epoch for terminal output.

    Returns:
        avg_loss:
          A double value indicating average training loss for the current epoch.
    """

    _epoch_loss = 0
    _counter = 0

    for _batch_idx, (_data_0, _targ_0) in enumerate(loader):
        t, c, h, d, w = _data_0.shape
        print(_data_0.shape)
        _data = _data_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _data = _data.view(-1, 1, h, d, w).float().to(device)
        _targ = _targ_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _targ = _targ.view(-1, 1, h, d, w).float().to(device)

        print(_data.shape)

        with torch.cuda.amp.autocast():
            _pred = model_i(_data).float().to(device=device)
            _pred = torch.add(_pred, -1.0).float().to(device=device)

            _loss = criterion(_pred, _targ)

            _epoch_loss += _loss.item()
            _counter += 1

        _loss.backward(retain_graph=True)
        optimizer_i.step()
        optimizer_i.zero_grad()

    _avg_loss = _epoch_loss/_counter
    return _avg_loss

def mc_dropout_predict(model, data, mc_samples=20):
    #Sample from model for mc_samples for different dropouts
    model.eval()
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            pred = model(data)
            preds.append(pred.cpu().numpy())
    preds = np.stack(preds, axis=0)
    return preds


def valid_AE_aleatoric(loader, model_i, optimizer_i, model_identifier_i, criterion, scaler, current_epoch):
    """The valid_AE_aleatoric function computes the average loss on a given single
    channel dataset without updating/optimizing the learnable model parameters.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the model to be trained.
        optimizer:
          The optimization algorithm applied during training.
        criterion:
          The loss function applied to quantify the error.
        scaler:
          Object of torch.cuda.amp.GradScaler to conveniently help perform the
          steps of gradient scaling.
        model_identifier:
          A unique string to identify the model. Here, the learning rate is
          used to identify which model is being trained.
        current_epoch:
          A string containing the current epoch for terminal output.

    Returns:
        avg_loss:
          A double value indicating average training loss for the current epoch.
    """

    _epoch_loss = 0
    _counter = 0

    for _batch_idx, (_data_0, _targ_0) in enumerate(loader):
        t, c, h, d, w = _data_0.shape
        _data = _data_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _data = _data.view(-1, 1, h, d, w).float().to(device)
        _targ = _targ_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _targ = _targ.view(-1, 1, h, d, w).float().to(device)

        with torch.cuda.amp.autocast():
            _pred = model_i(_data).float().to(device=device)
            _pred = torch.add(_pred, -1.0).float().to(device=device)

            _loss = criterion(_pred, _targ)

            _epoch_loss += _loss.item()
            _counter += 1

        
    _avg_loss = _epoch_loss/_counter
    return _avg_loss


def get_latentspace_AE_aleatoric(loader, model_i, out_file_name):
    """The get_latentspace_AE_aleatoric function extracts the model-specific latentspace
    for a given dataset and saves it to file.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the model from which to extract
          the latentspace.
        out_file_name:
          A string containing the name of the file that the latentspace should
          be saved to.

    Returns:
        NONE:
          This function does not have a return value. Instead it saves the
          latentspace to file.
    """
    latentspace_x = []
    latentspace_y = []
    latentspace_z = []

    for batch_idx, (_data, _) in enumerate(loader):
        t, c, h, d, w = _data.shape
        _data = torch.add(_data, 1.0).float().to(device)
        _data_x = _data
        #_data_x = _data[:, 0:1, :, :, :]
        #_data_y = _data[:, 1:2, :, :, :]
        #_data_z = _data[:, 2:3, :, :, :]

        with torch.cuda.amp.autocast():
            bottleneck_x, _ = model_i(_data_x,  y='get_bottleneck')
            #bottleneck_y, _ = model_i(_data_y,  y='get_bottleneck')
            #bottleneck_z, _ = model_i(_data_z,  y='get_bottleneck')
            latentspace_x.append(bottleneck_x.cpu().detach().numpy())
            #latentspace_y.append(bottleneck_y.cpu().detach().numpy())
            #latentspace_z.append(bottleneck_z.cpu().detach().numpy())

    np_latentspace_x = np.vstack(latentspace_x)
    #np_latentspace_y = np.vstack(latentspace_y)
    #np_latentspace_z = np.vstack(latentspace_z)

    path = "" #"/home/sabrina/Schreibtisch/Masterarbeit/Diffusion_test/"
    np.save(f"{out_file_name}_x.npy",np_latentspace_x )
    #np.save(f"{out_file_name}_y.npy",np_latentspace_y)
    #np.save(f"{out_file_name}_z.npy",np_latentspace_z)

    dataset2csv(
        dataset=np_latentspace_x,
        dataset_name=f'{out_file_name}_x'
    )
    """
    dataset2csv(
        dataset=np_latentspace_y,
        dataset_name=f'{out_file_name}_y'
    )
    dataset2csv(
        dataset=np_latentspace_z,
        dataset_name=f'{out_file_name}_z'
    )
    """


def get_latentspace_AE_aleatoric_helper():
    """Helper function to extract latentspace from trained aleatoric AE model."""
    _1_train_loaders, _2_valid_loaders = get_AE_loaders(
        data_distribution="get_KVS",
        batch_size=1,
        shuffle=False
    )

    _model_i = AE_aleatoric(
        device=device,
        in_channels=1,
        out_channels=2,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_i.load_state_dict(torch.load(
        'Model_AE_aleatoric_LR0_001_i', #'/home/sabrina/Schreibtisch/Masterarbeit/TrainedModels/Model_AE_aleatoric_LR0_001_i', 
        map_location='cpu'))
    _model_i.eval()

    get_latentspace_AE_aleatoric(_1_train_loaders[0], _model_i, "latentspace_aleatoric_train")
    get_latentspace_AE_aleatoric(_2_valid_loaders[0], _model_i, "latentspace_aleatoric_valid")


def trial_1_AE_aleatoric(alpha, alpha_string, train_loaders, valid_loaders):
    """The trial_1_AE_aleatoric function trains the given aleatoric uncertainty-aware
    model and documents its progress via saving average training and validation 
    losses to file and comparing them in a plot.

    Args:
        alpha:
          A double value indicating the chosen learning rate.
        alpha_string:
          Object of type string used as a model identifier.
        train_loaders:
          Object of PyTorch-type DataLoader to automatically pass training
          dataset to model.
        valid_loaders:
          Object of PyTorch-type DataLoader to automatically pass validation
          dataset to model.

    Returns:
        NONE:
          This function documents model progress by saving results to file and
          creating meaningful plots.
    """
    #_criterion = CoverageWidthLoss1().to(device)
    #_criterion = CoverageWidthLoss().to(device)
    #_criterion = AleatoricLoss().to(device)
    #_criterion = QuantileLoss().to(device)
    _criterion = nn.L1Loss().to(device)
    num_epochs = 1
    mc_samples = 20

    _file_prefix = "" #"/home/sabrina/Schreibtisch/Masterarbeit/TrainedModels/"
    _model_identifier_i = f'GPTest_1epochs_LR{alpha_string}_i'

    print('Initializing AE_aleatoric model.')
    _model_i = AE_aleatoric(
        device=device,
        in_channels=1,
        out_channels=2,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    #_optimizer_i = optim.Adam(_model_i.parameters(), lr=alpha)
    _optimizer_i = optim.Adam(_model_i.parameters(), lr=alpha)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(_optimizer_i, mode='min', factor=0.5, patience=10)#, verbose=True)
    
    train_losses = []
    val_losses = []

    start_training = time.time()
    print('Beginning training.')
    for epoch in range(num_epochs):
        _avg_loss = 0
        print('Hardware: ', platform.processor())

        start = time.time()
     
        for _train_loader in train_loaders:
            _loss = train_AE_aleatoric(
                loader=_train_loader,
                model_i=_model_i,
                optimizer_i=_optimizer_i,
                model_identifier_i=_model_identifier_i,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch+1
            )
            _avg_loss += _loss

        end = time.time()
        print('Duration of one ML Calculation = 1 Coupling Cycle: ', end - start)
        _avg_loss = _avg_loss/len(train_loaders)
        print('------------------------------------------------------------')
        print(f'[{_model_identifier_i}] Training Epoch: {epoch+1}')
        print(f'[{_model_identifier_i}] -> Avg aleatoric loss {_avg_loss:.3f}')
        train_losses.append(_avg_loss)

        _sum_loss = 0
        for _valid_loader in valid_loaders:
            _loss = valid_AE_aleatoric(
                loader=_train_loader,
                model_i=_model_i,
                optimizer_i=_optimizer_i,
                model_identifier_i=_model_identifier_i,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch+1
            )
            _sum_loss += _loss
        _avg_valid = _sum_loss/len(train_loaders)
        print('------------------------------------------------------------')
        print(f'[{_model_identifier_i}] Validation Epoch: {epoch+1}')
        print(f'[{_model_identifier_i}] -> Avg aleatoric loss {_avg_valid:.3f}')
        val_losses.append(_avg_valid)    

    torch.save(
        _model_i.state_dict(),
        f'{_file_prefix}Model_AE_aleatoric_{_model_identifier_i}'
    )
    end_training = time.time()
    time_training = end_training - start_training
    np.save("time_mc.npy",np.array([time_training]))
    np.save("train_losses_gp.npy",np.array(train_losses))
    np.save("valid_losses_gp.npy",np.array(val_losses))
    # After training, run inference on the validation set, collect mean and std, and plot with plot_flow_profile_std
    print("gets here")

    get_latentspace_AE_aleatoric(valid_loaders[0], _model_i, "latentspace_aleatoric_valid")
    #Plot losses
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GP Loss Curves')
    plt.savefig('gp_loss_curves.png')
    plt.close()

    #Get uncertainties/inference test
    
    #Fitting GPR
    train_latents, train_targets = extract_latent_features(_model_i, train_loaders[0])
    valid_latents, valid_targets = extract_latent_features(_model_i, valid_loaders[0])
    #Use radial basis function for the kernel for fitting
    #Use radial basis function for the kernel for fitting
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-2)
    gp.fit(train_latents, train_targets)
    #print("GP fit complete. Predicting epistemic uncertainty on validation set...")
    y_pred, y_std = gp_predict(gp, valid_latents)
    np.save("gp_epistemic.npy",np.array(y_std))

    #aleatoric from loss function
    _model_i.eval()
    aleatoric_stds = []
    with torch.no_grad():
        for data, target in valid_loaders[0]:
            channel_data = data[:, 0:1, :, :, :].float().to(device)
            channel_data = torch.add(channel_data, 1.0).float().to(device)
            pred = _model_i(channel_data).float().to(device)
            pred = torch.add(pred, -1.0).float().to(device)
            log_var = pred[:, 1, :, :, :].cpu().numpy()

            aleatoric_std = np.sqrt(np.exp(log_var))
            aleatoric_stds.append(aleatoric_std)
    aleatoric_std = np.concatenate(aleatoric_stds)
    #total_std = np.sqrt(y_std**2 + aleatoric_std**2)
    
    time_steps = np.arange(len(valid_targets))

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, valid_targets, 'o-', color='black', linewidth=2, markersize=6, label='Target', alpha=0.8)
    plt.plot(time_steps, y_pred, 's-', color='blue', linewidth=2, markersize=6, label='GP Mean Prediction', alpha=0.8)
    plt.fill_between(time_steps, y_pred - y_std, y_pred + y_std, alpha=0.3, color='orange', label='±1σ Epistemic (GP)')
    plt.fill_between(time_steps, y_pred - aleatoric_std, y_pred + aleatoric_std, alpha=0.3, color='green', label='±1σ Aleatoric (AE)')
    plt.xlabel('Time Step')
    plt.ylabel('u_x Value')
    plt.title('GP AE: Target, Prediction, Aleatoric & Epistemic Uncertainty (Cell 6,6,6)')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'gp_aleatoric_epistemic_temporal_{timestamp}.png', dpi=150)
    plt.close()
    print(f"GP uncertainty plot saved to: gp_aleatoric_epistemic_temporal_{timestamp}.png")
    print(f"epistemic (y_std): shape={np.shape(y_std)}, min={np.min(y_std)}, max={np.max(y_std)}, sample={y_std[:5]}")
    print(f"aleatoric_std: shape={np.shape(aleatoric_std)}, min={np.min(aleatoric_std)}, max={np.max(aleatoric_std)}, sample={aleatoric_std[:5]}")
    print(f"y_pred: shape={np.shape(y_pred)}, min={np.min(y_pred)}, max={np.max(y_pred)}, sample={y_pred[:5]}")
    print(f"valid_targets: shape={np.shape(valid_targets)}, min={np.min(valid_targets)}, max={np.max(valid_targets)}, sample={valid_targets[:5]}")
    reference_target = np.load('npy_results/targets_reference.npy')
    np.save('npy_results/gp_aleatoric_epistemic_lightweight_nll_results.npy', {
        'mean': y_pred,
        'std': aleatoric_std,
        'epistemic': y_std,
        'target': reference_target
    })
    print('Saved NLL-based results for GP comparison.')
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, reference_target, 'o-', color='black', linewidth=2, markersize=6, label='Target', alpha=0.8)
    plt.plot(time_steps, y_pred, 's-', color='blue', linewidth=2, markersize=6, label='GP Mean Prediction', alpha=0.8)
    plt.fill_between(time_steps, y_pred - y_std, y_pred + y_std, alpha=0.3, color='orange', label='±1σ Epistemic (GP)')
    plt.fill_between(time_steps, y_pred - aleatoric_std, y_pred + aleatoric_std, alpha=0.3, color='green', label='±1σ Aleatoric (AE)')
    plt.xlabel('Time Step')
    plt.ylabel('u_x Value')
    plt.title('GP AE: Target, Prediction, Aleatoric & Epistemic Uncertainty (Cell 6,6,6)')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'gp_aleatoric_epistemic_temporal_{timestamp}.png', dpi=150)
    plt.close()
    print(f"GP uncertainty plot saved to: gp_aleatoric_epistemic_temporal_{timestamp}.png")
    return


def trial_1_AE_aleatoric_mp():
    """The trial_1_AE_aleatoric_mp function is essentially a helper function to
    facilitate the training of multiple concurrent models via multiprocessing
    of the trial_1_AE_aleatoric function. Here, models are trained using different
    learning rates.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 1: AE_aleatoric_mp (KVS)')
    _alphas = [0.001, 0.0005, 0.0001, 0.00005,'0_000001']
    _alpha_strings = ['0_001', '0_0005', '0_0001', '0_00005','0_00001']
    _train_loaders,_valid_loaders = get_AE_loaders(path="",data_distribution="get_KVS",batch_size=32,shuffle=True)

    _processes = []
    #i=2
    #j=0
    for i in range(0,1,1):
        start_time = time.time()
        
        #torch.manual_seed(i)
        #random.seed(i)
        #trial_1_AE_aleatoric(_alphas[j], _alpha_strings[j]+f"_seed{j}_",
        #      _train_loaders, _valid_loaders,)
        
        trial_1_AE_aleatoric(_alphas[i], _alpha_strings[i],
                  _train_loaders, _valid_loaders,)
        
        end_time = time.time()
        print(f"Needed time: {end_time - start_time}")
    return

if __name__ == "__main__":
    trial_1_AE_aleatoric_mp() 
