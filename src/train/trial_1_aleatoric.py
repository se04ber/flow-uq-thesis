"""trial_1_aleatoric

This script focuses on training the convolutional autoencoder with aleatoric
uncertainty as used in the triple model approach. The loss function includes
both reconstruction loss and aleatoric uncertainty terms.

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
from model_aleatoric import AE_aleatoric
from model_MCDropout import AE_dropout
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
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, i:i+1, ...]
            losses.append(torch.max((q-1)*errors, q*errors).mean())
        return sum(losses) #losses.sum()

class AleatoricLoss(nn.Module):
    """
    This loss function implements the negative log-likelihood loss for
    heteroscedastic regression for predicting the mean and the aleatoric uncertainty.
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
        mean = pred[:, 0:1, :, :, :]  
        log_var = pred[:, 1:2, :, :, :]  
        # Ensure log_var doesn't get too negative (numerical stability)
        log_var = torch.clamp(log_var, min=-10, max=10)    
        #NLL loss for heteroscedastic noise
        loss = 0.5 * torch.exp(-log_var) * (target - mean)**2 + 0.5 * log_var
        return loss.mean()



"""class CoverageWidthLoss1(nn.Module):
    def __init__(self, alpha=0.95, lambd=0.5):
        super(CoverageWidthLoss1, self).__init__()
        self.alpha = alpha
        self.lambd = lambd
    def forward(self, pred, target):
        lower = pred[:, 0:1, :, :, :]
        upper = pred[:, 1:2, :, :, :]

        # Ensure lower <= upper
        lower, upper = torch.min(lower, upper), torch.max(lower, upper)
        # Mean Prediction Interval Width
        MPIW = (upper - lower).mean()
        # Prediction Interval Coverage Probability
        covered = ((target >= lower) & (target <= upper)).float()
        PICP = covered.mean()
        # Coverage-width loss
        coverage_penalty = torch.pow(torch.clamp((1 - self.alpha) - PICP, min=0), 2)
        loss = MPIW + self.lambd * coverage_penalty

        return loss

class CoverageWidthLoss2(nn.Module):
    def __init__(self, alpha=0.95, lambd=0.5):
        super(CoverageWidthLoss, self).__init__()
        self.alpha = alpha
        self.lambd = lambd
        self.l1 = nn.L1Loss()
    def forward(self, pred, target):
        lower = pred[:, 0:1, :, :, :]
        upper = pred[:, 1:2, :, :, :]
        # Ensure lower <= upper
        lower, upper = torch.min(lower, upper), torch.max(lower, upper)
        MPIW = (upper - lower).mean()
        covered = ((target >= lower) & (target <= upper)).float()

        min_width_penalty = 0.01 * (upper - lower).mean()
        #TEST Hybrid loss with NLL regularisation
        mean = (lower + upper) / 2
        std = (upper - lower) / 2 + 1e-6
        nll = 0.5 * torch.log(std**2) + 0.5 * ((target - mean) / std)**2
        nll_loss = nll.mean()
        #TEST
        #Grid search found weights (100,5)
        loss = 5.0 * MPIW + self.lambd * torch.pow(torch.clamp((1 - self.alpha) - PICP, min=0), 2) + 100.0 * l1_loss + 5.0 * nll_loss
        return loss
"""

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
    _criterion = QuantileLoss().to(device)
    _file_prefix = "" #"/home/sabrina/Schreibtisch/Masterarbeit/TrainedModels/"
    
    _model_identifier_i = f'cw_LR{alpha_string}_i'

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
    _optimizer_i = optim.Adam(_model_i.parameters(), lr=alpha)


    print('Beginning training.')
    for epoch in range(50):
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

    torch.save(
        _model_i.state_dict(),
        f'{_file_prefix}Model_AE_aleatoric_{_model_identifier_i}'
    )

    # After training, run inference on the validation set, collect mean and std, and plot with plot_flow_profile_std
    print("gets here")
    get_latentspace_AE_aleatoric(valid_loaders[0], _model_i, "latentspace_aleatoric_valid")
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
    _alphas = [0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_001', '0_0005', '0_0001', '0_00005']
    _train_loaders,_valid_loaders = get_AE_loaders(path="",data_distribution="get_KVS",batch_size=32,shuffle=True)

    _processes = []
    i=2
    #j=0
    for j in range(0,5,1):
        start_time = time.time()
        
        #torch.manual_seed(i)
        #random.seed(i)
        #trial_1_AE_aleatoric(_alphas[j], _alpha_strings[j]+f"_seed{j}_",
        #      _train_loaders, _valid_loaders,)
        
        #trial_1_AE_aleatoric(_alphas[i], _alpha_strings[i],
        #          _train_loaders, _valid_loaders,)
        
        end_time = time.time()
        print(f"Needed time: {end_time - start_time}")
    return

if __name__ == "__main__":
    trial_1_AE_aleatoric_mp() 
