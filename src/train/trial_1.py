""""trial_1

This script focuses on training the convolutional autoencoder as used in the
triple model approach. Afterwards, the latentspaces are retrieved from the
trained model. Plots are created.
"""

import seaborn as sns
import torch
import random
import copy
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import numpy as np
import platform
import time
import tracemalloc  #Maybe instead ressource and using usage[2]/1024

from model import AE_u_i
from model_MCDropout import AE_dropout
from model_bnn import AE_bnn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import roc_auc_score, roc_curve,brier_score_loss
from scipy.ndimage import rotate
import scipy as sc
from scipy.signal import correlate, correlation_lags
from utils import get_AE_loaders, get_RNN_loaders, dataset2csv, mlready2dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotting import plot_flow_profile, plotPredVsTargKVS
from model_aleatoric import AE_aleatoric

#Turn of for setting manually for ensemble
torch.manual_seed(10)
random.seed(10)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False


Ensemble=True #False# True #False #True
GP=False #True #True
MC=False #True #False #True #True #False #True
num_samples=10 #20
BNN=False
ensemble_models=[
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed1_20250715_230301.pth",
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed2_20250715_230301.pth",
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed3_20250715_230301.pth",
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed4_20250715_230301.pth",
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed5_20250715_230301.pth",
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed6_20250715_230301.pth",
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed7_20250715_230301.pth",
        
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed8_20250715_230301.pth",
        
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed9_20250715_230301.pth",
        "TEST_Model_AE_aleatoric_epistemic_lightweight_ux_seed10_20250715_230301.pth",
        
        #"Model_AE_aleatoric_aleatoric_LR0_001_seed0__i"
        #"Model_AE_u_i_LRKVS_full_RandomWeightInit9_i_Piet_fullTraining_justX"
        #"Model_AE_aleatoric_quantile_LR0_00005_i"
        #"Model_AE_aleatoric_McdropoutTest_50epochs_LR0_001_i"
        #"Model_AE_aleatoric_McdropoutTest_50epochs_LR0_0005_i"
        #"Model_AE_aleatoric_McdropoutTest_50epochs_LR0_0001_i"
        #"Model_AE_aleatoric_GPTest_1epochs_LR0_001_i"
        #"Model_AE_aleatoric_McdropoutTest_5epochs_LR0_001_i"
        #"Model_AE_aleatoric_1epoch_MCdropoutTest_LR0_001_i"
        #"Model_AE_aleatoric_quantile_LR0_001_i",
        #"Model_AE_aleatoric_quantile_LR0_0005_i",
        #"Model_AE_aleatoric_quantile_LR0_0001_i",
        #"Model_AE_aleatoric_quantile_LR0_00005_i"
        #"Model_AE_aleatoric_LR0_001_i",
        #"Model_AE_aleatoric_LR0_0005_i",
        #"Model_AE_aleatoric_LR0_0001_i",
        #"Model_AE_aleatoric_LR0_00005_i"
        ]
ensemble=True
Prediction=True
Target=True
epistemic=True
aleatoric=True #True
modelcomparison=True
models=[
        ensemble_models,
        "Model_AE_aleatoric_McdropoutTest_50epochs_LR0_001_i",
        ]

Testing=True

StandardDeviation=True
ID=True #True
CellShuffle=False #True #False
Rotation=False #True #False #False #False #True#False
Shift=False#True #False #False
Random=False#True #True #False #False #In Range of Amplitude of Original
TrueRandom=True #True #True #False #False #False    #Completly random Amplitude range
validPath = "Data/Validation"
_id="145922"
MultipleIDFiles=True
_ids=["145922","121428","118626","143128"]

OOD=True #False #True #True
file_path_ood="Data/OOD"
id_ood="1921"
MultipleOODFiles=True
_ids_ood= ["1921","1949","1200","1445","2166","2194"]
Couette=True #True #False
file_path_couette = "Data/Couette"
id_couette = "6_u_wall_5_0_0_top_couette_md_domain_top_0_oscil_5_0_u_wall"
MultipleCouetteFiles=True
_ids_couette=["6_u_wall_5_0_0_top_couette_md_domain_top_0_oscil_5_0_u_wall"]
"""_ids_couette=[
        "6_u_wall_5_0_0_top_couette_md_domain_top_0_oscil_5_0_u_wall",
"4_u_wall_3_0_1_middle_couette_md_domain_middle_0_oscil_3_0_u_wall",
"2_u_wall_1_5_2_bottom_couette_md_domain_bottom_0_oscil_1_5_u_wall",
"5_u_wall_4_0_1_middle_couette_md_domain_middle_0_oscil_4_0_u_wall",
"0_u_wall_0_5_0_top_couette_md_domain_top_0_oscil_0_5_u_wall",
"3_u_wall_2_0_2_bottom_couette_md_domain_bottom_0_oscil_2_0_u_wall",
"6_u_wall_5_0_0_top_couette_md_domain_top_0_oscil_5_0_u_wall"]
"""
#randomPath= "Data/Random" #Wird dynamisch erzeugt

def train_AE_aleatoric_ux(loader, model_i, optimizer_i, criterion, current_epoch):
    model_i.train()
    epoch_loss = 0
    counter = 0
    stds = []
    for batch_idx, (data, target) in enumerate(loader):
        channel_data = data[:, 0:1, :, :, :].float().to(device)
        channel_target = target[:, 0:1, :, :, :].float().to(device)
        channel_data = torch.add(channel_data, 1.0).float().to(device)
        channel_target = torch.add(channel_target, 1.0).float().to(device)
        pred = model_i(channel_data).float().to(device)
        pred = torch.add(pred, -1.0).float().to(device)
        loss = criterion(pred, channel_target)
        # Track mean std for this batch
        log_var = pred[:, 1, :, :, :].cpu().detach().numpy()
        std = np.sqrt(np.exp(log_var)).mean()
        stds.append(std)
        loss.backward()
        optimizer_i.step()
        optimizer_i.zero_grad()
        epoch_loss += loss.item()
        counter += 1
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.6f}")
    avg_loss = epoch_loss / counter
    avg_std = np.mean(stds)
    return avg_loss, avg_std


def valid_AE_aleatoric_ux(loader, model_i, criterion, current_epoch):
    model_i.eval()
    epoch_loss = 0
    counter = 0
    stds = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            channel_data = data[:, 0:1, :, :, :].float().to(device)
            channel_target = target[:, 0:1, :, :, :].float().to(device)
            channel_data = torch.add(channel_data, 1.0).float().to(device)
            channel_target = torch.add(channel_target, 1.0).float().to(device)
            pred = model_i(channel_data).float().to(device)
            pred = torch.add(pred, -1.0).float().to(device)
            loss = criterion(pred, channel_target)
            log_var = pred[:, 1, :, :, :].cpu().detach().numpy()
            std = np.sqrt(np.exp(log_var)).mean()
            stds.append(std)
            epoch_loss += loss.item()
            counter += 1
    avg_loss = epoch_loss / counter
    avg_std = np.mean(stds)
    return avg_loss, avg_std


class AleatoricLoss(nn.Module):
    """Custom loss function that includes aleatoric uncertainty estimation.

    This loss function implements the negative log-likelihood loss for
    heteroscedastic regression, which allows the model to learn both
    the mean prediction and the aleatoric uncertainty.

    The loss is: 0.5 * exp(-log_var) * (target - mean)^2 + 0.5 * log_var
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

        # Compute the aleatoric loss using
        #NLL loss for heteroscedastic noise
        loss = 0.5 * torch.exp(-log_var) * (target - mean)**2 + 0.5 * log_var

        return loss.mean()



def train_AE_u_i(loader, model_i, optimizer_i, model_identifier_i, criterion, scaler, current_epoch):
    """The train_AE function trains the single channel model and computes the
    average loss on the training set.

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
        print(_data.shape)
        _data = torch.add(_data, 1.0).float().to(device)
        #_data = torch.add(_data, 1.0).float().to(device)
        _targ = _targ_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _targ = torch.reshape(_targ, (t*c, 1, h, d, w)).float().to(device)

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


def valid_AE_u_i(loader, model_i, optimizer_i, model_identifier_i, criterion, scaler, current_epoch):
    """The valid_AE_u_i function computes the average loss on a given single
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
        _data = torch.add(_data, 1.0).float().to(device)
        _targ = _targ_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _targ = torch.reshape(_targ, (t*c, 1, h, d, w)).float().to(device)

        with torch.cuda.amp.autocast():
            _pred = model_i(_data).float().to(device=device)
            _pred = torch.add(_pred, -1.0).float().to(device=device)

            _loss = criterion(_pred, _targ)

            _epoch_loss += _loss.item()
            _counter += 1

    _avg_loss = _epoch_loss/_counter
    return _avg_loss


def get_latentspace_AE_u_i(loader, model_i, out_file_name):
    """The get_latentspace_AE function extracts the model-specific latentspace
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
        _data_x = _data[:, 0, :, :, :]
        _data_y = _data[:, 1, :, :, :]
        _data_z = _data[:, 2, :, :, :]

        with torch.cuda.amp.autocast():
            bottleneck_x = model_i(_data_x,  y='get_bottleneck')
            bottleneck_y = model_i(_data_y,  y='get_bottleneck')
            bottleneck_z = model_i(_data_z,  y='get_bottleneck')
            latentspace_x.append(bottleneck_x.cpu().detach().numpy())
            latentspace_y.append(bottleneck_y.cpu().detach().numpy())
            latentspace_z.append(bottleneck_z.cpu().detach().numpy())

    np_latentspace_x = np.vstack(latentspace_x)
    np_latentspace_y = np.vstack(latentspace_y)
    np_latentspace_z = np.vstack(latentspace_z)
    
    print(out_file_name)
    np.save(f"{out_file_name}_x",np_latentspace_x)
    np.save(f"{out_file_name}_y",np_latentspace_y)
    np.save(f"{out_file_name}_z",np_latentspace_z)

    """
    dataset2csv(
        dataset=np_latentspace_x,
        dataset_name=f'{out_file_name}_x'
    )
    dataset2csv(
        dataset=np_latentspace_y,
        dataset_name=f'{out_file_name}_y'
    )
    dataset2csv(
        dataset=np_latentspace_z,
        dataset_name=f'{out_file_name}_z'
    )
    """
    return


def get_latentspace_AE_u_i_helper():
    """The get_latentspace_AE_helper function contains the additional steps to
    create the model-specific latentspace. It loads an already trained model in
    model.eval() mode, loads the dataset loaders and calls the get_latentspace_AE
    function for each individual subdataset in the training and validation
    datasets.

    Args:
        NONE

    Returns:
        NONE:
    """
    print('Starting Trial 1: Get Latentspace (KVS)')

    model_directory = "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE'
    model_name_i = 'Model_AE_u_i_LR0_0001_i_Piet22' #'Model_AE_u_i_LR0_001_i'
    _model_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_i.load_state_dict(torch.load(
        f'{model_directory}{model_name_i}', map_location='cpu'))
    _model_i.eval()

    _loader_1, _loader_2_ = get_AE_loaders(
        #path = "/data/dust/user/sabebert/TrainingData/Training3Kvs/init_22000/",
        data_distribution='get_KVS',
        batch_size=1,
        shuffle=False
    )

    _loaders = _loader_1 + _loader_2_
    _out_directory = "Data/DatasetFull/" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready/KVS/Latentspace'
    
    """
    _out_file_names = [
        'kvs_latentspace_22000_SW',
        'kvs_latentspace_22000_NE',
        'kvs_latentspace_22000_SE',
        'kvs_latentspace_22000_NW',
    ]
    """

    
    _out_file_names = [
        'kvs_latentspace_20000_NW',
        'kvs_latentspace_20000_SE',
        'kvs_latentspace_20000_SW',
        'kvs_latentspace_22000_NE',
        'kvs_latentspace_22000_SE',
        'kvs_latentspace_22000_SW',
        'kvs_latentspace_24000_NE',
        'kvs_latentspace_24000_NW',
        'kvs_latentspace_24000_SE',
        'kvs_latentspace_24000_SW',
        'kvs_latentspace_26000_NE',
        'kvs_latentspace_26000_NW',
        'kvs_latentspace_26000_SW',
        'kvs_latentspace_28000_NE',
        'kvs_latentspace_28000_NW',
        'kvs_latentspace_28000_SE',
        'kvs_latentspace_20000_NE',
        'kvs_latentspace_22000_NW',
        'kvs_latentspace_26000_SE',
        'kvs_latentspace_28000_SW',
    ]

    for idx, _loader in enumerate(_loaders):
        print(f"idx:{idx}")
        get_latentspace_AE_u_i(
            loader=_loader,
            model_i=_model_i,
            out_file_name=f'{_out_directory}{_out_file_names[idx]}'
        )



def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

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





from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import rotate

def plot_3d_flow_comparison_2(_dataset,_dataset_2=None, t_idx=0, component=0, angle=45, axes=(0,1), threshold_percentile=80):
    """
    Visualize 3D flow patterns before and after rotation using semi-transparent blocks.

    Parameters:
    - _dataset: array with shape (t, uvw, x, y, z)
    - t_idx: time step index
    - component: velocity component index
    - angle: rotation angle in degrees
    - axes: rotation axes tuple
    - threshold_percentile: only show values above this percentile for clarity
    """
    # Extract and rotate data
    data_3d = _dataset[t_idx, component]
    if(Rotation==True):
        data_rotated = rotate(data_3d, angle=angle, axes=axes, reshape=False)
    elif(_dataset_2!=None):
        data_rotated = _dataset_2

    fig = plt.figure(figsize=(15, 6))

    for i, (data, title) in enumerate([(data_3d, 'Original'), (data_rotated, f'Rotated {angle}°')]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')

        # Create coordinate grids
        x, y, z = np.mgrid[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]]

        """
        # Only plot high-magnitude values for clarity
        threshold = np.percentile(np.abs(data), threshold_percentile)
        mask = np.abs(data) > threshold
        # Create scatter plot with transparency based on magnitude
        magnitudes = np.abs(data[mask])
        colors = data[mask]  # Use actual values for color
        """

        # Filter out upper outliers to better see the main wind speed distribution
        magnitudes_all = np.abs(data)
        lower_threshold = np.percentile(magnitudes_all, 20)  # Remove very low values
        upper_threshold = np.percentile(magnitudes_all, 95)  # Remove upper 5% outliers

        mask = (magnitudes_all > lower_threshold) & (magnitudes_all < upper_threshold)
        # Create scatter plot with filtered data for better dynamic range
        magnitudes = magnitudes_all[mask]
        colors = magnitudes  # Use magnitudes for color
        # Normalize colors to the filtered range for better contrast
        colors_normalized = (colors - colors.min()) / (colors.max() - colors.min())
        
        scatter = ax.scatter(x[mask], y[mask], z[mask],
                           c=colors_normalized,
                           s=20 + colors_normalized * 40,  # Size based on normalized magnitude
                           alpha=0.6,
                           cmap='RdBu_r')
        """
        scatter = ax.scatter(x[mask], y[mask], z[mask],
                           c=colors,
                           s=magnitudes*20/magnitudes.max(),  # Size based on magnitude
                           alpha=0.6,
                           cmap='RdBu_r')
        """
        ax.set_title(title)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax, shrink=0.8)

    plt.suptitle(f't={t_idx}, component={component}, threshold>{threshold_percentile}%')
    plt.tight_layout()
    plt.savefig("/home/sabebert/FlowComp2.png")
    plt.show()

def plot_3d_isosurfaces(_dataset, t_idx=0, component=0, angle=45, axes=(0,1), num_levels=3):
    """
    Alternative: Plot isosurfaces to show 3D structure more clearly.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure

    data_3d = _dataset[t_idx, component]
    data_rotated = rotate(data_3d, angle=angle, axes=axes, reshape=False)

    fig = plt.figure(figsize=(15, 6))
    for i, (data, title) in enumerate([(data_3d, 'Original'), (data_rotated, f'Rotated {angle}°')]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        #Isosurfaces at different levels
        data_range = data.max() - data.min()
        levels = np.linspace(data.min() + 0.3*data_range, data.max() - 0.1*data_range, num_levels)
        colors = plt.cm.RdBu_r(np.linspace(0, 1, num_levels))
        for j, (level, color) in enumerate(zip(levels, colors)):
            try:
                verts, faces, _, _ = measure.marching_cubes(data, level=level)
                mesh = Poly3DCollection(verts[faces], alpha=0.3, facecolor=color, edgecolor='none')
                ax.add_3d_collection(mesh)
            except:
                continue  # Skip if no surface found at this level
        max_range = max(data.shape)
        ax.set_xlim([0, data.shape[0]])
        ax.set_ylim([0, data.shape[1]])
        ax.set_zlim([0, data.shape[2]])
        ax.set_title(title)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.suptitle(f't={t_idx}, component={component}, {num_levels} isosurfaces')
    plt.tight_layout()
    plt.show()

def plot_3d_slices(_dataset, t_idx=0, component=0, angle=45, axes=(0,1), alpha=0.7):
    """
    Show semi-transparent orthogonal slices through the 3D volume.
    """
    # Extract and rotate data
    data_3d = _dataset[t_idx, component]
    data_rotated = rotate(data_3d, angle=angle, axes=axes, reshape=False)

    fig = plt.figure(figsize=(15, 6))

    for i, (data, title) in enumerate([(data_3d, 'Original'), (data_rotated, f'Rotated {angle}°')]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')

        # Get middle slices
        mid_x, mid_y, mid_z = [s//2 for s in data.shape]

        # Create coordinate arrays for each slice
        y_slice, z_slice = np.mgrid[0:data.shape[1], 0:data.shape[2]]
        x_slice, z_slice2 = np.mgrid[0:data.shape[0], 0:data.shape[2]]
        x_slice2, y_slice2 = np.mgrid[0:data.shape[0], 0:data.shape[1]]

        # Plot YZ slice (constant X)
        ax.plot_surface(np.full_like(y_slice, mid_x), y_slice, z_slice,
                       facecolors=plt.cm.RdBu_r((data[mid_x, :, :] - data.min()) / (data.max() - data.min())),
                       alpha=alpha, shade=False)

        # Plot XZ slice (constant Y)
        ax.plot_surface(x_slice, np.full_like(z_slice2, mid_y), z_slice2,
                       facecolors=plt.cm.RdBu_r((data[:, mid_y, :] - data.min()) / (data.max() - data.min())),
                       alpha=alpha, shade=False)

        # Plot XY slice (constant Z)
        ax.plot_surface(x_slice2, y_slice2, np.full_like(y_slice2, mid_z),
                       facecolors=plt.cm.RdBu_r((data[:, :, mid_z] - data.min()) / (data.max() - data.min())),
                       alpha=alpha, shade=False)

        ax.set_title(title)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.suptitle(f't={t_idx}, component={component}, 3D orthogonal slices')
    plt.tight_layout()
    plt.savefig("/home/sabebert/FlowSlice.png")
    plt.show()



def plot_rotation_comparison(_dataset, t_idx=0, component=0, angle=45, axes=(0,1)):
    """
    Visualize spatial data before and after rotation.

    Parameters:
    - _dataset: array with shape (t, uvw, x, y, z)
    - t_idx: time step index (default: 0)
    - component: velocity component index (default: 0)
    - angle: rotation angle in degrees (default: 45)
    - axes: rotation axes tuple (default: (0,1) for xy-plane)
    """
    # Extract 3D data and create rotation
    data_3d = _dataset[t_idx, component]
    data_rotated = rotate(data_3d, angle=angle, axes=axes, reshape=False)

    z_mid, y_mid = data_3d.shape[2]//2, data_3d.shape[1]//2
    slices = [
        (data_3d[:,:,z_mid], data_rotated[:,:,z_mid], 'XY', 'X', 'Y'),
        (data_3d[:,y_mid,:], data_rotated[:,y_mid,:], 'XZ', 'X', 'Z')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, (orig, rot, plane, xlabel, ylabel) in enumerate(slices):
        vmin, vmax = min(orig.min(), rot.min()), max(orig.max(), rot.max())
        im1 = axes[i,0].imshow(orig, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[i,0].set_title(f'Original - {plane}')
        axes[i,0].set_xlabel(xlabel); axes[i,0].set_ylabel(ylabel)
        im2 = axes[i,1].imshow(rot, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[i,1].set_title(f'Rotated {angle}° - {plane}')
        axes[i,1].set_xlabel(xlabel); axes[i,1].set_ylabel(ylabel)
        plt.colorbar(im1, ax=axes[i,0], shrink=0.7)
        plt.colorbar(im2, ax=axes[i,1], shrink=0.7)
    plt.suptitle(f't={t_idx}, component={component}')
    plt.tight_layout()
    plt.savefig("/home/sabebert/RotationCompare.png")
    plt.show()

# Example usage:
# plot_rotation_comparison(_dataset)
# plot_rotation_comparison(_dataset, t_idx=5, component=1, angle=90, axes=(1,2))


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


def trial_1_AE_u_i(alpha, alpha_string, train_loaders, valid_loaders):
    """The trial_1_AE function trains the given model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

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
    _criterion = nn.L1Loss().to(device)
    #_criterion = AleatoricLoss().to(device)
    _file_prefix = ""#"/data/dust/sabebert/TrainingData/Training3Kvs/Numpy/old/"#'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/' + \
        #'4_ICCS/Results/1_Conv_AE/'

    _model_identifier_i = f'LR{alpha_string}_i_Piet_fullTraining_justX'

    print('Initializing AE_u_i model.')
    

    _model_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer_i = optim.Adam(_model_i.parameters(), lr=alpha)

    print('Beginning training.')
    for epoch in range(100):
        _avg_loss = 0
        print('Hardware: ', platform.processor())

        start = time.time()

        for _train_loader in train_loaders:
            _loss = train_AE_u_i(
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
        print(f'[{_model_identifier_i}] -> Avg u_i {_avg_loss:.3f}')

        _sum_loss = 0

        for _valid_loader in valid_loaders:
            _loss = valid_AE_u_i(
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
        print(f'[{_model_identifier_i}] -> Avg u_i {_avg_valid:.3f}')

    torch.save(
        _model_i.state_dict(),
        f'{_file_prefix}Model_AE_u_i_{_model_identifier_i}'
    )
    return


def trial_1_AE_u_i_mp():
    """The trial_1_AE_mp function is essentially a helper function to
    facilitate the training of multiple concurrent models via multiprocessing
    of the trial_1_AE/trial_1_AE_u_i function. Here, 6 unique models are trained
    using the 6 learning rates (_alphas) respectively. Refer to the trial_1_AE
    function for more details.
t
    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 1: AE_u_i_mp (KVS)')
    _alphas = [0.005,0.001, 0.0005, 0.0001, 0.00001,0.00005]
    _alpha_strings = ['0_001', '0_0005', '0_0001','0.00001', '0_00005']
    #_train_loaders, _valid_loaders = get_AE_loaders(
    #    path = "", #"/data/dust/user/sabebert/Mamico_ml/", #/data/dust/user/sabebert/TrainingData/Training3Kvs/init_22000/",
    #    data_distribution= 'get_KVS', #'get_couette', #'get_KVS',
    #    batch_size=32,
    #    shuffle=True
    #)
    #_processes = []
    
    #Iterate for different randomized weight starts
    #SEEDS
    
    i=2
    j=0
    splits=["First","Half","None"]

    for k in range(0,3,1):
        split = splits[k]
        torch.manual_seed(j)
        random.seed(j)
        _train_loaders, _valid_loaders = get_AE_loaders(
        path = "",#"/data/dust/user/sabebert/Mamico_ml/", #/data/dust/user/sabebert/TrainingData/Training3Kvs/init_22000/",
        data_distribution= 'get_KVS', #'get_couette', #'get_KVS',
        batch_size=32,
        shuffle=True,
        split=split
         )
        _alpha_strings=f'KVS_full_RandomWeightInit_{j}_split{split}'
        trial_1_AE_u_i(_alphas[i], _alpha_strings,
                  _train_loaders, _valid_loaders,)
    """

    for j in range(0,7,1):
        
        _train_loaders, _valid_loaders = get_AE_loaders(
        path = "",#"/data/dust/user/sabebert/Mamico_ml/", #/data/dust/user/sabebert/TrainingData/Training3Kvs/init_22000/",
        data_distribution= 'get_KVS', #'get_couette', #'get_KVS',
        batch_size=32,
        shuffle=True
         )
        alpha_strings=f'LearningRate{_alpha_strings[j]}'
        trial_1_AE_u_i(_alphas[j], alpha_strings,
                  _train_loaders, _valid_loaders,)

    """
    #Iterate for different training learning rates
    #for i in range(len(_alphas)):
    #    trial_1_AE_u_i(_alphas[i], _alpha_strings[i],
    #              _train_loaders, _valid_loaders,)
    
    #i=2
    #trial_1_AE_u_i(_alphas[i], _alpha_strings[i],
    #_train_loaders, _valid_loaders,)
    
    #for i in range(1):
    #    _p = mp.Process(
    #        target=trial_1_AE_u_i,
    #        args=(_alphas[i], _alpha_strings[i],
    # _train_loaders, _valid_loaders,)
    #    )
    #    _p.start()
    #    _processes.append(_p)
    #    print(f'Creating Process Number: {i+1}')

    #for _process in _processes:
    #    _process.join()
    #    print('Joining Process')
    return


def prediction_retriever_u_i(model_directory, model_name_i, dataset_name, save2file_name_1, save2file_name_2):
    """The prediction_retriever function is used to evaluate model performance
    of a trained model. This is done by loading the saved model, feeding it
    with datasets and then saving the corresponding predictions for later
    visual comparison.

    Args:
        model_directory:

        model_name:

        dataset_name:

    Returns:
        NONE
    """
    _1_train_loaders, _2_valid_loaders = get_AE_loaders(
            data_distribution=dataset_name,
            batch_size=1,
            shuffle=False
        )

    _model_x = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_y = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_z = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_x.load_state_dict(torch.load(
        f'{model_directory}/{model_name_i}', map_location='cpu'))
    _model_x.eval()
    _model_y.load_state_dict(torch.load(
        f'{model_directory}/{model_name_i}', map_location='cpu'))
    _model_y.eval()
    _model_z.load_state_dict(torch.load(
        f'{model_directory}/{model_name_i}', map_location='cpu'))
    _model_z.eval()

    for train_loader in _1_train_loaders:
        _preds = []
        _targs = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(device=device)
            data = torch.add(data, 1.0).float().to(device=device)
            with torch.cuda.amp.autocast():
                data_pred_x = _model_x(data[:, 0, :, :, :])
                data_pred_y = _model_y(data[:, 0, :, :, :]) #1
                data_pred_z = _model_z(data[:, 0, :, :, :]) #2
                data_pred = torch.cat(
                    (data_pred_x, data_pred_y, data_pred_z), 1).to(device)
                data_pred = torch.add(
                    data_pred, -1.0).float().to(device=device)
                _preds.append(data_pred.cpu().detach().numpy())
                _targs.append(target.cpu().detach().numpy())
        _preds = np.vstack(_preds)
        _targs = np.vstack(_targs)
    _lbm_1 = np.loadtxt(
        f'dataset_mlready/01_clean_lbm/kvs_{save2file_name_1}_lbm.csv', delimiter=";")
    _lbm_1 = _lbm_1.reshape(1000, 3)
    plotPredVsTargKVS(input_pred=_preds, input_targ=_targs,
                      input_lbm=_lbm_1, file_name=save2file_name_1)

    for valid_loader in _2_valid_loaders:
        _preds = []
        _targs = []
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.float().to(device=device)
            data = torch.add(data, 1.0).float().to(device=device)
            with torch.cuda.amp.autocast():
                data_pred_x = _model_x(data[:, 0, :, :, :])
                data_pred_y = _model_y(data[:, 0, :, :, :]) #1
                data_pred_z = _model_z(data[:, 0, :, :, :])#2
                data_pred = torch.cat(
                    (data_pred_x, data_pred_y, data_pred_z), 1).to(device)
                data_pred = torch.add(
                    data_pred, -1.0).float().to(device=device)
                _preds.append(data_pred.cpu().detach().numpy())
                _targs.append(target.cpu().detach().numpy())
        _preds = np.vstack(_preds)
        _targs = np.vstack(_targs)
    _lbm_2 = np.loadtxt(
        f'dataset_mlready/01_clean_lbm/kvs_{save2file_name_2}_lbm.csv', delimiter=";")
    _lbm_2 = _lbm_2.reshape(1000, 3)
    plotPredVsTargKVS(input_pred=_preds, input_targ=_targs,
                      input_lbm=_lbm_2, file_name=save2file_name_2)


def prediction_retriever_latentspace_u_i(model_directory, model_name_i, dataset_name, save2file_name):
    """The prediction_retriever function_latentspace_u_i is used to evaluate
    correctness of extracted latentspaces. This is done by loading the trained
    AE_u_i model, feeding it with the extracted latentspaces, saving the corresponding predictions for later
    visual comparison.

    Args:
        model_directory:

        model_name_i:

        dataset_name:

        save2file_prefix:

        save2file_name:

    Returns:
        NONE
    """
    _t_x, _t_y, _t_z, _v_x, _v_y, _v_z = get_RNN_loaders(
            data_distribution=dataset_name,
            batch_size=1,
            shuffle=False
        )

    _trains, _valids = get_AE_loaders(
            data_distribution=dataset_name,
            batch_size=32,
            shuffle=False
        )

    _model_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_i.load_state_dict(torch.load(
        f'{model_directory}/{model_name_i}', map_location='cpu'))
    _model_i.eval()

    data_preds_x = torch.zeros(1, 1, 24, 24, 24).to(device=device)
    data_preds_y = torch.zeros(1, 1, 24, 24, 24).to(device=device)
    data_preds_z = torch.zeros(1, 1, 24, 24, 24).to(device=device)
    targs = []

    for data, target in _t_x[0]:
        data = data.float().to(device=device)
        with torch.cuda.amp.autocast():
            data_pred_x = _model_i(data, 'get_MD_output').to(device=device)
            data_preds_x = torch.cat((data_preds_x, data_pred_x), 0).to(device)

    for data, target in _t_y[0]:
        data = data.float().to(device=device)
        with torch.cuda.amp.autocast():
            data_pred_y = _model_i(data, 'get_MD_output').to(device=device)
            data_preds_y = torch.cat((data_preds_y, data_pred_y), 0).to(device)

    for data, target in _t_z[0]:
        data = data.float().to(device=device)
        with torch.cuda.amp.autocast():
            data_pred_z = _model_i(data, 'get_MD_output').to(device=device)
            data_preds_z = torch.cat((data_preds_z, data_pred_z), 0).to(device)

    for data, target in _trains[0]:
        with torch.cuda.amp.autocast():
            target = target.cpu().detach().numpy()
            print(target.shape)
            targs.append(target)

    targs = np.vstack(targs)

    print('data_preds_x.shape: ', data_preds_x.shape)
    print('data_preds_y.shape: ', data_preds_y.shape)
    print('data_preds_z.shape: ', data_preds_z.shape)
    preds = torch.cat((data_preds_x, data_preds_y, data_preds_z), 1).to(device)
    preds = torch.add(preds, -1.0).float().to(device=device)
    preds = preds[1:, :, :, :, :].cpu().detach().numpy()

    plotPredVsTargKVS(input_pred=preds, input_targ=targs,
                      file_name=save2file_name)



#Ensemble +aleatoric figure maker
def fig_maker_2(_id,_ids=None,t_max=899,file_path="Data/Validation",file_path_ood="None",id_ood="None",_ids_ood="None",file_path_couette="None",id_couette="None",_ids_couette="None",augments=[],outname="test",epistemic_models=ensemble_models,aleatoric=True,getEuclidianDistance=False,onlyX=True):
    _directory = "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready/KVS/Validation/'
    #_id = id  # '28000_SW'
    #_file_name = f"processed_file_{_id}.npy" #f'clean_kvs_combined_domain_init_{_id}.csv'
    #_file_name = f"../../build/0_couette/processed_file_6_u_wall_5_0_0_top_couette_md_domain_top_0_oscil_5_0_u_wall.npy"
    #_file_name = f"Data/Validation/processed_file_{_id}.npy"
    if(MultipleIDFiles==True):
        _file_names = [f"{file_path}/processed_file_{_id}.npy" for _id in _ids]
    else:
        _file_names = [f"{file_path}/processed_file_{_id}.npy"]
    if(MultipleOODFiles==True):
        _file_names_ood = [f"{file_path_ood}/processed_file_{_id_ood}.npy" for _id_ood in _ids_ood]
    else:
        _file_names_ood = ["{file_path_ood}/processed_file_{id_ood}.npy"]
    if(MultipleCouetteFiles==True):
        _file_names_couette = [f"{file_path_couette}/processed_file_{_id_couette}.npy" for _id_couette in _ids_couette]
    else:
        _file_names_couette = [f"{file_path_couette}/processed_file_{_id_couette}.npy" for _id_couette in _ids_couette]
    
    measured_time_ensemble = [] #Array of arrays of computional cost/time for all files of one ensemble member
    measured_time_ensemble_std = [] # std between all files for ensembles
    measured_memory_ensemble = []
    measured_memory_ensemble_std = []

    mean_preds=[]
    if(Rotation==True):
        mean_preds_rot=[]
    if(CellShuffle==True):
        mean_preds_shuffle=[]
    if(Shift==True):
        mean_preds_shift=[]
    if(Random==True):
         mean_preds_random=[]
    if(TrueRandom==True):
        mean_preds_randomtrue=[]
    if(OOD==True):
        mean_preds_ood=[]
    if(Couette==True):
        mean_preds_couette=[]

    uncertainties=[]
    uncertainties_epistemic=[]
    flag=0
    _model_directory = "" #'Results/1_Conv_AE'
    for model_name in epistemic_models:
        _model_name_i = model_name   #"Model_AE_u_i_LR0_0001_i_Piet22" #"Model_AE_u_i_LR0_0001_i_Piet_allPerm0" # #"Model_RNN_LR1e-4_Lay1_Seq25_i #'Model_AE_u_i_LR0_0001_i' # 'Model_AE_u_i_LR0_001_i'
        #_dataset = torch.from_numpy(mlready2dataset(
        #    f'{_directory}{_file_name}')[:, :, :,:,:]) #1:-1, 1:-1, 1:-1])
        
        _datasets = []
        for _file_name in _file_names:
            _dataset = mlready2dataset(f'{_directory}{_file_name}')
            _dataset = _dataset[:, :, :, :, :]
            print('Dataset shape: ', _dataset.shape)
            if(_dataset.shape == (900,1,24,24,24)):
                print(_dataset.shape)
                _dataset = np.concatenate([_dataset, _dataset, _dataset], axis=1)
            print(_dataset.shape)
            _datasets.append(_dataset)
        _dataset = np.concatenate(np.array(_datasets), axis=0)

        #AUGMENTATION in case of shuffle, rotation OOD test
        if(augments != []):
            if("OOD" in augments):
                _datasets_ood=[]
                for _file_name in _file_names_ood:
                    _file_name = f"{file_path_ood}/processed_file_{id_ood}.npy"
                    _dataset_ood = mlready2dataset(f'{_directory}{_file_name}')
                    _datase_ood = _dataset_ood[:, :, :, :, :]
                    if(_dataset_ood.shape == (900,1,24,24,24)):
                        print(_dataset_ood.shape)
                        _dataset_ood = np.concatenate([_dataset_ood, _dataset_ood, _dataset_ood], axis=1)
                    _dataset_ood = torch.from_numpy(_dataset_ood.copy()).to(device)
                    _dataset_ood = torch.add(_dataset_ood, 1.0).float().to(device)
                    _datasets_ood.append(_dataset_ood)
                _dataset_ood = torch.cat(_datasets_ood, axis=0)

            if("Couette" in augments):
                _datasets_couette=[]
                for _file_name in _file_names_couette:
                    _file_name = f"{file_path_couette}/processed_file_{id_couette}.npy"
                    _dataset_couette = mlready2dataset(f'{_directory}{_file_name}')
                    _dataset_couette = _dataset_couette[:, :, :, :, :]
                    if(_dataset_ood.shape != (900,1,24,24,24)):
                        print(_dataset_couette.shape)
                        _dataset_couette = np.concatenate([_dataset_couette, _dataset_couette, _dataset_couette], axis=1)
                    _dataset_couette = torch.from_numpy(_dataset_couette.copy()).to(device)
                    _dataset_couette = torch.add(_dataset_couette, 1.0).float().to(device)
                    _datasets_couette.append(_dataset_couette)
                _dataset_couette = torch.cat(_datasets_couette, axis=0)

                #augmented = np.copy(dataset)
            if("Rotation" in augments): # (90°) Rotation nur für x,y,z (letzte 3 Dims)
                _datasets_rot=[]
                #for _file_name in _file_names:
                Angle=90
                print("Gets here: Rotation")
                # Rotiere um alle 3 Achsen: x-y, x-z, y-z
                _dataset_rot = np.rot90(_dataset, k=1, axes=(-1, -2)) #x,y #torch.rot90 dims=(-2,-1)
                    
                print(_dataset_rot.shape)
                plot_rotation_comparison(_dataset_rot, t_idx=500, component=1, angle=90, axes=(-1,-2))
                plot_3d_flow_comparison_2(_dataset_rot, angle=90,threshold_percentile=20)
                #plot_3d_slices(_dataset)  # Often the clearest option
                #if(_dataset_rot.shape != (len(_dataset_rot),1,24,24,24)):
                #    print(_dataset_rot.shape)
                #    _dataset_rot = np.concatenate([_dataset_rot, _dataset_rot, _dataset_rot], axis=1)
                _dataset_rot = torch.from_numpy(_dataset_rot.copy()).to(device)
                _dataset_rot = torch.add(_dataset_rot, 1.0).float().to(device)
                #_datasets_rot.append(_dataset_rot)    
                #_dataset_rot = torch.cat(_datasets_rot, axis=0)
                print(_dataset_rot.shape)
                """
                                temp = _dataset[i,j,t,uvw]
                                temp = rotate(temp, 90, axes=(0,1), reshape=False)  # x-y Rotation
                                temp = rotate(temp, 90, axes=(0,2), reshape=False)  # x-z Rotation
                                augmented[i,j,t,uvw] = rotate(temp, 90, axes=(1,2), reshape=False) 
                                #augmented[i,j,t,uvw] = rotate(_dataset[i,j,t,uvw], Angle, axes=(0,1), reshape=False)
                """

            if("Shift" in augments): # (90°) Rotation nur für x,y,z (letzte 3 Dims)
                relativeAmplitudeShift=1.7
                _datasets_shift=[]
                #for _file_name in _file_names:
                print(f"Gets here: Shift with {relativeAmplitudeShift}")
                #Shift by relative Amplitude
                _dataset_shift = _dataset * relativeAmplitudeShift

                #if(_dataset_shift.shape != (len(_dataset_shift),1,24,24,24)):
                #    print(_dataset_shift.shape)
                #    _dataset_shift = np.concatenate([_dataset_shift, _dataset_shift, _dataset_shift], axis=1)
                _dataset_shift = torch.from_numpy(_dataset_shift.copy()).to(device)
                _dataset_shift = torch.add(_dataset_shift, 1.0).float().to(device)
                #_datasets_shift.append(_dataset_shift)
                #_dataset_shift = torch.cat(_datasets_shift, axis=0)
                #print(_dataset_shift.shape)

            if("CellShuffle" in augments):
                _datasets_shuffle=[]
                #for _file_name in _file_names:
                augmented_2 = np.zeros(_dataset.shape)
                for t in range(_dataset.shape[0]):
                    print(f"At beginning shape:{_dataset.shape}")
                    _dataset = _dataset
                    flat = _dataset[t,0].flatten() # Flatten the 3D cube to 1D
                    np.random.shuffle(flat)  # Shuffle the 1D array
                    augmented_2[t,0] = flat.reshape((24, 24, 24))  #Reshape back to 3D
                    #print(f"After shuffle: {_dataset.shape}")
                #if(augmented_2.shape == (len(_dataset),1,24,24,24)):
                #    _dataset_shuffle = np.concatenate([augmented_2,augmented_2,augmented_2],axis=1)
                        
                _dataset_shuffle = torch.from_numpy(augmented_2.copy()).to(device)
                _dataset_shuffle = torch.add(_dataset_shuffle, 1.0).float().to(device)
                #_datasets_shuffle.append(_dataset_shuffle)
                #_dataset_shuffle = torch.cat(_datasets_shuffle, axis=0)
                print(f"Shape CellShuffle at  end: {_dataset_shuffle.shape}")

            if("Random" in augments): # Random noise zu x,y,z
                _datasets_random = []
                #for _file_name in _file_names:
                magnitude_range = (0.5, 2.0)  # Min, Max für Randomrange
                mag = np.random.uniform(*magnitude_range)
                augmented_3 = np.zeros(_dataset.shape)
                for t in range(_dataset.shape[0]):
                    data = _dataset[t,0]
                    noise = np.random.normal(0, np.std(data)*mag, data.shape)
                    augmented_3[t,0] = data + noise

                _dataset_random = torch.from_numpy(augmented_3.copy()).to(device)
                _dataset_random = torch.add(_dataset_random, 1.0).float().to(device)
                #_datasets_random.append(_dataset_random)
                #_dataset_random = torch.cat(_datasets_random, axis=0)

            if("TrueRandom" in augments): # (90°) Rotation nur für x,y,z (letzte 3 Dims)
                _datasets_randomtrue=[]
                magnitude_range = (2.0, 6.0)  # Min, Max für Randomrange
                augmented_4 = np.zeros(_dataset.shape)
                for t in range(_dataset.shape[0]):
                    augmented_4[t,0] = np.random.randint(1, 11, size=(24, 24, 24))
                _dataset_randomtrue = torch.from_numpy(augmented_4.copy()).to(device)
                _dataset_randomtrue = torch.add(_dataset_randomtrue, 1.0).float().to(device)
                #_datasets_randomtrue.append(_dataset_randomtrue)
                #_dataset_randomtrue = torch.cat(_datasets_randomtrue, axis=0)
                print(_dataset_randomtrue.shape)
        
        print("gets here 5")
        #dataset = np.concatenate([dataset, augmented], axis=0)
        #print(f"After {augment} augmentation: {_dataset.shape}")
        #dataset = dataset[np.random.permutation(dataset.shape[0])] #Final shuffle


        _targs = copy.deepcopy(_dataset[1:, :, :, :, :])
        _targs2 = copy.deepcopy(_dataset[:, :, :, :, :])
        _input_a = torch.from_numpy(copy.deepcopy(_dataset[:-1, :, :, :, :]))
        _input_b = torch.from_numpy(copy.deepcopy(_dataset[:-1, :, :, :, :]))

        _dataset = torch.from_numpy(_dataset).to(device)
        #_targs = copy.deepcopy(_dataset)
        _dataset = torch.add(_dataset, 1.0).float().to(device)

        """
        _model_u_i = AE_u_i(
            device=device,
            in_channels=1,
            out_channels=1,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
        ).to(device)
        """
        
        if(MC==True):
             _model_u_i =  AE_dropout(
            device=device,
            in_channels=1,
            out_channels=2,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
        ).to(device)
        elif(GP==True):
            _model_u_i =  AE_aleatoric(
            device=device,
            in_channels=1,
            out_channels=2,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
        ).to(device)
        elif(BNN==True):
             _model_u_i =  AE_BNN(
            device=device,
            in_channels=1,
            out_channels=2,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
        ).to(device)
        else:
            """
            _model_u_i = AE_u_i(
            device=device,
            in_channels=1,
            out_channels=1,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
        ).to(device)
            """
            
            _model_u_i =  AE_aleatoric(
                device=device,
                in_channels=1,
                out_channels=2,
                features=[4, 8, 16],
                activation=nn.ReLU(inplace=True)
            ).to(device)
            
        _model_u_i.load_state_dict(torch.load(
            f'{_model_directory}{_model_name_i}', map_location='cpu'))
        _model_u_i.eval()
        
        if(MC==False and BNN==False and GP==False):
            if(Rotation == True):
                #INCLUDING ROTATION FOR COMPARISON
                print(f"Rot:{_dataset_rot.shape}")
                _preds_x = _model_u_i(_dataset_rot[:, 0:1, :, :, :])
                # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
                # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])
                _pred = torch.add(_preds_x, -1.0).float().to(device)
                print(f"Rot finished")
                mean_pred_rot = _pred[:, 0, :, :, :].cpu().detach().numpy()
                if(aleatoric==True):
                    log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
                    uncertainty_aleatoric_rot = np.sqrt(np.exp(log_var))
                mean_preds_rot.append(mean_pred_rot)
            if(Shift==True):
                print(f"Shift:{_dataset_shift.shape}")
                #INCLUDING RANDOM FOR COMPARISON
                _preds_x = _model_u_i(_dataset_shift[:, 0:1, :, :, :])
                # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
                # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])
                _pred = torch.add(_preds_x, -1.0).float().to(device)
                mean_pred_shift = _pred[:, 0, :, :, :].cpu().detach().numpy()
                print("Shift finished")
                if(aleatoric==True):
                    log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
                    uncertainty_aleatoric_rot = np.sqrt(np.exp(log_var))
                mean_preds_shift.append(mean_pred_shift)
            if(CellShuffle==True):
                #INCLUDING SHUFFLE FOR COMPARISON
                print(f"Shuffle{_dataset_shuffle.shape}")
                _preds_x = _model_u_i(_dataset_shuffle[:, 0:1, :, :, :])
                # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
                # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])
                _pred = torch.add(_preds_x, -1.0).float().to(device)
                mean_pred_shuffle = _pred[:, 0, :, :, :].cpu().detach().numpy()
                print("Shuffle finished")
                if(aleatoric==True):
                    log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
                    uncertainty_aleatoric_rot = np.sqrt(np.exp(log_var))
                mean_preds_shuffle.append(mean_pred_shuffle)
            if(Random==True):
                print(f"Random:{_dataset_random.shape}")
                #INCLUDING RANDOM FOR COMPARISON
                _preds_x = _model_u_i(_dataset_random[:, 0:1, :, :, :])
                # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
                # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])
                _pred = torch.add(_preds_x, -1.0).float().to(device)
                mean_pred_random = _pred[:, 0, :, :, :].cpu().detach().numpy()
                print("Random finished")
                if(aleatoric==True):
                    log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
                    uncertainty_aleatoric_rot = np.sqrt(np.exp(log_var))
                mean_preds_random.append(mean_pred_random)
            if(TrueRandom==True):
                print(f"True Random:{_dataset_randomtrue.shape}")
                #INCLUDING RANDOM FOR COMPARISON
                _preds_x = _model_u_i(_dataset_randomtrue[:, 0:1, :, :, :])
                # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
                # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])
                _pred = torch.add(_preds_x, -1.0).float().to(device)
                mean_pred_randomtrue = _pred[:, 0, :, :, :].cpu().detach().numpy()
                print("True Random finished")
                if(aleatoric==True):
                    log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
                    uncertainty_aleatoric_rot = np.sqrt(np.exp(log_var))
                mean_preds_randomtrue.append(mean_pred_randomtrue)
            if(OOD==True):
                print(f"OOD start:{_dataset_ood.shape}")
                #INCLUDING RANDOM FOR COMPARISON
                _preds_x = _model_u_i(_dataset_ood[:, 0:1, :, :, :])
                # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
                # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])
                _pred = torch.add(_preds_x, -1.0).float().to(device)
                mean_pred_ood = _pred[:, 0, :, :, :].cpu().detach().numpy()
                print("OOD finished")
                if(aleatoric==True):
                    log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
                    uncertainty_aleatoric_rot = np.sqrt(np.exp(log_var))
                mean_preds_ood.append(mean_pred_ood)

            if(Couette==True):
                #INCLUDING RANDOM FOR COMPARISON
                _preds_x = _model_u_i(_dataset_couette[:, 0:1, :, :, :])
                # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
                # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])
                _pred = torch.add(_preds_x, -1.0).float().to(device)
                mean_pred_couette = _pred[:, 0, :, :, :].cpu().detach().numpy()
                if(aleatoric==True):
                    log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
                    uncertainty_aleatoric_rot = np.sqrt(np.exp(log_var))
                mean_preds_couette.append(mean_pred_couette)
                    
            #JUST NORMAL CASE
            index=0
            measured_time_files = []
            measured_memory_files = []
            mean_preds_i=[]

            for file in _file_names:
                start_time = time.time()
                tracemalloc.start()
                _preds_x = _model_u_i(_dataset[index:index+t_max, 0:1, :, :, :])
                index = index + t_max
                # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
                # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])
                _pred = torch.add(_preds_x, -1.0).float().to(device)
                mean_pred = _pred[:, 0, :, :, :].cpu().detach().numpy()
                if(aleatoric==True):
                    log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
                    uncertainty = np.sqrt(np.exp(log_var))
                    uncertainties.append(uncertainty)    
                mean_preds_i.append(mean_pred)
                
                measured_time_file = (time.time() - start_time)
                measured_time_files.append(measured_time_file)
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                measured_memory_file = peak / 1024 / 1024  # Convert to MB
                measured_time_files.append(measured_time_file)
                measured_memory_files.append(measured_memory_file)
                print(f"time needed total currently for computation, ensemble member, one file: {measured_time_file}")
            print(f"Total times ensemble member, all files: {measured_time_files}")
            measured_time_ensemble.append(measured_time_files)
            measured_memory_ensemble.append(measured_memory_files)
            mean_preds_i = np.array(mean_preds_i)
            print(f"Shape before mean_preds_i:{mean_preds_i.shape}")
            mean_preds_i = mean_preds_i.reshape((len(_file_names)*mean_preds_i.shape[1],) + mean_preds_i.shape[2:])
            print(f"Shape after mean_preds_i:{mean_preds_i.shape}")
            mean_preds.append(mean_preds_i)
            print(f"Shape after mean_preds:{np.array(mean_preds).shape}")

        if(GP==True):
            #Fitting GPR
            train_loaders, valid_loaders = get_AE_loaders(
               path = "", #"/data/dust/user/sabebert/Mamico_ml/", #/data/dust/user/sabebert/TrainingData/Training3Kvs/init_22000/",
                data_distribution= 'get_KVS', #'get_couette', #'get_KVS',
                batch_size=32,
                shuffle=True
            )
            newFitting=False
            if(newFitting==True):
                train_latents, train_targets = extract_latent_features(_model_u_i, train_loaders[0])
                valid_latents, valid_targets = extract_latent_features(_model_u_i, valid_loaders[0])
                #Use radial basis function for the kernel for fitting
                kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-2)
                gp.fit(train_latents, train_targets)
                #print("GP fit complete. Predicting epistemic uncertainty on validation set...")
                y_pred, y_std = gp_predict(gp, valid_latents)
                mean_preds.append(y_pred)
                mean_pred = y_pred
                np.save("gp_prediction_2.npy",np.array(y_pred))
                np.save("gp_epistemic_2.npy",np.array(y_std))
            #aleatoric from loss function
            _model_u_i.eval()
            #JUST NORMAL CASE
            _preds_x = _model_u_i(_dataset[:, 0:1, :, :, :])
            # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
            # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])
            _pred = torch.add(_preds_x, -1.0).float().to(device)
            mean_pred = _pred[:, 0, :, :, :].cpu().detach().numpy()
            if(aleatoric==True):
                log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
                uncertainty = np.sqrt(np.exp(log_var))
                uncertainties.append(uncertainty)
            #mean_pred_i = []
            mean_pred = np.array([mean_pred])
            """
            _model_u_i.eval()
            aleatoric_stds = []
            with torch.no_grad():
                for data, target in valid_loaders[0]:
                    channel_data = data[:, 0:1, :, :, :].float().to(device)
                channel_data = torch.add(channel_data, 1.0).float().to(device)
                    pred = _model_u_i(channel_data).float().to(device)
                    pred = torch.add(pred, -1.0).float().to(device)
                    log_var = pred[:, 1, :, :, :].cpu().numpy()
                    mean_preds.append(pred)

                    aleatoric_std = np.sqrt(np.exp(log_var))
                    aleatoric_stds.append(aleatoric_std)
                aleatoric_std = np.concatenate(aleatoric_stds)
                mean_preds = np.concatenate(mean_preds)
                #total_std = np.sqrt(y_std**2 + aleatoric_std**2)
            """
            y_std = np.load("gp_epistemic_2.npy")[0:len(_dataset)]
            #for i,y_std_i in enumerate(y_std):
            #    y_std_2 = [[np.ones_like(mean_pred[0,0]] * y_std_i
            y_std = [[np.ones_like(mean_pred[0,0])] + y_std_i for y_std_i in y_std]
            print(f"shape gp y_std right? {y_std}")
            uncertainties_epistemic.append([np.array(y_std)])
            uncertainties_epistemic = np.array(uncertainties_epistemic)[:,:,0]
            print(f"Y_std/ espitemic shape gp: {np.array(y_std).shape}")
            print(f"prediction has ok shape gp: {np.array(mean_preds).shape}")
            

        if(MC==True):
            mc_preds_MC = mc_dropout_predict(_model_u_i, _dataset[:,0:1,:,:,:], mc_samples=num_samples)
            print(f"full mc {mc_preds_MC.shape}")
            mc_preds_MC = torch.add(torch.from_numpy(mc_preds_MC),-1.0).float().to(device)
            mc_preds_MC = mc_preds_MC.detach().numpy()
            _preds_x = _model_u_i(_dataset[:,0:1,:,:,:])  #torch.from_numpy(np.mean(mc_preds_MC[:,:,0:1,:,:,:],axis=0))
            mean_pred_i = mc_preds_MC[:, :, 0, :, :, :]
            mean_pred = mean_pred_i #np.mean(mean_pred_i, axis=0)
            mean_preds = mean_pred
            log_var = mc_preds_MC[:, :, 1, :, :, :]
            print(f"log before: {log_var.shape}")
            aleatoric_std = np.mean(np.sqrt(np.exp(log_var)), axis=0)
            epistemic_std = np.std(mc_preds_MC[:,:,0], axis=0)
            #mean_preds.append(mean_pred)
            uncertainties_epistemic.append(epistemic_std)
            uncertainties.append(aleatoric_std)
            print(f"epistemic {epistemic_std.shape}")
            print(f"{uncertainties_epistemic[0]}")
            print(f"aleatoric {aleatoric_std.shape}")
            
            _dataset_original = _dataset
            mean_preds_original = mean_preds
            for case in augments:
                if(case == "Rotation"):
                    _dataset = _dataset_rot
                if(case == "Shift"):
                    _dataset = _dataset_shift
                if(case == "CellShuffle"):
                    _dataset = _dataset_shuffle
                if(case == "Random"):
                    _dataset = _dataset_random
                if(case == "TrueRandom"):
                    _dataset = _dataset_randomtrue
                if(case == "OOD"):
                    _dataset = _dataset_ood
                if(case == "Couette"):
                    _dataset = _dataset_couette
        
                mc_preds_MC = mc_dropout_predict(_model_u_i, _dataset[:,0:1,:,:,:], mc_samples=num_samples)
                print(f"full mc {mc_preds_MC.shape}")
                mc_preds_MC = torch.add(torch.from_numpy(mc_preds_MC),-1.0).float().to(device)
                mc_preds_MC = mc_preds_MC.detach().numpy()
                _preds_x = _model_u_i(_dataset[:,0:1,:,:,:])  #torch.from_numpy(np.mean(mc_preds_MC[:,:,0:1,:,:,:],axis=0))
                mean_pred_i = mc_preds_MC[:, :, 0, :, :, :]
                mean_pred = mean_pred_i
                mean_preds = mean_pred_i
                #mean_pred = np.mean(mean_pred_i, axis=0)
                if(aleatoric == True):
                    log_var = mc_preds_MC[:, :, 1, :, :, :]
                    print(f"log before: {log_var.shape}")
                    aleatoric_std = np.mean(np.sqrt(np.exp(log_var)), axis=0)
                    uncertainties.append(aleatoric_std)
                    print(f"aleatoric {aleatoric_std.shape}")
                
                #mc_preds_MC = mc_dropout_predict(_model_u_i, _dataset[:,0:1,:,:,:], mc_samples=num_samples)
                #epistemic_std = np.std(mc_preds_MC[:,:,0], axis=0)
                #mean_preds.append(mean_pred)
                #uncertainties_epistemic.append(epistemic_std)
                print(f"epistemic {epistemic_std.shape}")
                print(f"{uncertainties_epistemic[0]}")
            
                if(case == "Rotation"):
                    mean_preds_rot = mean_pred
                    print("Gets here Rotation")
                    #uncertainties_epistemic_rot.append(epistemic_std)
                if(case == "Shift"):
                    print("Gets here Shift")
                    mean_preds_shift = mean_pred
                    print(f"{np.mean(_dataset.cpu().detach().numpy(),axis=(1,2,3,4))}")
                    print(f"{np.mean(_dataset_shift.cpu().detach().numpy(),axis=(1,2,3,4))}")
                    print(f"{np.mean(mean_preds_shift,axis=(0,2,3,4))}")
                    print(f"{np.mean(mean_preds_original,axis=(0,2,3,4))}")
                    #uncertainties_epistemic_shift.append(epistemic_std)
                if(case == "CellShuffle"):
                    print("Gets here shuffle")
                    mean_preds_shuffle = mean_pred
                    #uncertainties_epistmic_shuffle.append(epistemic_std)
                if(case == "Random"):
                    print("Gets here random")
                    mean_preds_random = mean_pred
                    #uncertainties_epistemic_random.append(epistemic_std)
                if(case == "TrueRandom"):
                    print("Gets here true random")
                    mean_preds_randomtrue = mean_pred
                    #uncertainties_epistemic_randomtrue.append(epistemic_std)
                if(case == "OOD"):
                    print("Gets here OOD")
                    mean_preds_ood = mean_pred
                    #uncertainties_epistemic_ood.append(epistemic_std)
                if(case == "Couette"):
                    print("Gets here Couette")
                    mean_preds_couette = mean_pred
                    #uncertainties_epistemic_couette.append(epistemic_std)

            _dataset = _dataset_original
            mean_preds = mean_preds_original

        else:
            if(onlyX==True):
                print("gets here3")
                #_preds_x = _model_u_i(_dataset[:, 0, :, :, :])
                #_preds_y = _model_u_i(_dataset[:, 1, :, :, :])
                #_preds_z = _model_u_i(_dataset[:, 2, :, :, :])
            else:
                _preds_x = _model_u_i(_dataset[:, 0, :, :, :])
                _preds_y = _model_u_i(_dataset[:, 1, :, :, :])
                _preds__ = _model_u_i(_dataset[:, 2, :, :, :])

        """
        _preds = torch.cat((_preds_x, _preds_x, _preds_x), 1).to(device)
        #print(_preds_y)
        _preds = torch.add(_preds, -1.0).float().to(device)
        _preds = _preds.cpu().detach().numpy()
         # _targs = _targs.numpy()
        """

        """
        if(getEuclidianDistance==True):
            # Calculate Euclidean distance for each channel separately
            euclidean_distances = []
            diff_x = _preds_x.cpu().detach().numpy()[:,0,:,:,:] - _targs2[:,0,:,:,:]
            print(diff_x.shape) #(900, 900, 24, 24, 24)
            print(_preds_x.shape) #torch.Size([900, 1, 24, 24, 24]
            print(diff_x[0,0,0,0])
            #diff_y = _preds_y.cpu().detach().numpy() - _targs[:,1,:,:,:]
            #diff_z = _preds_z.cpu().detach().numpy() - _targs[:,2,:,:,:]
            euclidean_dist_x = np.sqrt(np.sum(diff_x**2, axis=(1, 2, 3)))
            print(euclidean_dist_x.shape)
            #euclidean_dist_y = np.sqrt(np.sum(diff_y**2, axis=(1, 2, 3)))
            #euclidean_dist_z = np.sqrt(np.sum(diff_z**2, axis=(1, 2, 3)))
            euclidean_distances.append(euclidean_dist_x)
            #euclidean_distances.append(euclidean_dist_x,euclidean_dist_y,euclidean_dist_z)
            euclidean_distances = np.array(euclidean_distances)  # shape (3, 1000)

            #np.save('euclidean_distances_ch1.npy', euclidean_distances[0])
            #np.save('euclidean_distances_ch2.npy', euclidean_distances[1])
            #np.save('euclidean_distances_ch3.npy', euclidean_distances[2])
            np.save(f'euclidean_distances_all_{_id}.npy', euclidean_distances)
            print(f"Euclidean distances for file {_id}.")
            print(f"Shape of distances per channel: {euclidean_distances[0].shape}")
            print(f"Mean distance - Channel 1: {np.mean(euclidean_distances[0]):.4f}")
            #print(f"Mean distance - Channel 2: {np.mean(euclidean_distances[1]):.4f}")
            #print(f"Mean distance - Channel 3: {np.mean(euclidean_distances[2]):.4f}")
        """

    time_steps = np.arange(0,t_max,1)
    

    #ALEATORIC
    
    if(aleatoric==True):
        #INCOMMENT
        """plot_flow_profile(
                np_datasets=[_targs, np.mean(np.array(mean_preds),axis=0)],
                dataset_legends=['MD', 'Autoencoder'],
                save2file=f'{_id}_Fig_Maker_5_a_ConvAE_vs_MD_22_new',
                t_max=t_max
        )"""
        #mean_pred = mean_pred - np.ones_like(mean_pred)
        #plt.plot(mean_pred[0:t_max,12,12,12] + (uncertainty[0:t_max,12,12,12]),label="Aleatoric",color="green")
        #plt.plot(mean_pred[0:t_max,12,12,12] - (uncertainty[0:t_max,12,12,12]),color="green")
        i=0
        #for uncertainty in uncertainties:
        plt.figure()
        if(Ensemble==True):
            mean_preds = np.array(mean_preds)
            mean_pred_i = np.mean(mean_preds,axis=0)
            if(ensemble==True):
                uncertainties = np.array(uncertainties)
                print(f"Uncertainties: {uncertainties.shape}")
                uncertainty = np.mean(uncertainties,axis=0)
                plt.fill_between(time_steps,mean_pred_i[0:t_max,12,12,12] - uncertainty[0:t_max,12,12,12] , mean_pred_i[0:t_max,12,12,12] + uncertainty[0:t_max,12,12,12], alpha=0.6,color="darkgreen", linestyle='dotted',zorder=5) #color="green"
        else:
            for uncertainty in uncertainties:
                plt.fill_between(time_steps,np.mean(mean_pred[:,0:t_max,12,12,12],axis=0) - uncertainty[0:t_max,12,12,12] , np.mean(mean_pred[:,0:t_max,12,12,12],axis=0) + uncertainty[0:t_max,12,12,12], alpha=0.6,color="darkgreen", linestyle='dotted',zorder=5) #color="green"
            #plt.fill_between(time_steps,mean_pred[0:t_max,12,12,12] - uncertainty[0:t_max,12,12,12] / 10, mean_pred[0:t_max,12,12,12] + uncertainty[0:t_max,12,12,12] / 10, alpha=0.6,color="darkgreen", linestyle='dotted',zorder=5) #color="green"

    #EPISTEMIC
    if(epistemic==True):
        if(MC==True or GP==True):
   
            for uncertainty_epistemic in uncertainties_epistemic:
                print(f"uncertainty2: {uncertainty_epistemic.shape}")
                print(f"{uncertainty_epistemic[0:t_max].shape}")
                print(f"mean: {mean_pred.shape}")
                plt.fill_between(time_steps,np.mean(mean_pred[:,0:t_max,12,12,12],axis=0) - uncertainty_epistemic[0:t_max,12,12,12], np.mean(mean_pred[:,0:t_max,12,12,12],axis=0) + uncertainty_epistemic[0:t_max,12,12,12], alpha=0.6,color="purple",zorder=9)
        if(Ensemble==True):
            #caluulate epistemic uncertainty for ensemble
            mean_preds = np.array(mean_preds)
            mean_pred_i = np.mean(mean_preds,axis=0)
            uncertainties_epistemic.append(np.std(mean_preds,axis=0))
            for uncertainty_epistemic in uncertainties_epistemic:
                plt.fill_between(time_steps,mean_pred_i[0:t_max,12,12,12] - uncertainty_epistemic[0:t_max,12,12,12], mean_pred_i[0:t_max,12,12,12] + uncertainty_epistemic[0:t_max,12,12,12], alpha=0.6,color="purple",zorder=9)

    #PREDICTION
    if(Prediction==True):
        if(Ensemble==True):
            mean_preds = np.array(mean_preds)
            print(mean_preds.shape)
            print(np.mean(mean_preds,axis=0).shape)
            print(t_max)
            mean_preds_k = np.mean(mean_preds,axis=0)
            plt.plot(mean_preds_k[0:t_max,12,12,12],color="darkblue",zorder=7)
            if(ensemble==True):
                i=0
                for mean_pred in mean_preds:
                    plt.plot(mean_pred[0:t_max,12,12,12],label=f"seed{i}",zorder=7)
                    i=i+1
            #mean_pred = mean_pred - np.ones_like(mean_pred)
        else:
            #for mean_pred in mean_preds:
            plt.plot(np.mean(mean_pred[:,0:t_max,12,12,12],axis=0),color="darkblue",zorder=7)
    print("test") 

    #TARGET
    if(Target==True):
        if(Ensemble==True):
            if(flag==0):
                plt.plot(_targs[0:t_max,0,12,12,12],color="orange",zorder=5)
                flag=1
        else:
            plt.plot(_targs[0:t_max,0,12,12,12],color="orange",zorder=5)
    #plt.fill_between(time_steps,mean_pred[0:t_max,12,12,12]- np.std(_targs[0:t_max,0,12,12,12]), mean_pred[0:t_max,12,12,12]+ np.std(_targs[0:t_max,0,12,12,12]), alpha=0.6,color="lightgreen", linestyle='dotted',zorder=4) #color="green"
    #plt.plot(time_steps,mean_pred[0:t_max,12,12,12]- np.ones_like(mean_pred[0:t_max,12,12,12]) * np.min(np.std(_targs[0:t_max,0,12,12,12])),"--", alpha=0.6,color="darkblue", linestyle='dotted',zorder=6) #color="green"
    #plt.plot(time_steps, mean_pred[0:t_max,12,12,12]+ np.ones_like(mean_pred[0:t_max,12,12,12]) * np.max(np.std(_targs[0:t_max,0,12,12,12])),"--", alpha=0.6,color="darkblue", linestyle='dotted',zorder=6) #color="green"
    

    #plt.plot(_targs[0:t_max,0,12,12,12]+np.std(_targs[0:t_max,0,12,12,12]),"--",color="darkgreen",zorder=5)
    #plt.plot(_targs[0:t_max,0,12,12,12]-np.std(_targs[0:t_max,0,12,12,12]),"--",color="darkgreen",zorder=5)
    # Sample custom handles
    custom_lines = [
        Line2D([0], [0], color='orange', lw=4),
        Line2D([0],[0], color="lightgreen",lw=4),
        Line2D([0],[0],color="black",lw=4),
        Line2D([0], [0], color='darkblue', lw=4),
        Line2D([0],[0],color="purple",lw=4),
        Line2D([0], [0], color='darkgreen', lw=4)

    ]
    plt.legend(custom_lines, ['Target', 'Target_std','Target_std_min_max','Predictions','Epistemic','Aleatoric'],loc="lower right")
    plt.xlabel("Coupling Cycles")
    plt.ylabel("ux [m/s]")
    #plt.ylim(0,10)
    plt.savefig(f"/home/sabebert/{outname}.png")
    
    Labels=["Original"]
    BarEpistemics = []
    BarMeans = []
    ViolinMeans=[]
    StdGT=[]
    ViolinStdGT=[]

    if(OOD==True or Couette==True):
        BarMeansOOD=[]
        BarEpistemicsOOD=[]
        LabelsOOD=[]
        StdGTOOD = []
        if(StandardDeviation==True):
            BarMeansOOD_std=[]
            StdGTOOD_std=[]
            BarEpistemicsOOD_std=[]

    if(StandardDeviation==True):
        BarMeans_std=[]
        StdGT_std=[]
        BarEpistemics_std=[]

    if np.array(mean_preds).ndim == 5:  # If single case, to also visualise
        print("GETS HERE")
        #mean_preds = [mean_preds] #np.array(mean_preds)[np.newaxis, :]  # Add dimension: (1, 1000, 3, 24, 24, 24)
        print(f"mean preds shape final: {np.array(mean_preds).shape}")
        #print(f"mean preds shift shape final: {np.array(mean_preds_shift).shape}")
        #print(f"mean preds ood shape final: {np.array(mean_preds_ood).shape}")
    
    #print(f"measured time shape file: {np.array(measured_time_file).shape}")
    if(len(measured_time_ensemble) < 1):
        #measured_time_files = np.zeros((len(_file_names)))
        measured_time_ensemble = np.zeros((len(_file_names),len(ensemble_models)))
    #print(f"measured time shape files: {np.array(measured_time_files).shape}")
    print(f"measured time shape ensemble: {np.array(measured_time_ensemble).shape}")
    ComputationalCost_time = np.mean(np.sum(np.array(measured_time_ensemble), axis=0),axis=0)  # Sum across ensemble members first, then mean across files
    print(f"ComputationalCost_time shape: {np.array(np.mean(np.sum(np.array(measured_time_ensemble),axis=0),axis=0))}")
    #print(f"ComputationalCost_time shape: {np.array(np.sum(np.array(measured_time_ensemble),axis=0))}")
    #print(f"ComputationalCost_time shape: {np.array(ComputationalCost_time).shape}")
    ComputationalCost_time = np.array([ComputationalCost_time for i in range(0,len(Labels))])
    print(len(ComputationalCost_time))
    print(f"SHape after concat: {ComputationalCost_time.shape}")
    ComputationalCost_time_std = np.std(np.sum(np.array(measured_time_ensemble), axis=0))  # Sum across ensemble members first, then std across files
    ComputationalCost_time_std = np.array([ComputationalCost_time_std for i in range(0,len(Labels))])
    ComputationalCost_memory = np.mean(np.sum(np.array(measured_memory_ensemble), axis=0))  # Sum across ensemble members first, then mean across files
    ComputationalCost_memory = np.array([ComputationalCost_memory for i in range(0,len(Labels))])
    ComputationalCost_memory_std = np.std(np.sum(np.array(measured_memory_ensemble), axis=0))  # Sum across ensemble members first, then s
    ComputationalCost_memory_std = np.array([ComputationalCost_memory_std for i in range(0,len(Labels))])

    BarMeans.append(np.mean(np.array(mean_preds)[:, 0:t_max], axis=(0,2,3,4)))
    StdGT.append(np.abs(np.mean(np.array(_dataset)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds)[:, 0:t_max], axis=(0,2,3,4))))
    ViolinMeans.append(np.mean(np.array(mean_preds)[:, 0:t_max], axis=(2,3,4)))
    ViolinStdGT.append([np.abs(np.mean(np.array(_dataset)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_i)[0:t_max], axis=(1,2,3))) for mean_preds_i in mean_preds])
    BarEpistemics.append(np.mean(uncertainty_epistemic[0:t_max],axis=(1,2,3)))
    if(StandardDeviation==True):
        BarMeans_std.append(np.std([np.mean(np.array(mean_preds)[:,:], axis=(0,2,3,4))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))
        print(f"BarMeans_std:{np.array(BarMeans_std).shape}")
        StdGT_std.append(np.std([np.abs(np.mean(np.array(_dataset)[:,0], axis=(1,2,3))[f*t_max:(f+1)*t_max]-np.mean(np.array(mean_preds)[:,:],axis=(0,2,3,4))[f*t_max:(f+1)*t_max]) for f in range(len(_file_names))]))
        BarEpistemics_std.append(np.std([np.mean(np.array(uncertainty_epistemic)[:,:], axis=(1,2,3))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))


    if(Rotation==True):
        uncertainty_epistemic_rot = np.std(np.array(mean_preds_rot),axis=0)
        Labels.append("ID-IT-Rotated")
        BarMeans.append(np.mean(np.array(mean_preds_rot)[:,0:t_max],axis=(0,2,3,4)))
        StdGT.append(np.abs(np.mean(np.array(_dataset_rot)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_rot)[:, 0:t_max], axis=(0,2,3,4))))
        ViolinMeans.append(np.mean(np.array(mean_preds_rot)[:, 0:t_max], axis=(2,3,4)))
        ViolinStdGT.append([np.abs(np.mean(np.array(_dataset_rot)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_rot_i)[0:t_max], axis=(1,2,3))) for mean_preds_rot_i in mean_preds_rot])
        BarEpistemics.append(np.mean(uncertainty_epistemic_rot[0:t_max],axis=(1,2,3)))

        if(StandardDeviation==True):
            BarMeans_std.append(np.std([np.mean(np.array(mean_preds_rot)[:,:], axis=(0,2,3,4))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))
            print(f"BarMeans_std:{np.array(BarMeans_std).shape}")
            StdGT_std.append(np.std([np.abs(np.mean(np.array(_dataset_rot)[:,0], axis=(1,2,3))[f*t_max:(f+1)*t_max]-np.mean(np.array(mean_preds_rot)[:,:],axis=(0,2,3,4))[f*t_max:(f+1)*t_max]) for f in range(len(_file_names))]))
            BarEpistemics_std.append(np.std([np.mean(np.array(uncertainty_epistemic_rot)[:,:], axis=(1,2,3))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))

    
    if(CellShuffle==True):
        uncertainty_epistemic_shuffle = np.std(np.array(mean_preds_shuffle),axis=0)
        Labels.append("ID-IT-CellShuffle")
        BarMeans.append(np.mean(np.array(mean_preds_shuffle)[:,0:t_max],axis=(0,2,3,4)))
        StdGT.append(np.abs(np.mean(np.array(_dataset_shuffle)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_shuffle)[:,0:t_max], axis=(0,2,3,4))))
        ViolinMeans.append(np.mean(np.array(mean_preds_shuffle)[:, 0:t_max], axis=(2,3,4)))
        ViolinStdGT.append([np.abs(np.mean(np.array(_dataset_shuffle)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_shuffle_i)[0:t_max], axis=(1,2,3))) for mean_preds_shuffle_i in mean_preds_shuffle])
        BarEpistemics.append(np.mean(uncertainty_epistemic_shuffle[0:t_max],axis=(1,2,3)))

        if(StandardDeviation==True):
            BarMeans_std.append(np.std([np.mean(np.array(mean_preds_shuffle)[:,:], axis=(0,2,3,4))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))
            print(f"BarMeans_std:{np.array(BarMeans_std).shape}")
            StdGT_std.append(np.std([np.abs(np.mean(np.array(_dataset_shuffle)[:,0], axis=(1,2,3))[f*t_max:(f+1)*t_max]-np.mean(np.array(mean_preds_shuffle)[:,:],axis=(0,2,3,4))[f*t_max:(f+1)*t_max]) for f in range(len(_file_names))]))
            BarEpistemics_std.append(np.std([np.mean(np.array(uncertainty_epistemic_shuffle)[:,:], axis=(1,2,3))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))
    

    if(Shift==True):
        uncertainty_epistemic_shift = np.std(np.array(mean_preds_shift),axis=0)
        Labels.append("ID-IT-Shift")
        BarMeans.append(np.mean(np.array(mean_preds_shift)[:,0:t_max],axis=(0,2,3,4)))
        StdGT.append(np.abs(np.mean(np.array(_dataset_shift)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_shift)[:, 0:t_max], axis=(0,2,3,4))))
        ViolinMeans.append(np.mean(np.array(mean_preds_shift)[:, 0:t_max], axis=(2,3,4)))
        ViolinStdGT.append([np.abs(np.mean(np.array(_dataset_shift)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_shift_i)[0:t_max], axis=(1,2,3))) for mean_preds_shift_i in mean_preds_shift])
        BarEpistemics.append(np.mean(uncertainty_epistemic_shift[0:t_max],axis=(1,2,3)))

        if(StandardDeviation==True):
            BarMeans_std.append(np.std([np.mean(np.array(mean_preds_shift)[:,:], axis=(0,2,3,4))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))
            print(f"BarMeans_std:{np.array(BarMeans_std).shape}")
            StdGT_std.append(np.std([np.abs(np.mean(np.array(_dataset)[:,0], axis=(1,2,3))[f*t_max:(f+1)*t_max]-np.mean(np.array(mean_preds_shift)[:,:],axis=(0,2,3,4))[f*t_max:(f+1)*t_max]) for f in range(len(_file_names))]))
            BarEpistemics_std.append(np.std([np.mean(np.array(uncertainty_epistemic_shift)[:,:], axis=(1,2,3))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))

    if(Random==True):
        uncertainty_epistemic_random = np.std(np.array(mean_preds_random),axis=0)
        Labels.append("ID-IT+Noise(0-2)")
        BarMeans.append(np.mean(np.array(mean_preds_random)[:,0:t_max],axis=(0,2,3,4)))
        StdGT.append(np.abs(np.mean(np.array(_dataset_random)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_random)[:,0:t_max], axis=(0,2,3,4))))
        ViolinMeans.append(np.mean(np.array(mean_preds_random)[:, 0:t_max], axis=(2,3,4)))
        ViolinStdGT.append([np.abs(np.mean(np.array(_dataset_random)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_random_i)[0:t_max], axis=(1,2,3))) for mean_preds_random_i in mean_preds_random])
        BarEpistemics.append(np.mean(uncertainty_epistemic_random[0:t_max],axis=(1,2,3)))

        if(StandardDeviation==True):
            BarMeans_std.append(np.std([np.mean(np.array(mean_preds_random)[:,:], axis=(0,2,3,4))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))
            print(f"BarMeans_std:{np.array(BarMeans_std).shape}")
            StdGT_std.append(np.std([np.abs(np.mean(np.array(_dataset_random)[:,0], axis=(1,2,3))[f*t_max:(f+1)*t_max]-np.mean(np.array(mean_preds_random)[:,:],axis=(0,2,3,4))[f*t_max:(f+1)*t_max]) for f in range(len(_file_names))]))
            BarEpistemics_std.append(np.std([np.mean(np.array(uncertainty_epistemic_random)[:,:], axis=(1,2,3))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))
    
    if(TrueRandom==True):
        uncertainty_epistemic_randomtrue = np.std(np.array(mean_preds_randomtrue),axis=0)
        Labels.append("TrueRandom(1-11)")
        BarMeans.append(np.mean(np.array(mean_preds_randomtrue)[:,0:t_max],axis=(0,2,3,4)))
        StdGT.append(np.abs(np.mean(np.array(_dataset_randomtrue)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_randomtrue)[:,0:t_max], axis=(0,2,3,4))))
        ViolinMeans.append(np.mean(np.array(mean_preds_randomtrue)[:, 0:t_max], axis=(2,3,4)))
        ViolinStdGT.append([np.abs(np.mean(np.array(_dataset_randomtrue)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_randomtrue_i)[0:t_max], axis=(1,2,3))) for mean_preds_randomtrue_i in mean_preds_randomtrue])
        BarEpistemics.append(np.mean(uncertainty_epistemic_randomtrue[0:t_max],axis=(1,2,3)))

        if(StandardDeviation==True):
            BarMeans_std.append(np.std([np.mean(np.array(mean_preds_randomtrue)[:,:], axis=(0,2,3,4))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))
            print(f"BarMeans_std:{np.array(BarMeans_std).shape}")
            StdGT_std.append(np.std([np.abs(np.mean(np.array(_dataset_randomtrue)[:,0], axis=(1,2,3))[f*t_max:(f+1)*t_max]-np.mean(np.array(mean_preds_randomtrue)[:,:],axis=(0,2,3,4))[f*t_max:(f+1)*t_max]) for f in range(len(_file_names))]))
            BarEpistemics_std.append(np.std([np.mean(np.array(uncertainty_epistemic_randomtrue)[:,:], axis=(1,2,3))[f*t_max:(f+1)*t_max] for f in range(len(_file_names))]))
     
    if(OOD==True):
        uncertainty_epistemic_ood = np.std(np.array(mean_preds_ood),axis=0)
        LabelsOOD.append("ID-OOT")
        BarMeansOOD.append(np.mean(np.array(mean_preds_ood)[:,0:t_max],axis=(0,2,3,4)))
        StdGTOOD.append(np.abs(np.mean(np.array(_dataset_ood)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_ood)[:,0:t_max], axis=(0,2,3,4))))
        ViolinMeans.append(np.mean(np.array(mean_preds_ood)[:, 0:t_max], axis=(2,3,4)))
        ViolinStdGT.append([np.abs(np.mean(np.array(_dataset_ood)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_ood_i)[0:t_max], axis=(1,2,3))) for mean_preds_ood_i in mean_preds_ood])
        BarEpistemicsOOD.append(np.mean(uncertainty_epistemic_ood[0:t_max],axis=(1,2,3)))
        #BarMeans.append(np.mean(mean_preds_ood[0:t_max],axis=(0,2,3,4)))
        #BarEpistemics.append(np.mean(uncertainty_epistemic_ood[0:t_max],axis=(1,2,3)))
        if(StandardDeviation==True):
            BarMeansOOD_std.append(np.std([np.mean(np.array(mean_preds_ood)[:,:], axis=(0,2,3,4))[f*t_max:(f+1)*t_max] for f in range(len(_file_names_ood))]))
            print(f"BarMeans_std:{np.array(BarMeans_std).shape}")
            StdGTOOD_std.append(np.std([np.abs(np.mean(np.array(_dataset_ood)[:,0], axis=(1,2,3))[f*t_max:(f+1)*t_max]-np.mean(np.array(mean_preds_ood)[:,:],axis=(0,2,3,4))[f*t_max:(f+1)*t_max]) for f in range(len(_file_names_ood))]))
            BarEpistemicsOOD_std.append(np.std([np.mean(np.array(uncertainty_epistemic_ood)[:,:], axis=(1,2,3))[f*t_max:(f+1)*t_max] for f in range(len(_file_names_ood))]))


    if(Couette==True):
        uncertainty_epistemic_couette = np.std(np.array(mean_preds_couette),axis=0)
        LabelsOOD.append("OOD-OOT-Couette")
        BarMeansOOD.append(np.mean(np.array(mean_preds_couette)[:,0:t_max],axis=(0,2,3,4)))
        StdGTOOD.append(np.abs(np.mean(np.array(_dataset_couette)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_couette)[:, 0:t_max], axis=(0,2,3,4))))
        
        ViolinMeans.append(np.mean(np.array(mean_preds_couette)[:, 0:t_max], axis=(2,3,4)))
        ViolinStdGT.append([np.abs(np.mean(np.array(_dataset_couette)[0:t_max,0],axis=(1,2,3)) - np.mean(np.array(mean_preds_couette_i)[0:t_max], axis=(1,2,3))) for mean_preds_couette_i in mean_preds_couette])
        BarEpistemicsOOD.append(np.mean(uncertainty_epistemic_couette[0:t_max],axis=(1,2,3)))
        #BarMeans.append(np.mean(mean_preds_couette[0:t_max],axis=(0,2,3,4)))
        #BarEpistemics.append(np.mean(uncertainty_epistemic_couette[0:t_max],axis=(1,2,3)))
        #Plotting for comparison:

        # Prepare data
        gt = np.array(_dataset_couette)[0:t_max,0]
        pred = np.mean(np.array(mean_preds_couette),axis=0)[0:t_max] #np.mean(np.array(mean_preds_couette)[:, 0:t_max],axis=0)
        mid_t = 500 # len(gt) // 2
        # Calculate MAE and means
        mae = np.abs(np.mean(gt, axis=(1,2,3)) - np.mean(pred, axis=(1,2,3)))
        gt_means = np.mean(gt, axis=(1,2,3))
        pred_means = np.mean(pred, axis=(1,2,3))
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        # 1. Time series comparison and mae
        axes[0,0].plot(gt_means, 'b-', label='Ground Truth', lw=2)
        axes[0,0].plot(pred_means, 'r--', label='Prediction', lw=2)
        axes[0,0].plot(mae,'g--',label='MAE',lw=2) #mae
        axes[0,0].set_title('Spatial Averages vs Time')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        im1 = axes[0,1].imshow(gt[mid_t, 0], cmap='RdBu_r', aspect='auto')
        axes[0,1].set_title(f'GT U-field (t={mid_t},X,Z=12)')
        plt.colorbar(im1, ax=axes[0,1])
        # 2. 2D field comparison - Prediction
        im2 = axes[0,2].imshow(pred[mid_t, 0], cmap='RdBu_r', aspect='auto')
        axes[0,2].set_title(f'Pred U-field (t={mid_t},X,Z=12)')
        plt.colorbar(im2, ax=axes[0,2])
        # 3. Error field between 2D slices
        error = gt[mid_t, 0] - pred[mid_t, 0]
        im3 = axes[1,0].imshow(error, cmap='RdBu_r', aspect='auto')
        axes[1,0].set_title(f'Error (Max: {np.max(np.abs(error)):.4f},X,Z=12)')
        plt.colorbar(im3, ax=axes[1,0])
        # 4. Normalisation 2D slices
        im4 = axes[1,1].imshow(gt[mid_t, 0]/np.mean(gt[mid_t,0]), cmap='RdBu_r', aspect='auto')
        axes[1,1].set_title(f'GT Normalized U-filed (t={mid_t},X,Z=12)')
        axes[1,1].grid(True, alpha=0.3)
        plt.colorbar(im4, ax=axes[1,1])
        im5 = axes[1,2].imshow(pred[mid_t, 0]/np.mean(pred[mid_t,0]), cmap='RdBu_r', aspect='auto')
        axes[1,2].set_title(f'Pred Normalized U-filed (t={mid_t},X,Z=12)')
        axes[1,2].grid(True, alpha=0.3)
        plt.colorbar(im5, ax=axes[1,2])
        # 6. Velocity profile (check if linear Couette)
        profile_gt = gt[mid_t, 12, :, gt.shape[3]//2]
        profile_pred = pred[mid_t, 12, :, pred.shape[3]//2]
        axes[2,0].plot(profile_gt, 'b-', label='GT', lw=2)
        axes[2,0].plot(profile_pred, 'r--', label='Pred', lw=2)
        axes[2,0].set_title('Velocity Profile Slice X=12,Z=12')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        axes[2,1].hist(gt.flatten(), bins=50)
        axes[2,2].hist(pred.flatten(),bins=50)
        axes[2,1].set_title("Velocitydistribution Cell GT")
        axes[2,2].set_title("Velocitydistribution Cell Pred")
        axes[2,1].set_ylabel("Count")
        axes[2,1].set_xlabel("Velocity [m/s]")
        axes[2,2].set_xlabel("Velocity [m/s]")
        axes[2,2].set_ylabel("Count")
        plt.tight_layout()
        plt.savefig("/home/sabebert/Couette_comparision.png")
        # Quick diagnostics
        print(f"MAE: {np.mean(mae):.6f}")
        print(f"Data range GT: [{np.min(gt):.4f}, {np.max(gt):.4f}]")
        print(f"Data range Pred: [{np.min(pred):.4f}, {np.max(pred):.4f}]")
        np.save("targ_couette.npy",gt)
        np.save("pred_couette_ae.npy",pred)
        if(StandardDeviation==True):
            BarMeansOOD_std.append(np.std([np.mean(np.array(mean_preds_couette)[:,:], axis=(0,2,3,4))[f*t_max:(f+1)*t_max] for f in range(len(_file_names_couette))]))
            print(f"BarMeans_std:{np.array(BarMeans_std).shape}")
            StdGTOOD_std.append(np.std([np.abs(np.mean(np.array(_dataset_couette)[:,0], axis=(1,2,3))[f*t_max:(f+1)*t_max]-np.mean(np.array(mean_preds_couette)[:,:],axis=(0,2,3,4))[f*t_max:(f+1)*t_max]) for f in range(len(_file_names_couette))]))
            BarEpistemicsOOD_std.append(np.std([np.mean(np.array(uncertainty_epistemic_couette)[:,:], axis=(1,2,3))[f*t_max:(f+1)*t_max] for f in range(len(_file_names_couette))]))

    #ID
    #MEAN OVER TIME

    #Bar/Boxplot
    colors = ["blue","orange","green","red","purple","brown"]
    colors = colors[0:len(BarEpistemics)]
    fig, (ax1, ax2, ax0) = plt.subplots(3, 1, figsize=(8, 6))
    sns.set_style("whitegrid")
    ax1.bar(Labels,[np.mean(bar,axis=0) for bar in BarEpistemics],yerr=np.array(BarEpistemics_std),capsize=3.0,color=colors)
    #ax1.bar(Labels,[np.mean(bar,axis=0) for bar in BarEpistemics])
    ax1.set_ylabel("Mean Epistemic Uncertainty[m/s]")
    ax2.bar(Labels,[np.mean(st,axis=0) for st in StdGT],yerr= np.array(StdGT_std),capsize=3.0,color=colors)
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("MAE [m/s]")
    print(BarMeans_std)
    plots = ax0.violinplot([np.mean(violin,axis=1).flatten() for violin in ViolinMeans],showmeans=True)
    # Set the color of the violin patches
    for pc, color in zip(plots['bodies'], colors):
        pc.set_facecolor(color)
    # Set the color of the median lines
    plots['cmeans'].set_colors(colors)
    # Set the labels
    ax0.set_xticks(np.array(range(1,len(BarMeans)+1,1)), labels=Labels)
    ax0.set_ylabel("Mean Epistemic Uncertainty[m/s]")
    ax0.set_xlabel("Dataset")
    #ax3.bar(Labels,[np.mean(bar,axis=0) for bar in BarMeans])
    #ax3.set_ylabel("Mean Amplitudes [m/s]")
    #for index,epi2 in enumerate(BarMeans):
    #    ax4.plot(epi2,label=Labels[index])
    #ax4.set_ylabel("Mean Amplitudes [m/s]")
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()
    plt.savefig("/home/sabebert/EpistemicStd_Augment.png")
    

    #TIME BEHAVIOUR
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 6))
    #ax1.errorbar(Labels,[np.mean(bar,axis=0) for bar in BarEpistemics],yerr=[np.std(bar,axis=0) for bar in BarEpistemics])
    for index,epi in enumerate(BarEpistemics):
        ax3.plot(epi ,label=Labels[index])
    #ax1.bar(Labels,[np.mean(bar,axis=0) for bar in BarEpistemics])
    ax3.set_ylabel("Mean Epistemic Uncertainty[m/s]")
    ax3.set_xlabel("Time")
    for index,st in enumerate(StdGT):
        ax4.plot(st ,label=Labels[index])
    ax4.set_xlabel("Time")
    ax4.set_ylabel("MAE [m/s]")
    ax3.legend()
    ax4.legend()
    ax3.grid(True)
    ax4.grid(True)
    print([np.mean(bar).shape for bar in BarMeans])
    print([np.mean(bar,axis=0).shape for bar in BarMeans])
    #ax3.bar(Labels,[np.mean(bar,axis=0) for bar in BarMeans])
    #ax3.set_ylabel("Mean Amplitudes [m/s]")
    #for index,epi2 in enumerate(BarMeans):
    #    ax4.plot(epi2,label=Labels[index])
    #ax4.set_ylabel("Mean Amplitudes [m/s]")
    plt.savefig("/home/sabebert/EpistemicStd_Augment_Time.png")

    fig, (ax5,ax6) = plt.subplots(2, 1, figsize=(8, 6))
    #ax3.bar(Labels,[np.mean(bar,axis=0) for bar in BarMeans])
    ax5.bar(Labels,[np.mean(bar,axis=0) for bar in BarMeans],yerr=np.array(BarMeans_std),color=colors,capsize=3.0)
    ax5.set_ylabel("Mean Amplitude [m/s]")
    for index,epi2 in enumerate(BarMeans):
        ax6.plot(epi2,label=Labels[index])
    ax6.set_ylabel("Mean Amplitude [m/s]")
    ax5.grid(True)
    ax6.grid(True)
    ax6.legend()
    plt.savefig("/home/sabebert/EpistemicMeans_Augment.png")
    
    #OOD
    if(OOD==True or Couette==True):
        #MEANS
        colors = ["blue","orange"]
        colors = colors[0:len(BarEpistemicsOOD)]
        fig, (ax1_2, ax2_2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1_2.bar(LabelsOOD,[np.mean(bar,axis=0) for bar in BarEpistemicsOOD],yerr=np.array(BarEpistemicsOOD_std),color=colors)
        #ax1.bar(Labels,[np.mean(bar,axis=0) for bar in BarEpistemics])
        ax1_2.set_ylabel("Mean Epistemic Uncertainty[m/s]")
        ax2_2.bar(LabelsOOD,[np.mean(st,axis=0) for st in StdGTOOD],yerr=np.array(StdGTOOD_std),color=colors)
        ax2_2.set_xlabel("Dataset")
        ax2_2.set_ylabel("MAE [m/s]")
        #print([np.mean(bar).shape for bar in BarMeans])
        #print([np.mean(bar,axis=0).shape for bar in BarMeans])
        ax1_2.grid(True)
        ax2_2.grid(True)
        ax1_2.legend()
        ax2_2.legend()
        plt.savefig("/home/sabebert/EpistemicStd_OOD.png")

        #TIME BEHAVIOUR
        fig, (ax3_2, ax4_2) = plt.subplots(2, 1, figsize=(8, 6))
        #ax1.errorbar(Labels,[np.mean(bar,axis=0) for bar in BarEpistemics],yerr=[np.std(bar,axis=0) for bar in BarEpistemics])
        for index,epi in enumerate(BarEpistemicsOOD):
            ax3_2.plot(epi,label=LabelsOOD[index])
        #ax1.bar(Labels,[np.mean(bar,axis=0) for bar in BarEpistemics])
        ax3_2.set_ylabel("Mean Epistemic Uncertainty[m/s]")
        ax3_2.set_xlabel("Time")
        for index,st in enumerate(StdGTOOD):
            ax4_2.plot(st,label=LabelsOOD[index])
        ax4_2.set_xlabel("Time")
        ax4_2.set_ylabel("MAE [m/s]")
        ax3_2.grid(True)
        ax4_2.grid(True)
        ax3_2.legend()
        ax4_2.legend()
        print(BarMeansOOD)
        print(BarMeansOOD)
        #ax3.bar(Labels,[np.mean(bar,axis=0) for bar in BarMeans])
        #ax3.set_ylabel("Mean Amplitudes [m/s]")
        #for index,epi2 in enumerate(BarMeans):
        #    ax4.plot(epi2,label=Labels[index])
        #ax4.set_ylabel("Mean Amplitudes [m/s]")
        plt.savefig("/home/sabebert/EpistemicStd_OOD_Time.png")

        fig, (ax5_2,ax6_2) = plt.subplots(2, 1, figsize=(8, 6))
        #ax3.bar(Labels,[np.mean(bar,axis=0) for bar in BarMeans])
        ax5_2.bar(LabelsOOD,[np.mean(bar,axis=0) for bar in BarMeansOOD],yerr=np.array(BarMeansOOD_std),color=colors)
        ax5_2.set_ylabel("Mean Amplitude [m/s]")
        for index,epi2 in enumerate(BarMeansOOD):
            ax6_2.plot(epi2,label=LabelsOOD[index])
        ax6_2.set_ylabel("Mean Amplitude [m/s]")
        ax5_2.grid(True)
        ax6_2.grid(True)
        ax6_2.legend()
        plt.savefig("/home/sabebert/EpistemicMeans_OOD.png")
        
        #TESTS
        #print(sc.stats.ttest_ind(BarEpistemics[0],BarEpistemics[1]))
        #print(sc.stats.ttest_ind(BarEpistemics[0],BarEpistemics[2]))
        #print(sc.stats.ttest_ind(BarEpistemics[0],BarEpistemics[3]))
        #print(sc.stats.ttest_ind(BarEpistemics[0],BarEpistemicsOOD[0]))
        #print(sc.stats.ttest_ind(BarEpistemics[0],BarEpistemicsOOD[1]))
        
        # All statistical tests in one loop
        comparisons = [(BarEpistemics[0], BarEpistemics[i], f"ID[0]vs[{i}]") for i in range(1,len(augments)-1)] + \
              [(BarEpistemics[0], BarEpistemicsOOD[i], f"ID[0]vsOOD[{i}]") for i in [0,1]]
        for x, y, label in comparisons:
            t_stat, t_p = sc.stats.ttest_ind(x, y)                   #T-Test two-sided, Mean comparision
            ks_stat, ks_p = sc.stats.ks_2samp(x, y)                  #Kolmogorov-Smirnov Test, Distribution comparison (for OOD detection)
            ad_stat, ad_crit, ad_p = sc.stats.anderson_ksamp([x, y]) #Anderson-Darling Test, Tail-sensitive distribution comparison
            lev_stat, lev_p = sc.stats.levene(x, y)                  #Levene's Test, Variance homogeneity for method reliability
            print(f"{label}: T:{t_stat:.3f}({t_p:.3f}) KS:{ks_stat:.3f}({ks_p:.3f}) AD:{ad_stat:.3f}({ad_p:.3f}) L:{lev_stat:.3f}({lev_p:.3f})")
        # Correlations of epistemic uncertainty to Ground Truth deviation
        print("\nCorrelations:")
        """
        for i in range(len(BarEpistemics)):
            p_r,p_p = sc.stats.pearsonr(BarEpistemics[i], StdGT[i])
            s_r,s_p = sc.stats.spearmanr(BarEpistemics[i], StdGT[i])
            k_tau, k_p = sc.stats.kendalltau(BarEpistemics[i], StdGT[i])
            correlation = sc.signal.correlate(BarEpistemics[i], StdGT[i])
            print(f"ID[{i}]: P:{p_r:.3f}({p_p:.3f}) S:{s_r:.3f}({s_p:.3f}) K:{k_tau:.3f}({k_p:.3f})")
            print(f"temporal cross-correlation:{correlation}")
        for i in range(len(BarEpistemicsOOD)):
            p_r,p_p = sc.stats.pearsonr(BarEpistemicsOOD[i], StdGTOOD[i])
            s_r,s_p = sc.stats.spearmanr(BarEpistemicsOOD[i], StdGTOOD[i])
            k_tau, k_p = sc.stats.kendalltau(BarEpistemics[i], StdGT[i])
            print(f"OOD[{i}]: P:{p_r:.3f}({p_p:.3f}) S:{s_r:.3f}({s_p:.3f}) K:{k_tau:.3f}({k_p:.3f})")
            """

        plt.figure(figsize=(6, 3))
        for i in range(len(BarEpistemics)):
            a = np.asarray(BarEpistemics[i])
            b = np.asarray(StdGT[i])
            # Normalize
            a = (a - np.mean(a)) / np.std(a)
            b = (b - np.mean(b)) / np.std(b)
            # Correlations
            p_r, p_p = sc.stats.pearsonr(a, b)
            s_r, s_p = sc.stats.spearmanr(a, b)
            k_tau, k_p = sc.stats.kendalltau(a, b)
            correlation = sc.signal.correlate(BarEpistemics[i],StdGT[i])
            print(f"ID[{i}]: P:{p_r:.3f})({p_p:.3f}) S:{s_r:.3f}({s_p:.3f} K:{k_tau:.3f}({k_p:.3f})")
            # Cross-correlation (lag)
            corr = sc.signal.correlate(a, b, mode='full')
            lags = sc.signal.correlation_lags(len(a), len(b), mode='full')
            peak_lag = lags[np.argmax(corr)]
            plt.plot(lags, corr / np.max(np.abs(corr)), label=f"{Labels[i]} f'P: {p_r:.2f}, Sp: {s_r:.2f}, K: {k_tau:.2f}')")
            plt.axvline(peak_lag, color='orange', ls='--', label=f'Peak lag: {peak_lag}')
        plt.title(f'Correlations and cross-correlations ID')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.legend()
        plt.tight_layout()
        plt.savefig("/home/sabebert/Correlations_ID.png")
        print(f"temporal cross-correlation:{correlation}")
        
        for i in range(len(BarEpistemicsOOD)):
            a_ood = np.asarray(BarEpistemicsOOD[i])
            b_ood = np.asarray(StdGTOOD[i])
            a_ood = (a_ood - np.mean(a_ood)) / np.std(a_ood)
            b_ood = (b_ood - np.mean(b_ood)) / np.std(b_ood)
            p_r_2,p_p_2 = sc.stats.pearsonr(a_ood,b_ood)
            s_r_2,s_p_2 = sc.stats.spearmanr(a_ood,b_ood)
            k_tau_2,k_p_2 = sc.stats.kendalltau(a_ood,b_ood)
            print(f"OOD[{i}]: P:{p_r_2:.3f})({p_p_2:.3f}) S:{s_r_2:.3f}({s_p_2:.3f} K:{k_tau:.3f}({k_p_2:.3f})")

        # ROC-AUC for OOD detection
        """
        aucs = []
        for i in range(len(BarEpistemics)):
            for j in range(min(len(BarEpistemicsOOD), 2)):
                unc = np.concatenate([BarEpistemics[i], BarEpistemicsOOD[j]])
                lbl = np.concatenate([np.zeros(len(BarEpistemics[i])), np.ones(len(BarEpistemicsOOD[j]))])
                aucs.append(roc_auc_score(lbl, unc))
                fpr, tpr, _ = roc_curve(lbl, unc)
                auc = roc_auc_score(lbl, unc)
                plt.plot(fpr, tpr, label=f'ID[{i}] vs OOD[{j}] (AUC={auc:.3f})')

        auc_str = " ".join([f"AUC{j}:{aucs[j]:.3f}" for j in range(len(aucs))])
        print(f"ID[{i}]: P:{p_r:.3f}({p_p:.3f}) S:{s_r:.3f}({s_p:.3f}) {auc_str}")
        plt.figure(figsize=(10, 6))

        plt.plot([0,1], [0,1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves: OOD Detection via Epistemic Uncertainty')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("/home/sabebert/ROC_test.png")
        """

    aucs_error, aucs_ood = [], []
    precisions, recalls, brier_scores,brier_scores_stds = [], [], [], []
    percentiles = [50, 70, 80, 90]
    #colors_comp = ["blue","orange","green","red","purple","brown"]
    colors_mae = ["grey","grey","grey","grey","grey","grey"] #["blue","orange","green","red","purple","brown"]
    colors_spear = ["blue","blue","blue","blue","blue","blue"]#["blue","orange","green","red","purple","brown"]
    colors_brier = ["red","red","red","red","red","red"]
    n_datasets = len(BarEpistemics) + len(BarEpistemicsOOD)
    dataset_colors = colors[0:n_datasets]
    #labels = [f"Dataset {i}" for i in range(n_datasets)]  # Adjust to your Labels variable
    labels = np.concatenate([Labels,LabelsOOD])
    #colors_auc = ["red","green","green","green","green","green"]#["blue","orange","green","red","purple","brown"]
    # Main ROC analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    for j in range(0,len(BarEpistemics)+len(BarEpistemicsOOD)):
        if(j < len(BarEpistemics)):
            unc = np.concatenate([BarEpistemics[0], BarEpistemics[j]])
            err = np.concatenate([StdGT[0], StdGT[j]])
            ood_lbl = np.concatenate([np.zeros(len(BarEpistemics[0])), np.ones(len(BarEpistemics[j]))])
        if(j >= len(BarEpistemics)):
            unc = np.concatenate([BarEpistemics[0], BarEpistemicsOOD[j-len(BarEpistemics)]])
            err = np.concatenate([StdGT[0], StdGTOOD[j - len(BarEpistemics)]])
            ood_lbl = np.concatenate([np.zeros(len(BarEpistemics[0])), np.ones(len(BarEpistemicsOOD[j-len(BarEpistemics)]))])

        # Error correlation analysis
        err_thresh = np.percentile(err, 20)
        err_lbl = (err > err_thresh).astype(int)
        auc_err = roc_auc_score(err_lbl, unc)
        aucs_error.append(auc_err)
        
        fpr_err, tpr_err, _ = roc_curve(err_lbl, unc)
        ax1.plot(fpr_err, tpr_err, label=f'Original vs OOD[{j}] (AUC={auc_err:.3f})')
        
        # OOD detection analysis  
        auc_ood = roc_auc_score(ood_lbl, unc)
        aucs_ood.append(auc_ood)
        
        fpr_ood, tpr_ood, _ = roc_curve(ood_lbl, unc)
        ax2.plot(fpr_ood, tpr_ood, label=f'Original vs OOD[{j}] (AUC={auc_ood:.3f})')
        
        # Dual threshold analysis for precision-recall and Brier Score
        case_prec, case_rec, case_brier, case_brier_stds = [], [], [], []
        
        print(f"\nOriginal vs OOD[{j}]: Error-AUC={auc_err:.3f}, OOD-AUC={auc_ood:.3f}")
        print(f"Error threshold (20th percentile): {err_thresh:.4f}")
        
        for unc_p in percentiles:
            #unc_p = 20
            unc_thresh = np.percentile(unc, unc_p)
            tp = np.sum((err > err_thresh) & (unc > unc_thresh))
            fp = np.sum((err <= err_thresh) & (unc > unc_thresh))
            tn = np.sum((err <= err_thresh) & (unc <= unc_thresh))
            fn = np.sum((err > err_thresh) & (unc <= unc_thresh))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            case_prec.append(prec)
            case_rec.append(rec)
            
        unc_p = 20
        unc_thresh = np.percentile(unc, unc_p)
        # Calculate Brier Score
        unc_prob = (unc - np.min(unc)) / (np.max(unc) - np.min(unc))
        brier = brier_score_loss(err_lbl, unc_prob)
        bs_ref = np.mean(err_lbl) * (1 - np.mean(err_lbl))
        bss = 1 - (brier / bs_ref) if bs_ref > 0 else 0
            
        brier_scores.append(bss)
        n = len(BarEpistemics[i] + BarEpistemicsOOD)
        brier_scores_stds.append(np.sqrt((1 - bss**2) / (n - 2)))
        print(f"  Unc {unc_p}th percentile: Precision={prec:.3f}, Recall={rec:.3f}, BSS={bss:.3f}")
        # Store results for plotting (only first ID vs first OOD) if(i==0):
        precisions.append(np.array(case_prec))
        recalls.append(case_rec)
    # Plot ROC curves
    for ax, title in zip([ax1, ax2], ['Uncertainty-Error Correlation', 'OOD Detection']):
        ax.plot([0,1], [0,1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    # Precision-Recall plot
    print(f"Precisions {np.array(precisions).shape}")
    print(f"2 {np.array(precisions)[0].shape}")
    for i,precision in enumerate(precisions):
        ax3.plot(percentiles, precision, 'bo-', label=labels[i], linewidth=2, markersize=8)
        ax3.plot(percentiles, recalls[i], 'ro-', label=labels[i], linewidth=2, markersize=8)
    ax3.set_xlabel('Uncertainty Percentile Threshold')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision vs Recall (ID[0] vs OOD[0])')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    # Brier Skill Score bar plot
    bars = ax4.bar(np.concatenate([Labels,LabelsOOD]), brier_scores, color='skyblue')
    ax4.set_xlabel('Uncertainty Percentile Threshold')
    ax4.set_ylabel('Brier Skill Score')
    ax4.set_title('Brier Skill Score by Threshold')
    ax4.set_xticks(range(len(percentiles)))
    ax4.set_xticklabels([f'{p}th' for p in percentiles])
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, brier_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{score:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("/home/sabebert/ROC_test.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate correlations and statistics
    print("\nStatistical Comparisons:")
    comparisons = [(BarEpistemics[0], BarEpistemicsOOD[i], f"ID[0]vsOOD[{i}]") for i in [0,1]]
    for x, y, label in comparisons:
        t_stat, t_p = sc.stats.ttest_ind(x, y)
        ks_stat, ks_p = sc.stats.ks_2samp(x, y)
        ad_stat, ad_crit, ad_p = sc.stats.anderson_ksamp([x, y])
        lev_stat, lev_p = sc.stats.levene(x, y)
        print(f"{label}: T:{t_stat:.3f}({t_p:.3f}) KS:{ks_stat:.3f}({ks_p:.3f}) AD:{ad_stat:.3f}({ad_p:.3f}) L:{lev_stat:.3f}({lev_p:.3f})")
    # Calculate Spearman correlations for each dataset
    spearman_corrs, spearman_stds = [], []
    for i in range(0,len(BarEpistemics)+len(BarEpistemicsOOD)):
        if(i < len(BarEpistemics)):
            corr, p_val = sc.stats.spearmanr(BarEpistemics[i], StdGT[i])
            n = len(BarEpistemics[i])
        if(i >= len(BarEpistemicsOOD)):
            corr, p_val = sc.stats.spearmanr(BarEpistemicsOOD[i-len(BarEpistemics)], StdGTOOD[i-len(StdGT)])
            n = len(BarEpistemicsOOD[i-len(BarEpistemics)])
        spearman_corrs.append(corr)
        # Approximate standard error for correlation
        #n = len(BarEpistemics[i])
        spearman_stds.append(np.sqrt((1 - corr**2) / (n - 2)))
    print(f"\nSpearman Correlations: {spearman_corrs}")
    # Create 4 barplots + ROC curves
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
    sns.set_style("whitegrid")
# Prepare data for barplots
    # 1. Computational Cost (placeholder - replace with actual values)
    comp_costs = ComputationalCost_time[:n_datasets]  # Replace with actual computational costs
    comp_costs_stds = ComputationalCost_time_std[:n_datasets]  # Replace with actual std
    memory_costs = ComputationalCost_memory[:n_datasets]  # Replace with actual computational costs
    memory_costs_stds = ComputationalCost_memory_std[:n_datasets]  # Replace with actual std
    print(f"Shape CompCost {np.array(ComputationalCost_time).shape}")
    print(f"Shape comp_costs {np.array(comp_costs).shape}")
    #ax1.bar(labels, comp_costs, yerr=comp_cost_stds, capsize=3.0, color=dataset_colors)
    x_pos = np.arange(len(labels))
    width = 0.35
    ax1_2 = ax1.twinx()  # Create secondary y-axis
    bars1 = ax1.bar(x_pos - width/2, comp_costs, width, yerr=comp_costs_stds,
                capsize=3.0, label='Inference time [s]', color='lightblue')
    bars2 = ax1_2.bar(x_pos + width/2, memory_costs, width, yerr=memory_costs_stds,
                capsize=3.0, label='Memory usage [MB]', color='lightcoral')
    ax1.set_ylabel("Inference time [s]")
    ax1_2.set_ylabel("Memory usage [MB]")
    ax1.set_title("Computational Cost")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.legend(loc="upper left")
    ax1_2.legend(loc="upper right")
    # 2. MAE (Mean Absolute Error)
    #mae_means = [np.mean(mae) for mae in StdGT]
    mae_means = np.concatenate([[np.mean(np.array(mae)) for mae in StdGT], [np.mean(np.array(std)) for std in StdGTOOD]])
    mae_stds = np.concatenate([[StdGT_std], [StdGTOOD_std]]) if 'StdGT_std' in globals() else [0.1] * n_datasets
    #mae_stds = StdGT_std if 'StdGT_std' in globals() else [0.1] * n_datasets
    ax2.bar(labels, mae_means, yerr=mae_stds, capsize=3.0, color=colors_mae[0:n_datasets])
    ax2.set_ylabel("MAE [m/s]")
    ax2.set_title("Mean Absolute Error by Dataset")
    # 3. Spearman Correlation
    ax3.bar(labels, spearman_corrs, yerr=spearman_stds, capsize=3.0, color=colors_spear[0:n_datasets])
    ax3.set_ylabel("Spearman Correlation")
    ax3.set_title("Uncertainty-Error Correlation")
    #   4. ROC-AUC Scores (Error correlation and OOD detection)
    # Reshape AUC data for plotting
    print(f"{np.array(aucs_error).shape}")
    print(f"{np.array(aucs_ood).shape}")
    auc_error_means = aucs_error #[aucs_error[i*2:(i+1)*2] for i in range(len(aucs_error)//2)]
    auc_ood_means = aucs_ood #[aucs_ood[i*2:(i+1)*2] for i in range(len(aucs_ood)//2)]
    print(f"{np.array(auc_error_means).shape}")
    print(f"{np.array(auc_ood_means).shape}")
    if len(auc_error_means) > 0:
        auc_err_vals = [np.mean(aucs) for aucs in auc_error_means]
        auc_err_stds = [np.std(aucs) for aucs in auc_error_means]
        auc_ood_vals = [np.mean(aucs) for aucs in auc_ood_means]
        auc_ood_stds = [np.std(aucs) for aucs in auc_ood_means]
    else:
        auc_err_vals = aucs_error[:n_datasets]
        auc_err_stds = [0.05] * len(auc_err_vals)
        auc_ood_vals = aucs_ood[:n_datasets]
        auc_ood_stds = [0.05] * len(auc_ood_vals)
    print(f"{np.array(auc_err_vals).shape}")
    print(f"{np.array(auc_ood_vals).shape}")
    x_pos = np.arange(len(labels))
    width = 0.35
    bars1 = ax4.bar(x_pos - width/2, auc_err_vals, width, yerr=auc_err_stds, 
                capsize=3.0, label='Error Correlation AUC', color='lightgreen')
    bars2 = ax4.bar(x_pos + width/2, auc_ood_vals, width, yerr=auc_ood_stds,
                capsize=3.0, label='OOD Detection AUC', color='orchid')
    print(f"Shape auc_ood_vals {np.array(auc_ood_vals).shape}")
    print(f"Shape x_pos {np.array(x_pos).shape}")
    print(f"Shape width {np.array(width).shape}")
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('ROC-AUC Score')
    ax4.set_title('ROC-AUC Scores')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.legend()
    # 5. Individual ROC Curves for ID[0] vs ID[1], ID[2], OOD[1], OOD[2]
    for comp_idx, (ref_data, comp_data, comp_label) in enumerate([
        (BarEpistemics[0], BarEpistemics[1] if len(BarEpistemics) > 1 else BarEpistemics[0], "ID[0] vs ID[1]"),
        (BarEpistemics[0], BarEpistemics[2] if len(BarEpistemics) > 2 else BarEpistemics[0], "ID[0] vs ID[2]"),
        (BarEpistemics[0], BarEpistemicsOOD[0], "ID[0] vs OOD[0]"),
        (BarEpistemics[0], BarEpistemicsOOD[1] if len(BarEpistemicsOOD) > 1 else BarEpistemicsOOD[0], "ID[0] vs OOD[1]")
    ]):
        unc_combined = np.concatenate([ref_data, comp_data])
        labels_combined = np.concatenate([np.zeros(len(ref_data)), np.ones(len(comp_data))])
        if len(np.unique(labels_combined)) > 1:  # Ensure we have both classes
            auc = roc_auc_score(labels_combined, unc_combined)
            fpr, tpr, _ = roc_curve(labels_combined, unc_combined)
            ax5.plot(fpr, tpr, label=f'{comp_label} (AUC={auc:.3f})', linewidth=2)
    ax5.plot([0,1], [0,1], 'k--', alpha=0.5, label='Random')
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC Curves: Individual Comparisons')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    # 6. Summary plot
    bars = ax6.bar(labels, brier_scores,yerr=brier_scores_stds, capsize=3.0, color=colors_brier[0:n_datasets])
    ax6.set_xlabel('Datasets')
    ax6.set_ylabel('Brier Skill Score')
    ax6.set_title('Brier Skill Score')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(labels)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.legend()
    summary_text = f"""Summary Statistics
    ID Datasets ({n_datasets}):
    Mean Error AUC: {np.mean(auc_err_vals)}
    Mean OOD AUC: {np.mean(auc_ood_vals)}
    Mean Spearman: {np.mean(spearman_corrs)}"""
    #ax7.text(0.5, 0.5, summary_text,
    #     horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes,
    #     fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    #ax7.set_title('Summary Statistics')
    #ax7.axis('off')
    # 8. Empty subplot for future use
    #ax8.axis('off')
    plt.tight_layout()
    plt.savefig("/home/sabebert/ROC_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.show()
    #auc_err_str = " ".join([f"ErrorAUC{k}:{aucs_error[k]:.3f}" for k in range(len(aucs_error))])
    #auc_ood_str = " ".join([f"OODAUC{k}:{aucs_ood[k]:.3f}" for k in range(len(aucs_ood))])
    #print(f"\nSummary: {auc_err_str} | {auc_ood_str}")

    pass

def fig_maker_1(id,t_max=899,aleatoric=False,getEuclidianDistance=False,onlyX=True):
    _directory = "" #'/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready/KVS/Validation/'
    _id = id  # '28000_SW'
    #_file_name = f"processed_file_{_id}.npy" #f'clean_kvs_combined_domain_init_{_id}.csv'
    #_file_name = f"../../build/0_couette/processed_file_6_u_wall_5_0_0_top_couette_md_domain_top_0_oscil_5_0_u_wall.npy"
    _file_name = f"Data/Validation/processed_file_{_id}.npy"
    _model_directory = "" #'Results/1_Conv_AE'
    _model_name_i = "Model_AE_quantile_LR0_0001_i" #"Model_AE_aleatoric_LR0_0001_i" #"Model_AE_u_i_LR0_0001_i_Piet22" #"Model_AE_u_i_LR0_0001_i_Piet_allPerm0" # #"Model_RNN_LR1e-4_Lay1_Seq25_i #'Model_AE_u_i_LR0_0001_i' # 'Model_AE_u_i_LR0_001_i'
    
    #_dataset = torch.from_numpy(mlready2dataset(
    #    f'{_directory}{_file_name}')[:, :, :,:,:]) #1:-1, 1:-1, 1:-1])
    _dataset = mlready2dataset(f'{_directory}{_file_name}')
    _dataset = _dataset[:, :, :, :, :]
    print('Dataset shape: ', _dataset.shape)
    if(_dataset.shape == (900,1,24,24,24)):
        print(_dataset.shape)
        _dataset = np.concatenate([_dataset, _dataset, _dataset], axis=1)
        print(_dataset.shape)

    _targs = copy.deepcopy(_dataset[1:, :, :, :, :])
    _targs2 = copy.deepcopy(_dataset[:, :, :, :, :])
    _input_a = torch.from_numpy(copy.deepcopy(_dataset[:-1, :, :, :, :]))
    _input_b = torch.from_numpy(copy.deepcopy(_dataset[:-1, :, :, :, :]))

    _dataset = torch.from_numpy(_dataset).to(device)
    #_targs = copy.deepcopy(_dataset)
    _dataset = torch.add(_dataset, 1.0).float().to(device)
    
    """
    _model_u_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    """

    _model_u_i =  AE_aleatoric(
        device=device,
        in_channels=1,
        out_channels=2,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_u_i.load_state_dict(torch.load(
        f'{_model_directory}{_model_name_i}', map_location='cpu'))
    _model_u_i.eval()
    
    if(aleatoric==True):
        _preds_x = _model_u_i(_dataset[:, 0:1, :, :, :])
       # _preds_y = _model_u_i(_dataset[:, 1:2, :, :, :])
       # _preds_z = _model_u_i(_dataset[:, 2:3, :, :, :])

        _pred = torch.add(_preds_x, -1.0).float().to(device)
        mean_pred = _pred[:, 0, :, :, :].cpu().detach().numpy()
        log_var = _pred[:, 1, :, :, :].cpu().detach().numpy()
        uncertainty = np.sqrt(np.exp(log_var))

    else:
        if(onlyX==True):
            _preds_x = _model_u_i(_dataset[:, 0, :, :, :])
            _preds_y = _model_u_i(_dataset[:, 1, :, :, :])
            _preds_z = _model_u_i(_dataset[:, 2, :, :, :])
        else:
            _preds_x = _model_u_i(_dataset[:, 0, :, :, :])
            _preds_y = _model_u_i(_dataset[:, 1, :, :, :])
            _preds__ = _model_u_i(_dataset[:, 2, :, :, :])

    _preds = torch.cat((_preds_x, _preds_x, _preds_x), 1).to(device)
    #print(_preds_y)
    _preds = torch.add(_preds, -1.0).float().to(device)
    _preds = _preds.cpu().detach().numpy()
        # _targs = _targs.numpy()



    if(getEuclidianDistance==True):
        # Calculate Euclidean distance for each channel separately
        euclidean_distances = []
        diff_x = _preds_x.cpu().detach().numpy()[:,0,:,:,:] - _targs2[:,0,:,:,:]
        print(diff_x.shape) #(900, 900, 24, 24, 24)
        print(_preds_x.shape) #torch.Size([900, 1, 24, 24, 24]
        print(diff_x[0,0,0,0])
        #diff_y = _preds_y.cpu().detach().numpy() - _targs[:,1,:,:,:]
        #diff_z = _preds_z.cpu().detach().numpy() - _targs[:,2,:,:,:]
        euclidean_dist_x = np.sqrt(np.sum(diff_x**2, axis=(1, 2, 3)))
        print(euclidean_dist_x.shape)
        #euclidean_dist_y = np.sqrt(np.sum(diff_y**2, axis=(1, 2, 3)))
        #euclidean_dist_z = np.sqrt(np.sum(diff_z**2, axis=(1, 2, 3)))
        euclidean_distances.append(euclidean_dist_x)
        #euclidean_distances.append(euclidean_dist_x,euclidean_dist_y,euclidean_dist_z)
        euclidean_distances = np.array(euclidean_distances)  # shape (3, 1000)

        #np.save('euclidean_distances_ch1.npy', euclidean_distances[0])
        #np.save('euclidean_distances_ch2.npy', euclidean_distances[1])
        #np.save('euclidean_distances_ch3.npy', euclidean_distances[2])
        np.save(f'euclidean_distances_all_{_id}.npy', euclidean_distances)
        print(f"Euclidean distances for file {_id}.")
        print(f"Shape of distances per channel: {euclidean_distances[0].shape}")
        print(f"Mean distance - Channel 1: {np.mean(euclidean_distances[0]):.4f}")
        #print(f"Mean distance - Channel 2: {np.mean(euclidean_distances[1]):.4f}")
        #print(f"Mean distance - Channel 3: {np.mean(euclidean_distances[2]):.4f}")

    if(aleatoric==True):
        plot_flow_profile(
            np_datasets=[_targs, _preds],
            dataset_legends=['MD', 'Autoencoder'],
            save2file=f'{_id}_Fig_Maker_5_a_ConvAE_vs_MD_22_new',
            t_max=t_max
        )
        time_steps = np.arange(0,t_max,1)
        #mean_pred = mean_pred - np.ones_like(mean_pred)
        plt.plot(_targs[0:t_max,0,12,12,12],label="Target",color="orange")
        plt.plot(mean_pred[0:t_max,12,12,12],label="Prediction",color="darkblue")
        #plt.plot(mean_pred[0:t_max,12,12,12] + (uncertainty[0:t_max,12,12,12]),label="Aleatoric",color="green")
        #plt.plot(mean_pred[0:t_max,12,12,12] - (uncertainty[0:t_max,12,12,12]),color="green")
        plt.fill_between(time_steps,mean_pred[0:899,12,12,12] - uncertainty[0:899,12,12,12], mean_pred[0:899,12,12,12] + uncertainty[0:899,12,12,12], alpha=0.6, label="Aleatoric", color="green", linestyle='dotted')
        plt.legend(loc="upper right")
        plt.xlabel("Coupling Cycles")
        plt.ylabel("ux [m/s]")
        plt.savefig(f"/home/sabebert/{outname}.png")

    pass


if __name__ == "__main__":
    '''
    _model_directory = 'Results/1_Conv_AE'
    _model_name_i = 'Model_AE_u_i_LR0_001_i'
    _dataset_name = 'get_KVS_eval'
    _save2file_name = 'latentspace_validation_20k_NE'

    prediction_retriever_latentspace_u_i(
        model_directory=_model_directory,
        model_name_i=_model_name_i,
        dataset_name=_dataset_name,
        save2file_name=_save2file_name)

    _ids = ['20000_NE', '22000_NW', '26000_SE', '28000_SW']
    for _id in _ids:
        fig_maker_1(id=_id)
    '''
    trial_1_AE_u_i_mp()
    
    #get_latentspace_AE_u_i_helper()
   
    """
    augments=[]
    if(Rotation==True):
        augments.append("Rotation")
    if(Shift==True):
        augments.append("Shift")
    if(CellShuffle==True):
        augments.append("CellShuffle")
    if(Random==True):
        augments.append("Random")
    if(TrueRandom==True):
        augments.append("TrueRandom")
    if(OOD==True):
        augments.append("OOD")
    if(Couette==True):
        augments.append("Couette")

        
    if(ensemble==True and OOD==True and Couette==True): #and Testing==True):
        fig_maker_2(_id=_id,_ids=_ids,outname="Test",file_path=f"{validPath}",file_path_ood=file_path_ood,id_ood=id_ood,_ids_ood=_ids_ood,file_path_couette=file_path_couette,id_couette = id_couette,_ids_couette=_ids_couette,augments=augments,epistemic_models=ensemble_models,t_max=899,aleatoric=aleatoric,getEuclidianDistance=True) #145922
    elif(ensemble==True and OOD==True): #and Testing==True):
        fig_maker_2(_id=_id,_ids=_ids,outname="Test",file_path=f"{validPath}",file_path_ood=file_path_ood,id_ood=id_ood,_ids_ood=_ids_ood,augments=augments,epistemic_models=ensemble_models,t_max=899,aleatoric=aleatoric,getEuclidianDistance=True)
    elif(ensemble==True and Testing==True):
        fig_maker_2(_id=_id,_ids=_ids,outname="Test",file_path=f"{validPath}",file_path_ood=file_path_ood,id_ood=id_ood,_ids_ood=_ids_ood,augments=augments,epistemic_models=ensemble_models,t_max=899,aleatoric=aleatoric,getEuclidianDistance=True)
    elif(ensemble==True): # and Testing==True):
        fig_maker_1(_id=_id,_ids=_ids,outname="Test",file_path=f"{validPath}",t_max=899,aleatoric=aleatoric,getEuclidianDistance=True) #145922

    elif(ensemble==False): # and Testing==True):
        fig_maker_1(_id=_id,_ids=_ids,outname="Test",file_path=f"{validPath}",t_max=899,aleatoric=aleatoric,getEuclidianDistance=True) #145922
    
    """
    """"    
        if(ID):
            fig_maker_2(id=145922,outname="Test",file_path=f"{validPath}",epistemic_models=ensemble_models,t_max=899,aleatoric=aleatoric,getEuclidianDistance=True) #145922
        elif(OOD):
            fig_maker_2(id=145922,outname="Test",file_path=f"{oodPath}",epistemic_models=ensemble_models,t_max=899,aleatoric=aleatoric,getEuclidianDistance=True) #145922
        elif(CellShuffle):
            fig_maker_2(id=145922,outname="Test",file_path=f"{validPath}",augment="CellShuffle",epistemic_models=ensemble_models,t_max=899,aleatoric=aleatoric,getEuclidianDistance=True) #145922
        elif(Rotation):
            fig_maker_2(id=145922,outname="Test",file_path=f"{validPath}",augment="Rotation",epistemic_models=ensemble_models,t_max=899,aleatoric=aleatoric,getEuclidianDistance=True) #145922
        elif(Couette):
            fig_maker_2(id=145922,outname="Test",file_path=f"{couettePath}",epistemic_models=ensemble_models,t_max=899,aleatoric=aleatoric,getEuclidianDistance=True) #145922
        elif(Random):
            fig_maker_2(id=145922,outname="Test",file_path=f"{validPath}",augment="Random",epistemic_models=ensemble_models,t_max=899,aleatoric=aleatoric,getEuclidianDistance=True) #145922
    """
