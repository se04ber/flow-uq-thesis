#!/usr/bin/env python3
"""Gaussian Process regression for epistemic uncertainty estimation in latent space.

Note: This implementation uses spatially uniform epistemic uncertainty (scalar per timestep,
broadcast to full grid) as a computational simplification due to time constraints. A more
sophisticated approach would model spatial variation in epistemic uncertainty.
Also subsets of the data are used to reduce instead of diagonal approximation, because of possible
latent space feature correlations.

Reference:
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
    MIT Press. Available at http://www.gaussianprocess.org/gpml/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import time
import tracemalloc
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from utils import get_AE_loaders
from model_aleatoric import AE_aleatoric
from common_utils import save_method_results, calculate_std_across_files

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = os.environ.get('OUTNAME', "GP_LS10_Final_v1_2")
MODEL_DIR = os.environ.get('MODEL_DIR', "")
MODEL_NAME = os.environ.get('MODEL_NAME', "aleatoric/model_ae_nll_lr0.001_seed0.pth")

FILE_IDS = ["145922", "121428", "118626", "143128", "118620"]
FILE_NAMES = [f"Data/Validation/processed_file_{_id}.npy" for _id in FILE_IDS]

#If only some datasets should be processed, besides Original/Referenz
PROCESS_TRUERANDOM = True
PROCESS_OOD = True
PROCESS_COUETTE = True

#Copy here for continuity
def extract_latents(model, tensor_5d):
    """Extract latent bottleneck features from autoencoder for GP training/prediction."""
    latents = []
    with torch.no_grad():
        for i in range(tensor_5d.shape[0]):
            bottleneck, _ = model(tensor_5d[i:i+1, 0:1], y='get_bottleneck')
            latents.append(bottleneck.view(1, -1).cpu().numpy())
    return np.vstack(latents)

def broadcast_to_grid(series_1d, ref_grid):
    return np.stack([np.ones_like(ref_grid[i]) * series_1d[i] for i in range(len(series_1d))], axis=0)

def fit_gp(model, max_samples=4000, n_inducing=1500):
    """Train GP on latent space using center point (12,12,12) as target.
    
    Sparsity approach: Random subset selection (1500 points) is used instead of true sparse GP
    methods (e.g., inducing points/FITC) or diagonal approximations. This provides a reasonable
    balance between computational efficiency and capturing correlations in latent space via the
    full RBF kernel covariance structure, without requiring additional sparse GP libraries.
    """
    train_loaders, _ = get_AE_loaders(path="", data_distribution='get_KVS', batch_size=8, shuffle=True)
    
    train_lat_list, train_targ_list = [], []
    collected = 0
    
    with torch.no_grad():
        for data, target in train_loaders[0]:
            if collected >= max_samples:
                break
            data = torch.add(data[:, 0:1], 1.0).float().to(DEVICE)
            bottleneck, _ = model(data, y='get_bottleneck')
            train_lat_list.append(bottleneck.view(bottleneck.shape[0], -1).cpu().numpy())
            # Use center point (12,12,12) as representative target value
            train_targ_list.append(target[:, 0, 12, 12, 12].cpu().numpy())
            collected += bottleneck.shape[0]
    
    train_lat = np.vstack(train_lat_list) if train_lat_list else np.zeros((100, 32))
    train_targ = np.concatenate(train_targ_list) if train_targ_list else np.zeros((100,))
    # Use subset of data for computational efficiency
    n_inducing = min(n_inducing, train_lat.shape[0])
    idx = np.random.choice(train_lat.shape[0], size=n_inducing, replace=False)
    # Fixed RBF kernel with length scale 10 (found through hyperparameter analysis)
    kernel = C(1.0, constant_value_bounds='fixed') * RBF(length_scale=10.0, length_scale_bounds='fixed')
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-2, normalize_y=True)
    gp.fit(train_lat[idx], train_targ[idx])
    return gp
    
    # Alternative: Diagonal GP approximation (assumes independence between latent dimensions)
    # This would be O(n) instead of O(n³), but ignores correlations in latent space
    # from sklearn.gaussian_process.kernels import WhiteKernel
    # kernel_diag = WhiteKernel(noise_level=1.0, noise_level_bounds='fixed')
    # gp_diag = GaussianProcessRegressor(kernel=kernel_diag, alpha=1e-2, normalize_y=True)
    # gp_diag.fit(train_lat, train_targ)  # Can use full dataset due to diagonal structure
    # return gp_diag

def process_dataset(model, gp, dataset, t_max, file_names):
    """Process validation dataset: extract predictions and both aleatoric + epistemic uncertainty."""
    mean_preds_list, aleatoric_list, epistemic_list = [], [], []
    times, memories = [], []
    index = 0
    for file in file_names:

        #Track inference
        start_time = time.time()
        tracemalloc.start()
        # Extract latent features and predict epistemic uncertainty via GP
        lat_valid = extract_latents(model, dataset[index:index+t_max, 0:1, :, :, :])
        _, y_std = gp.predict(lat_valid, return_std=True)
        with torch.no_grad():
            pred_ae = model(dataset[index:index+t_max, 0:1, :, :, :])
            pred_ae = torch.add(pred_ae, -1.0)
            mean_ae = pred_ae[:, 0].cpu().numpy()
            # Extract aleatoric uncertainty: sqrt(exp(log_var))
            ale_std = np.sqrt(np.exp(pred_ae[:, 1].cpu().numpy()))
        
        mean_preds_list.append(mean_ae)
        aleatoric_list.append(ale_std)
        # Broadcast scalar epistemic uncertainty to full grid
        epistemic_list.append(broadcast_to_grid(y_std, mean_ae))
        times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memories.append(peak / 1024 / 1024)
        index += t_max

    mean_preds = np.array(mean_preds_list).reshape((len(file_names)*mean_preds_list[0].shape[0],) + mean_preds_list[0].shape[1:])
    return mean_preds, np.concatenate(epistemic_list, axis=0), np.concatenate(aleatoric_list, axis=0), times, memories

def process_augmented(model, gp, dataset, t_max, n_files=None):
    """Process augmented datasets (TrueRandom, OOD, Couette) with optional chunking."""
    dataset_len = dataset.shape[0]
    # If n_files is provided and dataset has enough data, process in chunks
    if n_files is not None and dataset_len >= n_files * t_max:
        mean_preds_list, epistemic_list = [], []
        times, memories = [], []
        for file_idx in range(n_files):
            start_time = time.time()
            tracemalloc.start()
            start_idx = file_idx * t_max
            end_idx = min((file_idx + 1) * t_max, dataset_len)
            # Process chunk: extract mean predictions and epistemic uncertainty
            lat = extract_latents(model, dataset[start_idx:end_idx, 0:1, :, :, :])
            _, y_std = gp.predict(lat, return_std=True)
            with torch.no_grad():
                pred_ae = model(dataset[start_idx:end_idx, 0:1, :, :, :])
                pred_ae = torch.add(pred_ae, -1.0)
                mean_ae = pred_ae[:, 0].cpu().numpy()

            mean_preds_list.append(mean_ae)
            epistemic_list.append(broadcast_to_grid(y_std, mean_ae))
            times.append(time.time() - start_time)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memories.append(peak / 1024 / 1024)    
        # Concatenate all files
        mean_preds = np.concatenate(mean_preds_list, axis=0)
        epistemic_combined = np.concatenate(epistemic_list, axis=0)
        return mean_preds, epistemic_combined, times, memories
    else:
        # Fallback: process entire dataset at once (original behavior)
        lat = extract_latents(model, dataset[:, 0:1, :, :, :])
        _, y_std = gp.predict(lat, return_std=True)
        with torch.no_grad():
            mean = torch.add(model(dataset[:, 0:1]), -1.0)[:, 0].cpu().numpy()
        epistemic = broadcast_to_grid(y_std, mean)
        return mean, epistemic, None, None

def main():
    #load datasets
    datasets = {
        'original': torch.load("dataset/original_2_dataset_tensor.pt", map_location=DEVICE),
        'truerandom': torch.load("dataset/truerandom_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_TRUERANDOM else None,
        'ood': torch.load("dataset/ood_dataset_tensor.pt", map_location=DEVICE) if PROCESS_OOD else None,
        'couette': torch.load("dataset/couette_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_COUETTE else None,
    }
    
    # Load pretrained aleatoric model (outputs mean + log_var)
    # Load pretrained aleatoric model (outputs mean + log_var)
    model = AE_aleatoric(device=DEVICE, in_channels=1, out_channels=2, features=[4, 8, 16], activation=nn.ReLU(inplace=True)).to(DEVICE)
    model.load_state_dict(torch.load(f'{MODEL_DIR}{MODEL_NAME}', map_location='cpu'))
    model.eval()
    # Train GP on latent space for epistemic uncertainty
    gp = fit_gp(model)
    mean_preds, epistemic_orig, aleatoric_orig, times, memories = process_dataset(model, gp, datasets['original'], T_MAX, FILE_NAMES)
    
    results = {"Original": mean_preds}
    epistemic = {"Original": epistemic_orig}
    #For the epistemic comparision
    aleatoric = {"Original": aleatoric_orig}
    n_files = {"Original": 5, "TrueRandom": 5, "OOD": 6, "Couette": 5}
    #In case not all datasets are used
    if PROCESS_TRUERANDOM:
        mean_tr, epi_tr, times_tr, memories_tr = process_augmented(model, gp, datasets['truerandom'], T_MAX, n_files["TrueRandom"])
        results["TrueRandom"], epistemic["TrueRandom"], aleatoric["TrueRandom"] = mean_tr, epi_tr, None
        if times_tr is not None:
            times.extend(times_tr)
            memories.extend(memories_tr)
    if PROCESS_OOD:
        mean_ood, epi_ood, times_ood, memories_ood = process_augmented(model, gp, datasets['ood'], T_MAX, n_files["OOD"])
        results["OOD"], epistemic["OOD"], aleatoric["OOD"] = mean_ood, epi_ood, None
        if times_ood is not None:
            times.extend(times_ood)
            memories.extend(memories_ood)
    if PROCESS_COUETTE:
        mean_cou, epi_cou, times_cou, memories_cou = process_augmented(model, gp, datasets['couette'], T_MAX, n_files["Couette"])
        results["Couette"], epistemic["Couette"], aleatoric["Couette"] = mean_cou, epi_cou, None
        if times_cou is not None:
            times.extend(times_cou)
            memories.extend(memories_cou)
 
    save_method_results(
        method_name="GP_LS10", outname=OUTNAME, results_dict=results, epistemic_dict=epistemic, aleatoric_dict=aleatoric,
        time_array=np.array(times), memory_array=np.array(memories),
        epistemic_std_dict=calculate_std_across_files(epistemic, n_files, T_MAX),
        aleatoric_std_dict=calculate_std_across_files(aleatoric, n_files, T_MAX),
        results_std_dict=calculate_std_across_files(results, n_files, T_MAX)
    )

main()
