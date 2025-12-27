#!/usr/bin/env python3
"""
Laplace approximation inference for epistemic uncertainty quantification.
Strongly based on the implementation of Daxberger et al. (2021).
(arXiv:2106.14806. https://github.com/aleximmer/Laplace)
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
from torch.utils.data import TensorDataset, DataLoader
from model_aleatoric import AE_aleatoric
from model_laplace import SparseLaplaceApproximation_AE
from common_utils import save_method_results, calculate_std_across_files

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = os.environ.get('OUTNAME', "Laplace_Final_v1")
MODEL_DIR = os.environ.get('MODEL_DIR', "")

MODEL_NAME = os.environ.get('MODEL_NAME', "aleatoric/model_ae_nll_lr0.001_seed0.pth")

SPARSITY_FRACTION = 0.001
MAX_PARAMS_TO_COMPUTE = 30
HESSIAN_DATA_SIZE = 15
PRIOR_PRECISION = 1e-4
USE_LAST_LAYER_ONLY = True


FILE_IDS = ["145922", "121428", "118626", "143128", "118620"]
FILE_NAMES = [f"Data/Validation/processed_file_{_id}.npy" for _id in FILE_IDS]

#If only some datasets should be processed, besides Original/Referenz
PROCESS_TRUERANDOM = True
PROCESS_OOD = True
PROCESS_COUETTE = True

def compute_uncertainty(laplace, dataset_tensor, t_max, dataset_name=""):
    t_max_actual = min(t_max, dataset_tensor.shape[0])
    data_on_device = dataset_tensor[:t_max_actual, 0:1].to(DEVICE)
    
    means_list, uncertainties_list = [], []
    batch_size = 128
    
    for i in range(0, t_max_actual, batch_size):
        end_idx = min(i + batch_size, t_max_actual)
        batch_data = data_on_device[i:end_idx]
        
        mean_pred, epistemic_std = laplace.predict_with_uncertainty(batch_data)
        mean_pred_np = mean_pred.detach().cpu().numpy()
        epistemic_std_np = epistemic_std.detach().cpu().numpy()
        
        if len(mean_pred_np.shape) == 5:
            mean_avg = np.mean(mean_pred_np[:, 0, :, :, :], axis=(1, 2, 3))
        elif len(mean_pred_np.shape) == 4:
            mean_avg = np.mean(mean_pred_np, axis=(1, 2, 3))
        else:
            mean_avg = np.mean(mean_pred_np, axis=tuple(range(1, len(mean_pred_np.shape))))
        
        if len(epistemic_std_np.shape) == 2:
            unc_avg = np.mean(epistemic_std_np, axis=1)
        elif len(epistemic_std_np.shape) == 1:
            unc_avg = epistemic_std_np
        else:
            unc_avg = np.mean(epistemic_std_np, axis=tuple(range(1, len(epistemic_std_np.shape))))
        
        means_list.extend(mean_avg.flatten())
        uncertainties_list.extend(unc_avg.flatten())
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Progress: {end_idx}/{t_max_actual} timesteps")
    
    means = np.array(means_list)
    uncertainties = np.array(uncertainties_list)
    spatial_shape = (len(means), 24, 24, 24)
    return (np.broadcast_to(means[:, np.newaxis, np.newaxis, np.newaxis], spatial_shape),
            np.broadcast_to(uncertainties[:, np.newaxis, np.newaxis, np.newaxis], spatial_shape))

def setup_laplace(model, train_data_tensor, train_target_tensor):
    train_subset_size = min(HESSIAN_DATA_SIZE, train_data_tensor.shape[0])
    train_dataset = TensorDataset(train_data_tensor[:train_subset_size], train_target_tensor[:train_subset_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    #from laplace import Laplace
    # Pre-trained model
    #model = load_map_model()
    # User-specified LA flavor
    #la = Laplace(model, "classification",
    #            subset_of_weights="all",
    #            hessian_structure="diag")
    #la.fit(train_loader)
    #la.optimize_prior_precision(
    #    method="gridsearch",
    #    pred_type="glm",
    #     link_approx="probit",
    #    val_loader=val_loader
    #)

    laplace = SparseLaplaceApproximation_AE(model, DEVICE)
    laplace.prior_precision = PRIOR_PRECISION
    
    def _estimate_diagonal_hessian_with_reduced_params(train_loader, criterion, num_samples):
        original_params = laplace._get_parameters()
        n_params = len(original_params)
        
        if USE_LAST_LAYER_ONLY:
            param_list = list(model.parameters())
            if len(param_list) > 0:
                last_layer_params = param_list[-1]
                last_layer_size = last_layer_params.numel()
                n_params_to_include = max(1, min(int(last_layer_size * SPARSITY_FRACTION), MAX_PARAMS_TO_COMPUTE))
                
                with torch.no_grad():
                    last_layer_flat = last_layer_params.view(-1)
                    _, important_indices = torch.topk(torch.abs(last_layer_flat), n_params_to_include, largest=True)
                    important_indices = important_indices.cpu().numpy()
                param_mask = torch.zeros(n_params, dtype=torch.bool, device=DEVICE)
                param_mask[(n_params - last_layer_size) + important_indices] = True
            else:
                param_mask = torch.ones(n_params, dtype=torch.bool, device=DEVICE)
        else:
            param_mask = torch.ones(n_params, dtype=torch.bool, device=DEVICE)
        
        hessian_diag = torch.zeros(n_params, device=DEVICE)
        data_samples, target_samples = [], []
        sample_count = 0
        for batch_data, batch_targets in train_loader:
            if sample_count >= HESSIAN_DATA_SIZE:
                break
            data_samples.append(batch_data.to(DEVICE))
            target_samples.append(batch_targets.to(DEVICE))
            sample_count += len(batch_data)
        
        data = torch.cat(data_samples, dim=0)[:HESSIAN_DATA_SIZE]
        targets = torch.cat(target_samples, dim=0)[:HESSIAN_DATA_SIZE]
        laplace._set_parameters(original_params)
        
        for param in model.parameters():
            param.requires_grad_(True)
        model.train()
        outputs = model(data)
        loss = criterion(outputs, targets)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
        grads_flat = torch.cat([g.reshape(-1) for g in grads])
        
        param_indices_to_compute = torch.where(param_mask)[0].cpu().numpy()
        for i in param_indices_to_compute:
            i = int(i)
            grad_i = grads_flat[i]
            hessian_row_i = torch.autograd.grad(grad_i, model.parameters(), retain_graph=True, create_graph=False, only_inputs=True)
            hessian_row_i_flat = torch.cat([h.reshape(-1) for h in hessian_row_i])
            hessian_diag[i] = hessian_row_i_flat[i]
        
        hessian_diag[param_mask] += laplace.prior_precision
        hessian_diag[~param_mask] = 1e6
        
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        
        return hessian_diag
    
    laplace._estimate_diagonal_hessian = _estimate_diagonal_hessian_with_reduced_params
    criterion = nn.MSELoss() #L1Loss()
    laplace.fit(train_loader, criterion, num_samples=HESSIAN_DATA_SIZE)
    return laplace

def main():
    #load datasets
    datasets = {
        'original': torch.load("dataset/original_2_dataset_tensor.pt", map_location=DEVICE),
        'truerandom': torch.load("dataset/truerandom_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_TRUERANDOM else None,
        'ood': torch.load("dataset/ood_dataset_tensor.pt", map_location=DEVICE) if PROCESS_OOD else None,
        'couette': torch.load("dataset/couette_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_COUETTE else None,
    }
    
    model = AE_aleatoric(device=DEVICE, in_channels=1, out_channels=2, features=[4, 8, 16], activation=nn.ReLU(inplace=True)).to(DEVICE)
    model.load_state_dict(torch.load(f'{MODEL_DIR}{MODEL_NAME}', map_location='cpu'))
    model.eval()
    
    train_data_tensor = datasets['original'][:, 0:1, :, :, :]
    train_target_tensor = datasets['original'][:, 0:1, :, :, :]
    laplace = setup_laplace(model, train_data_tensor, train_target_tensor)
    
    mean_preds_list, epistemic_list = [], []
    times, memories = [], []
    index = 0
    for file_idx, file in enumerate(FILE_NAMES):

        #Track inference
        start_time = time.time()
        tracemalloc.start()
        means_file, uncertainties_file = compute_uncertainty(laplace, datasets['original'][index:index+T_MAX], T_MAX, f"Original file {file_idx+1}/{len(FILE_NAMES)}")
        mean_preds_list.append(means_file)
        epistemic_list.append(uncertainties_file)
        
        times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memories.append(peak / 1024 / 1024)
        index += T_MAX
    mean_preds = np.array(mean_preds_list).reshape((len(FILE_NAMES)*mean_preds_list[0].shape[0],) + mean_preds_list[0].shape[1:])
    epistemic_orig = np.concatenate(epistemic_list, axis=0)
    
    results = {"Original": mean_preds}
    epistemic = {"Original": epistemic_orig}
    aleatoric = {"Original": None, "TrueRandom": None, "OOD": None, "Couette": None}
    
    #In case not all datasets are used
    if PROCESS_TRUERANDOM:
        print("Processing TrueRandom dataset...")
        means_tr, unc_tr = compute_uncertainty(laplace, datasets['truerandom'], datasets['truerandom'].shape[0], "TrueRandom")
        results["TrueRandom"], epistemic["TrueRandom"] = means_tr, unc_tr
    if PROCESS_OOD:
        print("Processing OOD dataset...")
        means_ood, unc_ood = compute_uncertainty(laplace, datasets['ood'], datasets['ood'].shape[0], "OOD")
        results["OOD"], epistemic["OOD"] = means_ood, unc_ood
    if PROCESS_COUETTE:
        print("Processing Couette dataset...")
        means_cou, unc_cou = compute_uncertainty(laplace, datasets['couette'], datasets['couette'].shape[0], "Couette")
        results["Couette"], epistemic["Couette"] = means_cou, unc_cou
 
    n_files = {"Original": 5, "TrueRandom": 5, "OOD": 6, "Couette": 5}
    aleatoric_std = {dataset: 0.0 for dataset in ["Original", "TrueRandom", "OOD", "Couette"]}
    
    save_method_results(
        method_name="Laplace2_3", outname=OUTNAME, results_dict=results, epistemic_dict=epistemic, aleatoric_dict=aleatoric,
        time_array=np.array(times), memory_array=np.array(memories),
        epistemic_std_dict=calculate_std_across_files(epistemic, n_files, T_MAX),
        aleatoric_std_dict=aleatoric_std,
        results_std_dict=calculate_std_across_files(results, n_files, T_MAX)
    )

main()
