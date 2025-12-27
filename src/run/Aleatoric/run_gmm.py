#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import time
import tracemalloc
import random
from datetime import datetime
from model_gmm import AE_gmm
from methods.common_utils import save_method_results, load_dataset
from utils import mlready2dataset

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

MODEL_PATH = "aleatoric/model_ae_gmm_lr0.001_seed0.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = f"Test_GMM_2"
_model_directory = ""

_ids = ["145922", "121428", "118626", "143128","118620"]
_file_names = [f"Data/Validation/processed_file_{_id}.npy" for _id in _ids]

def load_gmm_model(model_path, device):
    model = AE_gmm(
        device=device,
        in_channels=1,
        out_channels=6,
        features=[4, 8, 16]
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded GMM model: {model_path}")
    
    model.eval()
    return model

def extract_gmm_uncertainty(model, data, device):
    with torch.no_grad():
        data = data.float().to(device)
        pred = model(data)
        # Denormalize: subtract 1.0 that was added before prediction (only for mean predictions, not log_var)
        pred_denorm = torch.add(pred, -1.0)
        mu1 = pred_denorm[:, 0:1]
        log_var1 = pred[:, 1:2]  # Log variance doesn't need denormalization
        mu2 = pred_denorm[:, 2:3]
        log_var2 = pred[:, 3:4]  # Log variance doesn't need denormalization
        w1 = torch.softmax(pred[:, 4:6], dim=1)[:, 0:1]  # Softmax weights
        w2 = 1.0 - w1
        
        # Compute mixture mean and variance
        mean_pred = w1 * mu1 + w2 * mu2
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        mixture_var = w1 * (var1 + (mu1 - mean_pred)**2) + w2 * (var2 + (mu2 - mean_pred)**2)
        aleatoric_uncertainty = torch.sqrt(torch.clamp(mixture_var, min=1e-10))
        
        # Additional denormalization: subtract 1.0 to fix MAE being +1 too large
        mean_pred = torch.add(mean_pred, 1.0)
        
        return mean_pred.cpu().numpy(), aleatoric_uncertainty.cpu().numpy()

def run_gmm_inference(model_path, file_names, t_max=899, device='cuda'):
    """
    Run GMM inference on multiple files and return standardized results.
    
    Returns:
        results_dict: {dataset_name: predictions_array}
        aleatoric_dict: {dataset_name: aleatoric_uncertainty_array}
        time_array: numpy array of inference times per dataset
        memory_array: numpy array of memory usage per dataset
    """
    print("GMM Aleatoric Uncertainty Inference")
    print(f"Loading GMM model: {model_path}")
    model = load_gmm_model(model_path, device)
    
    results_dict = {}
    aleatoric_dict = {}
    time_list = []
    memory_list = []
    all_predictions = []
    all_aleatoric = []
    all_times = []
    all_memories = []
    for file_idx, file_name in enumerate(file_names):
        print(f"  Processing file {file_idx+1}/{len(file_names)}: {file_name}")
        dataset = mlready2dataset(file_name)
        dataset = dataset[:, :, :, :, :]
        dataset = torch.from_numpy(dataset.copy()).to(device)
        dataset = torch.add(dataset, 1.0).float().to(device)
        
        #Track inference 
        start_time = time.time()
        tracemalloc.start()
        pred, aleatoric = extract_gmm_uncertainty(model, dataset, device)
        inference_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak / 1024 / 1024  # MB
        
        all_predictions.append(pred)
        all_aleatoric.append(aleatoric)
        all_times.append(inference_time)
        all_memories.append(memory_usage)    
        print(f"    Time: {inference_time:.3f}s, Memory: {memory_usage:.1f}MB")
    
    # Concatenate all files
    if len(all_predictions) > 0:
        results_dict['Original'] = np.concatenate(all_predictions, axis=0)
        aleatoric_dict['Original'] = np.concatenate(all_aleatoric, axis=0)
        time_list.append(np.mean(all_times))  # Mean time per file
        memory_list.append(np.mean(all_memories))  # Mean memory per file
        print(f" Original: shape {results_dict['Original'].shape}")
    # Epistemic is None for aleatoric-only methods
    epistemic_dict = {'Original': None}
    return results_dict, epistemic_dict, aleatoric_dict, np.array(time_list), np.array(memory_list)

def main():
    results_dict, epistemic_dict, aleatoric_dict, time_array, memory_array = run_gmm_inference(
        MODEL_PATH, _file_names, t_max=T_MAX, device=DEVICE
    )
    print(f"Saving results")
    save_method_results(
        method_name="GMM",
        results_dict=results_dict,
        epistemic_dict=epistemic_dict,
        aleatoric_dict=aleatoric_dict,
        time_array=time_array,
        memory_array=memory_array,
        outname=OUTNAME
    )
    print(f"GMM inference complete!")
    print(f"Results saved with outname: {OUTNAME}")

main()

