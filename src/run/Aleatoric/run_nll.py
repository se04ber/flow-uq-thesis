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
from model_aleatoric import AE_aleatoric
from methods.common_utils import save_method_results, load_dataset  
from utils import mlready2dataset
    
# Set random seeds
torch.manual_seed(10)
random.seed(10)
np.random.seed(10)
    
MODEL_PATH = "aleatoric/model_ae_nll_lr0.001_seed0.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899 #Legacy mostly used for plotting

OUTNAME = f"Test_NLL_2" #f"Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # OUTNAME includes timestamp to prevent overwriting old results
_model_directory = ""

#Inference files from Validation set
_ids = ["145922", "121428", "118626", "143128","118620"]
_file_names = [f"Data/Validation/processed_file_{_id}.npy" for _id in _ids]

def load_nll_model(model_path, device):
    """Load NLL model"""
    architectures_to_try = [
        {"in_channels": 1, "out_channels": 2, "features": [4, 8, 16]}
    ]
    
    model = None
    #for arch in architectures_to_try:
    model = AE_aleatoric(
        device=device,
        in_channels=arch["in_channels"],
        out_channels=arch["out_channels"],
        features=arch["features"]
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f" Loaded NLL model: {model_path} with arch {arch}")
    if model is None:
        raise RuntimeError(f"Could not load NLL model: {model_path}")
    model.eval()
    return model

def extract_nll_uncertainty(model, data, device):
    """Extract predictions and aleatoric uncertainty from NLL model"""
    with torch.no_grad():
        data = data.float().to(device)
        pred = model(data)
        # Denormalize: subtract 1.0 that was added before prediction
        pred_denorm = torch.add(pred, -1.0)
        mean_pred = pred_denorm[:, 0:1]  # Mean prediction
        log_var = pred[:, 1:2]    # Log variance doesn't need denormalization
        # Convert to std: sqrt(exp(log_var)) = sqrt(σ²) = σ
        aleatoric_uncertainty = torch.sqrt(torch.exp(log_var))  # Convert to std
        
        return mean_pred.cpu().numpy(), aleatoric_uncertainty.cpu().numpy()

def run_nll_inference(model_path, file_names, t_max=899, device='cuda'):
    """
    Run NLL inference on multiple files and return standardized results.
    
    Returns:
        results_dict: {dataset_name: predictions_array}
        aleatoric_dict: {dataset_name: aleatoric_uncertainty_array}
        time_array: numpy array of inference times per dataset
        memory_array: numpy array of memory usage per dataset
    """

    print("NLL Aleatoric Uncertainty Inference") 
    print(f"Loading NLL model: {model_path}")
    model = load_nll_model(model_path, device)
 
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
        # Load and process dataset file
        dataset = mlready2dataset(file_name)
        dataset = dataset[:, :, :, :, :]
        dataset = torch.from_numpy(dataset.copy()).to(device)
        dataset = torch.add(dataset, 1.0).float().to(device)  # Normalize
        # Run and track inference
        start_time = time.time()
        tracemalloc.start()
        pred, aleatoric = extract_nll_uncertainty(model, dataset, device)
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
    results_dict, epistemic_dict, aleatoric_dict, time_array, memory_array = run_nll_inference(
        MODEL_PATH, _file_names, t_max=T_MAX, device=DEVICE
    )
    # Save results
    print(f"Saving results")
    save_method_results(
        method_name="NLL",
        results_dict=results_dict,
        epistemic_dict=epistemic_dict,
        aleatoric_dict=aleatoric_dict,
        time_array=time_array,
        memory_array=memory_array,
        outname=OUTNAME
    )
    print(f" NLL inference complete!")
    print(f"   Results saved with outname: {OUTNAME}")


main()
