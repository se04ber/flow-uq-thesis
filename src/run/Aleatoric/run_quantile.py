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

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

MODEL_PATH = "aleatoric/model_ae_quantile_lr0.0001_seed0.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = f"Test_Quantile_2"
_model_directory = ""

_ids = ["145922", "121428", "118626", "143128","118620"]
_file_names = [f"Data/Validation/processed_file_{_id}.npy" for _id in _ids]

def load_quantile_model(model_path, device):
    model = AE_aleatoric(
        device=device,
        in_channels=1,
        out_channels=2,  # [lower_quantile, upper_quantile]
        features=[4, 8, 16]
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded Quantile model: {model_path}")
    
    model.eval()
    return model

def extract_quantile_uncertainty(model, data, device):
    with torch.no_grad():
        data = data.float().to(device)
        pred = model(data)
        # Denormalize: subtract 1.0 that was added before prediction
        # Note: Only denormalize the quantile bounds, not the mean_pred calculation
        pred_denorm = torch.add(pred, -1.0)
        lower_q = pred_denorm[:, 0:1]  # Lower quantile (e.g., 0.05) - already denormalized
        upper_q = pred_denorm[:, 1:2]  # Upper quantile (e.g., 0.95) - already denormalized
        # Mean as midpoint of denormalized quantiles
        mean_pred = 0.5 * (lower_q + upper_q)  # Mean as midpoint
        aleatoric_uncertainty = 0.5 * torch.abs(upper_q - lower_q)  # Half the range
        
        # Additional denormalization: subtract 1.0 to fix MAE being +1 too large
        mean_pred = torch.add(mean_pred, -1.0)
        return mean_pred.cpu().numpy(), aleatoric_uncertainty.cpu().numpy()

def run_quantile_inference(model_path, file_names, t_max=899, device='cuda'):
    """
    Run Quantile inference on multiple files and return standardized results.
    
    Returns:
        results_dict: {dataset_name: predictions_array}
        aleatoric_dict: {dataset_name: aleatoric_uncertainty_array}
        time_array: numpy array of inference times per dataset
        memory_array: numpy array of memory usage per dataset
    """
    print("Quantile Aleatoric Uncertainty Inference")
    print(f"Loading Quantile model: {model_path}")
    model = load_quantile_model(model_path, device)
    
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
        
        # Load dataset
        dataset = mlready2dataset(file_name)
        dataset = dataset[:, :, :, :, :]
        dataset = torch.from_numpy(dataset.copy()).to(device)
        dataset = torch.add(dataset, 1.0).float().to(device)  # Normalize
        
        #Track inference
        start_time = time.time()
        tracemalloc.start()
        pred, aleatoric = extract_quantile_uncertainty(model, dataset, device)
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
    results_dict, epistemic_dict, aleatoric_dict, time_array, memory_array = run_quantile_inference(
        MODEL_PATH, _file_names, t_max=T_MAX, device=DEVICE
    )
    
    print(f"Saving results")
    save_method_results(
        method_name="Quantile",
        results_dict=results_dict,
        epistemic_dict=epistemic_dict,
        aleatoric_dict=aleatoric_dict,
        time_array=time_array,
        memory_array=memory_array,
        outname=OUTNAME
    )
    print(f"Quantile inference complete!")
    print(f"Results saved with outname: {OUTNAME}")

main()

