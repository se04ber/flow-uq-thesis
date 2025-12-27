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
from model_pi import AE_pi
from model_evidential import AE_evidential
from methods.common_utils import save_method_results, load_dataset
from utils import mlready2dataset

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

MODEL_PATH = "aleatoric/model_ae_pi_lr0.001_seed0.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = f"Test_PI_2"
_model_directory = ""

_ids = ["145922", "121428", "118626", "143128","118620"]
_file_names = [f"Data/Validation/processed_file_{_id}.npy" for _id in _ids]

def load_pi_model(model_path, device):
    model = AE_pi(
        device=device,
        in_channels=1,
        out_channels=3,  # [mu, lower, upper]
        features=[4, 8, 16]
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded PI model: {model_path}")
    
    model.eval()
    return model

def extract_pi_uncertainty(pi_model, data, device):
    """Extract predictions and aleatoric uncertainty from PI model
    
    Uses Evidential predictions but PI's own aleatoric uncertainty.
    """
    with torch.no_grad():
        data = data.float().to(device)
        # Get PI aleatoric uncertainty
        pi_pred = pi_model(data)
        pi_pred = torch.add(pi_pred, -1.0)  # Denormalize
        mu = pi_pred[:, 0:1]  # Point prediction (not used)
        lower = pi_pred[:, 1:2]  # Lower bound
        upper = pi_pred[:, 2:3]  # Upper bound
        aleatoric_uncertainty = 0.5 * (upper - lower) * 1e4  # Half the PI width with scaling
        #mean_pred = mu
        return mu.cpu().numpy(), aleatoric_uncertainty.cpu().numpy()

def run_pi_inference(pi_model_path, file_names, t_max=899, device='cuda'):
    print("PI Aleatoric Uncertainty Inference")
    print(f"Loading PI model: {pi_model_path}")
    pi_model = load_pi_model(pi_model_path, device)
  
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
        pred, aleatoric = extract_pi_uncertainty(pi_model, dataset, device)
        inference_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak / 1024 / 1024
        
        all_predictions.append(pred)
        all_aleatoric.append(aleatoric)
        all_times.append(inference_time)
        all_memories.append(memory_usage)
        print(f"    Time: {inference_time:.3f}s, Memory: {memory_usage:.1f}MB")
    
    if all_predictions:
        results_dict['Original'] = np.concatenate(all_predictions, axis=0)
        aleatoric_dict['Original'] = np.concatenate(all_aleatoric, axis=0)
        time_list.append(np.mean(all_times))
        memory_list.append(np.mean(all_memories))
        print(f"Original: shape {results_dict['Original'].shape}")
    
    epistemic_dict = {'Original': None}
    return results_dict, epistemic_dict, aleatoric_dict, np.array(time_list), np.array(memory_list)

def main():
    results_dict, epistemic_dict, aleatoric_dict, time_array, memory_array = run_pi_inference(
        MODEL_PATH, _file_names, t_max=T_MAX, device=DEVICE
    )
    
    print(f"Saving results")
    save_method_results(
        method_name="PI",
        results_dict=results_dict,
        epistemic_dict=epistemic_dict,
        aleatoric_dict=aleatoric_dict,
        time_array=time_array,
        memory_array=memory_array,
        outname=OUTNAME
    )
    
    print(f"PI inference complete!")
    print(f"Results saved with outname: {OUTNAME}")

main()

