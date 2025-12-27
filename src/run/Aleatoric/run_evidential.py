#!/usr/bin/env python3

import sys
import os
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import time
import tracemalloc
import random
from datetime import datetime
from model_evidential import AE_evidential
from methods.common_utils import save_method_results, load_dataset
from utils import mlready2dataset

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

MODEL_PATH = "model_ae_evidential_lr0.0001_seed0.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = f"Test_Evidential_2"
_model_directory = ""

_ids = ["145922", "121428", "118626", "143128","118620"]
_file_names = [f"Data/Validation/processed_file_{_id}.npy" for _id in _ids]

def load_evidential_model(model_path, device):
    model = AE_evidential(
        device=device,
        in_channels=1,
        out_channels=4,  # mu, log_lambda, log_alpha, log_beta
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f" Loaded Evidential model: {model_path}")
    #except RuntimeError as e:
    #    if "Missing key" in str(e) or "final_projection" in str(e) or "helper_up_2" in str(e):
    #        model.load_weights_selectively(state_dict)
    #    else:
    #        raise RuntimeError(f"Failed to load Evidential model: {e}"
    model.eval()
    return model

def extract_evidential_uncertainty(model, data, device):
    with torch.no_grad():
        data = data.float().to(device)
        pred = model(data)
        mu, v, alpha, beta = AE_evidential.output_to_nig_params(pred)
        aleatoric_uncertainty, _ = AE_evidential.nig_uncertainties_from_params(mu, v, alpha, beta)
        # Denormalize: subtract 1.0 that was added before prediction
        mean_pred = torch.add(mu, -1.0) 
        return mean_pred.cpu().numpy(), aleatoric_uncertainty.cpu().numpy()

def run_evidential_inference(model_path, file_names, t_max=899, device='cuda'):
    """
    Run Evidential inference on multiple files and return standardized results.
    
    Returns:
        results_dict: {dataset_name: predictions_array}
        aleatoric_dict: {dataset_name: aleatoric_uncertainty_array}
        time_array: numpy array of inference times per dataset
        memory_array: numpy array of memory usage per dataset
    """
    print("Evidential Aleatoric Uncertainty Inference")
    print(f"Loading Evidential model: {model_path}")
    model = load_evidential_model(model_path, device)
    
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
        pred, aleatoric = extract_evidential_uncertainty(model, dataset, device)
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
    results_dict, epistemic_dict, aleatoric_dict, time_array, memory_array = run_evidential_inference(
        MODEL_PATH, _file_names, t_max=T_MAX, device=DEVICE
    )
    
    print(f"Saving results")
    save_method_results(
        method_name="Evidential",
        results_dict=results_dict,
        epistemic_dict=epistemic_dict,
        aleatoric_dict=aleatoric_dict,
        time_array=time_array,
        memory_array=memory_array,
        outname=OUTNAME
    )
    print(f"Evidential inference complete!")
    print(f"Results saved with outname: {OUTNAME}")

main()

