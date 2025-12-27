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
from model_MCDropout import AE_dropout
from common_utils import save_method_results, calculate_std_across_files

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = os.environ.get('OUTNAME', "MCDropout_Final_v1")
MODEL_DIR = os.environ.get('MODEL_DIR', "")
MODEL_NAME = os.environ.get('MODEL_NAME', "model_ae_mcdropout_lr0.001_epochs50_seed0.pth")
NUM_SAMPLES = 30

FILE_IDS = ["145922", "121428", "118626", "143128", "118620"]
FILE_NAMES = [f"Data/Validation/processed_file_{_id}.npy" for _id in FILE_IDS]

#If only some datasets should be processed, besides Original/Referenz
PROCESS_TRUERANDOM = True
PROCESS_OOD = True
PROCESS_COUETTE = True

def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def mc_dropout_predict(model, data, mc_samples=30):
    model.eval()
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            preds.append(model(data).cpu().numpy())
    return np.stack(preds, axis=0)

def process_dataset(model, dataset, mc_samples):

    mc_preds = torch.add(torch.from_numpy(mc_dropout_predict(model, dataset[:, 0:1, :, :, :], mc_samples)), -1.0).float().detach().numpy()
    mean_pred = mc_preds[:, :, 0, :, :, :]
    log_var = mc_preds[:, :, 1, :, :, :]
    return mean_pred, np.std(mc_preds[:, :, 0], axis=0), np.mean(np.sqrt(np.exp(log_var)), axis=0)

def main():
    #load datasets
    datasets = {
        'original': torch.load("dataset/original_2_dataset_tensor.pt", map_location=DEVICE),
        'truerandom': torch.load("dataset/truerandom_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_TRUERANDOM else None,
        'ood': torch.load("dataset/ood_dataset_tensor.pt", map_location=DEVICE) if PROCESS_OOD else None,
        'couette': torch.load("dataset/couette_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_COUETTE else None,
    }
    
    model = AE_dropout(device=DEVICE, in_channels=1, out_channels=2, features=[4, 8, 16], activation=nn.ReLU(inplace=True)).to(DEVICE)
    model.load_state_dict(torch.load(f'{MODEL_DIR}{MODEL_NAME}', map_location='cpu'))
    model.eval()
    
    #Track inference
    start_time = time.time()
    tracemalloc.start()
    mean_pred_orig, epi_orig, ale_orig = process_dataset(model, datasets['original'], NUM_SAMPLES)
    measured_time_mc = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    measured_memory_mc = peak / 1024 / 1024
    
    time_per_file = measured_time_mc / len(FILE_NAMES)
    memory_per_file = measured_memory_mc / len(FILE_NAMES)
    
    results = {"Original": np.mean(mean_pred_orig, axis=0)}
    epistemic = {"Original": epi_orig}
    #For the epistemic comparision
    aleatoric = {"Original": ale_orig}
    
    #In case not all datasets are used
    if PROCESS_TRUERANDOM:
        mean_tr, epi_tr, ale_tr = process_dataset(model, datasets['truerandom'], NUM_SAMPLES)
        results["TrueRandom"], epistemic["TrueRandom"], aleatoric["TrueRandom"] = np.mean(mean_tr, axis=0), epi_tr, ale_tr
    if PROCESS_OOD:
        mean_ood, epi_ood, ale_ood = process_dataset(model, datasets['ood'], NUM_SAMPLES)
        results["OOD"], epistemic["OOD"], aleatoric["OOD"] = np.mean(mean_ood, axis=0), epi_ood, ale_ood
    if PROCESS_COUETTE:
        mean_cou, epi_cou, ale_cou = process_dataset(model, datasets['couette'], NUM_SAMPLES)
        results["Couette"], epistemic["Couette"], aleatoric["Couette"] = np.mean(mean_cou, axis=0), epi_cou, ale_cou
 
    n_files = {"Original": 5, "TrueRandom": 5, "OOD": 6, "Couette": 5}
    time_list = np.array([time_per_file] * len(epistemic))
    memory_list = np.array([memory_per_file] * len(epistemic))
    
    save_method_results(
        method_name="MCDropout", outname=OUTNAME, results_dict=results, epistemic_dict=epistemic, aleatoric_dict=aleatoric,
        time_array=time_list, memory_array=memory_list,
        epistemic_std_dict=calculate_std_across_files(epistemic, n_files, T_MAX),
        aleatoric_std_dict=calculate_std_across_files(aleatoric, n_files, T_MAX),
        results_std_dict=calculate_std_across_files(results, n_files, T_MAX)
    )

main()
