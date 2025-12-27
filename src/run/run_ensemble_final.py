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
from model import AE_u_i
from common_utils import save_method_results, calculate_std_across_files

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = os.environ.get('OUTNAME', "Ensemble_Final_v1")
MODEL_DIR = os.environ.get('MODEL_DIR', "")

ENSEMBLE_MODELS = [
    "ensemble/model_ae_baseline_seed0.pth",
    "ensemble/model_ae_baseline_seed1.pth",
    "ensemble/model_ae_baseline_seed2.pth",
    "ensemble/model_ae_baseline_seed3.pth",
    "ensemble/model_ae_baseline_seed4.pth",
    "ensemble/model_ae_baseline_seed5.pth"
]

FILE_IDS = ["145922", "121428", "118626", "143128", "118620"]
FILE_NAMES = [f"Data/Validation/processed_file_{_id}.npy" for _id in FILE_IDS]

#If only some datasets should be processed, besides Original/Referenz
PROCESS_TRUERANDOM = True
PROCESS_OOD = True
PROCESS_COUETTE = True

def load_model(model_name):
    model = AE_u_i(device=DEVICE, in_channels=1, out_channels=1, features=[4, 8, 16], activation=nn.ReLU(inplace=True)).to(DEVICE)
    model.load_state_dict(torch.load(f'{MODEL_DIR}{model_name}', map_location='cpu'))
    model.eval()
    return model

def predict_dataset(model, dataset, t_max, file_names):

    predictions, times, memories = [], [], []
    index = 0
    for file in file_names:
        #Track inference
        start_time = time.time()
        tracemalloc.start()
        pred = torch.add(model(dataset[index:index+t_max, 0, :, :, :]), -1.0).float().to(DEVICE)
        predictions.append(pred.cpu().detach().numpy())
        times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memories.append(peak / 1024 / 1024)
        index += t_max
    
    preds = np.array(predictions)
    return preds.reshape((len(file_names)*preds.shape[1],) + preds.shape[2:]), times, memories

def denormalize(model, dataset):
    return torch.add(model(dataset[:, 0, :, :, :]), -1.0).float().cpu().detach().numpy()

def main():
    #load datasets
    datasets = {
        'original': torch.load("dataset/original_2_dataset_tensor.pt", map_location=DEVICE),
        'truerandom': torch.load("dataset/truerandom_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_TRUERANDOM else None,
        'ood': torch.load("dataset/ood_dataset_tensor.pt", map_location=DEVICE) if PROCESS_OOD else None,
        'couette': torch.load("dataset/couette_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_COUETTE else None,
    }
    
    all_preds_orig, all_preds_tr, all_preds_ood, all_preds_cou = [], [], [], []
    all_times, all_memories = [], []
    for model_name in ENSEMBLE_MODELS:
        model = load_model(model_name)
        pred_orig, times, memories = predict_dataset(model, datasets['original'], T_MAX, FILE_NAMES)
        all_preds_orig.append(pred_orig)
        all_times.append(times)
        all_memories.append(memories)
        
        if PROCESS_TRUERANDOM:
            all_preds_tr.append(denormalize(model, datasets['truerandom']))
        if PROCESS_OOD:
            all_preds_ood.append(denormalize(model, datasets['ood']))
        if PROCESS_COUETTE:
            all_preds_cou.append(denormalize(model, datasets['couette']))
    
    all_preds_orig = np.array(all_preds_orig)
    results = {"Original": np.mean(all_preds_orig, axis=0)}
    epistemic = {"Original": np.std(all_preds_orig, axis=0)}
    #For the epistemic comparision
    aleatoric = {"Original": None}
    
    #In case not all datasets are used
    if PROCESS_TRUERANDOM:
        all_preds_tr = np.array(all_preds_tr)
        results["TrueRandom"] = np.mean(all_preds_tr, axis=0)
        epistemic["TrueRandom"] = np.std(all_preds_tr, axis=0)
        aleatoric["TrueRandom"] = None
    if PROCESS_OOD:
        all_preds_ood = np.array(all_preds_ood)
        results["OOD"] = np.mean(all_preds_ood, axis=0)
        epistemic["OOD"] = np.std(all_preds_ood, axis=0)
        aleatoric["OOD"] = None
    if PROCESS_COUETTE:
        all_preds_cou = np.array(all_preds_cou)
        results["Couette"] = np.mean(all_preds_cou, axis=0)
        epistemic["Couette"] = np.std(all_preds_cou, axis=0)
        aleatoric["Couette"] = None

    time_list = np.array([np.mean(np.sum(np.array(all_times), axis=0), axis=0)] * len(epistemic))
    memory_list = np.array([np.mean(np.sum(np.array(all_memories), axis=0))] * len(epistemic))
    n_files = {"Original": 5, "TrueRandom": 5, "OOD": 6, "Couette": 5}
    
    save_method_results(
        method_name="Ensemble", outname=OUTNAME, results_dict=results, epistemic_dict=epistemic, aleatoric_dict=aleatoric,
        time_array=time_list, memory_array=memory_list,
        epistemic_std_dict=calculate_std_across_files(epistemic, n_files, T_MAX),
        aleatoric_std_dict=calculate_std_across_files(aleatoric, n_files, T_MAX),
        results_std_dict=calculate_std_across_files(results, n_files, T_MAX)
    )

main()
