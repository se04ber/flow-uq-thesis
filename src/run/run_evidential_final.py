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
from model_evidential import AE_evidential
from common_utils import save_method_results, calculate_std_across_files

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = os.environ.get('OUTNAME', "Evidential_Final_v1")
MODEL_DIR = os.environ.get('MODEL_DIR', "")
MODEL_NAME = os.environ.get('MODEL_NAME', "model_ae_evidential_lr0.0001_seed0.pth")

FILE_IDS = ["145922", "121428", "118626", "143128", "118620"]
FILE_NAMES = [f"Data/Validation/processed_file_{_id}.npy" for _id in FILE_IDS]

#If only some datasets should be processed, besides Original/Referenz
PROCESS_TRUERANDOM = True
PROCESS_OOD = True
PROCESS_COUETTE = True

def process_dataset(model, dataset, t_max, file_names):

    mean_preds_list, aleatoric_list, epistemic_list = [], [], []
    times, memories = [], []
    index = 0
    for file in file_names:

        #Track inference
        start_time = time.time()
        tracemalloc.start()
        with torch.no_grad():
            preds = torch.add(model(dataset[index:index+t_max, 0:1, :, :, :]), -1.0)
            mean_pred, v, alpha, beta = AE_evidential.output_to_nig_params(preds)
            ale, epi = AE_evidential.nig_uncertainties_from_params(mean_pred, v, alpha, beta)
            
            mean_preds_list.append(mean_pred.cpu().detach().numpy())
            aleatoric_list.append(ale.cpu().detach().numpy())
            epistemic_list.append(epi.cpu().detach().numpy())
        times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memories.append(peak / 1024 / 1024)
        index += t_max
    
    mean_preds = np.array(mean_preds_list).reshape((len(file_names)*mean_preds_list[0].shape[0],) + mean_preds_list[0].shape[1:])
    return mean_preds, np.concatenate(epistemic_list, axis=0), np.concatenate(aleatoric_list, axis=0), times, memories

def denormalize(model, dataset):

    with torch.no_grad():
        preds = torch.add(model(dataset[:, 0:1, :, :, :]), -1.0)
        mean_pred, v, alpha, beta = AE_evidential.output_to_nig_params(preds)
        _, epi = AE_evidential.nig_uncertainties_from_params(mean_pred, v, alpha, beta)
        return mean_pred.cpu().detach().numpy(), epi.cpu().detach().numpy()

def main():
    #load datasets
    datasets = {
        'original': torch.load("dataset/original_2_dataset_tensor.pt", map_location=DEVICE),
        'truerandom': torch.load("dataset/truerandom_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_TRUERANDOM else None,
        'ood': torch.load("dataset/ood_dataset_tensor.pt", map_location=DEVICE) if PROCESS_OOD else None,
        'couette': torch.load("dataset/couette_2_dataset_tensor.pt", map_location=DEVICE) if PROCESS_COUETTE else None,
    }
    
    model = AE_evidential(device=DEVICE, in_channels=1, out_channels=4, features=[4, 8, 16], activation=nn.ReLU(inplace=True)).to(DEVICE)
    model.load_state_dict(torch.load(f'{MODEL_DIR}{MODEL_NAME}', map_location='cpu'))
    model.eval()
    
    mean_preds, epistemic_orig, aleatoric_orig, times, memories = process_dataset(model, datasets['original'], T_MAX, FILE_NAMES)
    
    results = {"Original": mean_preds}
    epistemic = {"Original": epistemic_orig}
    #For the epistemic comparision
    aleatoric = {"Original": aleatoric_orig}
    
    #In case not all datasets are used
    if PROCESS_TRUERANDOM:
        mean_tr, epi_tr = denormalize(model, datasets['truerandom'])
        results["TrueRandom"], epistemic["TrueRandom"], aleatoric["TrueRandom"] = mean_tr, epi_tr, None
    
    if PROCESS_OOD:
        mean_ood, epi_ood = denormalize(model, datasets['ood'])
        results["OOD"], epistemic["OOD"], aleatoric["OOD"] = mean_ood, epi_ood, None
    if PROCESS_COUETTE:
        mean_cou, epi_cou = denormalize(model, datasets['couette'])
        results["Couette"], epistemic["Couette"], aleatoric["Couette"] = mean_cou, epi_cou, None
 
    n_files = {"Original": 5, "TrueRandom": 5, "OOD": 6, "Couette": 5}
    
    save_method_results(
        method_name="Evidential", outname=OUTNAME, results_dict=results, epistemic_dict=epistemic, aleatoric_dict=aleatoric,
        time_array=np.array(times), memory_array=np.array(memories),
        epistemic_std_dict=calculate_std_across_files(epistemic, n_files, T_MAX),
        aleatoric_std_dict=calculate_std_across_files(aleatoric, n_files, T_MAX),
        results_std_dict=calculate_std_across_files(results, n_files, T_MAX)
    )

main()
