#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
import copy
import random
from model_evidential import AE_evidential
from model_rnn_evidential import Hybrid_MD_RNN_AE_evidential_combined, RNN_evidential
from common_utils import save_method_results, calculate_std_across_files
from utils import mlready2dataset

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = os.environ.get('OUTNAME', "Evidential_RNN_Final_v1")
MODEL_DIR = os.environ.get('MODEL_DIR', "")

AE_MODEL = "model_ae_evidential_lr0.0001_seed0.pth"
RNN_MODEL_NAME = "evidential/model_rnn_evidential_lr1e-3_layers1_seq25.pth"

UNCERTAINTY_METHOD = 'bounds'
AE_UNCERTAINTY_METHOD = 'bounds'

FILE_IDS = ["145922", "121428", "118626", "143128", "118620"]
FILE_NAMES = [f"Data/Validation/processed_file_{_id}.npy" for _id in FILE_IDS]

#If only some datasets should be processed, besides Original/Referenz
PROCESS_TRUERANDOM = True
PROCESS_OOD = True
PROCESS_COUETTE = True

def load_dataset(file_name):
    dataset = mlready2dataset(file_name)
    if dataset.shape == (900, 1, 24, 24, 24):
        dataset = np.concatenate([dataset, dataset, dataset], axis=1)
    return torch.from_numpy(copy.deepcopy(dataset[:-1, :, :, :, :]))

def extract_epistemic(hybrid_model, data):
    with torch.no_grad():
        result = hybrid_model(data, return_uncertainty=True)
        if isinstance(result, tuple) and len(result) == 2:
            pred, unc_tuple = result
            if isinstance(unc_tuple, tuple) and len(unc_tuple) >= 1:
                final_unc = unc_tuple[0] if len(unc_tuple) > 0 else None
                if final_unc is not None and isinstance(final_unc, torch.Tensor) and torch.all(final_unc == 0) and len(unc_tuple) >= 2:
                    final_unc = unc_tuple[1]
                return pred, final_unc
        return result, None

def process_inference(ae_model, rnn_model_name, file_name, t_max):
    input_data = load_dataset(file_name)
    
    rnn_evidential_x = RNN_evidential(256, 256, 25, 1, DEVICE)
    rnn_evidential_y = RNN_evidential(256, 256, 25, 1, DEVICE)
    rnn_evidential_z = RNN_evidential(256, 256, 25, 1, DEVICE)
    
    for rnn, suffix in [(rnn_evidential_x, "_x"), (rnn_evidential_y, "_y"), (rnn_evidential_z, "_z")]:
        model_path = f'{MODEL_DIR}{rnn_model_name}'#{suffix}'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RNN model file not found: {model_path}")
        rnn.load_state_dict(torch.load(model_path, map_location='cpu'))
        rnn.eval()
    
    hybrid_model = Hybrid_MD_RNN_AE_evidential_combined(device=DEVICE, AE_Model_x=ae_model, AE_Model_y=ae_model, AE_Model_z=ae_model,
                                                         RNN_Model_x=rnn_evidential_x, RNN_Model_y=rnn_evidential_y, RNN_Model_z=rnn_evidential_z,
                                                         seq_length=25, n_mc_samples=20, uncertainty_method=UNCERTAINTY_METHOD,
                                                         ae_uncertainty_method=AE_UNCERTAINTY_METHOD)
    
    predictions, epistemic_list = [], []
    for t in range(t_max):
        data = torch.reshape(input_data[t, :, :, :, :], (1, 3, 24, 24, 24))
        data = torch.add(data, 1.0).float().to(DEVICE)
        
        with torch.cuda.amp.autocast():
            pred, unc_tuple = extract_epistemic(hybrid_model, data)
            pred = torch.add(pred, -1).float().to(DEVICE)
            
            if len(pred.shape) == 5 and pred.shape[1] == 4:
                mean_pred, _, _, _ = AE_evidential.output_to_nig_params(pred)
                mean_pred_np = mean_pred.squeeze(0).cpu().numpy()
            elif len(pred.shape) == 5:
                mean_pred_np = pred[:, 0].cpu().numpy()
                if mean_pred_np.shape[0] == 1:
                    mean_pred_np = mean_pred_np[0]
            else:
                mean_pred_np = pred.squeeze().cpu().numpy()
            
            mean_pred_np = np.squeeze(mean_pred_np)
            if mean_pred_np.shape != (24, 24, 24):
                mean_pred_np = mean_pred_np.reshape(24, 24, 24) if mean_pred_np.size == 24*24*24 else np.zeros((24, 24, 24))
            
            predictions.append(mean_pred_np)
            
            if unc_tuple is not None and isinstance(unc_tuple, torch.Tensor):
                if len(unc_tuple.shape) == 5:
                    epi_np = unc_tuple[:, 0, :, :, :].cpu().numpy()
                elif len(unc_tuple.shape) == 4:
                    epi_np = unc_tuple[0, :, :, :].cpu().numpy() if unc_tuple.shape[0] == 1 else unc_tuple.cpu().numpy()
                elif len(unc_tuple.shape) == 3:
                    epi_np = unc_tuple.cpu().numpy()
                else:
                    epi_np = unc_tuple.cpu().numpy()
                
                epi_np = np.squeeze(epi_np)
                if epi_np.shape != (24, 24, 24):
                    epi_np = epi_np.reshape(24, 24, 24) if epi_np.size == 24*24*24 else np.zeros((24, 24, 24))
                epistemic_list.append(epi_np)
            else:
                epistemic_list.append(np.zeros((24, 24, 24)))
    
    return np.array(predictions), np.array(epistemic_list)

def process_from_tensor(ae_model, rnn_model_name, dataset_tensor, start_idx, t_max):
    available_timesteps = dataset_tensor.shape[0] - start_idx
    actual_t_max = min(t_max, available_timesteps - 1)
    
    if actual_t_max <= 0:
        return np.array([]), np.array([])
    
    dataset = dataset_tensor[start_idx:start_idx + actual_t_max + 1].cpu().numpy()
    if dataset.shape[1] == 1:
        dataset = np.concatenate([dataset, dataset, dataset], axis=1)
    elif dataset.shape[1] != 3:
        dataset = np.stack([dataset[:, 0, :, :, :], dataset[:, 0, :, :, :], dataset[:, 0, :, :, :]], axis=1)
    
    input_data = torch.from_numpy(copy.deepcopy(dataset[:-1, :, :, :, :]))
    
    rnn_evidential_x = RNN_evidential(256, 256, 25, 1, DEVICE)
    rnn_evidential_y = RNN_evidential(256, 256, 25, 1, DEVICE)
    rnn_evidential_z = RNN_evidential(256, 256, 25, 1, DEVICE)
    
    for rnn, suffix in [(rnn_evidential_x, "_x"), (rnn_evidential_y, "_y"), (rnn_evidential_z, "_z")]:
        model_path = f'{MODEL_DIR}{rnn_model_name}'#{suffix}'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RNN model file not found: {model_path}")
        rnn.load_state_dict(torch.load(model_path, map_location='cpu'))
        rnn.eval()
    
    hybrid_model = Hybrid_MD_RNN_AE_evidential_combined(device=DEVICE, AE_Model_x=ae_model, AE_Model_y=ae_model, AE_Model_z=ae_model,
                                                         RNN_Model_x=rnn_evidential_x, RNN_Model_y=rnn_evidential_y, RNN_Model_z=rnn_evidential_z,
                                                         seq_length=25, n_mc_samples=20, uncertainty_method=UNCERTAINTY_METHOD,
                                                         ae_uncertainty_method=AE_UNCERTAINTY_METHOD)
    
    predictions, epistemic_list = [], []
    for t in range(actual_t_max):
        data = torch.reshape(input_data[t, :, :, :, :], (1, 3, 24, 24, 24))
        data = torch.add(data, 1.0).float().to(DEVICE)
        
        with torch.cuda.amp.autocast():
            pred, unc_tuple = extract_epistemic(hybrid_model, data)
            pred = torch.add(pred, -1).float().to(DEVICE)
            
            if len(pred.shape) == 5 and pred.shape[1] == 4:
                mean_pred, _, _, _ = AE_evidential.output_to_nig_params(pred)
                mean_pred_np = mean_pred.squeeze(0).cpu().numpy()
            elif len(pred.shape) == 5:
                mean_pred_np = pred[:, 0].cpu().numpy()
                if mean_pred_np.shape[0] == 1:
                    mean_pred_np = mean_pred_np[0]
            else:
                mean_pred_np = pred.squeeze().cpu().numpy()
            
            mean_pred_np = np.squeeze(mean_pred_np)
            if mean_pred_np.shape != (24, 24, 24):
                mean_pred_np = mean_pred_np.reshape(24, 24, 24) if mean_pred_np.size == 24*24*24 else np.zeros((24, 24, 24))
            
            predictions.append(mean_pred_np)
            
            if unc_tuple is not None and isinstance(unc_tuple, torch.Tensor):
                if len(unc_tuple.shape) == 5:
                    epi_np = unc_tuple[:, 0, :, :, :].cpu().numpy()
                elif len(unc_tuple.shape) == 4:
                    epi_np = unc_tuple[0, :, :, :].cpu().numpy() if unc_tuple.shape[0] == 1 else unc_tuple.cpu().numpy()
                elif len(unc_tuple.shape) == 3:
                    epi_np = unc_tuple.cpu().numpy()
                else:
                    epi_np = unc_tuple.cpu().numpy()
                
                epi_np = np.squeeze(epi_np)
                if epi_np.shape != (24, 24, 24):
                    epi_np = epi_np.reshape(24, 24, 24) if epi_np.size == 24*24*24 else np.zeros((24, 24, 24))
                epistemic_list.append(epi_np)
            else:
                epistemic_list.append(np.zeros((24, 24, 24)))
    
    return np.array(predictions), np.array(epistemic_list)

def main():
    #load datasets
    datasets = {
        'truerandom': torch.load("dataset/truerandom_2_dataset_tensor.pt", map_location=DEVICE),
        'ood': torch.load("dataset/ood_dataset_tensor.pt", map_location=DEVICE),
        'couette': torch.load("dataset/couette_2_dataset_tensor.pt", map_location=DEVICE),
    }
    
    for name, ds in datasets.items():
        if len(ds.shape) == 5 and ds.shape[1] > 3:
            datasets[name] = ds.transpose(0, 1).contiguous()
        if ds.shape[1] == 1:
            datasets[name] = ds.repeat(1, 3, 1, 1, 1)
    
    ae_model = AE_evidential(device=DEVICE, in_channels=1, out_channels=4, features=[4, 8, 16], activation=nn.ReLU(inplace=True)).to(DEVICE)
    ae_model.load_state_dict(torch.load(f'{MODEL_DIR}{AE_MODEL}', map_location='cpu'))
    ae_model.eval()
    
    mean_preds_files, epistemic_files = [], []
    for file_name in FILE_NAMES:
        mean_pred, epistemic = process_inference(ae_model, RNN_MODEL_NAME, file_name, T_MAX)
        mean_preds_files.append(mean_pred)
        epistemic_files.append(epistemic)
    results = {"Original": np.concatenate(mean_preds_files, axis=0)}
    epistemic = {"Original": np.concatenate(epistemic_files, axis=0)}
    
    #In case not all datasets are used
    if PROCESS_TRUERANDOM:
        predictions_files_tr = [process_from_tensor(ae_model, RNN_MODEL_NAME, datasets['truerandom'], file_idx * T_MAX, T_MAX) 
                                for file_idx in range(5)]
        results["TrueRandom"] = np.concatenate([p[0] for p in predictions_files_tr], axis=0)
        epistemic["TrueRandom"] = np.concatenate([p[1] for p in predictions_files_tr], axis=0)
        expected_len = 5 * T_MAX
        actual_len = results["TrueRandom"].shape[0]
       
    if PROCESS_OOD:
        predictions_files_ood = [process_from_tensor(ae_model, RNN_MODEL_NAME, datasets['ood'], file_idx * T_MAX, T_MAX) 
                                    for file_idx in range(6)]
        results["OOD"] = np.concatenate([p[0] for p in predictions_files_ood], axis=0)
        epistemic["OOD"] = np.concatenate([p[1] for p in predictions_files_ood], axis=0)
        expected_len = 6 * T_MAX
        actual_len = results["OOD"].shape[0]
                
    if PROCESS_COUETTE:
            predictions_files_cou = [process_from_tensor(ae_model, RNN_MODEL_NAME, datasets['couette'], file_idx * T_MAX, T_MAX) 
                                     for file_idx in range(5)]
            results["Couette"] = np.concatenate([p[0] for p in predictions_files_cou], axis=0)
            epistemic["Couette"] = np.concatenate([p[1] for p in predictions_files_cou], axis=0)
            expected_len = 5 * T_MAX
            actual_len = results["Couette"].shape[0]
        
    #For the epistemic comparision
    aleatoric = {"Original": None, "TrueRandom": None, "OOD": None, "Couette": None}
    n_files = {"Original": 5, "TrueRandom": 5, "OOD": 6, "Couette": 5}
    
    save_method_results(
        method_name="Fixed_AE_Evidential_RNN", outname=OUTNAME, results_dict=results, epistemic_dict=epistemic, aleatoric_dict=aleatoric,
        time_array=np.array([0.0]), memory_array=np.array([0.0]),
        epistemic_std_dict=calculate_std_across_files(epistemic, n_files, T_MAX),
        aleatoric_std_dict=calculate_std_across_files(aleatoric, n_files, T_MAX),
        results_std_dict=calculate_std_across_files(results, n_files, T_MAX)
    )

main()
