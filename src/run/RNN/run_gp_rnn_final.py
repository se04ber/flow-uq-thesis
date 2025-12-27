#!/usr/bin/env python3
"""Gaussian Process RNN hybrid model for epistemic uncertainty estimation.

Reference:
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
    MIT Press. Available at http://www.gaussianprocess.org/gpml/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
import copy
import random
from model import AE_u_i
from model_rnn_gp import Hybrid_MD_RNN_AE_gp, RNN_gp_predictor
from common_utils import save_method_results, calculate_std_across_files
from utils import mlready2dataset
from investigate_ensembles import RNN_old as RNN

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = os.environ.get('OUTNAME', "GP_RNN_Final_v1")
MODEL_DIR = os.environ.get('MODEL_DIR', "")

AE_MODEL = "Model_AE_u_i_LR0_0001_i_Piet22"
RNN_MODEL = "Model_RNN_LR1e-5_Lay1_Seq25_i"
GP_MODEL_PATH = "Model_RNN_GP_LR1e-5_Lay1_Seq25_i"

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

def create_gp_hybrid(ae_model, gp_model_path):
    rnn_gp_x = RNN_gp_predictor(256, 256, 25, 1, DEVICE)
    rnn_gp_y = RNN_gp_predictor(256, 256, 25, 1, DEVICE)
    rnn_gp_z = RNN_gp_predictor(256, 256, 25, 1, DEVICE)
    
    for rnn_gp, suffix in [(rnn_gp_x, "_x"), (rnn_gp_y, "_y"), (rnn_gp_z, "_z")]:
        full_path = os.path.join(MODEL_DIR, f'{gp_model_path}{suffix}') if MODEL_DIR else f'{gp_model_path}{suffix}'
        rnn_path = f'{full_path}_rnn.pth'
        gp_path = f'{full_path}_gp.pkl'
        
        if os.path.exists(rnn_path) and os.path.exists(gp_path):
            print(f" Loading GP model from {full_path}")
            rnn_gp.load_models(full_path)
        else:
            print(f"GP model not found at {full_path}, training GP...")
            print(f"   Expected files: {rnn_path}, {gp_path}")
            from utils import get_AE_loaders
            train_loaders, _ = get_AE_loaders(path="", data_distribution='get_KVS', batch_size=8, shuffle=True)
            rnn_gp.train_gp(train_loaders[0], ae_model)
            rnn_gp.save_models(full_path)
            print(f" GP model trained and saved to {full_path}")
    
    return Hybrid_MD_RNN_AE_gp(device=DEVICE, AE_Model_x=ae_model, AE_Model_y=ae_model, AE_Model_z=ae_model,
                                RNN_gp_predictor_x=rnn_gp_x, RNN_gp_predictor_y=rnn_gp_y, RNN_gp_predictor_z=rnn_gp_z, seq_length=25)

def extract_gp_uncertainty(hybrid_model, data):
    with torch.no_grad():
        if not hybrid_model.rnn_gp_x.gp_trained:
            return None
        
        interim_x = torch.reshape(hybrid_model.sequence_x, (1, hybrid_model.seq_length, 256)).to(DEVICE)
        interim_y = torch.reshape(hybrid_model.sequence_y, (1, hybrid_model.seq_length, 256)).to(DEVICE)
        interim_z = torch.reshape(hybrid_model.sequence_z, (1, hybrid_model.seq_length, 256)).to(DEVICE)
        
        rnn_output_x_seq = hybrid_model.rnn_gp_x.rnn_model(interim_x).unsqueeze(1).repeat(1, hybrid_model.seq_length, 1)
        rnn_output_y_seq = hybrid_model.rnn_gp_y.rnn_model(interim_y).unsqueeze(1).repeat(1, hybrid_model.seq_length, 1)
        rnn_output_z_seq = hybrid_model.rnn_gp_z.rnn_model(interim_z).unsqueeze(1).repeat(1, hybrid_model.seq_length, 1)
        
        _, std_x = hybrid_model.rnn_gp_x.predict_gp(rnn_output_x_seq)
        _, std_y = hybrid_model.rnn_gp_y.predict_gp(rnn_output_y_seq)
        _, std_z = hybrid_model.rnn_gp_z.predict_gp(rnn_output_z_seq)
        
        spatial_shape = (1, 1, 24, 24, 24)
        epistemic_x = torch.ones(spatial_shape, device=DEVICE) * float(std_x[0])
        epistemic_y = torch.ones(spatial_shape, device=DEVICE) * float(std_y[0])
        epistemic_z = torch.ones(spatial_shape, device=DEVICE) * float(std_z[0])
        
        return torch.cat([epistemic_x, epistemic_y, epistemic_z], dim=1)

def process_inference(ae_model, rnn_model_path, gp_model_path, file_name, t_max):
    input_data = load_dataset(file_name)
    
    rnn_x = RNN(256, 256, 25, 1, DEVICE)
    rnn_y = RNN(256, 256, 25, 1, DEVICE)
    rnn_z = RNN(256, 256, 25, 1, DEVICE)
    
    state_dict = torch.load(f'{MODEL_DIR}{rnn_model_path}', map_location='cpu')
    for rnn in [rnn_x, rnn_y, rnn_z]:
        rnn.load_state_dict(state_dict)
        rnn.eval()
    
    hybrid_model = create_gp_hybrid(ae_model, gp_model_path)
    hybrid_model.rnn_gp_x.rnn_model = rnn_x
    hybrid_model.rnn_gp_y.rnn_model = rnn_y
    hybrid_model.rnn_gp_z.rnn_model = rnn_z
    
    predictions, epistemic_list = [], []
    
    for t in range(t_max):
        data = torch.reshape(input_data[t, :, :, :, :], (1, 3, 24, 24, 24))
        data = torch.add(data, 1.0).float().to(DEVICE)
        
        with torch.cuda.amp.autocast():
            result = hybrid_model(data, return_uncertainty=True, dataset=input_data)
            
            if isinstance(result, tuple) and len(result) == 2:
                pred, unc = result
                epi_unc = unc[1] if isinstance(unc, tuple) and len(unc) >= 2 else unc
            else:
                pred = result
                epi_unc = extract_gp_uncertainty(hybrid_model, data)
            
            pred = torch.add(pred, -1.0)
            mean_pred_np = pred[:, 0].cpu().detach().numpy()
            if mean_pred_np.shape[0] == 1:
                mean_pred_np = mean_pred_np[0]
            
            mean_pred_np = np.squeeze(mean_pred_np)
            if mean_pred_np.shape != (24, 24, 24):
                mean_pred_np = mean_pred_np.reshape(24, 24, 24) if mean_pred_np.size == 24*24*24 else np.zeros((24, 24, 24))
            
            predictions.append(mean_pred_np)
            
            if epi_unc is not None:
                epi_np = epi_unc[:, 0, :, :, :].cpu().detach().numpy()
                if epi_np.shape[0] == 1:
                    epi_np = epi_np[0]
                epistemic_list.append(epi_np)
            else:
                epistemic_list.append(np.zeros((24, 24, 24)))
    
    return np.array(predictions), np.array(epistemic_list)

def process_from_tensor(ae_model, rnn_model_path, gp_model_path, dataset_tensor, start_idx, t_max):
    available_timesteps = dataset_tensor.shape[0] - start_idx
    actual_t_max = min(t_max, available_timesteps - 1)
    
    dataset = dataset_tensor[start_idx:start_idx + actual_t_max + 1].cpu().numpy()
    if dataset.shape[1] == 1:
        dataset = np.concatenate([dataset, dataset, dataset], axis=1)
    elif dataset.shape[1] != 3:
        dataset = np.stack([dataset[:, 0, :, :, :], dataset[:, 0, :, :, :], dataset[:, 0, :, :, :]], axis=1)
    
    input_data = torch.from_numpy(copy.deepcopy(dataset[:-1, :, :, :, :]))
    
    rnn_x = RNN(256, 256, 25, 1, DEVICE)
    rnn_y = RNN(256, 256, 25, 1, DEVICE)
    rnn_z = RNN(256, 256, 25, 1, DEVICE)
    
    state_dict = torch.load(f'{MODEL_DIR}{rnn_model_path}', map_location='cpu')
    for rnn in [rnn_x, rnn_y, rnn_z]:
        rnn.load_state_dict(state_dict)
        rnn.eval()
    
    hybrid_model = create_gp_hybrid(ae_model, gp_model_path)
    hybrid_model.rnn_gp_x.rnn_model = rnn_x
    hybrid_model.rnn_gp_y.rnn_model = rnn_y
    hybrid_model.rnn_gp_z.rnn_model = rnn_z
    
    predictions, epistemic_list = [], []
    
    for t in range(actual_t_max):
        data = torch.reshape(input_data[t, :, :, :, :], (1, 3, 24, 24, 24))
        data = torch.add(data, 1.0).float().to(DEVICE)
        
        with torch.cuda.amp.autocast():
            result = hybrid_model(data, return_uncertainty=True, dataset=input_data)
            
            if isinstance(result, tuple) and len(result) == 2:
                pred, unc = result
                epi_unc = unc[1] if isinstance(unc, tuple) and len(unc) >= 2 else unc
            else:
                pred = result
                epi_unc = extract_gp_uncertainty(hybrid_model, data)
            
            pred = torch.add(pred, -1.0)
            mean_pred_np = pred[:, 0].cpu().detach().numpy()
            if mean_pred_np.shape[0] == 1:
                mean_pred_np = mean_pred_np[0]
            
            mean_pred_np = np.squeeze(mean_pred_np)
            if mean_pred_np.shape != (24, 24, 24):
                mean_pred_np = mean_pred_np.reshape(24, 24, 24) if mean_pred_np.size == 24*24*24 else np.zeros((24, 24, 24))
            
            predictions.append(mean_pred_np)
            
            if epi_unc is not None:
                epi_np = epi_unc[:, 0, :, :, :].cpu().detach().numpy()
                if epi_np.shape[0] == 1:
                    epi_np = epi_np[0]
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
    
    ae_model = AE_u_i(device=DEVICE, in_channels=1, out_channels=1, features=[4, 8, 16], activation=nn.ReLU(inplace=True)).to(DEVICE)
    ae_model.load_state_dict(torch.load(f'{MODEL_DIR}{AE_MODEL}', map_location='cpu'))
    ae_model.eval()
    
    mean_preds_files, epistemic_files = [], []
    for file_name in FILE_NAMES:
        mean_pred, epistemic = process_inference(ae_model, RNN_MODEL, GP_MODEL_PATH, file_name, T_MAX)
        mean_preds_files.append(mean_pred)
        epistemic_files.append(epistemic)
    
    results = {"Original": np.concatenate(mean_preds_files, axis=0)}
    epistemic = {"Original": np.concatenate(epistemic_files, axis=0)}
    
    #In case not all datasets are used
    if PROCESS_TRUERANDOM:
        predictions_files_tr = [process_from_tensor(ae_model, RNN_MODEL, GP_MODEL_PATH, datasets['truerandom'], file_idx * T_MAX, T_MAX) 
                                for file_idx in range(5)]
        results["TrueRandom"] = np.concatenate([p[0] for p in predictions_files_tr], axis=0)
        epistemic["TrueRandom"] = np.concatenate([p[1] for p in predictions_files_tr], axis=0)
    if PROCESS_OOD:
        predictions_files_ood = [process_from_tensor(ae_model, RNN_MODEL, GP_MODEL_PATH, datasets['ood'], file_idx * T_MAX, T_MAX) 
                                 for file_idx in range(6)]
        results["OOD"] = np.concatenate([p[0] for p in predictions_files_ood], axis=0)
        epistemic["OOD"] = np.concatenate([p[1] for p in predictions_files_ood], axis=0)
    if PROCESS_COUETTE:
        predictions_files_cou = [process_from_tensor(ae_model, RNN_MODEL, GP_MODEL_PATH, datasets['couette'], file_idx * T_MAX, T_MAX) 
                                 for file_idx in range(5)]
        results["Couette"] = np.concatenate([p[0] for p in predictions_files_cou], axis=0)
        epistemic["Couette"] = np.concatenate([p[1] for p in predictions_files_cou], axis=0)
 
    #For the epistemic comparision
    aleatoric = {"Original": None, "TrueRandom": None, "OOD": None, "Couette": None}
    n_files = {"Original": 5, "TrueRandom": 5, "OOD": 6, "Couette": 5}
    
    save_method_results(
        method_name="GP_RNN", outname=OUTNAME, results_dict=results, epistemic_dict=epistemic, aleatoric_dict=aleatoric,
        time_array=np.array([0.0]), memory_array=np.array([0.0]),
        epistemic_std_dict=calculate_std_across_files(epistemic, n_files, T_MAX),
        aleatoric_std_dict=calculate_std_across_files(aleatoric, n_files, T_MAX),
        results_std_dict=calculate_std_across_files(results, n_files, T_MAX)
    )

main()
