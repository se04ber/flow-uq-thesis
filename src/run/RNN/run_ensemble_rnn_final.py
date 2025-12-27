#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
import copy
from model import AE_u_i, RNN
from model_rnn_ensemble import Hybrid_MD_RNN_AE_ensemble
from common_utils import save_method_results, calculate_std_across_files
from utils import mlready2dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 899
OUTNAME = os.environ.get('OUTNAME', "Ensemble_RNN_Final_v1")
MODEL_DIR = os.environ.get('MODEL_DIR', "")

RNN_MODELS = [
    "ensemble/model_rnn_lr1e-4_layers1_seq25_seed0_epoch15.pth",
    "ensemble/model_rnn_lr1e-4_layers1_seq25_seed1_epoch15.pth",
    "ensemble/model_rnn_lr1e-4_layers1_seq25_seed2_epoch15.pth",
    "ensemble/model_rnn_lr1e-4_layers1_seq25_seed3_epoch15.pth",
    "ensemble/model_rnn_lr1e-4_layers1_seq25_seed4_epoch15.pth",
    "ensemble/model_rnn_lr1e-4_layers1_seq25_seed5_epoch15.pth",
]

AE_MODELS = [
    "ensemble/model_ae_baseline_seed0.pth",
    "ensemble/model_ae_baseline_seed1.pth",
    "ensemble/model_ae_baseline_seed2.pth",
    "ensemble/model_ae_baseline_seed3.pth",
    "ensemble/model_ae_baseline_seed4.pth",
    "ensemble/model_ae_baseline_seed5.pth",
]

FILE_IDS = ["145922", "121428", "118626", "143128", "118620"]
FILE_NAMES = [f"Data/Validation/processed_file_{_id}.npy" for _id in FILE_IDS]

def load_dataset(file_name):
    dataset = mlready2dataset(file_name)
    if dataset.shape == (900, 1, 24, 24, 24):
        dataset = np.concatenate([dataset, dataset, dataset], axis=1)
    return torch.from_numpy(copy.deepcopy(dataset[:-1, :, :, :, :]))

def load_rnn(rnn_model_path):
    rnn_x = RNN(256, 256, 25, 1, DEVICE)
    rnn_y = RNN(256, 256, 25, 1, DEVICE)
    rnn_z = RNN(256, 256, 25, 1, DEVICE)
    
    state_dict = torch.load(f'{MODEL_DIR}{rnn_model_path}', map_location='cpu')
    for rnn in [rnn_x, rnn_y, rnn_z]:
        rnn.load_state_dict(state_dict)
        rnn.eval()
    return rnn_x, rnn_y, rnn_z

def run_inference(ae_model, rnn_model_path, input_data, t_max):
    rnn_x, rnn_y, rnn_z = load_rnn(rnn_model_path)
    hybrid_model = Hybrid_MD_RNN_AE_ensemble(device=DEVICE, AE_Model_x=ae_model, AE_Model_y=ae_model, AE_Model_z=ae_model,
                                              RNN_ensemble_predictor_x=[rnn_x], RNN_ensemble_predictor_y=[rnn_y],
                                              RNN_ensemble_predictor_z=[rnn_z], seq_length=25)
    
    preds_a = torch.zeros(1, 3, 24, 24, 24).to(DEVICE)
    
    for t in range(t_max):
        data = torch.reshape(input_data[t, :, :, :, :], (1, 3, 24, 24, 24))
        data = torch.add(data, 1.0).float().to(DEVICE)
        
        with torch.cuda.amp.autocast():
            pred = torch.add(hybrid_model(data, return_uncertainty=False), -1).float().to(DEVICE)
            if len(pred.shape) == 4:
                pred = pred.unsqueeze(1)
            if pred.shape[1] == 1:
                pred = pred.repeat(1, 3, 1, 1, 1)
            preds_a = torch.cat((preds_a, pred), 0).to(DEVICE)
    
    return preds_a[1:, 0, :, :, :].cpu().detach().numpy()

def process_inference(ae_model, rnn_model_path, file_name, t_max):
    return run_inference(ae_model, rnn_model_path, load_dataset(file_name), t_max)

def process_from_tensor(ae_model, rnn_model_path, dataset_tensor, start_idx, t_max):
    available_timesteps = dataset_tensor.shape[0] - start_idx
    actual_t_max = min(t_max, available_timesteps - 1)
    
    dataset = dataset_tensor[start_idx:start_idx + actual_t_max + 1].cpu().numpy()
    if dataset.shape[1] == 1:
        dataset = np.concatenate([dataset, dataset, dataset], axis=1)
    elif dataset.shape[1] != 3:
        dataset = np.stack([dataset[:, 0, :, :, :], dataset[:, 0, :, :, :], dataset[:, 0, :, :, :]], axis=1)
    
    input_data = torch.from_numpy(copy.deepcopy(dataset[:-1, :, :, :, :]))
    return run_inference(ae_model, rnn_model_path, input_data, actual_t_max)

def main():
    torch.manual_seed(10)
    np.random.seed(10)
    
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
    
    n_pairs = min(len(AE_MODELS), len(RNN_MODELS))
    all_predictions, all_preds_tr, all_preds_ood, all_preds_cou = [], [], [], []
    
    for pair_idx in range(n_pairs):
        ae_model = AE_u_i(device=DEVICE, in_channels=1, out_channels=1, features=[4, 8, 16], activation=nn.ReLU(inplace=True)).to(DEVICE)
        ae_model.load_state_dict(torch.load(f'{MODEL_DIR}{AE_MODELS[pair_idx]}', map_location='cpu'))
        ae_model.eval()
        
        predictions_files = [process_inference(ae_model, RNN_MODELS[pair_idx], file_name, T_MAX) for file_name in FILE_NAMES]
        all_predictions.append(np.concatenate(predictions_files, axis=0))
        
        truerandom_length = datasets['truerandom'].shape[0]
        predictions_files_tr = [process_from_tensor(ae_model, RNN_MODELS[pair_idx], datasets['truerandom'], file_idx * T_MAX, T_MAX) 
                                for file_idx in range(5)]
        all_preds_tr.append(np.concatenate(predictions_files_tr, axis=0))
        
        ood_length = datasets['ood'].shape[0]
        predictions_files_ood = [process_from_tensor(ae_model, RNN_MODELS[pair_idx], datasets['ood'], file_idx * T_MAX, T_MAX) 
                                 for file_idx in range(6)]
        all_preds_ood.append(np.concatenate(predictions_files_ood, axis=0))
        
        couette_length = datasets['couette'].shape[0]
        predictions_files_cou = [process_from_tensor(ae_model, RNN_MODELS[pair_idx], datasets['couette'], file_idx * T_MAX, T_MAX) 
                                 for file_idx in range(5)]
        all_preds_cou.append(np.concatenate(predictions_files_cou, axis=0))
    
    all_predictions = np.array(all_predictions)
    
    results = {
        "Original": np.mean(all_predictions, axis=0),
        "TrueRandom": np.mean(all_preds_tr, axis=0),
        "OOD": np.mean(all_preds_ood, axis=0),
        "Couette": np.mean(all_preds_cou, axis=0)
    }
    
    epistemic = {
        "Original": np.std(all_predictions, axis=0),
        "TrueRandom": np.std(all_preds_tr, axis=0),
        "OOD": np.std(all_preds_ood, axis=0),
        "Couette": np.std(all_preds_cou, axis=0)
    }
    
    aleatoric = {"Original": None, "TrueRandom": None, "OOD": None, "Couette": None}
    n_files = {"Original": 5, "TrueRandom": 5, "OOD": 6, "Couette": 5}
    
    save_method_results(
        method_name="Ensemble_3_AE_RNN", outname=OUTNAME, results_dict=results, epistemic_dict=epistemic, aleatoric_dict=aleatoric,
        time_array=np.array([0.0] * len(epistemic)), memory_array=np.array([0.0] * len(epistemic)),
        epistemic_std_dict=calculate_std_across_files(epistemic, n_files, T_MAX),
        aleatoric_std_dict=calculate_std_across_files(aleatoric, n_files, T_MAX),
        results_std_dict=calculate_std_across_files(results, n_files, T_MAX)
    )

if __name__ == "__main__":
    main()
