#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import torch
import random
from pathlib import Path
from utils import mlready2dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_MAX = 900
N_FILES = 5
OUTPUT_DIR = "dataset"

COUETTE_CONFIGS = [
    {"id": "6_u_wall_5_0_0_top", "wall_pos": "top", "velocity": "5_0"},
    {"id": "4_u_wall_3_0_1_middle", "wall_pos": "middle", "velocity": "3_0"},
    {"id": "2_u_wall_1_5_2_bottom", "wall_pos": "bottom", "velocity": "1_5"},
    {"id": "5_u_wall_4_0_1_middle", "wall_pos": "middle", "velocity": "4_0"},
    {"id": "0_u_wall_0_5_0_top", "wall_pos": "top", "velocity": "0_5"}
]

def generate_truerandom(n_timesteps, spatial_dims=(24, 24, 24), value_range=(1, 11)):
    augmented = np.zeros((n_timesteps, 1, *spatial_dims))
    for t in range(n_timesteps):
        augmented[t, 0] = np.random.randint(value_range[0], value_range[1], size=spatial_dims)
    dataset_tensor = torch.from_numpy(augmented.copy()).to(DEVICE)
    dataset_tensor = torch.add(dataset_tensor, 1.0).float().to(DEVICE)
    return dataset_tensor

def load_and_concatenate_files(file_paths, t_max=T_MAX):
    datasets = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        data = mlready2dataset(file_path)
        data = data[:t_max]
        datasets.append(data)
    
    if not datasets:
        return None
    
    combined = np.concatenate(datasets, axis=0)
    dataset_tensor = torch.from_numpy(combined.copy()).to(DEVICE)
    dataset_tensor = torch.add(dataset_tensor, 1.0).float().to(DEVICE)
    return dataset_tensor

def generate_original(validation_path="Data/Validation", validation_ids=None, output_path=None):
    if validation_ids is None:
        validation_ids = ["145922", "121428", "118626", "143128", "118620"]
    
    file_paths = [f"{validation_path}/processed_file_{_id}.npy" for _id in validation_ids]
    
    dataset = load_and_concatenate_files(file_paths)
    if dataset is None:
        return False
    
    output_path = output_path or f"{OUTPUT_DIR}/original_2_dataset_tensor.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset.cpu(), output_path)
    print(f"Generated Original: {dataset.shape} -> {output_path}")
    return True

def generate_truerandom_dataset(n_files=N_FILES, t_max_per_file=T_MAX, output_path=None):
    total_timesteps = n_files * t_max_per_file
    dataset = generate_truerandom(total_timesteps)
    
    output_path = output_path or f"{OUTPUT_DIR}/truerandom_2_dataset_tensor.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset.cpu(), output_path)
    print(f"Generated TrueRandom: {dataset.shape} -> {output_path}")
    return True

def generate_ood(ood_path="Data/OOD", ood_ids=None, output_path=None):
    if ood_ids is None:
        ood_ids = ["1921", "1949", "1200", "1445", "2166"]
    
    file_paths = [f"{ood_path}/processed_file_{_id}.npy" for _id in ood_ids]
    dataset = load_and_concatenate_files(file_paths)
    if dataset is None:
        return False
    
    output_path = output_path or f"{OUTPUT_DIR}/ood_dataset_tensor.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset.cpu(), output_path)
    print(f"Generated OOD: {dataset.shape} -> {output_path}")
    return True

def generate_couette(couette_path="Data/Couette", configs=None, output_path=None):
    if configs is None:
        configs = COUETTE_CONFIGS
    
    file_paths = []
    for config in configs[:N_FILES]:
        velocity_str = str(config['velocity']).replace('.', '_')
        full_id = f"{config['id']}_couette_md_domain_{config['wall_pos']}_0_oscil_{velocity_str}_u_wall"
        file_path = f"{couette_path}/processed_file_{full_id}.npy"
        file_paths.append(file_path)
    
    dataset = load_and_concatenate_files(file_paths)
    if dataset is None:
        return False
    
    output_path = output_path or f"{OUTPUT_DIR}/couette_2_dataset_tensor.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset.cpu(), output_path)
    print(f"Generated Couette: {dataset.shape} -> {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate datasets for UQ evaluation')
    parser.add_argument('--datasets', type=str, nargs='+',
                       choices=['original', 'truerandom', 'ood', 'couette', 'all'],
                       default=['all'],
                       help='Which datasets to generate')
    parser.add_argument('--validation-path', type=str, default='Data/Validation',
                       help='Path to Validation data')
    parser.add_argument('--ood-path', type=str, default='Data/OOD',
                       help='Path to OOD data')
    parser.add_argument('--couette-path', type=str, default='Data/Couette',
                       help='Path to Couette data')
    parser.add_argument('--output-dir', type=str, default='dataset',
                       help='Output directory for generated datasets')
    parser.add_argument('--validation-ids', type=str, nargs='+',
                       default=["145922", "121428", "118626", "143128", "118620"],
                       help='Validation file IDs')
    parser.add_argument('--ood-ids', type=str, nargs='+',
                       default=["1921", "1949", "1200", "1445", "2166"],
                       help='OOD file IDs')
    
    args = parser.parse_args()
    
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)
    
    datasets_to_generate = args.datasets
    if 'all' in datasets_to_generate:
        datasets_to_generate = ['original', 'truerandom', 'ood', 'couette']
    
    success = True
    if 'original' in datasets_to_generate:
        success &= generate_original(
            args.validation_path,
            args.validation_ids
        )
    
    if 'truerandom' in datasets_to_generate:
        success &= generate_truerandom_dataset()
    
    if 'ood' in datasets_to_generate:
        success &= generate_ood(args.ood_path, args.ood_ids)
    
    if 'couette' in datasets_to_generate:
        success &= generate_couette(args.couette_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

