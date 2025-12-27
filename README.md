#Flow-UQ-Thesis


## Overview

This is the repository containing weights, scripts and results from my master thesis on incorporating multiple (mostly epistemic) uncertainty quantification (UQ) methods in a neural network surrogate for a multiscale coupled MD-CFD flow prediction model.


## Repository structure

- `config/` – MaMiCo config files for the simulation data generation of the training data for the kvs and couette flow scenarios used here for the training and evaluation 
- `src/` – Source code for the raw data preprocessing, models, training, UQ methods run scripts, evaluation, and visualization scripts.
- `dataset/` – .npy files of the datasets used for evaluation
- `models/` – Trained model weights used to generate the results
- `results/` – Saved .npy results and figures generated through the vis scripts
- `docs/` – Additional documentation


## Setup

To reproduce the results either 1) the results can be directly used and visualized with the src/vis scripts, the model weights and method run scripts in models/ and src/run can be used to regenerate the methods results, or the full training can be done by running the src/training scripts.


