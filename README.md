#Flow-UQ-Thesis


## Overview
Master thesis repository on incorporating multiple (primarily epistemic) uncertainty quantification methods into an existing neural-network surrogate (AE‑RNN) for a multiscale coupled MD–CFD flow prediction model, including trained weights, some helpful scripts for data generation, wrappers and visualization, and thesis experiment results.
The baseline AE‑RNN architecture (model.py) into which the methods were build in is based on a paper from (Jamarz et al. 2023: https://doi.org/10.1007/978-3-031-36027-5_42)

## Repository structure

- `config/` – MaMiCo config files for the simulation data generation of the training data for the kvs and couette flow scenarios used here for the training and evaluation 
- `src/` – Source code for the raw data preprocessing, models, training, UQ methods run scripts, evaluation, and visualization scripts.
- `dataset/` – .npy files of the datasets used for evaluation
- `models/` – Trained model weights used to generate the results
- `results/` – Saved .npy results and figures generated through the vis scripts
- `docs/` – Additional documentation


## Setup

To reproduce the results either 1) the results can be directly used and visualized with the src/vis scripts, the model weights and method run scripts in models/ and src/run can be used to regenerate the methods results, or the full training can be done by running the src/training scripts.

