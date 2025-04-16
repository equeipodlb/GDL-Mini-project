
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bayes_opt import BayesianOptimization
from src.train import run_experiment_dataset

from src.utils import set_seed,get_device
from src.config import training_params,layer_configs,activations, seed

import pickle

def bo_sigmas(dataset_name='Cora'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dictionary to store the best sigma and best accuracy for each combination.
    best_params = {}

    for nl in layer_configs:
        best_params[nl] = {}
        for act in activations:
            act_name = act.__class__.__name__
            best_params[nl][act_name] = {}
            print(f"\n=== Running BO for {nl} layers, activation: {act_name} ===")

            # Define the objective function for the current combination.
            def objective_sigma(sigma):
                sigma_val = float(sigma)
                # For model_type 'init_gcn', pass sigma.
                acc,energy = run_experiment_dataset(
                    model_type='init_gcn',
                    sigma=sigma_val,
                    weight_decay=0,
                    num_layers=nl,
                    hidden_channels=training_params["hidden_channels"],
                    max_epochs=training_params["max_epochs"],
                    patience=training_params["patience"],
                    lr=training_params["lr"],
                    activation=act,
                    device=device
                )
                print(f"[{nl} layers, {act_name}] sigma={sigma_val:.3f} -> Test Accuracy={acc:.4f}, Energy:{energy:.4f}")
                return acc

            pbounds = {'sigma': (0.001, 3.0)}
            optimizer = BayesianOptimization(
                f=objective_sigma,
                pbounds=pbounds,
                random_state=seed,
            )
            optimizer.maximize(init_points=10, n_iter=15)

            # Extract best sigma and best accuracy.
            best_sigma = optimizer.max['params']['sigma']
            best_acc = optimizer.max['target']
            best_params[nl][act_name]['sigma'] = best_sigma
            best_params[nl][act_name]['test_accuracy'] = best_acc
            print(f"Best for {nl} layers, {act_name}: sigma = {best_sigma:.7f}, Test Accuracy = {best_acc:.4f}")


    for nl in best_params:
        for act_name in best_params[nl]:
            print(f"Layers: {nl}, Activation: {act_name} --> Best sigma: {best_params[nl][act_name]['sigma']:.4f}, Best Acc: {best_accs[nl][act_name]['test_accuracy']:.4f}")

    with open("bo_sigma_results.pkl", "wb") as f:
        pickle.dump(best_params, f)

    print("Best sigma results saved to bo_sigma_results.pkl.")
    return best_params


def bo_OrthRegGCN(dataset_name='Cora'):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dictionaries to store best parameters and best accuracy for each combination.
    # best_params[num_layers][activation_name] = {'gamma': best_gamma, 'orth_reg_weight': best_orth_reg_weight, 'acc': best_acc}
    best_params = {}
    best_accs = {}
    # Dictionary to store full BO results for each configuration.
    results_by_layers = {}

    for nl in layer_configs:
        best_params[nl] = {}
        best_accs[nl] = {}
        results_by_layers[nl] = {}
        for act in activations:
            act_name = act.__class__.__name__
            print(f"\n=== Running BO for {nl} layers, activation: {act_name} ===")

            # Define the objective function for the current combination.
            def objective_gamma(gamma, orth_reg_weight):
                gamma_val = float(gamma)
                orth_reg_val = float(orth_reg_weight)
                # Assume run_experiment_dataset returns (test_accuracy, dirichlet_energy)
                acc,energy = run_experiment_dataset(
                    model_type='orth_reg',
                    gamma=gamma_val,
                    num_layers=nl,
                    hidden_channels=training_params["hidden_channels"],
                    max_epochs=training_params["max_epochs"],
                    patience=training_params["patience"],
                    lr=training_params["lr"],
                    weight_decay=0,
                    orth_reg_weight=orth_reg_val,
                    dataset_name=dataset_name,
                    activation=act,
                    device=device
                )
                print(f"[{nl} layers, {act_name}] gamma={gamma_val:.3f}, orth_reg_weight={orth_reg_val:.5f} -> Test Accuracy={acc:.4f}, energy: {energy:.4f}")
                return acc

            pbounds = {'gamma': (0.01, 2.0), 'orth_reg_weight': (1e-5, 1e-1)}
            optimizer = BayesianOptimization(
                f=objective_gamma,
                pbounds=pbounds,
                random_state=seed,
            )
            optimizer.maximize(init_points=10, n_iter=15)

            # Extract best parameters and best accuracy.
            best_parameters = optimizer.max['params']
            best_gamma = best_parameters['gamma']
            best_orth_reg_weight = best_parameters['orth_reg_weight']
            best_acc = optimizer.max['target']
            best_params[nl][act_name] = {
                'gamma': best_gamma,
                'orth_reg_weight': best_orth_reg_weight,
                'acc': best_acc
            }
            best_accs[nl][act_name] = best_acc
            results_by_layers.setdefault(nl, {})[act_name] = optimizer.res
            print(f"Best for {nl} layers, {act_name}: gamma = {best_gamma:.3f}, orth_reg_weight = {best_orth_reg_weight:.5f}, Test Accuracy = {best_acc:.4f}")

    
    with open("./results/bo_results_orth.pkl", "wb") as f:
        pickle.dump(best_params, f)
    print("Best results saved to bo_results_orth.pkl.")

    return best_params

