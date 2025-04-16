from src.models import erf_activation
import torch
import torch.nn as nn

seed = 42
layer_configs = [4,10,20,30,40,50,60]
model_types = ['baseline','init_gcn','orth_reg','orth_no_penalty']
training_params = {
    'hidden_channels': 128,
    'max_epochs': 151,
    'patience': 200,
    'lr': 0.001
}
activations = [erf_activation,nn.Tanh(),nn.ReLU(),nn.Identity()]