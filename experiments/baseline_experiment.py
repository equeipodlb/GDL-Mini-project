from src.train import run_for_results
from src.plotting import plot_experiment

from src.config import activations,layer_configs
import pickle

results = {
    4: {
        "function": (0.7922, 0.0072),
        "Tanh": (0.7902, 0.0170),
        "ReLU": (0.7682, 0.0074),
        "Identity": (0.7846, 0.0107)
    },
    10: {
        "function": (0.7738, 0.0259),
        "Tanh": (0.7700, 0.0130),
        "ReLU": (0.7170, 0.0142),
        "Identity": (0.7626, 0.0280)
    },
    20: {
        "function": (0.7742, 0.0178),
        "Tanh": (0.7622, 0.0114),
        "ReLU": (0.3744, 0.0506),
        "Identity": (0.7520, 0.0118)
    },
    30: {
        "function": (0.6908, 0.0547),
        "Tanh": (0.6428, 0.1771),
        "ReLU": (0.3292, 0.0420),
        "Identity": (0.7518, 0.0086)
    },
    40: {
        "function": (0.5008, 0.1223),
        "Tanh": (0.3996, 0.1804),
        "ReLU": (0.3316, 0.0362),
        "Identity": (0.7116, 0.0388)
    },
    50: {
        "function": (0.3254, 0.0366),
        "Tanh": (0.3164, 0.0032),
        "ReLU": (0.2630, 0.1298),
        "Identity": (0.5392, 0.1789)
    },
    60: {
        "function": (0.3362, 0.0370),
        "Tanh": (0.3184, 0.0146),
        "ReLU": (0.1438, 0.0078),
        "Identity": (0.3646, 0.0447)
    }
}

    

def main():
    
    # Run experiment for baseline model on the "Cora" dataset for 5 runs.
    #results = run_for_results('baseline', 'baseline', dataset_name='Cora', num_runs=5)
    
    #Or load previous results:
    with open("./results/baseline_results_Cora.pkl", "rb") as f:
        results = pickle.load(f)
    
    print("Baseline Experiment Results:")
    print(results)
    # Plot the experiment results.
    plot_experiment('baseline', results,layer_configs,activations)

if __name__ == '__main__':
    main()