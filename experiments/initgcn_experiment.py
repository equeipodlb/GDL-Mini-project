from src.train import run_for_results
from src.plotting import plot_experiment

from src.config import activations,layer_configs
import pickle

# EXPERIMENT RESULTS
results = {
    4: {
        "Identity": (0.7994, 0.0116),
        "ReLU": (0.7936, 0.0097),
        "Tanh": (0.7918, 0.0163),
        "function": (0.7914, 0.0092),
    },
    10: {
        "Identity": (0.7704, 0.0184),
        "ReLU": (0.7766, 0.0128),
        "Tanh": (0.7898, 0.0166),
        "function": (0.7822, 0.0128),
    },
    20: {
        "Identity": (0.7778, 0.0119),
        "ReLU": (0.7596, 0.0208),
        "Tanh": (0.7748, 0.0268),
        "function": (0.7766, 0.0165),
    },
    30: {
        "Identity": (0.7600, 0.0264),
        "ReLU": (0.7380, 0.0253),
        "Tanh": (0.7490, 0.0242),
        "function": (0.7518, 0.0140),
    },
    40: {
        "Identity": (0.7620, 0.0120),
        "ReLU": (0.7016, 0.0169),
        "Tanh": (0.7070, 0.0177),
        "function": (0.7128, 0.0362),
    },
    50: {
        "Identity": (0.7198, 0.0215),
        "ReLU": (0.6242, 0.0579),
        "Tanh": (0.5966, 0.1501),
        "function": (0.6608, 0.0866),
    },
    60: {
        "Identity": (0.6392, 0.0550),
        "ReLU": (0.6070, 0.0824),
        "Tanh": (0.6606, 0.0290),
        "function": (0.6262, 0.0520),
    },
}

def main():

    # Run experiment for baseline model on the "Cora" dataset for 5 runs.
    """
    with open("./results/bo_sigma_results.pkl","rb") as f:
        best_params = pickle.load(f)
    best_sigmas = {nl: {act: info['sigma'] for act, info in best_params[nl].items()} for nl in best_params}
    best_accs = {nl: {act: info['test_accuracy'] for act, info in best_params[nl].items()} for nl in best_params}
    results = run_for_results('init_gcn', 'InitGCN', dataset_name='Cora', best_sigmas=best_sigmas,num_runs=5)
    """
    #Or load previous results
    with open("./results/InitGCN_results_Cora.pkl", "rb") as f:
        results = pickle.load(f)

    print("InitGCN Experiment Results:")
    print(results)
    # Plot the experiment results.
    plot_experiment('InitGCN', results,layer_configs,activations)

if __name__ == '__main__':
    main()