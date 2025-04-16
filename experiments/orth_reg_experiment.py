from src.train import run_for_results
from src.plotting import plot_experiment

from src.config import activations,layer_configs
import pickle

# EXPERIMENT RESULTS
"""results = {4: {'Identity': (0.8068, 0.0055),
     'ReLU': (0.7532, 0.043),
     'Tanh': (0.8046, 0.003),
     'function': (0.8038, 0.0051)},
 10: {'Identity': (0.8046, 0.0048),
      'ReLU': (0.787, 0.0076),
      'Tanh': (0.7964, 0.0067),
      'function': (0.8054, 0.0046)},
 20: {'Identity': (0.7896, 0.003),
      'ReLU': (0.4576, 0.2698),
      'Tanh': (0.7834, 0.0146),
      'function': (0.7876, 0.0149)},
 30: {'Identity': (0.7864, 0.0106),
      'ReLU': (0.2772, 0.1999),
      'Tanh': (0.7896, 0.0045),
      'function': (0.7898, 0.0078)},
 40: {'Identity': (0.7808, 0.0111),
      'ReLU': (0.176, 0.0808),
      'Tanh': (0.792, 0.0065),
      'function': (0.7918, 0.0126)},
 50: {'Identity': (0.7766, 0.0084),
      'ReLU': (0.265, 0.2126),
      'Tanh': (0.7886, 0.0067),
      'function': (0.7852, 0.0112)},
 60: {'Identity': (0.7456, 0.0247),
      'ReLU': (0.1754, 0.0805),
      'Tanh': (0.7796, 0.0151),
      'function': (0.7918, 0.0088)}}"""

def main():

    # Run experiment for baseline model on the "Cora" dataset for 5 runs.
    """
    with open("./results/bo_best_params_orth.pkl", "rb") as f:
        best_params = pickle.load(f)
    results = run_for_results('orth_reg', 'orth_reg', dataset_name='Cora', best_orth_params=best_params,num_runs=5)
    """
    #Or load previous results
    with open("./results/orth_reg_results_Cora.pkl", "rb") as f:
        results = pickle.load(f)

    print("OrthRegGCN Experiment Results:")
    print(results)
    # Plot the experiment results.
    plot_experiment('OrthRegGCN', results,layer_configs,activations)

if __name__ == '__main__':
    main()