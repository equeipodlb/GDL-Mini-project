
from src.train import run_for_results
from src.config import layer_configs,activations

from src.plotting import plot_ablation_all_activations
import pickle

# Dictionary for configuration: orthogonal gamma=1, mu=0
orthogonal_gamma1_mu0 = {
    4: {
        "function": (0.8002, 0.0073),
        "Tanh":     (0.7996, 0.0113),
        "ReLU":     (0.7870, 0.0203),
        "Identity": (0.8060, 0.0037)
    },
    10: {
        "function": (0.7834, 0.0152),
        "Tanh":     (0.7768, 0.0118),
        "ReLU":     (0.7512, 0.0143),
        "Identity": (0.7776, 0.0118)
    },
    20: {
        "function": (0.7774, 0.0110),
        "Tanh":     (0.7782, 0.0104),
        "ReLU":     (0.2622, 0.0969),
        "Identity": (0.7874, 0.0079)
    },
    30: {
        "function": (0.7738, 0.0149),
        "Tanh":     (0.7728, 0.0134),
        "ReLU":     (0.2926, 0.0891),
        "Identity": (0.7642, 0.0109)
    },
    40: {
        "function": (0.7216, 0.0576),
        "Tanh":     (0.7058, 0.0913),
        "ReLU":     (0.2952, 0.0797),
        "Identity": (0.6652, 0.0916)
    },
    50: {
        "function": (0.4366, 0.0463),
        "Tanh":     (0.4638, 0.0903),
        "ReLU":     (0.2240, 0.1045),
        "Identity": (0.6212, 0.0747)
    },
    60: {
        "function": (0.3658, 0.0547),
        "Tanh":     (0.3950, 0.0737),
        "ReLU":     (0.1760, 0.0802),
        "Identity": (0.4582, 0.0725)
    }
}

# Dictionary for configuration: orthogonal BO gamma, mu=0
orthogonal_BO_gamma_mu0 = {
    4: {
        "function": (0.8002, 0.0073),
        "Tanh":     (0.7996, 0.0113),
        "ReLU":     (0.7870, 0.0203),
        "Identity": (0.8060, 0.0037)
    },
    10: {
        "function": (0.7834, 0.0152),
        "Tanh":     (0.7768, 0.0118),
        "ReLU":     (0.7476, 0.0207),
        "Identity": (0.7776, 0.0118)
    },
    20: {
        "function": (0.7774, 0.0110),
        "Tanh":     (0.7782, 0.0104),
        "ReLU":     (0.2698, 0.0942),
        "Identity": (0.7874, 0.0079)
    },
    30: {
        "function": (0.7738, 0.0149),
        "Tanh":     (0.7728, 0.0134),
        "ReLU":     (0.2986, 0.0906),
        "Identity": (0.7646, 0.0115)
    },
    40: {
        "function": (0.7460, 0.0246),
        "Tanh":     (0.7036, 0.0900),
        "ReLU":     (0.2834, 0.0738),
        "Identity": (0.6660, 0.0925)
    },
    50: {
        "function": (0.4532, 0.0580),
        "Tanh":     (0.4612, 0.0979),
        "ReLU":     (0.2302, 0.1038),
        "Identity": (0.6184, 0.0734)
    },
    60: {
        "function": (0.3506, 0.0612),
        "Tanh":     (0.3722, 0.0527),
        "ReLU":     (0.1772, 0.0795),
        "Identity": (0.4714, 0.0874)
    }
}

# Dictionary for configuration: orthogonal gamma=1, BO mu
orthogonal_gamma1_BO_mu = {
    4: {
        "function": (0.7988, 0.0062),
        "Tanh":     (0.7994, 0.0015),
        "ReLU":     (0.8028, 0.0041),
        "Identity": (0.8056, 0.0035)
    },
    10: {
        "function": (0.7936, 0.0102),
        "Tanh":     (0.7940, 0.0111),
        "ReLU":     (0.7876, 0.0040),
        "Identity": (0.7950, 0.0091)
    },
    20: {
        "function": (0.7884, 0.0072),
        "Tanh":     (0.7902, 0.0087),
        "ReLU":     (0.2948, 0.2534),
        "Identity": (0.7684, 0.0086)
    },
    30: {
        "function": (0.7776, 0.0130),
        "Tanh":     (0.7878, 0.0056),
        "ReLU":     (0.1822, 0.0791),
        "Identity": (0.7582, 0.0086)
    },
    40: {
        "function": (0.7526, 0.0188),
        "Tanh":     (0.7466, 0.0239),
        "ReLU":     (0.1994, 0.0606),
        "Identity": (0.7398, 0.0227)
    },
    50: {
        "function": (0.7314, 0.0064),
        "Tanh":     (0.7430, 0.0311),
        "ReLU":     (0.1820, 0.0768),
        "Identity": (0.5226, 0.0695)
    },
    60: {
        "function": (0.7460, 0.0265),
        "Tanh":     (0.7200, 0.0149),
        "ReLU":     (0.1896, 0.0758),
        "Identity": (0.5062, 0.0998)
    }
}

# Dictionary for configuration: BO gamma, BO mu
orthogonal_BO_gamma_BO_mu = {
    4: {'Identity': (0.8068, 0.0055),
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
      'function': (0.7918, 0.0088)}
}


def main():
    """params1 = {
    nl: {
        act.__class__.__name__: {'gamma': 1.0, 'orth_reg_weight': 0}
        for act in activations
    }
    for nl in layer_configs
    }
    params2 = {
    nl: {
        act.__class__.__name__: {'gamma': best_params[nl][act.__class__.__name__], 'orth_reg_weight': 0}
        for act in activations
    }
    for nl in layer_configs
    }
    params3 = {
    nl: {
        act.__class__.__name__: {'gamma': 1.0, 'orth_reg_weight': best_params[nl][act.__class__.__name__]}
        for act in activations
    }
    for nl in layer_configs
    }

    orthogonal_gamma1_mu0 = run_for_results('orth_reg',
                                            "no-gamma-no-mu",
                                            dataset_name='Cora',
                                            best_orth_params=params1,
                                            num_runs=5)
    
    orthogonal_BO_gamma_mu0 = run_for_results('orth_reg',
                                             "ablation2",
                                             dataset_name='Cora',
                                             best_orth_params=params2,
                                             num_runs=5)
    orthogonal_gamma1_BO_mu = run_for_results('orth_reg',
                                             "ablation3",
                                             dataset_name='Cora',
                                             best_orth_params=params3,
                                             num_runs=5)"""
    with open("./results/ablation1_results_Cora.pkl","rb") as f:
        orthogonal_gamma1_mu0 = pickle.load(f)

    with open("./results/ablation2_results_Cora.pkl","rb") as f:
        orthogonal_BO_gamma_mu0 = pickle.load(f)

    with open("./results/ablation3_results_Cora.pkl","rb") as f:
        orthogonal_gamma1_BO_mu = pickle.load(f)

    with open("./results/orth_reg_results_Cora.pkl") as f:
        orthogonal_BO_gamma_BO_mu = pickle.load(f)

    plot_ablation_all_activations(orthogonal_gamma1_mu0,
                                 orthogonal_BO_gamma_mu0,
                                 orthogonal_gamma1_BO_mu,
                                 orthogonal_BO_gamma_BO_mu)

if __name__ == '__main__':
    main()