import pickle
from src.config import layer_configs,activations

from src.plotting import plot_sigmas_distr
from src.bayesian_opt import bo_sigmas

"""


best_params = {
    4: {
        "function": {"sigma": 1.8030, "test_accuracy": 0.8100},
        "Tanh": {"sigma": 1.7985, "test_accuracy": 0.8160},
        "ReLU": {"sigma": 1.7950, "test_accuracy": 0.8180},
        "Identity": {"sigma": 2.1970, "test_accuracy": 0.8230},
    },
    10: {
        "function": {"sigma": 1.8043, "test_accuracy": 0.8010},
        "Tanh": {"sigma": 1.1222, "test_accuracy": 0.8070},
        "ReLU": {"sigma": 1.8097, "test_accuracy": 0.7950},
        "Identity": {"sigma": 2.1932, "test_accuracy": 0.8030},
    },
    20: {
        "function": {"sigma": 1.1242, "test_accuracy": 0.7980},
        "Tanh": {"sigma": 1.1251, "test_accuracy": 0.7850},
        "ReLU": {"sigma": 1.7671, "test_accuracy": 0.7920},
        "Identity": {"sigma": 1.1242, "test_accuracy": 0.7910},
    },
    30: {
        "function": {"sigma": 1.7955, "test_accuracy": 0.7890},
        "Tanh": {"sigma": 1.8036, "test_accuracy": 0.7750},
        "ReLU": {"sigma": 1.6941, "test_accuracy": 0.7720},
        "Identity": {"sigma": 1.8039498, "test_accuracy": 0.7850},
    },
    40: {
        "function": {"sigma": 1.8050601, "test_accuracy": 0.7550},
        "Tanh": {"sigma": 2.1962498, "test_accuracy": 0.7400},
        "ReLU": {"sigma": 1.6254202, "test_accuracy": 0.7510},
        "Identity": {"sigma": 1.0712572, "test_accuracy": 0.7760},
    },
    50: {
        "function": {"sigma": 1.7963768, "test_accuracy": 0.7390},
        "Tanh": {"sigma": 1.7967506, "test_accuracy": 0.7370},
        "ReLU": {"sigma": 1.7963768, "test_accuracy": 0.6910},
        "Identity": {"sigma": 2.1245097, "test_accuracy": 0.7720},
    },
    60: {
        "function": {"sigma": 1.7963768, "test_accuracy": 0.7360},
        "Tanh": {"sigma": 2.1033186, "test_accuracy": 0.6840},
        "ReLU": {"sigma": 1.7916745, "test_accuracy": 0.6970},
        "Identity": {"sigma": 2.1259761, "test_accuracy": 0.7280},
    },
}

"""

def main():

    #best_params = bo_sigmas()
    
    with open("./results/bo_sigma_results.pkl","rb") as f:
        best_params = pickle.load(f)
    best_sigmas = {nl: {act: info['sigma'] for act, info in best_params[nl].items()} for nl in best_params}
    best_accs = {nl: {act: info['test_accuracy'] for act, info in best_params[nl].items()} for nl in best_params}
    
    print(best_params)
    plot_sigmas_distr(best_sigmas,best_accs,activations=activations,layer_configs=layer_configs)

if __name__ == '__main__':
    main()