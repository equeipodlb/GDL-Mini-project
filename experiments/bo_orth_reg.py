from src.bayesian_opt import bo_OrthRegGCN

## BEST PARAMS
"""best_params = {
    4: {
        "function": {"gamma": 1.338, "orth_reg_weight": 0.06422},
        "Tanh": {"gamma": 1.588, "orth_reg_weight": 0.08840},
        "ReLU": {"gamma": 0.126, "orth_reg_weight": 0.08662},
        "Identity": {"gamma": 1.759, "orth_reg_weight": 0.00079},
    },
    10: {
        "function": {"gamma": 1.667, "orth_reg_weight": 0.02124},
        "Tanh": {"gamma": 1.353, "orth_reg_weight": 0.05753},
        "ReLU": {"gamma": 1.667, "orth_reg_weight": 0.02124},
        "Identity": {"gamma": 1.465, "orth_reg_weight": 0.06031},
    },
    20: {
        "function": {"gamma": 1.806, "orth_reg_weight": 0.00061},
        "Tanh": {"gamma": 1.316, "orth_reg_weight": 0.00015},
        "ReLU": {"gamma": 1.583, "orth_reg_weight": 0.06504},
        "Identity": {"gamma": 1.864, "orth_reg_weight": 0.09743},
    },
    30: {
        "function": {"gamma": 1.329, "orth_reg_weight": 0.09774},
        "Tanh": {"gamma": 1.667, "orth_reg_weight": 0.02124},
        "ReLU": {"gamma": 1.349, "orth_reg_weight": 0.05309},
        "Identity": {"gamma": 1.206, "orth_reg_weight": 0.07081},
    },
    40: {
        "function": {"gamma": 1.695, "orth_reg_weight": 0.09975},
        "Tanh": {"gamma": 1.210, "orth_reg_weight": 0.07511},
        "ReLU": {"gamma": 1.911, "orth_reg_weight": 0.02615},
        "Identity": {"gamma": 1.910, "orth_reg_weight": 0.09944},
    },
    50: {
        "function": {"gamma": 1.905, "orth_reg_weight": 0.09949},
        "Tanh": {"gamma": 1.896, "orth_reg_weight": 0.09991},
        "ReLU": {"gamma": 1.911, "orth_reg_weight": 0.02615},
        "Identity": {"gamma": 1.122, "orth_reg_weight": 0.09702},
    },
    60: {
        "function": {"gamma": 1.467, "orth_reg_weight": 0.05987},
        "Tanh": {"gamma": 1.718, "orth_reg_weight": 0.09851},
        "ReLU": {"gamma": 0.755, "orth_reg_weight": 0.09507},
        "Identity": {"gamma": 1.787, "orth_reg_weight": 0.00124},
    },
}"""

def main():
    best_params = bo_OrthRegGCN()
    print_best_gammas_mus(best_params)
    plot_gammas(best_params)

if __name__ == '__main__':
    main()

