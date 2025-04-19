from src.config import layer_configs,activations,training_params
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/filangel/mplstyles/raw/main/matlab.mplstyle')
plt.rcParams['figure.dpi'] = 150

def print_best_gammas_mus():
    print("Best parameters for each (num_layers, activation) combination:")
    for nl in sorted(best_params.keys()):
        for act_name, params in best_params[nl].items():
            print(f"Layers: {nl}, Activation: {act_name}  -->  Best gamma: {params['gamma']:.4f}, orth_reg_weight: {params['orth_reg_weight']:.5f}")

def plot_gammas(best_params):# Define the x-axis: the number of layers (keys of the dictionary)
    layers = sorted(best_params.keys())

    # Specify the order of activations (internally still 'function', 'Tanh', 'Identity', 'ReLU')
    activations = ["function", "Tanh", "Identity", "ReLU"]

    plt.figure(figsize=(8, 6))

    # Loop over the activation functions and plot gamma values
    for activation in activations:
        gamma_values = [best_params[layer][activation]["gamma"] for layer in layers]
        # For the legend, replace the label 'function' with 'erf'
        label = "erf" if activation == "function" else activation
        plt.plot(layers, gamma_values, marker='o', label=label)

    plt.xlabel("Number of Layers")
    plt.ylabel("Gamma")
    plt.title("Distribution of Gamma w.r.t. Number of Layers")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_gammas(best_params)
def plot_experiment(model_name, summary_results, layer_configs, activations):
    plt.figure(figsize=(10, 6))
    for act in activations:
        act_name = act.__class__.__name__
        x = []
        y = []
        yerr = []
        # For each activation, get the result for each layer from summary_results
        for nl in sorted(layer_configs):
            mean_acc, std_acc = summary_results[nl][act_name]
            x.append(nl)
            y.append(mean_acc)
            yerr.append(std_acc)
        # Plot the line for this activation function
        plt.plot(x, y, marker='o', label=act_name)
        # Fill the area between mean-std and mean+std
        plt.fill_between(x, 
                        [m - s for m, s in zip(y, yerr)], 
                        [m + s for m, s in zip(y, yerr)], 
                        alpha=0.2)

    plt.xlabel("Number of Layers")
    plt.ylabel("Test Accuracy")
    plt.title(f"{model_name} Test Accuracy vs. Number of Layers (Cora)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/{model_name}_results')
    plt.show()

    # Optionally, print a summary of the results.
    print("\nSummary of Results for Cora:")
    for nl in sorted(summary_results.keys()):
        for act_name in sorted(summary_results[nl].keys()):
            mean_acc, std_acc = summary_results[nl][act_name]
            print(f"  Activation: {act_name}, {nl} layers: {mean_acc:.4f} ± {std_acc:.4f}")


def plot_sigmas_distr(best_sigmas,best_accs,activations,layer_configs):
    # --- Plotting: Best Test Accuracy vs. Number of Layers for each Activation ---
    plt.figure(figsize=(10, 6))
    for act in activations:
        act_name = act.__class__.__name__
        x_layers = []
        y_acc = []
        for nl in sorted(layer_configs):
            x_layers.append(nl)
            y_acc.append(best_accs[nl][act_name])
        plt.plot(x_layers, y_acc, marker='o', label=f"{act_name}")
    plt.xlabel("Number of Layers")
    plt.ylabel("Best Test Accuracy")
    plt.title("Best Test Accuracy vs. Number of Layers (Best Sigma per Activation)")
    plt.legend()
    plt.grid(True)
    plt.ylim(top=0.85,bottom=0.1)  # Set the smallest value on the y-axis to 0.1
    plt.tight_layout()
    plt.savefig(f'figures/best_test_accuracy_best_sigmas')
    plt.show()

    # --- Line Plot: Distribution of Best Sigma Values ---
    plt.figure(figsize=(10, 6))
    for act in activations:
        act_name = act.__class__.__name__
        x_layers = sorted(best_sigmas.keys())
        y_sigma = [best_sigmas[nl][act_name] for nl in x_layers]
        plt.plot(x_layers, y_sigma, marker='o', label=act_name)
    plt.xlabel("Number of Layers")
    plt.ylabel("Best Sigma")
    plt.title("Distribution of Best Sigma Values Across Layers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/best_sigmas_distribution')
    plt.ylim(top=3.0,bottom=0.0)  # Set the smallest value on the y-axis to 0.1
    plt.show()

    print("Best sigma for each (num_layers, activation) combination:")
    for nl in sorted(best_sigmas.keys()):
        for act_name, sigma_val in best_sigmas[nl].items():
            print(f"Layers: {nl}, Activation: {act_name}  -->  Best sigma: {sigma_val}")



def plot_ablation_activation(activation,orthogonal_gamma1_mu0,
                                 orthogonal_BO_gamma_mu0,
                                 orthogonal_gamma1_BO_mu,
                                 orthogonal_BO_gamma_BO_mu):
    """
    Plots the results (mean and uncertainty) as a function of the number of layers
    for a given activation function from all 4 configurations.
    """
    # Combine the 4 dictionaries into one dictionary mapping configuration name to its dictionary.
    config_dicts = {
        "No scaling, no penalty (γ=1, μ=0)": orthogonal_gamma1_mu0,
        "Scaling, no penalty (BO γ, μ=0)": orthogonal_BO_gamma_mu0,
        "No scaling, penalty (γ=1, BO μ)": orthogonal_gamma1_BO_mu,
        "OrthRegGCN (BO for γ,μ)": orthogonal_BO_gamma_BO_mu
    }

    plt.figure(figsize=(8, 6))

    # Iterate over each configuration and plot its line and error band.
    for label, data in config_dicts.items():
        layers = sorted(data.keys())  # list of layer numbers
        # Extract means and uncertainties for the given activation function
        means = [data[layer][activation][0] for layer in layers]
        errors = [data[layer][activation][1] for layer in layers]
        x = np.array(layers)
        y = np.array(means)
        err = np.array(errors)

        # Plot the line with markers
        plt.plot(x, y, marker='o', label=label)
        # Shade the area between (mean - error) and (mean + error)
        plt.fill_between(x, y - err, y + err, alpha=0.2)

    plt.xlabel("Number of Layers")
    plt.ylabel("Test Accuracy")
    activation_name = activation if activation != "function" else "erf"
    plt.title(f"Results for Activation: {activation_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figures/ablation_{activation_name}')
    plt.ylim(top=0.85,bottom=0.1)  # Set the smallest value on the y-axis to 0.1
    plt.show()


def plot_ablation_all_activations(orthogonal_gamma1_mu0,
                                 orthogonal_BO_gamma_mu0,
                                 orthogonal_gamma1_BO_mu,
                                 orthogonal_BO_gamma_BO_mu):
    """
    Loops over all activation functions and creates a separate plot for each one.
    """
    activations = ["function", "Tanh", "ReLU", "Identity"]
    for act in activations:
        plot_ablation_activation(act,orthogonal_gamma1_mu0,
                                 orthogonal_BO_gamma_mu0,
                                 orthogonal_gamma1_BO_mu,
                                 orthogonal_BO_gamma_BO_mu)
