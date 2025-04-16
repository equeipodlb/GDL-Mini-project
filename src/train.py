import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from src.models import GCN, InitGCN, OrthRegGCN
from src.utils import get_device

from src.config import layer_configs,activations,training_params

def train(model, model_type, data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    #data.train_mask = fix_mask(data.train_mask)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    # For orth_reg model, add the regularization term.
    if model_type == 'orth_reg':
        loss = loss + model.orth_reg_loss()
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function: returns [train_acc, val_acc, test_acc].
def test(model, data, device):
    model.eval()
    data_x = data.x.to(device)
    data_edge_index = data.edge_index.to(device)
    data_y = data.y.to(device)
    logits = model(data_x, data_edge_index)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        #mask = fix_mask(mask)
        mask = mask.to(device)
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data_y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def dirichlet_energy_from_data(x, data):
    # x: latent representation of shape [N, d]
    # data.edge_index: tensor of shape [2, num_edges]
    row, col = data.edge_index
    diff = x[row] - x[col]
    # Sum over all edges and divide by 2 (if each edge is counted twice).
    energy = (diff ** 2).sum() / 2.0
    return energy.item()

# A unified experiment function that accepts a dataset name.
def run_experiment_dataset(model_type='baseline', dataset_name='Cora', sigma=0.155,
                             num_layers=4, hidden_channels=128, max_epochs=150, gamma=1.0,
                             patience=150, lr=0.001, weight_decay=0, orth_reg_weight=1e-4,
                             activation=nn.ReLU(),device='cpu'):
    
    device = get_device()
    if dataset_name == 'Texas':
      dataset = WebKB(root='data/WebKB', name='Texas',transform=T.GCNNorm())
    else:
      dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name, transform=T.GCNNorm())
    data = dataset[0]

    # Select the model variant.
    if model_type == 'baseline':
        model = GCN(in_channels=dataset.num_node_features,
                        hidden_channels=hidden_channels,
                        out_channels=dataset.num_classes,
                        num_layers=num_layers, activation=activation).to(device)
    elif model_type == 'init_gcn':
        model = InitGCN(in_channels=dataset.num_node_features,
                        hidden_channels=hidden_channels,
                        out_channels=dataset.num_classes,
                        num_layers=num_layers, activation=activation,
                        sigma=sigma).to(device)
    elif model_type == 'orth_reg':
        model = OrthRegGCN(in_channels=dataset.num_node_features,
                               hidden_channels=hidden_channels,
                               out_channels=dataset.num_classes, gamma=gamma,mu=orth_reg_weight,
                               num_layers=num_layers, activation=activation).to(device)
    else:
        raise ValueError("Unknown model_type. Choose 'baseline', 'init_gcn' or 'orth_reg'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5)

    best_val_acc = 0
    best_epoch = 0
    test_acc_at_best_val = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        loss = train(model, model_type, data, optimizer, device)
        train_acc, val_acc, test_acc = test(model, data, device)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            test_acc_at_best_val = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 10 == 0:
          print(f"[{dataset_name} | {model_type} | Layers: {num_layers}] Epoch: {epoch:03d}, Loss: {loss:.4f}, "
              f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch} with val acc {best_val_acc:.4f})")
            break

    model.eval()
    with torch.no_grad():
        _, latent = model(data.x.to(device), data.edge_index.to(device), return_latent=True)
    energy = dirichlet_energy_from_data(latent, data)
    return test_acc_at_best_val,energy

def run_for_results(model_type, model_name, dataset_name = "Cora", best_sigmas={}, best_orth_params={}, num_runs=5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dictionary to store raw results.
    # Structure: results[activation_name][num_layers] = list of test accuracies from num_runs.
    results = {}

    # Loop over each activation function.
    for act in activations:
        act_name = act.__class__.__name__
        results[act_name] = {}
        for nl in layer_configs:
            results[act_name][nl] = []
            for run in range(num_runs):
                print(f"Running experiment on {dataset_name} | Activation: {act_name} | Layers: {nl} | Run {run+1}/{num_runs}")
                sigma = best_sigmas[nl][act_name] if model_type == 'init_gcn' else None
                gamma = best_orth_params[nl][act_name]['gamma'] if model_type in ['orth_reg', 'orth_no_penalty'] else 1
                orth_reg_weight = best_orth_params[nl][act_name]['orth_reg_weight'] if model_type == 'orth_reg' else None

                test_acc,_ = run_experiment_dataset(
                    model_type=model_type,
                    dataset_name=dataset_name,
                    num_layers=nl,
                    hidden_channels=training_params["hidden_channels"],
                    max_epochs=training_params["max_epochs"],
                    patience=training_params["patience"],
                    lr=training_params["lr"],
                    weight_decay=0,
                    activation=act,
                    sigma=sigma,
                    gamma=gamma,
                    orth_reg_weight=orth_reg_weight,
                    device=device
                )
                results[act_name][nl].append(test_acc)

    # Compute summary statistics (mean and std) for each combination.
    summary_results = {}  # summary_results[activation_name][num_layers] = (mean, std)
    for act_name in results:
        summary_results[act_name] = {}
        for nl in results[act_name]:
            accs = np.array(results[act_name][nl])
            mean_acc = np.mean(accs)
            std_acc = np.std(accs, ddof=1)
            summary_results[act_name][nl] = (mean_acc, std_acc)

    # Save the summary results to a file.
    with open(f"./{model_name}_results_Cora.pkl", "wb") as f:
        pickle.dump(summary_results, f)
    print(f"Summary results saved to {model_name}_results_Cora.pkl.")
    return summary_results

