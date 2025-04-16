import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# Define the erf activation: erf(sqrt(pi)/2 * x)
def erf_activation(x):
    constant = math.sqrt(math.pi) / 2  # sqrt(pi)/2
    return torch.erf(constant * x)

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, activation):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index,return_latent=False):
        x = self.layers[0](x)
        x = self.activation(x)

        for conv in self.layers[1:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)

        latent = x.clone()
        logits = self.layers[-1](x)
        if return_latent:
            return logits, latent
        else:
            return logits

# Define a simple deep GCN model using the erf activation function.
class InitGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, sigma, activation):
        """
        Args:
            in_channels: Dimension of input features.
            hidden_channels: Dimension of hidden layers.
            out_channels: Number of output classes.
            num_layers: Total number of layers (including first and last).
            sigma: Standard deviation for weight initialization.
            dropout: Dropout probability.
        """
        super(InitGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        # Hidden layers.
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(nn.Linear(hidden_channels, out_channels))
        self.sigma = sigma
        self.activation = activation
        self.hidden_dim = hidden_channels
        self.reset_parameters()

    def reset_parameters(self):
        # Reset parameters and reinitialize weights from N(0, sigma)
        self.convs[0].reset_parameters()
        self.convs[-1].reset_parameters()

        for conv in self.convs[1:-1]:
            conv.reset_parameters()
            if hasattr(conv, 'lin'):
                nn.init.normal_(conv.lin.weight, mean=0, std=self.sigma / math.sqrt(self.hidden_dim))
                if conv.lin.bias is not None:
                    nn.init.zeros_(conv.lin.bias)

        #for bn in self.bns:
         #   bn.reset_parameters()

    def forward(self, x, edge_index, return_latent=False):
        x = self.convs[0](x)
        x = self.activation(x)

        for i,conv in enumerate(self.convs[1:-1]):
            residual = x  # store the input for the skip connection
            x = conv(x, edge_index)
            x = self.activation(x)
            #x = residual + x  # add residual connection

        latent = x.clone()
        logits = self.convs[-1](x)
        if return_latent:
            return logits, latent
        else:
            return logits

class OrthRegGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, activation=erf_activation, gamma=1.0,mu=1e-4):
        """
        A deep GCN with orthogonal weight initialization and an additional orthogonal regularization penalty.

        Args:
            in_channels: Dimension of input features.
            hidden_channels: Dimension of hidden layers.
            out_channels: Number of output classes.
            num_layers: Total number of layers (including first and last).
            sigma: Standard deviation for weight initialization (unused here since we use orthogonal init).
            activation: Activation function.
            dropout: Dropout probability.
        """
        super(OrthRegGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(nn.Linear(hidden_channels, out_channels))
        self.activation = activation
        self.gamma = gamma
        self.mu=mu
        self.reset_parameters()

    def reset_parameters(self):
        # Reset parameters and reinitialize weights using orthogonal initialization.
        self.convs[0].reset_parameters()
        self.convs[-1].reset_parameters()
        for conv in self.convs[1:-1]:
            conv.reset_parameters()
            if hasattr(conv, 'lin'):
                nn.init.orthogonal_(conv.lin.weight)
                conv.lin.weight.data.mul(self.gamma)
                if conv.lin.bias is not None:
                    nn.init.zeros_(conv.lin.bias)
        #for bn in self.bns:
         #   bn.reset_parameters()

    def orth_reg_loss(self):
        """Compute the orthogonal regularization penalty: sum ||WᵀW - I||_F² for each linear layer."""
        reg_loss = 0
        for conv in self.convs[1:-1]:
            if hasattr(conv, 'lin'):
                W = conv.lin.weight  # Shape: [out_features, in_features]
                # Create identity of appropriate size (using in_features dimension)
                I = torch.eye(W.size(1), device=W.device, dtype=W.dtype)
                reg_loss += ((W.t() @ W - (self.gamma**2)*I) ** 2).sum()
        return self.mu * reg_loss

    def forward(self, x, edge_index, return_latent=False):
        x = self.convs[0](x)
        x = self.activation(x)

        # Hidden layers with residual connections.
        for i,conv in enumerate(self.convs[1:-1]):
            residual = x  # store the input for the skip connection
            x = conv(x, edge_index)
            x = self.activation(x)
            #x = residual + x  # add residual connection

        latent = x.clone()
        logits = self.convs[-1](x)
        if return_latent:
            return logits, latent
        else:
            return logits