import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv
torch.manual_seed(0)

# Define a Graph Convolutional Network (GCN) using GENConv layers
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.conv1 = GENConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GENConv(hidden_channels, out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None, training = True):
        x = F.dropout(x, p = 0.1, training = training)
        x = self.conv1(x, edge_index, edge_attr)  # First convolution
        x = F.relu(x)  # Apply ReLU activation
        x = F.dropout(x, p = 0.1, training = training)
        x = self.conv2(x, edge_index, edge_attr)  # Second convolution
        return x

# Define a NormalGCN for node classification
class NormalGCN(nn.Module):
    def __init__(self, input_dim, output_dim, edge_feature_dim, k, device, hidden1=16, hidden2=16, temperature=1.0):
        """
        NormalGCN for node classification.

        Args:
        - input_dim (int): Node feature size.
        - output_dim (int): Number of classes.
        - edge_feature_dim (int): Edge feature dimension.
        - k (int): Number of top edges to keep.
        - hidden1, hidden2 (int): Hidden layer sizes.
        - weight_decay (float): L2 regularization.
        - temperature (float): Normal-Softmax temperature.
        """
        super(NormalGCN, self).__init__()

        self.k = k
        self.temperature = temperature
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # Edge feature transformation layer
        self.MLP = nn.Linear(edge_feature_dim + 2 * input_dim, 1)

        # GCN layers
        self.conv = GCN(input_dim, hidden1, hidden2, edge_feature_dim)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden2, output_dim)

    def forward(self, num_nodes, edge_index, edge_attr, x, node_mask, training = True):
        """
        Forward pass for NormalGCN.

        Args:
        - num_nodes (int): Number of nodes.
        - edge_index (Tensor): Edge indices.
        - edge_attr (Tensor): Edge attributes.
        - x (Tensor): Node features.
        - node_mask (Tensor): Node mask.
        - training (bool): Training mode.
        """
        # Apply GCN layers

        x = self.conv(x, edge_index, edge_attr, training = training)
        x = F.relu(x)
        x = self.fc(x)
        x = x[node_mask!=0]

        return x
