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

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)  # First convolution
        x = F.relu(x)  # Apply ReLU activation
        x = self.conv2(x, edge_index, edge_attr)  # Second convolution
        return x

# Define a GumbelGCN for node classification
class GumbelGCN(nn.Module):
    def __init__(self, input_dim, output_dim, edge_feature_dim, k, device, hidden1=16, hidden2=16, temperature=1.0):
        """
        GumbelGCN for node classification.

        Args:
        - input_dim (int): Node feature size.
        - output_dim (int): Number of classes.
        - edge_feature_dim (int): Edge feature dimension.
        - k (int): Number of top edges to keep.
        - hidden1, hidden2 (int): Hidden layer sizes.
        - weight_decay (float): L2 regularization.
        - temperature (float): Gumbel-Softmax temperature.
        """
        super(GumbelGCN, self).__init__()

        self.k = k
        self.temperature = temperature
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_feature_dim = edge_feature_dim
        self.device = device

        # Edge feature transformation layer
        self.MLP = nn.Linear(edge_feature_dim + 2 * input_dim, 1)

        # GCN layers
        self.conv = GCN(input_dim, hidden1, hidden2, edge_feature_dim)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden2, output_dim)

    def sample_gumbel(self, shape, eps=1e-20):
        """Samples from a Gumbel distribution."""
        U = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax(self, logits, training=True):
        """Applies the Gumbel-Softmax trick."""
        noise = self.sample_gumbel(logits.shape) if training else 0
        return F.softmax((logits + noise) / self.temperature, dim=-1)

    def forward(self, num_nodes, edge_index, edge_attr, x, node_mask, training=True):
        """
        Forward pass for GumbelGCN.

        Args:
        - num_nodes (int): Number of nodes.
        - edge_index (Tensor): Edge indices.
        - edge_attr (Tensor): Edge attributes.
        - x (Tensor): Node features.
        - node_mask (Tensor): Node mask.
        - training (bool): Training mode.
        """
        if training:
            # Create adjacency matrix
            adj_batch = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), device=self.device), (num_nodes, num_nodes), device=self.device).to_dense()

            # Create node embeddings
            node_embedding = x.unsqueeze(-1).repeat(1, 1, num_nodes)

            # Create neighbor embeddings
            neighbor_embedding = torch.zeros((num_nodes, self.input_dim, num_nodes), device=self.device)
            for u in range(num_nodes):
                neighbors = edge_index[1, edge_index[0] == u]
                for v in neighbors:
                    neighbor_embedding[u, :, v] = x[v]

            # Create edge embeddings
            edge_embedding = torch.zeros((num_nodes, edge_attr.size(1), num_nodes), device=self.device)
            for u in range(num_nodes):
                neighbors = edge_index[1, edge_index[0] == u]
                for v in neighbors:
                    mask = (edge_index[0] == u) & (edge_index[1] == v)
                    edge_embedding[u, :, v] = edge_attr[mask].squeeze()

            # Concatenate all features
            all_feats = torch.cat([node_embedding, neighbor_embedding, edge_embedding], dim=1).transpose(1, 2)

            # Compute scores using MLP
            score = self.MLP(all_feats).squeeze()
            score[adj_batch == 0] = -1e9  # Mask non-adjacent nodes
            z = F.softmax(score, dim=-1)
            z = self.gumbel_softmax(z, training=training)

            # Get top-k indices for each node
            top_k_indices = torch.topk(z, self.k, dim=-1).indices

            # Create new edge index
            new_edge_index = torch.cat([
                torch.arange(num_nodes, device=edge_index.device).view(-1, 1).expand(-1, self.k).reshape(-1, 1),
                top_k_indices.reshape(-1, 1)
            ], dim=-1).t()

            # Filter valid edges based on z values
            valid_mask = z[new_edge_index[0], new_edge_index[1]] > 0
            new_edge_index = new_edge_index[:, valid_mask]

            # Extract edge attributes
            edge_attr_indices = (edge_index[0].view(1, -1) == new_edge_index[0].view(-1, 1)) & \
                                (edge_index[1].view(1, -1) == new_edge_index[1].view(-1, 1))
            edge_attr_indices = edge_attr_indices.float().argmax(dim=-1)

            if edge_attr_indices.numel() > 0:
                new_edge_attr = edge_attr[edge_attr_indices]
            else:
                new_edge_attr = torch.empty((0, edge_attr.shape[1]), device=edge_attr.device)

        else:
            new_edge_index = edge_index
            new_edge_attr = edge_attr

        # Apply GCN layers
        x = self.conv(x, new_edge_index, new_edge_attr)
        x = F.relu(x)
        x = self.fc(x)
        x = x[node_mask != 0]

        return x
