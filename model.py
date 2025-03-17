import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GumbelGCN(nn.Module):
    def __init__(self, input_dim, output_dim, edge_feature_dim, k, hidden1=16, hidden2=16, weight_decay=5e-4, temperature=1.0):
        """
        GumbelGCN for node classification.

        Args:
        - num_nodes (int): Number of nodes.
        - input_dim (int): Node feature size (for ogbn-proteins, use 1-hot or identity).
        - output_dim (int): Number of classes (112 for ogbn-proteins).
        - edge_feature_dim (int): Edge feature dimension (8 for ogbn-proteins).
        - k (int): Number of top edges to keep.
        - hidden1, hidden2 (int): Hidden layer sizes.
        - weight_decay (float): L2 regularization.
        - temperature (float): Gumbel-Softmax temperature.
        """
        super(GumbelGCN, self).__init__()

        self.k = k
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Edge feature transformation layers
        self.MLP = nn.Linear(edge_feature_dim + 2 * input_dim, 1)

        # GCN layers
        self.conv = GCN(input_dim, hidden1, hidden2)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden2, output_dim)

    def sample_gumbel(self, shape, eps=1e-20):
        """Samples from a Gumbel distribution."""
        U = torch.rand(shape, device=self.MLP.weight.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax(self, logits, training=True):
        """Applies the Gumbel-Softmax trick."""
        noise = self.sample_gumbel(logits.shape) if training else 0
        return F.softmax((logits + noise) / self.temperature, dim=-1)

    def forward(self, num_nodes, edge_index, edge_attr, x, node_mask, training=True):
        """
        edge_index: (2, num_edges)
        edge_attr: (num_edges, edge_feature_dim)
        x: (num_nodes, input_dim)
        mask: (num_nodes)
        """
        adj_batch = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes), device=self.MLP.weight.device).to_dense()

        node_embedding = x.unsqueeze(-1) #(num_nodes, input_dim, 1)
        node_embedding = node_embedding.repeat(1, 1, num_nodes) #(num_nodes, input_dim, num_nodes)

        neighbor_embedding = torch.zeros((num_nodes, self.input_dim, num_nodes), device=self.MLP.weight.device)
        for u in range(num_nodes):
            neighbors = edge_index[1, edge_index[0] == u]
            for v in neighbors:
                neighbor_embedding[u, :, v] = x[v]

        edge_embedding = torch.zeros((num_nodes, edge_attr.size(1), num_nodes), device=self.MLP.weight.device)
        for u in range(num_nodes):
            neighbors = edge_index[1, edge_index[0] == u]
            for v in neighbors:
                mask = (edge_index[0] == u) & (edge_index[1] == v)
                edge_embedding[u, :, v] = edge_attr[mask].squeeze()

        all_feats = torch.cat([node_embedding, neighbor_embedding, edge_embedding], dim=1) #(num_nodes, 2*input_dim + edge_feature_dim, num_nodes)

        all_feats = all_feats.transpose(1, 2) # (num_nodes, num_nodes, 2*input_dim + edge_feature_dim)
        score = self.MLP(all_feats).squeeze()
        score[adj_batch == 0] = -1e9
        z = F.softmax(score, dim=-1)
        z[adj_batch == 0] = -1e9
        z = self.gumbel_softmax(z, training=training)

        top_k_indices = torch.topk(z, self.k, dim=-1).indices.long()
        kth_value = torch.topk(z, self.k, dim=-1).values[:, -1]

        mask = (node_mask[edge_index[0]] != 0) & (kth_value[edge_index[0]] != 0)
        top_k_mask = torch.zeros_like(z, dtype=torch.bool)
        top_k_mask.scatter_(1, top_k_indices, True)
        mask &= top_k_mask[edge_index[0], edge_index[1]]

        new_edge_index = edge_index[:, mask]

        x = self.conv(x, new_edge_index)
        x = F.relu(x)
        x = self.fc(x)
        x = x[node_mask != 0]

        return x
