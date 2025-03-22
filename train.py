import torch
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import scatter
from model import GumbelGCN

# Load the dataset
dataset = PygNodePropPredDataset('ogbn-proteins', root='./data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)

# Initialize features of nodes by aggregating edge features
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

# Create a mask for the training nodes
mask = torch.zeros(data.num_nodes, dtype=torch.bool)
mask[splitted_idx["train"]] = True
data['train_mask'] = mask

# Create a data loader for the training nodes
train_loader = RandomNodeLoader(data, num_parts=100, shuffle=True)

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = GumbelGCN(input_dim=8, output_dim=112, edge_feature_dim=8, k=5, hidden1=16, hidden2=16, weight_decay=5e-4, temperature=0.05).to(device)

# Set up the optimizer with weight decay for the first convolutional layer
optimizer = torch.optim.Adam([
    {'params': model.conv.conv1.parameters(), 'weight_decay': 5e-4},
    {'params': [p for n, p in model.named_parameters() if 'conv.conv1' not in n], 'weight_decay': 0}
], lr=0.001)

# Define the loss function
criterion = torch.nn.BCEWithLogitsLoss()

def train(epoch):
    model.train()

    # Progress bar for tracking training progress
    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:03d}')

    total_loss = total_examples = 0

    # Iterate over the training data
    for data in train_loader:
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = data.x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        num_nodes = data.num_nodes

        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(num_nodes, edge_index, edge_attr, x, data.train_mask, training=True)

        # Compute the loss
        loss = criterion(logits, data.y[data.train_mask])
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.train_mask.sum().item()
        total_examples += data.train_mask.sum().item()

        # Update the progress bar
        pbar.set_postfix({'loss': total_loss / total_examples})
        pbar.update(1)

    pbar.close()

# Train the model for 100 epochs
for epoch in range(1, 101):
    train(epoch)