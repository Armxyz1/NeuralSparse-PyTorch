import torch
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import scatter
from model import GumbelGCN

dataset = PygNodePropPredDataset('ogbn-proteins', root='./data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)

# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

mask = torch.zeros(data.num_nodes, dtype=torch.bool)
mask[splitted_idx["train"]] = True
data['train_mask'] = mask

train_loader = RandomNodeLoader(data, num_parts=100, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GumbelGCN(input_dim=8, output_dim=112, edge_feature_dim=8, k=5, hidden1=16, hidden2=16, weight_decay=5e-4, temperature=0.05).to(device)

optimizer = torch.optim.Adam([
    {'params': model.conv.conv1.parameters(), 'weight_decay': 5e-4},
    {'params': [p for n, p in model.named_parameters() if 'conv.conv1' not in n], 'weight_decay': 0}
], lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:03d}')

    total_loss = total_examples = 0

    for data in train_loader:
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = data.x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        num_nodes = data.num_nodes

        optimizer.zero_grad()
        logits = model(num_nodes, edge_index, edge_attr, x, data.train_mask, training=True)

        loss = criterion(logits, data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.train_mask.sum().item()
        total_examples += data.train_mask.sum().item()

        pbar.set_postfix({'loss': total_loss / total_examples})

        pbar.update(1)

    pbar.close()

for epoch in range(1, 101):
    train(epoch)