import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import DataLoader
import numpy as np

class SumPoolLayer(torch.nn.Module):
    # sz_in=F, sz_out=F'
    # we will apply the nonlinearity later
    def __init__(self, sz_in, sz_out):
        super().__init__()
        self.W = nn.Parameter(torch.empty(size=(sz_in, sz_out)))
        nn.init.xavier_uniform_(self.W.data)
    
    # Propagation rule: H' = sigma(AHW) 
    def forward(self, fts, adj):
        new_fts = torch.mm(fts, self.W)
        ret_fts = torch.mm(adj, new_fts)
        return ret_fts

class LinearLayer(torch.nn.Module):
    def __init__(self, sz_in, sz_out):
        super().__init__()
        self.W = nn.Parameter(torch.empty(size=(sz_in, sz_out)))
        nn.init.xavier_uniform_(self.W.data)
    
    def forward(self, fts, adj):
        # Simple linear layer application
        new_fts = torch.mm(fts, self.W)
        return new_fts

class MeanPoolLayer(torch.nn.Module):
    def __init__(self, sz_in, sz_out):
        super().__init__()
        self.W = nn.Parameter(torch.empty(size=(sz_in, sz_out)))
        nn.init.xavier_uniform_(self.W.data)
    
    def forward(self, fts, adj):
        new_fts = torch.mm(fts, self.W)
        deg = adj.sum(axis=1) # New!
        ret_fts = torch.mm(adj/deg, new_fts) # New!
        return ret_fts

class GCNLayer(torch.nn.Module):
    def __init__(self, sz_in, sz_out):
        super().__init__()
        self.W = nn.Parameter(torch.empty(size=(sz_in, sz_out)))
        nn.init.xavier_uniform_(self.W.data)
    
    def forward(self, fts, adj):
        new_fts = torch.mm(fts, self.W)

        deg = adj.sum(axis=1)
        deg_inv_half = torch.diag(1.0 / torch.sqrt(deg)) # New!
        adj_norm = deg_inv_half @ adj @ deg_inv_half # New!
        
        ret_fts = torch.mm(adj_norm, new_fts) # New!
        return ret_fts

class GraphRegressionModel(torch.nn.Module):

    def __init__(self, layer_type, num_layers=3, sz_in=9, sz_hid=256, sz_out=1):
        super().__init__()

        # GNN layers with ReLU, as before
        layers = []
        layers.append(layer_type(sz_in, sz_hid))
        layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            layers.append(layer_type(sz_hid, sz_hid))
            layers.append(nn.ReLU())
        layers.append(layer_type(sz_hid, sz_hid)) # New!
        self.layers = nn.ModuleList(layers)

        # Final classificator
        self.f = nn.Linear(sz_hid, sz_out)
    
    def forward(self, fts, adj, batch):
        # 1: obtain node latents
        for l in self.layers:
            if isinstance(l, nn.ReLU):
                fts = l(fts)
            else:
                fts = l(fts, adj)

        # 2: pool
        h = torch_geometric.nn.global_mean_pool(fts, batch)

        # 3: final classifier
        return self.f(h)

# Train the given model on the given dataset for num_epochs
def train(model, train_loader, test_loader, num_epochs):
    # Set up the loss and the optimizer
    loss_fn = nn.MSELoss() #BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # A utility function to compute the accuracy
    def get_acc(model, loader):
        n_total = 0
        n_ok = 0
        for data in loader:
            outs = model(data.x, data.edge_index, data.batch).squeeze()
            n_ok += ((outs>0) == data.y).sum().item()
            n_total += data.y.shape[0]
        return n_ok/n_total

    for epoch in range(num_epochs):
        for data in train_loader:
            # Zero grads -> forward pass -> compute loss -> backprop
            optimizer.zero_grad()
            outs = model(data.x, data.edge_index, data.batch).squeeze()
            #print("outs: \n", outs)
            loss = loss_fn(outs, data.y.float()) # no train_mask!
            #print("loss: \n", loss)
            loss.backward()
            optimizer.step()

        # Compute accuracies
        acc_train = get_acc(model, train_loader)
        acc_test = get_acc(model, test_loader)
        print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Train: {acc_train:.3f} | Test: {acc_test:.3f}')