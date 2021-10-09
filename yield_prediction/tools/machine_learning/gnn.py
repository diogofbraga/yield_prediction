import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from torch.autograd import Function

class GraphRegressionModel(torch.nn.Module):

    def __init__(self, layer_type, num_layers=3, graph_readout='mean', sz_in=9, sz_hid=256, sz_out=1):
        super().__init__()

        self.graph_readout = graph_readout

        layers = []
        layers.append(layer_type(sz_in, sz_hid))
        layers.append(nn.LeakyReLU())

        for _ in range(num_layers-2):
            layers.append(layer_type(sz_hid, sz_hid))
            layers.append(nn.LeakyReLU())
            #layers.append(nn.Dropout(p=0.2))
        
        layers.append(layer_type(sz_hid, sz_hid))

        layers.append(nn.Linear(sz_hid, sz_hid))
        layers.append(nn.LeakyReLU())
        self.layers = nn.ModuleList(layers)

        # Final classificator
        self.f = nn.Linear(sz_hid, sz_out)
    
    def forward(self, fts, adj, batch):
        # 1: obtain node latents
        for l in self.layers:
            if isinstance(l, nn.Linear) or isinstance(l, nn.LeakyReLU) or isinstance(l, nn.Dropout):
                fts = l(fts)
            else:
                fts = l(fts, adj)

        # 2: pool
        if self.graph_readout == 'mean':
            h = torch_geometric.nn.global_mean_pool(fts, batch)
        elif self.graph_readout == 'sum':
            h = torch_geometric.nn.global_add_pool(fts, batch)
        elif self.graph_readout == 'max':
            h = torch_geometric.nn.global_max_pool(fts, batch)

        # 3: final classifier
        return self.f(h)

# Train the given model on the given dataset for num_epochs
def train(model, train_loader, test_loader, num_epochs, lr, mol_comb):
    # Set up the loss and the optimizer
    loss_fn = nn.MSELoss() #ReactionMSEloss.apply #BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_rmse_test = float("inf")
    best_rmse_train = float("inf")
    best_r2_test = float("-inf")
    best_r2_train = float("-inf")

    # A utility function to compute the accuracy
    def get_acc(model, loader):
        final_r2 = 0
        final_rmse = 0
        total = 0
        for data in loader:

            additive = model(data['additive_molg'].x, data['additive_molg'].edge_index, data['additive_molg'].batch).squeeze()
            aryl_halide = model(data['aryl_halide_molg'].x, data['aryl_halide_molg'].edge_index, data['aryl_halide_molg'].batch).squeeze()
            base = model(data['base_molg'].x, data['base_molg'].edge_index, data['base_molg'].batch).squeeze()
            ligand = model(data['ligand_molg'].x, data['ligand_molg'].edge_index, data['ligand_molg'].batch).squeeze()

            if mol_comb == 'product':
                outs = additive * aryl_halide * base * ligand
            elif mol_comb == 'sum':
                outs = additive + aryl_halide + base + ligand
            elif mol_comb == 'mean':
                sum = additive + aryl_halide + base + ligand
                outs = sum/4

            outs = outs.detach().numpy()
            data['y'] = data['y'].detach().numpy()

            if outs.size > 1:
                final_r2 += r2_score(data['y'], outs)
                final_rmse += mean_squared_error(data['y'], outs, squared=False)
                total += 1
        
        r2 = final_r2/total
        rmse = final_rmse/total
        return r2, rmse

    for epoch in range(num_epochs):
        running_loss = 0
        for data in train_loader:
            # Zero grads -> forward pass -> compute loss -> backprop

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            #additivex = data['additive_molg'].x

            additive = model(data['additive_molg'].x, data['additive_molg'].edge_index, data['additive_molg'].batch).squeeze()
            aryl_halide = model(data['aryl_halide_molg'].x, data['aryl_halide_molg'].edge_index, data['aryl_halide_molg'].batch).squeeze()
            base = model(data['base_molg'].x, data['base_molg'].edge_index, data['base_molg'].batch).squeeze()
            ligand = model(data['ligand_molg'].x, data['ligand_molg'].edge_index, data['ligand_molg'].batch).squeeze()

            if mol_comb == 'product':
                outs = additive * aryl_halide * base * ligand
            elif mol_comb == 'sum':
                outs = additive + aryl_halide + base + ligand
            elif mol_comb == 'mean':
                sum = additive + aryl_halide + base + ligand
                outs = sum/4
            #print(f"outs -> data: {outs.data}\nrequires_grad: {outs.requires_grad}\n grad: {outs.grad}\ngrad_fn: {outs.grad_fn}\nis_leaf: {outs.is_leaf}\n")
            #print("next functions", outs.grad_fn.next_functions)

            loss = loss_fn(outs, data['y'].float()) # no train_mask!

            # Propagate the loss backward
            loss.backward()
            
            # Update the gradients
            optimizer.step()

            running_loss += loss.item()
        
        # Compute accuracies
        r2_train, rmse_train = get_acc(model, train_loader)
        r2_test, rmse_test = get_acc(model, test_loader)

        if rmse_train < best_rmse_train:
            best_running_loss = running_loss
            best_rmse_train = rmse_train

        if r2_train < best_r2_train:
            best_r2_train = r2_train

        if rmse_test < best_rmse_test:
            best_rmse_test = rmse_test
        
        if r2_test < best_r2_test:
            best_r2_test = r2_test

        print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.3f} | Train R-squared: {r2_train:.3f} | Train RMSE: {rmse_train:.3f} | Test R-squared: {r2_test:.3f} | Test RMSE: {rmse_test:.3f}') 

    return round(best_running_loss/len(train_loader),3), round(best_r2_train,3), round(best_rmse_train,3), round(best_r2_test,3), round(best_rmse_test,3)