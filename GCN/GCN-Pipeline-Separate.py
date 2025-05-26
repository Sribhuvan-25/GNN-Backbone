#%% Imports and Setup
import os
import itertools
import collections

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, PNA

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% Data Loading and Preprocessing
data_path = "../Data/New_data.csv"
df = pd.read_csv(data_path)
df = df[~df.isin(['x']).any(axis=1)]

otu_cols  = [c for c in df.columns if "d__" in c]
meta_cols = [c for c in df.columns if c not in otu_cols]

def extract_family_from_colname(colname):
    for part in colname.split(';'):
        part = part.strip()
        if part.startswith('f__'):
            return part[3:] or "UnclassifiedFamily"
    return "UnclassifiedFamily"

col_to_family = {c: extract_family_from_colname(c) for c in otu_cols}
family_to_cols = collections.defaultdict(list)
for c, fam in col_to_family.items():
    family_to_cols[fam].append(c)

df_fam = pd.DataFrame({
    fam: df[cols].sum(axis=1)
    for fam, cols in family_to_cols.items()
}, index=df.index)

param_df   = df[meta_cols].copy()
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)

df_fam_rel.to_csv("family_abundance.csv", index=False)
param_df.to_csv("parameters.csv", index=False)


#%% Filtering Families
df_fam_rel = pd.read_csv("family_abundance.csv")
param_df   = pd.read_csv("parameters.csv")
param_df.columns = param_df.columns.str.strip()

presence_count = (df_fam_rel > 0).sum(0)
prevalence     = presence_count / df_fam_rel.shape[0]
high_prev      = prevalence[prevalence >= 0.05].index

mean_abund     = df_fam_rel.mean(0)
high_abund     = mean_abund[mean_abund >= 0.01].index

selected_families   = high_prev.intersection(high_abund)
df_fam_rel_filtered = df_fam_rel[selected_families].copy()
print(f"Selected {len(selected_families)} families after filtering.")


#%% Construct Node Features & Targets
target_cols    = ['ACE-km', 'H2-km']
target_data    = param_df[target_cols].copy()
df_microbe     = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))
node_feature_names = list(df_microbe.columns)
num_nodes      = len(node_feature_names)
feature_matrix = df_microbe.values.astype(np.float32)
print(f"Node feature matrix: {feature_matrix.shape} (samples × nodes)")


#%% Build Edges via Mantel Test
os.makedirs('intermediate_results', exist_ok=True)

def compute_distance_matrix(vec, metric='braycurtis'):
    dm = squareform(pdist(vec[:, None], metric=metric))
    return np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)

def mantel_test(d1, d2, permutations=999):
    n   = d1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1, v2 = d1[idx], d2[idx]
    if v1.std()==0 or v2.std()==0:
        return 1.0, 0.0
    r_obs = pearsonr(v1, v2)[0]
    count = sum(
        abs(pearsonr(v1, np.random.permutation(v2))[0]) >= abs(r_obs)
        for _ in range(permutations)
    )
    return (count + 1) / (permutations + 1), r_obs

dist_mats = {
    fam: compute_distance_matrix(feature_matrix[:, i], metric='braycurtis')
    for i, fam in enumerate(node_feature_names)
}

edge_i, edge_j, edge_weights = [], [], []
for i, j in itertools.combinations(range(num_nodes), 2):
    p, r = mantel_test(dist_mats[node_feature_names[i]],
                       dist_mats[node_feature_names[j]], permutations=999)
    if p < 0.05:
        edge_i += [i, j]
        edge_j += [j, i]
        edge_weights += [abs(r), abs(r)]  # Use correlation strength as edge weight

edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
print("Total undirected edges:", edge_index.shape[1] // 2)


#%% Apply KNN Sparsification
def create_knn_sparsified_graph(edge_index, edge_weight, num_nodes, k=10):
    """Create a k-nearest neighbor sparsified version of the graph"""
    print(f"Creating KNN sparsified graph with k={k}...")
    
    # Create adjacency matrix from edge_index
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        adj_matrix[u, v] = edge_weight[i]
    
    # KNN sparsification
    adj_matrix_np = adj_matrix.numpy()
    
    # For each node, keep only the k strongest connections
    for i in range(num_nodes):
        # Get weights of all neighbors
        neighbors = adj_matrix_np[i]
        
        # Sort neighbors by weight and keep only top k
        if np.sum(neighbors > 0) > k:
            threshold = np.sort(neighbors)[-k]
            adj_matrix_np[i, neighbors < threshold] = 0
    
    # Make matrix symmetric (undirected graph)
    adj_matrix_np = np.maximum(adj_matrix_np, adj_matrix_np.T)
    
    # Convert back to edge_index and edge_weight format
    new_edge_index = []
    new_edge_weight = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix_np[i, j] > 0:
                new_edge_index.append([i, j])
                new_edge_weight.append(adj_matrix_np[i, j])
    
    new_edge_index = torch.tensor(new_edge_index).t().contiguous()
    new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32)
    
    print(f"KNN sparsified graph has {new_edge_index.shape[1]//2} undirected edges")
    
    return new_edge_index, new_edge_weight

# Apply KNN sparsification
k_neighbors = 12  # Set the number of neighbors to keep
sparsified_edge_index, sparsified_edge_weight = create_knn_sparsified_graph(
    edge_index, edge_weight, num_nodes, k=k_neighbors
)

# Use the sparsified graph instead of the original one
edge_index = sparsified_edge_index
print(f"Using sparsified graph with {edge_index.shape[1]//2} undirected edges")


#%% Create PyG Data Objects
graphs = []
for idx in range(len(feature_matrix)):
    x = torch.tensor(feature_matrix[idx], dtype=torch.float).unsqueeze(1)  
    # Store y as 2D tensor with shape [1, 2] for consistency
    y = torch.tensor(target_data.iloc[idx].values.astype(np.float32), dtype=torch.float).view(1, 2)  # shape [1, 2]
    graphs.append(Data(x=x, edge_index=edge_index, y=y))

print(f"Created {len(graphs)} graphs with {num_nodes} nodes each.")
print(f"Target shape for each graph: {graphs[0].y.shape}")


#%% Built-in GNN Wrapper
class BuiltinGNN(nn.Module):
    def __init__(self,
                 model_name: str,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: int,
                 dropout: float = 0.3,
                 act: str = 'relu',
                 jk: str  = None,
                 **kwargs):
        super().__init__()
        
        # For GCN, use our custom implementation with GCNConv
        if model_name == 'GCN':
            self.use_custom_impl = True
            from torch_geometric.nn import GCNConv, global_mean_pool
            
            # Create layers
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            
            # First layer
            self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Middle layers
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Activation function
            if act == 'relu':
                self.act_fn = F.relu
            elif act == 'elu':
                self.act_fn = F.elu
            else:
                self.act_fn = F.relu
            
            self.dropout = nn.Dropout(dropout)
        else:
            # For other models, use the high-level wrappers
            self.use_custom_impl = False
            model_map = {
                'GraphSAGE': GraphSAGE,
                'GIN':       GIN,
                'GAT':       GAT,
                'PNA':       PNA,
            }
            Wrapper = model_map[model_name]
            
            self.gnn = Wrapper(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=hidden_channels,
                dropout=dropout,
                act=act,
                jk=jk,
                **kwargs
            )
        
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        if self.use_custom_impl:
            # Custom implementation for GCN
            from torch_geometric.nn import global_mean_pool
            
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # Apply GCN layers
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = self.act_fn(x)
                x = self.dropout(x)
            
            # Global pooling
            x = global_mean_pool(x, batch)
        else:
            # High-level wrappers for other models
            x = self.gnn(data.x, data.edge_index, data.batch)
            
            # Apply pooling if needed
            if x.size(0) != data.num_graphs:
                from torch_geometric.nn import global_mean_pool
                x = global_mean_pool(x, data.batch)
        
        # Final projection
        return self.head(x)  # [batch_size, out_channels]


#%% Training Function
def train_model(target_idx, target_name,
                model_name='GIN',
                hidden_channels=64,
                num_layers=3,
                dropout=0.3,
                jk=None,
                **kwargs):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_trues = [], []

    # Check the structure of a single graph's y
    print(f"Single graph y shape: {graphs[0].y.shape}, dtype: {graphs[0].y.dtype}")
    print(f"First few y values: {graphs[0].y}")

    for fold, (tr, va) in enumerate(kf.split(graphs), 1):
        # Create new graphs with only the target of interest for cleaner handling
        tr_graphs = []
        for i in tr:
            # Extract the specific target value
            target_val = graphs[i].y[0, target_idx].item()
            # Create a new graph with only this target
            new_graph = Data(
                x=graphs[i].x,
                edge_index=graphs[i].edge_index,
                y=torch.tensor([[target_val]], dtype=torch.float)  # Shape: [1, 1]
            )
            tr_graphs.append(new_graph)
        
        va_graphs = []
        for i in va:
            target_val = graphs[i].y[0, target_idx].item()
            new_graph = Data(
                x=graphs[i].x,
                edge_index=graphs[i].edge_index,
                y=torch.tensor([[target_val]], dtype=torch.float)  # Shape: [1, 1]
            )
            va_graphs.append(new_graph)
        
        tr_loader = DataLoader(tr_graphs, batch_size=16, shuffle=True)
        va_loader = DataLoader(va_graphs, batch_size=16)

        # Get the first batch to inspect
        for debug_batch in tr_loader:
            print(f"Batch y shape: {debug_batch.y.shape}, dtype: {debug_batch.y.dtype}")
            print(f"Batch size: {debug_batch.num_graphs}")
            # Only checking the first batch
            break

        model = BuiltinGNN(
            model_name=model_name,
            in_channels=1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=1,
            dropout=dropout,
            act='relu',
            jk=jk,
            **kwargs
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        loss_fn   = nn.MSELoss()

        for epoch in range(1, 301):
            model.train()
            total_loss = 0
            for batch in tr_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)  # [B,1]
                
                # Target is already in the right shape [B,1]
                target = batch.y
                
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            
            if epoch % 50 == 0:
                print(f"{target_name} [{model_name}] Fold {fold} Epoch {epoch} Train MSE: {total_loss/len(tr_loader.dataset):.4f}")

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                p = model(batch)
                
                # Target is already in the right shape [B,1]
                t = batch.y
                
                preds.append(p.cpu().numpy())
                trues.append(t.cpu().numpy())
        
        all_preds.append(np.vstack(preds))
        all_trues.append(np.vstack(trues))
        print(f"{target_name} [{model_name}] Fold {fold} done.")

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    mse = mean_squared_error(all_trues, all_preds)
    r2  = r2_score(all_trues, all_preds)
    print(f"\n{target_name} [{model_name}] CV MSE: {mse:.4f}, R^2: {r2:.3f}")

    plt.figure(figsize=(6,5))
    plt.scatter(all_trues, all_preds, alpha=0.7)
    
    # Set consistent axis limits
    if target_name == "ACE-km":
        plt.xlim(0, 50)
        plt.ylim(0, 50)
    elif target_name == "H2-km":
        plt.xlim(0, 140)
        plt.ylim(0, 140)
    
    # Add 45-degree line
    plt.plot([0, plt.xlim()[1]], [0, plt.ylim()[1]], 'r--')
    
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"{target_name} [{model_name}] Pred vs Actual\nMSE={mse:.4f}, R^2={r2:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{target_name}_{model_name}_sparsified_k{k_neighbors}.png", dpi=300)
    plt.show()

    return all_preds, all_trues, mse, r2


#%% Train Separate Models
print("=== Training ACE-km ===")

# For PNA model, we need to calculate the degree histogram
if 'PNA' in ['PNA', 'pna']:
    # Compute in-degree histogram over training data
    from torch_geometric.utils import degree
    deg = torch.zeros(100, dtype=torch.long)
    for data in graphs:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

ace_preds, ace_trues, ace_mse, ace_r2 = train_model(
    target_idx=0,
    target_name="ACE-km",
    model_name='GCN',
    hidden_channels=128,  # Increased from 64
    num_layers=8,         # Increased from 3
    dropout=0.2,          # Decreased from 0.6
    # jk='cat',
    # heads=4   # extra arg for GAT
    # Required PNA parameters
    # aggregators=['mean', 'min', 'max', 'std'],
    # scalers=['identity', 'amplification', 'attenuation'],
    # deg=deg
)

print("\n=== Training H2-km ===")
h2_preds, h2_trues, h2_mse, h2_r2 = train_model(
    target_idx=1,
    target_name="H2-km",
    model_name='GCN',
    hidden_channels=128,  # Increased from 64
    num_layers=8,         # Increased from 3
    dropout=0.2,          # Decreased from 0.6
    # jk='cat',
    # heads=4   # extra arg for GAT
    # Required PNA parameters
    # aggregators=['mean', 'min', 'max', 'std'],
    # scalers=['identity', 'amplification', 'attenuation'],
    # deg=deg
)

# %%
