#%%
import pandas as pd
import numpy as np
import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch
from torch_geometric.data import Data
import collections
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_squared_error, r2_score
import os
import itertools
import argparse
from scipy.spatial.distance import pdist, squareform

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Methane GNN Pipeline')
parser.add_argument('--data_path', type=str, default='../Data/New_data.csv', help='Path to the CSV file with microbial data')
parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors for KNN graph sparsification')
parser.add_argument('--mantel_threshold', type=float, default=0.05, help='p-value threshold for Mantel test')
parser.add_argument('--model_type', type=str, default='gat', help='Type of GNN model (gcn, gat, gatv2, gin)')
parser.add_argument('--model_architecture', type=str, default='default', help='Architecture to use (default, simple_gcn_res, simple_rggc, simple_gat)')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
parser.add_argument('--num_epochs', type=int, default=300, help='Maximum number of epochs')
parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping')
parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--save_dir', type=str, default='./methane_results', help='Directory to save results')
parser.add_argument('--visualize_graphs', type=bool, default=True, help='Whether to visualize the graphs')

args = parser.parse_args()

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#%% Data Loading and Preprocessing
# Load OTU table from the correct path
data_path = args.data_path
df = pd.read_csv(data_path)
print(f"Loaded df shape: {df.shape}")
# Filter out rows containing 'x' values
df = df[~df.isin(['x']).any(axis=1)]
print(f"Filtered df shape (no 'x'): {df.shape}")

# Identify OTU and metadata columns
otu_cols = [c for c in df.columns if "d__" in c]
meta_cols = [c for c in df.columns if c not in otu_cols]
print(f"Number of OTU columns: {len(otu_cols)}")
print(f"Number of meta columns: {len(meta_cols)}")

# Extract family from taxonomy string
def extract_family_from_colname(colname):
    for part in colname.split(';'):
        part = part.strip()
        if part.startswith('f__'):
            return part[3:] or "UnclassifiedFamily"
    return "UnclassifiedFamily"

# Map OTUs to families and create a family-level dataframe
col_to_family = {c: extract_family_from_colname(c) for c in otu_cols}
family_to_cols = collections.defaultdict(list)
for c, fam in col_to_family.items():
    family_to_cols[fam].append(c)

df_fam = pd.DataFrame({
    fam: df[cols].sum(axis=1)
    for fam, cols in family_to_cols.items()
}, index=df.index)
print(f"df_fam shape: {df_fam.shape}")

# Store metadata separately
param_df = df[meta_cols].copy()
param_df.columns = param_df.columns.str.strip()

# Convert counts to relative abundances
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)
print(f"df_fam_rel shape: {df_fam_rel.shape}")

#%% Filtering Families by Prevalence and Abundance
print("\n#%% Filtering Families")
print("Selecting families based on prevalence and abundance thresholds...")

# Filter families based on prevalence and abundance
presence_count = (df_fam_rel > 0).sum(axis=0)
prevalence = presence_count / df_fam_rel.shape[0]
high_prev = prevalence[prevalence >= 0.05].index  # Keep families present in at least 5% of samples

mean_abund = df_fam_rel.mean(axis=0)
high_abund = mean_abund[mean_abund >= 0.01].index  # Keep families with at least 1% mean abundance

selected_families = high_prev.intersection(high_abund)
df_fam_rel_filtered = df_fam_rel[selected_families].copy()
print(f"Selected {len(selected_families)} families after filtering (out of {df_fam_rel.shape[1]}).")

# Feature transformation: apply double square root transformation for variance stabilization
df_microbe = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))
node_feature_names = list(df_microbe.columns)
num_families = len(node_feature_names)
feature_matrix = df_microbe.values.T.astype(np.float32)  # Shape: [num_families, num_samples]
print(f"Node feature matrix: {feature_matrix.shape} (families × samples)")

# Extract target variables
target_cols = ['ACE-km', 'H2-km']
# Convert target values to float
param_df[target_cols] = param_df[target_cols].astype(float)
target_data = param_df[target_cols].copy()
ace_targets = param_df['ACE-km'].values
h2_targets = param_df['H2-km'].values

#%% Graph Construction via Mantel Test
print("\n#%% Graph Construction via Mantel Test")
print("Constructing microbial interaction network using Mantel test...")

# Create a directory for intermediate results
os.makedirs('intermediate_results', exist_ok=True)

def compute_distance_matrix(vec, metric='braycurtis'):
    """Compute distance matrix between samples for a given microbial family"""
    dm = squareform(pdist(vec[:, None], metric=metric))
    return np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)

def mantel_test(d1, d2, permutations=999):
    """Mantel test to assess correlation between two distance matrices"""
    n = d1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1, v2 = d1[idx], d2[idx]
    if v1.std() == 0 or v2.std() == 0:
        return 1.0, 0.0
    r_obs = pearsonr(v1, v2)[0]
    count = sum(
        abs(pearsonr(v1, np.random.permutation(v2))[0]) >= abs(r_obs)
        for _ in range(permutations)
    )
    return (count + 1) / (permutations + 1), r_obs

# Compute distance matrices for each family
dist_mats = {
    fam: compute_distance_matrix(feature_matrix[i], metric='braycurtis')
    for i, fam in enumerate(node_feature_names)
}

# Construct edges based on Mantel test (p < 0.05)
edge_i, edge_j, edge_weights = [], [], []
for i, j in itertools.combinations(range(num_families), 2):
    p, r = mantel_test(dist_mats[node_feature_names[i]], 
                     dist_mats[node_feature_names[j]], 
                     permutations=999)
    if p < 0.05:  # Significant correlation between distance matrices
        edge_i += [i, j]
        edge_j += [j, i]
        edge_weights += [abs(r), abs(r)]  # Use correlation strength as edge weight

edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
print(f"Mantel test identified {len(edge_i)//2} undirected edges")

# Create edge types based on the sign of the correlation
edge_types = []
for i, j in zip(edge_i, edge_j):
    # Calculate correlation coefficient between the two families
    corr, _ = pearsonr(feature_matrix[i], feature_matrix[j])
    # Edge type: 0 = negative correlation, 1 = positive correlation
    edge_type = 1 if corr > 0 else 0
    edge_types.append(edge_type)
    
edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)

#%% Enhanced Graph Visualization with Family Names
print("\n#%% Enhanced Graph Visualization")
print("Visualizing the microbial family network with family names...")

# Create a NetworkX graph from the edge_index with family names
G = nx.Graph()

# Add all nodes first (including isolated ones)
for i, family in enumerate(node_feature_names):
    G.add_node(i, name=family)

# Add edges with weights and types
for i in range(len(edge_i)//2):  # Only add undirected edges once
    u, v = edge_i[i*2], edge_j[i*2]
    weight = edge_weights[i*2]
    edge_type = edge_types[i*2]
    G.add_edge(u, v, weight=weight, type=edge_type)

# Create more informative layout
pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42, weight='weight')

# Analyze the graph
connected_components = list(nx.connected_components(G))
print(f"Graph has {len(connected_components)} connected components")
largest_cc = max(connected_components, key=len)
print(f"Largest component has {len(largest_cc)} nodes out of {len(G.nodes)}")

# Calculate node centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality_numpy(G)

# Create a combined centrality measure
combined_centrality = {}
for node in G.nodes():
    combined_centrality[node] = (
        0.4 * degree_centrality[node] + 
        0.3 * betweenness_centrality[node] + 
        0.3 * eigenvector_centrality[node]
    )

sorted_by_centrality = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 most central families (potential keystone taxa):")
for i, (node_idx, centrality) in enumerate(sorted_by_centrality[:10]):
    family = node_feature_names[node_idx]
    print(f"{i+1}. {family} (centrality: {centrality:.3f}, degree: {G.degree(node_idx)})")

# Visualize with edge types, node importance and family names
plt.figure(figsize=(20, 16))

# Scale node size by combined centrality
node_size = [1000 * (0.1 + combined_centrality[node]) for node in G.nodes()]

# Scale edge width by correlation strength and color by type
edge_colors = []
edge_width = []

for u, v, data in G.edges(data=True):
    # Edge type determines color: 0 = negative correlation, 1 = positive correlation
    if data['type'] == 0:
        edge_colors.append('red')  # negative correlation
    else:
        edge_colors.append('green')  # positive correlation
    
    # Width based on weight
    edge_width.append(abs(data['weight']) * 2 + 0.5)

# Color nodes by module/community
try:
    # Attempt to find communities in the graph
    communities = nx.community.greedy_modularity_communities(G)
    print(f"\nDetected {len(communities)} communities/modules")
    
    # Create a mapping of nodes to community ID
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
    
    # Color nodes by community
    node_colors = [community_map.get(node, 0) for node in G.nodes()]
    
except:
    # Fallback if community detection fails
    node_colors = list(range(len(G.nodes)))

# Create better visualization
nx.draw_networkx(
    G, 
    pos=pos,
    with_labels=False,  # We'll add custom labels
    node_size=node_size,
    node_color=node_colors,
    width=edge_width,
    edge_color=edge_colors,
    alpha=0.8,
    cmap=plt.cm.tab20
)

# Add labels for ALL nodes
labels = {node: node_feature_names[node] for node in G.nodes()}
nx.draw_networkx_labels(
    G, 
    pos=pos,
    labels=labels,
    font_size=8,
    font_weight='bold',
    font_color='black'
)

# Create a legend for edge types
plt.plot([], [], 'g-', linewidth=2, label='Positive correlation')
plt.plot([], [], 'r-', linewidth=2, label='Negative correlation')

plt.legend(loc='upper right')
plt.title('Microbial Family Interaction Network (All Taxa Labeled)', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig('microbial_network_with_names.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key network statistics
print(f"\nNetwork Statistics:")
print(f"Number of nodes (microbial families): {len(G.nodes)}")
print(f"Number of edges (interactions): {len(G.edges)}")
avg_degree = sum(dict(G.degree()).values()) / len(G)
print(f"Average degree: {avg_degree:.2f}")
density = nx.density(G)
print(f"Network density: {density:.3f}")
print(f"Number of connected components: {len(connected_components)}")
print(f"Largest component size: {len(largest_cc)} nodes ({len(largest_cc)/len(G.nodes):.1%} of network)")

# Calculate network modularity
try:
    modularity = nx.community.modularity(G, communities)
    print(f"Network modularity: {modularity:.3f}")
except:
    print("Could not calculate modularity")

#%% Create Graph Data Objects
print("\n#%% Creating Graph Data Objects")
# Transpose feature matrix to shape [num_samples, num_families]
feature_matrix_samples = feature_matrix.T  # Shape: [num_samples, num_families]

data_list = []
edge_index_tensor = torch.tensor([edge_i, edge_j], dtype=torch.long)  # [2, num_edges]
edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)  # [num_edges]
edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)  # [num_edges]

for s in range(feature_matrix_samples.shape[0]):
    # Node feature for each family in this sample - the abundance value in this sample
    x = torch.tensor(feature_matrix_samples[s], dtype=torch.float32).view(-1, 1)  # [num_families, 1]
    
    # Graph-level target (ACE-km and H2-km) for this sample
    ace_val = float(ace_targets[s])
    h2_val = float(h2_targets[s])
    
    # Create tensor with shape [1, 2] for proper batching
    y = torch.tensor([[ace_val, h2_val]], dtype=torch.float32)
    
    # Create the Data object with edge attributes
    data = Data(
        x=x, 
        edge_index=edge_index_tensor, 
        edge_weight=edge_weight_tensor,
        edge_attr=edge_weight_tensor.view(-1, 1),  # Reshape for PyG
        y=y
    )
    data_list.append(data)

print(f"Created {len(data_list)} graph data objects with {num_families} nodes each.")
print(f"Sample graph data: {data_list[0]}")
print(f"Node features shape: {data_list[0].x.shape}")
print(f"Target shape: {data_list[0].y.shape}")
print(f"Edge index shape: {data_list[0].edge_index.shape}")

# Save the processed data
torch.save({
    'edge_index': edge_index_tensor,
    'edge_weight': edge_weight_tensor,
    'edge_type': edge_type_tensor,
    'node_feature_names': node_feature_names,
    'num_families': num_families
}, 'mantel_graph.pt')

#%% GNN Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, GraphSAGE, global_mean_pool, global_add_pool

class ImprovedMethaneGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, num_targets=1, dropout_rate=0.3, 
                 num_layers=4, model_type='gat', use_edge_attr=True):
        super(ImprovedMethaneGNN, self).__init__()
        self.num_layers = num_layers
        self.model_type = model_type
        self.use_edge_attr = use_edge_attr
        
        # Choose GNN layer type
        if model_type == 'gcn':
            self.conv_layers = nn.ModuleList()
            # First layer
            self.conv_layers.append(GCNConv(num_node_features, hidden_dim))
            # Middle layers
            for i in range(num_layers - 1):
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        elif model_type == 'gat':
            self.conv_layers = nn.ModuleList()
            # First layer with 8 attention heads
            self.conv_layers.append(GATConv(num_node_features, hidden_dim // 8, heads=8, dropout=dropout_rate))
            # Middle layers
            for i in range(num_layers - 2):
                self.conv_layers.append(GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout_rate))
            # Final GAT layer with 1 attention head for concatenation
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout_rate))
            
        elif model_type == 'gatv2':
            self.conv_layers = nn.ModuleList()
            # First layer with 8 attention heads
            self.conv_layers.append(GATv2Conv(num_node_features, hidden_dim // 8, heads=8, dropout=dropout_rate))
            # Middle layers
            for i in range(num_layers - 2):
                self.conv_layers.append(GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout_rate))
            # Final GATv2 layer
            self.conv_layers.append(GATv2Conv(hidden_dim, hidden_dim, heads=1, dropout=dropout_rate))
        
        elif model_type == 'gin':
            self.conv_layers = nn.ModuleList()
            # For GIN, we need to define MLPs for each layer
            mlp1 = nn.Sequential(
                nn.Linear(num_node_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.conv_layers.append(GINConv(mlp1))
            
            # Middle GIN layers
            for i in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.conv_layers.append(GINConv(mlp))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Attention pooling layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        # Xavier initialization for better training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def attention_pooling(self, x, batch):
        # Compute attention weights
        attn_weights = self.attention(x).squeeze(-1)
        # Apply softmax over nodes in same graph (per batch)
        # Need to handle differently for each graph in the batch
        output = torch.zeros(batch.max().item() + 1, x.size(-1), device=x.device)
        
        for i in range(batch.max().item() + 1):
            mask = (batch == i)
            # Get nodes for this graph
            graph_x = x[mask]
            # Get attention weights for this graph
            graph_weights = attn_weights[mask]
            # Apply softmax to get normalized weights
            graph_weights = F.softmax(graph_weights, dim=0)
            # Apply attention pooling
            graph_output = torch.sum(graph_x * graph_weights.unsqueeze(-1), dim=0)
            # Store result
            output[i] = graph_output
            
        return output
        
    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
        # Handle batch being None for single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Initial features
        last_x = x
        
        # Apply GNN layers with residual connections and batch normalization
        for i, conv in enumerate(self.conv_layers):
            if self.model_type == 'gcn':
                # For GCN, we can pass edge weights
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.model_type in ['gat', 'gatv2']:
                # For GAT, we don't pass edge weights directly 
                x = conv(x, edge_index)
            elif self.model_type == 'gin':
                # For GIN
                x = conv(x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
            # Apply residual connection for all but first layer
            if i > 0 and x.size() == last_x.size():
                x = x + last_x
            
            # Apply dropout except last layer
            if i < len(self.conv_layers) - 1:
                x = F.dropout(x, p=0.2, training=self.training)
            
            last_x = x
        
        # Multiple pooling methods combined
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        # Also use attention pooling
        x_attn = self.attention_pooling(x, batch)
        
        # Combine different pooling methods
        x_combined = x_mean + 0.5 * x_attn + 0.1 * x_add
        
        # Final MLP layers
        out = self.mlp(x_combined)
        return out

#%% Training
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

# Prepare to store results for each target
target_names = ['ACE-km', 'H2-km']
results_by_target = {name: {'preds': [], 'trues': []} for name in target_names}

# Iterate through each target and train a separate model
for target_idx, target_name in enumerate(target_names):
    print(f"\n{'='*50}")
    print(f"Training improved model for {target_name}")
    print(f"{'='*50}")
    
    # Prepare data with single target
    single_target_data_list = []
    for data in data_list:
        # Create a copy with only the specific target
        target_value = data.y[0, target_idx].item()
        single_target_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_weight,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
            y=torch.tensor([[target_value]], dtype=torch.float32)
        )
        single_target_data_list.append(single_target_data)
    
    # Create k-fold training for this target
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    fold_results = []
    
    for train_index, test_index in kf.split(single_target_data_list):
        print(f"Fold {fold}: Train on {len(train_index)} samples, Test on {len(test_index)} samples")
        # Split data_list into train and test subsets
        train_dataset = [single_target_data_list[i] for i in train_index]
        test_dataset = [single_target_data_list[i] for i in test_index]
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Try different GNN variants for different targets
        model_type = 'gat' if target_name == 'ACE-km' else 'gatv2'
        
        # Initialize model for this fold with improved parameters
        model = ImprovedMethaneGNN(
            num_node_features=1, 
            hidden_dim=128,  # Larger hidden dimensions
            num_targets=1,
            dropout_rate=0.3,
            num_layers=4,   # Deeper model
            model_type=model_type
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        num_epochs = 300  # More epochs
        patience = 30  # More patience
        patience_counter = 0
        
        for epoch in range(1, num_epochs+1):
            model.train()
            total_loss = 0
            # Training step
            for batch_data in train_loader:
                batch_data = batch_data.to(device)  # Move to device
                optimizer.zero_grad()
                out = model(batch_data)
                # Reshape target to match output
                target = batch_data.y.view(-1, 1)
                
                loss = criterion(out, target)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * batch_data.num_graphs
            
            avg_loss = total_loss / len(train_dataset)
            
            # Evaluate on test set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_data = batch_data.to(device)  # Move to device
                    pred = model(batch_data)
                    # Reshape target
                    target = batch_data.y.view(-1, 1)
                    val_loss += criterion(pred, target).item() * batch_data.num_graphs
                val_loss /= len(test_dataset)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0 or epoch == num_epochs:
                print(f"Epoch {epoch:03d} - Train MSE: {avg_loss:.4f}, Val MSE: {val_loss:.4f}")
        
        # Load best model for this fold
        model.load_state_dict(best_model_state)
        
        # After training, evaluate final performance on test fold
        model.eval()
        with torch.no_grad():
            test_loss = 0
            fold_preds, fold_trues = [], []
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                pred = model(batch_data)
                # Reshape target
                target = batch_data.y.view(-1, 1)
                fold_preds.append(pred)
                fold_trues.append(target)
                test_loss += criterion(pred, target).item()
            fold_preds = torch.cat(fold_preds, dim=0)
            fold_trues = torch.cat(fold_trues, dim=0)
            fold_mse = test_loss / len(test_dataset)
            fold_rmse = fold_mse**0.5
        print(f"Fold {fold} results: MSE = {fold_mse:.3f}, RMSE = {fold_rmse:.3f}\n")
        
        # Store predictions and true values for combined evaluation
        fold_results.append({
            'fold': fold,
            'preds': fold_preds.cpu().numpy(),
            'trues': fold_trues.cpu().numpy()
        })
        
        # Store all results for this target
        results_by_target[target_name]['preds'].extend(fold_preds.flatten().cpu().numpy())
        results_by_target[target_name]['trues'].extend(fold_trues.flatten().cpu().numpy())
        
        # Save the best model for this fold
        torch.save(model.state_dict(), f"{target_name}_improved_model_fold{fold}.pt")
        
        fold += 1
    
    # Create a final model trained on all data for this target
    print(f"\nTraining final improved {target_name} model on all data...")
    all_loader = DataLoader(single_target_data_list, batch_size=16, shuffle=True)
    
    # Initialize model and optimizer - use GATv2 for H2-km
    model_type = 'gat' if target_name == 'ACE-km' else 'gatv2'
    
    final_model = ImprovedMethaneGNN(
        num_node_features=1, 
        hidden_dim=128,
        num_targets=1,
        dropout_rate=0.3,
        num_layers=4,
        model_type=model_type
    ).to(device)
    
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )
    criterion = nn.MSELoss()
    
    # Train on all data
    epochs = 300
    for epoch in range(1, epochs+1):
        final_model.train()
        total_loss = 0
        # Training step
        for batch_data in all_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            out = final_model(batch_data)
            target = batch_data.y.view(-1, 1)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} - Train MSE: {total_loss/len(all_loader):.4f}")
    
    # Save the final model
    torch.save(final_model.state_dict(), f"{target_name}_improved_final_model.pt")
    print(f"Final improved {target_name} model saved to {target_name}_improved_final_model.pt")
    
    # Evaluate the model on all data to get a sense of training fit
    final_model.eval()
    with torch.no_grad():
        all_preds, all_trues = [], []
        total_loss = 0
        for batch_data in all_loader:
            batch_data = batch_data.to(device)
            pred = final_model(batch_data)
            target = batch_data.y.view(-1, 1)
            all_preds.append(pred)
            all_trues.append(target)
            total_loss += criterion(pred, target).item()
        all_preds = torch.cat(all_preds, dim=0)
        all_trues = torch.cat(all_trues, dim=0)
        
        # Calculate metrics
        mse = total_loss / len(all_loader)
        rmse = mse**0.5
        print(f"Training performance on all data: MSE = {mse:.4f}, RMSE = {rmse:.4f}")

#%% Combined Performance Evaluation - For each target model
print("\n#%% Overall Performance Metrics")
for target_name in target_names:
    preds = np.array(results_by_target[target_name]['preds'])
    trues = np.array(results_by_target[target_name]['trues'])
    
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(trues, preds)
    
    print(f"{target_name} model cross-validation metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Create scatter plot of predicted vs actual
    plt.figure(figsize=(10, 8))
    plt.scatter(trues, preds, alpha=0.6, edgecolor='k')
    
    # Add 45-degree line for reference
    min_val = min(min(trues), min(preds))
    max_val = max(max(trues), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'{target_name} Improved Model - Predicted vs. Actual\nMSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    
    # Add annotated text
    plt.text(0.05, 0.95, f'All Validation Folds Combined', 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.savefig(f'{target_name}_improved_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create combined plot with both targets side by side
plt.figure(figsize=(15, 7))
for i, target_name in enumerate(target_names):
    preds = np.array(results_by_target[target_name]['preds'])
    trues = np.array(results_by_target[target_name]['trues'])
    
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(trues, preds)
    
    plt.subplot(1, 2, i+1)
    plt.scatter(trues, preds, alpha=0.6, edgecolor='k')
    
    # Add 45-degree line for reference
    min_val = min(min(trues), min(preds))
    max_val = max(max(trues), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'{target_name} Improved Model\nMSE={mse:.4f}, R²={r2:.4f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('combined_improved_target_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
