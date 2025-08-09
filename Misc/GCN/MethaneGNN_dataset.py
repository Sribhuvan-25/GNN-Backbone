import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx, k_hop_subgraph
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

class MethaneGNNDataset:
    def __init__(self, data_path='../Data/New_data.csv', k_neighbors=5, mantel_threshold=0.05):
        """
        Dataset class for methane prediction using microbial abundance data
        
        Args:
            data_path: Path to the CSV file with microbial data
            k_neighbors: Number of neighbors to keep in the sparsified graph
            mantel_threshold: p-value threshold for the Mantel test
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        
        # Load and process the data
        self.df = pd.read_csv(data_path)
        
        # Filter out rows containing 'x' values
        self.df = self.df[~self.df.isin(['x']).any(axis=1)]
        
        # Identify OTU and metadata columns
        self.otu_cols = [c for c in self.df.columns if "d__" in c]
        self.meta_cols = [c for c in self.df.columns if c not in self.otu_cols]
        
        # Extract families from taxonomy strings
        self.df_fam, self.node_feature_names = self._process_families()
        
        # Target variables
        self.param_df = self.df[self.meta_cols].copy()
        self.param_df.columns = self.param_df.columns.str.strip()
        
        # Create node features
        self.df_microbe, self.feature_matrix = self._create_node_features()
        
        # Create graphs
        self.edge_index, self.edge_weight, self.edge_type = self._create_graph_structure()
        
        # Create PyG data objects
        self.data_list = self._create_data_objects()
        
        # Store original graph data for visualization
        self.original_graph_data = {
            'edge_index': self.edge_index.clone(),
            'edge_weight': self.edge_weight.clone(),
            'edge_type': self.edge_type.clone()
        }
        
        # Initialize sparsified graph data as None (will be set when create_knn_sparsified_graph is called)
        self.sparsified_graph_data = None
        
        # Create directory for visualizations
        os.makedirs('graph_visualizations', exist_ok=True)
    
    def _process_families(self):
        """Extract family level taxonomy and aggregate OTUs"""
        # Function to extract family from taxonomy string
        def extract_family(colname):
            for part in colname.split(';'):
                part = part.strip()
                if part.startswith('f__'):
                    return part[3:] or "UnclassifiedFamily"
            return "UnclassifiedFamily"
        
        # Map OTUs to families
        col_to_family = {c: extract_family(c) for c in self.otu_cols}
        family_to_cols = {}
        for c, fam in col_to_family.items():
            if fam not in family_to_cols:
                family_to_cols[fam] = []
            family_to_cols[fam].append(c)
        
        # Aggregate OTUs at family level
        df_fam = pd.DataFrame({
            fam: self.df[cols].sum(axis=1)
            for fam, cols in family_to_cols.items()
        }, index=self.df.index)
        
        # Convert to relative abundance
        df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)
        
        # Filter low abundance/prevalence families using the exact thresholds provided
        presence_count = (df_fam_rel > 0).sum(axis=0)
        prevalence = presence_count / df_fam_rel.shape[0]
        high_prev = prevalence[prevalence >= 0.05].index  # Keep families present in at least 5% of samples
        
        mean_abund = df_fam_rel.mean(axis=0)
        high_abund = mean_abund[mean_abund >= 0.01].index  # Keep families with at least 1% mean abundance
        
        selected_families = high_prev.intersection(high_abund)
        df_fam_rel_filtered = df_fam_rel[selected_families].copy()
        
        print(f"Selected {len(selected_families)} families after filtering (out of {df_fam_rel.shape[1]}).")
        
        return df_fam_rel_filtered, list(df_fam_rel_filtered.columns)
    
    def _create_node_features(self):
        """Create node features using double square root transformation"""
        # Apply double square root transformation (variance stabilization)
        df_microbe = self.df_fam.apply(lambda x: np.sqrt(np.sqrt(x)))
        
        # Convert to numpy array with shape [num_families, num_samples]
        feature_matrix = df_microbe.values.T.astype(np.float32)
        
        print(f"Node feature matrix: {feature_matrix.shape} (families × samples)")
        
        return df_microbe, feature_matrix
    
    def _compute_distance_matrix(self, vec, metric='braycurtis'):
        """Compute distance matrix between samples for a given microbial family"""
        dm = squareform(pdist(vec[:, None], metric=metric))
        return np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _mantel_test(self, d1, d2, permutations=999):
        """Mantel test to assess correlation between two distance matrices"""
        n = d1.shape[0]
        idx = np.triu_indices(n, k=1)
        v1, v2 = d1[idx], d2[idx]
        
        if v1.std() == 0 or v2.std() == 0:
            return 1.0, 0.0
            
        r_obs = pearsonr(v1, v2)[0]
        
        # Permutation test
        count = 0
        for _ in range(permutations):
            perm_v2 = np.random.permutation(v2)
            r_perm = abs(pearsonr(v1, perm_v2)[0])
            if r_perm >= abs(r_obs):
                count += 1
                
        p_value = (count + 1) / (permutations + 1)
        return p_value, r_obs
    
    def _create_graph_structure(self):
        """Create graph structure using Mantel test for edge determination"""
        print("Constructing graph using Mantel test...")
        
        # Compute distance matrices for each family
        dist_mats = {
            fam: self._compute_distance_matrix(self.feature_matrix[i], metric='braycurtis')
            for i, fam in enumerate(self.node_feature_names)
        }
        
        # Construct edges based on Mantel test
        num_families = len(self.node_feature_names)
        edge_i, edge_j, edge_weights = [], [], []
        
        # Create edges between families with significant Mantel test
        for i in range(num_families):
            for j in range(i+1, num_families):
                p, r = self._mantel_test(
                    dist_mats[self.node_feature_names[i]],
                    dist_mats[self.node_feature_names[j]],
                    permutations=999
                )
                
                if p < self.mantel_threshold:  # Significant correlation
                    edge_i += [i, j]
                    edge_j += [j, i]
                    edge_weights += [abs(r), abs(r)]  # Use correlation strength as edge weight
        
        # Create edge types based on correlation sign
        edge_types = []
        for i, j in zip(edge_i, edge_j):
            # Calculate correlation coefficient between the two families
            corr, _ = pearsonr(self.feature_matrix[i], self.feature_matrix[j])
            # Edge type: 0 = negative correlation, 1 = positive correlation
            edge_type = 1 if corr > 0 else 0
            edge_types.append(edge_type)
        
        print(f"Created graph with {len(edge_i)//2} undirected edges")
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_weight, edge_type
    
    def _create_data_objects(self):
        """Create PyG Data objects for each sample"""
        # Extract target variables
        target_cols = ['ACE-km', 'H2-km']
        target_data = self.param_df[target_cols].copy()
        target_data = target_data.astype(float)
        
        # Transpose feature matrix to shape [num_samples, num_families]
        feature_matrix_samples = self.feature_matrix.T
        
        # Create a list of PyG Data objects
        data_list = []
        
        for s in range(feature_matrix_samples.shape[0]):
            # Node features for this sample - abundance values
            x = torch.tensor(feature_matrix_samples[s], dtype=torch.float32).view(-1, 1)
            
            # Graph-level targets
            targets = torch.tensor(target_data.iloc[s].values, dtype=torch.float32).view(1, 2)
            
            # Create the Data object
            data = Data(
                x=x,
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
                edge_attr=self.edge_weight.view(-1, 1),
                edge_type=self.edge_type,
                y=targets
            )
            
            data_list.append(data)
        
        print(f"Created {len(data_list)} graph data objects with {len(self.node_feature_names)} nodes each")
        
        return data_list
    
    def get_data(self):
        """Return the list of graph data objects"""
        return self.data_list
    
    def save_processed_data(self, path='processed_methane_data.pt'):
        """Save the processed data for later use"""
        torch.save({
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'edge_type': self.edge_type,
            'node_feature_names': self.node_feature_names,
            'feature_matrix': self.feature_matrix,
            'data_list': self.data_list
        }, path)
        print(f"Saved processed data to {path}")
    
    def create_knn_sparsified_graph(self, k=10):
        """Create a k-nearest neighbor sparsified version of the graph"""
        print(f"Creating KNN sparsified graph with k={k}...")
        
        # Create adjacency matrix from edge_index
        num_nodes = len(self.node_feature_names)
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for i in range(self.edge_index.shape[1]):
            u, v = self.edge_index[0, i], self.edge_index[1, i]
            adj_matrix[u, v] = self.edge_weight[i]
        
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
        new_edge_type = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix_np[i, j] > 0:
                    new_edge_index.append([i, j])
                    new_edge_weight.append(adj_matrix_np[i, j])
                    
                    # Determine edge type (sign of correlation)
                    corr, _ = pearsonr(self.feature_matrix[i], self.feature_matrix[j])
                    new_edge_type.append(1 if corr > 0 else 0)
        
        new_edge_index = torch.tensor(new_edge_index).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32)
        new_edge_type = torch.tensor(new_edge_type, dtype=torch.long)
        
        print(f"KNN sparsified graph has {new_edge_index.shape[1]//2} undirected edges")
        
        # Store sparsified graph data for visualization
        self.sparsified_graph_data = {
            'edge_index': new_edge_index.clone(),
            'edge_weight': new_edge_weight.clone(),
            'edge_type': new_edge_type.clone()
        }
        
        # Create new data objects with sparsified graph
        new_data_list = []
        feature_matrix_samples = self.feature_matrix.T
        target_cols = ['ACE-km', 'H2-km']
        target_data = self.param_df[target_cols].copy().astype(float)
        
        for s in range(feature_matrix_samples.shape[0]):
            x = torch.tensor(feature_matrix_samples[s], dtype=torch.float32).view(-1, 1)
            targets = torch.tensor(target_data.iloc[s].values, dtype=torch.float32).view(1, 2)
            
            data = Data(
                x=x,
                edge_index=new_edge_index,
                edge_weight=new_edge_weight,
                edge_attr=new_edge_weight.view(-1, 1),
                edge_type=new_edge_type,
                y=targets
            )
            
            new_data_list.append(data)
        
        # Save sparsified graph
        torch.save({
            'edge_index': new_edge_index,
            'edge_weight': new_edge_weight,
            'edge_type': new_edge_type,
            'node_feature_names': self.node_feature_names,
            'feature_matrix': self.feature_matrix,
            'data_list': new_data_list
        }, 'sparsified_graph.pt')
        
        print(f"Saved sparsified graph to sparsified_graph.pt")
        
        return new_data_list
    
    def visualize_graphs(self, save_dir='graph_visualizations'):
        """Visualize both original and sparsified graphs for comparison
        
        Args:
            save_dir: Directory to save the visualization images
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Get number of nodes
        num_nodes = len(self.node_feature_names)
        
        # Visualize original graph
        self._visualize_single_graph(
            self.original_graph_data['edge_index'],
            self.original_graph_data['edge_weight'],
            self.original_graph_data['edge_type'],
            ax1,
            title=f"Original Graph ({num_nodes} nodes, {self.original_graph_data['edge_index'].shape[1]//2} edges)"
        )
        
        # Check if sparsified graph exists
        if self.sparsified_graph_data is not None:
            # Visualize sparsified graph
            self._visualize_single_graph(
                self.sparsified_graph_data['edge_index'],
                self.sparsified_graph_data['edge_weight'],
                self.sparsified_graph_data['edge_type'],
                ax2,
                title=f"Sparsified Graph (k={self.k_neighbors}, {num_nodes} nodes, {self.sparsified_graph_data['edge_index'].shape[1]//2} edges)"
            )
        else:
            ax2.text(0.5, 0.5, "Sparsified graph not created yet.\nCall create_knn_sparsified_graph() first.",
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            ax2.set_title("Sparsified Graph (Not Available)")
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/graph_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved to {save_dir}/graph_comparison.png")
        
        # Also create individual high-resolution visualizations
        plt.figure(figsize=(15, 15))
        self._visualize_single_graph(
            self.original_graph_data['edge_index'],
            self.original_graph_data['edge_weight'],
            self.original_graph_data['edge_type'],
            plt.gca(),
            title=f"Original Microbial Interaction Graph ({num_nodes} nodes, {self.original_graph_data['edge_index'].shape[1]//2} edges)"
        )
        plt.tight_layout()
        plt.savefig(f"{save_dir}/original_graph.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.sparsified_graph_data is not None:
            plt.figure(figsize=(15, 15))
            self._visualize_single_graph(
                self.sparsified_graph_data['edge_index'],
                self.sparsified_graph_data['edge_weight'],
                self.sparsified_graph_data['edge_type'],
                plt.gca(),
                title=f"Sparsified Microbial Interaction Graph (k={self.k_neighbors}, {num_nodes} nodes, {self.sparsified_graph_data['edge_index'].shape[1]//2} edges)"
            )
            plt.tight_layout()
            plt.savefig(f"{save_dir}/sparsified_graph.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _visualize_single_graph(self, edge_index, edge_weight, edge_type, ax, title):
        """Helper method to visualize a single graph
        
        Args:
            edge_index: Edge indices tensor [2, num_edges]
            edge_weight: Edge weights tensor [num_edges]
            edge_type: Edge types tensor [num_edges]
            ax: Matplotlib axis to draw on
            title: Plot title
        """
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes with family names
        for i, family in enumerate(self.node_feature_names):
            G.add_node(i, name=family)
        
        # Add edges with weights and types
        for i in range(0, edge_index.shape[1], 2):  # Process only one direction for undirected edges
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            weight = edge_weight[i].item()
            edge_t = edge_type[i].item()
            G.add_edge(u, v, weight=weight, type=edge_t)
        
        # Calculate node centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
        except:
            # Fallback if eigenvector centrality fails
            eigenvector_centrality = {node: 0.1 for node in G.nodes()}
        
        # Create a combined centrality measure
        combined_centrality = {}
        for node in G.nodes():
            combined_centrality[node] = (
                0.4 * degree_centrality[node] + 
                0.3 * betweenness_centrality[node] + 
                0.3 * eigenvector_centrality[node]
            )
        
        # Create layout
        pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42, weight='weight')
        
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
        
        # Try to find communities in the graph
        try:
            communities = nx.community.greedy_modularity_communities(G)
            community_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = i
            node_colors = [community_map.get(node, 0) for node in G.nodes()]
        except:
            # Fallback if community detection fails
            node_colors = list(range(len(G.nodes)))
        
        # Draw the graph
        nx.draw_networkx(
            G, 
            pos=pos,
            with_labels=True,  # Show all labels
            labels={node: self.node_feature_names[node] for node in G.nodes()},
            node_size=node_size,
            node_color=node_colors,
            width=edge_width,
            edge_color=edge_colors,
            alpha=0.8,
            cmap=plt.cm.tab20,
            font_size=8,  # Smaller font size to avoid overlap
            font_weight='bold',
            font_color='black',
            ax=ax
        )
        
        # Create a legend for edge types
        ax.plot([], [], 'g-', linewidth=2, label='Positive correlation')
        ax.plot([], [], 'r-', linewidth=2, label='Negative correlation')
        ax.legend(loc='upper right')
        
        ax.set_title(title, fontsize=16)
        ax.axis('off') 