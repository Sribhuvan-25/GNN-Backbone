import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from torch_geometric.data import Data
import networkx as nx

class MicrobialGNNDataset:
    """Dataset class for GNN-based regression on microbial data"""
    
    def __init__(self, data_path, k_neighbors=5, mantel_threshold=0.05, use_fast_correlation=True, graph_mode='otu', family_filter_mode='relaxed'):
        """
        Initialize the microbial GNN dataset
        
        Args:
            data_path: Path to the CSV file containing microbial abundance data
            k_neighbors: Number of neighbors for KNN graph construction
            mantel_threshold: P-value threshold for Mantel test
            use_fast_correlation: If True, use fast correlation-based graph construction
            graph_mode: Mode for graph construction ('otu' or 'family')
            family_filter_mode: Mode for family filtering ('relaxed' or 'strict')
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        self.use_fast_correlation = use_fast_correlation
        self.graph_mode = graph_mode
        self.family_filter_mode = family_filter_mode
        
        # Initialize data containers
        self.feature_df = None
        self.target_df = None
        self.node_feature_names = None
        self.target_names = None
        self.data_list = None
        
        # Graph data containers
        self.edge_index = None
        self.edge_weight = None
        self.edge_type = None
        self.original_node_count = None  # Track original node count before pruning
        self.original_graph_data = None
        self.explainer_sparsified_graph_data = None
        
        # Load and process data
        self._load_data()
        
        # Create node features (must be done before graph structure)
        self.df_features, self.feature_matrix = self._create_node_features()
        
        # Create graph structure (now feature_matrix is available)
        self.full_edge_index, self.full_edge_weight, self.full_edge_type = self._create_graph_structure()
        
        # Create KNN sparsified graph structure (always use KNN for initial graph)
        self.edge_index, self.edge_weight, self.edge_type = self._create_knn_graph(k=self.k_neighbors)
        
        # Create PyG data objects
        self.data_list = self._create_data_objects()
        
        # Store original data list for reset capability (after data_list is created)
        self.original_data_list = [data.clone() for data in self.data_list]
        
        # Store original graph data for visualization
        self.original_graph_data = {
            'edge_index': self.edge_index.clone(),
            'edge_weight': self.edge_weight.clone(),
            'edge_type': self.edge_type.clone(),
            'original_node_names': self.node_feature_names.copy()  # Store original node names
        }
        
        # Initialize explainer-sparsified graph data as None
        self.explainer_sparsified_graph_data = None
        
        # Create directory for visualizations
        os.makedirs('graph_visualizations', exist_ok=True)
        
        # Save original state for reset capability
        self._save_original_state()
    
    def _save_original_state(self):
        """Save the original dataset state for resetting between targets"""
        self.original_node_feature_names = self.node_feature_names.copy() if self.node_feature_names else None
        self.original_data_list = None  # Will be set after data_list is created
        
    def reset_to_original_state(self):
        """Reset dataset to original state before any pruning"""
        print("Resetting dataset to original state...")
        
        # Reset node feature names
        if self.original_node_feature_names:
            self.node_feature_names = self.original_node_feature_names.copy()
        
        # Reset data list to original
        if self.original_data_list:
            self.data_list = [data.clone() for data in self.original_data_list]
        else:
            # Recreate data list from scratch
            self.data_list = self._create_data_objects()
        
        # Reset explainer data
        self.explainer_sparsified_graph_data = None
        
        # Reset graph data to original k-NN graph
        if hasattr(self, 'original_graph_data') and self.original_graph_data:
            self.edge_index = self.original_graph_data['edge_index'].clone()
            self.edge_weight = self.original_graph_data['edge_weight'].clone()  
            self.edge_type = self.original_graph_data['edge_type'].clone()
            
        print(f"Dataset reset complete - back to {len(self.node_feature_names)} nodes")
    
    def _load_data(self):
        """Load and process the data"""
        # Load and process the data
        self.df = pd.read_csv(self.data_path)
        
        # Filter out rows containing 'x' values if they exist
        if self.df.isin(['x']).any().any():
            self.df = self.df[~self.df.isin(['x']).any(axis=1)]
        
        # Identify feature and target columns based on graph mode
        # Use only the two specific target columns requested
        self.target_cols = ['ACE-km', 'H2-km']
        
        # Check if the target columns exist in the data
        missing_targets = [col for col in self.target_cols if col not in self.df.columns]
        if missing_targets:
            print(f"Warning: Missing target columns: {missing_targets}")
            # Use only the available target columns
            self.target_cols = [col for col in self.target_cols if col in self.df.columns]
        
        if self.graph_mode == 'otu':
            # Identify OTU feature columns (likely with taxonomic identifiers)
            # Look for columns with taxonomic patterns like 'd__', 'k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__'
            taxonomic_patterns = ['d__', 'k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
            self.otu_cols = []
            
            for col in self.df.columns:
                if col not in self.target_cols:  # Exclude target columns
                    # Check if column contains taxonomic patterns
                    if any(pattern in col for pattern in taxonomic_patterns):
                        self.otu_cols.append(col)
            
            self.feature_cols = self.otu_cols
            print(f"OTU mode: Identified {len(self.feature_cols)} OTU feature columns and {len(self.target_cols)} target columns")
            
            # Verify we have the expected number of OTU columns
            if len(self.feature_cols) != 1086:
                print(f"Warning: Expected 1086 OTU columns, found {len(self.feature_cols)}")
                # Show a few examples of feature column names for debugging
                print(f"Sample feature columns: {self.feature_cols[:5]}")
                if len(self.feature_cols) > 5:
                    print(f"... and {len(self.feature_cols) - 5} more")
                    
        elif self.graph_mode == 'family':
            # First identify OTU columns, then process at family level
            taxonomic_patterns = ['d__', 'k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
            self.otu_cols = []
            
            for col in self.df.columns:
                if col not in self.target_cols:  # Exclude target columns
                    # Check if column contains taxonomic patterns
                    if any(pattern in col for pattern in taxonomic_patterns):
                        self.otu_cols.append(col)
            
            print(f"Family mode: Found {len(self.otu_cols)} OTU columns, processing at family level...")
            
            # Process families and get filtered family columns
            self.df_family_filtered, self.feature_cols = self._process_families()
            
            print(f"Family mode: Selected {len(self.feature_cols)} family feature columns and {len(self.target_cols)} target columns")
        
        else:
            raise ValueError(f"Invalid graph_mode: {self.graph_mode}. Must be 'otu' or 'family'")
        
        print(f"Target columns: {self.target_cols}")
        
        # Extract and clean target data
        self.target_df = self.df[self.target_cols].copy()
        self.target_df.columns = self.target_df.columns.str.strip()
        
        # Store original indices before cleaning
        original_indices = self.target_df.index
        
        # Clean target data
        self.target_df = self._clean_target_data(self.target_df)
        
        # Store target names
        self.target_names = list(self.target_df.columns)
        
        # Update target column names after cleaning
        self.target_cols = list(self.target_df.columns)
        
        # If rows were removed during cleaning, update the main dataframe
        if len(self.target_df) != len(original_indices):
            self.df = self.df.loc[self.target_df.index]
        
    def _create_node_features(self):
        """Create node features from input data"""
        # Extract feature data based on graph mode
        if self.graph_mode == 'otu':
            df_features = self.df[self.feature_cols].copy()
        elif self.graph_mode == 'family':
            # Use the already filtered family data
            df_features = self.df_family_filtered.copy()
        else:
            raise ValueError(f"Invalid graph_mode: {self.graph_mode}")
        
        # Apply variance stabilization if needed
        # For microbial data, double square root transformation is common
        # Adjust this based on your data characteristics
        if df_features.min().min() >= 0:  # Check if all values are non-negative
            df_features = df_features.apply(lambda x: np.sqrt(np.sqrt(x + 1e-10)))
        
        # Convert to numpy array with shape [num_features, num_samples]
        feature_matrix = df_features.values.T.astype(np.float32)
        
        print(f"Node feature matrix: {feature_matrix.shape} (features × samples)")
        
        # Store the feature names for later use
        self.node_feature_names = list(df_features.columns)
        # Track original node count before any pruning
        if self.original_node_count is None:
            self.original_node_count = len(self.node_feature_names)
        
        return df_features, feature_matrix
    
    def _compute_distance_matrix(self, vec, metric='euclidean'):
        """Compute distance matrix between samples for a given feature"""
        if vec.ndim == 1:
            vec = vec.reshape(-1, 1)
        dm = squareform(pdist(vec, metric=metric))
        return np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _mantel_test(self, d1, d2, permutations=100):  # Reduced permutations for efficiency
        """Mantel test to assess correlation between two distance matrices"""
        n = d1.shape[0]
        idx = np.triu_indices(n, k=1)
        v1, v2 = d1[idx], d2[idx]
        
        if v1.std() == 0 or v2.std() == 0:
            return 1.0, 0.0
            
        r_obs = pearsonr(v1, v2)[0]
        
        # Reduced permutation test for efficiency
        count = 0
        for _ in range(permutations):
            perm_v2 = np.random.permutation(v2)
            r_perm = abs(pearsonr(v1, perm_v2)[0])
            if r_perm >= abs(r_obs):
                count += 1
                
        p_value = (count + 1) / (permutations + 1)
        return p_value, r_obs
    
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
        
        print(f"Total families before filtering: {df_fam_rel.shape[1]}")
        
        # FILTERING CRITERIA based on family_filter_mode
        presence_count = (df_fam_rel > 0).sum(axis=0)
        prevalence = presence_count / df_fam_rel.shape[0]
        mean_abund = df_fam_rel.mean(axis=0)
        
        # Set thresholds based on filter mode
        if self.family_filter_mode == 'strict':
            prevalence_threshold = 0.05  # 5% of samples
            abundance_threshold = 0.01   # 1% mean abundance
            use_intersection = True      # Both criteria must be met
            target_min_families = 20
        elif self.family_filter_mode == 'relaxed':
            prevalence_threshold = 0.02  # 2% of samples  
            abundance_threshold = 0.001  # 0.1% mean abundance
            use_intersection = False     # Either criterion (UNION)
            target_min_families = 50
        elif self.family_filter_mode == 'permissive':
            prevalence_threshold = 0.018 # ~1 sample (1.8%)
            abundance_threshold = 0.0005 # 0.05% mean abundance
            use_intersection = False     # Either criterion (UNION)
            target_min_families = 80
        else:
            raise ValueError(f"Invalid family_filter_mode: {self.family_filter_mode}")
        
        high_prev = prevalence[prevalence >= prevalence_threshold].index
        high_abund = mean_abund[mean_abund >= abundance_threshold].index
        
        # Apply filtering logic
        if use_intersection:
            selected_families = high_prev.intersection(high_abund)
            filter_method = "INTERSECTION (prevalence AND abundance)"
        else:
            selected_families = high_prev.union(high_abund)
            filter_method = "UNION (prevalence OR abundance)"
        
        # If still too few families, use ultra-permissive criteria
        if len(selected_families) < target_min_families:
            print(f"Only {len(selected_families)} families with {self.family_filter_mode} criteria. Using ultra-permissive filtering...")
            ultra_prev = prevalence[prevalence >= 0.015].index  # ~0.8 samples
            ultra_abund = mean_abund[mean_abund >= 0.0001].index  # 0.01% abundance
            selected_families = ultra_prev.union(ultra_abund)
            filter_method = "ULTRA-PERMISSIVE (prevalence OR abundance)"
        
        # Ensure we don't include completely absent families
        non_zero_families = df_fam_rel.columns[df_fam_rel.sum(axis=0) > 0]
        selected_families = selected_families.intersection(non_zero_families)
        
        df_fam_rel_filtered = df_fam_rel[selected_families].copy()
        
        print(f"Selected {len(selected_families)} families after {self.family_filter_mode} filtering (out of {df_fam_rel.shape[1]}).")
        print(f"Filter mode: {self.family_filter_mode}")
        print(f"Prevalence threshold: {prevalence_threshold*100:.1f}% of samples")
        print(f"Abundance threshold: {abundance_threshold*100:.3f}% mean abundance")
        print(f"Filtering method: {filter_method}")
        
        # Show some statistics
        final_prevalence = (df_fam_rel_filtered > 0).sum(axis=0) / df_fam_rel_filtered.shape[0]
        final_abundance = df_fam_rel_filtered.mean(axis=0)
        
        print(f"Final prevalence range: {final_prevalence.min():.3f} - {final_prevalence.max():.3f}")
        print(f"Final abundance range: {final_abundance.min():.6f} - {final_abundance.max():.3f}")
        
        return df_fam_rel_filtered, list(df_fam_rel_filtered.columns)
    
    def _create_graph_structure(self):
        """Create graph structure based on correlation or distance metrics"""
        if self.use_fast_correlation:
            return self._create_graph_structure_fast()
        else:
            return self._create_graph_structure_mantel()
    
    def _create_graph_structure_fast(self):
        """Create graph structure using correlation-based approach (much faster than Mantel tests)"""
        print("Constructing graph using fast correlation method...")
        
        num_features = len(self.node_feature_names)
        print(f"Computing correlations for {num_features} features...")
        
        # Use direct correlation approach which is much faster
        # Compute correlation matrix between features
        correlation_matrix = np.corrcoef(self.feature_matrix)
        
        # Replace NaN values with 0
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        edge_i, edge_j, edge_weights = [], [], []
        
        # Adaptive correlation threshold based on feature count
        if num_features <= 30:
            correlation_threshold = 0.7  # Very strict for small graphs
        elif num_features <= 100:
            correlation_threshold = 0.5  # Moderate for medium graphs
        else:
            correlation_threshold = 0.3  # Relaxed for large graphs
            
        print(f"Using correlation threshold {correlation_threshold} for {num_features} features")
        
        # Create edges based on correlation strength
        for i in range(num_features):
            for j in range(i+1, num_features):
                corr = correlation_matrix[i, j]
                
                # Use absolute correlation above threshold
                if abs(corr) > correlation_threshold:
                    edge_i += [i, j]
                    edge_j += [j, i]
                    edge_weights += [abs(corr), abs(corr)]
        
        # If no edges meet the strict threshold, use top-k approach
        if len(edge_i) == 0:
            print(f"No edges meet threshold {correlation_threshold}, using top-k approach...")
            # Select top k pairs by correlation strength
            k = min(num_features * 2, 40)  # Adaptive k based on num_features
            
            # Get all correlation pairs
            corr_pairs = []
            for i in range(num_features):
                for j in range(i+1, num_features):
                    corr = abs(correlation_matrix[i, j])
                    if corr > 0.1:  # Minimum threshold to avoid noise
                        corr_pairs.append((i, j, corr))
            
            # Sort by correlation strength and take top k
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            corr_pairs = corr_pairs[:k]
            
            print(f"Selected top {len(corr_pairs)} edges by correlation strength")
            
            for i, j, corr in corr_pairs:
                edge_i += [i, j]
                edge_j += [j, i]
                edge_weights += [corr, corr]
        
        # Create edge types based on correlation sign
        edge_types = []
        for i, j in zip(edge_i, edge_j):
            corr = correlation_matrix[i, j] if i < len(correlation_matrix) and j < len(correlation_matrix) else 0
            edge_type = 1 if corr > 0 else 0
            edge_types.append(edge_type)
        
        print(f"Created graph with {len(edge_i)//2} undirected edges using correlation method")
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_weight, edge_type
    
    def _create_graph_structure_mantel(self):
        """Create graph structure using Mantel test approach (slower but more rigorous)"""
        print("Constructing graph using Mantel test method (this may take a while)...")
        
        # Compute distance matrices for each feature
        dist_mats = {
            feat: self._compute_distance_matrix(self.feature_matrix[i], metric='euclidean')
            for i, feat in enumerate(self.node_feature_names)
        }
        
        # Construct edges based on Mantel test
        num_features = len(self.node_feature_names)
        edge_i, edge_j, edge_weights = [], [], []
        
        # Create edges between features with significant correlation
        for i in range(num_features):
            for j in range(i+1, num_features):
                # Use Mantel test to assess correlation
                p, r = self._mantel_test(
                    dist_mats[self.node_feature_names[i]],
                    dist_mats[self.node_feature_names[j]],
                    permutations=100  # Reduced for efficiency
                )
                
                if p < self.mantel_threshold:  # Significant correlation
                    # Calculate correlation coefficient to get the sign
                    corr, _ = pearsonr(self.feature_matrix[i], self.feature_matrix[j])
                    # Use actual signed correlation as edge weight (preserve sign information)
                    signed_weight = corr
                    edge_i += [i, j]
                    edge_j += [j, i]
                    edge_weights += [signed_weight, signed_weight]  # Use signed correlation as edge weight
        
        # Create edge types based on correlation sign
        edge_types = []
        for i, j in zip(edge_i, edge_j):
            # Calculate correlation coefficient between the two features
            corr, _ = pearsonr(self.feature_matrix[i], self.feature_matrix[j])
            # Edge type: 0 = negative correlation, 1 = positive correlation
            edge_type = 1 if corr > 0 else 0
            edge_types.append(edge_type)
        
        print(f"Created graph with {len(edge_i)//2} undirected edges using Mantel test")
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_weight, edge_type
    
    def _create_knn_graph(self, k=None):
        """Create a k-nearest neighbor sparsified version of the graph"""
        if k is None:
            k = self.k_neighbors
            
        print(f"Creating KNN graph with k={k}...")
        
        # Create adjacency matrix from full edge_index
        num_nodes = len(self.node_feature_names)
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for i in range(self.full_edge_index.shape[1]):
            u, v = self.full_edge_index[0, i], self.full_edge_index[1, i]
            adj_matrix[u, v] = self.full_edge_weight[i]
        
        # KNN sparsification
        adj_matrix_np = adj_matrix.numpy()
        
        # For each node, keep only the k strongest connections by absolute value
        for i in range(num_nodes):
            # Get weights of all neighbors
            neighbors = adj_matrix_np[i]
            
            # Find non-zero connections and keep only top k by absolute weight
            nonzero_mask = neighbors != 0
            if np.sum(nonzero_mask) > k:
                # Sort by absolute value to get strongest connections regardless of sign
                abs_neighbors = np.abs(neighbors)
                threshold = np.sort(abs_neighbors[nonzero_mask])[-k]
                # Keep only edges with absolute weight >= threshold
                keep_mask = abs_neighbors >= threshold
                adj_matrix_np[i, ~keep_mask] = 0
        
        # Make matrix symmetric (undirected graph)
        adj_matrix_np = np.maximum(adj_matrix_np, adj_matrix_np.T)
        
        # Convert back to edge_index and edge_weight format
        new_edge_index = []
        new_edge_weight = []
        new_edge_type = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix_np[i, j] != 0:  # Check for non-zero (can be positive or negative)
                    new_edge_index.append([i, j])
                    new_edge_weight.append(adj_matrix_np[i, j])
                    
                    # Determine edge type based on sign of the weight
                    new_edge_type.append(1 if adj_matrix_np[i, j] > 0 else 0)
        
        new_edge_index = torch.tensor(new_edge_index).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32)
        new_edge_type = torch.tensor(new_edge_type, dtype=torch.long)
        
        print(f"KNN graph has {new_edge_index.shape[1]//2} undirected edges")
        
        # Create immediate visualization of k-NN graph with all node names intact
        self._visualize_knn_graph_immediate(new_edge_index, new_edge_weight, new_edge_type)
        
        return new_edge_index, new_edge_weight, new_edge_type

    def _visualize_knn_graph_immediate(self, edge_index, edge_weight, edge_type):
        """Create immediate visualization of k-NN graph when it's created to preserve node names"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            print("DEBUG: _visualize_knn_graph_immediate called!")
            print(f"DEBUG: node_feature_names = {self.node_feature_names[:5]}...")  # Show first 5 names
            
            # Create output directory - use a generic location first
            viz_dir = './graph_visualizations_debug'
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create NetworkX graph with guaranteed correct node names
            G = nx.Graph()
            
            # Add all nodes with their original names (guaranteed to be intact at this point)
            for i, name in enumerate(self.node_feature_names):
                G.add_node(i, name=name)
            
            # Add edges with uniform styling and weight labels
            edge_labels = {}
            for i in range(0, edge_index.shape[1], 2):  # Process only one direction for undirected edges
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                weight = edge_weight[i].item()
                G.add_edge(u, v, weight=weight)
                edge_labels[(u, v)] = f'{abs(weight):.2f}'
            
            # Create layout
            pos = nx.spring_layout(G, k=6, iterations=100, seed=42)
            
            # Create figure
            plt.figure(figsize=(20, 16))
            
            # Draw nodes with proper names
            node_labels = {i: name for i, name in enumerate(self.node_feature_names)}
            
            # Draw the graph components
            nx.draw_networkx_nodes(G, pos, node_size=800, alpha=0.9, 
                                 node_color=range(len(G.nodes())), cmap=plt.cm.tab20,
                                 edgecolors='black', linewidths=0.5)
            
            # Draw edges with uniform style
            nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.8, edge_color='gray')
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')
            
            # Draw edge weight labels
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
            
            plt.title(f'k-NN Graph - Created Immediately ({len(G.nodes())} nodes, {len(G.edges())} edges)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')
            
            # Save the graph with unique filename
            filename = f"{viz_dir}/knn_graph_IMMEDIATE_CREATION.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"IMMEDIATE k-NN graph visualization saved to: {filename}")
            print(f"DEBUG: Graph had {len(G.nodes())} nodes with names: {list(node_labels.values())[:5]}...")
            
        except Exception as e:
            print(f"Warning: Immediate k-NN visualization failed: {e}")
    
    def _create_data_objects(self):
        """Create PyG Data objects for each sample"""
        # Transpose feature matrix to shape [num_samples, num_features]
        feature_matrix_samples = self.feature_matrix.T
        
        # Create a list of PyG Data objects
        data_list = []
        
        for s in range(feature_matrix_samples.shape[0]):
            # Node features for this sample - feature values
            x = torch.tensor(feature_matrix_samples[s], dtype=torch.float32).view(-1, 1)
            
            # Graph-level targets
            targets = torch.tensor(self.target_df.iloc[s].values, dtype=torch.float32).view(1, -1)
            
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
    
    def visualize_graphs(self, save_dir='graph_visualizations'):
        """Visualize both original and sparsified graphs for comparison"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Get original and current number of nodes
        original_nodes = self.original_node_count if self.original_node_count is not None else len(self.node_feature_names)
        current_nodes = len(self.node_feature_names)
        
        # Visualize original graph
        self._visualize_single_graph(
            self.original_graph_data['edge_index'],
            self.original_graph_data['edge_weight'],
            self.original_graph_data['edge_type'],
            ax1,
            title=f"KNN Graph ({original_nodes} nodes, {self.original_graph_data['edge_index'].shape[1]//2} edges)",
            graph_type='original'
        )
        
        # Check if explainer-sparsified graph exists
        if self.explainer_sparsified_graph_data is not None:
            # Visualize explainer-sparsified graph
            self._visualize_single_graph(
                self.explainer_sparsified_graph_data['edge_index'],
                self.explainer_sparsified_graph_data['edge_weight'],
                self.explainer_sparsified_graph_data['edge_type'],
                ax2,
                title=f"GNNExplainer Graph ({current_nodes} nodes, {self.explainer_sparsified_graph_data['edge_index'].shape[1]//2} edges)",
                graph_type='explainer'
            )
        else:
            ax2.text(0.5, 0.5, "GNNExplainer graph not created yet.",
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            ax2.set_title("GNNExplainer Graph (Not Available)")
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/graph_comparison_OLD_METHOD.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OLD METHOD graph visualization saved to {save_dir}/graph_comparison_OLD_METHOD.png")
        
        # Also create individual high-resolution visualizations
        plt.figure(figsize=(15, 15))
        self._visualize_single_graph(
            self.original_graph_data['edge_index'],
            self.original_graph_data['edge_weight'],
            self.original_graph_data['edge_type'],
            plt.gca(),
            title=f"KNN Graph ({original_nodes} nodes, {self.original_graph_data['edge_index'].shape[1]//2} edges)",
            graph_type='original'
        )
        plt.tight_layout()
        plt.savefig(f"{save_dir}/knn_graph_OLD_SINGLE.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.explainer_sparsified_graph_data is not None:
            plt.figure(figsize=(15, 15))
            self._visualize_single_graph(
                self.explainer_sparsified_graph_data['edge_index'],
                self.explainer_sparsified_graph_data['edge_weight'],
                self.explainer_sparsified_graph_data['edge_type'],
                plt.gca(),
                title=f"GNNExplainer Graph ({current_nodes} nodes, {self.explainer_sparsified_graph_data['edge_index'].shape[1]//2} edges)",
                graph_type='explainer'
            )
            plt.tight_layout()
            plt.savefig(f"{save_dir}/explainer_graph_OLD_SINGLE.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _get_node_labels(self, G, graph_type):
        """Get appropriate node labels for visualization based on graph type."""
        if graph_type == 'explainer' and hasattr(self, 'explainer_sparsified_graph_data'):
            explainer_data = self.explainer_sparsified_graph_data
            if 'pruned_node_names' in explainer_data and explainer_data['pruned_node_names']:
                # Use pruned node names if available
                pruned_names = explainer_data['pruned_node_names']
                return {node: pruned_names[node] if node < len(pruned_names) else f"Node_{node}" 
                       for node in G.nodes()}
            elif 'kept_nodes' in explainer_data:
                # Use original node names for kept nodes
                kept_nodes = explainer_data['kept_nodes']
                return {node: self.node_feature_names[kept_nodes[node]] if node < len(kept_nodes) else f"Node_{node}"
                       for node in G.nodes()}
        elif graph_type == 'original' and hasattr(self, 'original_graph_data') and 'original_node_names' in self.original_graph_data:
            # Use stored original node names for k-NN graph to ensure consistency
            original_names = self.original_graph_data['original_node_names']
            return {node: original_names[node] if node < len(original_names) else f"Node_{node}"
                   for node in G.nodes()}
        
        # Default: use current node feature names (with bounds checking)
        return {node: self.node_feature_names[node] if node < len(self.node_feature_names) else f"Node_{node}"
               for node in G.nodes()}

    def _visualize_single_graph(self, edge_index, edge_weight, edge_type, ax, title, graph_type='original'):
        """Helper method to visualize a single graph"""
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes with appropriate names based on graph type
        if graph_type == 'original' and hasattr(self, 'original_graph_data') and 'original_node_names' in self.original_graph_data:
            # Use stored original node names for k-NN graph
            node_names = self.original_graph_data['original_node_names']
            for i in range(len(node_names)):
                G.add_node(i, name=node_names[i])
        else:
            # Use current node names for explainer graphs
            for i, name in enumerate(self.node_feature_names):
                G.add_node(i, name=name)
        
        # Add edges with weights and types
        for i in range(0, edge_index.shape[1], 2):  # Process only one direction for undirected edges
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            
            # Handle None edge_weight gracefully
            if edge_weight is not None:
                weight = edge_weight[i].item()
            else:
                weight = 1.0  # Default weight
                
            # Handle None edge_type gracefully  
            if edge_type is not None:
                edge_t = edge_type[i].item()
            else:
                edge_t = 1  # Default type (positive)
                
            G.add_edge(u, v, weight=weight, type=edge_t)
        
        # Create layout with bulletproof NetworkX handling
        def create_custom_layout(G):
            """Create a custom circular layout as fallback"""
            import math
            pos = {}
            nodes = list(G.nodes())
            n = len(nodes)
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / n
                pos[node] = (math.cos(angle), math.sin(angle))
            return pos
        
        # Try NetworkX layout with multiple fallbacks
        pos = None
        try:
            # Try basic spring_layout first
            pos = nx.spring_layout(G, k=0.3)
        except Exception as e1:
            try:
                # Try with seed parameter
                pos = nx.spring_layout(G, k=0.3, seed=42)
            except Exception as e2:
                try:
                    # Try with random_state parameter
                    pos = nx.spring_layout(G, k=0.3, random_state=42)
                except Exception as e3:
                    try:
                        # Try without any parameters
                        pos = nx.spring_layout(G)
                    except Exception as e4:
                        try:
                            # Try circular layout
                            pos = nx.circular_layout(G)
                        except Exception as e5:
                            try:
                                # Try shell layout
                                pos = nx.shell_layout(G)
                            except Exception as e6:
                                # Final fallback - custom circular layout
                                print(f"NetworkX layout failed, using custom layout. Errors: {e1}, {e2}, {e3}, {e4}, {e5}, {e6}")
                                pos = create_custom_layout(G)
        
        # Calculate node size - make uniform for better visualization
        # Option 1: Uniform node size for cleaner visualization
        node_size = [800] * len(G.nodes())  # Uniform size for all nodes
        
        # Option 2: Size based on absolute sum of edge weights (commented out)
        # node_size = []
        # for node in G.nodes():
        #     # Use sum of absolute edge weights as node importance
        #     abs_weight_sum = sum(abs(data['weight']) for u, v, data in G.edges(node, data=True))
        #     node_size.append(200 + abs_weight_sum * 1000)  # Scale to visible range
        
        # Use uniform edge styling with weight labels
        edge_colors = []
        edge_width = []
        edge_labels = {}
        
        for u, v, data in G.edges(data=True):
            # Use uniform gray color for all edges
            edge_colors.append('gray')
            
            # Use uniform width for all edges
            edge_width.append(0.8)
            
            # Add edge weight labels (absolute values)
            weight = data['weight']
            edge_labels[(u, v)] = f'{abs(weight):.2f}'
        
        # Try to find communities for node coloring
        try:
            from community import best_partition
            partition = best_partition(G)
            node_colors = [partition[node] for node in G.nodes()]
        except:
            # Fallback if community detection fails
            node_colors = list(range(len(G.nodes)))
        
        # Draw the graph
        nx.draw_networkx(
            G, 
            pos=pos,
            with_labels=True,
            labels=self._get_node_labels(G, graph_type),
            node_size=node_size,
            node_color=node_colors,
            width=edge_width,
            edge_color=edge_colors,
            alpha=0.8,
            cmap=plt.cm.tab20,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        # Draw edge weight labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
        
        ax.set_title(title, fontsize=16)
        ax.axis('off')
    
    def _clean_target_data(self, target_df):
        """Clean and convert target data to numeric format"""
        print("Cleaning target data...")
        print(f"Target columns before cleaning: {list(target_df.columns)}")
        
        # Convert all columns to numeric, replacing non-numeric values with NaN
        for col in target_df.columns:
            target_df[col] = pd.to_numeric(target_df[col], errors='coerce')
        
        # Check for completely missing targets
        missing_data_summary = target_df.isna().sum()
        print(f"Missing values per target: {missing_data_summary.to_dict()}")
        
        # Check for rows with all NaN values and remove them
        all_nan_rows = target_df.isna().all(axis=1)
        if all_nan_rows.any():
            print(f"Removing {all_nan_rows.sum()} samples with all NaN target values")
            target_df = target_df[~all_nan_rows]
        
        # Fill remaining NaN values with column means
        target_df = target_df.fillna(target_df.mean())
        
        # If there are still NaN values (columns with all NaN), fill with 0
        target_df = target_df.fillna(0)
        
        # Convert all columns to float32 to ensure consistent dtypes for PyTorch
        target_df = target_df.astype(np.float32)
        
        print(f"Cleaned target data: {target_df.shape[1]} targets, {target_df.shape[0]} samples")
        print(f"Target data types: {target_df.dtypes.value_counts().to_dict()}")
        print(f"Target value ranges:")
        for col in target_df.columns:
            print(f"  {col}: min={target_df[col].min():.3f}, max={target_df[col].max():.3f}, mean={target_df[col].mean():.3f}")
        
        return target_df 