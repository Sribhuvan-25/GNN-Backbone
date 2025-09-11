"""
Correlation Network Analysis Implementation
Based on Thapa et al. (2023) methodology from "Network-Analysis.pdf"

This implementation replicates the exact correlation analysis process described in:
- Section 2.5: Network analysis using Vegan package in R-studio
- Section 3.3: Network and phylogenetic analysis with co-occurrence networks

Paper Reference: Thapa et al. (2023) "Elucidation of microbial interactions, dynamics, 
and keystone microbes in high pressure anaerobic digestion"
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Tuple, Dict, List, Optional
import community.community_louvain as community_louvain
warnings.filterwarnings('ignore')

class CorrelationNetworkAnalyzer:
    """
    Implementation of correlation-based network analysis following Thapa et al. (2023) methodology.
    
    The paper uses:
    1. Vegan package (R) equivalent data preprocessing
    2. Spearman correlation with significance testing (p < 0.05)
    3. Co-occurrence network construction with nodes as OTUs and edges as correlations
    4. Network analysis metrics: modularity, centrality, clustering coefficient
    5. Keystone species identification through topological importance
    """
    
    def __init__(self, significance_threshold: float = 0.05, correlation_threshold: float = 0.3):
        """
        Initialize the analyzer with thresholds matching the paper's methodology.
        
        Args:
            significance_threshold: p-value threshold for correlation significance (paper uses p < 0.05)
            correlation_threshold: Minimum correlation strength to include edge (inferred from paper results)
        """
        # Paper Section 2.5: "edges indicate the significance level of co-occurrences between each OTU"
        self.significance_threshold = significance_threshold
        self.correlation_threshold = correlation_threshold
        
        # Network storage
        self.correlation_matrix = None
        self.pvalue_matrix = None
        self.network = None
        self.modules = None
        
        # Results storage matching paper's Table 1 structure
        self.network_metrics = {}
        self.keystone_species = {}
        
    def preprocess_data(self, data: pd.DataFrame, abundance_threshold: float = 0.001) -> pd.DataFrame:
        """
        Preprocess microbial abundance data following Vegan package methodology.
        
        Paper Section 3.3: "48 core OTUs were taken from all of the samples (a total of 387 OTUs). 
        The core OTUs were defined as having >0.1 % of the total sequences in each sample."
        
        Args:
            data: Raw microbial abundance data (samples x features)
            abundance_threshold: Minimum relative abundance threshold (paper uses >0.1% = 0.001)
            
        Returns:
            Preprocessed data with core OTUs only
        """
        print(f"Starting data preprocessing...")
        print(f"Original data shape: {data.shape}")
        
        # Convert to relative abundances if not already
        if data.sum(axis=1).max() > 1.1:  # Likely absolute counts
            print("Converting to relative abundances...")
            data = data.div(data.sum(axis=1), axis=0)
        
        # Filter core OTUs based on abundance threshold (Paper methodology)
        print(f"Filtering OTUs with abundance > {abundance_threshold*100}% in each sample...")
        core_otus_mask = (data > abundance_threshold).all(axis=0)
        core_data = data.loc[:, core_otus_mask]
        
        print(f"Core OTUs selected: {core_data.shape[1]} (Paper had 48 core OTUs)")
        print(f"Processed data shape: {core_data.shape}")
        
        # Apply variance stabilization (equivalent to Vegan package preprocessing)
        print("Applying variance stabilization...")
        processed_data = np.log1p(core_data)  # log(1+x) transformation
        
        return processed_data
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, method: str = 'spearman') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlation matrix with statistical significance testing.
        
        Paper Section 3.3: Uses Spearman correlation for co-occurrence analysis.
        "In the network analysis, nodes represent OTUs and edges represent correlations between OTUs"
        
        Args:
            data: Preprocessed abundance data
            method: Correlation method ('spearman' as used in paper)
            
        Returns:
            Tuple of (correlation_matrix, pvalue_matrix)
        """
        print(f"Calculating {method} correlation matrix...")
        
        n_features = data.shape[1]
        correlation_matrix = np.zeros((n_features, n_features))
        pvalue_matrix = np.zeros((n_features, n_features))
        
        feature_names = data.columns.tolist()
        
        # Calculate pairwise correlations with significance testing
        for i in range(n_features):
            for j in range(i, n_features):
                if method == 'spearman':
                    corr, pval = spearmanr(data.iloc[:, i], data.iloc[:, j])
                else:
                    corr, pval = stats.pearsonr(data.iloc[:, i], data.iloc[:, j])
                
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
                pvalue_matrix[i, j] = pval
                pvalue_matrix[j, i] = pval
        
        # Convert to DataFrames
        corr_df = pd.DataFrame(correlation_matrix, index=feature_names, columns=feature_names)
        pval_df = pd.DataFrame(pvalue_matrix, index=feature_names, columns=feature_names)
        
        # Store results
        self.correlation_matrix = corr_df
        self.pvalue_matrix = pval_df
        
        print(f"Correlation matrix calculated: {corr_df.shape}")
        return corr_df, pval_df
    
    def construct_cooccurrence_network(self, correlation_matrix: pd.DataFrame, pvalue_matrix: pd.DataFrame) -> nx.Graph:
        """
        Construct co-occurrence network based on significant correlations.
        
        Paper Section 3.3: "The co-occurrence network consisted of 48 OTUs and 178 edges. 
        Regarding nodes, except for one OTU, 47 of the 48 OTUs were strongly correlated with each other"
        
        Args:
            correlation_matrix: Correlation coefficients
            pvalue_matrix: Statistical significance p-values
            
        Returns:
            NetworkX graph representing co-occurrence network
        """
        print("Constructing co-occurrence network...")
        
        # Create empty graph
        G = nx.Graph()
        
        # Add all OTUs as nodes
        node_names = correlation_matrix.index.tolist()
        G.add_nodes_from(node_names)
        
        # Add edges based on significance and correlation thresholds
        edges_added = 0
        total_possible_edges = len(node_names) * (len(node_names) - 1) // 2
        
        for i, otu1 in enumerate(node_names):
            for j, otu2 in enumerate(node_names[i+1:], i+1):
                corr = correlation_matrix.loc[otu1, otu2]
                pval = pvalue_matrix.loc[otu1, otu2]
                
                # Paper criterion: significant correlations (p < 0.05)
                if pval < self.significance_threshold and abs(corr) >= self.correlation_threshold:
                    G.add_edge(otu1, otu2, 
                              weight=abs(corr), 
                              correlation=corr,
                              pvalue=pval)
                    edges_added += 1
        
        print(f"Network constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        print(f"Paper reference: 48 OTUs and 178 edges")
        print(f"Edge density: {edges_added/total_possible_edges:.3f}")
        
        # Store network
        self.network = G
        return G
    
    def analyze_network_topology(self, network: nx.Graph) -> Dict:
        """
        Analyze network topological properties matching paper's methodology.
        
        Paper Section 3.3 reports:
        - Average network distance: 2.1
        - Longest distance: 5.0
        - Clustering coefficient: 0.52
        - Modularity: 0.53
        
        Returns:
            Dictionary of network metrics
        """
        print("Analyzing network topology...")
        
        metrics = {}
        
        # Basic network properties
        metrics['num_nodes'] = network.number_of_nodes()
        metrics['num_edges'] = network.number_of_edges()
        metrics['density'] = nx.density(network)
        
        # Distance metrics (Paper: average distance 2.1, longest distance 5.0)
        if nx.is_connected(network):
            avg_path_length = nx.average_shortest_path_length(network)
            diameter = nx.diameter(network)
            metrics['average_path_length'] = avg_path_length
            metrics['diameter'] = diameter
            print(f"Average path length: {avg_path_length:.2f} (Paper: 2.1)")
            print(f"Diameter: {diameter} (Paper: 5.0)")
        else:
            print("Network is not connected - calculating for largest component")
            largest_cc = max(nx.connected_components(network), key=len)
            subgraph = network.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
            metrics['average_path_length'] = avg_path_length
            metrics['diameter'] = diameter
        
        # Clustering coefficient (Paper: 0.52)
        clustering_coeff = nx.average_clustering(network)
        metrics['clustering_coefficient'] = clustering_coeff
        print(f"Clustering coefficient: {clustering_coeff:.3f} (Paper: 0.52)")
        
        # Modularity using community detection (Paper: 0.53)
        partition = community_louvain.best_partition(network)
        modularity = community_louvain.modularity(partition, network)
        metrics['modularity'] = modularity
        metrics['num_communities'] = len(set(partition.values()))
        print(f"Modularity: {modularity:.3f} (Paper: 0.53)")
        print(f"Number of communities: {metrics['num_communities']} (Paper: 5 modules)")
        
        # Store community assignments
        self.modules = partition
        
        # Centrality measures for keystone identification
        degree_centrality = nx.degree_centrality(network)
        betweenness_centrality = nx.betweenness_centrality(network)
        closeness_centrality = nx.closeness_centrality(network)
        
        metrics['degree_centrality'] = degree_centrality
        metrics['betweenness_centrality'] = betweenness_centrality
        metrics['closeness_centrality'] = closeness_centrality
        
        # Store metrics
        self.network_metrics = metrics
        
        return metrics
    
    def identify_keystone_species(self, network: nx.Graph, top_k: int = 10) -> Dict:
        """
        Identify keystone microbes based on network topology and correlation strength.
        
        Paper Section 3.3: "Alkaliphilus, Petrimonas, Methanobacterium, and Methanomicrobiaceae 
        could be keystone microbes in the HPAD reactor due to their stronger correlation and 
        critical roles in the anaerobic degradation"
        
        Args:
            network: Co-occurrence network
            top_k: Number of top keystone species to identify
            
        Returns:
            Dictionary of keystone species with their importance metrics
        """
        print("Identifying keystone species...")
        
        # Calculate multiple centrality measures
        degree_centrality = nx.degree_centrality(network)
        betweenness_centrality = nx.betweenness_centrality(network)
        closeness_centrality = nx.closeness_centrality(network)
        
        # Calculate node strength (sum of edge weights)
        node_strength = {}
        for node in network.nodes():
            strength = sum([network[node][neighbor]['weight'] for neighbor in network.neighbors(node)])
            node_strength[node] = strength
        
        # Normalize node strength
        max_strength = max(node_strength.values()) if node_strength.values() else 1
        normalized_strength = {node: strength/max_strength for node, strength in node_strength.items()}
        
        # Composite keystone score (combining multiple measures)
        keystone_scores = {}
        for node in network.nodes():
            score = (degree_centrality[node] + 
                    betweenness_centrality[node] + 
                    closeness_centrality[node] + 
                    normalized_strength[node]) / 4
            keystone_scores[node] = score
        
        # Rank keystone species
        ranked_keystones = sorted(keystone_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create detailed keystone information
        keystone_info = {}
        for i, (species, score) in enumerate(ranked_keystones[:top_k]):
            keystone_info[species] = {
                'rank': i + 1,
                'keystone_score': score,
                'degree_centrality': degree_centrality[species],
                'betweenness_centrality': betweenness_centrality[species],
                'closeness_centrality': closeness_centrality[species],
                'node_strength': node_strength[species],
                'degree': network.degree(species),
                'module': self.modules.get(species, 'Unknown') if self.modules else 'Unknown'
            }
        
        print(f"Top {top_k} keystone species identified:")
        for species, info in keystone_info.items():
            print(f"  {info['rank']}. {species[:50]}... (score: {info['keystone_score']:.3f})")
        
        # Store keystone species
        self.keystone_species = keystone_info
        
        return keystone_info
    
    def visualize_network(self, network: nx.Graph, keystone_species: Dict, 
                         figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None):
        """
        Visualize the co-occurrence network with keystone species highlighted.
        
        Paper visualization shows:
        - Node size proportional to abundance/importance
        - Edge thickness proportional to correlation strength
        - Different colors for different modules
        - Keystone species prominently displayed
        """
        plt.figure(figsize=figsize)
        
        # Set up layout
        pos = nx.spring_layout(network, k=1, iterations=50)
        
        # Node sizes based on degree centrality
        node_sizes = [self.network_metrics['degree_centrality'][node] * 1000 + 100 
                     for node in network.nodes()]
        
        # Node colors based on modules
        if self.modules:
            module_colors = plt.cm.Set3(np.linspace(0, 1, len(set(self.modules.values()))))
            node_colors = [module_colors[self.modules[node]] for node in network.nodes()]
        else:
            node_colors = 'lightblue'
        
        # Highlight keystone species
        keystone_nodes = list(keystone_species.keys())
        keystone_mask = [node in keystone_nodes for node in network.nodes()]
        
        # Draw regular nodes
        regular_nodes = [node for i, node in enumerate(network.nodes()) if not keystone_mask[i]]
        regular_sizes = [node_sizes[i] for i, node in enumerate(network.nodes()) if not keystone_mask[i]]
        regular_colors = [node_colors[i] for i, node in enumerate(network.nodes()) if not keystone_mask[i]] if self.modules else 'lightblue'
        
        if regular_nodes:
            nx.draw_networkx_nodes(network, pos, nodelist=regular_nodes, 
                                 node_size=regular_sizes, node_color=regular_colors, 
                                 alpha=0.7)
        
        # Draw keystone nodes
        keystone_sizes = [node_sizes[i] for i, node in enumerate(network.nodes()) if keystone_mask[i]]
        if keystone_nodes:
            nx.draw_networkx_nodes(network, pos, nodelist=keystone_nodes, 
                                 node_size=[s*1.5 for s in keystone_sizes], 
                                 node_color='red', alpha=0.9, edgecolors='black', linewidths=2)
        
        # Draw edges with thickness proportional to correlation strength
        edges = network.edges()
        weights = [network[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(network, pos, alpha=0.3, 
                             width=[w*5 for w in weights])
        
        # Add labels for keystone species
        keystone_labels = {node: node.split(';g__')[-1][:15] if ';g__' in node else node[:15] 
                          for node in keystone_nodes}
        nx.draw_networkx_labels(network, pos, labels=keystone_labels, 
                               font_size=8, font_weight='bold')
        
        plt.title("Co-occurrence Network Analysis\n(Red nodes = Keystone Species)", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Add network statistics
        stats_text = f"Nodes: {network.number_of_nodes()}\n"
        stats_text += f"Edges: {network.number_of_edges()}\n"
        stats_text += f"Modularity: {self.network_metrics.get('modularity', 'N/A'):.3f}\n"
        stats_text += f"Clustering: {self.network_metrics.get('clustering_coefficient', 'N/A'):.3f}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to: {save_path}")
        
        plt.show()
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                  figsize: Tuple[int, int] = (12, 10), save_path: Optional[str] = None):
        """
        Create correlation heatmap similar to paper's Figure 5b.
        """
        plt.figure(figsize=figsize)
        
        # Mask non-significant correlations
        mask = self.pvalue_matrix >= self.significance_threshold
        
        # Create heatmap
        sns.heatmap(correlation_matrix, mask=mask, cmap='RdBu_r', center=0,
                   square=True, cbar_kws={"shrink": .8}, 
                   xticklabels=False, yticklabels=False)
        
        plt.title('Correlation Matrix Heatmap\n(Only Significant Correlations Shown)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to: {save_path}")
        
        plt.show()
    
    def run_complete_analysis(self, data: pd.DataFrame, output_dir: str = None) -> Dict:
        """
        Run the complete correlation network analysis pipeline.
        
        Args:
            data: Raw microbial abundance data
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all analysis results
        """
        print("=" * 60)
        print("CORRELATION NETWORK ANALYSIS")
        print("Based on Thapa et al. (2023) Methodology")
        print("=" * 60)
        
        # Step 1: Data preprocessing
        processed_data = self.preprocess_data(data)
        
        # Step 2: Calculate correlation matrix
        corr_matrix, pval_matrix = self.calculate_correlation_matrix(processed_data)
        
        # Step 3: Construct co-occurrence network
        network = self.construct_cooccurrence_network(corr_matrix, pval_matrix)
        
        # Step 4: Analyze network topology
        network_metrics = self.analyze_network_topology(network)
        
        # Step 5: Identify keystone species
        keystone_species = self.identify_keystone_species(network)
        
        # Step 6: Generate visualizations
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            network_path = os.path.join(output_dir, "cooccurrence_network.png")
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        else:
            network_path = heatmap_path = None
        
        self.visualize_network(network, keystone_species, save_path=network_path)
        self.create_correlation_heatmap(corr_matrix, save_path=heatmap_path)
        
        # Compile results
        results = {
            'processed_data': processed_data,
            'correlation_matrix': corr_matrix,
            'pvalue_matrix': pval_matrix,
            'network': network,
            'network_metrics': network_metrics,
            'keystone_species': keystone_species,
            'modules': self.modules
        }
        
        print("=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        return results


if __name__ == "__main__":
    # Example usage
    print("Correlation Network Analysis Module")
    print("Based on Thapa et al. (2023) methodology")
    print("\nTo use this module:")
    print("1. analyzer = CorrelationNetworkAnalyzer()")
    print("2. results = analyzer.run_complete_analysis(your_data)")