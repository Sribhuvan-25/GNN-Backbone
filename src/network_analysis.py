#!/usr/bin/env python3
"""
Network Analysis of Microbial Correlation Graph

This script performs comprehensive network analysis on the Spearman correlation graph
including centrality measures, community detection, and biological interpretation.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def load_correlation_graph_from_dataset():
    """Load the correlation graph from the dataset pipeline"""
    try:
        # Import the dataset to get the exact same graph structure
        import sys
        sys.path.append('.')
        from datasets.domain_expert_dataset import AnchoredMicrobialGNNDataset

        # Create the same dataset as used in the pipeline
        dataset = AnchoredMicrobialGNNDataset(
            data_path='../Data/New_Data.csv',
            case_type='case1',
            anchored_features=['Methanoregulaceae', 'Methanobacteriaceae', 'Methanospirillaceae'],
            graph_construction_method='paper_correlation',
            k_neighbors=10
        )

        # Get the correlation graph data
        edge_index = dataset.original_graph_data['edge_index']
        edge_weight = dataset.original_graph_data['edge_weight']
        node_names = dataset.node_feature_names

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes with names
        for i, name in enumerate(node_names):
            G.add_node(i, name=name)

        # Add edges with weights
        for i in range(0, edge_index.shape[1], 2):  # Undirected edges
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            weight = edge_weight[i].item()
            G.add_edge(u, v, weight=abs(weight), correlation=weight)  # Use absolute weight for analysis

        return G, node_names

    except Exception as e:
        print(f"Error loading from dataset: {e}")
        return None, None

def classify_microorganisms(node_names):
    """
    Classify microorganisms based on their known metabolic roles in anaerobic digestion
    """
    classifications = {}

    # Methanogenic archaea (methane producers)
    methanogens = [
        'Methanobacteriaceae',      # Hydrogenotrophic methanogens
        'Methanoregulaceae',        # Hydrogenotrophic methanogens
        'Methanospirillaceae',      # Hydrogenotrophic methanogens
        'Methanosaetaceae'          # Acetoclastic methanogens
    ]

    # Syntrophic bacteria (produce substrates for methanogens)
    syntrophs = [
        'Syntrophobacteraceae',     # Syntrophic propionate oxidizers
        'Syntrophorhabdaceae',      # Syntrophic fatty acid oxidizers
        'Smithellaceae',            # Syntrophic propionate oxidizers
        'Syntrophaceae',            # Syntrophic fatty acid oxidizers
        'Syntrophomonadaceae'       # Syntrophic fatty acid oxidizers
    ]

    # Fermentative bacteria (primary fermenters)
    fermenters = [
        'Rikenellaceae',            # Fermentative bacteria
        'Christensenellaceae',      # Fermentative bacteria
        'Cloacimonadaceae',         # Fermentative bacteria
        'Bacteroidetes_vadinHA17',  # Fermentative bacteria
        'Dysgonomonadaceae'         # Fermentative bacteria
    ]

    # Sulfate reducers and other anaerobic bacteria
    other_anaerobes = [
        'SBR1031',                  # Sulfate-reducing bacteria
        'Geobacteraceae',           # Anaerobic metal/sulfur reducers
        'Gracilibacteraceae',       # Anaerobic bacteria
        'Desulfotomaculales'        # Sulfate-reducing bacteria
    ]

    # Classify each organism
    for name in node_names:
        found = False
        for family in methanogens:
            if family in name:
                classifications[name] = 'Methanogen'
                found = True
                break

        if not found:
            for family in syntrophs:
                if family in name:
                    classifications[name] = 'Syntrophic'
                    found = True
                    break

        if not found:
            for family in fermenters:
                if family in name:
                    classifications[name] = 'Fermentative'
                    found = True
                    break

        if not found:
            for family in other_anaerobes:
                if family in name:
                    classifications[name] = 'Other_Anaerobe'
                    found = True
                    break

        if not found:
            classifications[name] = 'Unclassified'

    return classifications

def calculate_centrality_measures(G):
    """Calculate various centrality measures"""
    print("Calculating centrality measures...")

    centrality_measures = {}

    # Degree centrality
    centrality_measures['degree'] = nx.degree_centrality(G)

    # Betweenness centrality
    centrality_measures['betweenness'] = nx.betweenness_centrality(G, weight='weight')

    # Closeness centrality
    centrality_measures['closeness'] = nx.closeness_centrality(G, distance='weight')

    # Eigenvector centrality
    try:
        centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except:
        print("Eigenvector centrality failed, using unweighted version")
        centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)

    # PageRank
    centrality_measures['pagerank'] = nx.pagerank(G, weight='weight')

    return centrality_measures

def detect_communities(G):
    """Detect communities using multiple algorithms"""
    print("Detecting communities...")

    communities = {}

    try:
        # Louvain community detection (best for weighted networks)
        import community as community_louvain
        louvain_communities = community_louvain.best_partition(G, weight='weight')
        communities['louvain'] = louvain_communities

        # Calculate modularity
        modularity = community_louvain.modularity(louvain_communities, G, weight='weight')
        print(f"Louvain modularity: {modularity:.3f}")

    except ImportError:
        print("python-louvain not available, using NetworkX community detection")

        # Use NetworkX greedy modularity communities
        greedy_communities = nx.community.greedy_modularity_communities(G, weight='weight')
        louvain_communities = {}
        for i, community in enumerate(greedy_communities):
            for node in community:
                louvain_communities[node] = i
        communities['louvain'] = louvain_communities

    # Edge betweenness community detection
    try:
        edge_communities = nx.community.edge_betweenness_communities(G)
        edge_comm_dict = {}
        for i, community in enumerate(edge_communities):
            for node in community:
                edge_comm_dict[node] = i
        communities['edge_betweenness'] = edge_comm_dict
    except:
        print("Edge betweenness community detection failed")

    return communities

def analyze_network_topology(G):
    """Analyze basic network topology metrics"""
    print("Analyzing network topology...")

    topology = {}

    # Basic metrics
    topology['nodes'] = G.number_of_nodes()
    topology['edges'] = G.number_of_edges()
    topology['density'] = nx.density(G)
    topology['average_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()

    # Connectivity
    topology['connected_components'] = nx.number_connected_components(G)
    topology['largest_component_size'] = len(max(nx.connected_components(G), key=len))

    # Path metrics
    if nx.is_connected(G):
        topology['diameter'] = nx.diameter(G)
        topology['average_path_length'] = nx.average_shortest_path_length(G)
    else:
        # Use largest component
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
        topology['diameter'] = nx.diameter(largest_cc)
        topology['average_path_length'] = nx.average_shortest_path_length(largest_cc)

    # Clustering
    topology['average_clustering'] = nx.average_clustering(G, weight='weight')
    topology['transitivity'] = nx.transitivity(G)

    # Small world properties
    # Compare to random graph
    random_G = nx.erdos_renyi_graph(G.number_of_nodes(), topology['density'])
    random_clustering = nx.average_clustering(random_G)
    random_path_length = nx.average_shortest_path_length(random_G) if nx.is_connected(random_G) else float('inf')

    topology['small_world_sigma'] = (topology['average_clustering'] / random_clustering) / (topology['average_path_length'] / random_path_length) if random_path_length != float('inf') else 'N/A'

    return topology

def create_centrality_analysis_report(G, node_names, centrality_measures, classifications, output_dir):
    """Create detailed centrality analysis report"""

    # Create DataFrame with all centrality measures
    centrality_df = pd.DataFrame(index=range(len(node_names)))
    centrality_df['node_id'] = centrality_df.index
    centrality_df['organism'] = [node_names[i] for i in centrality_df.index]
    centrality_df['functional_group'] = [classifications.get(node_names[i], 'Unclassified') for i in centrality_df.index]
    centrality_df['degree'] = [G.degree(i) for i in centrality_df.index]

    # Add centrality measures
    for measure, values in centrality_measures.items():
        centrality_df[f'{measure}_centrality'] = [values[i] for i in centrality_df.index]

    # Save centrality analysis
    centrality_df.to_csv(f"{output_dir}/centrality_analysis.csv", index=False)

    # Create top organisms report for each centrality measure
    top_organisms = {}
    for measure in centrality_measures.keys():
        top_5 = centrality_df.nlargest(5, f'{measure}_centrality')[['organism', 'functional_group', f'{measure}_centrality']]
        top_organisms[measure] = top_5

    return centrality_df, top_organisms

def create_community_analysis_report(G, node_names, communities, classifications, output_dir):
    """Create detailed community analysis report"""

    community_reports = {}

    for method, comm_dict in communities.items():
        # Create DataFrame
        comm_df = pd.DataFrame(index=range(len(node_names)))
        comm_df['node_id'] = comm_df.index
        comm_df['organism'] = [node_names[i] for i in comm_df.index]
        comm_df['functional_group'] = [classifications.get(node_names[i], 'Unclassified') for i in comm_df.index]
        comm_df['community'] = [comm_dict.get(i, -1) for i in comm_df.index]
        comm_df['degree'] = [G.degree(i) for i in comm_df.index]

        # Analyze community composition
        community_composition = {}
        for comm_id in set(comm_dict.values()):
            members = comm_df[comm_df['community'] == comm_id]
            functional_groups = members['functional_group'].value_counts().to_dict()
            organisms = members['organism'].tolist()

            community_composition[comm_id] = {
                'size': len(members),
                'organisms': organisms,
                'functional_groups': functional_groups,
                'avg_degree': members['degree'].mean()
            }

        # Save results
        comm_df.to_csv(f"{output_dir}/community_analysis_{method}.csv", index=False)
        community_reports[method] = {
            'dataframe': comm_df,
            'composition': community_composition
        }

    return community_reports

def create_biological_interpretation(centrality_df, community_reports, topology, node_names, output_dir):
    """Create biological interpretation of network analysis results"""

    interpretation = []

    interpretation.append("=" * 80)
    interpretation.append("MICROBIAL NETWORK ANALYSIS - BIOLOGICAL INTERPRETATION")
    interpretation.append("=" * 80)

    # Network overview
    interpretation.append(f"\n1. NETWORK OVERVIEW:")
    interpretation.append(f"   • Total organisms: {topology['nodes']}")
    interpretation.append(f"   • Correlation relationships: {topology['edges']}")
    interpretation.append(f"   • Network density: {topology['density']:.3f}")
    interpretation.append(f"   • Average degree: {topology['average_degree']:.2f}")
    interpretation.append(f"   • Connected components: {topology['connected_components']}")

    # Small world properties
    interpretation.append(f"\n2. NETWORK TOPOLOGY:")
    interpretation.append(f"   • Average clustering: {topology['average_clustering']:.3f}")
    interpretation.append(f"   • Average path length: {topology['average_path_length']:.2f}")
    interpretation.append(f"   • Small-world coefficient σ: {topology['small_world_sigma']}")
    if isinstance(topology['small_world_sigma'], float) and topology['small_world_sigma'] > 1:
        interpretation.append("   → Network exhibits small-world properties (high clustering, short paths)")

    # Central organisms
    interpretation.append(f"\n3. KEY HUB ORGANISMS (High Centrality):")

    # Get top organisms by different centrality measures
    top_degree = centrality_df.nlargest(3, 'degree_centrality')
    top_betweenness = centrality_df.nlargest(3, 'betweenness_centrality')
    top_eigenvector = centrality_df.nlargest(3, 'eigenvector_centrality')

    interpretation.append(f"   Most Connected (Degree Centrality):")
    for _, row in top_degree.iterrows():
        interpretation.append(f"   • {row['organism']} ({row['functional_group']}) - Degree: {row['degree']}")

    interpretation.append(f"   \n   Key Intermediates (Betweenness Centrality):")
    for _, row in top_betweenness.iterrows():
        interpretation.append(f"   • {row['organism']} ({row['functional_group']}) - Score: {row['betweenness_centrality']:.3f}")

    interpretation.append(f"   \n   Most Influential (Eigenvector Centrality):")
    for _, row in top_eigenvector.iterrows():
        interpretation.append(f"   • {row['organism']} ({row['functional_group']}) - Score: {row['eigenvector_centrality']:.3f}")

    # Functional group analysis
    interpretation.append(f"\n4. FUNCTIONAL GROUP ANALYSIS:")
    functional_groups = centrality_df['functional_group'].value_counts()
    for group, count in functional_groups.items():
        avg_degree = centrality_df[centrality_df['functional_group'] == group]['degree_centrality'].mean()
        interpretation.append(f"   • {group}: {count} organisms (avg. degree centrality: {avg_degree:.3f})")

    # Community analysis
    if 'louvain' in community_reports:
        interpretation.append(f"\n5. COMMUNITY STRUCTURE (Louvain Method):")
        louvain_data = community_reports['louvain']

        for comm_id, data in louvain_data['composition'].items():
            interpretation.append(f"   Community {comm_id} ({data['size']} organisms):")
            for group, count in data['functional_groups'].items():
                interpretation.append(f"   • {count} {group}")
            interpretation.append(f"   • Average degree: {data['avg_degree']:.2f}")
            interpretation.append(f"   • Key members: {', '.join(data['organisms'][:3])}{'...' if len(data['organisms']) > 3 else ''}")
            interpretation.append("")

    # Biological insights
    interpretation.append(f"6. BIOLOGICAL INSIGHTS:")

    # Check for methanogen centrality
    methanogens = centrality_df[centrality_df['functional_group'] == 'Methanogen']
    if len(methanogens) > 0:
        avg_methanogen_centrality = methanogens['degree_centrality'].mean()
        interpretation.append(f"   • Methanogens centrality: {avg_methanogen_centrality:.3f}")
        if avg_methanogen_centrality > centrality_df['degree_centrality'].mean():
            interpretation.append("   → Methanogens are highly connected, suggesting they're key players")

    # Check syntrophic relationships
    syntrophs = centrality_df[centrality_df['functional_group'] == 'Syntrophic']
    if len(syntrophs) > 0:
        avg_syntrophic_centrality = syntrophs['betweenness_centrality'].mean()
        interpretation.append(f"   • Syntrophic bacteria betweenness: {avg_syntrophic_centrality:.3f}")
        if avg_syntrophic_centrality > centrality_df['betweenness_centrality'].mean():
            interpretation.append("   → Syntrophic bacteria serve as important intermediates")

    # Metabolic pathway insights
    interpretation.append(f"\n7. METABOLIC PATHWAY IMPLICATIONS:")

    # Check for hydrogenotrophic pathway
    hydrogenotrophic_methanogens = ['Methanobacteriaceae', 'Methanoregulaceae', 'Methanospirillaceae']
    hydro_present = any(any(hm in org for hm in hydrogenotrophic_methanogens)
                       for org in node_names)

    if hydro_present:
        interpretation.append("   • Hydrogenotrophic methanogenesis pathway well-represented")

    # Check for acetoclastic pathway
    acetoclastic_methanogens = ['Methanosaetaceae', 'Methanosarcinaceae']
    aceto_present = any(any(am in org for am in acetoclastic_methanogens)
                       for org in node_names)

    if aceto_present:
        interpretation.append("   • Acetoclastic methanogenesis pathway present")

    # Network stability implications
    interpretation.append(f"\n8. ECOSYSTEM STABILITY IMPLICATIONS:")
    interpretation.append(f"   • Network connectivity suggests functional redundancy")
    interpretation.append(f"   • High clustering indicates robust local interactions")
    interpretation.append(f"   • Community structure suggests specialized functional niches")

    # Save interpretation
    interpretation_text = "\n".join(interpretation)
    with open(f"{output_dir}/biological_interpretation.txt", "w") as f:
        f.write(interpretation_text)

    return interpretation_text

def create_visualizations(G, node_names, centrality_measures, communities, classifications, output_dir):
    """Create comprehensive network visualizations"""

    print("Creating visualizations...")

    # Set up the plotting style
    plt.style.use('default')

    # 1. Network layout for all plots
    from utils.visualization_utils import get_optimal_layout
    pos = get_optimal_layout(G, seed=42, scale=3.0)

    # 2. Centrality visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    centrality_plots = [
        ('degree', 'Degree Centrality'),
        ('betweenness', 'Betweenness Centrality'),
        ('eigenvector', 'Eigenvector Centrality'),
        ('pagerank', 'PageRank Centrality')
    ]

    for idx, (measure, title) in enumerate(centrality_plots):
        ax = axes[idx // 2, idx % 2]

        # Get centrality values
        centrality_values = [centrality_measures[measure][i] for i in range(len(node_names))]

        # Create node colors and sizes based on centrality
        node_colors = centrality_values
        node_sizes = [300 + 1000 * val for val in centrality_values]

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                              cmap=plt.cm.viridis, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)

        # Add labels for top nodes
        top_nodes = sorted(range(len(centrality_values)), key=lambda i: centrality_values[i], reverse=True)[:5]
        top_labels = {i: node_names[i].split('.')[-1] for i in top_nodes}
        nx.draw_networkx_labels(G, pos, labels=top_labels, font_size=8, ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/centrality_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Community detection visualization
    if 'louvain' in communities:
        plt.figure(figsize=(15, 10))

        # Get community colors
        community_dict = communities['louvain']
        unique_communities = list(set(community_dict.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))

        node_colors = [colors[unique_communities.index(community_dict[i])] for i in range(len(node_names))]

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

        # Add labels
        labels = {i: node_names[i].split('.')[-1] for i in range(len(node_names))}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

        plt.title("Community Structure (Louvain Method)", fontsize=16, fontweight='bold')
        plt.axis('off')

        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                                     markersize=10, label=f'Community {unique_communities[i]}')
                          for i in range(len(unique_communities))]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.savefig(f"{output_dir}/community_structure.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Functional group visualization
    plt.figure(figsize=(15, 10))

    # Define colors for functional groups
    group_colors = {
        'Methanogen': '#FF6B6B',      # Red
        'Syntrophic': '#4ECDC4',      # Teal
        'Fermentative': '#45B7D1',    # Blue
        'Other_Anaerobe': '#96CEB4',  # Green
        'Unclassified': '#FFEAA7'     # Yellow
    }

    node_colors = [group_colors.get(classifications.get(node_names[i], 'Unclassified'), '#CCCCCC')
                   for i in range(len(node_names))]

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

    # Add labels
    labels = {i: node_names[i].split('.')[-1] for i in range(len(node_names))}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title("Functional Groups in Microbial Network", fontsize=16, fontweight='bold')
    plt.axis('off')

    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                 markersize=10, label=group)
                      for group, color in group_colors.items()]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.savefig(f"{output_dir}/functional_groups.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualizations saved successfully!")

def main():
    """Main function to run comprehensive network analysis"""

    print("Starting comprehensive microbial network analysis...")

    # Create output directory
    output_dir = "network_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load the correlation graph
    G, node_names = load_correlation_graph_from_dataset()

    if G is None:
        print("Failed to load correlation graph")
        return

    print(f"Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Classify microorganisms
    classifications = classify_microorganisms(node_names)

    # Perform network analysis
    centrality_measures = calculate_centrality_measures(G)
    communities = detect_communities(G)
    topology = analyze_network_topology(G)

    # Create detailed reports
    centrality_df, top_organisms = create_centrality_analysis_report(
        G, node_names, centrality_measures, classifications, output_dir)

    community_reports = create_community_analysis_report(
        G, node_names, communities, classifications, output_dir)

    # Create biological interpretation
    biological_interpretation = create_biological_interpretation(
        centrality_df, community_reports, topology, node_names, output_dir)

    # Create visualizations
    create_visualizations(G, node_names, centrality_measures, communities,
                         classifications, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("NETWORK ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    print("\nFiles generated:")
    print("• centrality_analysis.csv - Detailed centrality measures")
    print("• community_analysis_*.csv - Community detection results")
    print("• biological_interpretation.txt - Biological insights")
    print("• centrality_analysis.png - Centrality visualizations")
    print("• community_structure.png - Community structure")
    print("• functional_groups.png - Functional group network")

    print("\n" + biological_interpretation)

if __name__ == "__main__":
    main()