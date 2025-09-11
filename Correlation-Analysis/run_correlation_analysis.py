"""
Example Usage Script for Correlation Network Analysis
Based on Thapa et al. (2023) methodology

This script demonstrates how to run the correlation analysis on your microbial data
and compares results with the paper's findings.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the correlation analysis module to path
sys.path.append(str(Path(__file__).parent))
from correlation_network_analysis import CorrelationNetworkAnalyzer

def load_microbial_data(data_path: str) -> pd.DataFrame:
    """
    Load and prepare microbial abundance data.
    
    Paper context: They analyzed microbial communities from high-pressure anaerobic digestion
    with focus on family-level taxonomic classifications.
    
    Args:
        data_path: Path to the microbial abundance data CSV file
        
    Returns:
        DataFrame with samples as rows and microbial taxa as columns
    """
    print(f"Loading data from: {data_path}")
    
    # Load the data
    data = pd.read_csv(data_path, index_col=0)
    
    print(f"Original data shape: {data.shape}")
    
    # Filter to family-level classifications (similar to your GNN pipeline)
    # Look for family-level taxonomic identifiers
    family_columns = [col for col in data.columns if ';f__' in col and not col.endswith('ACE-km') and not col.endswith('H2-km')]
    
    if family_columns:
        print(f"Found {len(family_columns)} family-level features")
        family_data = data[family_columns]
    else:
        print("No family-level features found, using all microbial features")
        # Exclude target columns
        microbial_columns = [col for col in data.columns if not col.endswith('ACE-km') and not col.endswith('H2-km')]
        family_data = data[microbial_columns]
    
    print(f"Selected microbial data shape: {family_data.shape}")
    
    return family_data

def compare_with_paper_results(analyzer: CorrelationNetworkAnalyzer, results: dict):
    """
    Compare analysis results with paper findings from Thapa et al. (2023).
    
    Paper key findings:
    - 48 core OTUs and 178 edges in co-occurrence network
    - Average network distance: 2.1, longest distance: 5.0
    - Clustering coefficient: 0.52, Modularity: 0.53
    - 5 modules identified
    - Keystone species: Methanobacterium, Methanomicrobiaceae, Alkaliphilus, Petrimonas
    """
    print("\n" + "="*70)
    print("COMPARISON WITH PAPER RESULTS (Thapa et al. 2023)")
    print("="*70)
    
    # Paper benchmarks
    paper_metrics = {
        'num_nodes': 48,
        'num_edges': 178,
        'average_path_length': 2.1,
        'diameter': 5.0,
        'clustering_coefficient': 0.52,
        'modularity': 0.53,
        'num_communities': 5
    }
    
    our_metrics = results['network_metrics']
    
    print("Network Structure Comparison:")
    print(f"  Nodes:              Paper: {paper_metrics['num_nodes']:>6} | Ours: {our_metrics['num_nodes']:>6}")
    print(f"  Edges:              Paper: {paper_metrics['num_edges']:>6} | Ours: {our_metrics['num_edges']:>6}")
    print(f"  Communities:        Paper: {paper_metrics['num_communities']:>6} | Ours: {our_metrics.get('num_communities', 'N/A'):>6}")
    
    print("\nNetwork Topology Comparison:")
    print(f"  Avg Path Length:    Paper: {paper_metrics['average_path_length']:>6.2f} | Ours: {our_metrics.get('average_path_length', 0):>6.2f}")
    print(f"  Diameter:           Paper: {paper_metrics['diameter']:>6} | Ours: {our_metrics.get('diameter', 'N/A'):>6}")
    print(f"  Clustering Coeff:   Paper: {paper_metrics['clustering_coefficient']:>6.3f} | Ours: {our_metrics.get('clustering_coefficient', 0):>6.3f}")
    print(f"  Modularity:         Paper: {paper_metrics['modularity']:>6.3f} | Ours: {our_metrics.get('modularity', 0):>6.3f}")
    
    print("\nPaper's Keystone Species:")
    paper_keystone = ['Methanobacterium', 'Methanomicrobiaceae', 'Alkaliphilus', 'Petrimonas']
    for i, species in enumerate(paper_keystone, 1):
        print(f"  {i}. {species}")
    
    print("\nOur Top Keystone Species:")
    keystone_species = results['keystone_species']
    for species, info in list(keystone_species.items())[:4]:
        # Extract genus name for cleaner display
        genus_name = species.split(';g__')[-1] if ';g__' in species else species.split('__')[-1]
        print(f"  {info['rank']}. {genus_name} (score: {info['keystone_score']:.3f})")
    
    # Check for overlap with paper's keystone species
    our_keystone_names = [species.split(';g__')[-1] if ';g__' in species else species.split('__')[-1] 
                          for species in keystone_species.keys()]
    
    overlap = []
    for paper_species in paper_keystone:
        for our_species in our_keystone_names:
            if paper_species.lower() in our_species.lower() or our_species.lower() in paper_species.lower():
                overlap.append((paper_species, our_species))
    
    if overlap:
        print("\nKeystone Species Overlap Found:")
        for paper_sp, our_sp in overlap:
            print(f"  Paper: {paper_sp} <-> Our: {our_sp}")
    else:
        print("\nNo direct keystone species overlap found.")
        print("Note: This is expected as datasets and conditions differ.")

def generate_detailed_report(results: dict, output_dir: str):
    """
    Generate a detailed analysis report.
    """
    report_path = os.path.join(output_dir, "correlation_analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("CORRELATION NETWORK ANALYSIS REPORT\n")
        f.write("Based on Thapa et al. (2023) Methodology\n")
        f.write("="*60 + "\n\n")
        
        # Network Overview
        f.write("NETWORK OVERVIEW\n")
        f.write("-"*20 + "\n")
        metrics = results['network_metrics']
        f.write(f"Number of nodes (OTUs): {metrics['num_nodes']}\n")
        f.write(f"Number of edges: {metrics['num_edges']}\n")
        f.write(f"Network density: {metrics['density']:.4f}\n")
        f.write(f"Average path length: {metrics.get('average_path_length', 'N/A')}\n")
        f.write(f"Network diameter: {metrics.get('diameter', 'N/A')}\n")
        f.write(f"Clustering coefficient: {metrics.get('clustering_coefficient', 'N/A'):.3f}\n")
        f.write(f"Modularity: {metrics.get('modularity', 'N/A'):.3f}\n")
        f.write(f"Number of communities: {metrics.get('num_communities', 'N/A')}\n\n")
        
        # Keystone Species
        f.write("TOP KEYSTONE SPECIES\n")
        f.write("-"*20 + "\n")
        keystone_species = results['keystone_species']
        for species, info in keystone_species.items():
            f.write(f"\n{info['rank']}. {species}\n")
            f.write(f"   Keystone Score: {info['keystone_score']:.4f}\n")
            f.write(f"   Degree Centrality: {info['degree_centrality']:.4f}\n")
            f.write(f"   Betweenness Centrality: {info['betweenness_centrality']:.4f}\n")
            f.write(f"   Closeness Centrality: {info['closeness_centrality']:.4f}\n")
            f.write(f"   Node Strength: {info['node_strength']:.4f}\n")
            f.write(f"   Degree: {info['degree']}\n")
            f.write(f"   Module: {info['module']}\n")
        
        # Methodology
        f.write("\n\nMETHODOLOGY NOTES\n")
        f.write("-"*20 + "\n")
        f.write("This analysis replicates the methodology from:\n")
        f.write("Thapa et al. (2023) 'Elucidation of microbial interactions, dynamics, and\n")
        f.write("keystone microbes in high pressure anaerobic digestion'\n\n")
        f.write("Key steps:\n")
        f.write("1. Data preprocessing with abundance filtering (>0.1%)\n")
        f.write("2. Spearman correlation calculation with significance testing (p < 0.05)\n")
        f.write("3. Co-occurrence network construction\n")
        f.write("4. Network topology analysis (modularity, centrality, clustering)\n")
        f.write("5. Keystone species identification\n")
    
    print(f"Detailed report saved to: {report_path}")

def main():
    """
    Main function to run the correlation network analysis.
    """
    # Configuration
    DATA_PATH = "/Users/sb/TReNDS/DNA Project/gnn_backbone/Data/New_Data.csv"
    OUTPUT_DIR = "/Users/sb/TReNDS/DNA Project/gnn_backbone/Correlation-Analysis/results"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("CORRELATION NETWORK ANALYSIS")
    print("Based on Thapa et al. (2023) methodology")
    print("="*50)
    
    try:
        # Step 1: Load data
        print("\n1. Loading microbial abundance data...")
        microbial_data = load_microbial_data(DATA_PATH)
        
        # Step 2: Initialize analyzer with paper's parameters
        print("\n2. Initializing correlation network analyzer...")
        analyzer = CorrelationNetworkAnalyzer(
            significance_threshold=0.05,  # Paper uses p < 0.05
            correlation_threshold=0.3     # Reasonable threshold for strong correlations
        )
        
        # Step 3: Run complete analysis
        print("\n3. Running correlation network analysis...")
        results = analyzer.run_complete_analysis(microbial_data, OUTPUT_DIR)
        
        # Step 4: Compare with paper results
        print("\n4. Comparing results with paper findings...")
        compare_with_paper_results(analyzer, results)
        
        # Step 5: Generate detailed report
        print("\n5. Generating detailed report...")
        generate_detailed_report(results, OUTPUT_DIR)
        
        print(f"\n✓ Analysis complete! Results saved to: {OUTPUT_DIR}")
        print("📊 Generated files:")
        print("  - cooccurrence_network.png (network visualization)")
        print("  - correlation_heatmap.png (correlation matrix)")
        print("  - correlation_analysis_report.txt (detailed report)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()