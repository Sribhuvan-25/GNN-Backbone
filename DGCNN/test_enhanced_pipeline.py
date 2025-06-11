#!/usr/bin/env python3
"""
Test script for Enhanced DGCNN Pipeline
Tests the improved DGCNN architecture and training enhancements
"""

import torch
import numpy as np
from mixed_embedding_pipeline import MixedEmbeddingPipeline

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enhanced pipeline configuration
    config = {
        'data_path': '../Data/phenomapping.csv',  # Adjust path as needed
        'k_neighbors': 8,  # Increased for better connectivity
        'mantel_threshold': 0.05,
        'hidden_dim': 96,  # Increased hidden dimension
        'dropout_rate': 0.4,  # Slightly increased for regularization
        'batch_size': 16,  # Increased batch size
        'learning_rate': 0.002,  # Slightly higher learning rate
        'weight_decay': 5e-5,  # Reduced weight decay
        'num_epochs': 250,  # More epochs with better early stopping
        'patience': 25,  # Increased patience
        'num_folds': 5,
        'save_dir': './enhanced_dgcnn_results',
        'importance_threshold': 0.15,  # Lowered threshold for more edges
        'use_fast_correlation': False,
        'graph_mode': 'family',
        'family_filter_mode': 'strict',
        # Enhanced parameters
        'use_feature_scaling': True,
        'use_data_augmentation': True,
        'augmentation_noise_std': 0.005,  # Light augmentation
        'use_graph_enhancement': True,
        'adaptive_k_neighbors': True
    }
    
    print("=" * 80)
    print("ENHANCED DGCNN PIPELINE TEST")
    print("=" * 80)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Initialize enhanced pipeline
        pipeline = MixedEmbeddingPipeline(**config)
        
        # Run the pipeline
        print("Starting enhanced pipeline...")
        results = pipeline.run_pipeline()
        
        print("\n" + "=" * 80)
        print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Analyze DGCNN performance improvements
        print("\nDGCNN PERFORMANCE ANALYSIS:")
        print("-" * 50)
        
        for target_name, target_results in results.items():
            if target_name == 'summary':
                continue
            
            print(f"\nTarget: {target_name}")
            
            # Check DGCNN performance in both phases
            for phase in ['knn', 'explainer']:
                if phase in target_results and 'dgcnn' in target_results[phase]:
                    dgcnn_r2 = target_results[phase]['dgcnn']['avg_metrics']['r2']
                    dgcnn_rmse = target_results[phase]['dgcnn']['avg_metrics']['rmse']
                    
                    # Get all models for comparison
                    all_r2s = [results['avg_metrics']['r2'] for results in target_results[phase].values()]
                    rank = sorted(all_r2s, reverse=True).index(dgcnn_r2) + 1
                    
                    print(f"  {phase.upper()} phase - DGCNN: R² = {dgcnn_r2:.4f}, RMSE = {dgcnn_rmse:.4f}, Rank = {rank}/4")
        
        print(f"\nResults saved to: {config['save_dir']}")
        print("Check the comprehensive plots and analysis files!")
        
        return results
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 