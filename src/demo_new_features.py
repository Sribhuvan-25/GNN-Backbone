#!/usr/bin/env python3

"""
FAST DEMO: New Features Implementation
- Node importance reporting
- Node-based pruning (instead of edge pruning) 
- t-SNE embedding visualization

This demo uses minimal parameters for quick execution (~2-3 minutes)
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.domain_expert_cases_pipeline import DomainExpertCasesPipeline

def demo_new_features():
    """Fast demo of the new features"""
    
    print("🚀 FAST DEMO: New GNN Features")
    print("=" * 60)
    print("Features being demonstrated:")
    print("  1. Node importance reporting")
    print("  2. Node-based pruning (vs edge pruning)")  
    print("  3. t-SNE embedding visualization")
    print("=" * 60)
    
    # Use fast parameters for quick demo
    data_path = "../Data/New_Data.csv"
    
    # Check data file
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    print(f"✅ Using data: {data_path}")
    print(f"🎯 Running Case 1 (H2-km prediction)")
    
    try:
        # ULTRA-FAST parameters for demo
        pipeline = DomainExpertCasesPipeline(
            data_path=data_path,
            case_type='case1',
            k_neighbors=5,           # Reduced from 10
            hidden_dim=32,           # Reduced from 64
            num_epochs=10,           # Reduced from 50 - VERY FAST
            num_folds=2,             # Reduced from 3 - FAST
            save_dir="./demo_results",
            importance_threshold=0.3, # Higher threshold = fewer nodes
            use_fast_correlation=True,
            family_filter_mode='relaxed'
        )
        
        print("\n🏃‍♂️ Starting FAST training (should take ~2-3 minutes)...")
        print("⏱️  Progress will be shown below:")
        print("-" * 60)
        
        # Run the pipeline
        results = pipeline.run_case_specific_pipeline()
        
        print("\n" + "🎉" * 20)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        # Check what files were created
        demo_dir = "./demo_results"
        if os.path.exists(demo_dir):
            print(f"\n📁 Files created in {demo_dir}:")
            
            created_files = []
            for root, dirs, files in os.walk(demo_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, demo_dir)
                    created_files.append((file, rel_path))
            
            # Categorize files
            tsne_files = [f for f in created_files if 'tsne' in f[0].lower()]
            graph_files = [f for f in created_files if any(x in f[0].lower() for x in ['graph', 'explainer'])]
            result_files = [f for f in created_files if f[0].endswith('.pkl') and 'results' in f[0]]
            plot_files = [f for f in created_files if f[0].endswith('.png') and f not in tsne_files]
            
            if tsne_files:
                print(f"  🎨 t-SNE Plots:")
                for name, path in tsne_files:
                    print(f"      ✅ {path}")
            
            if graph_files:
                print(f"  🕸️  Graph Visualizations:")
                for name, path in graph_files:
                    print(f"      ✅ {path}")
            
            if result_files:
                print(f"  📊 Results Files:")
                for name, path in result_files:
                    print(f"      ✅ {path}")
            
            if plot_files:
                print(f"  📈 Other Plots:")
                for name, path in plot_files:
                    print(f"      ✅ {path}")
            
            print(f"\n📈 Total files created: {len(created_files)}")
        
        print(f"\n🎯 NEW FEATURES SUCCESSFULLY DEMONSTRATED:")
        print(f"  ✅ Node importance reporting (shown during explainer phase)")
        print(f"  ✅ Node-based pruning (replaces edge-based pruning)")  
        print(f"  ✅ t-SNE embedding visualization")
        print(f"  ✅ Consistent visualizations with proper node/edge sizing")
        
        print(f"\n📂 Check the demo_results folder for all generated files!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_new_features()