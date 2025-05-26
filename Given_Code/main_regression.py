import argparse
from pipeline_regression import RegressionPipeline, DEFAULT_SAVE_DIR

def main():
    """Main entry point for running the regression pipeline"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='GNN Regression Pipeline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors for KNN graph sparsification')
    parser.add_argument('--mantel_threshold', type=float, default=0.05, help='P-value threshold for Mantel test')
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'gat', 'rggc'], help='Type of GNN model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR, help='Directory to save results')
    parser.add_argument('--importance_threshold', type=float, default=0.3, help='Threshold for edge importance in GNNExplainer')
    parser.add_argument('--estimate_uncertainty', action='store_true', help='Estimate uncertainty in predictions')
    parser.add_argument('--use_fast_correlation', action='store_true', default=True, help='Use fast correlation-based graph construction (default: True). Use --no-use_fast_correlation for Mantel test approach.')
    parser.add_argument('--no-use_fast_correlation', dest='use_fast_correlation', action='store_false', help='Use slower Mantel test-based graph construction instead of fast correlation method')
    parser.add_argument('--graph_mode', type=str, default='otu', choices=['otu', 'family'], help='Graph construction mode: "otu" for OTU-based graphs, "family" for family-level aggregated graphs')
    parser.add_argument('--family_filter_mode', type=str, default='relaxed', choices=['strict', 'relaxed', 'permissive'], help='Family filtering strictness: strict (5%+1%), relaxed (2%+0.1%), permissive (1%+0.05%)')
    
    args = parser.parse_args()
    
    # Print arguments
    print("Pipeline parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Create and run pipeline
    pipeline = RegressionPipeline(
        data_path=args.data_path,
        k_neighbors=args.k_neighbors,
        mantel_threshold=args.mantel_threshold,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        num_folds=args.num_folds,
        save_dir=args.save_dir,
        importance_threshold=args.importance_threshold,
        estimate_uncertainty=args.estimate_uncertainty,
        use_fast_correlation=args.use_fast_correlation,
        graph_mode=args.graph_mode,
        family_filter_mode=args.family_filter_mode
    )
    
    # Run the pipeline
    results = pipeline.run_pipeline()
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 