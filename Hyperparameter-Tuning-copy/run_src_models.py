#!/usr/bin/env python3
"""
Run the domain expert cases pipeline using models from src/models/GNNmodelsRegression.py
This includes the GraphTransformer model in addition to GCN, RGGC, and GAT.
"""

from domain_expert_cases_src_models import DomainExpertCasesPipeline

def test_single_case(case_name='case1', reduced_params=True):
    """Test single case with the src models"""
    print(f"Testing {case_name} with src models...")
    
    # Configure for reduced testing if needed
    if reduced_params:
        pipeline = DomainExpertCasesPipeline(
            data_path='../Data/New_Data.csv',
            case_type=case_name,
            k_neighbors=10,
            mantel_threshold=0.05,
            hidden_dim=64,
            num_epochs=100,  # Reduced for testing
            batch_size=4,
            learning_rate=0.01,
            patience=10,
            num_folds=5,
            importance_threshold=0.2,
            use_fast_correlation=False,
            graph_mode='family',
            family_filter_mode='strict',
            use_nested_cv=True,
            save_dir='./domain_expert_results_src_models'
        )
    else:
        # Full parameters
        pipeline = DomainExpertCasesPipeline(
            data_path='../Data/New_Data.csv',
            case_type=case_name,
            save_dir='./domain_expert_results_src_models'
        )
    
    print(f"Pipeline configured for {case_name.upper()}")
    print("Available GNN models:", pipeline.gnn_models_to_train)
    print("Using models from: ../src/models/GNNmodelsRegression.py")
    
    try:
        case_results = pipeline.run_case_specific_pipeline()
        print(f"Successfully completed {case_name}!")
        return case_results
    except Exception as e:
        print(f"Error testing {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the pipeline"""
    print("="*80)
    print("DOMAIN EXPERT CASES PIPELINE - SRC MODELS VERSION")
    print("="*80)
    print("Models: GCN, RGGC, GAT, GraphTransformer")
    print("Source: ../src/models/GNNmodelsRegression.py")
    print("="*80)
    
    # Test with case1 first
    results = test_single_case('case1', reduced_params=True)
    
    if results:
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Results saved to: ./domain_expert_results_src_models/")
        print("Available GNN models used: GCN, RGGC, GAT, GraphTransformer")
    else:
        print("Pipeline failed. Check the error messages above.")

if __name__ == "__main__":
    main()