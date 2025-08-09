# Methane GNN Pipeline

This pipeline implements a Graph Neural Network (GNN) approach for predicting methane-related properties (ACE-km and H2-km) using microbial community data. The pipeline includes graph construction, sparsification, model training, and model explanation.

## Components

The pipeline consists of three main components:

1. **MethaneGNN_dataset.py**: Handles data loading, preprocessing, graph construction using Mantel test, and graph sparsification using KNN.

2. **MethaneGNN_models.py**: Implements various GNN architectures (GCN, GAT, GATv2, GIN) and the GNNExplainer for model interpretability.

3. **MethaneGNN_pipeline.py**: Orchestrates the complete workflow, including training, evaluation, and visualization.

## Features

- **Graph Construction**: Uses the Mantel test to identify significant correlations between microbial families.
- **Graph Sparsification**: Applies k-nearest neighbor (KNN) approach to reduce graph complexity.
- **Multiple GNN Architectures**: Supports GCN, GAT, GATv2, and GIN models.
- **Cross-Validation**: Implements k-fold cross-validation for robust evaluation.
- **Model Explainability**: Uses GNNExplainer to identify important edges and nodes for predictions.
- **Visualization**: Provides detailed visualizations of model performance and explanations.

## Usage

```python
from MethaneGNN_pipeline import MethanePipeline

# Initialize the pipeline
pipeline = MethanePipeline(
    data_path='../Data/New_data.csv',
    k_neighbors=10,
    mantel_threshold=0.05,
    model_type='gat',
    hidden_dim=128,
    num_layers=4,
    dropout_rate=0.3,
    batch_size=8,
    learning_rate=0.001,
    weight_decay=1e-4,
    num_epochs=300,
    patience=30,
    num_folds=5,
    save_dir='./methane_results'
)

# Run the complete pipeline
results = pipeline.run_pipeline()
```

## Pipeline Workflow

1. **Data Loading and Preprocessing**:
   - Loads microbial abundance data
   - Filters low-abundance/prevalence families
   - Transforms data using double square root transformation

2. **Graph Construction**:
   - Uses Mantel test to identify significant correlations between families
   - Creates edges between significantly correlated families
   - Applies KNN sparsification to reduce graph complexity

3. **Model Training**:
   - Trains separate models for each target variable (ACE-km and H2-km)
   - Uses k-fold cross-validation to evaluate model performance
   - Implements early stopping to prevent overfitting

4. **Model Explainability**:
   - Uses GNNExplainer to identify important edges for predictions
   - Visualizes explanations as network graphs

5. **Results and Visualization**:
   - Creates scatter plots of predicted vs actual values
   - Calculates performance metrics (MSE, RMSE, R²)
   - Visualizes model explanations

## Requirements

- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- NetworkX
- scikit-learn

## References

This implementation combines approaches from:

1. The GNN backbone developed by Badhan Mazumder for brain connectivity analysis
2. The microbial community GNN approach in the MethaneGNN codebase
3. Standard GNN architectures and explainability tools from PyTorch Geometric 