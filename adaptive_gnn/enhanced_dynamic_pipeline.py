#!/usr/bin/env python3
"""
Enhanced Dynamic Microbial GNN Pipeline
=======================================

This pipeline maintains the same structure as mixed_embedding_pipeline.py but uses
advanced Dynamic GNN models based on latest 2024 research.

Research Integration:
1. "Modelling Microbial Communities with Graph Neural Networks" (ICLR 2024)
2. "HG-LGBM: A Hybrid Model for Microbiome-Disease Prediction" (MDPI 2024)  
3. "Learning Dynamics from Multicellular Graphs" (ArXiv 2024)

Pipeline Flow (Enhanced):
1. Load microbial data and create dynamic data objects
2. Train Dynamic GNN models (DynamicGNN, HeterogeneousGNN, AdaptiveGNN)
3. Use best Dynamic GNN for biological interaction discovery
4. Extract multi-scale embeddings with learned interaction patterns
5. Train ML models (LinearSVR, ExtraTrees) on enhanced embeddings with 5-fold CV
6. Provide biological interpretability through learned interaction networks
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import our new Dynamic GNN models
from models.dynamic_microbial_gnn import (
    DynamicMicrobialGNN, 
    BiologicalConstraintLoss
)
from models.adaptive_microbial_gnn import AdaptiveMicrobialGNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class EnhancedDynamicPipeline:
    """
    Enhanced Dynamic GNN Pipeline for Microbial Interaction Networks
    
    Maintains compatibility with mixed_embedding_pipeline.py structure while
    incorporating cutting-edge Dynamic GNN research for microbial communities.
    """
    
    def __init__(self, 
                 data_path,
                 target_columns=['ACE-km', 'H2-km'],
                 hidden_dim=128,
                 dropout_rate=0.2,
                 batch_size=8,
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 num_epochs=300,
                 patience=30,
                 num_folds=5,
                 save_dir='./enhanced_dynamic_results',
                 sparsity_factor=0.15,
                 use_uncertainty=True,
                 biological_constraints=True,
                 random_state=42):
        """
        Initialize the enhanced dynamic pipeline
        
        Args:
            data_path: Path to the CSV file with microbial data
            target_columns: List of target variable names
            hidden_dim: Hidden dimension size for Dynamic GNNs
            dropout_rate: Dropout rate
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Maximum number of epochs
            patience: Patience for early stopping
            num_folds: Number of folds for cross-validation
            save_dir: Directory to save results
            sparsity_factor: Sparsity factor for graph learning
            use_uncertainty: Whether to use uncertainty quantification
            biological_constraints: Whether to apply biological constraints
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.target_columns = target_columns
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_folds = num_folds
        self.save_dir = save_dir
        self.sparsity_factor = sparsity_factor
        self.use_uncertainty = use_uncertainty
        self.biological_constraints = biological_constraints
        self.random_state = random_state
        
        # Model types to train (enhanced dynamic versions)
        self.dynamic_models_to_train = ['dynamic_gnn', 'heterogeneous_gnn', 'adaptive_gnn']
        
        print(f"🧬 Enhanced Dynamic GNN Pipeline initialized")
        print(f"📱 Device: {device}")
        print(f"🎯 Targets: {target_columns}")
        print(f"🔬 Dynamic Models: {self.dynamic_models_to_train}")
        
        # Create comprehensive save directories
        os.makedirs(save_dir, exist_ok=True)
        for subdir in ['dynamic_models', 'ml_models', 'plots', 'embeddings', 
                      'interactions', 'biological_insights', 'detailed_results']:
            os.makedirs(f"{save_dir}/{subdir}", exist_ok=True)
        
        # Data containers
        self.data = None
        self.microbial_features = None
        self.targets = None
        self.microbial_families = None
        self.scaler = StandardScaler()
        
        # Results containers
        self.dynamic_results = {}
        self.ml_results = {}
        self.embeddings = {}
        self.learned_interactions = {}
        
    def load_data(self):
        """
        Load and preprocess microbial abundance data
        """
        print(f"\n📂 Loading data from: {self.data_path}")
        
        # Load CSV data
        self.data = pd.read_csv(self.data_path)
        print(f"📊 Data shape: {self.data.shape}")
        
        # Extract microbial family columns (excluding targets and metadata)
        exclude_cols = self.target_columns + ['Sample_ID'] if 'Sample_ID' in self.data.columns else self.target_columns
        self.microbial_families = [col for col in self.data.columns if col not in exclude_cols]
        
        print(f"🦠 Number of microbial families: {len(self.microbial_families)}")
        
        # Extract features and targets
        self.microbial_features = self.data[self.microbial_families].values.astype(np.float32)
        self.targets = self.data[self.target_columns].values.astype(np.float32)
        
        # Normalize features (preserve original for biological interpretation)
        self.original_features = self.microbial_features.copy()
        self.microbial_features = self.scaler.fit_transform(self.microbial_features)
        
        print(f"✅ Data loaded successfully")
        print(f"   - Features: {self.microbial_features.shape}")
        print(f"   - Targets: {self.targets.shape}")
        
        # Print statistics
        print(f"\n📈 Feature Statistics:")
        print(f"   - Mean abundance: {np.mean(self.microbial_features):.4f}")
        print(f"   - Std abundance: {np.std(self.microbial_features):.4f}")
        print(f"   - Zero abundance ratio: {np.mean(self.original_features == 0):.4f}")
        
    def create_dynamic_data_objects(self):
        """
        Create PyTorch Geometric data objects for Dynamic GNN training
        """
        print(f"\n🔧 Creating dynamic data objects...")
        
        data_objects = []
        num_samples, num_features = self.microbial_features.shape
        
        for i in range(num_samples):
            # Node features: each microbial family is a node
            x = torch.tensor(self.microbial_features[i].reshape(-1, 1), dtype=torch.float)
            
            # Target values for this sample
            y = torch.tensor(self.targets[i], dtype=torch.float)
            
            # Additional metadata for biological constraints
            metadata = {
                'sample_id': i,
                'original_abundances': self.original_features[i],
                'family_names': self.microbial_families
            }
            
            # Create data object (no fixed edge_index - will be learned dynamically)
            data = Data(x=x, y=y)
            data.metadata = metadata
            data_objects.append(data)
        
        print(f"✅ Created {len(data_objects)} dynamic data objects")
        print(f"   - Node features per sample: {data_objects[0].x.shape}")
        print(f"   - Target shape per sample: {data_objects[0].y.shape}")
        
        return data_objects
    
    def create_dynamic_model(self, model_type, num_targets=1):
        """
        Create enhanced Dynamic GNN model
        """
        num_nodes = len(self.microbial_families)
        
        if model_type == 'dynamic_gnn':
            model = DynamicMicrobialGNN(
                num_nodes=num_nodes,
                input_dim=1,
                hidden_dim=self.hidden_dim,
                output_dim=num_targets,
                num_heads=8,
                dropout=self.dropout_rate,
                sparsity_factor=self.sparsity_factor,
                num_gnn_layers=4,
                use_uncertainty=self.use_uncertainty,
                biological_constraints=self.biological_constraints
            ).to(device)
        elif model_type == 'heterogeneous_gnn':
            # Use DynamicMicrobialGNN with heterogeneous focus
            model = DynamicMicrobialGNN(
                num_nodes=num_nodes,
                input_dim=1,
                hidden_dim=self.hidden_dim,
                output_dim=num_targets,
                num_heads=6,
                dropout=self.dropout_rate,
                sparsity_factor=self.sparsity_factor,
                num_gnn_layers=3,
                use_uncertainty=self.use_uncertainty,
                biological_constraints=True
            ).to(device)
        elif model_type == 'adaptive_gnn':
            model = AdaptiveMicrobialGNN(
                num_nodes=num_nodes,
                input_dim=1,
                hidden_dim=self.hidden_dim,
                output_dim=num_targets,
                num_heads=4,
                dropout=self.dropout_rate,
                sparsity_factor=self.sparsity_factor,
                num_gnn_layers=3
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def train_dynamic_model(self, model_type, target_idx, data_objects):
        """
        Train Dynamic GNN model with enhanced biological constraints
        """
        target_name = self.target_columns[target_idx]
        print(f"\n🧠 Training {model_type.upper()} for target: {target_name}")
        
        # Setup k-fold cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        fold_results = []
        best_model = None
        best_r2 = -float('inf')
        
        # Loss functions
        mse_criterion = nn.MSELoss()
        if self.biological_constraints:
            bio_criterion = BiologicalConstraintLoss(alpha=0.1, beta=0.05)
        
        # Iterate through folds
        for fold, (train_index, test_index) in enumerate(kf.split(data_objects)):
            fold_num = fold + 1
            print(f"  Fold {fold_num}/{self.num_folds}")
            
            # Create fold-specific data
            train_dataset = [data_objects[i] for i in train_index]
            test_dataset = [data_objects[i] for i in test_index]
            
            # Adjust targets for single target training
            for data in train_dataset + test_dataset:
                data.y = data.y[target_idx:target_idx+1]
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Create model for this fold
            model = self.create_dynamic_model(model_type, num_targets=1)
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=self.learning_rate, 
                                        weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=15, factor=0.5, verbose=False
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.num_epochs):
                # Training phase
                model.train()
                epoch_train_loss = 0
                num_train_batches = 0
                
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if self.use_uncertainty:
                        pred_mean, pred_log_var, dynamics = model(batch.x, batch.batch, return_dynamics=True)
                        
                        # Uncertainty-aware loss
                        precision = torch.exp(-pred_log_var)
                        mse_loss = torch.mean(precision * (batch.y - pred_mean)**2 + pred_log_var)
                    else:
                        pred, dynamics = model(batch.x, batch.batch, return_dynamics=True)
                        mse_loss = mse_criterion(pred, batch.y)
                    
                    total_loss = mse_loss
                    
                    # Add biological constraints if enabled
                    if self.biological_constraints:
                        bio_loss, bio_metrics = bio_criterion(
                            dynamics['edge_weights'], 
                            dynamics['interaction_matrices']
                        )
                        total_loss += bio_loss
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_train_loss += total_loss.item()
                    num_train_batches += 1
                
                avg_train_loss = epoch_train_loss / num_train_batches
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        
                        if self.use_uncertainty:
                            pred_mean, pred_log_var = model(batch.x, batch.batch)
                            val_loss += mse_criterion(pred_mean, batch.y).item()
                            val_predictions.extend(pred_mean.cpu().numpy())
                        else:
                            pred = model(batch.x, batch.batch)
                            val_loss += mse_criterion(pred, batch.y).item()
                            val_predictions.extend(pred.cpu().numpy())
                        
                        val_targets.extend(batch.y.cpu().numpy())
                
                avg_val_loss = val_loss / len(test_loader)
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best model for this fold
                    if model_type == 'dynamic_gnn':  # Only save the main dynamic model
                        best_fold_model = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Evaluate final model
            val_predictions = np.array(val_predictions).flatten()
            val_targets = np.array(val_targets).flatten()
            
            fold_r2 = r2_score(val_targets, val_predictions)
            fold_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
            fold_mae = mean_absolute_error(val_targets, val_predictions)
            
            fold_results.append({
                'fold': fold_num,
                'r2': fold_r2,
                'rmse': fold_rmse,
                'mae': fold_mae,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            })
            
            print(f"    Fold {fold_num} Results: R² = {fold_r2:.4f}, RMSE = {fold_rmse:.4f}")
            
            # Keep best model overall
            if fold_r2 > best_r2:
                best_r2 = fold_r2
                if model_type == 'dynamic_gnn':
                    best_model = model
        
        # Calculate average metrics
        avg_r2 = np.mean([result['r2'] for result in fold_results])
        avg_rmse = np.mean([result['rmse'] for result in fold_results])
        avg_mae = np.mean([result['mae'] for result in fold_results])
        
        print(f"  📊 {model_type.upper()} Average Results:")
        print(f"     R² = {avg_r2:.4f} ± {np.std([r['r2'] for r in fold_results]):.4f}")
        print(f"     RMSE = {avg_rmse:.4f} ± {np.std([r['rmse'] for r in fold_results]):.4f}")
        
        return {
            'model': best_model,
            'fold_results': fold_results,
            'avg_metrics': {
                'r2': avg_r2,
                'rmse': avg_rmse,
                'mae': avg_mae
            }
        }
    
    def extract_enhanced_embeddings(self, model, data_objects):
        """
        Extract multi-scale embeddings with learned interaction patterns
        """
        print(f"\n🔬 Extracting enhanced embeddings with biological interactions...")
        
        model.eval()
        all_embeddings = []
        all_interactions = []
        
        # Create data loader for embedding extraction
        data_loader = DataLoader(data_objects, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                
                # Extract embeddings and interaction patterns
                embedding_data = model.get_embeddings(batch.x, batch.batch)
                
                # Store graph-level embeddings
                graph_embeddings = embedding_data['graph_embeddings'].cpu().numpy()
                all_embeddings.append(graph_embeddings)
                
                # Store interaction patterns for interpretability
                interaction_strengths = embedding_data['interaction_strengths']
                all_interactions.extend(interaction_strengths)
        
        # Concatenate all embeddings
        enhanced_embeddings = np.vstack(all_embeddings)
        
        print(f"✅ Extracted enhanced embeddings: {enhanced_embeddings.shape}")
        print(f"🔗 Captured {len(all_interactions)} interaction networks")
        
        return {
            'embeddings': enhanced_embeddings,
            'interactions': all_interactions,
            'embedding_dim': enhanced_embeddings.shape[1]
        }
    
    def train_ml_models_on_embeddings(self, embeddings, target_idx):
        """
        Train ML models on enhanced embeddings with 5-fold cross-validation
        """
        target_name = self.target_columns[target_idx]
        target_values = self.targets[:, target_idx]
        
        print(f"\n🤖 Training ML models on enhanced embeddings for {target_name}")
        
        # ML models to train
        ml_models = {
            'LinearSVR': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', LinearSVR(random_state=self.random_state, max_iter=2000))
            ]),
            'ExtraTrees': Pipeline([
                ('scaler', StandardScaler()),
                ('et', ExtraTreesRegressor(
                    n_estimators=200,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])
        }
        
        # 5-fold cross-validation results
        ml_results = {}
        
        for model_name, model_pipeline in ml_models.items():
            print(f"  Training {model_name}...")
            
            # Perform 5-fold cross-validation
            cv_scores = cross_val_score(
                model_pipeline, embeddings, target_values,
                cv=self.num_folds, scoring='r2',
                n_jobs=-1
            )
            
            # Also get RMSE scores
            cv_rmse_scores = -cross_val_score(
                model_pipeline, embeddings, target_values,
                cv=self.num_folds, scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            
            ml_results[model_name] = {
                'r2_scores': cv_scores,
                'rmse_scores': cv_rmse_scores,
                'mean_r2': np.mean(cv_scores),
                'std_r2': np.std(cv_scores),
                'mean_rmse': np.mean(cv_rmse_scores),
                'std_rmse': np.std(cv_rmse_scores)
            }
            
            print(f"    {model_name}: R² = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"    {model_name}: RMSE = {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}")
        
        return ml_results
    
    def analyze_biological_interactions(self, interactions, target_name):
        """
        Analyze learned biological interaction patterns
        """
        print(f"\n🔬 Analyzing biological interactions for {target_name}...")
        
        # Aggregate interaction strengths across samples
        num_families = len(self.microbial_families)
        avg_interaction_matrix = np.zeros((num_families, num_families))
        
        for interaction_matrix in interactions:
            avg_interaction_matrix += interaction_matrix.cpu().numpy()
        
        avg_interaction_matrix /= len(interactions)
        
        # Find strongest interactions
        interaction_threshold = np.percentile(avg_interaction_matrix, 95)
        strong_interactions = np.where(avg_interaction_matrix > interaction_threshold)
        
        print(f"  🔗 Found {len(strong_interactions[0])} strong microbial interactions")
        
        # Create interaction DataFrame for analysis
        interaction_data = []
        for i, j in zip(strong_interactions[0], strong_interactions[1]):
            if i != j:  # Exclude self-interactions
                interaction_data.append({
                    'microbe_1': self.microbial_families[i],
                    'microbe_2': self.microbial_families[j],
                    'interaction_strength': avg_interaction_matrix[i, j],
                    'avg_abundance_1': np.mean(self.original_features[:, i]),
                    'avg_abundance_2': np.mean(self.original_features[:, j])
                })
        
        interaction_df = pd.DataFrame(interaction_data)
        interaction_df = interaction_df.sort_values('interaction_strength', ascending=False)
        
        # Save biological insights
        interaction_df.to_csv(
            f"{self.save_dir}/biological_insights/{target_name}_interactions.csv",
            index=False
        )
        
        print(f"  💾 Saved interaction analysis to biological_insights/")
        
        return {
            'interaction_matrix': avg_interaction_matrix,
            'strong_interactions': interaction_df,
            'interaction_threshold': interaction_threshold
        }
    
    def visualize_interaction_network(self, interaction_analysis, target_name):
        """
        Create visualization of learned interaction network
        """
        print(f"  📊 Creating interaction network visualization...")
        
        plt.figure(figsize=(12, 10))
        
        # Plot interaction matrix heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(
            interaction_analysis['interaction_matrix'],
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Interaction Strength'}
        )
        plt.title(f'Microbial Interaction Network - {target_name}')
        plt.xlabel('Microbial Families')
        plt.ylabel('Microbial Families')
        
        # Plot strongest interactions
        plt.subplot(2, 2, 2)
        top_interactions = interaction_analysis['strong_interactions'].head(20)
        y_pos = np.arange(len(top_interactions))
        plt.barh(y_pos, top_interactions['interaction_strength'], color='coral')
        plt.yticks(y_pos, [f"{row['microbe_1'][:15]}\n↔ {row['microbe_2'][:15]}" 
                          for _, row in top_interactions.iterrows()], fontsize=8)
        plt.xlabel('Interaction Strength')
        plt.title('Top 20 Microbial Interactions')
        
        # Plot interaction strength distribution
        plt.subplot(2, 2, 3)
        plt.hist(interaction_analysis['interaction_matrix'].flatten(), 
                bins=50, alpha=0.7, color='skyblue')
        plt.axvline(interaction_analysis['interaction_threshold'], 
                   color='red', linestyle='--', label='Significance Threshold')
        plt.xlabel('Interaction Strength')
        plt.ylabel('Frequency')
        plt.title('Interaction Strength Distribution')
        plt.legend()
        
        # Plot abundance vs interaction correlation
        plt.subplot(2, 2, 4)
        strong_interactions = interaction_analysis['strong_interactions']
        plt.scatter(strong_interactions['avg_abundance_1'], 
                   strong_interactions['interaction_strength'],
                   alpha=0.6, color='green', label='Microbe 1')
        plt.scatter(strong_interactions['avg_abundance_2'], 
                   strong_interactions['interaction_strength'],
                   alpha=0.6, color='orange', label='Microbe 2')
        plt.xlabel('Average Abundance')
        plt.ylabel('Interaction Strength')
        plt.title('Abundance vs Interaction Strength')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/{target_name}_interaction_network.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    💾 Saved interaction network plot")
    
    def run_enhanced_pipeline(self):
        """
        Run the complete enhanced dynamic pipeline
        """
        print("🚀 Starting Enhanced Dynamic Microbial GNN Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Create dynamic data objects
        data_objects = self.create_dynamic_data_objects()
        
        # Step 3: Train Dynamic GNN models for each target
        all_results = {}
        
        for target_idx, target_name in enumerate(self.target_columns):
            print(f"\n🎯 Processing target: {target_name}")
            print("-" * 40)
            
            target_results = {}
            
            # Train all dynamic models
            for model_type in self.dynamic_models_to_train:
                model_results = self.train_dynamic_model(model_type, target_idx, data_objects)
                target_results[model_type] = model_results
                
                # Store results
                self.dynamic_results[f"{target_name}_{model_type}"] = model_results
            
            # Step 4: Select best dynamic model
            best_model_type = max(
                self.dynamic_models_to_train,
                key=lambda m: target_results[m]['avg_metrics']['r2']
            )
            best_model = target_results[best_model_type]['model']
            
            print(f"\n🏆 Best Dynamic Model for {target_name}: {best_model_type.upper()}")
            print(f"    Best R² = {target_results[best_model_type]['avg_metrics']['r2']:.4f}")
            
            # Step 5: Extract enhanced embeddings
            embedding_data = self.extract_enhanced_embeddings(best_model, data_objects)
            self.embeddings[target_name] = embedding_data
            
            # Step 6: Train ML models on enhanced embeddings
            ml_results = self.train_ml_models_on_embeddings(
                embedding_data['embeddings'], target_idx
            )
            self.ml_results[target_name] = ml_results
            
            # Step 7: Analyze biological interactions
            interaction_analysis = self.analyze_biological_interactions(
                embedding_data['interactions'], target_name
            )
            self.learned_interactions[target_name] = interaction_analysis
            
            # Step 8: Create visualizations
            self.visualize_interaction_network(interaction_analysis, target_name)
            
            # Store complete results for this target
            all_results[target_name] = {
                'dynamic_models': target_results,
                'best_model': best_model_type,
                'ml_models': ml_results,
                'biological_insights': interaction_analysis,
                'embeddings': embedding_data
            }
        
        # Step 9: Create comprehensive comparison plots
        self.create_comprehensive_plots(all_results)
        
        # Step 10: Save complete results
        self.save_all_results(all_results)
        
        print(f"\n🎉 Enhanced Dynamic Pipeline completed successfully!")
        print(f"📁 Results saved to: {self.save_dir}")
        
        return all_results
    
    def create_comprehensive_plots(self, all_results):
        """
        Create comprehensive comparison plots
        """
        print(f"\n📊 Creating comprehensive comparison plots...")
        
        # Model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, target_name in enumerate(self.target_columns):
            # Dynamic model comparison
            ax1 = axes[i, 0]
            model_types = []
            r2_scores = []
            rmse_scores = []
            
            for model_type in self.dynamic_models_to_train:
                model_types.append(model_type.replace('_', ' ').title())
                results = all_results[target_name]['dynamic_models'][model_type]
                r2_scores.append(results['avg_metrics']['r2'])
                rmse_scores.append(results['avg_metrics']['rmse'])
            
            x = np.arange(len(model_types))
            ax1.bar(x - 0.2, r2_scores, 0.4, label='R²', alpha=0.8)
            ax1_twin = ax1.twinx()
            ax1_twin.bar(x + 0.2, rmse_scores, 0.4, label='RMSE', alpha=0.8, color='orange')
            
            ax1.set_xlabel('Dynamic GNN Models')
            ax1.set_ylabel('R² Score', color='blue')
            ax1_twin.set_ylabel('RMSE', color='orange')
            ax1.set_title(f'{target_name} - Dynamic GNN Performance')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_types, rotation=45)
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
            
            # ML model comparison
            ax2 = axes[i, 1]
            ml_models = list(all_results[target_name]['ml_models'].keys())
            ml_r2_scores = [all_results[target_name]['ml_models'][model]['mean_r2'] 
                          for model in ml_models]
            ml_r2_stds = [all_results[target_name]['ml_models'][model]['std_r2'] 
                         for model in ml_models]
            
            ax2.bar(ml_models, ml_r2_scores, yerr=ml_r2_stds, capsize=5, alpha=0.8)
            ax2.set_xlabel('ML Models on Enhanced Embeddings')
            ax2.set_ylabel('R² Score')
            ax2.set_title(f'{target_name} - ML Model Performance')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/comprehensive_model_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved comprehensive comparison plots")
    
    def save_all_results(self, all_results):
        """
        Save all results to files
        """
        print(f"\n💾 Saving all results...")
        
        # Save results summary
        results_summary = {}
        for target_name in self.target_columns:
            target_data = all_results[target_name]
            
            # Dynamic model summary
            dynamic_summary = {}
            for model_type in self.dynamic_models_to_train:
                model_results = target_data['dynamic_models'][model_type]
                dynamic_summary[model_type] = model_results['avg_metrics']
            
            # ML model summary
            ml_summary = {}
            for model_name, model_results in target_data['ml_models'].items():
                ml_summary[model_name] = {
                    'mean_r2': model_results['mean_r2'],
                    'std_r2': model_results['std_r2'],
                    'mean_rmse': model_results['mean_rmse'],
                    'std_rmse': model_results['std_rmse']
                }
            
            results_summary[target_name] = {
                'dynamic_models': dynamic_summary,
                'best_dynamic_model': target_data['best_model'],
                'ml_models': ml_summary
            }
        
        # Save summary as JSON
        import json
        with open(f"{self.save_dir}/results_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save embeddings
        for target_name in self.target_columns:
            embedding_data = all_results[target_name]['embeddings']
            np.save(f"{self.save_dir}/embeddings/{target_name}_enhanced_embeddings.npy",
                   embedding_data['embeddings'])
        
        # Save models (state dicts)
        for target_name in self.target_columns:
            best_model_type = all_results[target_name]['best_model']  
            model_results = all_results[target_name]['dynamic_models'][best_model_type]
            if model_results['model'] is not None:
                torch.save(
                    model_results['model'].state_dict(),
                    f"{self.save_dir}/dynamic_models/{target_name}_best_{best_model_type}.pth"
                )
        
        print(f"  ✅ All results saved successfully")


def main():
    """
    Main function to run the enhanced dynamic pipeline
    """
    # Configuration
    data_path = "microbial_data.csv"  # Update with your data path
    
    # Initialize and run pipeline
    pipeline = EnhancedDynamicPipeline(
        data_path=data_path,
        target_columns=['ACE-km', 'H2-km'],
        hidden_dim=128,
        num_epochs=300,
        patience=30,
        sparsity_factor=0.15,
        use_uncertainty=True,
        biological_constraints=True,
        save_dir='./enhanced_dynamic_results'
    )
    
    # Run complete pipeline
    results = pipeline.run_enhanced_pipeline()
    
    return results


if __name__ == "__main__":
    results = main() 