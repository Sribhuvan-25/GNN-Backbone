"""
Improved Microbial Data Analysis Pipeline

This pipeline addresses the fundamental issues with GCN-based approaches for microbial data:
1. Small dataset size (54 samples)
2. Compositional nature of microbiome data
3. Need for proper feature engineering
4. More appropriate model selection

Key improvements:
- Compositional data preprocessing (CLR transformation)
- Feature selection based on biological relevance
- Simpler models with proper regularization
- Ensemble approaches
- Cross-validation optimized for small datasets
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import the working dataset class
from dataset_regression import MicrobialGNNDataset

class ImprovedMicrobialPipeline:
    """
    Improved pipeline specifically designed for microbial abundance data.
    
    Addresses key issues:
    - Small sample size (54 samples)
    - Compositional data nature
    - Need for feature selection
    - Proper cross-validation
    """
    
    def __init__(self, 
                 data_path="../Data/df.csv",
                 num_folds=5,
                 test_size=0.2,
                 random_state=42,
                 save_dir='./improved_results',
                 k_neighbors=5,
                 mantel_threshold=0.05,
                 graph_mode='family'):
        """
        Initialize the improved pipeline using the same data loading as other pipelines.
        """
        self.data_path = data_path
        self.num_folds = num_folds
        self.test_size = test_size
        self.random_state = random_state
        self.save_dir = save_dir
        
        # Create save directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        
        # Load data using the same method as other pipelines
        print("Loading and processing data using MicrobialGNNDataset...")
        self.dataset = MicrobialGNNDataset(
            data_path=data_path,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold,
            use_fast_correlation=False,
            graph_mode=graph_mode,
            family_filter_mode='strict'
        )
        
        # Extract features and targets from the dataset
        self.X, self.y, self.target_names = self.extract_features_and_targets()
        
    def extract_features_and_targets(self):
        """
        Extract features and targets from the MicrobialGNNDataset.
        """
        print("Extracting features and targets from dataset...")
        
        # Get features from the first data object (all have same node features structure)
        data_sample = self.dataset.data_list[0]
        num_nodes = data_sample.x.shape[0]
        num_samples = len(self.dataset.data_list)
        
        # Extract node features (abundance data) from all samples
        X = np.zeros((num_samples, num_nodes))
        y = np.zeros((num_samples, len(self.dataset.target_cols)))
        
        for i, data in enumerate(self.dataset.data_list):
            # Use mean of node features as sample features (since nodes represent families/OTUs)
            X[i] = data.x.numpy().flatten()
            y[i] = data.y.numpy().flatten()
        
        print(f"Extracted features shape: {X.shape}")
        print(f"Extracted targets shape: {y.shape}")
        print(f"Target names: {self.dataset.target_cols}")
        
        return X, y, self.dataset.target_cols
    
    def preprocess_microbiome_features(self, X):
        """
        Proper preprocessing for microbiome abundance data.
        """
        print("Applying microbiome-specific preprocessing...")
        
        # Convert to DataFrame for easier manipulation
        X_df = pd.DataFrame(X)
        
        # 1. Handle zeros (add small pseudocount for log transformation)
        X_nonzero = X_df + 1e-6
        
        # 2. Relative abundance (compositional data)
        X_rel = X_nonzero.div(X_nonzero.sum(axis=1), axis=0)
        
        # 3. Centered Log-Ratio (CLR) transformation - standard for microbiome data
        # CLR(x) = log(x/geometric_mean(x))
        geometric_mean = np.exp(np.log(X_rel).mean(axis=1))
        X_clr = np.log(X_rel.div(geometric_mean, axis=0))
        
        # 4. Remove features with very low variance (uninformative)
        feature_vars = X_clr.var()
        high_var_features = feature_vars > feature_vars.quantile(0.1)  # Keep top 90% by variance
        X_filtered = X_clr.loc[:, high_var_features]
        
        print(f"Feature preprocessing:")
        print(f"  - Applied CLR transformation")
        print(f"  - Removed low-variance features: {X_df.shape[1]} → {X_filtered.shape[1]}")
        
        return X_filtered.values  # Return as numpy array
    
    def create_optimized_models(self):
        """
        Create models optimized for small dataset with proper regularization.
        """
        models = {
            'Ridge_CV': Pipeline([
                ('scaler', RobustScaler()),
                ('feature_select', SelectKBest(f_regression, k=10)),  # Feature selection for small dataset
                ('regressor', Ridge(alpha=1.0))
            ]),
            
            'Lasso_CV': Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', Lasso(alpha=0.1, max_iter=5000))  # Built-in feature selection
            ]),
            
            'ElasticNet_CV': Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000))
            ]),
            
            'RandomForest_Small': Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', RandomForestRegressor(
                    n_estimators=50,      # Smaller ensemble for small data
                    max_depth=4,          # Limit depth
                    min_samples_split=8,  # Higher to prevent overfitting
                    min_samples_leaf=4,   # Higher to prevent overfitting
                    max_features='sqrt',
                    random_state=self.random_state
                ))
            ]),
            
            'GradientBoosting_Small': Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=50,      # Smaller ensemble
                    max_depth=3,          # Shallow trees
                    learning_rate=0.1,
                    subsample=0.8,        # Stochastic gradient boosting
                    random_state=self.random_state
                ))
            ]),
            
            'SVR_RBF': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', SVR(kernel='rbf', C=1.0, gamma='scale'))
            ])
        }
        
        return models
    
    def train_and_evaluate_models(self, X, y, target_name):
        """
        Train and evaluate models using proper cross-validation for small datasets.
        """
        print(f"\nTraining models for target: {target_name}")
        print("-" * 50)
        
        models = self.create_optimized_models()
        results = {}
        
        # Use Leave-One-Out or small k-fold for small dataset
        cv = KFold(n_splits=min(self.num_folds, len(X)), shuffle=True, random_state=self.random_state)
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
            cv_mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            
            # Fit on full data for feature importance (if available)
            model.fit(X, y)
            
            # Store results
            results[model_name] = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'cv_mse_mean': cv_mse_scores.mean(),
                'cv_mse_std': cv_mse_scores.std(),
                'cv_r2_scores': cv_scores,
                'model': model
            }
            
            print(f"    CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def plot_results(self, results, target_name):
        """
        Create comprehensive plots for model comparison.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Performance Analysis - {target_name}', fontsize=16)
        
        # 1. R² comparison
        ax1 = axes[0, 0]
        model_names = list(results.keys())
        r2_means = [results[name]['cv_r2_mean'] for name in model_names]
        r2_stds = [results[name]['cv_r2_std'] for name in model_names]
        
        bars = ax1.bar(model_names, r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
        ax1.set_title('Cross-Validation R² Scores')
        ax1.set_ylabel('R²')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Color bars based on performance
        for bar, r2 in zip(bars, r2_means):
            if r2 > 0.1:
                bar.set_color('green')
            elif r2 > 0:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 2. CV score distribution
        ax2 = axes[0, 1]
        cv_data = [results[name]['cv_r2_scores'] for name in model_names]
        ax2.boxplot(cv_data, labels=model_names)
        ax2.set_title('R² Score Distribution Across Folds')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. MSE comparison
        ax3 = axes[1, 0]
        mse_means = [results[name]['cv_mse_mean'] for name in model_names]
        mse_stds = [results[name]['cv_mse_std'] for name in model_names]
        
        ax3.bar(model_names, mse_means, yerr=mse_stds, capsize=5, alpha=0.7, color='lightblue')
        ax3.set_title('Cross-Validation MSE')
        ax3.set_ylabel('MSE')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for name in model_names:
            r2 = results[name]['cv_r2_mean']
            r2_std = results[name]['cv_r2_std']
            mse = results[name]['cv_mse_mean']
            table_data.append([name, f"{r2:.4f} ± {r2_std:.3f}", f"{mse:.2f}"])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Model', 'CV R² (±std)', 'CV MSE'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        ax4.set_title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/plots/model_comparison_{target_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_pipeline(self):
        """
        Run the complete improved pipeline.
        """
        print("="*80)
        print("IMPROVED MICROBIAL DATA ANALYSIS PIPELINE")
        print("="*80)
        
        # Preprocess the features
        print("\nPreprocessing microbial features...")
        X_processed = self.preprocess_microbiome_features(self.X)
        
        all_results = {}
        
        # Process each target
        for target_idx, target_name in enumerate(self.target_names):
            print(f"\n{'='*60}")
            print(f"PROCESSING TARGET: {target_name}")
            print(f"{'='*60}")
            
            target_values = self.y[:, target_idx]
            print(f"Target range: {target_values.min():.2f} - {target_values.max():.2f}")
            
            # Train and evaluate models
            results = self.train_and_evaluate_models(X_processed, target_values, target_name)
            
            # Create plots
            self.plot_results(results, target_name)
            
            all_results[target_name] = results
        
        # Summary
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETED - SUMMARY")
        print(f"{'='*80}")
        
        summary_data = []
        for target_name, target_results in all_results.items():
            print(f"\n{target_name}:")
            best_model = max(target_results.keys(), key=lambda k: target_results[k]['cv_r2_mean'])
            best_r2 = target_results[best_model]['cv_r2_mean']
            print(f"  Best Model: {best_model}")
            print(f"  Best CV R²: {best_r2:.4f}")
            
            summary_data.append({
                'Target': target_name,
                'Best_Model': best_model,
                'CV_R2': best_r2,
                'CV_R2_std': target_results[best_model]['cv_r2_std'],
                'CV_MSE': target_results[best_model]['cv_mse_mean']
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{self.save_dir}/performance_summary.csv', index=False)
        
        print(f"\nResults saved to: {self.save_dir}")
        return all_results

def main():
    """Main execution function."""
    print("Starting Improved Microbial Data Analysis Pipeline...")
    
    # Initialize pipeline
    pipeline = ImprovedMicrobialPipeline(
        data_path="../Data/df.csv",
        num_folds=5,
        random_state=42,
        save_dir='./improved_results'
    )
    
    # Run pipeline
    results = pipeline.run_pipeline()
    
    print("\nImproved Pipeline completed successfully!")
    return results

if __name__ == "__main__":
    main() 