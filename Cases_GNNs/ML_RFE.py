import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create necessary directories
def create_directories():
    """Create necessary directories for outputs"""
    directories = [
        'results_extension',
        'results_extension/metrics',
        'results_extension/plots'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_feature_lists():
    """Load the specific feature lists based on the image provided"""
    # These are the specific OTUs from the image with their classifications
    acetoclastic = [
        "d__Archaea;p__Halobacterota;c__Methanosarcinia;o__Methanosarciniales;f__Methanosaetaceae;g__Methanosaeta"
    ]
    
    hydrogenotrophic = [
        "d__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium",
        # "d__Archaea;p__Halobacterota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium"  # This is an alternate format that might be in the data
    ]
    
    syntrophic = [
        "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Smithellaceae;g__Smithella",
        "d__Bacteria;p__Desulfobacterota;c__Syntrophobacteria;o__Syntrophobacterales;f__Syntrophobacteraceae;g__Syntrophobacter",
        "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Syner-01",
        "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__uncultured",
        "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Syntrophaceae;g__Syntrophus",
        "d__Bacteria;p__Desulfobacterota;c__Desulfuromonadia;o__Geobacterales;f__Geobacteraceae;g__uncultured"
    ]
    
    print(f"Using hardcoded taxonomic features from the image:")
    print(f"- Acetoclastic: {len(acetoclastic)} feature(s)")
    print(f"- Hydrogenotrophic: {len(hydrogenotrophic)} feature(s)")
    print(f"- Syntrophic: {len(syntrophic)} feature(s)")
    
    return acetoclastic, hydrogenotrophic, syntrophic

def select_features_with_rfe(X, y, anchored_features, n_features=5):
    """
    Perform RFE while keeping anchored features and return feature importance
    """
    # Separate anchored and non-anchored features
    non_anchored_features = [col for col in X.columns if col not in anchored_features]
    X_non_anchored = X[non_anchored_features]
    
    if n_features > 0 and len(non_anchored_features) > 0:
        # Perform RFE and get feature importance with increased max_iter to avoid convergence warnings
        estimator = LinearSVR(random_state=42, max_iter=1000, tol=1e-4, dual=True)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X_non_anchored, y)
        
        # Get feature importance for selected non-anchored features
        selected_mask = rfe.support_
        selected_features = X_non_anchored.columns[selected_mask]
        final_model = LinearSVR(random_state=42, max_iter=10000, tol=1e-4, dual=True)
        final_model.fit(X_non_anchored[selected_features], y)
        
        # Calculate importance scores and sort features
        importance_scores = np.abs(final_model.coef_)
        feature_importance_pairs = list(zip(selected_features, importance_scores))
        sorted_features = [f for f, _ in sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)]
        sorted_importance = [i for _, i in sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)]
        
        # Combine anchored and sorted non-anchored features
        final_features = anchored_features + sorted_features
        final_importance = [0] * len(anchored_features) + sorted_importance
    else:
        final_features = anchored_features
        final_importance = [0] * len(anchored_features)
    
    return final_features, final_importance

def create_performance_plot(all_actual, all_predictions, target, mode, r2, mse, n_features, df=None):
    """
    Create and save a scatter plot of actual vs. predicted values.
    For Case 9, color points by biomass type if df is provided.
    
    Parameters:
    ----------
    all_actual : list
        Actual target values
    all_predictions : list
        Predicted target values
    target : str
        Name of the target variable
    mode : str
        Mode of analysis (case4, case5, etc.)
    r2 : float
        R² score
    mse : float
        Mean squared error
    n_features : int
        Number of features used
    df : pandas.DataFrame, optional
        DataFrame with biomass information for coloring points
    """
    # Add debugging for biomass coloring
    print(f"\n{'#'*50}")
    print(f"DEBUG - create_performance_plot for {mode}")
    print(f"df parameter provided: {df is not None}")
    if df is not None:
        print(f"df shape: {df.shape}")
        print(f"all_actual length: {len(all_actual)}")
        print(f"Length check passes: {len(df) == len(all_actual)}")
        
        # Check which biomass columns are available
        biomass_cols = [col for col in df.columns if col.startswith('Biomass_')]
        print(f"Biomass columns found: {len(biomass_cols)}")
        if biomass_cols:
            print(f"Biomass columns: {biomass_cols}")
    print(f"{'#'*50}\n")
    
    plt.figure(figsize=(10, 8))
    
    # Find min and max values for setting axis limits
    min_val = min(min(all_actual), min(all_predictions))
    max_val = max(max(all_actual), max(all_predictions))
    
    # Add some padding to the limits
    range_val = max_val - min_val
    min_val -= range_val * 0.05
    max_val += range_val * 0.05
    
    # Plot the perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7, label='Perfect Prediction')
    
    # For Case 9, color points by biomass type if df is provided
    if mode == "case9" and df is not None and len(df) == len(all_actual):
        # Check which biomass columns are available
        biomass_cols = [col for col in df.columns if col.startswith('Biomass_')]
        
        if biomass_cols:
            print(f"Found {len(biomass_cols)} biomass types for coloring in Case 9 plot")
            
            # Create a color map with distinct colors
            colors = plt.cm.tab10(np.linspace(0, 1, len(biomass_cols) + 1))
            biomass_colors = {f"Biomass_{c}": colors[i] for i, c in enumerate(
                [col.split('_')[1] for col in biomass_cols])}
            biomass_colors['Unknown'] = colors[-1]
            
            # Create a mask for each biomass type
            for i, col in enumerate(biomass_cols):
                biomass_type = col.split('_')[1]
                mask = df[col] == 1
                samples_count = sum(mask)
                
                if samples_count > 0:
                    print(f"  Adding {samples_count} samples for biomass type '{biomass_type}'")
                    plt.scatter(
                        np.array(all_actual)[mask], 
                        np.array(all_predictions)[mask],
                        alpha=0.8, 
                        color=biomass_colors[col], 
                        label=f'Biomass {biomass_type} (n={samples_count})',
                        s=50,
                        edgecolors='k',
                        linewidths=0.5
                    )
            
            # Plot any points not assigned to a biomass type
            unknown_mask = ~df[biomass_cols].any(axis=1)
            unknown_count = sum(unknown_mask)
            
            if unknown_count > 0:
                print(f"  Adding {unknown_count} samples with unknown biomass type")
                plt.scatter(
                    np.array(all_actual)[unknown_mask], 
                    np.array(all_predictions)[unknown_mask],
                    alpha=0.8, 
                    color=biomass_colors['Unknown'], 
                    label=f'Unknown Biomass (n={unknown_count})',
                    s=50,
                    edgecolors='k',
                    linewidths=0.5
                )
        else:
            print("No biomass columns found in the DataFrame. Using default coloring.")
            # Default scatter plot if no biomass columns
            plt.scatter(all_actual, all_predictions, alpha=0.7, color='#1f77b4', 
                       s=50, label=f'Samples (n={len(all_actual)})')
    else:
        # Default scatter plot for other modes
        plt.scatter(all_actual, all_predictions, alpha=0.7, color='#1f77b4', 
                   s=50, label=f'Samples (n={len(all_actual)})')
    
    # Configure plot aesthetics
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.title(f'{mode} - {target} Prediction Performance\nR² = {r2:.4f}, MSE = {mse:.4f}, Features = {n_features}')
    
    # Set equal aspect and limits
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add a grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    # Add text with performance metrics in the bottom right
    plt.annotate(f'R² = {r2:.4f}\nMSE = {mse:.4f}\nRMSE = {np.sqrt(mse):.4f}',
                xy=(0.02, 0.96), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='top')
    
    # Save the plot as PNG
    plot_filename = f"results_extension/plots/{mode}_{target}_n{n_features}.png"
    
    # For Case 9, save a special version with biomass coloring
    if mode == "case9" and df is not None and any(col.startswith('Biomass_') for col in df.columns):
        biomass_plot_filename = f"results_extension/plots/{mode}_{target}_n{n_features}_biomass_colored.png"
        plt.savefig(biomass_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved biomass-colored plot to: {biomass_plot_filename}")
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

def filter_available_features(df, feature_list):
    """Check which features from the provided list are available in the dataset"""
    available_features = [feature for feature in feature_list if feature in df.columns]
    missing_features = [feature for feature in feature_list if feature not in df.columns]
    
    return available_features, missing_features

def run_linearsvr_cv(data_path, target="ACE-km", mode="case4", n_features=50):
    """
    Run LinearSVR with different feature anchoring strategies for extension cases
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file
    target : str
        Target variable to predict ("ACE-km")
    mode : str
        Feature selection mode ("case4", "case5", "case6", "case7", "case8", "case9")
    n_features : int
        Number of additional features to select beyond anchored features
        
    Returns:
    -------
    final_r2 : float
        R² score on the test set
    final_mse : float
        Mean Squared Error on the test set
    final_rmse : float
        Root Mean Squared Error on the test set
    results_df : DataFrame
        DataFrame with actual and predicted values
    sample_info : dict
        Dictionary with sample count information
    """
    # Add directory creation at the start
    create_directories()
    
    # Track sample counts
    sample_info = {}
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Add debugging for biomass columns before any processing
    print(f"\n{'*'*50}")
    print(f"DEBUG - Data loaded from: {data_path}")
    print(f"Original data shape: {df.shape}")
    
    # Check for biomass columns in original data
    original_biomass_cols = [col for col in df.columns if col.startswith('Biomass_')]
    print(f"Biomass columns in original data: {len(original_biomass_cols)}")
    if original_biomass_cols:
        print(f"Original biomass columns: {original_biomass_cols}")
    else:
        print("No biomass columns found in the original data!")
    print(f"{'*'*50}\n")
    
    sample_info['original'] = len(df)
    
    # Remove rows containing 'x'
    df = df[~df.isin(['x']).any(axis=1)]
    sample_info['after_removing_x'] = len(df)
    
    print(f"\n{'*' * 20} SAMPLE COUNT INFORMATION {'*' * 20}")
    print(f"Original dataset: {sample_info['original']} samples")
    print(f"After removing rows with 'x': {sample_info['after_removing_x']} samples")
    
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)
    
    # Load feature lists
    acetoclastic, hydrogenotrophic, syntrophic = load_feature_lists()
    
    # Check which features are available in the dataset
    acetoclastic_available, acetoclastic_missing = filter_available_features(df, acetoclastic)
    hydrogenotrophic_available, hydrogenotrophic_missing = filter_available_features(df, hydrogenotrophic)
    syntrophic_available, syntrophic_missing = filter_available_features(df, syntrophic)
    
    # Print availability information
    print("\nFeature availability:")
    print(f"- Acetoclastic: {len(acetoclastic_available)}/{len(acetoclastic)} available")
    if acetoclastic_missing:
        print(f"  Missing: {acetoclastic_missing}")
    
    print(f"- Hydrogenotrophic: {len(hydrogenotrophic_available)}/{len(hydrogenotrophic)} available")
    if hydrogenotrophic_missing:
        print(f"  Missing: {hydrogenotrophic_missing}")
    
    print(f"- Syntrophic: {len(syntrophic_available)}/{len(syntrophic)} available")
    if syntrophic_missing:
        print(f"  Missing: {syntrophic_missing}")
    
    # Get all available features (excluding target columns)
    target_columns = ["Average-Total-ISD-Cells", "ACE-Xi", "ACE-km", "ACE-Ks", 
                     "H2-Xi", "H2-km", "H2-Ks"]
    all_features = [col for col in df.columns if col not in target_columns]
    
    # Remove outlier at ACE-km ~ 33 (actual ~ 50)
    # Using a threshold of 30 for ACE-km to identify the outlier
    outlier_mask = df[target] > 30
    outliers_removed = sum(outlier_mask)
    sample_info['outliers_removed'] = outliers_removed
    
    if outliers_removed > 0:
        print(f"\nRemoving {outliers_removed} outlier(s) with {target} > 30")
        df = df[~outlier_mask]
    
    sample_info['after_outlier_removal'] = len(df)
    print(f"Samples after outlier removal: {sample_info['after_outlier_removal']}")
    
    # Store original DataFrame for coloring plots by biomass type
    # This must be stored BEFORE we apply any filtering based on Biomass types
    original_df = df.copy()
    
    # Check for biomass type columns
    biomass_cols = [col for col in df.columns if col.startswith('Biomass_')]
    if biomass_cols:
        print(f"\nFound {len(biomass_cols)} biomass type columns: {', '.join(biomass_cols)}")
        for col in biomass_cols:
            biomass_type = col.split('_')[1]
            count = df[col].sum()
            percentage = count / len(df) * 100
            print(f"  - Biomass {biomass_type}: {count} samples ({percentage:.1f}%)")
        
        # Count samples with no biomass type assigned
        no_biomass = (~df[biomass_cols].any(axis=1)).sum()
        if no_biomass > 0:
            percentage = no_biomass / len(df) * 100
            print(f"  - Unknown biomass: {no_biomass} samples ({percentage:.1f}%)")

    # Apply additional filtering based on the case
    if mode == "case4":
        # Case 4: Use all 8 features plus additional features and all ACE-km values
        anchored_features = acetoclastic_available + hydrogenotrophic_available + syntrophic_available
        print(f"\n*** CASE 4: Using all {len(df)} samples with all 8 classified features ***")
        sample_info['final'] = len(df)
        # No additional filtering
    elif mode == "case5":
        # Case 5: Use acetoclastic feature plus additional features and ACE-km values <= 10
        anchored_features = acetoclastic_available
        samples_before = len(df)
        df = df[df[target] <= 10]
        samples_after = len(df)
        sample_info['before_filtering'] = samples_before
        sample_info['final'] = samples_after
        print(f"\n*** CASE 5: Filtered to {samples_after}/{samples_before} samples with {target} ≤ 10 ***")
        print(f"Percentage of total: {samples_after/samples_before:.1%}")
    elif mode == "case6":
        # Case 6: Use acetoclastic feature plus additional features for Biomass "F"
        anchored_features = acetoclastic_available
        if "Biomass_F" in df.columns:
            samples_before = len(df)
            df = df[df["Biomass_F"] == 1]
            samples_after = len(df)
            sample_info['before_filtering'] = samples_before
            sample_info['final'] = samples_after
            print(f"\n*** CASE 6: Filtered to {samples_after}/{samples_before} samples with Biomass F ***")
            print(f"Percentage of total: {samples_after/samples_before:.1%}")
            
            # Print distribution of target values in this subset
            print(f"\nDistribution of {target} values in Biomass F subset ({samples_after} samples):")
            print(f"  Mean: {df[target].mean():.2f}")
            print(f"  Median: {df[target].median():.2f}")
            print(f"  Min: {df[target].min():.2f}")
            print(f"  Max: {df[target].max():.2f}")
            print(f"  Standard deviation: {df[target].std():.2f}")
        else:
            print("Warning: Biomass_F column not found")
            sample_info['final'] = len(df)
    elif mode == "case7":
        # Case 7: Use all 8 features plus additional features and ACE-km values > 10
        anchored_features = acetoclastic_available + hydrogenotrophic_available + syntrophic_available
        samples_before = len(df)
        df = df[df[target] > 10]
        samples_after = len(df)
        sample_info['before_filtering'] = samples_before
        sample_info['final'] = samples_after
        print(f"\n*** CASE 7: Filtered to {samples_after}/{samples_before} samples with {target} > 10 ***")
        print(f"Percentage of total: {samples_after/samples_before:.1%}")
    elif mode == "case8":
        # Case 8: Use all 8 features plus additional features for Biomass "G"
        anchored_features = acetoclastic_available + hydrogenotrophic_available + syntrophic_available
        if "Biomass_G" in df.columns:
            samples_before = len(df)
            df = df[df["Biomass_G"] == 1]
            samples_after = len(df)
            sample_info['before_filtering'] = samples_before
            sample_info['final'] = samples_after
            print(f"\n*** CASE 8: Filtered to {samples_after}/{samples_before} samples with Biomass G ***")
            print(f"Percentage of total: {samples_after/samples_before:.1%}")
            
            # Print distribution of target values in this subset
            print(f"\nDistribution of {target} values in Biomass G subset ({samples_after} samples):")
            print(f"  Mean: {df[target].mean():.2f}")
            print(f"  Median: {df[target].median():.2f}")
            print(f"  Min: {df[target].min():.2f}")
            print(f"  Max: {df[target].max():.2f}")
            print(f"  Standard deviation: {df[target].std():.2f}")
        else:
            print("Warning: Biomass_G column not found")
            sample_info['final'] = len(df)
    elif mode == "case9":
        # Case 9: Use acetoclastic feature plus additional features and all ACE-km values (similar to Case 5 but no filtering)
        anchored_features = acetoclastic_available
        print(f"\n*** CASE 9: Using all {len(df)} samples with 1 acetoclastic feature (no filtering) ***")
        sample_info['final'] = len(df)
        
        # Print distribution of target values in the full dataset
        print(f"\nDistribution of {target} values in full dataset ({len(df)} samples):")
        print(f"  Mean: {df[target].mean():.2f}")
        print(f"  Median: {df[target].median():.2f}")
        print(f"  Min: {df[target].min():.2f}")
        print(f"  Max: {df[target].max():.2f}")
        print(f"  Standard deviation: {df[target].std():.2f}")
        
        # For Case 9, print biomass type distribution if available
        biomass_cols = [col for col in df.columns if col.startswith('Biomass_')]
        if biomass_cols:
            print("\nBiomass type distribution:")
            for col in biomass_cols:
                biomass_type = col.split('_')[1]
                count = df[col].sum()
                percentage = count / len(df) * 100
                print(f"  - Biomass {biomass_type}: {count} samples ({percentage:.1f}%)")
            
            # Count samples with no biomass type assigned
            no_biomass = (~df[biomass_cols].any(axis=1)).sum()
            if no_biomass > 0:
                percentage = no_biomass / len(df) * 100
                print(f"  - Unknown biomass: {no_biomass} samples ({percentage:.1f}%)")
    else:
        raise ValueError("Invalid mode specified")
    
    print(f"\n{'*' * 20} FINAL SAMPLE COUNT: {sample_info['final']} {'*' * 20}")
    
    # Check if we have enough samples to proceed
    if len(df) < 5:
        print(f"Warning: Only {len(df)} samples left after filtering. Cannot proceed with cross-validation.")
        return None, None, None, None, sample_info
    
    # Check if any anchored features are available
    if not anchored_features:
        print("Warning: No anchored features available in the dataset. Cannot proceed.")
        return None, None, None, None, sample_info
    
    # Perform feature selection
    selected_features, importance = select_features_with_rfe(
        df[all_features], 
        df[target], 
        anchored_features, 
        n_features
    )
    X = df[selected_features]
    y = df[target]
    
    print(f"\nDataset shape after cleaning and filtering: {df.shape}")
    print(f"Number of features used: {len(X.columns)}")
    print(f"Number of anchored features used: {len(anchored_features)}/{len(selected_features)} ({len(anchored_features)/len(selected_features):.1%})")

    # Add sample to feature ratio information
    sample_to_feature_ratio = len(df) / len(selected_features)
    sample_info['sample_to_feature_ratio'] = sample_to_feature_ratio
    print(f"Sample to feature ratio: {sample_to_feature_ratio:.2f}")
    
    # Check if the number of samples is low relative to the number of features
    if sample_to_feature_ratio < 5:
        print(f"\nWarning: Sample size ({len(df)}) is less than 5 times the number of features ({len(selected_features)}).")
        print("This may lead to overfitting and unreliable model performance metrics.")
    
    # Initialize K-Fold
    n_splits = min(5, len(df))  # Adjust number of folds if sample size is very small
    sample_info['n_folds'] = n_splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results
    all_predictions = []
    all_actual = []
    fold_r2_scores = []
    
    # Modify LinearSVR parameters for better convergence
    model = LinearSVR(
        random_state=42,
        max_iter=10000,  # Increased from 2000 to 10000 to avoid convergence warnings
        tol=1e-4,       # Adjust tolerance
        dual=True       # Use dual formulation
    )
    
    # Perform cross-validation
    print(f"\nRunning {n_splits}-fold cross-validation:")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"  Fold {fold+1}: Training on {len(X_train)} samples, Validating on {len(X_val)} samples")
        
        # Standard scaling before model fitting
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        
        # Calculate fold-specific R²
        fold_r2 = r2_score(y_val, y_pred)
        fold_r2_scores.append(fold_r2)
        
        all_predictions.extend(y_pred)
        all_actual.extend(y_val)
        
        print(f"  Fold {fold+1} R²: {fold_r2:.4f}")

    # Calculate and display metrics
    # Note on R² calculation: R² (coefficient of determination) measures how well the model explains
    # the variance in the data compared to a simple mean model. It can range from -inf to 1:
    # - R² = 1: Perfect prediction, all variance explained
    # - R² = 0: Model performs same as predicting the mean value
    # - R² < 0: Model performs worse than predicting the mean value (can happen with small samples or poor fit)
    # Negative R² values indicate that the model is worse than just predicting the mean of the data.
    # This often occurs with small sample sizes, noisy data, or when the model doesn't fit the data well.
    final_r2 = r2_score(all_actual, all_predictions)
    final_mse = mean_squared_error(all_actual, all_predictions)
    final_rmse = np.sqrt(final_mse)
    
    # Store metrics in sample_info
    sample_info['R2'] = final_r2
    sample_info['MSE'] = final_mse
    sample_info['RMSE'] = final_rmse

    print(f"\nResults for {target} ({mode} mode with {sample_info['final']} samples):")
    print(f"R² Score: {final_r2:.4f}")
    print(f"Mean Fold R²: {np.mean(fold_r2_scores):.4f}")
    print(f"Fold R² Std Dev: {np.std(fold_r2_scores):.4f}")
    print(f"MSE: {final_mse:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    
    # If R² is negative, add an explanation
    if final_r2 < 0:
        print("\nNote: Negative R² value indicates the model performs worse than predicting the mean value.")
        print("This may be due to:")
        print(f"  - Small sample size (n={len(df)})")
        print(f"  - High variance in the data (target std dev: {df[target].std():.2f})")
        print("  - Non-linear relationships that LinearSVR can't capture")
        print("  - Selected features may not be predictive for this subset")
        
        # Simple mean predictor baseline
        y_mean = np.mean(all_actual)
        mean_mse = mean_squared_error(all_actual, [y_mean] * len(all_actual))
        print(f"\nBaseline (mean predictor) MSE: {mean_mse:.4f}")
        print(f"Model MSE: {final_mse:.4f}")
        if final_mse > mean_mse:
            print(f"Model performs worse than baseline by {final_mse/mean_mse - 1:.1%}")

    # Track selected features
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Is_Anchored': [f in anchored_features for f in selected_features],
        'Feature_Importance': importance
    })
    
    # Save feature importance information
    feature_importance.to_csv(f'results_extension/metrics/features_{target}_{mode}_{n_features}features.csv', index=False)

    # Create and save plot
    # For Case 9, pass the original DataFrame to color by biomass type
    if mode == "case9":
        print(f"\n{'*'*50}")
        print(f"DEBUG - Calling create_performance_plot for case9")
        print(f"original_df shape: {original_df.shape}")
        biomass_cols = [col for col in original_df.columns if col.startswith('Biomass_')]
        print(f"Biomass columns in original_df: {len(biomass_cols)}")
        if biomass_cols:
            print(f"Biomass columns: {biomass_cols}")
        print(f"all_actual length: {len(all_actual)}")
        print(f"{'*'*50}\n")
        
        create_performance_plot(all_actual, all_predictions, target, mode, final_r2, final_mse, n_features, original_df)
    else:
        create_performance_plot(all_actual, all_predictions, target, mode, final_r2, final_mse, n_features)

    # Save numerical results
    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    results_df.to_csv(f'results_extension/metrics/results_{target}_{mode}_{n_features}features.csv', index=False)

    # Update the main method to handle the additional return value
    return final_r2, final_mse, final_rmse, results_df, sample_info

def create_combined_plot(case_results, case_names):
    """
    Create a combined plot showing multiple cases with separate regression lines
    
    Parameters:
    ----------
    case_results : dict
        Dictionary with case name as key and DataFrame with 'Actual' and 'Predicted' columns as value
    case_names : list
        List of case names to include in the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Define colors for different cases
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Initialize for axis limits
    all_actual = []
    all_predicted = []
    all_r2 = {}
    all_mse = {}
    
    # Map case names to more readable descriptions
    case_descriptions = {
        'case4': 'All Features',
        'case5': 'ACE-km ≤ 10',
        'case6': 'Biomass F',
        'case7': 'ACE-km > 10',
        'case8': 'Biomass G',
        'case9': 'All Values (1 Feature)'
    }
    
    # Plot each case with its own color and regression line
    for i, case in enumerate(case_names):
        if case not in case_results:
            print(f"Warning: {case} not found in results")
            continue
            
        results = case_results[case]
        actual = results['Actual']
        predicted = results['Predicted']
        
        # Get readable case name
        case_desc = case_descriptions.get(case, case)
        
        # Add to aggregate data for axis limits
        all_actual.extend(actual)
        all_predicted.extend(predicted)
        
        # Calculate metrics
        r2 = r2_score(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        all_r2[case] = r2
        all_mse[case] = mse
        
        # Plot scatter for this case
        color = colors[i % len(colors)]
        plt.scatter(actual, predicted, alpha=0.7, s=40, 
                   color=color, label=f'{case_desc} (R²={r2:.4f})')
        
        # Calculate and plot regression line for this case
        coeffs = np.polyfit(actual, predicted, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Create label for regression line
        reg_label = f'{case_desc} Regression (y={slope:.2f}x+{intercept:.2f})'
        
        # Add regression line with label
        x_min, x_max = min(actual), max(actual)
        x_reg = np.array([x_min, x_max])
        y_reg = slope * x_reg + intercept
        plt.plot(x_reg, y_reg, color=color, linestyle='-', alpha=0.8, label=reg_label)
    
    # Calculate padding for axis limits
    min_val_x = min(all_actual)
    max_val_x = max(all_actual)
    min_val_y = min(all_predicted)
    max_val_y = max(all_predicted)
    
    x_padding = (max_val_x - min_val_x) * 0.1
    y_padding = (max_val_y - min_val_y) * 0.1
    
    # If padding is too small, use a minimum padding
    min_padding = max(max_val_x, max_val_y) * 0.05
    x_padding = max(x_padding, min_padding)
    y_padding = max(y_padding, min_padding)
    
    # Set axis limits with padding
    x_min = max(0, min_val_x - x_padding)
    x_max = max_val_x + x_padding
    y_min = max(0, min_val_y - y_padding)
    y_max = max_val_y + y_padding
    
    # Use the maximum range for both axes
    plot_min = min(x_min, y_min)
    plot_max = max(x_max, y_max)
    
    # Add perfect prediction line
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5, label='Perfect Prediction (y=x)')
    
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    
    # Create title based on the cases being combined
    case_desc_list = [case_descriptions.get(case, case) for case in case_names if case in case_results]
    title = f"Combined Plot: {' vs '.join(case_desc_list)}"
    
    # Calculate combined R² and MSE if needed
    if len(case_names) > 1:
        all_actual_combined = []
        all_predicted_combined = []
        for case in case_names:
            if case in case_results:
                all_actual_combined.extend(case_results[case]['Actual'])
                all_predicted_combined.extend(case_results[case]['Predicted'])
        
        if all_actual_combined:
            combined_r2 = r2_score(all_actual_combined, all_predicted_combined)
            combined_mse = mean_squared_error(all_actual_combined, all_predicted_combined)
            combined_metrics = f'Combined: R² = {combined_r2:.4f}, MSE = {combined_mse:.4f}'
            plt.title(f'{title}\n{combined_metrics}', fontsize=14)
        else:
            plt.title(title, fontsize=14)
    else:
        plt.title(title, fontsize=14)
    
    # Improve the legend by adjusting size and position
    plt.legend(fontsize=9, loc='best', framealpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot with specific filename
    case_str = '_and_'.join(case_names)
    filename = f'results_extension/plots/combined_results_{case_str}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot saved as {filename}")
    
    return all_r2, all_mse

def run_case9_with_biomass_coloring(data_path="../Data/New_data.csv", feature_counts=[50, 100, 200, 400, 800]):
    """
    Run Case 9 analysis with biomass coloring for plots
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file
    feature_counts : list
        List of additional feature counts to test
    """
    print(f"\n{'='*80}")
    print(f"RUNNING CASE 9 WITH BIOMASS COLORING:")
    print(f"Using data from: {data_path}")
    print(f"Testing feature counts: {feature_counts}")
    print(f"{'='*80}")
    
    # Initialize results dictionary
    case9_results = {}
    
    for n_features in feature_counts:
        print(f"\n{'-'*60}")
        print(f"Running Case 9 with {n_features} additional features")
        print(f"{'-'*60}")
        
        # Run the model for Case 9
        r2, mse, rmse, results, sample_info = run_linearsvr_cv(
            data_path=data_path,
            target="ACE-km",
            mode="case9",
            n_features=n_features
        )
        
        # Store results if model ran successfully
        if r2 is not None:
            case9_results[n_features] = {
                'R2': r2,
                'MSE': mse,
                'RMSE': rmse,
                'Sample_Count': sample_info.get('final', 0),
                'Sample_to_Feature_Ratio': sample_info.get('sample_to_feature_ratio', 0)
            }
    
    # Print summary of results
    if case9_results:
        print(f"\n{'='*80}")
        print(f"CASE 9 RESULTS SUMMARY WITH BIOMASS COLORING:")
        print(f"{'='*80}")
        
        # Create formatted table
        print(f"\nFeature count comparison:")
        features_data = []
        
        for n_features, results in sorted(case9_results.items()):
            features_data.append({
                'Features': n_features,
                'R²': f"{results['R2']:.4f}",
                'MSE': f"{results['MSE']:.4f}",
                'RMSE': f"{results['RMSE']:.4f}",
                'Sample/Feature': f"{results['Sample_to_Feature_Ratio']:.2f}"
            })
        
        if features_data:
            # Print as a small table
            cols = list(features_data[0].keys())
            widths = {col: max(len(str(row[col])) for row in features_data) for col in cols}
            widths = {col: max(widths[col], len(col)) for col in widths}
            
            # Header
            feat_header = ' | '.join(f"{col:{widths[col]}}" for col in cols)
            print(feat_header)
            print('-' * len(feat_header))
            
            # Rows
            for row in features_data:
                print(' | '.join(f"{str(row[col]):{widths[col]}}" for col in cols))
        
        # Identify best feature count based on R²
        best_count = max(case9_results.keys(), key=lambda k: case9_results[k]['R2'])
        best_r2 = case9_results[best_count]['R2']
        
        print(f"\nBest performing feature count: {best_count} additional features")
        print(f"Best R² Score: {best_r2:.4f}")
        print(f"Sample-to-Feature Ratio: {case9_results[best_count]['Sample_to_Feature_Ratio']:.2f}")
        
        # Check if there's a warning about sample size
        if any(case9_results[n]['Sample_to_Feature_Ratio'] < 5 for n in case9_results):
            print("\nWarning: Some feature counts resulted in small sample-to-feature ratios (<5).")
            print("This may lead to overfitting. Consider using fewer features for more reliable results.")
        
        # Provide path to generated plots
        print("\nCheck the generated plots at: results_extension/plots/")
        print("Look for files with 'biomass_colored' in the name.")
    else:
        print("No valid results were generated. Check for errors.")
    
    return case9_results

if __name__ == "__main__":
    # Create directories before running
    create_directories()
    
    data_path = "../Data/New_data.csv"
    additional_features = [50, 100, 200, 400, 800]  # Can modify to try different numbers of features
    all_results = {}
    sample_counts = {}  # Dictionary to store sample counts for each case
    sample_infos = {}   # Store all sample info by case
    
    # Define all cases to run
    cases = [
        ("case4", "Using all 8 features plus ≤50 additional features and all ACE-km values"),
        ("case5", "Using acetoclastic feature plus ≤50 additional features and ACE-km values ≤10"),
        ("case6", "Using acetoclastic feature plus ≤50 additional features for Biomass F"),
        ("case7", "Using all 8 features plus ≤50 additional features and ACE-km values >10"),
        ("case8", "Using all 8 features plus ≤50 additional features for Biomass G"),
        ("case9", "Using acetoclastic feature plus ≤50 additional features and all ACE-km values (no filtering)")
    ]
    
    # Run all cases with specified number of features
    for mode, description in cases:
        print(f"\n{'=' * 80}")
        print(f"Running {mode}: {description}")
        print(f"{'=' * 80}")
        
        for n_features in additional_features:
            print(f"\nUsing {n_features} additional features")
            
            # Run the model
            r2, mse, rmse, results, sample_info = run_linearsvr_cv(
                data_path=data_path,
                target="ACE-km",
                mode=mode,
                n_features=n_features
            )
            
            # Store results if model ran successfully
            if r2 is not None:
                # Store the sample information
                key = f'ACE_km_{mode}_{n_features}'
                sample_counts[key] = sample_info.get('final', 0)
                sample_infos[key] = sample_info
                
                all_results[key] = {
                    'R2': r2,
                    'MSE': mse,
                    'RMSE': rmse,
                    'Additional_Features': n_features,
                    'Sample_Count': sample_info.get('final', 0),
                    'Sample_to_Feature_Ratio': sample_info.get('sample_to_feature_ratio', 0)
                }
    
    # Save results to CSV with enhanced sample info
    if all_results:
        results_df = pd.DataFrame(all_results).T
        results_df.index = results_df.index.astype(str)  # Convert index to string type
        results_df.index.name = 'Model_Configuration'
        results_df.to_csv('results_extension/metrics/extension_comparison_results.csv')
        
        # Print summary of results with sample counts
        print("\nSummary of results:")
        
        # Create a formatted summary table for all feature counts
        summary_table = []
        for mode, description in cases:
            # First create entries with just 50 features for quick reference
            result_key = f'ACE_km_{mode}_50'
            if result_key in all_results:
                result = all_results[result_key]
                sample_info = sample_infos.get(result_key, {})
                
                summary_entry = {
                    'Case': mode,
                    'Description': description,
                    'Features': "50",
                    'Sample Count': sample_info.get('final', 0),
                    'Sample/Feature': f"{sample_info.get('sample_to_feature_ratio', 0):.2f}",
                    'R²': f"{result['R2']:.4f}",
                    'MSE': f"{result['MSE']:.4f}",
                    'RMSE': f"{result['RMSE']:.4f}"
                }
                
                # Add warning flag for small sample sizes
                if sample_info.get('sample_to_feature_ratio', 0) < 5:
                    summary_entry['Warning'] = "Small sample size"
                else:
                    summary_entry['Warning'] = ""
                    
                summary_table.append(summary_entry)
        
        # Create and print the formatted table
        if summary_table:
            # Compute column widths
            col_widths = {col: max(len(str(row[col])) for row in summary_table) for col in summary_table[0].keys()}
            col_widths = {col: max(col_widths[col], len(col)) for col in col_widths}
            
            # Print header
            header = ' | '.join(f"{col:{col_widths[col]}}" for col in summary_table[0].keys())
            print('\n' + header)
            print('-' * len(header))
            
            # Print rows
            for row in summary_table:
                print(' | '.join(f"{str(row[col]):{col_widths[col]}}" for col in row.keys()))
        
        # Save the detailed summary table to CSV for future reference
        pd.DataFrame(summary_table).to_csv('results_extension/metrics/case_summary_with_samples.csv', index=False)
        
        # Create a full detailed table with all feature counts
        full_summary_table = []
        for mode, description in cases:
            for n_features in additional_features:
                result_key = f'ACE_km_{mode}_{n_features}'
                if result_key in all_results:
                    result = all_results[result_key]
                    sample_info = sample_infos.get(result_key, {})
                    
                    full_summary_entry = {
                        'Case': mode,
                        'Description': description,
                        'Additional Features': n_features,
                        'Sample Count': sample_info.get('final', 0),
                        'Sample/Feature Ratio': f"{sample_info.get('sample_to_feature_ratio', 0):.2f}",
                        'R²': f"{result['R2']:.4f}",
                        'MSE': f"{result['MSE']:.4f}",
                        'RMSE': f"{result['RMSE']:.4f}"
                    }
                    
                    # Add warning flag for small sample sizes
                    if sample_info.get('sample_to_feature_ratio', 0) < 5:
                        full_summary_entry['Warning'] = "Small sample size"
                    else:
                        full_summary_entry['Warning'] = ""
                        
                    full_summary_table.append(full_summary_entry)
        
        # Save the full detailed table to CSV
        pd.DataFrame(full_summary_table).to_csv('results_extension/metrics/full_case_summary_with_samples.csv', index=False)
        print("\nFull detailed summary table saved to: results_extension/metrics/full_case_summary_with_samples.csv")
        
        # Print feature count comparison for each case
        print("\nPerformance comparison across different feature counts:")
        for mode, description in cases:
            print(f"\n{mode}: {description}")
            features_data = []
            
            for n_features in additional_features:
                result_key = f'ACE_km_{mode}_{n_features}'
                if result_key in all_results:
                    result = all_results[result_key]
                    features_data.append({
                        'Features': n_features,
                        'R²': f"{result['R2']:.4f}",
                        'MSE': f"{result['MSE']:.4f}",
                        'RMSE': f"{result['RMSE']:.4f}",
                        'Sample/Feature': f"{result.get('Sample_to_Feature_Ratio', 0):.2f}"
                    })
            
            if features_data:
                # Print as a small table
                cols = list(features_data[0].keys())
                widths = {col: max(len(str(row[col])) for row in features_data) for col in cols}
                widths = {col: max(widths[col], len(col)) for col in widths}
                
                # Header
                feat_header = ' | '.join(f"{col:{widths[col]}}" for col in cols)
                print(feat_header)
                print('-' * len(feat_header))
                
                # Rows
                for row in features_data:
                    print(' | '.join(f"{str(row[col]):{widths[col]}}" for col in cols))
        
        print("\nDetailed summary tables saved to: results_extension/metrics/")
        
        # After running all cases, create combined plots
        # For example, combine case5 (ACE-km ≤ 10) and case7 (ACE-km > 10)
        if all_results:
            # Organize results by case for combined plotting
            case_results = {}
            for mode, description in cases:
                n_features = 50  # Use the results with 50 additional features
                result_key = f'ACE_km_{mode}_{n_features}'
                
                if result_key in all_results:
                    # Get the saved result files
                    results_file = f'results_extension/metrics/results_ACE-km_{mode}_{n_features}features.csv'
                    if os.path.exists(results_file):
                        results_df = pd.read_csv(results_file)
                        # Rename columns to standard names if needed
                        if 'Actual' not in results_df.columns and 'actual' in results_df.columns:
                            results_df = results_df.rename(columns={'actual': 'Actual'})
                        if 'Predicted' not in results_df.columns and 'predicted' in results_df.columns:
                            results_df = results_df.rename(columns={'predicted': 'Predicted'})
                        
                        case_results[mode] = results_df
            
            # Create combined plot for case5 and case7 (low and high ACE-km values)
            if 'case5' in case_results and 'case7' in case_results:
                print("\nCreating combined plot for case5 (ACE-km ≤ 10) and case7 (ACE-km > 10)")
                print(f"Sample counts: case5={len(case_results['case5'])} samples, case7={len(case_results['case7'])} samples")
                combined_r2, combined_mse = create_combined_plot(
                    case_results, 
                    ['case5', 'case7']
                )
                
                print("\nCombined plot metrics:")
                for case, r2 in combined_r2.items():
                    sample_to_feature = sample_infos.get(f'ACE_km_{case}_50', {}).get('sample_to_feature_ratio', 0)
                    print(f"{case}: R² = {r2:.4f}, MSE = {combined_mse[case]:.4f}, " +
                          f"Samples = {len(case_results[case])}, Sample/Feature = {sample_to_feature:.2f}")
            
            # Create combined plot for case6 and case8 (Biomass F and G)
            if 'case6' in case_results and 'case8' in case_results:
                print("\nCreating combined plot for case6 (Biomass F) and case8 (Biomass G)")
                print(f"Sample counts: case6={len(case_results['case6'])} samples, case8={len(case_results['case8'])} samples")
                combined_r2, combined_mse = create_combined_plot(
                    case_results, 
                    ['case6', 'case8']
                )
                
                print("\nCombined plot metrics:")
                for case, r2 in combined_r2.items():
                    sample_to_feature = sample_infos.get(f'ACE_km_{case}_50', {}).get('sample_to_feature_ratio', 0)
                    print(f"{case}: R² = {r2:.4f}, MSE = {combined_mse[case]:.4f}, " +
                          f"Samples = {len(case_results[case])}, Sample/Feature = {sample_to_feature:.2f}")
                    
            # Create combined plot for case5 and case9 (filtered vs. unfiltered with 1 feature)
            if 'case5' in case_results and 'case9' in case_results:
                print("\nCreating combined plot for case5 (ACE-km ≤ 10) and case9 (All values, 1 feature)")
                print(f"Sample counts: case5={len(case_results['case5'])} samples, case9={len(case_results['case9'])} samples")
                combined_r2, combined_mse = create_combined_plot(
                    case_results, 
                    ['case5', 'case9']
                )
                
                print("\nCombined plot metrics:")
                for case, r2 in combined_r2.items():
                    sample_to_feature = sample_infos.get(f'ACE_km_{case}_50', {}).get('sample_to_feature_ratio', 0)
                    print(f"{case}: R² = {r2:.4f}, MSE = {combined_mse[case]:.4f}, " +
                          f"Samples = {len(case_results[case])}, Sample/Feature = {sample_to_feature:.2f}")
    else:
        print("\nNo results to summarize. Check for errors in the cases.")
    
    # Run Case 9 with biomass coloring if requested
    run_case9 = True  # Set to True to run Case 9 with biomass coloring
    if run_case9:
        run_case9_with_biomass_coloring(data_path="../Data/New_data.csv", feature_counts=[50, 100, 200]) 