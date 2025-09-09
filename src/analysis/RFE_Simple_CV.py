import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from collections import Counter

# Import XGBoost and LightGBM with availability checking
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Domain Expert Case Feature Definitions
CASE_FEATURES = {
    'case1': [  # Hydrogenotrophic features only
        "d__Archaea;p__Halobacterota;c__Methanomicrobia;o__Methanomicrobiales;f__Methanoregulaceae;g__Methanolinea",
        "d__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium", 
        "d__Archaea;p__Halobacterota;c__Methanomicrobia;o__Methanomicrobiales;f__Methanospirillaceae;g__Methanospirillum"
    ],
    'case2': [  # Acetoclastic features only
        "d__Archaea;p__Halobacterota;c__Methanosarcinia;o__Methanosarciniales;f__Methanosaetaceae;g__Methanosaeta"
    ],
    'case3': [  # All feature groups (acetoclastic + hydrogenotrophic + syntrophic)
        "d__Archaea;p__Halobacterota;c__Methanosarcinia;o__Methanosarciniales;f__Methanosaetaceae;g__Methanosaeta",
        "d__Archaea;p__Halobacterota;c__Methanomicrobia;o__Methanomicrobiales;f__Methanoregulaceae;g__Methanolinea",
        "d__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium",
        "d__Archaea;p__Halobacterota;c__Methanomicrobia;o__Methanomicrobiales;f__Methanospirillaceae;g__Methanospirillum",
        "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Smithellaceae;g__Smithella",
        "d__Bacteria;p__Desulfobacterota;c__Syntrophorhabdia;o__Syntrophorhabdales;f__Syntrophorhabdaceae;g__Syntrophorhabdus",
        "d__Bacteria;p__Desulfobacterota;c__Syntrophobacteria;o__Syntrophobacterales;f__Syntrophobacteraceae;g__Syntrophobacter",
        "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Syner-01",
        "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__uncultured;g__uncultured",
        "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__uncultured",
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Rikenellaceae;g__DMER64",
        "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Thermovirga",
        "d__Bacteria;p__Firmicutes;c__Syntrophomonadia;o__Syntrophomonadales;f__Syntrophomonadaceae;g__Syntrophomonas",
        "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Syntrophaceae;g__Syntrophus",
        "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__JGI-0000079-D21",
        "d__Bacteria;p__Desulfobacterota;c__Desulfuromonadia;o__Geobacterales;f__Geobacteraceae;__",
        "d__Bacteria;p__Firmicutes;c__Desulfotomaculia;o__Desulfotomaculales;f__Desulfotomaculales;g__Pelotomaculum"
    ]
}

def create_directories():
    """Create directories for organizing outputs for all cases"""
    cases = ['case1', 'case2', 'case3']
    base_directories = ['plots', 'metrics', 'selected_features']
    
    # Create main directory
    main_dir = 'results_rfe_simple_cv'
    os.makedirs(main_dir, exist_ok=True)
    
    # Create main metrics directory for comprehensive results
    main_metrics_dir = f'{main_dir}/metrics'
    os.makedirs(main_metrics_dir, exist_ok=True)
    print(f"Created directory: {main_metrics_dir}")
    
    # Create case-specific directories
    for case in cases:
        for sub_dir in base_directories:
            directory = f'{main_dir}/{case}/{sub_dir}'
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

def create_performance_plot(all_actual, all_predictions, target, n_features, final_r2, final_mse, model_type, case_type='case3'):
    """Create and save performance plot with detailed information"""
    print(f"Creating performance plot for {target} with {model_type}...")
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    sns.scatterplot(x=all_actual, y=all_predictions, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(min(all_actual), min(all_predictions))
    max_val = max(max(all_actual), max(all_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Add metrics to title
    metrics = f'R² = {final_r2:.4f}\nMSE = {final_mse:.4f}'
    feature_display = "All" if n_features is None else str(n_features)
    title = f'{target} Prediction ({model_type})\nAll Features' if n_features is None else f'{target} Prediction ({model_type})\nTop {feature_display} Features'
    plt.title(f'{title}\n{metrics}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    feature_suffix = "all" if n_features is None else str(n_features)
    filename = f'results_rfe_simple_cv/{case_type}/plots/results_{target}_{feature_suffix}_features_{model_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def select_features_with_rfe(X_candidate, y, n_features=None, model_type='extratrees', fixed_features=None):
    """Perform RFE feature selection on candidate features, optionally combining with fixed features"""
    if n_features is None:
        # If no RFE selection needed, return all candidate features + fixed features
        all_features = list(X_candidate.columns)
        if fixed_features:
            all_features.extend(fixed_features)
        return all_features
    
    # If we want more features than available candidates, use all candidates
    if n_features >= len(X_candidate.columns):
        print(f"        RFE Debug: Requested {n_features} features but only {len(X_candidate.columns)} candidates available")
        print(f"        RFE Debug: Using all {len(X_candidate.columns)} candidate features")
        selected_candidate_features = X_candidate.columns.tolist()
    else:
        print(f"        RFE Debug: Starting with {len(X_candidate.columns)} candidate features, selecting {n_features}")
        start_time = time.time()
            
        # Create estimator based on model type
        if model_type == 'linearsvr':
            estimator = LinearSVR(random_state=42, max_iter=100000, tol=1e-4, dual=True)
        elif model_type == 'extratrees':
            estimator = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        elif model_type == 'randomforest':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
        elif model_type == 'gradientboosting':
            estimator = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=6)
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            estimator = xgb.XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            estimator = lgb.LGBMRegressor(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=-1)
        else:
            # Default fallback to ExtraTreesRegressor
            estimator = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            print(f"        RFE Debug: Unknown model_type '{model_type}', using ExtraTreesRegressor as fallback")
        
        # Create RFE object
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        
        # Fit RFE
        print(f"        RFE Debug: Fitting RFE with {type(estimator).__name__}...")
        rfe.fit(X_candidate, y)
        
        # Get selected features
        selected_candidate_features = X_candidate.columns[rfe.support_].tolist()
        
        # Debug output
        print(f"        RFE Debug: Selected {len(selected_candidate_features)} candidate features in {time.time() - start_time:.2f}s")
        print(f"        RFE Debug: First 5 selected candidate features: {selected_candidate_features[:5]}")
    
    # Combine fixed features with RFE-selected features
    final_features = []
    if fixed_features:
        final_features.extend(fixed_features)
        print(f"        RFE Debug: Added {len(fixed_features)} fixed features")
    final_features.extend(selected_candidate_features)
    
    print(f"        RFE Debug: Final feature set: {len(fixed_features) if fixed_features else 0} fixed + {len(selected_candidate_features)} RFE-selected = {len(final_features)} total")
    
    return final_features

def run_model_simple_cv(data_path, target="ACE-km", model_type='extratrees', case_type='case3', n_features=50):
    """
    Run model with simple 5-fold CV for feature selection with domain expert cases
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file
    target : str
        Target variable to predict (must exist in the data)
    model_type : str
        Type of model to use ('extratrees', 'linearsvr', etc.)
    case_type : str
        Domain expert case type ('case1', 'case2', 'case3')
    n_features : int
        Number of additional features to select via RFE (default: 50)
    """
    # Create directories if they don't exist
    create_directories()
    
    print(f"Loading data from {data_path}...")
    # Load data
    df = pd.read_csv(data_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")
    
    # Check if target exists
    if target not in df.columns:
        print(f"ERROR: Target column '{target}' not found in data!")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Define target columns to exclude from features
    target_columns = ["Average-Total-ISD-Cells", "ACE-Xi", "ACE-km", "ACE-Ks", 
                     "H2-Xi", "H2-km", "H2-Ks"]
    
    # Remove rows containing 'x' or missing values
    df = df[~df.isin(['x']).any(axis=1)]
    df = df.dropna(subset=[target])
    
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[target])
    
    # Get features (all columns except target columns)
    all_features = [col for col in df.columns if col not in target_columns]
    
    # Apply case-specific feature filtering with hybrid approach
    if case_type in CASE_FEATURES:
        case_features = CASE_FEATURES[case_type]
        # Find intersection between case features and available features
        available_case_features = [f for f in case_features if f in df.columns]
        
        if not available_case_features:
            print(f"ERROR: No case features found in dataset for {case_type}")
            print(f"Required features: {case_features[:3]}...")  # Show first 3
            print(f"Available features: {list(df.columns)[:10]}...")  # Show first 10
            return None
        
        print(f"\n{case_type.upper()} HYBRID FEATURE SELECTION:")
        print(f"Case features required: {len(case_features)}")
        print(f"Case features available: {len(available_case_features)}")
        print(f"Available case features: {available_case_features}")
        
        # HYBRID APPROACH: Fixed case features + RFE on remaining features
        # Step 1: Get remaining features (excluding case features)
        remaining_features = [f for f in all_features if f not in available_case_features]
        print(f"Remaining features for RFE: {len(remaining_features)}")
        
        # Step 2: Create separate datasets for fixed and candidate features
        X_fixed = df[available_case_features]  # Always included features
        X_candidate = df[remaining_features]   # Features for RFE selection
        
        # Store the fixed features for later use
        fixed_features = available_case_features
        
        print(f"Fixed features (always included): {len(fixed_features)}")
        print(f"Candidate features (for RFE): {len(remaining_features)}")
        
    else:
        # Use all features if case_type is not recognized (no fixed features)
        X_fixed = None
        X_candidate = df[all_features]
        fixed_features = None
        print(f"\nNo case filtering - using all {len(all_features)} features for RFE")
    
    y = df[target]
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Fixed features: {len(fixed_features) if fixed_features else 0}")
    print(f"Candidate features: {len(X_candidate.columns)}")
    print(f"Target: {target}")
    print(f"Target range: {y.min():.4f} to {y.max():.4f}")
    print(f"Model type: {model_type}")
    print(f"Case type: {case_type}")
    print(f"Additional features to select: {n_features}")
    
    # Use simple 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    all_predictions = []
    all_actual = []
    all_selected_features = []
    
    print(f"\n{'='*60}")
    print(f"SIMPLE 5-FOLD CV RESULTS FOR {model_type.upper()} - {target}")
    print(f"{'='*60}")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        fold_num = fold + 1
        print(f"\n  {'-'*50}")
        print(f"  FOLD {fold_num}/5")
        print(f"  {'-'*50}")
        print(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Perform hybrid feature selection: fixed features + RFE on candidate features
        print(f"  Performing hybrid feature selection...")
        
        if n_features is None:
            # Use all features
            selected_features = list(X_candidate.columns) if 'X_candidate' in locals() else []
            if fixed_features:
                selected_features.extend(fixed_features)
            print(f"  Using all {len(selected_features)} features...")
        else:
            # Use hybrid selection: fixed + RFE on candidates
            if 'X_candidate' in locals() and len(X_candidate.columns) > 0:
                X_candidate_train = X_candidate.iloc[train_idx]
                selected_candidate_features = select_features_with_rfe(X_candidate_train, y_train, n_features, model_type, None)
            else:
                selected_candidate_features = []
            
            # Combine fixed and RFE-selected features
            selected_features = []
            if fixed_features:
                selected_features.extend(fixed_features)
            selected_features.extend(selected_candidate_features)
            print(f"  Hybrid selection: {len(fixed_features) if fixed_features else 0} fixed + {len(selected_candidate_features)} RFE-selected = {len(selected_features)} total")
        
        # Create final feature matrices
        X_train_selected = df.loc[y_train.index, selected_features]
        X_test_selected = df.loc[y_test.index, selected_features]
        
        # Apply scaling and create model based on model type
        if model_type == 'linearsvr':
            # LinearSVR needs scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            model = LinearSVR(random_state=42, max_iter=100000, tol=1e-4, dual=True)
        elif model_type == 'extratrees':
            # Tree-based models don't need scaling
            X_train_scaled, X_test_scaled = X_train_selected, X_test_selected
            model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'randomforest':
            # Tree-based models don't need scaling
            X_train_scaled, X_test_scaled = X_train_selected, X_test_selected
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        elif model_type == 'gradientboosting':
            # Gradient boosting benefits from scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost benefits from scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # LightGBM benefits from scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=-1)
        else:
            # Default fallback to ExtraTreesRegressor (no scaling needed)
            X_train_scaled, X_test_scaled = X_train_selected, X_test_selected
            model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            print(f"  WARNING: Unknown model_type '{model_type}', using ExtraTreesRegressor as fallback")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  Results:")
        print(f"    R² = {r2:.4f}, MSE = {mse:.4f}")
        
        # Store results
        cv_results.append({
            'fold': fold_num,
            'r2': r2,
            'mse': mse,
            'selected_features': selected_features,
            'y_pred': y_pred,
            'y_test': y_test.values
        })
        
        # Collect predictions for final plot
        all_predictions.extend(y_pred)
        all_actual.extend(y_test.values)
        all_selected_features.extend(selected_features)
    
    # Calculate mean ± std metrics
    r2_scores = [r['r2'] for r in cv_results]
    mse_scores = [r['mse'] for r in cv_results]
    
    avg_r2 = np.mean(r2_scores)
    avg_mse = np.mean(mse_scores)
    std_r2 = np.std(r2_scores)
    std_mse = np.std(mse_scores)
    
    # 4. Comprehensive Summary
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE 5-FOLD CV SUMMARY")
    print(f"{'='*60}")
    
    print(f"Model: {model_type.upper()}")
    print(f"Target: {target}")
    print(f"Case: {case_type}")
    print(f"Number of folds: 5")
    
    print(f"\nFold-by-Fold Results:")
    print(f"{'Fold':<6} {'R²':<10} {'MSE':<10}")
    print(f"{'-'*30}")
    
    for result in cv_results:
        fold = result['fold']
        r2 = result['r2']
        mse = result['mse']
        print(f"{fold:<6} {r2:<10.4f} {mse:<10.4f}")
    
    print(f"\nOverall Performance (Mean ± Std):")
    print(f"  R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"  MSE = {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"  RMSE = {np.sqrt(avg_mse):.4f} ± {std_mse/(2*np.sqrt(avg_mse)):.4f}")
    
    # Create final plot
    create_performance_plot(all_actual, all_predictions, target, n_features, avg_r2, avg_mse, model_type, case_type)
    
    # Save results
    print(f"\nSaving results...")
    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    results_path = f'results_rfe_simple_cv/{case_type}/metrics/results_{target}_{model_type}_simple_cv.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Saved results: {results_path}")
    
    # Save selected features summary
    feature_counts = Counter(all_selected_features)
    feature_df = pd.DataFrame([
        {'Feature': feature, 'Frequency': count} 
        for feature, count in feature_counts.items()
    ]).sort_values('Frequency', ascending=False)
    
    features_path = f'results_rfe_simple_cv/{case_type}/selected_features/{target}_{model_type}_feature_frequency.csv'
    feature_df.to_csv(features_path, index=False)
    print(f"Saved feature frequency: {features_path}")
    
    # Save metrics in mean ± std format
    save_metrics_summary(avg_r2, std_r2, avg_mse, std_mse, target, model_type, case_type)
    
    return {
        'avg_r2': avg_r2,
        'avg_mse': avg_mse,
        'std_r2': std_r2,
        'std_mse': std_mse,
        'cv_results': cv_results,
        'feature_frequency': feature_df
    }

def save_metrics_summary(avg_r2, std_r2, avg_mse, std_mse, target, model_type, case_type='case3'):
    """Save metrics summary in mean ± std format"""
    print(f"Saving metrics summary in mean ± std format...")
    
    # Calculate RMSE and its standard deviation
    rmse = np.sqrt(avg_mse)
    rmse_std = std_mse / (2 * np.sqrt(avg_mse))  # Delta method for RMSE std
    
    # Create metrics summary
    metrics_data = {
        'Metric': ['R²', 'MSE', 'RMSE'],
        'Mean': [avg_r2, avg_mse, rmse],
        'Std': [std_r2, std_mse, rmse_std],
        'Mean_Plus_Minus_Std': [
            f"{avg_r2:.4f} ± {std_r2:.4f}",
            f"{avg_mse:.4f} ± {std_mse:.4f}",
            f"{rmse:.4f} ± {rmse_std:.4f}"
        ],
        'Target': [target, target, target],
        'Model': [model_type, model_type, model_type],
        'Case': [case_type, case_type, case_type]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = f'results_rfe_simple_cv/{case_type}/metrics/{target}_{model_type}_metrics_summary.csv'
    metrics_df.to_csv(csv_path, index=False)
    
    print(f"Metrics summary saved: {csv_path}")
    
    return csv_path

def save_comprehensive_metrics_summary(all_results):
    """Save comprehensive metrics summary in mean ± std format for all configurations"""
    print(f"Saving comprehensive metrics summary in mean ± std format...")
    
    comprehensive_data = []
    
    for config_name, result in all_results.items():
        case = result['Case']
        target = result['Target']
        model = result['Model']
        r2_mean = result['R2']
        r2_std = result['Std_R2']
        mse_mean = result['MSE']
        mse_std = result['Std_MSE']
        
        # Calculate RMSE and its standard deviation
        rmse_mean = np.sqrt(mse_mean)
        rmse_std = mse_std / (2 * np.sqrt(mse_mean))  # Delta method for RMSE std
        
        comprehensive_data.append({
            'Configuration': config_name,
            'Case': case,
            'Target': target,
            'Model': model,
            'R2_Mean': r2_mean,
            'R2_Std': r2_std,
            'R2_Mean_Plus_Minus_Std': f"{r2_mean:.4f} ± {r2_std:.4f}",
            'MSE_Mean': mse_mean,
            'MSE_Std': mse_std,
            'MSE_Mean_Plus_Minus_Std': f"{mse_mean:.4f} ± {mse_std:.4f}",
            'RMSE_Mean': rmse_mean,
            'RMSE_Std': rmse_std,
            'RMSE_Mean_Plus_Minus_Std': f"{rmse_mean:.4f} ± {rmse_std:.4f}"
        })
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    csv_path = 'results_rfe_simple_cv/metrics/comprehensive_metrics_summary_mean_std.csv'
    comprehensive_df.to_csv(csv_path, index=False)
    
    print(f"Comprehensive metrics summary saved: {csv_path}")
    
    return csv_path

if __name__ == "__main__":
    # Create directories first
    create_directories()
    
    data_path = "../../Data/New_Data.csv"
    
    # Define all available model types
    model_types = ['extratrees', 'linearsvr', 'randomforest', 'gradientboosting']
    
    # Add XGBoost and LightGBM if available
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')
    
    cases = ['case1', 'case2', 'case3']
    
    # Focus on the two main targets: ACE-km and H2-km
    main_targets = ['ACE-km', 'H2-km']
    
    # Number of additional features to select via RFE
    n_features_options = [50, 100, 200]
    
    print(f"Available model types ({len(model_types)}): {model_types}")
    print(f"Available feature counts: {n_features_options}")
    print(f"Total experiments to run: {len(cases)} cases × {len(main_targets)} targets × {len(model_types)} models × {len(n_features_options)} feature counts = {len(cases) * len(main_targets) * len(model_types) * len(n_features_options)}")
    
    # Dictionary to store results
    all_results = {}
    
    # Run simple CV for all combinations
    for case_type in cases:
        for target in main_targets:
            for model_type in model_types:
                for n_features in n_features_options:
                    print(f"\n{'='*80}")
                    print(f"Running SIMPLE CV for {case_type.upper()} - {target} with {model_type.upper()} ({n_features} features)")
                    print(f"{'='*80}")
                    
                    try:
                        results = run_model_simple_cv(
                            data_path, 
                            target=target, 
                            model_type=model_type,
                            case_type=case_type,
                            n_features=n_features
                        )
                        
                        if results is not None:
                            config_name = f"{case_type}_{target}_{model_type}_{n_features}features"
                            all_results[config_name] = {
                                'Case': case_type,
                                'Target': target,
                                'Model': model_type,
                                'N_Features': n_features,
                                'R2': results['avg_r2'],
                                'MSE': results['avg_mse'],
                                'Std_R2': results['std_r2'],
                                'Std_MSE': results['std_mse']
                            }
                            
                            print(f"Completed {case_type} - {target} with {model_type} ({n_features} features)")
                        else:
                            print(f"Skipping {case_type} - {target} with {model_type} ({n_features} features) due to errors")
                        
                    except Exception as e:
                        print(f"Error running {case_type} - {target} with {model_type} ({n_features} features): {str(e)}")
                        continue
    
    # Save overall results
    if all_results:
        print(f"\nSaving overall results...")
        results_df = pd.DataFrame(all_results).T
        results_df.index.name = 'Configuration'
        overall_path = 'results_rfe_simple_cv/metrics/overall_results_simple_cv.csv'
        results_df.to_csv(overall_path)
        print(f"Saved overall results: {overall_path}")
        
        # Save comprehensive metrics summary in mean ± std format
        save_comprehensive_metrics_summary(all_results)
        
        # Print final results
        print(f"\n{'='*80}")
        print("FINAL SIMPLE CV RESULTS")
        print(f"{'='*80}")
        print(results_df)
        
        # Print best configurations for each case and target
        print(f"\n{'='*60}")
        print("BEST CONFIGURATIONS BASED ON R² SCORE")
        print(f"{'='*60}")
        for case in cases:
            print(f"\n{case.upper()}:")
            case_results = results_df[results_df['Case'] == case]
            if not case_results.empty:
                for target in main_targets:
                    target_case_results = case_results[case_results['Target'] == target]
                    if not target_case_results.empty:
                        best_config = target_case_results.loc[target_case_results['R2'].idxmax()]
                        print(f"  {target}:")
                        print(f"    Best Model: {best_config['Model']}")
                        print(f"    Best N_Features: {best_config['N_Features']}")
                        print(f"    Best R²: {best_config['R2']:.4f} ± {best_config['Std_R2']:.4f}")
                        print(f"    MSE: {best_config['MSE']:.4f} ± {best_config['Std_MSE']:.4f}")
                        print(f"    Mean ± Std Format: R² = {best_config['R2']:.4f} ± {best_config['Std_R2']:.4f}, MSE = {best_config['MSE']:.4f} ± {best_config['Std_MSE']:.4f}")
            else:
                print(f"  No results available for {case}")
                
        # Print overall best across all cases
        print(f"\n{'='*60}")
        print("OVERALL BEST CONFIGURATIONS ACROSS ALL CASES")
        print(f"{'='*60}")
        for target in main_targets:
            target_results = results_df[results_df['Target'] == target]
            if not target_results.empty:
                best_config = target_results.loc[target_results['R2'].idxmax()]
                print(f"\n{target} (Overall Best):")
                print(f"Best Case: {best_config['Case']}")
                print(f"Best Model: {best_config['Model']}")
                print(f"Best N_Features: {best_config['N_Features']}")
                print(f"Best R²: {best_config['R2']:.4f} ± {best_config['Std_R2']:.4f}")
                print(f"MSE: {best_config['MSE']:.4f} ± {best_config['Std_MSE']:.4f}")
                print(f"Mean ± Std Format: R² = {best_config['R2']:.4f} ± {best_config['Std_R2']:.4f}, MSE = {best_config['MSE']:.4f} ± {best_config['Std_MSE']:.4f}")
    
    print(f"\nAll results saved successfully!")
