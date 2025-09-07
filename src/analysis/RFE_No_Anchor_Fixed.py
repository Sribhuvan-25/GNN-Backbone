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
    main_dir = 'results_rfe_nested_cv_fixed'
    os.makedirs(main_dir, exist_ok=True)
    
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
    filename = f'results_rfe_nested_cv_fixed/{case_type}/plots/results_{target}_{feature_suffix}_features_{model_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def select_features_with_rfe(X, y, n_features=None, model_type='extratrees'):
    """Perform RFE feature selection with detailed debugging for all model types"""
    if n_features is None:
        return X.columns.tolist()
    
    print(f"        RFE Debug: Starting with {len(X.columns)} features, selecting {n_features}")
    start_time = time.time()
        
    # Create estimator based on model type
    if model_type == 'linearsvr':
        estimator = LinearSVR(random_state=42, max_iter=100000, tol=1e-4, dual=True)
    elif model_type == 'extratrees':
        estimator = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_type == 'randomforest':
        estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
    elif model_type == 'gradientboosting':
        estimator = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=6)
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        estimator = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
    elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        estimator = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=-1)
    else:
        # Default fallback to ExtraTreesRegressor
        estimator = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        print(f"        RFE Debug: Unknown model_type '{model_type}', using ExtraTreesRegressor as fallback")
    
    # Create RFE object
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    
    # Fit RFE
    print(f"        RFE Debug: Fitting RFE with {type(estimator).__name__}...")
    rfe.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[rfe.support_].tolist()
    
    # Debug output
    print(f"        RFE Debug: Selected {len(selected_features)} features in {time.time() - start_time:.2f}s")
    print(f"        RFE Debug: First 5 selected features: {selected_features[:5]}")
    
    # Ensure we got the right number
    if len(selected_features) != n_features:
        print(f"        RFE ERROR: Expected {n_features} features but got {len(selected_features)}")
        # This shouldn't happen, but let's handle it
        if len(selected_features) == 0:
            print(f"        RFE ERROR: No features selected! Falling back to all features.")
            return X.columns.tolist()
    
    return selected_features

def _train_and_evaluate_once(X, y, train_idx, val_idx, n_features, model_type):
    """Train model on train_idx and evaluate on val_idx, returning MSE and R² scores"""
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Perform RFE feature selection (or use all features if n_features is None)
    if n_features is None:
        selected_features = X_train.columns.tolist()
        print(f"      Using all {len(selected_features)} features")
    else:
        print(f"      Performing RFE to select {n_features} features from {len(X_train.columns)}...")
        selected_features = select_features_with_rfe(X_train, y_train, n_features, model_type)
        print(f"      RFE selected {len(selected_features)} features")
        
        # Validate that RFE actually selected the right number of features
        if len(selected_features) != n_features:
            print(f"      WARNING: RFE selected {len(selected_features)} features but expected {n_features}")
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    
    # Apply scaling and create model based on model type
    if model_type == 'linearsvr':
        # LinearSVR needs scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        model = LinearSVR(random_state=42, max_iter=100000, tol=1e-4, dual=True)
    elif model_type == 'extratrees':
        # Tree-based models don't need scaling
        X_train_scaled, X_val_scaled = X_train_selected, X_val_selected
        model = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_type == 'randomforest':
        # Tree-based models don't need scaling
        X_train_scaled, X_val_scaled = X_train_selected, X_val_selected
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
    elif model_type == 'gradientboosting':
        # Gradient boosting benefits from scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        model = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=6)
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        # XGBoost benefits from scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
    elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        # LightGBM benefits from scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=-1)
    else:
        # Default fallback to ExtraTreesRegressor (no scaling needed)
        X_train_scaled, X_val_scaled = X_train_selected, X_val_selected
        model = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        print(f"      WARNING: Unknown model_type '{model_type}', using ExtraTreesRegressor as fallback")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_val_scaled)
    
    # Calculate both MSE (for hyperparameter selection) and R² (for reporting)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    return mse, r2, selected_features

def _inner_loop_select(X, y, model_type):
    """Inner loop for hyperparameter selection using K-fold CV with MSE"""
    # FIXED: Use proper 5-fold inner CV for reliable hyperparameter selection
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = float('inf')  # FIXED: Best MSE is lowest, so start with infinity
    best_n_features = None
    all_combinations_results = []
    
    # Define hyperparameter search space
    # n_features_options = [None, 50, 75, 100]
    n_features_options = [50]
    
    print(f"    Inner CV: Testing {len(n_features_options)} feature count combinations...")
    print(f"    {'Combination':<15} {'N_Features':<12} {'Mean MSE':<10} {'Std MSE':<10} {'Mean R²':<10} {'Best':<6}")
    print(f"    {'-'*75}")
    
    for i, n_features in enumerate(n_features_options):
        val_mse_scores = []
        val_r2_scores = []
        
        for tr_idx, val_idx in inner_kf.split(X):
            mse_val, r2_val, _ = _train_and_evaluate_once(X, y, tr_idx, val_idx, n_features, model_type)
            val_mse_scores.append(mse_val)
            val_r2_scores.append(r2_val)
        
        mean_mse = np.mean(val_mse_scores)
        std_mse = np.std(val_mse_scores)
        mean_r2 = np.mean(val_r2_scores)
        
        # Store results for this combination
        feature_display = "All" if n_features is None else str(n_features)
        combination_result = {
            'combination': i + 1,
            'n_features': n_features,
            'mean_mse': mean_mse,
            'std_mse': std_mse,
            'mean_r2': mean_r2,
            'val_mse_scores': val_mse_scores,
            'val_r2_scores': val_r2_scores,
            'is_best': False
        }
        all_combinations_results.append(combination_result)
        
        # FIXED: Use MSE for hyperparameter selection (lower is better)
        if mean_mse < best_score:
            best_score = mean_mse
            best_n_features = n_features
            # Mark this as best
            combination_result['is_best'] = True
        
        # Print progress for this combination
        best_marker = "✓" if combination_result['is_best'] else " "
        print(f"    {i+1:<15} {feature_display:<12} {mean_mse:<10.4f} {std_mse:<10.4f} {mean_r2:<10.4f} {best_marker:<6}")
    
    print(f"    {'-'*75}")
    feature_display = "All" if best_n_features is None else str(best_n_features)
    print(f"    Best hyperparameters: n_features={feature_display} (MSE = {best_score:.4f})")
    
    return best_n_features

def run_model_nested_cv(data_path, target="ACE-km", model_type='extratrees', case_type='case3'):
    """
    FIXED: Run model with proper nested CV hyperparameter tuning for feature selection with domain expert cases
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file
    target : str
        Target variable to predict (must exist in the data)
    model_type : str
        Type of model to use ('extratrees' or 'linearsvr')
    case_type : str
        Domain expert case type ('case1', 'case2', 'case3')
    """
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
    
    # Apply case-specific feature filtering
    if case_type in CASE_FEATURES:
        case_features = CASE_FEATURES[case_type]
        # Find intersection between case features and available features
        available_case_features = [f for f in case_features if f in df.columns]
        
        if not available_case_features:
            print(f"ERROR: No case features found in dataset for {case_type}")
            print(f"Required features: {case_features[:3]}...")  # Show first 3
            print(f"Available features: {list(df.columns)[:10]}...")  # Show first 10
            return None
        
        print(f"\n{case_type.upper()} FEATURE FILTERING:")
        print(f"Case features required: {len(case_features)}")
        print(f"Case features available: {len(available_case_features)}")
        print(f"Available case features: {available_case_features}")
        
        # Use only the case-specific features
        X = df[available_case_features]
        filtered_features = available_case_features
    else:
        # Use all features if case_type is not recognized
        X = df[all_features]
        filtered_features = all_features
    
    y = df[target]
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Total features after case filtering: {len(filtered_features)}")
    print(f"Target: {target}")
    print(f"Target range: {y.min():.4f} to {y.max():.4f}")
    print(f"Model type: {model_type}")
    print(f"Case type: {case_type}")
    
    # FIXED: Use proper 5-fold outer CV for reliable performance estimation
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_results = []
    best_outer_r2 = -float('inf')
    best_n_features = None
    
    # FIXED: Store actual CV splits for proper prediction collection
    cv_splits = list(outer_kf.split(X))
    
    print(f"\n{'='*60}")
    print(f"NESTED CV RESULTS FOR {model_type.upper()} - {target}")
    print(f"{'='*60}")
    
    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        fold_num = fold + 1
        print(f"\n  {'-'*50}")
        print(f"  OUTER FOLD {fold_num}/5")
        print(f"  {'-'*50}")
        print(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. Inner loop: pick hyperparameters
        print(f"  Inner CV Hyperparameter Selection:")
        best_n_features_inner = _inner_loop_select(X_train, y_train, model_type)
        
        # 2. Retrain on full train_data with best_params
        print(f"  Retraining with best n_features={best_n_features_inner}")
        
        # Perform RFE feature selection with best n_features (or use all features if None)
        if best_n_features_inner is None:
            selected_features = X_train.columns.tolist()
            print(f"  Using all {len(selected_features)} features...")
        else:
            selected_features = select_features_with_rfe(X_train, y_train, best_n_features_inner, model_type)
        
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
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
            model = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        elif model_type == 'randomforest':
            # Tree-based models don't need scaling
            X_train_scaled, X_test_scaled = X_train_selected, X_test_selected
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
        elif model_type == 'gradientboosting':
            # Gradient boosting benefits from scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            model = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=6)
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost benefits from scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # LightGBM benefits from scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=-1)
        else:
            # Default fallback to ExtraTreesRegressor (no scaling needed)
            X_train_scaled, X_test_scaled = X_train_selected, X_test_selected
            model = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            print(f"  WARNING: Unknown model_type '{model_type}', using ExtraTreesRegressor as fallback")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # 3. Evaluate on test_data
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  Final Evaluation on Test Set:")
        print(f"    R² = {r2:.4f}, MSE = {mse:.4f}")
        
        outer_results.append({
            'fold': fold_num,
            'r2': r2,
            'mse': mse,
            'best_n_features': best_n_features_inner,
            'selected_features': selected_features,
            'train_idx': train_idx,  # FIXED: Store actual indices
            'test_idx': test_idx,    # FIXED: Store actual indices
            'y_pred': y_pred,        # FIXED: Store actual predictions
            'y_test': y_test.values  # FIXED: Store actual test values
        })
        
        if r2 > best_outer_r2:
            best_outer_r2 = r2
            best_n_features = best_n_features_inner
    
    # 4. Comprehensive Summary
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE NESTED CV SUMMARY")
    print(f"{'='*60}")
    
    r2_scores = [r['r2'] for r in outer_results]
    mse_scores = [r['mse'] for r in outer_results]
    
    avg_r2 = np.mean(r2_scores)
    avg_mse = np.mean(mse_scores)
    std_r2 = np.std(r2_scores)
    std_mse = np.std(mse_scores)
    
    print(f"Model: {model_type.upper()}")
    print(f"Target: {target}")
    print(f"Number of outer folds: 5")
    
    print(f"\nFold-by-Fold Results:")
    print(f"{'Fold':<6} {'R²':<10} {'MSE':<10} {'Best N_Features':<15}")
    print(f"{'-'*45}")
    
    for result in outer_results:
        fold = result['fold']
        r2 = result['r2']
        mse = result['mse']
        best_n_features = result['best_n_features']
        feature_display = "All" if best_n_features is None else str(best_n_features)
        print(f"{fold:<6} {r2:<10.4f} {mse:<10.4f} {feature_display:<15}")
    
    print(f"\nOverall Performance:")
    print(f"  R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"  MSE = {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"  RMSE = {np.sqrt(avg_mse):.4f}")
    
    # Show hyperparameter selection frequency
    print(f"\nHyperparameter Selection Frequency:")
    n_features_list = [r['best_n_features'] for r in outer_results]
    
    n_features_counts = Counter(n_features_list)
    
    print(f"  n_features selections:")
    for n_features, count in n_features_counts.items():
        feature_display = "All" if n_features is None else str(n_features)
        print(f"    {feature_display}: {count}/5 folds")
    
    # FIXED: Collect predictions properly using stored values
    print(f"\nCollecting predictions for final plot...")
    all_predictions = []
    all_actual = []
    
    for result in outer_results:
        all_predictions.extend(result['y_pred'])
        all_actual.extend(result['y_test'])
    
    # Create final plot
    # Use the most frequently selected n_features for plot title
    most_common_n_features = n_features_counts.most_common(1)[0][0] if n_features_counts else None
    create_performance_plot(all_actual, all_predictions, target, most_common_n_features, avg_r2, avg_mse, model_type, case_type)
    
    # Save results
    print(f"Saving results...")
    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    results_path = f'results_rfe_nested_cv_fixed/{case_type}/metrics/results_{target}_{model_type}_nested_cv.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Saved results: {results_path}")
    
    # Save selected features summary
    all_selected_features = []
    for result in outer_results:
        all_selected_features.extend(result['selected_features'])
    
    # Get unique features and their frequency
    feature_counts = Counter(all_selected_features)
    feature_df = pd.DataFrame([
        {'Feature': feature, 'Frequency': count} 
        for feature, count in feature_counts.items()
    ]).sort_values('Frequency', ascending=False)
    
    features_path = f'results_rfe_nested_cv_fixed/{case_type}/selected_features/{target}_{model_type}_feature_frequency.csv'
    feature_df.to_csv(features_path, index=False)
    print(f"Saved feature frequency: {features_path}")
    
    # Save hyperparameter selection summary
    save_hyperparameter_summary(outer_results, target, model_type, case_type)
    
    return {
        'avg_r2': avg_r2,
        'avg_mse': avg_mse,
        'std_r2': std_r2,
        'std_mse': std_mse,
        'best_n_features': best_n_features,
        'fold_results': outer_results,
        'feature_frequency': feature_df
    }

def save_hyperparameter_summary(outer_results, target, model_type, case_type='case3'):
    """Save hyperparameter selection summary for RFE nested CV"""
    print(f"Saving hyperparameter summary...")
    # Collect hyperparameter selections
    n_features_selections = [result['best_n_features'] for result in outer_results]
    
    # Create summary
    value_counts = Counter(n_features_selections)
    total_folds = len(n_features_selections)
    
    summary_data = []
    for value, count in value_counts.items():
        summary_data.append({
            'parameter': 'n_features',
            'value': value,
            'frequency': count,
            'percentage': f"{(count/total_folds)*100:.1f}%",
            'total_folds': total_folds
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = f'results_rfe_nested_cv_fixed/{case_type}/metrics/{target}_{model_type}_hyperparameter_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    
    print(f"Hyperparameter selection summary saved: {csv_path}")
    
    return csv_path

if __name__ == "__main__":
    # Create directories first
    create_directories()
    
    data_path = "../Data/New_Data.csv"
    
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
    
    print(f"Available model types ({len(model_types)}): {model_types}")
    print(f"Total experiments to run: {len(cases)} cases × {len(main_targets)} targets × {len(model_types)} models = {len(cases) * len(main_targets) * len(model_types)}")
    
    # Dictionary to store results
    all_results = {}
    
    # Run nested CV for all combinations of cases, targets, and model types
    for case_type in cases:
        for target in main_targets:
            for model_type in model_types:
                print(f"\n{'='*80}")
                print(f"Running FIXED NESTED CV for {case_type.upper()} - {target} with {model_type.upper()}")
                print(f"{'='*80}")
                
                try:
                    results = run_model_nested_cv(
                        data_path, 
                        target=target, 
                        model_type=model_type,
                        case_type=case_type
                    )
                    
                    if results is not None:
                        config_name = f"{case_type}_{target}_{model_type}"
                        all_results[config_name] = {
                            'Case': case_type,
                            'Target': target,
                            'Model': model_type,
                            'R2': results['avg_r2'],
                            'MSE': results['avg_mse'],
                            'Std_R2': results['std_r2'],
                            'Std_MSE': results['std_mse'],
                            'Best_N_Features': results['best_n_features']
                        }
                        
                        print(f"Completed {case_type} - {target} with {model_type}")
                    else:
                        print(f"Skipping {case_type} - {target} with {model_type} due to errors")
                    
                except Exception as e:
                    print(f"Error running {case_type} - {target} with {model_type}: {str(e)}")
                    continue
    
    # Save overall results
    if all_results:
        print(f"\nSaving overall results...")
        results_df = pd.DataFrame(all_results).T
        results_df.index.name = 'Configuration'
        overall_path = 'results_rfe_nested_cv_fixed/metrics/overall_results_nested_cv.csv'
        results_df.to_csv(overall_path)
        print(f"Saved overall results: {overall_path}")
        
        # Print final results
        print(f"\n{'='*80}")
        print("FINAL FIXED NESTED CV RESULTS")
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
                        print(f"    Best R²: {best_config['R2']:.4f} ± {best_config['Std_R2']:.4f}")
                        print(f"    MSE: {best_config['MSE']:.4f} ± {best_config['Std_MSE']:.4f}")
                        print(f"    Best N_Features: {best_config['Best_N_Features']}")
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
                print(f"Best R²: {best_config['R2']:.4f} ± {best_config['Std_R2']:.4f}")
                print(f"MSE: {best_config['MSE']:.4f} ± {best_config['Std_MSE']:.4f}")
                print(f"Best N_Features: {best_config['Best_N_Features']}")
    
    print(f"\nAll results saved successfully!")