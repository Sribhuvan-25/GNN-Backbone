#!/usr/bin/env python3
"""
Recursive Feature Elimination (RFE) Feature Selector

This module implements RFE for feature importance calculation to identify
the most relevant genus features for predictive tasks using sklearn models.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor

# Import XGBoost and LightGBM with availability checking
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class RFEFeatureSelector:
    """
    Feature selector using Recursive Feature Elimination (RFE).
    
    Uses sklearn's RFE with various estimator types to select
    top N features based on model importance.
    """
    
    def __init__(self, model_type: str = 'extratrees', random_state: int = 42):
        """
        Initialize RFE feature selector.
        
        Args:
            model_type: Type of estimator to use ('extratrees', 'linearsvr', 
                       'randomforest', 'gradientboosting', 'xgboost', 'lightgbm')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        
        # Validate model type
        valid_types = ['extratrees', 'linearsvr', 'randomforest', 'gradientboosting']
        if XGBOOST_AVAILABLE:
            valid_types.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            valid_types.append('lightgbm')
            
        if model_type not in valid_types:
            print(f"Warning: Unknown model_type '{model_type}', using 'extratrees' as fallback")
            self.model_type = 'extratrees'
    
    def _create_estimator(self, n_estimators: int = 50):
        """
        Create estimator based on model type.
        
        Args:
            n_estimators: Number of estimators for tree-based models
            
        Returns:
            Estimator instance
        """
        if self.model_type == 'linearsvr':
            return LinearSVR(random_state=self.random_state, max_iter=100000, tol=1e-4, dual=True)
        elif self.model_type == 'extratrees':
            return ExtraTreesRegressor(n_estimators=n_estimators, random_state=self.random_state, n_jobs=-1)
        elif self.model_type == 'randomforest':
            return RandomForestRegressor(n_estimators=n_estimators, random_state=self.random_state, n_jobs=-1, max_depth=10)
        elif self.model_type == 'gradientboosting':
            return GradientBoostingRegressor(n_estimators=n_estimators, random_state=self.random_state, max_depth=6)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(n_estimators=n_estimators, max_depth=6, learning_rate=0.1, 
                                   random_state=self.random_state, n_jobs=-1, verbosity=0)
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=6, learning_rate=0.1, 
                                   random_state=self.random_state, n_jobs=-1, verbosity=-1)
        else:
            # Default fallback to ExtraTreesRegressor
            return ExtraTreesRegressor(n_estimators=n_estimators, random_state=self.random_state, n_jobs=-1)
    
    def select_features(self, 
                       X: np.ndarray, 
                       y: np.ndarray, 
                       n_features: int,
                       feature_names: Optional[List[str]] = None) -> Tuple[List[int], List[str]]:
        """
        Select top N features using RFE.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            n_features: Number of features to select
            feature_names: Optional list of feature names
            
        Returns:
            selected_indices: List of selected feature indices
            selected_names: List of selected feature names (or indices if names not provided)
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Validate inputs
        if len(feature_names) != X.shape[1]:
            raise ValueError(f"Feature names length ({len(feature_names)}) must match X columns ({X.shape[1]})")
        
        if n_features > X.shape[1]:
            print(f"Warning: Requested {n_features} features but only {X.shape[1]} available")
            n_features = X.shape[1]
        
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        
        # If we want all features, return all
        if n_features >= X.shape[1]:
            selected_indices = list(range(X.shape[1]))
            selected_names = feature_names
            print(f"RFE: Using all {len(selected_indices)} features (requested {n_features})")
            return selected_indices, selected_names
        
        # Convert to DataFrame for easier handling
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=feature_names)
        
        print(f"RFE: Starting with {X.shape[1]} features, selecting {n_features} using {self.model_type}")
        
        # Create estimator
        estimator = self._create_estimator()
        
        # Create RFE object
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        
        # Fit RFE
        rfe.fit(X_df, y)
        
        # Get selected features
        selected_mask = rfe.support_
        selected_indices = [i for i, selected in enumerate(selected_mask) if selected]
        selected_names = [feature_names[i] for i in selected_indices]
        
        print(f"✅ RFE: Selected {len(selected_indices)} features using {type(estimator).__name__}")
        print(f"   Top 5 selected features: {selected_names[:5]}")
        
        return selected_indices, selected_names


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing RFE Feature Selector")
    
    np.random.seed(42)
    n_samples, n_features = 100, 50
    
    # Create synthetic data with some important features
    X = np.random.randn(n_samples, n_features)
    
    # Create target that depends on first 10 features
    y = np.sum(X[:, :10], axis=1) + 0.1 * np.random.randn(n_samples)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Test selector
    selector = RFEFeatureSelector(model_type='extratrees', random_state=42)
    selected_indices, selected_names = selector.select_features(
        X, y, n_features=10, feature_names=feature_names
    )
    
    print(f"\n✅ Test completed!")
    print(f"Selected {len(selected_indices)} features")
    print(f"Top 10 selected indices: {selected_indices[:10]}")

