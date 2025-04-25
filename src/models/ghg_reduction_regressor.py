import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class GHGReductionRegressor:
    """
    Random Forest regressor for estimating greenhouse gas (GHG) emission reductions
    from converting land to agroforestry.
    """
    
    def __init__(self, 
                n_estimators: int = 100, 
                max_depth: int = None,
                min_samples_split: int = 2,
                min_samples_leaf: int = 1,
                random_state: int = 42):
        """
        Initialize the GHGReductionRegressor.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.feature_importances = None
        self.feature_names = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None) -> None:
        """
        Train the regressor.
        
        Args:
            X_train: Training features
            y_train: Training target values
            feature_names: Names of the features
        """
        self.model.fit(X_train, y_train)
        self.feature_importances = self.model.feature_importances_
        self.feature_names = feature_names
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted GHG reduction values
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importances from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importances is None:
            raise ValueError("Model not trained yet. Call train first.")
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances))]
        else:
            feature_names = self.feature_names
        
        return dict(zip(feature_names, self.feature_importances))
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = joblib.load(model_path)
        self.feature_importances = self.model.feature_importances_
