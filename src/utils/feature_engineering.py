import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeatureEngineering:
    """
    Class for feature engineering operations on agroforestry data.
    """
    
    def __init__(self):
        """
        Initialize the FeatureEngineering class.
        """
        self.preprocessor = None
        self.feature_names = None
    
    def create_preprocessor(self, 
                           numeric_features: List[str], 
                           categorical_features: List[str]) -> ColumnTransformer:
        """
        Create a preprocessor for transforming features.
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            
        Returns:
            ColumnTransformer for preprocessing features
        """
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_transform(self, 
                     X_train: pd.DataFrame, 
                     numeric_features: List[str], 
                     categorical_features: List[str]) -> np.ndarray:
        """
        Fit the preprocessor and transform the training data.
        
        Args:
            X_train: Training data
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            
        Returns:
            Transformed training data
        """
        if self.preprocessor is None:
            self.create_preprocessor(numeric_features, categorical_features)
        
        X_transformed = self.preprocessor.fit_transform(X_train)
        
        self._update_feature_names(numeric_features, categorical_features, X_train)
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        return self.preprocessor.transform(X)
    
    def _update_feature_names(self, 
                             numeric_features: List[str], 
                             categorical_features: List[str],
                             X_train: pd.DataFrame) -> None:
        """
        Update feature names after one-hot encoding.
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            X_train: Training data
        """
        ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        
        cat_feature_names = []
        for i, feature in enumerate(categorical_features):
            categories = ohe.categories_[i]
            for category in categories:
                cat_feature_names.append(f"{feature}_{category}")
        
        self.feature_names = numeric_features + cat_feature_names
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Call fit_transform first.")
        
        return self.feature_names
    
    def engineer_spatial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer spatial features for agroforestry suitability.
        
        Args:
            data: DataFrame containing spatial data
            
        Returns:
            DataFrame with additional spatial features
        """
        df = data.copy()
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['distance_from_equator'] = df['latitude'].abs()
            
            df['distance_from_prime_meridian'] = df['longitude'].abs()
        
        return df
    
    def engineer_climate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer climate-related features for agroforestry suitability.
        
        Args:
            data: DataFrame containing climate data
            
        Returns:
            DataFrame with additional climate features
        """
        df = data.copy()
        
        if 'temperature' in df.columns and 'precipitation' in df.columns:
            df['aridity_index'] = df['precipitation'] / (df['temperature'] + 10)
            
            if 'min_temperature' in df.columns and 'max_temperature' in df.columns:
                df['temperature_range'] = df['max_temperature'] - df['min_temperature']
        
        return df
    
    def engineer_soil_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer soil-related features for agroforestry suitability.
        
        Args:
            data: DataFrame containing soil data
            
        Returns:
            DataFrame with additional soil features
        """
        df = data.copy()
        
        if 'soil_ph' in df.columns:
            df['soil_ph_category'] = pd.cut(
                df['soil_ph'],
                bins=[0, 4.5, 5.5, 7.5, 8.5, 14],
                labels=['very_acidic', 'acidic', 'neutral', 'alkaline', 'very_alkaline']
            )
        
        return df
