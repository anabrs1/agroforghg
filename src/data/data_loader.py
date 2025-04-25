import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import os

class DataLoader:
    """
    Class for loading and preprocessing data for agroforestry land suitability 
    and GHG emission reduction prediction.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame containing the loaded data
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif file_extension == '.json':
            return pd.read_json(file_path)
        elif file_extension == '.geojson':
            return pd.read_json(file_path)
        elif file_extension == '.shp':
            try:
                import geopandas as gpd
                return gpd.read_file(file_path)
            except ImportError:
                raise ImportError("geopandas is required to read shapefile data")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def split_data(self, 
                  data: pd.DataFrame, 
                  target_column: str,
                  test_size: float = 0.2, 
                  validation_size: float = 0.1,
                  random_state: int = 42) -> Dict[str, Any]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            data: DataFrame containing the data
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            validation_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing the split data
        """
        from sklearn.model_selection import train_test_split
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        adjusted_val_size = validation_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=adjusted_val_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for model training.
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            Preprocessed DataFrame
        """
        df = data.copy()
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
        return df
