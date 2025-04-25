"""
Example script demonstrating how to use the AgroForGHG models with sample data.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.utils.feature_engineering import FeatureEngineering
from src.models.land_suitability_classifier import LandSuitabilityClassifier
from src.models.ghg_reduction_regressor import GHGReductionRegressor
from src.evaluation.model_evaluation import ModelEvaluator
from src.utils.visualization import DataVisualizer

def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_data.csv')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'sample_run')
    
    os.makedirs(output_dir, exist_ok=True)
    
    data_loader = DataLoader()
    feature_engineering = FeatureEngineering()
    evaluator = ModelEvaluator(output_dir=os.path.join(output_dir, 'evaluation'))
    visualizer = DataVisualizer(output_dir=os.path.join(output_dir, 'visualization'))
    
    print(f"Loading sample data from {data_path}")
    data = data_loader.load_data(data_path)
    
    print("Preprocessing data")
    data = data_loader.preprocess_data(data)
    
    print("Performing feature engineering")
    data = feature_engineering.engineer_spatial_features(data)
    data = feature_engineering.engineer_climate_features(data)
    data = feature_engineering.engineer_soil_features(data)
    
    print("Visualizing data distributions")
    visualizer.plot_feature_distributions(
        data, 
        save_path=os.path.join(output_dir, 'visualization', 'feature_distributions.png')
    )
    
    visualizer.plot_correlation_matrix(
        data,
        save_path=os.path.join(output_dir, 'visualization', 'correlation_matrix.png')
    )
    
    print("Splitting data for land suitability classification")
    suitability_data = data_loader.split_data(
        data, 
        target_column='is_suitable',
        test_size=0.2,
        validation_size=0.1,
        random_state=42
    )
    
    print("Splitting data for GHG reduction regression")
    ghg_data = data_loader.split_data(
        data, 
        target_column='ghg_reduction',
        test_size=0.2,
        validation_size=0.1,
        random_state=42
    )
    
    numeric_features = data.select_dtypes(include=['number']).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in ['is_suitable', 'ghg_reduction']]
    
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print("Preprocessing features")
    X_train_suitability = suitability_data['X_train']
    X_train_suitability_transformed = feature_engineering.fit_transform(
        X_train_suitability, numeric_features, categorical_features
    )
    
    X_val_suitability = suitability_data['X_val']
    X_val_suitability_transformed = feature_engineering.transform(X_val_suitability)
    
    X_test_suitability = suitability_data['X_test']
    X_test_suitability_transformed = feature_engineering.transform(X_test_suitability)
    
    feature_names = feature_engineering.get_feature_names()
    
    print("Training land suitability classifier")
    suitability_classifier = LandSuitabilityClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    suitability_classifier.train(
        X_train_suitability_transformed, 
        suitability_data['y_train'],
        feature_names=feature_names
    )
    
    print("Evaluating land suitability classifier")
    suitability_metrics = evaluator.evaluate_classifier(
        suitability_classifier,
        X_test_suitability_transformed,
        suitability_data['y_test']
    )
    
    print("\nLand Suitability Classification Metrics:")
    print(f"Accuracy: {suitability_metrics['accuracy']:.4f}")
    print(f"Precision: {suitability_metrics['weighted_precision']:.4f}")
    print(f"Recall: {suitability_metrics['weighted_recall']:.4f}")
    print(f"F1 Score: {suitability_metrics['weighted_f1']:.4f}")
    
    evaluator.plot_confusion_matrix(
        suitability_metrics['confusion_matrix'],
        title='Land Suitability Confusion Matrix',
        save_path=os.path.join(output_dir, 'evaluation', 'suitability_confusion_matrix.png')
    )
    
    feature_importance = suitability_classifier.get_feature_importance()
    evaluator.plot_feature_importance(
        feature_importance,
        title='Land Suitability Feature Importance',
        save_path=os.path.join(output_dir, 'evaluation', 'suitability_feature_importance.png')
    )
    
    suitability_model_path = os.path.join(output_dir, 'models', 'land_suitability_classifier.joblib')
    suitability_classifier.save_model(suitability_model_path)
    print(f"\nLand suitability classifier saved to {suitability_model_path}")
    
    X_train_ghg = ghg_data['X_train']
    X_train_ghg_transformed = feature_engineering.fit_transform(
        X_train_ghg, numeric_features, categorical_features
    )
    
    X_val_ghg = ghg_data['X_val']
    X_val_ghg_transformed = feature_engineering.transform(X_val_ghg)
    
    X_test_ghg = ghg_data['X_test']
    X_test_ghg_transformed = feature_engineering.transform(X_test_ghg)
    
    print("\nTraining GHG reduction regressor")
    ghg_regressor = GHGReductionRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    ghg_regressor.train(
        X_train_ghg_transformed, 
        ghg_data['y_train'],
        feature_names=feature_names
    )
    
    print("Evaluating GHG reduction regressor")
    ghg_metrics = evaluator.evaluate_regressor(
        ghg_regressor,
        X_test_ghg_transformed,
        ghg_data['y_test']
    )
    
    print("\nGHG Reduction Regression Metrics:")
    print(f"MSE: {ghg_metrics['mse']:.4f}")
    print(f"RMSE: {ghg_metrics['rmse']:.4f}")
    print(f"MAE: {ghg_metrics['mae']:.4f}")
    print(f"R²: {ghg_metrics['r2']:.4f}")
    
    evaluator.plot_actual_vs_predicted(
        ghg_metrics['actual'],
        ghg_metrics['predictions'],
        title='GHG Reduction: Actual vs Predicted',
        save_path=os.path.join(output_dir, 'evaluation', 'ghg_actual_vs_predicted.png')
    )
    
    evaluator.plot_residuals(
        ghg_metrics['actual'],
        ghg_metrics['predictions'],
        title='GHG Reduction Residuals',
        save_path=os.path.join(output_dir, 'evaluation', 'ghg_residuals.png')
    )
    
    feature_importance = ghg_regressor.get_feature_importance()
    evaluator.plot_feature_importance(
        feature_importance,
        title='GHG Reduction Feature Importance',
        save_path=os.path.join(output_dir, 'evaluation', 'ghg_feature_importance.png')
    )
    
    ghg_model_path = os.path.join(output_dir, 'models', 'ghg_reduction_regressor.joblib')
    ghg_regressor.save_model(ghg_model_path)
    print(f"GHG reduction regressor saved to {ghg_model_path}")
    
    print(f"\nSample run completed successfully! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
