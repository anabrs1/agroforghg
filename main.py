import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import joblib

from src.data.data_loader import DataLoader
from src.utils.feature_engineering import FeatureEngineering
from src.models.land_suitability_classifier import LandSuitabilityClassifier
from src.models.ghg_reduction_regressor import GHGReductionRegressor
from src.evaluation.model_evaluation import ModelEvaluator
from src.utils.visualization import DataVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Agroforestry Land Suitability and GHG Reduction Prediction')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the input data file')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--suitability_target', type=str, required=True,
                        help='Column name for land suitability target')
    parser.add_argument('--ghg_target', type=str, required=True,
                        help='Column name for GHG reduction target')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data to use for validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in the random forest')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Maximum depth of trees in the random forest')
    
    return parser.parse_args()

def main():
    """Main function to run the agroforestry prediction pipeline."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_loader = DataLoader()
    feature_engineering = FeatureEngineering()
    evaluator = ModelEvaluator(output_dir=os.path.join(args.output_dir, 'evaluation'))
    visualizer = DataVisualizer(output_dir=os.path.join(args.output_dir, 'visualization'))
    
    print(f"Loading data from {args.data_path}")
    data = data_loader.load_data(args.data_path)
    
    print("Preprocessing data")
    data = data_loader.preprocess_data(data)
    
    print("Performing feature engineering")
    data = feature_engineering.engineer_spatial_features(data)
    data = feature_engineering.engineer_climate_features(data)
    data = feature_engineering.engineer_soil_features(data)
    
    print("Visualizing data distributions")
    visualizer.plot_feature_distributions(
        data, 
        save_path=os.path.join(args.output_dir, 'visualization', 'feature_distributions.png')
    )
    
    visualizer.plot_correlation_matrix(
        data,
        save_path=os.path.join(args.output_dir, 'visualization', 'correlation_matrix.png')
    )
    
    print(f"Splitting data for land suitability classification (target: {args.suitability_target})")
    suitability_data = data_loader.split_data(
        data, 
        target_column=args.suitability_target,
        test_size=args.test_size,
        validation_size=args.val_size,
        random_state=args.random_state
    )
    
    print(f"Splitting data for GHG reduction regression (target: {args.ghg_target})")
    ghg_data = data_loader.split_data(
        data, 
        target_column=args.ghg_target,
        test_size=args.test_size,
        validation_size=args.val_size,
        random_state=args.random_state
    )
    
    numeric_features = data.select_dtypes(include=['number']).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in [args.suitability_target, args.ghg_target]]
    
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
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
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
    
    print("Land Suitability Classification Metrics:")
    print(f"Accuracy: {suitability_metrics['accuracy']:.4f}")
    print(f"Precision: {suitability_metrics['weighted_precision']:.4f}")
    print(f"Recall: {suitability_metrics['weighted_recall']:.4f}")
    print(f"F1 Score: {suitability_metrics['weighted_f1']:.4f}")
    
    evaluator.plot_confusion_matrix(
        suitability_metrics['confusion_matrix'],
        title='Land Suitability Confusion Matrix',
        save_path=os.path.join(args.output_dir, 'evaluation', 'suitability_confusion_matrix.png')
    )
    
    feature_importance = suitability_classifier.get_feature_importance()
    evaluator.plot_feature_importance(
        feature_importance,
        title='Land Suitability Feature Importance',
        save_path=os.path.join(args.output_dir, 'evaluation', 'suitability_feature_importance.png')
    )
    
    suitability_model_path = os.path.join(args.output_dir, 'models', 'land_suitability_classifier.joblib')
    suitability_classifier.save_model(suitability_model_path)
    print(f"Land suitability classifier saved to {suitability_model_path}")
    
    X_train_ghg = ghg_data['X_train']
    X_train_ghg_transformed = feature_engineering.fit_transform(
        X_train_ghg, numeric_features, categorical_features
    )
    
    X_val_ghg = ghg_data['X_val']
    X_val_ghg_transformed = feature_engineering.transform(X_val_ghg)
    
    X_test_ghg = ghg_data['X_test']
    X_test_ghg_transformed = feature_engineering.transform(X_test_ghg)
    
    print("Training GHG reduction regressor")
    ghg_regressor = GHGReductionRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
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
    
    print("GHG Reduction Regression Metrics:")
    print(f"MSE: {ghg_metrics['mse']:.4f}")
    print(f"RMSE: {ghg_metrics['rmse']:.4f}")
    print(f"MAE: {ghg_metrics['mae']:.4f}")
    print(f"R²: {ghg_metrics['r2']:.4f}")
    
    evaluator.plot_actual_vs_predicted(
        ghg_metrics['actual'],
        ghg_metrics['predictions'],
        title='GHG Reduction: Actual vs Predicted',
        save_path=os.path.join(args.output_dir, 'evaluation', 'ghg_actual_vs_predicted.png')
    )
    
    evaluator.plot_residuals(
        ghg_metrics['actual'],
        ghg_metrics['predictions'],
        title='GHG Reduction Residuals',
        save_path=os.path.join(args.output_dir, 'evaluation', 'ghg_residuals.png')
    )
    
    feature_importance = ghg_regressor.get_feature_importance()
    evaluator.plot_feature_importance(
        feature_importance,
        title='GHG Reduction Feature Importance',
        save_path=os.path.join(args.output_dir, 'evaluation', 'ghg_feature_importance.png')
    )
    
    ghg_model_path = os.path.join(args.output_dir, 'models', 'ghg_reduction_regressor.joblib')
    ghg_regressor.save_model(ghg_model_path)
    print(f"GHG reduction regressor saved to {ghg_model_path}")
    
    print("Agroforestry prediction pipeline completed successfully!")

if __name__ == "__main__":
    main()
