import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import os

class ModelEvaluator:
    """
    Class for evaluating machine learning models for agroforestry land suitability
    and GHG emission reduction prediction.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_classifier(self, 
                           model, 
                           X_test: np.ndarray, 
                           y_test: np.ndarray,
                           class_names: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a classification model.
        
        Args:
            model: Trained classifier model
            X_test: Test features
            y_test: Test labels
            class_names: Names of the classes
            
        Returns:
            Dictionary of evaluation metrics and results
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'confusion_matrix': cm,
            'classification_report': report,
            'accuracy': report['accuracy'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
        
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
            pr_auc = auc(recall, precision)
            
            results.update({
                'roc': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
                'pr': {'precision': precision, 'recall': recall, 'auc': pr_auc}
            })
        
        return results
    
    def evaluate_regressor(self, 
                          model, 
                          X_test: np.ndarray, 
                          y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a regression model.
        
        Args:
            model: Trained regressor model
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary of evaluation metrics and results
        """
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
        
        return results
    
    def plot_confusion_matrix(self, 
                             cm: np.ndarray, 
                             class_names: List[str] = None,
                             title: str = 'Confusion Matrix',
                             save_path: str = None) -> plt.Figure:
        """
        Plot a confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Names of the classes
            title: Title of the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, 
                      fpr: np.ndarray, 
                      tpr: np.ndarray, 
                      roc_auc: float,
                      title: str = 'ROC Curve',
                      save_path: str = None) -> plt.Figure:
        """
        Plot a ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under the ROC curve
            title: Title of the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, 
                                   precision: np.ndarray, 
                                   recall: np.ndarray, 
                                   pr_auc: float,
                                   title: str = 'Precision-Recall Curve',
                                   save_path: str = None) -> plt.Figure:
        """
        Plot a precision-recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            pr_auc: Area under the precision-recall curve
            title: Title of the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower left')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_feature_importance(self, 
                               feature_importance: Dict[str, float],
                               title: str = 'Feature Importance',
                               top_n: int = 20,
                               save_path: str = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            title: Title of the plot
            top_n: Number of top features to show
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
        df = df.sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=df)
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_actual_vs_predicted(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                title: str = 'Actual vs Predicted',
                                save_path: str = None) -> plt.Figure:
        """
        Plot actual vs predicted values for regression.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Title of the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_residuals(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      title: str = 'Residual Plot',
                      save_path: str = None) -> plt.Figure:
        """
        Plot residuals for regression.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Title of the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
