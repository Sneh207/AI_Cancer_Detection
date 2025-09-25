import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
import os

class ModelEvaluator:
    """
    Comprehensive model evaluation for cancer detection
    """
    def __init__(self, model, test_loader, device, save_path=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.save_path = save_path or "experiments/results"
        
        # Create save directory
        os.makedirs(self.save_path, exist_ok=True)
        
    def predict(self):
        """
        Generate predictions on test set
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Predicting"):
                data = data.to(self.device)
                output = self.model(data).squeeze()
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(output)
                
                predictions.extend(probs.cpu().numpy())
                targets.extend(target.numpy())
        
        return np.array(predictions), np.array(targets)
    
    def calculate_metrics(self, y_true, y_probs, threshold=0.5):
        """
        Calculate comprehensive evaluation metrics
        """
        y_pred = (y_probs > threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # Same as recall
        }
        
        # AUC metrics (only if we have both classes)
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
            metrics['pr_auc'] = average_precision_score(y_true, y_probs)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """
        Calculate specificity (True Negative Rate)
        """
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[0] < 2:
            return 0.0
        
        tn = cm[0, 0] if cm.shape == (2, 2) else 0
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def find_optimal_threshold(self, y_true, y_probs):
        """
        Find optimal threshold using Youden's J statistic
        """
        if len(np.unique(y_true)) <= 1:
            return 0.5
        
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold
    
    def plot_confusion_matrix(self, y_true, y_pred, save_name="confusion_matrix.png"):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Cancer', 'Cancer'],
                   yticklabels=['No Cancer', 'Cancer'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_probs, save_name="roc_curve.png"):
        """
        Plot ROC curve
        """
        if len(np.unique(y_true)) <= 1:
            print("Cannot plot ROC curve: only one class present")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_probs, save_name="pr_curve.png"):
        """
        Plot Precision-Recall curve
        """
        if len(np.unique(y_true)) <= 1:
            print("Cannot plot PR curve: only one class present")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {ap:.3f})')
        plt.axhline(y=np.mean(y_true), color='red', linestyle='--',
                   label=f'Baseline (AP = {np.mean(y_true):.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(self, y_true, y_probs, save_name="prediction_dist.png"):
        """
        Plot distribution of prediction probabilities
        """
        plt.figure(figsize=(10, 6))
        
        # Separate predictions by true class
        cancer_probs = y_probs[y_true == 1]
        no_cancer_probs = y_probs[y_true == 0]
        
        plt.hist(no_cancer_probs, bins=50, alpha=0.7, label='No Cancer', color='blue', density=True)
        plt.hist(cancer_probs, bins=50, alpha=0.7, label='Cancer', color='red', density=True)
        
        plt.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Prediction Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, threshold=None):
        """
        Comprehensive model evaluation
        """
        print("Evaluating model on test set...")
        
        # Generate predictions
        y_probs, y_true = self.predict()
        
        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self.find_optimal_threshold(y_true, y_probs)
            print(f"Optimal threshold found: {threshold:.3f}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_probs, threshold)
        
        # Print results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Threshold: {threshold:.3f}")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"F1-Score:    {metrics['f1']:.4f}")
        print(f"ROC AUC:     {metrics['roc_auc']:.4f}")
        print(f"PR AUC:      {metrics['pr_auc']:.4f}")
        print("="*60)
        
        # Generate predictions with optimal threshold
        y_pred = (y_probs > threshold).astype(int)
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['No Cancer', 'Cancer']))
        
        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_probs)
        self.plot_precision_recall_curve(y_true, y_probs)
        self.plot_prediction_distribution(y_true, y_probs)
        
        # Save results
        results = {
            'metrics': metrics,
            'threshold': threshold,
            'predictions': y_probs.tolist(),
            'targets': y_true.tolist(),
            'predicted_labels': y_pred.tolist()
        }
        
        import json
        with open(os.path.join(self.save_path, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {self.save_path}")
        
        return metrics, threshold
    
    def evaluate_at_different_thresholds(self, thresholds=None):
        """
        Evaluate model performance at different thresholds
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        y_probs, y_true = self.predict()
        
        results = []
        for threshold in thresholds:
            metrics = self.calculate_metrics(y_true, y_probs, threshold)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        # Plot metrics vs threshold
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        
        for idx, metric in enumerate(metrics_to_plot):
            row, col = idx // 2, idx % 2
            values = [r[metric] for r in results]
            axes[row, col].plot(thresholds, values, marker='o')
            axes[row, col].set_title(f'{metric.capitalize()} vs Threshold')
            axes[row, col].set_xlabel('Threshold')
            axes[row, col].set_ylabel(metric.capitalize())
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, 'threshold_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return results