"""
 * Performance metrics and visualization for the IDS model.
 * Created by Shoko on 2026/03/30
 """
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import os

class ModelEvaluator:
    """
    * Provides analysis of classifier performance.
    * @param detector Trained IntrusionDetector instance.
    * @param x_test Evaluation feature set.
    * @param y_test Evaluation labels.
    """

    def __init__(self, detector, x_test, y_test):
        self.detector = detector
        self.x_test = x_test
        self.y_test = y_test

    def generate_report(self, output_dir: str = "reports"):
        """
        * Prints classification report and saves confusion matrix plot.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        y_prob = self.detector.predict(self.x_test)
        y_pred = (y_prob > 0.5).astype(int)

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        auc = roc_auc_score(self.y_test, y_prob)
        print(f"ROC AUC Score: {auc:.4f}")

        self._plot_confusion_matrix(y_pred, output_dir)
        self._plot_importance(output_dir)

    def _plot_confusion_matrix(self, y_pred, output_dir):
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

    def _plot_importance(self, output_dir):
        importance = pd.DataFrame({
            'feature': self.x_test.columns,
            'importance': self.detector.model.feature_importance()
        }).sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importance.head(20))
        plt.title('Top 20 Features Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
