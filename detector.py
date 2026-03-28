"""
 * Intrusion Detection engine using Regularized LightGBM.
 * Created by Shoko on 2026/03/30
 """
import lightgbm as lgb
import os
import joblib
from typing import Dict, Any

class IntrusionDetector:
    """
    * Core classifier using Gradient Boosting Decision Trees.
    * @param params Dictionary of training hyperparameters.
    """

    def __init__(self, params: Dict[str, Any] = None):
        self.model = None
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1
        }

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        * Trains the model on network flow data.
        """
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_sets = [train_data]
        
        if x_val is not None:
            val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )

    def predict(self, x):
        """
        * Predicts probability of malicious flow.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(x)

    def save(self, model_dir: str = "models", filename: str = "lgb_model.joblib"):
        """
        * Serializes the model object to disk.
        """
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, filename))

    def load(self, model_path: str):
        """
        * Deserializes the model from a file.
        """
        self.model = joblib.load(model_path)
