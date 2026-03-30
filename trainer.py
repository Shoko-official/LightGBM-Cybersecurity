"""
 * ML Pipeline orchestrator for CSE-CIC-IDS2018.
 * Created by Shoko on 2026/03/30
 """
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from preprocessor import Preprocessor
from detector import IntrusionDetector
from evaluator import ModelEvaluator

class ModelTrainer:
    """
    * Manages the end-to-end training workflow.
    """

    def __init__(self, data_dir: str = "data"):
        self.loader = DataLoader(data_dir)
        self.preprocessor = Preprocessor()
        self.detector = IntrusionDetector()

    def run(self, sample_size: int = 500000):
        """
        * Executes the full pipeline: Load -> Preprocess -> Train -> Evaluate.
        """
        print(f"Starting training pipeline with {sample_size} records...")
        
        df = self.loader.load_sample(sample_size)
        if df.empty:
            print("No data found. Aborting.")
            return

        X, y = self.preprocessor.preprocess(df)
        
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.detector.train(x_train, y_train, x_val, y_val)
        self.detector.save()

        evaluator = ModelEvaluator(self.detector, x_val, y_val)
        evaluator.generate_report()

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
