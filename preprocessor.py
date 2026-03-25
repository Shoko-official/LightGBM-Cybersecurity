"""
 * Preprocessor for CSE-CIC-IDS2018 grouping features by categories.
 * Created by Shoko on 2026/03/30
 """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class Preprocessor:
    """
    * Handles feature selection and flow records cleaning.
    """

    CATEGORIES = {
        "metadata": ["Timestamp", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port"],
        "signatures": ["Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Seg Size Min"],
        "timing": ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
                   "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
                   "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Flow Duration"],
        "volume": ["Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
                   "Flow Bytes/s", "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s"],
        "packet_stats": ["Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
                         "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
                         "Pkt Len Max", "Pkt Len Min", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var"]
    }

    def __init__(self, excluded_categories: List[str] = None):
        """
        * @param excluded_categories List of CATEGORIES keys to exclude.
        """
        self.scaler = StandardScaler()
        self.excluded_categories = excluded_categories or ["metadata", "signatures", "timing"]
        
        self.drop_cols = []
        for cat in self.excluded_categories:
            if cat in self.CATEGORIES:
                self.drop_cols.extend(self.CATEGORIES[cat])

    def binary_label(self, label: str) -> int:
        """
        * Maps 'Benign' to 0 and all other categories to 1.
        """
        if str(label).lower().strip() == "benign":
            return 0
        return 1

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        * Prepares flow data for classification.
        * @return (X, y) tuple.
        """
        data = df.copy()
        y = data["Label"].apply(self.binary_label)

        to_drop = [c for c in self.drop_cols + ["Label"] if c in data.columns]
        data = data.drop(columns=to_drop)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)

        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if is_train:
            self.scaler.fit(data[num_cols])
        
        data[num_cols] = self.scaler.transform(data[num_cols])

        return data, y
