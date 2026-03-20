"""
 * Dataset loader for CSE-CIC-IDS2018.
 * Created by Shoko on 2026/03/30
 """
import os
import glob
import pandas as pd
from typing import List

class DataLoader:
    """
    * Handles the acquisition and loading of CSE-CIC-IDS2018 CSV files.
    * @param data_dir Local directory containing the CSV files.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_instructions(self) -> None:
        """
        * Displays S3 download instructions via AWS CLI.
        """
        print("Use the following command to download the dataset:")
        print("aws s3 sync --no-sign-request \"s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/\" data/")

    def load_all_data(self) -> pd.DataFrame:
        """
        * Loads all CSV files from the data directory.
        * @return Concatenated dataset.
        """
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not csv_files:
            self.download_instructions()
            return pd.DataFrame()

        print(f"Loading {len(csv_files)} files...")
        df_list = [pd.read_csv(f) for f in csv_files]
        return pd.concat(df_list, ignore_index=True)

    def load_sample(self, nrows: int = 100000) -> pd.DataFrame:
        """
        * Loads a sample of the data.
        * @param nrows Number of rows to load.
        """
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not csv_files:
            self.download_instructions()
            return pd.DataFrame()

        return pd.read_csv(csv_files[0], nrows=nrows)
