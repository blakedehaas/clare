# DataProcessor.py

import pandas as pd
import numpy as np
import json

class DataProcessor:
    def __init__(self, data_path):
        """
        Initializes the DataProcessor.
        
        Args:
            data_path (str): Path to the dataset.
        """
        self.data_path = data_path
        self.df = None
        self.medians = None
        self.iqr = None

    def preprocess_data(self):
        """
        Loads and preprocesses the dataset.
        """
        try:
            self.df = pd.read_csv(self.data_path, sep='\t')
            print("[INFO] Dataset loaded successfully.")
            # Additional preprocessing steps can be added here
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            raise

    def calculate_stats(self, df, columns):
        """
        Calculates median and interquartile range (IQR) for specified columns.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            columns (list of str): List of column names to calculate statistics for.
        
        Returns:
            pd.Series: Medians of the columns.
            pd.Series: IQRs of the columns.
        """
        self.medians = df[columns].median()
        q75 = df[columns].quantile(0.75)
        q25 = df[columns].quantile(0.25)
        self.iqr = q75 - q25
        return self.medians, self.iqr


    def normalize_df(self, df, medians, iqr, columns):
        """
        Normalizes the specified columns in the DataFrame using median and IQR.
        
        Args:
            df (pd.DataFrame): DataFrame to normalize.
            medians (pd.Series): Medians of the columns.
            iqr (pd.Series): IQRs of the columns.
            columns (list of str): List of column names to normalize.
        
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        df_norm = df.copy()
        for col in columns:
            if iqr[col] == 0:
                df_norm[col] = 0.0
            else:
                df_norm[col] = (df_norm[col] - medians[col]) / iqr[col]
        return df_norm

    def save_transformations(self, path):
        """
        Saves normalization statistics to a JSON file.
        
        Args:
            path (str): Path to save the JSON file.
        """
        try:
            stats = {
                'medians': self.medians.to_dict(),
                'iqr': self.iqr.to_dict()
            }
            with open(path, 'w') as f:
                json.dump(stats, f)
            print(f"[INFO] Normalization statistics saved to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save normalization statistics: {e}")
            raise
