# CustomDataset.py

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, input_columns, output_columns, seq_length=24):
        """
        Initializes the CustomDataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            input_columns (list of str): List of input feature column names.
            output_columns (list of str): List of target column names.
            seq_length (int): Sequence length for time series data.
        """
        self.df = df
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.seq_length = seq_length

    def __len__(self):
        return len(self.df) - self.seq_length

    def __getitem__(self, idx):
        """
        Retrieves the input and target sequences at the specified index.
        
        Args:
            idx (int): Index of the data point.
        
        Returns:
            torch.Tensor: Input sequence tensor of shape (seq_length, num_features).
            torch.Tensor: Target tensor.
        """
        input_seq = self.df.iloc[idx:idx + self.seq_length][self.input_columns].values
        target = self.df.iloc[idx + self.seq_length][self.output_columns].values
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
