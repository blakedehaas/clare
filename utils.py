import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class DataFrameDataset(Dataset):
    def __init__(self, dataframe, input_columns, output_column):
        self.X = torch.tensor(dataframe[input_columns].values, dtype=torch.float32)
        # Values
        # self.y = torch.tensor(dataframe[output_column].values, dtype=torch.float32).reshape(-1, 1)
        # Bins
        # self.y = torch.tensor((dataframe[output_column].values // 200).clip(0, 299), dtype=torch.long).squeeze()
        self.y = torch.tensor(((dataframe[output_column].values - 6) // 0.05).clip(0, 79), dtype=torch.long).squeeze()
        # self.y = torch.tensor(((dataframe[output_column].values + 3) // 0.05).clip(0, 139), dtype=torch.long).squeeze()

        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SamplingDataset(Dataset):
    def __init__(self, dataframe, input_columns, output_column, sampling_ratios=None):
        self.input_columns = input_columns
        self.output_column = output_column
        
        # Calculate quartiles
        quartiles = dataframe[output_column].quantile([0.25, 0.5, 0.75])
        
        # Create masks for each quartile
        q1_mask = dataframe[output_column] <= quartiles[0.25]
        q2_mask = (dataframe[output_column] > quartiles[0.25]) & (dataframe[output_column] <= quartiles[0.5])
        q3_mask = (dataframe[output_column] > quartiles[0.5]) & (dataframe[output_column] <= quartiles[0.75])
        q4_mask = dataframe[output_column] > quartiles[0.75]
        
        # Store indices for each quartile
        self.quartile_indices = [
            dataframe.index[q1_mask].tolist(),
            dataframe.index[q2_mask].tolist(),
            dataframe.index[q3_mask].tolist(),
            dataframe.index[q4_mask].tolist()
        ]
        
        # Convert dataframe to tensors for faster access
        self.X = torch.tensor(dataframe[input_columns].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[output_column].values, dtype=torch.float32).reshape(-1, 1)
        
        # Set default sampling ratios if not provided
        self.sampling_ratios = sampling_ratios or [0.25, 0.25, 0.25, 0.25]
        assert len(self.sampling_ratios) == 4, "Sampling ratios must be a list of 4 values"
        assert abs(sum(self.sampling_ratios) - 1.0) < 1e-6, "Sampling ratios must sum to 1"

    def __len__(self):
        return sum(len(q) for q in self.quartile_indices)

    def __getitem__(self, idx):
        # Determine which quartile to sample from based on the sampling ratios
        quartile_idx = np.random.choice(4, p=self.sampling_ratios)
        
        # Get the indices for the chosen quartile
        quartile_indices = self.quartile_indices[quartile_idx]
        
        # Sample a random index from the chosen quartile
        sample_idx = quartile_indices[np.random.randint(len(quartile_indices))]
        
        # Return the sampled data point
        return self.X[sample_idx], self.y[sample_idx]

# Function to calculate mean and std dev for specified columns
def calculate_stats(df, columns):
    data = df[columns].values
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return pd.Series(means, index=columns), pd.Series(stds, index=columns)

# Function to normalize specified columns in the DataFrame
def normalize_df(df, means, stds, columns):
    df[columns] = (df[columns] - means) / stds
    return df

def unnormalize_mean(pred, target_mean, target_std):
    return pred * target_std + target_mean

def unnormalize_var(var, target_std):
    return var * (target_std ** 2)