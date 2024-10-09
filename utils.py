import torch
from torch.utils.data import Dataset

class DataFrameDataset(Dataset):
    def __init__(self, dataframe, input_columns, output_column):
        self.X = torch.tensor(dataframe[input_columns].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[output_column].values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Function to calculate mean and std dev for specified columns
def calculate_stats(df, columns):
    means = df[columns].mean()
    stds = df[columns].std()
    return means, stds

# Function to normalize specified columns in the DataFrame
def normalize_df(df, means, stds, columns):
    df[columns] = (df[columns] - means) / stds
    return df

def unnormalize_mean(pred, target_mean, target_std):
    return pred * target_std + target_mean

def unnormalize_var(var, target_std):
    return var * (target_std ** 2)