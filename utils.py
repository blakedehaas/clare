import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
class DataFrameDataset(Dataset):
    def __init__(self, dataset, input_columns, output_columns):
        pass
        # # Convert input features to tensors for faster access
        # self.X = torch.tensor(np.array([self.dataset[col] for col in tqdm(input_columns, desc="Loading input columns")]).T, dtype=torch.float32)
        
        # # Convert output features to tensors and preprocess into bin indices
        # y_raw = torch.tensor(np.array([self.dataset[col] for col in tqdm(output_columns, desc="Loading output columns")]).T, dtype=torch.float32)
        # self.y = ((y_raw - 6) / 0.05).clamp(0, 79).long().squeeze()
        
        # self.y = torch.tensor((dataframe[output_column].values // 200).clip(0, 299), dtype=torch.long).squeeze()
        # self.y = torch.tensor(((dataframe[output_column].values + 3) // 0.05).clip(0, 139), dtype=torch.long).squeeze()
        # self.y = torch.tensor(((dataframe[output_column].values - 6) // 0.05).clip(0, 79), dtype=torch.long).squeeze()        
    
    def __len__(self):
        return len(self.dataset)

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

def calculate_stats(ds, columns):
    # Norm without invalids
    # # Convert DataFrame to NumPy array for faster operations
    # data = df[columns].values.astype(np.float64)
    
    # # Define invalid values
    # invalid_values = np.array([999.9, 9.999, 9999.0, 9999.99, 99999.99, 99999.99, 9999999, 9999999.0])
    
    # # Create a mask for valid values
    # mask = ~np.isin(data, invalid_values)
    
    # # Calculate means and stds using NumPy masked operations
    # means = np.ma.masked_array(data, ~mask).mean(axis=0)
    # stds = np.ma.masked_array(data, ~mask).std(axis=0)
    # means_dict = dict(zip(columns, means.tolist()))
    # stds_dict = dict(zip(columns, stds.tolist()))
    # return means_dict, stds_dict
    try:
        # Calculate means and stds for all columns at once using pandas
        # print("Calculating means for all columns at once using pandas")
        # import time
        # start_time = time.time()
        # means_dict = df[columns].mean().to_dict()
        # end_time = time.time()
        # print(f"Time taken to calculate means: {end_time - start_time:.2f} seconds")
        # print("Calculating stds for all columns at once using pandas")
        # start_time = time.time()
        # stds_dict = df[columns].std().to_dict()
        # end_time = time.time()
        # print(f"Time taken to calculate stds: {end_time - start_time:.2f} seconds")
        means_dict = {}
        stds_dict = {}
        from tqdm import tqdm
        import os
        import json
        for col in tqdm(["Te1"]):
            stats_dir = '/home/michael/auroral-precipitation-ml/data/1_23_stats'
            stats_file = os.path.join(stats_dir, f'{col}_stats.json')
            
            if not os.path.exists(stats_file):
                # Calculate and save new stats
                # Take natural log of column values before calculating stats
                log_values = np.log(ds.with_format("pandas")[col])
                mean = log_values.mean()
                std = log_values.std()
                means_dict[col] = float(mean)  # Convert to float for JSON serialization
                stds_dict[col] = float(std)
                
                os.makedirs(stats_dir, exist_ok=True)
                stats = {
                    'mean': means_dict[col], 
                    'std': stds_dict[col]
                }
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=4)

        return means_dict, stds_dict
    except Exception as e:
        print(f"Error calculating stats: {e}")
        import IPython; IPython.embed() 
        return None, None

# Function to normalize specified columns in the DataFrame
def normalize_df(df, means, stds, columns):
    # df[columns] = (df[columns] - means) / stds
    # Create a function to normalize a single row
    # Create a dictionary mapping of normalization operations
    # from tqdm import tqdm
    # import datasets
    # for col in tqdm(columns):
    #     if col in df.features:
    #         import IPython; IPython.embed()
    #         normalized_values = [(float(x) - means[col]) / stds[col] for x in df.data[col]]
    #         import IPython; IPython.embed()
    #         df = df.remove_columns([col])
    #         df = df.add_column(col, normalized_values)
    # Create a function to normalize a single column
    # Create a function that normalizes all specified columns in a batch
    import os
    from tqdm import tqdm
    def normalize_batch(batch):
        # Output column
        # batch['Te1'] = np.log(batch['Te1'])        
        
        for col in columns:
            if col in df.features:
                # Convert to numpy array for vectorized operations
                values = np.array(batch[col], dtype=np.float32)
                # Vectorized normalization
                batch[col] = (values - means[col]) / stds[col]
        return batch

    # Use ds.map() to normalize all columns
    df = df.map(normalize_batch, batched=True, batch_size=10000, num_proc=16)
    return df

def unnormalize_mean(pred, target_mean, target_std):
    return pred * target_std + target_mean

def unnormalize_var(var, target_std):
    return var * (target_std ** 2)
