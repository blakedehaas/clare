import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
class SamplingDataset(Dataset):
    """
    A dataset that samples from multiple datasets according to specified ratios.
    Optimized for fast sampling in training loops.
    """
    def __init__(self, datasets, sampling_ratios=None, batch_size=512):
        """
        Args:
            datasets (list): List of datasets to sample from
            sampling_ratios (list, optional): List of sampling ratios for each dataset.
                                           Must sum to 1. If None, samples equally from all datasets.
        """
        self.datasets = datasets
        
        # Set default sampling ratios if not provided
        if sampling_ratios is None:
            n_datasets = len(datasets)
            self.sampling_ratios = np.array([1.0/n_datasets] * n_datasets)
        else:
            self.sampling_ratios = np.array(sampling_ratios)
            
        assert len(self.sampling_ratios) == len(datasets), \
            f"Number of sampling ratios ({len(self.sampling_ratios)}) must match number of datasets ({len(datasets)})"
        assert abs(np.sum(self.sampling_ratios) - 1.0) < 1e-6, "Sampling ratios must sum to 1"
        
        # Store dataset lengths and precompute cumulative probabilities
        self.dataset_lengths = np.array([len(ds) for ds in datasets])
        self.total_length = np.sum(self.dataset_lengths)
        
        # Pre-generate random indices for faster sampling
        self.rng = np.random.Generator(np.random.PCG64())
        self.batch_size = batch_size
        self.dataset_indices = self._generate_indices()
        self.current_idx = 0

    def _generate_indices(self):
        return self.rng.choice(len(self.datasets), size=self.batch_size, p=self.sampling_ratios)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Use pre-generated dataset index
        if self.current_idx >= self.batch_size:
            self.dataset_indices = self._generate_indices()
            self.current_idx = 0
            
        dataset_idx = self.dataset_indices[self.current_idx]
        self.current_idx += 1
        
        # Fast random sampling using numpy's Generator
        sample_idx = int(self.rng.integers(self.dataset_lengths[dataset_idx]))
        
        return self.datasets[dataset_idx][sample_idx]

def calculate_stats(ds, columns):
    means_dict = {}
    stds_dict = {}
    for col in tqdm(columns, desc="Calculating stats"):
        values = ds.with_format("pandas")[col]
        mean = values.mean()
        std = values.std()
        means_dict[col] = float(mean)
        stds_dict[col] = float(std)
        
    return means_dict, stds_dict

def normalize_ds(ds, means, stds, columns, normalize_output=True):
    def normalize_batch(batch):
        if normalize_output:
            batch['Te1'] = np.log(batch['Te1'])        
        
        for col in columns:
            if col in ds.features:
                values = np.array(batch[col], dtype=np.float32)
                batch[col] = (values - means[col]) / stds[col]
        return batch

    ds = ds.map(normalize_batch, batched=True, batch_size=10000, num_proc=os.cpu_count())
    return ds

def unnormalize_mean(pred, target_mean, target_std):
    return pred * target_std + target_mean

def unnormalize_var(var, target_std):
    return var * (target_std ** 2)