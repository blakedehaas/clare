import os
import torch
import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset

from models.feed_forward import FF_2Network
import utils


class Evaluator:
    def __init__(self, 
                 model_path, 
                 norm_stats_path,
                 dataset_path,
                 input_columns,
                 output_columns,
                 input_size=188,
                 hidden_size=2048,
                 output_size=80,
                 batch_size=512,
                 device="cuda"):
        """
        Initializes the Evaluator class with model, dataset, etc.
        """
        self.model_path = model_path
        self.norm_stats_path = norm_stats_path
        self.dataset_path = dataset_path
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.all_columns = input_columns + output_columns
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device

        # Load model
        self.model = FF_2Network(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self._load_model()

        # Load dataset
        self.test_ds = self._load_dataset()

    def _load_model(self):
        """
        Loads the model checkpoint and sets it to evaluation mode.
        """
        print(f"Loading model from {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def _load_dataset(self):
        """
        Loads and normalizes the dataset, then converts it to torch tensors.
        """
        print(f"Loading dataset from {self.dataset_path}")
        ds = Dataset.load_from_disk(self.dataset_path)

        # Remove unused columns but keep DateTimeFormatted
        columns_to_remove = [
            'Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 
            'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'
        ]
        if 'DateTimeFormatted' not in ds.column_names:
            raise ValueError("DateTimeFormatted column is missing in the dataset.")

        ds = ds.remove_columns([col for col in columns_to_remove if col in ds.column_names])

        # Load normalization stats
        print(f"Loading normalization stats from {self.norm_stats_path}")
        with open(self.norm_stats_path, 'r') as f:
            stats = json.load(f)
            means = stats['mean']
            stds = stats['std']

        # Normalize dataset
        ds = utils.normalize_ds(ds, means, stds, self.input_columns, normalize_output=False)

        # Convert dataset to torch tensors while keeping DateTimeFormatted
        def convert_to_tensor(row):
            input_ids = torch.tensor([v for k, v in row.items() if k not in self.output_columns and k != "DateTimeFormatted"])
            label = torch.tensor([v for k, v in row.items() if k in self.output_columns])
            date_time = row.get("DateTimeFormatted", None)
            return {"input_ids": input_ids, "label": label, "DateTimeFormatted": date_time}

        ds = ds.map(
            convert_to_tensor,
            num_proc=1,
            remove_columns=[col for col in self.all_columns if col != "DateTimeFormatted"],
            desc="Converting test_ds to tensor"
        )
        ds.set_format(type="torch")
        return ds

    def evaluate(self):
        """
        Runs model inference on the test dataset and returns predictions, true values, and timestamps.
        """
        test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

        predictions, true_values, timestamps = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                x = batch["input_ids"].to(self.device)
                y = batch["label"].to(self.device)

                # Forward pass
                y_pred = self.model(x)
                # As per the original code for post-processing
                y_pred = torch.exp(torch.argmax(y_pred, dim=1) * 0.05 + 6 + 0.025)

                predictions.extend(y_pred.flatten().tolist())
                true_values.extend(y.flatten().tolist())
                timestamps.extend(batch.get("DateTimeFormatted", []))  # Include timestamps

        return predictions, true_values, timestamps


def main():
    # Configuration
    model_name = '1_24'
    model_path = "/glade/campaign/univ/ucul0008/auroral_precipitation_ml/Git_Repository/models/experiments/1.24/checkpoints/1_24.pth"
    norm_stats_path = "/glade/campaign/univ/ucul0008/auroral_precipitation_ml/data/1_24_norm_stats.json"
    dataset_path = "/glade/campaign/univ/ucul0008/auroral_precipitation_ml/data/akebono_solar_combined_v6_chu_val"
    input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 
                     'AL_index_0', 'SYM_H_0', 'f107_index_0']
    output_columns = ['Te1']

    # Common output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)  # Create output dir if it doesn't exist

    evaluator = Evaluator(
        model_path=model_path,
        norm_stats_path=norm_stats_path,
        dataset_path=dataset_path,
        input_columns=input_columns,
        output_columns=output_columns,
        input_size=188,
        hidden_size=2048,
        output_size=80
    )

    # Run evaluation
    predictions, true_values, timestamps = evaluator.evaluate()

    # Compute deviations
    deviations = [p - t for p, t in zip(predictions, true_values)]

    # Save results to a CSV in the output directory
    results_df = pd.DataFrame({
        "DateTimeFormatted": timestamps,
        "predictions": predictions,
        "true_values": true_values,
        "deviations": deviations
    })
    csv_path = os.path.join(output_dir, f"evaluation_results_{model_name}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to {csv_path}")


if __name__ == "__main__":
    main()
