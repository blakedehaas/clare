"""
Block-Split Multi-Seed Ensemble Evaluator for CLARE model.

Evaluates 3 seed-trained models on 7-day block-split test sets (quiet-time or storm-time),
calculates classification/regression performance metrics, computes 95% confidence intervals,
evaluates the ensemble mean prediction, and generates diagnostic deviation plots.
"""

import os
import json
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

import torch
from torch.utils.data import DataLoader
import datasets

import models.feed_forward as models
import constants


class EnsembleEvaluator:
    """Encapsulates the evaluation pipeline for multi-seed trained models."""

    def __init__(self, base_model_name: str, seeds: List[int], dataset_type: str, is_continuous: bool = False) -> None:
        """
        Initialize the evaluator with target models and dataset configurations.

        Args:
            base_model_name: Base identifier string for the model (e.g., '1_47').
            seeds: List of random seed IDs used during dataset generation and training.
            dataset_type: Target split name ('test-blocks' or 'test-storm').
            is_continuous: Flag to evaluate regression variants instead of classification.
        """
        self.base_model_name = base_model_name
        self.seeds = seeds
        self.dataset_type = dataset_type
        self.is_continuous = is_continuous

        # Define model architecture dimensions
        self.hidden_dim = 2048
        self.output_dim = 1 if self.is_continuous else 150

        # Define feature schema
        self.input_columns = [
            'Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON',
            'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5',
            'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11',
            'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17',
            'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23',
            'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29',
            'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5',
            'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12',
            'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19',
            'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26',
            'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33',
            'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40',
            'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47',
            'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54',
            'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61',
            'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68',
            'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75',
            'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82',
            'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89',
            'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96',
            'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103',
            'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110',
            'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117',
            'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124',
            'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131',
            'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138',
            'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144',
            'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index'
        ]
        self.output_columns = ['Te1']
        self.all_columns = self.input_columns + self.output_columns

        # Determine CPU core count dynamically from process affinity mask
        self.num_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 8

    def _load_and_preprocess_dataset(self, seed: int) -> DataLoader:
        """
        Load, normalize, and prepare PyTorch DataLoader for a specific seed dataset.

        Args:
            seed: Seed identifier corresponding to the target dataset split.

        Returns:
            DataLoader configured for parallel CPU fetching and CUDA transfers.
        """
        dataset_path = f"dataset/processed_dataset_blocksplit_s{seed}/{self.dataset_type}"
        # Stats file relies strictly on the base model name regardless of classification/continuous type
        stats_filepath = f"checkpoints/{self.base_model_name}_s{seed}_norm_stats.json"

        # Load HuggingFace dataset from disk
        raw_dataset = datasets.Dataset.load_from_disk(dataset_path)

        # Safely remove unused secondary electron temperature and density features
        metadata_columns_to_strip = [
            'Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'
        ]
        active_strip_columns = [col for col in metadata_columns_to_strip if col in raw_dataset.column_names]
        processed_dataset = raw_dataset.remove_columns(active_strip_columns)

        # Apply basic standardizations defined in constants
        def _apply_constant_normalizations(batch: Dict[str, Any]) -> Dict[str, Any]:
            for column, normalization_function in constants.NORMALIZATIONS.items():
                if column in batch:
                    batch[column] = normalization_function(batch[column])
            return batch

        processed_dataset = processed_dataset.map(
            _apply_constant_normalizations,
            batched=True,
            batch_size=10000,
            num_proc=self.num_workers,
            load_from_cache_file=False  # FORCE RECOMPUTE
        )

        # Load mean and std normalization stats cached during training
        with open(stats_filepath, 'r', encoding='utf-8') as file_handle:
            stats_data = json.load(file_handle)
            group_means = stats_data['mean']
            group_stds = stats_data['std']

        index_group_names = ['AL_index', 'SYM_H', 'f107_index']
        grouped_columns = [
            col for col in self.input_columns
            if any(col.startswith(group_prefix) for group_prefix in index_group_names)
        ]

        # Apply group z-score normalization using training set statistics
        def _apply_group_normalizations(batch: Dict[str, Any]) -> Dict[str, Any]:
            for column in grouped_columns:
                group_name = '_'.join(column.split('_')[:-1]) if column.split('_')[-1].isdigit() else column
                raw_values = np.array(batch[column], dtype=np.float32)
                batch[column] = (raw_values - group_means[group_name]) / group_stds[group_name]
            return batch

        processed_dataset = processed_dataset.map(
            _apply_group_normalizations,
            batched=True,
            batch_size=10000,
            num_proc=self.num_workers,
            load_from_cache_file=False  # FORCE RECOMPUTE
        )

        # Convert feature dictionary into PyTorch FloatTensors
        def _convert_row_to_tensor(row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
            input_tensor = torch.tensor([row[col] for col in self.input_columns], dtype=torch.float32)
            target_tensor = torch.tensor([row[col] for col in self.output_columns], dtype=torch.float32)
            return {"input_ids": input_tensor, "label": target_tensor}

        processed_dataset = processed_dataset.map(
            _convert_row_to_tensor,
            num_proc=self.num_workers,
            remove_columns=processed_dataset.column_names, # PURGE ALL REMAINING NON-TENSOR COLUMNS
            load_from_cache_file=False  # FORCE RECOMPUTE
        )

        # Explicitly lock to torch format so the DataLoader only sees inputs and labels
        processed_dataset.set_format(type="torch", columns=["input_ids", "label"])
        
        return DataLoader(
            processed_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=self.num_workers
        )

    def evaluate_single_seed(self, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate a single seed model model instance on its respective test split.

        Args:
            seed: Seed ID to evaluate.

        Returns:
            Tuple of NumPy arrays containing (predictions, true_values, predictive_entropies).
        """
        # Resolve correct checkpoint name based on model mode
        model_instance_name = f"{self.base_model_name}_s{seed}_continuous" if self.is_continuous else f"{self.base_model_name}_s{seed}"
        checkpoint_path = f"checkpoints/{model_instance_name}_best.pth"

        # Fall back to final checkpoint if best checkpoint is unavailable
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"checkpoints/{model_instance_name}.pth"

        print(f"\n[INFO] Evaluating Model Seed {seed} on {self.dataset_type}")
        print(f"[INFO] Mode: {'Continuous Regression' if self.is_continuous else 'Classification'}")
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")

        # Instantiate neural network architecture
        neural_network = models.FeedForwardNetwork(
            len(self.input_columns),
            self.hidden_dim,
            self.output_dim
        ).to("cuda")

        neural_network.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        neural_network.eval()

        test_loader = self._load_and_preprocess_dataset(seed)

        accumulated_predictions: List[np.ndarray] = []
        accumulated_targets: List[np.ndarray] = []
        accumulated_entropies: List[np.ndarray] = []

        # Run inference loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Inference Seed {seed}"):
                inputs = batch["input_ids"].to("cuda")
                targets = batch["label"].to("cuda")

                output_logits = neural_network(inputs)

                if self.is_continuous:
                    # Direct scalar outputs to temperature scale
                    predicted_temperatures = output_logits.squeeze(-1)
                    # Entropy is undefined/irrelevant for purely continuous MSE without probability distributions
                    entropy_values = np.zeros_like(predicted_temperatures.cpu().numpy())
                else:
                    # Compute predictive entropy of softmax classification distribution
                    softmax_probabilities = torch.softmax(output_logits, dim=1)
                    entropy_values = -torch.sum(
                        softmax_probabilities * torch.log(softmax_probabilities + 1e-10), dim=1
                    ).cpu().numpy()

                    # Convert class bins back to physical temperature values (K)
                    predicted_bins = torch.argmax(output_logits, dim=1)
                    predicted_temperatures = predicted_bins * 100 + 50

                accumulated_predictions.extend(predicted_temperatures.flatten().cpu().numpy())
                accumulated_targets.extend(targets.flatten().cpu().numpy())
                accumulated_entropies.extend(entropy_values)

        return (
            np.array(accumulated_predictions),
            np.array(accumulated_targets),
            np.array(accumulated_entropies)
        )

    @staticmethod
    def calculate_95_confidence_interval(data_array: np.ndarray) -> Tuple[float, float]:
        """
        Compute mean and 95% confidence interval margin using Student's t-distribution.

        Args:
            data_array: Array containing metric values across seeds.

        Returns:
            Tuple of (sample_mean, confidence_interval_margin).
        """
        sample_mean = float(np.mean(data_array))
        standard_error = st.sem(data_array)
        degrees_of_freedom = len(data_array) - 1
        confidence_margin = standard_error * st.t.ppf((1 + 0.95) / 2.0, degrees_of_freedom)
        return sample_mean, float(confidence_margin)

    def run_full_evaluation(self) -> None:
        """Execute evaluation across all seeds, compute ensemble metrics, and generate plots."""
        seed_predictions_list: List[np.ndarray] = []
        seed_targets_list: List[np.ndarray] = []
        seed_entropies_list: List[np.ndarray] = []

        individual_r2_scores = []
        individual_rmse_scores = []
        individual_mean_entropies = []

        for seed in self.seeds:
            predictions, targets, entropies = self.evaluate_single_seed(seed)
            seed_predictions_list.append(predictions)
            seed_targets_list.append(targets)
            seed_entropies_list.append(entropies)
            
            # Calculate metrics specific to this seed's unique test set
            individual_r2_scores.append(r2_score(targets, predictions))
            individual_rmse_scores.append(np.sqrt(mean_squared_error(targets, predictions)))
            individual_mean_entropies.append(np.mean(entropies))

        # Compute 95% Confidence Intervals of the METRICS across the 3 seeds
        mean_r2, ci_r2 = self.calculate_95_confidence_interval(np.array(individual_r2_scores))
        mean_rmse, ci_rmse = self.calculate_95_confidence_interval(np.array(individual_rmse_scores))
        mean_entropy, ci_entropy = self.calculate_95_confidence_interval(np.array(individual_mean_entropies))

        # Concatenate all disjoint predictions into one unified distribution for plotting
        all_predictions_flat = np.concatenate(seed_predictions_list)
        all_targets_flat = np.concatenate(seed_targets_list)
        overall_r2 = r2_score(all_targets_flat, all_predictions_flat)
        overall_rmse = np.sqrt(mean_squared_error(all_targets_flat, all_predictions_flat))

        # Print detailed report
        print("\n" + "=" * 60)
        print("                  EVALUATION RESULTS REPORT                  ")
        print("=" * 60)
        for index, seed in enumerate(self.seeds):
            print(
                f"Seed {seed} -> R²: {individual_r2_scores[index]:.4f} | "
                f"RMSE: {individual_rmse_scores[index]:.4f} | "
                f"Mean Entropy: {individual_mean_entropies[index]:.4f}"
            )

        print("-" * 60)
        print("Mean Across Seeds (95% Confidence Interval):")
        print(f"  R² Score:     {mean_r2:.4f} ± {ci_r2:.4f}")
        print(f"  RMSE:         {mean_rmse:.4f} ± {ci_rmse:.4f} K")
        if not self.is_continuous:
            print(f"  Mean Entropy: {mean_entropy:.4f} ± {ci_entropy:.4f}")
        print(f"\nOverall Concatenated Model -> R²: {overall_r2:.4f} | RMSE: {overall_rmse:.4f} K")
        print("=" * 60)

        # Plot Overall Deviation vs Ground Truth
        self.plot_ensemble_deviations(
            all_targets_flat,
            all_predictions_flat,
            overall_r2,
            overall_rmse,
            mean_r2,
            ci_r2,
            mean_rmse,
            ci_rmse
        )

    def plot_ensemble_deviations(
        self,
        ground_truth: np.ndarray,
        ensemble_predictions: np.ndarray,
        ensemble_r2: float,
        ensemble_rmse: float,
        mean_r2: float,
        ci_r2: float,
        mean_rmse: float,
        ci_rmse: float
    ) -> None:
        """
        Generate and save a 2D histogram plot comparing ensemble deviations against true values.

        Args:
            ground_truth: True temperature values.
            ensemble_predictions: Mean predicted temperature values across models.
            ensemble_r2: Calculated ensemble R² score.
            ensemble_rmse: Calculated ensemble RMSE value.
            mean_r2: Mean R² across seeds.
            ci_r2: R² 95% confidence interval margin.
            mean_rmse: Mean RMSE across seeds.
            ci_rmse: RMSE 95% confidence interval margin.
        """
        prediction_deviations = ensemble_predictions - ground_truth

        plt.figure(figsize=(10, 8))
        histogram_2d = plt.hist2d(
            ground_truth,
            prediction_deviations,
            bins=100,
            norm=LogNorm(),
            cmap='viridis'
        )
        plt.colorbar(histogram_2d[3], label='Obs Count')

        # Binned statistics for mean deviation trendline
        binned_means, bin_edges, _ = st.binned_statistic(
            ground_truth,
            prediction_deviations,
            statistic='mean',
            bins=50
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        plt.plot(bin_centers, binned_means, 'r-', linewidth=2, label='Mean Deviation')

        plt.xlabel('Te$_{obs}$ [K]')
        plt.ylabel('Te$_{model}$ - Te$_{obs}$ [K]')
        title_suffix = "Continuous" if self.is_continuous else "Classification"
        plt.title(f'Ensemble Model Deviation vs Ground Truth ({self.dataset_type} - {title_suffix})')
        plt.legend(loc='lower right')

        annotation_text = "\n".join([
            f"Ensemble R²: {ensemble_r2:.4f}",
            f"Ensemble RMSE: {ensemble_rmse:.4f} K",
            f"R² (95% CI): {mean_r2:.4f} ± {ci_r2:.4f}",
            f"RMSE (95% CI): {mean_rmse:.4f} ± {ci_rmse:.4f} K",
        ])
        plt.text(
            0.05, 0.95, annotation_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
        )

        plt.tight_layout()
        os.makedirs('checkpoints', exist_ok=True)
        # Update output string so standard plots are not overwritten
        output_prefix = f"{self.base_model_name}_continuous" if self.is_continuous else self.base_model_name
        output_plot_filepath = f"./checkpoints/{output_prefix}_{self.dataset_type}_ensemble_deviation.png"
        plt.savefig(output_plot_filepath, dpi=300)
        plt.close()
        print(f"\n[INFO] Saved ensemble evaluation plot to: {output_plot_filepath}")


def main() -> None:
    """Parse command line arguments and execute evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate CLARE Multi-Seed Ensemble Models.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="test-blocks",
        choices=["test-blocks", "test-storm"],
        help="Target test split ('test-blocks' or 'test-storm')."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="1_47",
        help="Base model architecture tag."
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Evaluate continuous regression models instead of classification."
    )
    args = parser.parse_args()

    evaluator = EnsembleEvaluator(
        base_model_name=args.model_name,
        seeds=[0, 1, 2],
        dataset_type=args.dataset,
        is_continuous=args.continuous
    )
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()