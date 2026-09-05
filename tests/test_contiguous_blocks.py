"""Unit test suite verifying contiguous equal-sized orbital block partitioning and boundary guard band isolation."""

import unittest
import numpy as np
import pandas as pd


class TestContiguousBlocks(unittest.TestCase):
    """Validates temporal continuity, equal block dimensions, and guard band boundary isolation across splits."""

    def test_block_partitioning_and_guard_bands(self):
        """Tests that all evaluation blocks are contiguous, equally sized, and temporally buffered from training data."""
        timestamps = pd.date_range("1991-01-01", periods=15000, freq="12s")
        synthetic_telemetry = pd.DataFrame({
            "Altitude": np.random.uniform(1000, 8000, size=len(timestamps)),
            "Te1": np.random.uniform(1000, 10000, size=len(timestamps)),
        }, index=timestamps)

        block_size = 150
        num_test_blocks = 10

        synthetic_telemetry.sort_index(inplace=True)
        total_samples = len(synthetic_telemetry)
        num_blocks = total_samples // block_size
        truncated_length = num_blocks * block_size

        telemetry_blocked = synthetic_telemetry.iloc[:truncated_length].copy()
        telemetry_blocked['block_id'] = np.repeat(np.arange(num_blocks), block_size)

        random_state = np.random.RandomState(42)
        candidate_blocks = list(range(2, num_blocks - 2))
        shuffled_candidates = random_state.permutation(candidate_blocks)
        selected_set = set()
        selected_test_blocks = []
        for block_index in shuffled_candidates:
            if (block_index - 1) not in selected_set and (block_index + 1) not in selected_set:
                selected_set.add(block_index)
                selected_test_blocks.append(block_index)
                if len(selected_test_blocks) == num_test_blocks:
                    break

        selected_test_blocks = set(selected_test_blocks)
        self.assertEqual(len(selected_test_blocks), num_test_blocks)

        guard_band_blocks = set()
        for test_block_index in selected_test_blocks:
            guard_band_blocks.add(test_block_index - 1)
            guard_band_blocks.add(test_block_index + 1)

        test_mask = telemetry_blocked['block_id'].isin(selected_test_blocks)
        train_mask = ~telemetry_blocked['block_id'].isin(selected_test_blocks | guard_band_blocks)

        test_split = telemetry_blocked[test_mask].copy()
        train_split = telemetry_blocked[train_mask].copy()

        self.assertEqual(len(test_split), num_test_blocks * block_size)

        block_sample_counts = test_split.groupby('block_id').size()
        self.assertEqual(len(block_sample_counts), num_test_blocks)
        self.assertTrue((block_sample_counts == block_size).all())

        training_block_ids = set(train_split['block_id'].unique())
        self.assertEqual(len(training_block_ids.intersection(selected_test_blocks)), 0)
        self.assertEqual(len(training_block_ids.intersection(guard_band_blocks)), 0)

        for test_block_id in selected_test_blocks:
            current_block = test_split[test_split['block_id'] == test_block_id]
            earliest_timestamp = current_block.index.min()
            latest_timestamp = current_block.index.max()

            self.assertTrue(current_block.index.is_monotonic_increasing)

            overlapping_training_samples = train_split[
                (train_split.index >= earliest_timestamp) & (train_split.index <= latest_timestamp)
            ]
            self.assertEqual(len(overlapping_training_samples), 0)

            # Check that adjacent guard band duration separates any training sample
            time_deltas_before = (earliest_timestamp - train_split.index[train_split.index < earliest_timestamp])
            if len(time_deltas_before) > 0:
                min_separation_seconds = time_deltas_before.min().total_seconds()
                self.assertGreater(min_separation_seconds, block_size * 10)


if __name__ == "__main__":
    unittest.main()
