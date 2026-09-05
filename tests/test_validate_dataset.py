"""Unit test suite verifying dataset validation, file audits, and data utilization checks in validate_dataset.py."""

import os
import unittest
import pandas as pd
from validate_dataset import DatasetValidator


class TestValidateDataset(unittest.TestCase):
    """Validates data integrity checks, logging assertions, and schema reporting."""

    def setUp(self):
        self.test_dir = "tests/test_output/mock_dataset"
        self.input_dir = os.path.join(self.test_dir, "input_dataset")
        os.makedirs(self.input_dir, exist_ok=True)

        # Create dummy Akebono file
        self.akebono_path = os.path.join(self.input_dir, "Akebono_combined.tsv")
        pd.DataFrame({
            "DateFormatted": ["1991-01-01", "1991-01-02"],
            "TimeFormatted": ["00:00:00", "00:00:12"],
            "Altitude": [2000.0, 3000.0],
            "ILAT": [45.0, 50.0],
            "XXLAT": [40.0, 42.0],
            "XXLON": [120.0, 125.0],
            "Te1": [2500.0, 3500.0]
        }).to_csv(self.akebono_path, sep="\t", index=False)

        # Create dummy Kp index file
        self.kp_path = os.path.join(self.input_dir, "omni_kp_index.lst")
        with open(self.kp_path, "w") as f:
            f.write("1991 01 01 00 20\n")

    def test_dataset_validator_initialization(self):
        """Tests path resolution and directory structure detection."""
        validator = DatasetValidator(base_dir=self.test_dir)
        self.assertEqual(validator.akebono_path, self.akebono_path)
        self.assertEqual(validator.kp_path, self.kp_path)

    def test_log_check_and_summary(self):
        """Tests status logging (PASS, WARN, FAIL) and summary tallying."""
        validator = DatasetValidator(base_dir=self.test_dir)
        validator.log_check("Test Check 1", "PASS", "Operational")
        validator.log_check("Test Check 2", "WARN", "Minor anomaly")
        validator.log_check("Test Check 3", "FAIL", "Critical deviation")

        self.assertEqual(len(validator.results["checks"]), 3)
        # Verify print_summary executes without error
        validator.print_summary()

    def test_validate_raw_files(self):
        """Tests raw input file presence and size detection."""
        validator = DatasetValidator(base_dir=self.test_dir)
        # akebono and kp exist, al_symh and f107 missing in mock dir -> should return False
        result = validator.validate_raw_files()
        self.assertFalse(result)

        # Check recorded statistics
        self.assertIn("akebono_size_mb", validator.results["statistics"])

    def test_validate_data_utilization_and_filtering(self):
        """Tests data filtering simulation and retention calculations."""
        validator = DatasetValidator(base_dir=self.test_dir)
        metrics = validator.validate_data_utilization_and_filtering(sample_chunk_size=10)
        self.assertIn("retention_rate_pct", metrics)
        self.assertEqual(metrics["total_raw_rows"], 2)

    def test_validate_processed_dataset_missing(self):
        """Tests validation reporting when processed dataset directory is absent."""
        validator = DatasetValidator(base_dir=self.test_dir)
        result = validator.validate_processed_dataset()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
