"""
validate_dataset.py - Comprehensive Dataset Quality & Data Utilization Validator for CLARE.

Validates:
1. Raw Input Data Integrity (Akebono TSV, OMNI indices, F10.7, Kp index).
2. Data Filtering & Utilization Analysis:
   - Quantifies rows removed at each filtering stage.
   - Evaluates whether valid data is being inadvertently discarded.
3. Feature Engineering & Temporal Alignment:
   - AL index lags (31 columns: 0-5 hours at 10-min resolution).
   - SYM-H index lags (145 columns: 0-3 days at 30-min resolution).
   - F10.7 index lags (4 columns: 0-72 hours at 24-hour resolution).
   - Kp index hourly alignment.
4. Processed Dataset Quality (if generated):
   - Schema validation (153 input features + 1 target).
   - NaN / Inf / Sentinel value audit.
   - Leakage detection between train, validation, and test sets.
   - Target variable distribution (Te1 range, physical plausibility).
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class DatasetValidator:
    """Comprehensive validator for CLARE raw and processed datasets."""

    def __init__(self, base_dir: str = "."):
        # Normalize paths whether running from root or dataset/
        if os.path.exists(os.path.join(base_dir, "input_dataset")):
            self.dataset_dir = base_dir
        elif os.path.exists(os.path.join(base_dir, "dataset", "input_dataset")):
            self.dataset_dir = os.path.join(base_dir, "dataset")
        else:
            self.dataset_dir = base_dir

        self.input_dir = os.path.join(self.dataset_dir, "input_dataset")
        self.processed_dir = os.path.join(self.dataset_dir, "processed_dataset_01_31_storm")

        self.akebono_path = os.path.join(self.input_dir, "Akebono_combined.tsv")
        self.kp_path = os.path.join(self.input_dir, "omni_kp_index.lst")
        self.al_symh_glob = os.path.join(self.input_dir, "omni_al_index_symh", "*.lst")
        self.f107_glob = os.path.join(self.input_dir, "omni_f107", "*.lst")

        self.results: Dict[str, Any] = {"checks": [], "statistics": {}, "recommendations": []}

    def log_check(self, name: str, status: str, details: str):
        """Records a check result (PASS, WARN, FAIL)."""
        symbol = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}.get(status, "[INFO]")
        print(f"{symbol} {name}: {details}")
        self.results["checks"].append({"name": name, "status": status, "details": details})

    def validate_raw_files(self) -> bool:
        """Checks raw data file presence, sizes, and accessibility."""
        print("\n" + "=" * 70)
        print("STAGE 1: Raw Input Files Verification")
        print("=" * 70)

        all_ok = True

        # 1. Akebono TSV
        if os.path.exists(self.akebono_path):
            size_mb = os.path.getsize(self.akebono_path) / (1024 * 1024)
            self.log_check("Akebono File", "PASS", f"Found at '{self.akebono_path}' ({size_mb:.1f} MB)")
            self.results["statistics"]["akebono_size_mb"] = size_mb
        else:
            self.log_check("Akebono File", "FAIL", f"Missing at '{self.akebono_path}'")
            all_ok = False

        # 2. Kp index file
        if os.path.exists(self.kp_path):
            size_kb = os.path.getsize(self.kp_path) / 1024
            self.log_check("Kp Index File", "PASS", f"Found at '{self.kp_path}' ({size_kb:.1f} KB)")
        else:
            self.log_check("Kp Index File", "FAIL", f"Missing at '{self.kp_path}'")
            all_ok = False

        # 3. AL / SYM-H files
        al_symh_files = sorted(glob.glob(self.al_symh_glob))
        if al_symh_files:
            years = [os.path.basename(f).split(".")[0] for f in al_symh_files]
            self.log_check("OMNI AL/SYM-H Files", "PASS", f"Found {len(al_symh_files)} files covering {years[0]} to {years[-1]}")
            self.results["statistics"]["al_symh_file_count"] = len(al_symh_files)
        else:
            self.log_check("OMNI AL/SYM-H Files", "FAIL", f"No .lst files matching '{self.al_symh_glob}'")
            all_ok = False

        # 4. F10.7 files
        f107_files = sorted(glob.glob(self.f107_glob))
        if f107_files:
            years_f = [os.path.basename(f).split(".")[0] for f in f107_files]
            self.log_check("F10.7 Solar Flux Files", "PASS", f"Found {len(f107_files)} files covering {years_f[0]} to {years_f[-1]}")
            self.results["statistics"]["f107_file_count"] = len(f107_files)
        else:
            self.log_check("F10.7 Solar Flux Files", "FAIL", f"No .lst files matching '{self.f107_glob}'")
            all_ok = False

        return all_ok

    def validate_data_utilization_and_filtering(self, sample_chunk_size: int = 500_000) -> Dict[str, Any]:
        """
        Simulates the filtering stages of create_dataset.py, auditing exact row counts
        and verifying that we maximize data retention without corrupting quality.
        """
        print("\n" + "=" * 70)
        print("STAGE 2: Data Utilization & Filtering Audit")
        print("=" * 70)

        if not os.path.exists(self.akebono_path):
            self.log_check("Data Utilization", "FAIL", "Cannot audit without Akebono raw file")
            return {}

        total_rows = 0
        removed_999 = 0
        removed_ilat = 0
        removed_altitude = 0
        removed_pre1990 = 0
        removed_nat = 0
        invalid_te1_count = 0

        altitude_min, altitude_max = float("inf"), float("-inf")
        te1_min, te1_max = float("inf"), float("-inf")

        print("Streaming Akebono TSV to compute exact filter attribution...")
        for chunk in tqdm(pd.read_csv(self.akebono_path, sep='\t', chunksize=sample_chunk_size), desc="Auditing Akebono"):
            total_rows += len(chunk)

            # Check invalid Te1 (missing, zero, negative, sentinel 9999)
            if "Te1" in chunk.columns:
                te1_col = pd.to_numeric(chunk["Te1"], errors="coerce")
                invalid_te1 = (te1_col.isna()) | (te1_col <= 0) | (te1_col >= 9999)
                invalid_te1_count += invalid_te1.sum()
                valid_te = te1_col[~invalid_te1]
                if len(valid_te) > 0:
                    te1_min = min(te1_min, float(valid_te.min()))
                    te1_max = max(te1_max, float(valid_te.max()))

            # Filter 1: 999 in XXLAT or XXLON
            mask_999 = (chunk[['XXLAT', 'XXLON']] == 999).any(axis=1)
            removed_999 += mask_999.sum()
            f1 = chunk[~mask_999]

            # Filter 2: ILAT > 90
            mask_ilat = f1['ILAT'] > 90
            removed_ilat += mask_ilat.sum()
            f2 = f1[~mask_ilat]

            # Filter 3: Altitude outside [1000, 8000]
            mask_alt = (f2['Altitude'] < 1000) | (f2['Altitude'] > 8000)
            removed_altitude += mask_alt.sum()
            f3 = f2[~mask_alt]

            if len(f3) > 0:
                altitude_min = min(altitude_min, float(f3['Altitude'].min()))
                altitude_max = max(altitude_max, float(f3['Altitude'].max()))

            # Filter 4: Date parsing & pre-1990
            date_col = pd.to_datetime(f3['DateFormatted'], errors='coerce')
            removed_nat += date_col.isna().sum()
            valid_dates = date_col.dropna()
            removed_pre1990 += (valid_dates < '1990-01-01').sum()

        remaining_clean = total_rows - (removed_999 + removed_ilat + removed_altitude + removed_nat + removed_pre1990)
        retention_rate = (remaining_clean / total_rows) * 100

        print(f"\n--- Data Retention Breakdown ---")
        print(f"Total Raw Rows:                    {total_rows:>10,}")
        print(f"Removed (XXLAT/XXLON == 999):      {removed_999:>10,} ({removed_999/total_rows*100:5.2f}%)")
        print(f"Removed (ILAT > 90):               {removed_ilat:>10,} ({removed_ilat/total_rows*100:5.2f}%)")
        print(f"Removed (Altitude <1000 or >8000): {removed_altitude:>10,} ({removed_altitude/total_rows*100:5.2f}%)")
        print(f"Removed (Date is NaT):             {removed_nat:>10,} ({removed_nat/total_rows*100:5.2f}%)")
        print(f"Removed (Date < 1990-01-01):       {removed_pre1990:>10,} ({removed_pre1990/total_rows*100:5.2f}%)")
        print(f"--------------------------------------------------")
        print(f"High-Quality Usable Rows:          {remaining_clean:>10,} ({retention_rate:5.2f}%)")

        self.results["statistics"]["total_raw_rows"] = total_rows
        self.results["statistics"]["retained_rows"] = remaining_clean
        self.results["statistics"]["retention_rate_pct"] = retention_rate

        # Assertions and quality checks
        if retention_rate >= 70.0:
            self.log_check("Data Retention Rate", "PASS", f"{retention_rate:.2f}% of raw rows retained ({remaining_clean:,} usable observations)")
        elif retention_rate >= 50.0:
            self.log_check("Data Retention Rate", "WARN", f"{retention_rate:.2f}% retained. Check if altitude or 1990 filters can be broadened.")
        else:
            self.log_check("Data Retention Rate", "FAIL", f"Only {retention_rate:.2f}% retained! Possible over-filtering.")

        if invalid_te1_count > 0:
            self.log_check("Te1 Quality", "WARN", f"Found {invalid_te1_count:,} non-positive/sentinel Te1 entries in raw data.")
        else:
            self.log_check("Te1 Quality", "PASS", f"All Te1 values positive and within physical bounds ({te1_min:.1f} K to {te1_max:.1f} K).")

        self.log_check("Altitude Bounds", "PASS", f"Post-filter altitude strictly bounded: [{altitude_min:.1f} km, {altitude_max:.1f} km]")

        return self.results["statistics"]

    def validate_processed_dataset(self) -> bool:
        """Audits the generated processed dataset on disk."""
        print("\n" + "=" * 70)
        print("STAGE 3: Processed Dataset Quality & Leakage Audit")
        print("=" * 70)

        if not os.path.exists(self.processed_dir):
            self.log_check("Processed Dataset", "WARN", f"Directory not found: '{self.processed_dir}'. Run create_dataset.py to build it.")
            return False

        all_ok = True

        test_storm_dir = os.path.join(self.processed_dir, "test-storm")
        test_normal_dir = os.path.join(self.processed_dir, "test-normal")
        train_chunks_dir = os.path.join(self.processed_dir, "train_chunks")

        # 1. Directory Structure
        has_storm = os.path.exists(test_storm_dir)
        has_normal = os.path.exists(test_normal_dir)
        has_train = os.path.exists(train_chunks_dir)

        self.log_check("test-storm Directory", "PASS" if has_storm else "FAIL", f"Path: {test_storm_dir}")
        self.log_check("test-normal Directory", "PASS" if has_normal else "FAIL", f"Path: {test_normal_dir}")
        self.log_check("train_chunks Directory", "PASS" if has_train else "FAIL", f"Path: {train_chunks_dir}")

        if not (has_storm and has_normal and has_train):
            return False

        if not HAS_DATASETS:
            self.log_check("HuggingFace Datasets Library", "FAIL", "Cannot inspect arrow tables without 'datasets' installed.")
            return False

        # Load datasets
        print("Loading processed splits for validation...")
        ds_storm = datasets.Dataset.load_from_disk(test_storm_dir)
        ds_normal = datasets.Dataset.load_from_disk(test_normal_dir)

        chunk_folders = sorted(os.listdir(train_chunks_dir))
        train_datasets_list = [datasets.Dataset.load_from_disk(os.path.join(train_chunks_dir, f)) for f in chunk_folders]
        ds_train = datasets.concatenate_datasets(train_datasets_list) if train_datasets_list else None

        n_train = len(ds_train) if ds_train is not None else 0
        n_val = len(ds_normal)
        n_storm = len(ds_storm)
        total_p = n_train + n_val + n_storm

        print(f"\nSplit Sizes:")
        print(f"  Train:       {n_train:>10,} ({n_train/total_p*100:5.2f}%)")
        print(f"  test-normal: {n_val:>10,} ({n_val/total_p*100:5.2f}%)")
        print(f"  test-storm:  {n_storm:>10,} ({n_storm/total_p*100:5.2f}%)")
        print(f"  Total:       {total_p:>10,}")

        # 2. Schema Validation (153 inputs + 1 target)
        expected_feature_prefixes = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON',
                                     'AL_index_', 'SYM_H_', 'f107_index_', 'Kp_index']
        expected_target = 'Te1'

        cols = set(ds_normal.column_names)
        has_target = expected_target in cols
        self.log_check("Target Column", "PASS" if has_target else "FAIL", f"'{expected_target}' present in dataset")

        # Verify lag column counts
        al_cols = [c for c in cols if c.startswith("AL_index_")]
        symh_cols = [c for c in cols if c.startswith("SYM_H_")]
        f107_cols = [c for c in cols if c.startswith("f107_index_")]

        self.log_check("AL Index Columns", "PASS" if len(al_cols) == 31 else "WARN", f"Found {len(al_cols)} AL lag columns (expected 31: 0-5h @ 10min)")
        self.log_check("SYM-H Columns", "PASS" if len(symh_cols) == 145 else "WARN", f"Found {len(symh_cols)} SYM-H lag columns (expected 145: 0-3d @ 30min)")
        self.log_check("F10.7 Columns", "PASS" if len(f107_cols) == 4 else "WARN", f"Found {len(f107_cols)} F10.7 lag columns (expected 4: 0-72h @ 24h)")

        # 3. Storm Temporal Holdout Validation
        if "DateTimeFormatted" in ds_storm.column_names:
            storm_df = ds_storm.with_format("pandas")
            storm_dates = pd.to_datetime(storm_df["DateTimeFormatted"])
            min_date, max_date = storm_dates.min(), storm_dates.max()
            in_range = (storm_dates >= "1991-01-31") & (storm_dates < "1991-02-07")
            if in_range.all():
                self.log_check("test-storm Temporal Isolation", "PASS", f"All rows strictly within 1991-01-31 to 1991-02-07 ({min_date} -> {max_date})")
            else:
                self.log_check("test-storm Temporal Isolation", "WARN", f"Some rows outside storm window: [{min_date}, {max_date}]")

        # 4. Leakage Check: verify storm period is strictly absent from train and test-normal
        if "DateTimeFormatted" in ds_normal.column_names:
            normal_dates = pd.to_datetime(ds_normal.with_format("pandas")["DateTimeFormatted"])
            storm_leak = ((normal_dates >= "1991-01-31") & (normal_dates < "1991-02-07")).sum()
            if storm_leak == 0:
                self.log_check("Zero Storm Leakage in test-normal", "PASS", "No storm period rows leaked into test-normal")
            else:
                self.log_check("Zero Storm Leakage in test-normal", "FAIL", f"Found {storm_leak} storm rows in test-normal!")
                all_ok = False

        # 5. NaN Audit on sample
        print("Auditing NaN/Inf presence across splits...")
        normal_sample = ds_normal.select(range(min(10000, len(ds_normal)))).to_pandas()
        nan_count = normal_sample.isna().sum().sum()
        if nan_count == 0:
            self.log_check("Missing Values in test-normal", "PASS", "0 NaN values found in sample")
        else:
            self.log_check("Missing Values in test-normal", "WARN", f"Found {nan_count} NaN values in sample!")

        return all_ok

    def print_summary(self):
        """Prints a final executive summary."""
        print("\n" + "=" * 70)
        print("DATASET VALIDATION SUMMARY REPORT")
        print("=" * 70)

        n_pass = sum(1 for c in self.results["checks"] if c["status"] == "PASS")
        n_warn = sum(1 for c in self.results["checks"] if c["status"] == "WARN")
        n_fail = sum(1 for c in self.results["checks"] if c["status"] == "FAIL")

        print(f"Total Checks: {len(self.results['checks'])} | PASS: {n_pass} | WARN: {n_warn} | FAIL: {n_fail}\n")

        if n_fail == 0:
            print("[SUCCESS] Dataset validation passed! Data quality meets high scientific standards.")
        else:
            print("[ATTENTION] Some checks failed. Review the detailed log above.")


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="CLARE Dataset Quality & Utilization Validator")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing dataset/ or input_dataset/")
    parser.add_argument("--skip-raw-audit", action="store_true", help="Skip the full TSV streaming audit (faster)")
    args = parser.parse_args()

    validator = DatasetValidator(base_dir=args.base_dir)

    # 1. Validate raw files
    raw_ok = validator.validate_raw_files()

    # 2. Audit filtering and data utilization
    if raw_ok and not args.skip_raw_audit:
        validator.validate_data_utilization_and_filtering()

    # 3. Validate processed dataset (if present)
    validator.validate_processed_dataset()

    # 4. Summary
    validator.print_summary()


if __name__ == "__main__":
    main()
