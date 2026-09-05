"""
tests/test_explainability.py - Unit tests for TEMPEST explainability and feature attribution.
"""

import unittest
import os
import shutil
import numpy as np
import torch

from train_v2 import TEMPEST
from explainability import (
    TempestExplainer,
    plot_global_feature_importance,
    plot_regime_comparison,
    plot_waterfall
)


class TestExplainability(unittest.TestCase):
    """Validates Path-Shapley attribution mechanics, efficiency axiom, and plotting routines."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.test_output_dir = "tests/test_output_explainability"
        os.makedirs(self.test_output_dir, exist_ok=True)

        self.num_features = 156
        self.feature_names = [f"feat_{i}" for i in range(self.num_features)]
        self.feature_names[-1] = "in_plasmapause"
        self.feature_names[0] = "Altitude"

        self.model = TEMPEST(
            num_features=self.num_features,
            d_model=32,
            expert_dim=64,
            num_experts=2,
            top_k=1,
            n_layers=1,
            n_hc=2,
            vocab_size=150
        )
        self.device = torch.device("cpu")
        self.explainer = TempestExplainer(
            model=self.model,
            feature_names=self.feature_names,
            device=self.device
        )

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_temperature_prediction_shapes_and_values(self):
        """Tests continuous temperature readout on tensors and numpy arrays."""
        x_tensor = torch.randn(4, self.num_features)
        t_tensor = self.explainer.predict_temperature_tensor(x_tensor)
        self.assertEqual(t_tensor.shape, (4,))
        self.assertTrue((t_tensor >= 50.0).all())

        x_np = np.random.randn(3, self.num_features).astype(np.float32)
        t_np = self.explainer.predict_temperature_numpy(x_np)
        self.assertEqual(t_np.shape, (3,))

    def test_path_shapley_efficiency_axiom(self):
        """
        Tests the Shapley Efficiency axiom:
            sum_i phi_i(x) == f(x) - f(baseline)
        Within numerical integration tolerance of Riemann sum on physical states.
        Avoids exact-zero baseline where RMSNorm unit-sphere projection exhibits a discontinuity.
        """
        baselines = torch.randn(1, self.num_features)
        inputs = baselines + 0.05 * torch.randn(2, self.num_features)

        attributions, base_val = self.explainer.compute_integrated_gradients(
            inputs, baselines=baselines, steps=50
        )

        self.assertEqual(attributions.shape, (2, self.num_features))
        sum_attributions = np.sum(attributions, axis=1)

        with torch.no_grad():
            preds = self.explainer.predict_temperature_tensor(inputs).cpu().numpy()

        diff = preds - base_val
        abs_err = np.abs(sum_attributions - diff)
        # Efficiency holds to high numerical accuracy (< 0.1 K or < 1% relative error)
        valid = (abs_err < 0.10) | ((abs_err / np.maximum(1e-4, np.abs(diff))) < 0.01)
        self.assertTrue(valid.all())

    def test_explain_interface(self):
        """Tests end-to-end explain() method output dictionary."""
        inputs = np.random.randn(3, self.num_features).astype(np.float32)
        background = np.zeros((5, self.num_features), dtype=np.float32)

        res = self.explainer.explain(inputs, background_samples=background, steps=10)
        self.assertIn("shap_values", res)
        self.assertIn("base_value", res)
        self.assertIn("predictions", res)
        self.assertEqual(res["shap_values"].shape, (3, self.num_features))

    def test_plotting_routines(self):
        """Tests generation and file saving of all explainability publication plots."""
        shap_vals = np.random.randn(10, self.num_features)
        feat_matrix = np.random.randn(10, self.num_features)
        feat_matrix[:, -1] = np.random.choice([0.0, 1.0], size=10)

        # 1. Global feature importance
        path_global = os.path.join(self.test_output_dir, "test_global.png")
        fig1 = plot_global_feature_importance(shap_vals, self.feature_names, top_n=10, save_path=path_global)
        self.assertTrue(os.path.exists(path_global))

        # 2. Regime comparison
        path_regime = os.path.join(self.test_output_dir, "test_regime.png")
        fig2 = plot_regime_comparison(shap_vals, feat_matrix, self.feature_names, top_n=5, save_path=path_regime)
        self.assertTrue(os.path.exists(path_regime))

        # 3. Waterfall plot
        path_waterfall = os.path.join(self.test_output_dir, "test_waterfall.png")
        fig3 = plot_waterfall(
            shap_sample=shap_vals[0],
            base_val=3000.0,
            pred_val=3500.0,
            feature_vals=feat_matrix[0],
            feature_names=self.feature_names,
            top_n=5,
            save_path=path_waterfall
        )
        self.assertTrue(os.path.exists(path_waterfall))
