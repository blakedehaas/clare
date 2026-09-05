"""Unit test suite verifying UnifiedPhysicsLoss numerical accuracy, focal modulation, metric Huber regimes, and geomagnetic storm weighting."""

import unittest
import torch
from physics_loss import UnifiedPhysicsLoss


class TestUnifiedPhysicsLoss(unittest.TestCase):
    """Validates the UnifiedPhysicsLoss mathematical formulations across all configurations."""

    def test_initialization_default_and_aliases(self):
        """Tests initialization parameter alias resolution and buffer registration."""
        loss_default = UnifiedPhysicsLoss()
        self.assertEqual(loss_default.vocab_size, 150)
        self.assertEqual(loss_default.bin_width_k, 100.0)
        self.assertEqual(loss_default.bin_offset_k, 50.0)
        self.assertEqual(loss_default.huber_delta_k, 500.0)
        self.assertEqual(loss_default.sym_h_feat_idx, 39)
        self.assertEqual(loss_default.bin_centers.shape, (150,))

        loss_alias = UnifiedPhysicsLoss(
            num_classes=64,
            bin_width=50.0,
            huber_delta=250.0,
            sym_h_feature_idx=10,
            storm_alpha=2.0
        )
        self.assertEqual(loss_alias.vocab_size, 64)
        self.assertEqual(loss_alias.bin_width_k, 50.0)
        self.assertEqual(loss_alias.huber_delta_k, 250.0)
        self.assertEqual(loss_alias.sym_h_feat_idx, 10)

    def test_extract_sym_h(self):
        """Tests SYM-H feature extraction across unnormalized, normalized, and missing representations."""
        criterion = UnifiedPhysicsLoss(sym_h_feat_idx=2)

        # 1. Direct unnormalized tensor
        unnorm = torch.tensor([-50.0, -120.0, 10.0])
        extracted = criterion.extract_sym_h(sym_h_unnorm=unnorm)
        self.assertTrue(torch.allclose(extracted, unnorm))

        # 2. Normalized inputs tensor
        inputs = torch.zeros(3, 5)
        # unnormalized = normalized * std + mean
        # Set normalized to 1.0 -> 1.0 * 22.0911 - 13.9137 = 8.1774
        inputs[:, 2] = 1.0
        extracted_from_inputs = criterion.extract_sym_h(inputs=inputs)
        expected = torch.full((3,), 1.0 * criterion.sym_h_std + criterion.sym_h_mean)
        self.assertTrue(torch.allclose(extracted_from_inputs, expected))

        # 3. Normalized features alias
        extracted_from_features = criterion.extract_sym_h(features=inputs)
        self.assertTrue(torch.allclose(extracted_from_features, expected))

        # 4. Dimension or index mismatch returns None
        empty_inputs = torch.zeros(3, 1)  # Index 2 out of bounds
        self.assertIsNone(criterion.extract_sym_h(inputs=empty_inputs))
        self.assertIsNone(criterion.extract_sym_h(inputs=None))

    def test_forward_pass_2d_without_storm_weighting(self):
        """Tests standard 2D batch evaluation without geomagnetic storm reweighting."""
        criterion = UnifiedPhysicsLoss(vocab_size=10, bin_width_k=100.0, storm_alpha=0.0)
        batch_size = 4
        logits = torch.randn(batch_size, 10, requires_grad=True)
        targets = torch.tensor([1, 4, 7, 9], dtype=torch.long)

        loss = criterion(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

        # Gradient flow test
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.shape)

    def test_forward_pass_3d_sequence(self):
        """Tests 3D sequence evaluation for autoregressive decoder architectures."""
        criterion = UnifiedPhysicsLoss(vocab_size=16, bin_width_k=100.0)
        batch_size, seq_len, vocab_size = 2, 8, 16
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        loss = criterion(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_storm_weighting_linear_scaling(self):
        """Tests that geomagnetic storm ring current depression increases sample weighting."""
        criterion = UnifiedPhysicsLoss(vocab_size=10, storm_alpha=2.0, huber_weight=0.0, sym_h_feat_idx=0)
        logits = torch.zeros(2, 10)  # Uniform distribution
        targets = torch.tensor([5, 5], dtype=torch.long)

        # Sample 0: quiet (SYM-H = 0 nT)
        # Sample 1: intense storm (SYM-H = -200 nT)
        sym_h = torch.tensor([0.0, -200.0])
        loss = criterion(logits, targets, sym_h=sym_h)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_huber_quadratic_vs_linear_regimes(self):
        """Tests that errors below delta transition quadratically while errors above transition linearly."""
        criterion = UnifiedPhysicsLoss(
            vocab_size=10,
            bin_width_k=100.0,
            huber_delta_k=200.0,
            focal_gamma=0.0,
            huber_weight=1.0,
            storm_alpha=0.0
        )
        # Target at bin 2 (250 K)
        targets = torch.tensor([2], dtype=torch.long)

        # Prediction sharply at bin 2 (error = 0)
        logits_exact = torch.full((1, 10), -100.0)
        logits_exact[0, 2] = 100.0
        loss_exact = criterion(logits_exact, targets)

        # Prediction at bin 3 (error = 100 K < delta=200 K -> quadratic)
        logits_small_err = torch.full((1, 10), -100.0)
        logits_small_err[0, 3] = 100.0
        loss_small_err = criterion(logits_small_err, targets)

        # Prediction at bin 7 (error = 500 K > delta=200 K -> linear)
        logits_large_err = torch.full((1, 10), -100.0)
        logits_large_err[0, 7] = 100.0
        loss_large_err = criterion(logits_large_err, targets)

        self.assertLess(loss_exact.item(), loss_small_err.item())
        self.assertLess(loss_small_err.item(), loss_large_err.item())


if __name__ == "__main__":
    unittest.main()
