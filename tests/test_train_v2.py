"""Unit test suite verifying DeepSeek-V4 MoE, mHC Sinkhorn-Knopp doubly stochastic projection, attention sink, and evaluation routines in train_v2.py."""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_v2 import (
    RMSNorm,
    SinkhornKnoppBirkhoff,
    ManifoldHyperConnection,
    DeepSeekMoE,
    AttentionWithSink,
    TransformerBlock,
    TEMPEST,
    SpaceWeatherDeepSeekV2,
    evaluate_model,
    build_preprocessor,
    generate_physical_diagnostics,
    NUM_INPUT_FEATURES
)
from physics_loss import UnifiedPhysicsLoss


class TestTrainV2(unittest.TestCase):
    """Validates the deep learning architecture and mathematical mechanics of train_v2.py."""

    def test_rmsnorm_invariance_and_shape(self):
        """Tests RMSNorm forward normalization and learnable gain scaling."""
        norm = RMSNorm(dim=32)
        x = torch.randn(4, 32) * 5.0 + 2.0
        out = norm(x)

        self.assertEqual(out.shape, x.shape)
        # Root mean square should be close to 1.0
        rms = torch.sqrt(out.pow(2).mean(dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-3))

    def test_sinkhorn_knopp_birkhoff_doubly_stochastic(self):
        """Tests that Sinkhorn-Knopp projects matrices into the Birkhoff Polytope (doubly stochastic)."""
        torch.manual_seed(42)
        sinkhorn = SinkhornKnoppBirkhoff(n_iters=30, tau=0.5)
        batch_size, n_hc = 4, 4
        unconstrained_matrix = torch.randn(batch_size, n_hc, n_hc)

        doubly_stochastic = sinkhorn(unconstrained_matrix)

        self.assertEqual(doubly_stochastic.shape, (batch_size, n_hc, n_hc))
        # Elements must be non-negative
        self.assertTrue((doubly_stochastic >= 0.0).all())

        # Row sums must equal 1
        row_sums = doubly_stochastic.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=5e-3))

        # Column sums must equal 1
        col_sums = doubly_stochastic.sum(dim=-2)
        self.assertTrue(torch.allclose(col_sums, torch.ones_like(col_sums), atol=5e-3))

    def test_manifold_hyper_connection(self):
        """Tests residual stream width expansion and dynamic mixing in ManifoldHyperConnection."""
        d_model, n_hc = 64, 4
        sublayer = nn.Linear(d_model, d_model)
        mhc = ManifoldHyperConnection(d_model=d_model, n_hc=n_hc)

        batch_size = 3
        stream_in = torch.randn(batch_size, n_hc, d_model)
        stream_out, aux = mhc(stream_in, sublayer)

        self.assertEqual(stream_out.shape, (batch_size, n_hc, d_model))
        self.assertIsNone(aux)

    def test_deepseek_moe_routing_and_balance_loss(self):
        """Tests DeepSeekMoE top-k routing, SwiGLU clamping, and load balancing loss."""
        d_model, expert_dim, num_experts, top_k = 64, 128, 6, 2
        moe = DeepSeekMoE(d_model=d_model, expert_dim=expert_dim, num_experts=num_experts, top_k=top_k)

        batch_size = 5
        x = torch.randn(batch_size, d_model)
        out, balance_loss = moe.forward_with_aux(x)

        self.assertEqual(out.shape, (batch_size, d_model))
        self.assertTrue(torch.is_tensor(balance_loss))
        self.assertGreaterEqual(balance_loss.item(), 0.0)

        # Direct forward
        out_direct = moe(x)
        self.assertEqual(out_direct.shape, (batch_size, d_model))

    def test_attention_with_sink(self):
        """Tests Multi-Head Attention with Attention Sink logit modulation."""
        d_model, n_heads = 64, 4
        attn = AttentionWithSink(d_model=d_model, n_heads=n_heads, dropout=0.0)

        batch_size = 4
        x = torch.randn(batch_size, d_model)
        out = attn(x)

        self.assertEqual(out.shape, (batch_size, d_model))

    def test_transformer_block(self):
        """Tests TransformerBlock combining Attention, DeepSeekMoE, and dual mHC streams."""
        d_model, expert_dim, num_experts, top_k, n_hc = 64, 128, 4, 2, 4
        block = TransformerBlock(d_model=d_model, expert_dim=expert_dim, num_experts=num_experts, top_k=top_k, n_hc=n_hc)

        batch_size = 2
        s_in = torch.randn(batch_size, n_hc, d_model)
        s_out, aux = block(s_in)

        self.assertEqual(s_out.shape, (batch_size, n_hc, d_model))
        self.assertIsNotNone(aux)

    def test_tempest_model(self):
        """Tests full TEMPEST model forward, parameter counts, alias, and temperature readout."""
        self.assertIs(SpaceWeatherDeepSeekV2, TEMPEST)
        num_features = 156
        vocab_size = 150
        model = TEMPEST(
            num_features=num_features,
            d_model=64,
            expert_dim=128,
            num_experts=4,
            top_k=2,
            n_layers=2,
            n_hc=4,
            vocab_size=vocab_size
        )

        total_p, active_p = model.count_parameters()
        self.assertGreater(total_p, active_p)
        self.assertGreater(active_p, 0)

        batch_size = 4
        inputs = torch.randn(batch_size, num_features)

        # Standard forward
        logits = model(inputs)
        self.assertEqual(logits.shape, (batch_size, vocab_size))

        # Forward returning auxiliary load balancing loss
        logits_aux, aux_loss = model(inputs, return_aux=True)
        self.assertEqual(logits_aux.shape, (batch_size, vocab_size))
        self.assertTrue(torch.is_tensor(aux_loss))

        # Temperature reconstruction
        expected_temps = model.predict_temperature(inputs, method="expected")
        self.assertEqual(expected_temps.shape, (batch_size,))
        self.assertTrue((expected_temps >= 0.0).all())

        argmax_temps = model.predict_temperature(inputs, method="argmax")
        self.assertEqual(argmax_temps.shape, (batch_size,))

    def test_evaluate_model_routine(self):
        """Tests evaluate_model metrics aggregation over mock DataLoader batches."""
        model = TEMPEST(num_features=156, d_model=32, expert_dim=64, num_experts=2, top_k=1, n_layers=1, n_hc=2)
        device = torch.device("cpu")
        criterion = UnifiedPhysicsLoss(vocab_size=150)

        # Mock dataset with input_ids and label
        mock_data = [
            {"input_ids": torch.randn(8, 156), "label": torch.randint(10, 50, (8,))}
            for _ in range(3)
        ]

        metrics = evaluate_model(model, mock_data, criterion, device)
        self.assertIn("loss", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("acc", metrics)

        # Empty loader handling
        empty_metrics = evaluate_model(model, [], criterion, device)
        self.assertEqual(empty_metrics["loss"], 0.0)

    def test_generate_physical_diagnostics(self):
        """Tests physical diagnostics plot generation across residual and plasmapause dimensions."""
        model = TEMPEST(num_features=156, d_model=32, expert_dim=64, num_experts=2, top_k=1, n_layers=1, n_hc=2)
        device = torch.device("cpu")

        # Mock test batches
        mock_loader = [
            {"input_ids": torch.randn(10, 156), "label": torch.randint(20, 60, (10,))}
            for _ in range(2)
        ]

        # Use tests/test_output to verify generation without polluting repo
        saved = generate_physical_diagnostics(model, mock_loader, device, output_dir="tests/test_output", model_name="test_tempest")
        self.assertIn("deviation", saved)
        self.assertIn("scatter", saved)
        self.assertIn("plasmapause", saved)

    def test_build_preprocessor(self):
        """Tests vectorize batch preprocessor feature engineering and L-shell computation."""
        means = {f"AL_index": -100.0, f"SYM_H": -15.0, f"f107_index": 120.0}
        stds = {f"AL_index": 150.0, f"SYM_H": 25.0, f"f107_index": 40.0}

        preprocessor = build_preprocessor(means, stds)

        # Construct minimal valid batch dictionary
        batch = {
            "Altitude": [2000.0, 4000.0],
            "ILAT": [45.0, 60.0],
            "Kp_index": [30.0, 10.0],
            "Te1": [2500.0, 3800.0],
            "GCLAT": [40.0, 55.0],
            "GCLON": [120.0, 240.0],
            "GLAT": [42.0, 58.0],
            "GMLT": [12.0, 18.0],
            "XXLAT": [40.0, 55.0],
            "XXLON": [120.0, 240.0],
        }
        for i in range(31):
            batch[f"AL_index_{i}"] = [-50.0, -200.0]
        for i in range(145):
            batch[f"SYM_H_{i}"] = [-10.0, -80.0]
        for i in range(4):
            batch[f"f107_index_{i}"] = [110.0, 140.0]

        processed = preprocessor(batch)
        self.assertIn("input_ids", processed)
        self.assertIn("label", processed)
        self.assertEqual(processed["input_ids"].shape, (2, NUM_INPUT_FEATURES))
        self.assertEqual(processed["label"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
