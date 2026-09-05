"""Unified physics-informed loss function balancing quiet-time plasmaspheric precision with geomagnetic storm shock resilience."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedPhysicsLoss(nn.Module):
    """Computes a multi-objective physics loss combining focal cross-entropy, expected Kelvin metric error, and ring-current weighting."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        bin_width_k: Optional[float] = None,
        bin_width: Optional[float] = None,
        bin_offset_k: float = 50.0,
        focal_gamma: float = 1.5,
        huber_weight: float = 0.5,
        huber_delta_k: Optional[float] = None,
        huber_delta: Optional[float] = None,
        storm_alpha: float = 1.5,
        sym_h_mean: float = -13.9137,
        sym_h_std: float = 22.0911,
        sym_h_feat_idx: Optional[int] = None,
        sym_h_feature_idx: Optional[int] = None,
        eps: float = 1e-7
    ):
        super().__init__()
        self.vocab_size = vocab_size or num_classes or 150
        self.bin_width_k = bin_width_k or bin_width or 100.0
        self.bin_offset_k = bin_offset_k
        self.focal_gamma = focal_gamma
        self.huber_weight = huber_weight
        self.huber_delta_k = huber_delta_k or huber_delta or 500.0
        self.storm_alpha = storm_alpha
        self.sym_h_mean = sym_h_mean
        self.sym_h_std = sym_h_std
        self.sym_h_feat_idx = sym_h_feat_idx if sym_h_feat_idx is not None else (sym_h_feature_idx if sym_h_feature_idx is not None else 39)
        self.eps = eps

        bin_centers = torch.arange(self.vocab_size, dtype=torch.float32) * self.bin_width_k + self.bin_offset_k
        self.register_buffer("bin_centers", bin_centers, persistent=False)

    def extract_sym_h(
        self,
        inputs: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        sym_h_unnorm: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Recovers physical ring current SYM-H index values in nanotesla from normalized telemetry tensors."""
        if sym_h_unnorm is not None:
            return sym_h_unnorm.float()
        active_features = features if features is not None else inputs
        if active_features is not None and active_features.dim() == 2 and active_features.size(1) > self.sym_h_feat_idx:
            normalized_sym_h = active_features[:, self.sym_h_feat_idx]
            return normalized_sym_h * self.sym_h_std + self.sym_h_mean
        return None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        sym_h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Evaluates joint focal classification loss and continuous expected temperature metric error across telemetry samples."""
        active_inputs = features if features is not None else inputs
        is_sequence_tensor = (logits.dim() == 3)
        if is_sequence_tensor:
            batch_dim, time_dim, vocab_dim = logits.shape
            logits = logits.reshape(-1, vocab_dim)
            targets = targets.reshape(-1)

        probabilities = F.softmax(logits.float(), dim=-1)
        log_probabilities = F.log_softmax(logits.float(), dim=-1)

        clamped_targets = targets.clamp(0, self.vocab_size - 1)
        target_probabilities = probabilities.gather(1, clamped_targets.unsqueeze(1)).squeeze(1).clamp(self.eps, 1.0 - self.eps)
        target_log_probabilities = log_probabilities.gather(1, clamped_targets.unsqueeze(1)).squeeze(1)

        focal_modulators = torch.pow(1.0 - target_probabilities, self.focal_gamma)
        focal_classification_loss = -focal_modulators * target_log_probabilities

        expected_temperatures = (probabilities * self.bin_centers.unsqueeze(0)).sum(dim=-1)
        true_temperatures = clamped_targets.float() * self.bin_width_k + self.bin_offset_k

        temperature_errors = expected_temperatures - true_temperatures
        absolute_errors = temperature_errors.abs()
        transition_delta = self.huber_delta_k

        quadratic_huber_loss = 0.5 * (temperature_errors / transition_delta).pow(2)
        linear_huber_loss = (absolute_errors / transition_delta) - 0.5
        metric_huber_loss = torch.where(absolute_errors <= transition_delta, quadratic_huber_loss, linear_huber_loss)

        samplewise_loss = focal_classification_loss + self.huber_weight * metric_huber_loss

        if not is_sequence_tensor:
            ring_current_sym_h = self.extract_sym_h(inputs=active_inputs, sym_h_unnorm=sym_h)
            if ring_current_sym_h is not None and self.storm_alpha > 0.0:
                geomagnetic_depression = torch.clamp(-ring_current_sym_h, min=0.0)
                raw_storm_weights = 1.0 + self.storm_alpha * (geomagnetic_depression / 100.0)
                batch_normalized_weights = raw_storm_weights / raw_storm_weights.mean().clamp(min=1e-5)
                return (batch_normalized_weights * samplewise_loss).mean()
        return samplewise_loss.mean()
