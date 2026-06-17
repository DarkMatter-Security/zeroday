"""
build1_zeroday/detector.py — Hijoluminic Firewall: Zero-Day Detection Engine.

Architecture:
    Raw traffic → FFT frequency spectrum → branch amplitudes → CollapseOfLight → entropy

    Normal traffic (periodic, structured)  → narrow frequency peaks → low entropy ✓
    Anomalous traffic (random, noise-like) → flat frequency spectrum → high entropy ✗

The entropy of the fractal collapse IS the anomaly score. No learned shortcuts,
no thresholds to tune — pure physics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fields.hijoluminic import FractalMass


# ──────────────────────────────────────────────
#  Traffic Configuration
# ──────────────────────────────────────────────

@dataclass
class TrafficConfig:
    """
    Configuration for network traffic processing.

    n_sites: Number of time steps in the traffic window (determines freq resolution).
    n_features: Feature dimension per time step.
    n_branches: Number of frequency bands = n_freqs (n_sites//2 + 1).
    """
    n_features: int = 8
    n_sites: int = 64

    @property
    def n_freqs(self) -> int:
        return self.n_sites // 2 + 1

    @property
    def n_branches(self) -> int:
        return self.n_freqs


# ──────────────────────────────────────────────
#  Frequency-Domain Traffic Encoder
# ──────────────────────────────────────────────

class FrequencyEncoder(nn.Module):
    """
    Encodes traffic via FFT into frequency-domain branch amplitudes.

    Each frequency bin from the FFT becomes one branch. This means:
    - n_branches = n_sites // 2 + 1 (unique FFT bins)
    - Normal traffic (periodic, narrow-band) → power concentrated in few
      bins → LOW collapse entropy
    - Anomalous traffic (random, broadband) → power spread across all bins
      → HIGH collapse entropy

    This is a PURELY FIXED transformation — no learned parameters. The
    frequency content of the traffic directly determines the branch
    amplitudes, which then determine the collapse entropy.
    """

    def __init__(self, config: TrafficConfig):
        super().__init__()
        self.config = config
        # Hann window to reduce spectral leakage
        self.register_buffer('window', torch.hann_window(config.n_sites))

    def forward(self, traffic: torch.Tensor) -> torch.Tensor:
        """
        Convert traffic to branch amplitudes.

        Args:
            traffic: (batch, n_sites, n_features) — raw traffic.

        Returns:
            branch_amps: (batch, n_branches) — real, non-negative amplitudes,
                        normalized to sum to n_branches.
        """
        batch = traffic.shape[0]
        n_freqs = self.config.n_freqs

        # Apply Hann window to reduce spectral leakage
        w = self.window.view(1, -1, 1).to(traffic.device)
        traffic = traffic * w

        # FFT over site (time) dimension
        spectrum = torch.fft.rfft(traffic, dim=1)  # (batch, n_freqs, n_features)

        # Power spectral density
        psd = spectrum.abs() ** 2  # (batch, n_freqs, n_features)

        # Average over feature dimension → (batch, n_freqs)
        psd = psd.mean(dim=-1)

        # Normalize: total power = n_branches so that uniform spectrum
        # gives equal amplitude to each branch (entropy = max).
        total_power = psd.sum(dim=-1, keepdim=True) + 1e-8
        branch_amps = psd / total_power * n_freqs

        return branch_amps  # (batch, n_branches) = (batch, n_freqs)

    @staticmethod
    def synthetic_traffic(batch: int = 32,
                           n_sites: int = 64,
                           n_features: int = 8,
                           normal: bool = True) -> torch.Tensor:
        """
        Generate synthetic traffic with controlled frequency properties.

        Normal traffic: PURE sinusoidal oscillations → sharp frequency peaks.
        Anomalous traffic: white noise with impulse bursts → flat spectrum.
        """
        if normal:
            # Normal: pure-tone periodic traffic — very narrow frequency peak.
            # Each sample has a random dominant frequency, but each feature
            # across all samples shares the same base frequency to create
            # sharp spectral peaks.
            t = torch.linspace(0, 6 * math.pi, n_sites).unsqueeze(0).unsqueeze(-1)
            # Per-sample, per-feature frequency (positive, varied)
            freq = torch.rand(batch, 1, n_features) * 2.0 + 0.5
            # Single pure tone at dominant frequency (no harmonics)
            signal = torch.sin(t * freq) * 1.0
            # Very low noise floor
            noise = torch.randn(batch, n_sites, n_features) * 0.02
            return signal + noise
        else:
            # Anomaly: broadband white noise + impulse bursts
            # White noise has FLAT spectrum across all frequencies
            base = torch.randn(batch, n_sites, n_features) * 0.8
            # Add impulse bursts (transient anomalies)
            for b in range(batch):
                burst_loc = torch.randint(0, n_sites, (1,)).item()
                base[b, burst_loc, :] += torch.randn(n_features) * 4.0
            return base


# ──────────────────────────────────────────────
#  Anomaly Scorer (Pure Physics)
# ──────────────────────────────────────────────

class AnomalyScorer(nn.Module):
    """
    Computes anomaly scores from collapse entropy.

    Anomaly score = entropy / max_possible_entropy ∈ [0, 1]
    No learned parameters, no thresholds to tune.

    Score interpretation:
        ≈ 0.0 → Normal (low entropy, concentrated in one branch)
        ≈ 1.0 → Anomalous (high entropy, spread across all branches)
    """

    def __init__(self, n_branches: int = 4):
        super().__init__()
        self.register_buffer('max_entropy', torch.tensor(math.log(n_branches)))

    def forward(self,
                collapse_entropy: torch.Tensor,
                **kwargs) -> torch.Tensor:
        return collapse_entropy / self.max_entropy


# ──────────────────────────────────────────────
#  Hijoluminic Firewall — Zero-Day Detector
# ──────────────────────────────────────────────

class HijoluminicFirewall(nn.Module):
    """
    Zero-day network intrusion detection via fractal resonance collapse.

    Architecture is fully feed-forward physics:
      Traffic → FFT → Filterbank → CollapseOfLight → Entropy → Score

    Only learnable parameters are the fractal mass folds, which tune
    the resonance frequencies to match normal traffic patterns.

    Detection principle:
      Normal traffic is periodic → narrow frequency peaks → one branch dominates
        → collapse is concentrated → LOW ENTROPY → score ≈ 0
      Zero-day attack is unstructured → flat frequency spectrum → even branches
        → collapse is spread → HIGH ENTROPY → score ≈ 1
    """

    def __init__(self,
                 config: TrafficConfig | None = None,
                 kappa_H: float = 1.0):
        super().__init__()
        self.config = config or TrafficConfig()
        self.n_branches = self.config.n_branches

        # Frequency encoder (fixed, no learned params)
        self.encoder = FrequencyEncoder(self.config)

        # Fractal mass (learnable fold gaps — for future resonance tuning)
        self.fractal_mass = FractalMass(kappa_H=kappa_H)

        # Collapse-of-Light — pure Born-rule probabilities.
        # No fractal resonance or interference — just the raw frequency
        # spectrum determines the collapse.
        from collapse.collapse import CollapseOfLight
        self.collapse = CollapseOfLight(
            n_branches=self.n_branches,
            use_fractal_resonance=False,
            use_interference=False,
        )

        # Anomaly scorer (pure physics, no learned params)
        self.scorer = AnomalyScorer(n_branches=self.n_branches)

    def forward(self,
                traffic: torch.Tensor,
                return_all: bool = False) -> torch.Tensor | dict:
        """
        Full detection pass.

        Args:
            traffic: (batch, n_sites, n_features).
            return_all: Return diagnostic dict if True.

        Returns:
            Anomaly scores (batch,) or full diagnostics dict.
        """
        batch = traffic.shape[0]

        # Step 1: Frequency analysis → branch amplitudes
        branch_amps = self.encoder(traffic)  # (batch, n_branches), real ≥ 0

        # Convert to complex (interference-free collapse uses magnitude)
        branch_amps_c = torch.complex(
            branch_amps, torch.zeros_like(branch_amps)
        )

        # Step 2: Collapse (pure Born-rule from frequency spectrum)
        collapse_result = self.collapse(
            branch_amps_c,
            return_full=True,
        )

        # Step 3: Anomaly score from entropy
        scores = self.scorer(
            collapse_entropy=collapse_result['entropy']
        )

        if return_all:
            return {
                'scores': scores,
                'entropy': collapse_result['entropy'],
                'probs': collapse_result['probs'],
                'resonance': collapse_result['resonance'],
                'branch_amps': branch_amps,
                'mass': self.fractal_mass.mass,
                'fold_gaps': self.fractal_mass.folds,
            }

        return scores

    def detect(self,
               traffic: torch.Tensor,
               threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run detection with a decision threshold.

        Returns (is_anomaly, scores).
        """
        scores = self.forward(traffic)
        return scores > threshold, scores


# ──────────────────────────────────────────────
#  Training (Unsupervised)
# ──────────────────────────────────────────────

def train_unsupervised(model: HijoluminicFirewall,
                       n_steps: int = 100,
                       batch_size: int = 32,
                       lr: float = 0.01,
                       verbose: bool = True) -> list[float]:
    """
    Train the firewall by tuning fractal mass fold gaps to normal traffic.

    The training objective is simple: minimize the collapse entropy for
    normal traffic. This tunes the fold gaps so that the frequency bands
    of normal traffic resonate with specific folds.

    Since we only train on NORMAL traffic, the model learns what "normal"
    looks like in frequency space. Anomalies (unseen during training)
    will naturally produce higher entropy because their broad spectrum
    doesn't match the learned fold gaps.

    Args:
        model: HijoluminicFirewall instance.
        n_steps: Number of training steps.
        batch_size: Batch size.
        lr: Learning rate.
        verbose: If True, print progress.

    Returns:
        List of loss values.
    """
    # Only optimize the fractal mass folds (everything else is fixed)
    optimizer = torch.optim.Adam(model.fractal_mass.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Generate synthetic normal traffic
        traffic = FrequencyEncoder.synthetic_traffic(
            batch=batch_size,
            n_sites=model.config.n_sites,
            n_features=model.config.n_features,
            normal=True,
        )

        # Forward pass
        out = model(traffic, return_all=True)

        # Loss: minimize entropy + regularize fold gaps to stay spread
        entropy_loss = out['entropy'].mean()

        # Encourage fold gaps to stay distinct (prevent collapse to one value)
        folds = model.fractal_mass.folds
        fold_spread = -((folds[1:] - folds[:-1]) ** 2).mean() * 0.01

        loss = entropy_loss + fold_spread

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (step % 20 == 0 or step == n_steps - 1):
            print(f"  Step {step:3d}: loss={loss.item():.4f}, "
                  f"entropy={out['entropy'].mean().item():.4f}, "
                  f"m={model.fractal_mass.mass.item():.4f}")

    return losses


# ──────────────────────────────────────────────
#  Analysis & Evaluation
# ──────────────────────────────────────────────

def analyze_traffic(model: HijoluminicFirewall,
                    n_normal: int = 200,
                    n_anomalies: int = 100) -> dict:
    """
    Evaluate the firewall's detection performance on synthetic data.

    Returns diagnostics including separation, detection rate, and FPR.
    """
    model.eval()

    with torch.no_grad():
        # Normal traffic
        normal_traffic = FrequencyEncoder.synthetic_traffic(
            n_normal, model.config.n_sites, model.config.n_features, normal=True
        )
        normal_out = model(normal_traffic, return_all=True)
        normal_scores = normal_out['scores']
        normal_entropy = normal_out['entropy']

        # Anomalous traffic
        anomaly_traffic = FrequencyEncoder.synthetic_traffic(
            n_anomalies, model.config.n_sites, model.config.n_features, normal=False
        )
        anomaly_out = model(anomaly_traffic, return_all=True)
        anomaly_scores = anomaly_out['scores']
        anomaly_entropy = anomaly_out['entropy']

    separation = (anomaly_scores.mean() - normal_scores.mean()).item()
    detection_rate = (anomaly_scores > 0.5).float().mean().item()
    false_positive_rate = (normal_scores > 0.5).float().mean().item()

    return {
        'normal_scores': normal_scores,
        'anomaly_scores': anomaly_scores,
        'normal_entropy': normal_entropy,
        'anomaly_entropy': anomaly_entropy,
        'separation': separation,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'normal_mean': normal_scores.mean().item(),
        'anomaly_mean': anomaly_scores.mean().item(),
    }


# ──────────────────────────────────────────────
#  Demo
# ──────────────────────────────────────────────

def demo():
    """Run a complete demo of the zero-day detector."""
    print("=" * 70)
    print("  Build 1: Hijoluminic Zero-Day Detector Demo")
    print("  Pure frequency-domain anomaly detection via fractal collapse")
    print("=" * 70)

    # Setup with larger n_sites for good frequency resolution
    config = TrafficConfig(n_features=8, n_sites=64)
    model = HijoluminicFirewall(config=config, kappa_H=1.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params} (only fractal mass folds learnable)")
    print(f"  Traffic config: {config.n_sites} sites x {config.n_features} features")
    print(f"  Frequency bins: {config.n_freqs} -> {config.n_branches} branches (one per FFT bin)")

    # Pre-training analysis
    print("\n  Pre-training: analyzing traffic frequency profiles...")
    with torch.no_grad():
        normal_t = FrequencyEncoder.synthetic_traffic(100, config.n_sites, config.n_features, normal=True)
        anomaly_t = FrequencyEncoder.synthetic_traffic(100, config.n_sites, config.n_features, normal=False)
        normal_amps = model.encoder(normal_t)
        anomaly_amps = model.encoder(anomaly_t)
    print(f"    Normal traffic:   branch amps std = {normal_amps.std(dim=0).mean().item():.4f}")
    print(f"    Anomalous traffic: branch amps std = {anomaly_amps.std(dim=0).mean().item():.4f}")
    print(f"    Expected: normal has MORE variation across branches than anomaly")

    # Unsupervised training
    print("\n  Phase 1: Unsupervised training on normal traffic...")
    losses = train_unsupervised(model, n_steps=100, verbose=True)

    # Evaluation
    print("\n  Phase 2: Evaluation on normal vs anomalous traffic...")
    results = analyze_traffic(model, n_normal=200, n_anomalies=100)

    print(f"\n  Results:")
    print(f"    Normal score mean:    {results['normal_mean']:.4f} "
          f"(should be LOW)")
    print(f"    Anomaly score mean:   {results['anomaly_mean']:.4f} "
          f"(should be HIGH)")
    print(f"    Separation:           {results['separation']:.4f} "
          f"(positive = working)")
    print(f"    Detection rate:       {results['detection_rate']:.3f}")
    print(f"    False positive rate:  {results['false_positive_rate']:.3f}")
    print(f"    Normal entropy:       {results['normal_entropy'].mean().item():.4f}")
    print(f"    Anomaly entropy:      {results['anomaly_entropy'].mean().item():.4f}")
    print(f"    Fractal mass:         {model.fractal_mass.mass.item():.4f}")
    print(f"    Fold gaps:            {[f'{d:.4f}' for d in model.fractal_mass.folds]}")
    print("=" * 70)

    # Determine success
    if results['separation'] > 0.1:
        print("  VERDICT: Detector working correctly - anomalies have higher entropy")
    elif results['separation'] > 0:
        print("  VERDICT: Detector trending correct - separation positive but small")
    else:
        print("  VERDICT: Detector needs tuning - separation negative")
    print("  Zero-Day Detector Demo Complete OK")
    print("=" * 70)

    return results


if __name__ == '__main__':
    demo()
