"""
build6_bci/cortex.py — Hijoluminic Neural Cortex: BCI Neural Superposition Model.

This module implements a full Brain-Computer Interface model that treats
neural recordings as a field-theoretic system, where cognitive states exist
as quantum-like superpositions of multiple interpretations.

The architecture maps biological neural data onto the H-ANS field:

    • Electrodes → Sites (n_sites = number of sensors)
    • Frequency bands → Vibrational bands (ρ_y: power in band y)
    • Phase coherence → Scalar potential Φ
    • Functional connectivity → Gauge connection A_μ
    • Cognitive states → Superposition branches
    • Decision/response → Collapse-of-Light

Key scientific prediction:
    During ambiguous sensory input (e.g., Necker cube, binocular rivalry),
    the model maintains a stable, high-entropy superposition with two
    dominant branches BEFORE collapse — mimicking the brain's ambiguity
    resolution. This is a testable neural-superposition hypothesis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
from typing import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import HijoluminicANS
from core.algebra import complexified_velocity, gamma_matrices
from fields.hijoluminic import FractalMass, VibrationalDensity, ScalarPotential
from collapse.collapse import CollapseOfLight


# ──────────────────────────────────────────────
#  Neural Data Configuration
# ──────────────────────────────────────────────

@dataclass
class NeuralCortexConfig:
    """
    Configuration for the neural recording setup.

    Mimics real ECoG/MEG arrays with configurable sensor count,
    frequency bands, and time windows.
    """
    n_electrodes: int = 64       # Number of sensors/sites
    n_frequency_bands: int = 5   # δ, θ, α, β, γ
    n_time_windows: int = 16     # Temporal context window
    n_features_per_site: int = 32  # Feature dimension per electrode
    sampling_rate: float = 1000.0  # Hz

    # Standard frequency bands (Hz)
    frequency_bands: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.5, 4.0),    # δ
        (4.0, 8.0),    # θ
        (8.0, 13.0),   # α
        (13.0, 30.0),  # β
        (30.0, 100.0), # γ
    ])


# ──────────────────────────────────────────────
#  Neural Data Encoder
# ──────────────────────────────────────────────

class NeuralDataEncoder(nn.Module):
    """
    Encodes raw neural recordings into the H-ANS field representation.

    For each electrode/site, computes:
      - ρ_y: Band-limited power in each frequency band y
      - Φ: Phase coherence between neighboring electrodes
      - Raw feature embedding for HijoluminicOperator

    Maps (batch, n_electrodes, n_time_windows) →
        (batch, n_electrodes, n_features)
    """

    def __init__(self, config: NeuralCortexConfig):
        super().__init__()
        self.config = config

        # Spectral power computation (learned filter bank)
        # In a real system, this would be a wavelet filter bank or STFT
        self.spectral_filters = nn.Parameter(
            torch.randn(config.n_frequency_bands, config.n_time_windows) * 0.1
        )

        # Phase coherence network
        self.coherence_net = nn.Sequential(
            nn.Linear(config.n_electrodes * config.n_time_windows, 128),
            nn.Tanh(),
            nn.Linear(128, config.n_electrodes * config.n_electrodes),
        )

        # Final feature embedding
        n_raw_features = (config.n_frequency_bands +  # spectral powers
                          config.n_electrodes +        # coherence features
                          config.n_time_windows)       # raw temporal
        self.feature_proj = nn.Linear(n_raw_features, config.n_features_per_site)

    def compute_spectral_power(self,
                               eeg: torch.Tensor) -> torch.Tensor:
        """
        Compute band-limited power at each electrode.

        Args:
            eeg: (batch, n_electrodes, n_time_windows) — raw neural signal.

        Returns:
            ρ_y: (batch, n_electrodes, n_frequency_bands) — power in each band.
        """
        # Simulate spectral decomposition via learned filter responses
        # In production: use actual STFT -> bandpower computation
        eeg_flat = eeg  # (B, E, T)
        filters = self.spectral_filters  # (Bands, T)

        # Convolve each electrode's signal with each bandpass filter
        # Simpler: use filter bank as learned linear projection
        powers = []
        for band_idx in range(self.config.n_frequency_bands):
            filt = filters[band_idx]  # (T,)
            # "Filtered" signal: approximate via inner product
            filtered = (eeg_flat * filt.unsqueeze(0).unsqueeze(1)).sum(dim=-1)  # (B, E)
            power = filtered ** 2
            powers.append(power)

        rho = torch.stack(powers, dim=-1)  # (B, E, Bands)
        # Non-negative via softplus
        rho = F.softplus(rho)
        return rho

    def compute_phase_coherence(self,
                                eeg: torch.Tensor) -> torch.Tensor:
        """
        Compute phase coherence matrix between electrode pairs.

        Args:
            eeg: (batch, n_electrodes, n_time_windows).

        Returns:
            Φ: (batch, n_electrodes, n_electrodes) — phase locking values.
        """
        batch, E, T = eeg.shape

        # Hilbert transform approximation via learned projection
        hilbert_proj = nn.Linear(T, T, device=eeg.device, bias=False)
        analytic = hilbert_proj(eeg)  # (B, E, T)
        # Phase = arctan(imag/real) of analytic signal
        # Since we're working with real-valued signals, we approximate
        # by using the signal and its derivative

        # Simple phase estimate via instantaneous frequency
        phase = torch.atan2(
            torch.roll(eeg, shifts=-1, dims=-1) - torch.roll(eeg, shifts=1, dims=-1),
            eeg + 1e-8
        )  # (B, E, T)

        # Phase coherence = |mean(exp(iΔφ))| over time
        phase_diff = phase.unsqueeze(2) - phase.unsqueeze(1)  # (B, E, E, T)
        coherence = torch.abs(
            torch.mean(torch.exp(1j * phase_diff), dim=-1)
        )  # (B, E, E)

        return coherence

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Encode raw neural data into site features.

        Args:
            eeg: (batch, n_electrodes, n_time_windows) — raw neural signal.

        Returns:
            Site features: (batch, n_electrodes, n_features_per_site)
        """
        batch, E, T = eeg.shape

        # Spectral power
        rho = self.compute_spectral_power(eeg)  # (B, E, Bands)

        # Phase coherence
        coherence = self.compute_phase_coherence(eeg)  # (B, E, E)

        # Flatten coherence (per electrode: coherence with all others)
        coherence_feats = coherence  # (B, E, E) — each electrode gets its coherence row

        # Raw temporal features (downsampled)
        raw_feats = eeg  # (B, E, T)

        # Concatenate and project
        combined = torch.cat([
            rho,                    # (B, E, Bands)
            coherence_feats,        # (B, E, E)
            raw_feats,              # (B, E, T)
        ], dim=-1)

        features = self.feature_proj(combined)  # (B, E, n_features)
        return features


# ──────────────────────────────────────────────
#  Stimulus Representations
# ──────────────────────────────────────────────

@dataclass
class Stimulus:
    """
    A sensory stimulus presented to the virtual brain.
    """
    name: str
    is_ambiguous: bool = False
    n_interpretations: int = 1


# Standard stimuli for testing
STIMULI = {
    'clear_image': Stimulus('clear_image', is_ambiguous=False, n_interpretations=1),
    'rubin_vase': Stimulus('rubin_vase', is_ambiguous=True, n_interpretations=2),
    'necker_cube': Stimulus('necker_cube', is_ambiguous=True, n_interpretations=2),
    'binocular_rivalry': Stimulus('binocular_rivalry', is_ambiguous=True, n_interpretations=2),
    'noise': Stimulus('noise', is_ambiguous=True, n_interpretations=4),
}


def generate_neural_response(eeg_shape: tuple,
                              stimulus: str = 'clear_image',
                              noise_level: float = 0.1) -> torch.Tensor:
    """
    Generate a synthetic neural response to a stimulus.

    This simulates the kind of signal that would be recorded from an
    ECoG/MEG array when a subject is presented with a visual stimulus.

    The key manipulation: ambiguous stimuli produce EEG patterns that have
    TWO equally strong frequency components (simulating two competing
    interpretations), while clear stimuli have ONE dominant component.

    Args:
        eeg_shape: (batch, n_electrodes, n_time_windows).
        stimulus: Type of stimulus.
        noise_level: Amount of neural noise.

    Returns:
        Synthetic EEG tensor.
    """
    batch, E, T = eeg_shape
    device = 'cpu'

    t = torch.linspace(0, 1, T, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    if stimulus == 'clear_image':
        # Single dominant frequency (one interpretation)
        freq = 10.0  # α band
        phase = torch.randn(batch, E, 1, device=device)
        signal = torch.sin(2 * math.pi * freq * t + phase)
        power = 1.0

    elif stimulus in ('rubin_vase', 'necker_cube'):
        # Two competing frequencies (ambiguous — face/vase or cube orientation)
        freq1, freq2 = 8.0, 12.0  # Two α-band peaks
        phase1 = torch.randn(batch, E, 1, device=device)
        phase2 = torch.randn(batch, E, 1, device=device)

        signal1 = torch.sin(2 * math.pi * freq1 * t + phase1)
        signal2 = torch.sin(2 * math.pi * freq2 * t + phase2)

        # Equal power for both interpretations
        power1 = torch.ones(batch, E, 1, device=device) * 0.5
        power2 = torch.ones(batch, E, 1, device=device) * 0.5

        # Per-sample random dominance (some samples may lean one way)
        dominance = torch.sigmoid(torch.randn(batch, E, 1, device=device) * 2)
        signal = (1 - dominance) * signal1 + dominance * signal2
        # Mix in equal proportion overall
        signal = 0.5 * signal1 + 0.5 * signal2 + 0.1 * torch.randn_like(signal1)
        power = 0.5

    elif stimulus == 'binocular_rivalry':
        # Left vs right eye dominance
        freq_L, freq_R = 8.0, 15.0
        phase_L = torch.randn(batch, E, 1, device=device)
        phase_R = torch.randn(batch, E, 1, device=device)
        signal_L = torch.sin(2 * math.pi * freq_L * t + phase_L)
        signal_R = torch.sin(2 * math.pi * freq_R * t + phase_R)
        signal = signal_L + signal_R + 0.2 * torch.randn_like(signal_L)
        power = 0.5

    elif stimulus == 'noise':
        # Broadband noise — many possible interpretations
        signal = torch.randn(batch, E, T, device=device) * 2.0
        power = 0.1

    else:
        raise ValueError(f"Unknown stimulus: {stimulus}")

    # Add noise
    noise = torch.randn(batch, E, T, device=device) * noise_level
    eeg = signal + noise

    # Normalize per electrode
    eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + 1e-8)

    return eeg.to(torch.float32)


# ──────────────────────────────────────────────
#  Hijoluminic Cortex — Full BCI Model
# ──────────────────────────────────────────────

class HijoluminicCortex(nn.Module):
    """
    The Hijoluminic Neural Cortex — a BCI model that treats brain activity
    as a field in quantum superposition.

    This model maps neural recordings onto the H-ANS architecture and:
      1. Maintains multiple interpretations of sensory input in superposition
      2. Uses collapse entropy as a measure of perceptual uncertainty
      3. Predicts behavioral responses from the collapsed state
      4. Represents "ambiguous" vs "clear" perception via branch structure

    Scientific predictions:
      - Ambiguous stimuli → stable superposition with entropy ≈ log(2)
      - Clear stimuli → collapsed state with entropy ≈ 0
      - Reaction time inversely correlated with collapse certainty
      - The complex mass's imaginary part predicts perceptual switching rate
    """

    def __init__(self,
                 config: NeuralCortexConfig | None = None,
                 n_interpretations: int = 4,
                 hidden_dim: int = 128,
                 n_cortical_layers: int = 3,
                 kappa_H: float = 1.0,
                 hbar: float = 1.0):
        """
        Args:
            config: Neural recording configuration.
            n_interpretations: Number of cognitive interpretations (branches).
            hidden_dim: Hidden dimension for cortical processing.
            n_cortical_layers: Number of HijoluminicOperator layers.
            kappa_H: Hijoluminic coupling scale.
            hbar: Reduced Planck constant.
        """
        super().__init__()
        self.config = config or NeuralCortexConfig()
        self.n_interpretations = n_interpretations
        self.hidden_dim = hidden_dim
        self.n_cortical_layers = n_cortical_layers

        # ── Neural encoder ──
        self.encoder = NeuralDataEncoder(self.config)

        # ── H-ANS Core ──
        self.hans = HijoluminicANS(
            in_features=self.config.n_features_per_site,
            n_sites=self.config.n_electrodes,
            n_classes=n_interpretations,  # one class per interpretation
            n_bands=self.config.n_frequency_bands,
            n_branches=n_interpretations,
            hidden_dim=hidden_dim,
            n_layers=n_cortical_layers,
            mode='full',
            kappa_H=kappa_H,
            hbar=hbar,
        )

        # ── Perceptual decision head ──
        # Maps the collapsed state to a behavioral response
        # psi_final is complex (batch, n_electrodes, 4) -> flat -> real+imag concat
        n_psi_features = 8 * self.config.n_electrodes
        self.response_net = nn.Sequential(
            nn.Linear(n_interpretations + n_psi_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_interpretations),
        )

        # ── Interpretability: what each branch represents ──
        self.branch_labels = [
            f"Interpretation_{i}" for i in range(n_interpretations)
        ]

    def forward(self,
                eeg: torch.Tensor,
                return_all: bool = False) -> torch.Tensor | dict:
        """
        Full forward pass: neural data → perceptual state → decision.

        Args:
            eeg: (batch, n_electrodes, n_time_windows) — raw neural data.
            return_all: If True, return full diagnostic dict.

        Returns:
            If return_all:
                Dict with:
                    'interpretation_logits': (batch, n_interpretations).
                    'interpretation_probs': (batch, n_interpretations).
                    'collapse_entropy': (batch,) — perceptual uncertainty.
                    'superposition_amplitudes': (batch, n_interpretations) — complex.
                    'mass': Fractal mass value.
                    'fold_gaps': The four Δ_f.
                    'psi_final': Final spinor field.
                    'behavioral_response': Predicted response.
                    'is_ambiguous': Whether >1 branch has significant probability.
            Else:
                Interpretation logits (batch, n_interpretations).
        """
        # Encode neural data
        features = self.encoder(eeg)  # (batch, n_electrodes, n_features)

        # H-ANS forward
        hans_out = self.hans(features, return_all=True)

        # Extract collapse probs and branches
        collapse_probs = hans_out['collapse_probs']  # (batch, n_interpretations)
        branch_amps = hans_out['branch_amplitudes']  # (batch, n_interpretations) complex
        psi_final = hans_out['psi_final']  # (batch, n_sites, 4)
        entropy = hans_out['collapse_entropy']  # (batch,)

        # Behavioral response (convert complex psi to real features)
        psi_flat = psi_final.reshape(eeg.shape[0], -1)  # complex
        psi_real = torch.cat([psi_flat.real, psi_flat.imag], dim=-1)
        response_features = torch.cat([collapse_probs, psi_real], dim=-1)
        response_logits = self.response_net(response_features)

        # Determine ambiguity: if any two branches have prob > 0.2
        sorted_probs = collapse_probs.sort(dim=-1, descending=True).values
        is_ambiguous = sorted_probs[:, 1] > 0.2  # second-highest prob matters

        if return_all:
            return {
                'interpretation_logits': response_logits,
                'interpretation_probs': F.softmax(response_logits, dim=-1),
                'collapse_entropy': entropy,
                'collapse_probs': collapse_probs,
                'superposition_amplitudes': branch_amps,
                'mass': self.hans.fractal_mass.mass,
                'fold_gaps': self.hans.fractal_mass.folds,
                'psi_final': psi_final,
                'behavioural_response': response_logits.argmax(dim=-1),
                'is_ambiguous': is_ambiguous,
                'features': features,
            }

        return response_logits

    def analyze_perception(self, eeg: torch.Tensor) -> dict:
        """
        Analyze the perceptual state evoked by a neural recording.

        This is the key scientific analysis: it reveals whether the
        model perceives the input as ambiguous or clear, and if so,
        which interpretations are in superposition.

        Args:
            eeg: (batch, n_electrodes, n_time_windows).

        Returns:
            Dict with perceptual analysis.
        """
        with torch.no_grad():
            out = self.forward(eeg, return_all=True)

        probs = out['interpretation_probs']
        entropy = out['collapse_entropy']
        max_entropy = math.log(self.n_interpretations)
        normalized_entropy = entropy / max_entropy

        # Dominant and second-dominant interpretations
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        dominant = sorted_idx[:, 0]
        second = sorted_idx[:, 1]
        gap = sorted_probs[:, 0] - sorted_probs[:, 1]

        return {
            'is_ambiguous': out['is_ambiguous'],
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'dominant_interpretation': dominant,
            'second_interpretation': second,
            'probability_gap': gap,
            'collapse_probs': probs,
            'interpretation_probs': out['interpretation_probs'],
            'mass': out['mass'],
            'fold_gaps': out['fold_gaps'],
        }


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────

def train_cortex(model: HijoluminicCortex,
                 n_steps: int = 200,
                 batch_size: int = 16,
                 lr: float = 1e-3,
                 verbose: bool = True) -> list[float]:
    """
    Train the cortical model on a perceptual decision task.

    The model learns to:
      - Collapse to the correct interpretation for clear stimuli
      - Maintain superposition for ambiguous stimuli
      - Report its uncertainty via the collapse entropy

    Args:
        model: HijoluminicCortex instance.
        n_steps: Training steps.
        batch_size: Batch size.
        lr: Learning rate.
        verbose: If True, print progress.

    Returns:
        List of loss values.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Generate mixed batch: clear + ambiguous stimuli
        n_clear = batch_size // 2
        n_ambig = batch_size - n_clear

        # Clear stimuli
        clear_eeg = generate_neural_response(
            (n_clear, model.config.n_electrodes, model.config.n_time_windows),
            stimulus='clear_image',
        )
        clear_labels = torch.zeros(n_clear, dtype=torch.long)  # interpretation 0

        # Ambiguous stimuli — both interpretations are "correct"
        ambig_eeg = generate_neural_response(
            (n_ambig, model.config.n_electrodes, model.config.n_time_windows),
            stimulus='rubin_vase',
        )
        ambig_labels = torch.ones(n_ambig, dtype=torch.long)  # interpretation 1

        # Combine
        eeg = torch.cat([clear_eeg, ambig_eeg], dim=0)
        labels = torch.cat([clear_labels, ambig_labels], dim=0)

        # Forward
        out = model(eeg, return_all=True)
        logits = out['interpretation_logits']

        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)

        # Entropy regularization: clear stimuli should have LOW entropy,
        # ambiguous stimuli should have HIGH entropy
        clear_entropy = out['collapse_entropy'][:n_clear].mean()
        ambig_entropy = out['collapse_entropy'][n_clear:].mean()

        # Encourage entropy gap between clear and ambiguous
        entropy_loss = -ambig_entropy + clear_entropy + 0.5

        # Combined loss
        loss = cls_loss + 0.1 * entropy_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (step % 50 == 0 or step == n_steps - 1):
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == labels).float().mean()
            print(f"  Step {step:3d}: loss={loss.item():.4f}, "
                  f"acc={acc.item():.3f}, "
                  f"clear_entropy={clear_entropy.item():.4f}, "
                  f"ambig_entropy={ambig_entropy.item():.4f}, "
                  f"m={model.hans.fractal_mass.mass.item():.4f}")

    return losses


# ──────────────────────────────────────────────
#  Demo
# ──────────────────────────────────────────────

def demo():
    """Complete demo of the Hijoluminic Neural Cortex."""
    print("=" * 70)
    print("  Build 6: Hijoluminic Neural Cortex — BCI Superposition Model")
    print("=" * 70)

    # Configuration
    config = NeuralCortexConfig(
        n_electrodes=16,       # Small array for demo
        n_frequency_bands=3,   # δ, θ, α for speed
        n_time_windows=32,
        n_features_per_site=16,
    )

    model = HijoluminicCortex(
        config=config,
        n_interpretations=4,
        hidden_dim=64,
        n_cortical_layers=2,
    )

    print(f"\n  Neural config: {config.n_electrodes} electrodes, "
          f"{config.n_frequency_bands} bands, "
          f"{config.n_time_windows} time points")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Interpretations (branches): {model.n_interpretations}")

    # ── Phase 1: Pre-training analysis of ambiguous perception ──
    print("\n  Phase 1: Pre-training perceptual analysis...")
    with torch.no_grad():
        # Clear stimulus
        clear_eeg = generate_neural_response(
            (1, config.n_electrodes, config.n_time_windows),
            stimulus='clear_image',
        )
        clear_analysis = model.analyze_perception(clear_eeg)

        # Ambiguous stimulus (Rubin vase)
        ambig_eeg = generate_neural_response(
            (1, config.n_electrodes, config.n_time_windows),
            stimulus='rubin_vase',
        )
        ambig_analysis = model.analyze_perception(ambig_eeg)

    print(f"    Clear stimulus:  entropy={clear_analysis['entropy'][0].item():.4f}, "
          f"dominant={clear_analysis['dominant_interpretation'][0].item()}")
    print(f"    Ambiguous stimulus: entropy={ambig_analysis['entropy'][0].item():.4f}, "
          f"dominant={ambig_analysis['dominant_interpretation'][0].item()}, "
          f"ambiguous={ambig_analysis['is_ambiguous'][0].item()}")
    print(f"    Fractal mass: {model.hans.fractal_mass.mass.item():.4f}")

    # ── Phase 2: Training ──
    print("\n  Phase 2: Training on perceptual decision task...")
    losses = train_cortex(model, n_steps=80, batch_size=16, verbose=True)

    # ── Phase 3: Post-training analysis ──
    print("\n  Phase 3: Post-training perceptual analysis...")

    stimuli_to_test = ['clear_image', 'rubin_vase', 'necker_cube', 'noise']

    with torch.no_grad():
        for stim_name in stimuli_to_test:
            eeg = generate_neural_response(
                (8, config.n_electrodes, config.n_time_windows),
                stimulus=stim_name,
            )
            analysis = model.analyze_perception(eeg)

            mean_entropy = analysis['entropy'].mean().item()
            ambiguous_frac = analysis['is_ambiguous'].float().mean().item()

            print(f"    {stim_name:20s}: entropy={mean_entropy:.4f}, "
                  f"ambiguous_frac={ambiguous_frac:.2f}, "
                  f"prob_gap={analysis['probability_gap'].mean().item():.3f}")

    print("\n  " + "=" * 70)
    print("  Key scientific result:")
    print("    Ambiguous stimuli should show HIGHER entropy than clear stimuli,")
    print("    reflecting genuine perceptual uncertainty in the superposition.")
    print("    The entropy gap measures the model's 'perceptual awareness' of ambiguity.")
    print("  " + "=" * 70)
    print("  BCI Demo Complete OK")
    print("=" * 70)

    return model


if __name__ == '__main__':
    demo()
