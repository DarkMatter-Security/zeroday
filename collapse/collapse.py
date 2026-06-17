"""
collapse/collapse.py — Collapse of Light: a physics-grounded alternative to softmax.

This module replaces the standard softmax activation with a physical "collapse"
mechanism inspired by quantum measurement and fractal resonance.​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍

The CollapseOfLight computes:

    P_i = |α_i exp(i S_i / ℏ)|² · R_fractal(S_i) / Z

where:
    • |α_i exp(i S_i / ℏ)|² — Born-rule probability of branch i
    • R_fractal(S_i) = ∏_f 1 / (1 + (Δ_f / |S_i|)²) — fractal resonance factor
    • Z = Σ_j |α_j exp(i S_j / ℏ)|² · R_fractal(S_j) — normalization

Properties:
    - Replaces softmax with a physics process that has a built-in "I don't know"
    - Branches resonant with a fold get amplified
    - Branches off-resonance with ALL folds produce near-uniform distribution
    - No temperature parameter — the fractal folds provide scale selection
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CollapseOfLight(nn.Module):
    """
    Collapse-of-Light — a differentiable measurement process.

    This module implements the transition from quantum superposition
    to classical probability distribution, inspired by the "collapse
    of the wavefunction" in quantum mechanics but generalized to use
    fractal resonance weighting.

    The collapse is applied at the "measurement" step — when the system
    must produce a definite output from its internal superposition.
    """

    def __init__(self,
                 n_branches: int = 4,
                 use_fractal_resonance: bool = True,
                 use_interference: bool = True,
                 temperature: float = 1.0,
                 learnable_temperature: bool = False):
        """
        Args:
            n_branches: Number of branches in the superposition.
            use_fractal_resonance: Whether to apply fractal resonance weighting.
            use_interference: Whether to include interference in probabilities.
            temperature: Softmax temperature (falls back to standard softmax
                        when fractal resonance is off).
            learnable_temperature: If True, temperature is a learned parameter.
        """
        super().__init__()
        self.n_branches = n_branches
        self.use_fractal_resonance = use_fractal_resonance
        self.use_interference = use_interference

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer('_temp_val', torch.tensor(temperature, dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        if hasattr(self, 'log_temperature'):
            return torch.exp(self.log_temperature)
        if hasattr(self, '_temp_val'):
            return self._temp_val
        return torch.tensor(1.0, dtype=torch.float32)

    def fractal_weight(self,
                       branch_actions: Tensor,
                       fold_gaps: Tensor) -> Tensor:
        """
        Compute fractal resonance weight for each branch.

        R_i = ∏_f 1 / (1 + (Δ_f / |S_i|)²)

        Args:
            branch_actions: Tensor of shape (batch, n_branches) — S_i.
            fold_gaps: Tensor of shape (4,) — the four Δ_f.

        Returns:
            Resonance weights of shape (batch, n_branches).
        """
        eps = 1e-8
        s = branch_actions.abs() + eps  # (batch, B)

        weight = torch.ones_like(s)
        for f in fold_gaps:
            # R_f(s) = 1 / (1 + (f / s)²) = s² / (s² + f²)
            r = s ** 2 / (s ** 2 + f ** 2 + eps)
            weight = weight * r

        return weight

    def forward(self,
                branch_amplitudes: Tensor,
                branch_actions: Tensor | None = None,
                fold_gaps: Tensor | None = None,
                return_full: bool = False) -> Tensor | dict:
        """
        Apply collapse to obtain a probability distribution.

        Args:
            branch_amplitudes: Complex tensor (batch, n_branches) —
                              α_i exp(iθ_i + iS_i/ℏ) from PathIntegralLayer.
            branch_actions: Real tensor (batch, n_branches) — S_i action values.
                           Required if use_fractal_resonance is True.
            fold_gaps: Tensor (4,) — the four Δ_f. Required if use_fractal_resonance
                      is True and no internal FractalMass is available.
            return_full: If True, return full dict with intermediate values.

        Returns:
            If return_full:
                Dict with keys:
                    'probs': (batch, n_branches) — collapsed probability distribution.
                    'born_probs': (batch, n_branches) — Born-rule |α_i|² probabilities.
                    'resonance': (batch, n_branches) — fractal resonance weights.
                    'entropy': (batch,) — Shannon entropy of the distribution.
                    'max_prob': (batch,) — maximum probability value.
                    'is_certain': (batch,) — whether any branch has prob > 0.9.
            Else:
                Tensor (batch, n_branches) — probability distribution.
        """
        batch, B = branch_amplitudes.shape

        # ── Step 1: Compute Born-rule probabilities ──
        # P_born(i) = |α_i|² (squared magnitude of branch amplitude)
        # Note: branch_amplitudes = α_i exp(iΦ_i), so |α_i|² = |amp_i|²
        born_probs = (branch_amplitudes * branch_amplitudes.conj()).real  # (B, B)
        born_probs = born_probs + 1e-12  # stability

        # ── Step 2: Compute interference pattern ──
        if self.use_interference:
            # The interference pattern is |Σ α_i exp(iΦ_i)|²
            # We compute the coherent sum first
            G = torch.sum(branch_amplitudes, dim=-1, keepdim=True)  # (B, 1)
            interference_mag = (G * G.conj()).real  # (B, 1)
            # Distribute the total magnitude among branches proportionally
            # to their contribution: P_i ∝ born_probs_i · |G|² / Σ born_probs_j
            G_mag = interference_mag.squeeze(-1)  # (B,)
            probs_raw = born_probs * G_mag.unsqueeze(-1) / (born_probs.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            probs_raw = born_probs

        # ── Step 3: Apply fractal resonance weighting ──
        if self.use_fractal_resonance and branch_actions is not None and fold_gaps is not None:
            resonance = self.fractal_weight(branch_actions, fold_gaps)
            probs_raw = probs_raw * resonance
        else:
            resonance = torch.ones_like(born_probs)

        # ── Step 4: Normalize to valid probability distribution ──
        # Apply temperature
        probs_raw = probs_raw / self.temperature
        probs = probs_raw / (probs_raw.sum(dim=-1, keepdim=True) + 1e-12)

        # ── Step 5: Compute diagnostics ──
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        max_prob = probs.max(dim=-1).values
        is_certain = max_prob > 0.9

        if return_full:
            return {
                'probs': probs,
                'born_probs': born_probs / (born_probs.sum(dim=-1, keepdim=True) + 1e-12),
                'resonance': resonance,
                'entropy': entropy,
                'max_prob': max_prob,
                'is_certain': is_certain,
            }

        return probs

    def sample(self,
               branch_amplitudes: Tensor,
               branch_actions: Tensor | None = None,
               fold_gaps: Tensor | None = None,
               n_samples: int = 1,
               hard: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """
        Sample branch indices from the collapse probability distribution.

        Args:
            branch_amplitudes: (batch, n_branches) — complex branch amplitudes.
            branch_actions: (batch, n_branches) — action values.
            fold_gaps: (4,) — fold gaps.
            n_samples: Number of samples per batch element.
            hard: If True, return one-hot encoded samples.

        Returns:
            If hard:
                One-hot tensor (batch, n_samples, n_branches).
            Else:
                Index tensor (batch, n_samples).
        """
        probs = self.forward(branch_amplitudes, branch_actions, fold_gaps)

        # Sample from categorical distribution
        dist = torch.distributions.Categorical(probs=probs)
        samples = dist.sample((n_samples,))  # (n_samples, batch)
        samples = samples.transpose(0, 1)  # (batch, n_samples)

        if hard:
            one_hot = F.one_hot(samples, num_classes=self.n_branches)
            return one_hot.float()

        return samples


class CollapseStack(nn.Module):
    """
    A stack of CollapseOfLight layers, each operating at a different scale.

    This implements a hierarchical collapse mechanism where the system
    can "partially collapse" at intermediate layers, each time resolving
    some uncertainty while leaving others in superposition.

    The hierarchical collapse probability:
        P_total = ∏_{ℓ} P^{(ℓ)}_selected · (1 - p_collapse_ℓ) + ...
    """

    def __init__(self,
                 n_branches: int,
                 n_layers: int = 3,
                 layer_dim: int = 64):
        super().__init__()
        self.n_layers = n_layers
        self.collapse_layers = nn.ModuleList([
            CollapseOfLight(n_branches=n_branches)
            for _ in range(n_layers)
        ])
        # Collapse probability per layer (learned)
        self.collapse_gates = nn.Sequential(
            nn.Linear(layer_dim, n_layers),
            nn.Sigmoid(),
        )

    def forward(self,
                branch_amplitudes: Tensor,
                branch_actions: Tensor,
                fold_gaps: Tensor,
                context: Tensor) -> dict:
        """
        Hierarchical collapse.

        Args:
            branch_amplitudes: (batch, n_branches).
            branch_actions: (batch, n_branches).
            fold_gaps: (4,).
            context: (batch, layer_dim) — features for collapse gate.

        Returns:
            Dict with per-layer collapse results plus final aggregated probs.
        """
        gate_probs = self.collapse_gates(context)  # (batch, n_layers)

        all_probs = []
        current_amps = branch_amplitudes

        for layer_idx, collapse in enumerate(self.collapse_layers):
            layer_out = collapse(
                current_amps, branch_actions, fold_gaps, return_full=True
            )
            all_probs.append(layer_out['probs'])

            # Partial collapse: mix between full superposition and collapsed
            g = gate_probs[:, layer_idx:layer_idx+1]  # (batch, 1)
            # After collapse, amplitude concentrates on the max-prob branch
            max_branch = layer_out['probs'].argmax(dim=-1)
            collapsed_amps = torch.zeros_like(current_amps)
            collapsed_amps.scatter_(1, max_branch.unsqueeze(-1), 1.0)

            # Blend: (1-g) * original + g * collapsed
            current_amps = (1 - g) * current_amps + g * collapsed_amps

        return {
            'per_layer_probs': all_probs,
            'final_probs': all_probs[-1],
            'gate_probs': gate_probs,
        }
