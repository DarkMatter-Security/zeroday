"""
core/model.py — HijoluminicANS: The complete two-stream neural architecture.

This module implements the full Hijoluminic Artificial Neural System,
which maintains two parallel computational streams:​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍

  1. Schrödinger stream (1st quantization):
       Ψ^{ℓ+1} = Ψ^ℓ − i Δt · H_Ψ Ψ^ℓ
     where H_Ψ is the Hijoluminic Hamiltonian acting on the spinor field Ψ.

  2. Path-Integral stream (2nd quantization):
       G = Σ_i α_i exp(i S_i / ℏ)
     where the sum runs over all branches/paths in superposition.

These streams are kept distinct throughout the forward pass and fused
only at the "collapse" step — honoring the principle that the path
integral is NOT equivalent to the Schrödinger equation.

The architecture:
    Input → Embedding → Spinor Field Ψ₀
                      ↓
    ┌─── Schrödinger Stream (HijoluminicOperator) ───┐
    │   Ψ_{ℓ+1} = Ψ_ℓ − iΔt · H Ψ_ℓ                  │
    │                                                 │
    └─────────────────────────────────────────────────┘
                                 ↓
    ┌─── Path-Integral Stream (BranchGenerator + PI) ─┐
    │   Divergence → Branches → Action → G = Σ α e^{iS}│
    │                                                 │
    └─────────────────────────────────────────────────┘
                                 ↓
    ┌─── Collapse-of-Light ───────────────────────────┐
    │   P_i ∝ |α_i e^{iS_i/ℏ}|² · ∏_f R_f(S_i)      │
    └─────────────────────────────────────────────────┘
                                 ↓
                              Output
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.algebra import (
    gamma_matrices,
    complexified_velocity,
    born_probability,
)
from fields.hijoluminic import (
    FractalMass,
    HijoluminicOperator,
    VibrationalDensity,
    ScalarPotential,
)
from paths.branches import (
    BranchGenerator,
    ActionFunctional,
    PathIntegralLayer,
)
from collapse.collapse import CollapseOfLight
from topology.path_graph import PathBundle


# ──────────────────────────────────────────────
#  Spinor Embedding
# ──────────────────────────────────────────────

class SpinorEmbedding(nn.Module):
    """
    Embeds input features into a 4-component complex spinor field Ψ₀.

    Maps a real input tensor of shape (batch, n_sites, in_features) to a
    complex spinor field of shape (batch, n_sites, 4) using learned linear
    transformations.

    The 4 components correspond to:
        Components 0-1: Left-handed Weyl spinor (upper two components)
        Components 2-3: Right-handed Weyl spinor (lower two components)
    """

    def __init__(self,
                 in_features: int,
                 n_sites: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.n_sites = n_sites
        self.in_features = in_features

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4 * 2),  # 4 components × 2 (real + imag)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Map input to complex spinor field.

        Args:
            x: Input tensor of shape (batch, n_sites, in_features).

        Returns:
            Complex tensor of shape (batch, n_sites, 4).
        """
        raw = self.net(x)  # (batch, n_sites, 8)
        real = raw[..., :4]
        imag = raw[..., 4:]
        return torch.complex(real, imag)


# ──────────────────────────────────────────────
#  HijoluminicANS — Full Two-Stream Model
# ──────────────────────────────────────────────

class HijoluminicANS(nn.Module):
    """
    Hijoluminic Artificial Neural System — full two-stream architecture.

    This is the end-to-end differentiable model combining:
      - Schrödinger stream (HijoluminicOperator)
      - Path-integral stream (Branches + PathIntegral)
      - Collapse-of-Light (fractal-resonance-weighted collapse)
      - Optional PathBundle (A→B topology)

    The model can operate in several modes:
      - 'schrodinger': Only the Schrödinger stream (H-operator only)
      - 'path_integral': Only the path-integral stream (branches + interference)
      - 'full': Both streams, fused at collapse (default)
      - 'bundle': Use PathBundle instead of BranchGenerator
    """

    def __init__(self,
                 in_features: int,
                 n_sites: int,
                 n_classes: int | None = None,
                 n_bands: int = 1,
                 n_branches: int = 4,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 mode: str = 'full',
                 kappa_H: float = 1.0,
                 hbar: float = 1.0,
                 init_folds: list[float] | None = None):
        """
        Args:
            in_features: Dimensionality of input features per site.
            n_sites: Number of lattice sites.
            n_classes: Number of output classes (None for regression).
            n_bands: Number of vibrational bands for ρ_y.
            n_branches: Number of branches for path-integral stream.
            hidden_dim: Hidden dimension for internal MLPs.
            n_layers: Number of HijoluminicOperator layers in Schrödinger stream.
            mode: 'schrodinger', 'path_integral', 'full', or 'bundle'.
            kappa_H: Hijoluminic coupling scale.
            hbar: Reduced Planck constant.
            init_folds: Initial fold gaps for FractalMass.
        """
        super().__init__()
        self.in_features = in_features
        self.n_sites = n_sites
        self.n_classes = n_classes
        self.n_branches = n_branches
        self.n_layers = n_layers
        self.mode = mode
        self.hbar = hbar

        # ── Shared components ──
        # Spinor embedding
        self.embedding = SpinorEmbedding(in_features, n_sites, hidden_dim)

        # Fractal mass (shared across entire network)
        self.fractal_mass = FractalMass(
            kappa_H=kappa_H,
            init_folds=init_folds or [1.0, 2.0, 4.0, 8.0],
        )

        # ── Schrödinger stream ──
        self.operator_layers = nn.ModuleList([
            HijoluminicOperator(
                n_sites=n_sites,
                site_dim=in_features,
                n_bands=n_bands,
                kappa_H=kappa_H,
                init_folds=init_folds,
            )
            for _ in range(n_layers)
        ])

        # ── Path-integral stream ──
        # Global context = flattened spinor (4 components * n_sites) real+imag = 8*n_sites
        self.branch_gen = BranchGenerator(
            in_features=8 * n_sites,  # global context (spinor real+imag)
            n_branches=n_branches,
            hidden_dim=hidden_dim,
        )
        self.action_func = ActionFunctional(
            state_dim=8 * n_sites,  # match branch_gen context dimension
            n_branches=n_branches,
        )
        self.path_integral = PathIntegralLayer(
            n_branches=n_branches,
            hbar=hbar,
        )

        # ── Collapse ──
        self.collapse = CollapseOfLight(
            n_branches=n_branches,
            use_fractal_resonance=True,
            use_interference=True,
        )

        # ── Output heads ──
        if n_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(8 * n_sites + n_branches, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, n_classes),
            )

        # ── Optional PathBundle ──
        self.path_bundle = PathBundle(
            state_dim=in_features,
            n_paths=n_branches,
            hbar=hbar,
        )

    def forward(self,
                x: Tensor,
                return_all: bool = False) -> Tensor | dict:
        """
        Full forward pass through the two-stream architecture.

        Args:
            x: Input tensor of shape (batch, n_sites, in_features).
            return_all: If True, return full diagnostic dict.

        Returns:
            If return_all:
                Dict with keys for every internal state.
            Else:
                Output logits / predictions of shape (batch, n_classes) or (batch,).
        """
        batch = x.shape[0]
        device = x.device

        # ── Step 1: Embed to spinor field ──
        psi = self.embedding(x)  # (batch, n_sites, 4)

        # ── Step 2: Schrödinger stream ──
        psi_schrodinger = psi
        all_psi_states = [psi_schrodinger]
        all_rho = []

        for op_layer in self.operator_layers:
            psi_schrodinger, rho_y, Phi_val, mass, op_norm = op_layer(
                psi_schrodinger, x, return_components=True
            )
            all_psi_states.append(psi_schrodinger)
            all_rho.append(rho_y)

        # ── Step 3: Create divergence point from Schrödinger stream ──
        # Global context = flattened final spinor state (complex -> real via concat)
        psi_flat = psi_schrodinger.reshape(batch, -1)  # (batch, 4 * n_sites) complex
        global_context = torch.cat([psi_flat.real, psi_flat.imag], dim=-1)  # (batch, 8 * n_sites) real

        # Divergence point features = global context (spinor-derived)
        # This matches the branch_gen and action_func dimensions
        div_features = global_context[:, :self.branch_gen.in_features]  # (batch, 8*n_sites)

        # ── Step 4: Path-integral stream ──
        # Generate branches
        amplitudes, phases, branch_offsets = self.branch_gen(div_features)

        # Compute trajectories and actions
        trajectories = self.action_func.compute_trajectory(
            div_features, amplitudes, branch_offsets
        )
        actions = self.action_func.compute_action(trajectories)  # (batch, n_branches)

        # Path integral superposition
        pi_result = self.path_integral(amplitudes, phases, actions)

        # ── Step 5: Collapse-of-Light ──
        collapse_result = self.collapse(
            pi_result['branch_amplitudes'],
            actions,
            self.fractal_mass.folds,
            return_full=True,
        )

        # ── Step 6: Bundle mode (optional alternative) ──
        bundle_result = None
        if self.mode == 'bundle':
            bundle_result = self.path_bundle(
                div_features.unsqueeze(1),  # A = current state
                torch.zeros_like(div_features.unsqueeze(1)),  # B = target/zero
            )

        # ── Step 7: Output ──
        if self.n_classes is not None:
            # Build combined features: final spinor (real+imag) + collapse probs
            psi_flat = psi_schrodinger.reshape(batch, -1)  # (batch, 4*n_sites) complex
            psi_real = torch.cat([psi_flat.real, psi_flat.imag], dim=-1)  # (batch, 8*n_sites)
            probs = collapse_result['probs']  # (batch, n_branches)
            combined = torch.cat([psi_real, probs], dim=-1)
            logits = self.classifier(combined)
            out = logits
        else:
            # Regression / reconstruction: use the collapsed mean
            probs = collapse_result['probs']
            # Weighted sum of branch trajectories
            out = torch.sum(
                trajectories.mean(dim=2) * probs.unsqueeze(-1), dim=1
            )  # (batch, in_features)

        if return_all:
            return {
                'psi_states': all_psi_states,
                'psi_final': psi_schrodinger,
                'rho_y': all_rho[-1],
                'mass': self.fractal_mass.mass,
                'fold_gaps': self.fractal_mass.folds,
                'amplitudes': amplitudes,
                'phases': phases,
                'actions': actions,
                'branch_amplitudes': pi_result['branch_amplitudes'],
                'G': pi_result['G'],
                'G_magnitude': pi_result['magnitudes'],
                'interference': pi_result['interference_terms'],
                'collapse_probs': collapse_result['probs'],
                'collapse_entropy': collapse_result['entropy'],
                'collapse_max_prob': collapse_result['max_prob'],
                'born_probs': collapse_result['born_probs'],
                'resonance': collapse_result['resonance'],
                'bundle': bundle_result,
                'output': out,
            }

        return out

    def get_uncertainty(self, x: Tensor) -> Tensor:
        """
        Get the model's inherent uncertainty about its prediction.

        This is the imaginary part of the fractal mass — a principled
        uncertainty estimate that falls out of the algebra, not from
        a learned confidence head.

        Args:
            x: Input tensor (batch, n_sites, in_features).

        Returns:
            Uncertainty tensor (batch,) — higher means more uncertain.
        """
        with torch.no_grad():
            out = self.forward(x, return_all=True)

        # Uncertainty = entropy of collapse distribution + imaginary mass contribution
        entropy = out['collapse_entropy']  # (batch,)
        imag_mass = self.fractal_mass.mass_imag.abs()  # scalar

        # Combine: higher entropy = more uncertain
        # The imag_mass provides a global uncertainty floor
        uncertainty = entropy - imag_mass
        return uncertainty

    def classify_with_uncertainty(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Classify with principled uncertainty estimates.

        Returns:
            Tuple (logits, probs, uncertainty):
                logits: Raw output logits (batch, n_classes).
                probs: Softmax probabilities (batch, n_classes).
                uncertainty: Per-sample uncertainty (batch,).
        """
        out = self.forward(x, return_all=True)
        logits = out['output']

        if self.n_classes:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)

        # Uncertainty: combination of collapse entropy and max probability
        entropy = out['collapse_entropy']
        max_prob = out['collapse_max_prob']
        # Normalize to [0, 1] range
        uncertainty = 1.0 - max_prob + 0.1 * entropy
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)

        return logits, probs, uncertainty


# ──────────────────────────────────────────────
#  HijoluminicANSLight — Simplified for rapid deployment
# ──────────────────────────────────────────────

class HijoluminicANSLight(nn.Module):
    """
    Lighter version of H-ANS for rapid prototyping and deployment.

    Uses only the essential components:
      - 1 HijoluminicOperator layer
      - 1 CollapseOfLight (no path-integral stream)
      - FractalMass for uncertainty

    Suitable for:
      - Anomaly detection (Build 1)
      - Real-time inference on edge devices
      - Quick baselines
    """

    def __init__(self,
                 in_features: int,
                 n_sites: int,
                 hidden_dim: int = 64,
                 kappa_H: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.n_sites = n_sites

        self.embedding = SpinorEmbedding(in_features, n_sites, hidden_dim)
        self.operator = HijoluminicOperator(
            n_sites=n_sites,
            site_dim=in_features,
            kappa_H=kappa_H,
        )
        self.fractal_mass = self.operator.mass  # shared

        # Simple output head (takes real+imag concat -> 8*n_sites features)
        self.output_head = nn.Sequential(
            nn.Linear(8 * n_sites, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> dict:
        psi = self.embedding(x)
        psi_out, rho_y, Phi, mass, op_norm = self.operator(
            psi, x, return_components=True
        )

        # Convert complex spinor to real features (concat real + imag)
        psi_flat = psi_out.reshape(x.shape[0], -1)
        features = torch.cat([psi_flat.real, psi_flat.imag], dim=-1)
        logit = self.output_head(features)

        # Entropy of spinor field as simple uncertainty
        field_entropy = -(psi_out.abs() ** 2 * torch.log(psi_out.abs() ** 2 + 1e-12)).sum(
            dim=(-1, -2)
        )

        return {
            'logit': logit,
            'psi': psi_out,
            'rho': rho_y,
            'mass': mass,
            'uncertainty': field_entropy,
        }
