"""
fields/hijoluminic.py — The Hijoluminic field-theoretic neural operator.

This module implements the core field-theoretic objects:

  • ρ_y(x)  — Learned non-negative vibrational density per "site"
  • Φ(x)    — Learned scalar potential
  • D_μ     — Finite-difference covariant derivative with gauge connection
  • FractalMass — Four-fold complex mass from learnable fold-gaps Δ_f
  • HijoluminicOperator — The master-equation operator acting on spinor fields

The Hijoluminic master equation:

    [γ^μ (iℏ D_μ − α ∂_p Φ) − β (κ_H ρ_y + λ) c] Ψ = 0

where:
    γ^μ    — Dirac gamma matrices in Weyl representation
    D_μ    — Covariant derivative: ∂_μ + i A_μ (gauge connection)
    Φ      — Learned scalar potential
    ρ_y    — Vibrational density (non-negative, per frequency band y)
    α, β   — Learnable coupling constants
    κ_H    — Hijoluminic coupling scale
    λ      — Bare mass coupling
    c, ℏ   — Speed of light and reduced Planck constant (set to 1 by default)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.algebra import gamma_matrices, gamma5


# ──────────────────────────────────────────────
#  Fractal Mass — Four-fold complex mass
# ──────────────────────────────────────────────

class FractalMass(nn.Module):
    """
    Four-fold fractal mass module.​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍

    The mass is complex by construction:
        m = (1 + Σ_f i / Δ_f) · κ_H · c²

    where Δ_f ∈ ℝ₊ are four learnable fold-gap parameters.

    Properties:
        - Re(m)  : Effective inertial mass (real part)
        - Im(m)  : Vibrational genesis/decay rate (imaginary part) —
                   positive → growth, negative → decay

    The four fold-gaps independently learn the resonances of the
    fractal geometry, encoding four characteristic scales.
    """

    def __init__(self,
                 kappa_H: float = 1.0,
                 c: float = 1.0,
                 init_folds: list[float] | None = None,
                 learnable: bool = True):
        """
        Args:
            kappa_H: Hijoluminic coupling scale κ_H.
            c: Speed of light (default 1.0 for natural units).
            init_folds: Initial values for the four fold-gaps Δ_f.
                        Defaults to [1.0, 2.0, 4.0, 8.0].
            learnable: Whether Δ_f are learnable parameters.
        """
        super().__init__()
        self.kappa_H = nn.Parameter(torch.tensor(kappa_H, dtype=torch.float32),
                                    requires_grad=learnable)
        self.c = c

        if init_folds is None:
            init_folds = [1.0, 2.0, 4.0, 8.0]
        assert len(init_folds) == 4, "Exactly 4 fold-gaps required"

        # Store fold gaps as learnable parameters (positive via softplus)
        raw_init = torch.tensor(init_folds, dtype=torch.float32)
        # We parameterize as log(exp(Δ) - 1) so Δ = softplus(raw) is always > 0
        self.raw_folds = nn.Parameter(raw_init, requires_grad=learnable)

    @property
    def folds(self) -> Tensor:
        """Four fold-gaps Δ_f, always positive via softplus."""
        return F.softplus(self.raw_folds)

    @property
    def mass(self) -> Tensor:
        """
        Complex mass m = (1 + Σ_f i / Δ_f) · κ_H · c².

        Returns:
            Complex scalar tensor.
        """
        kH = self.kappa_H
        c2 = self.c ** 2
        # Σ_f i / Δ_f = i · Σ_f 1/Δ_f
        sum_inv = torch.sum(1.0 / self.folds)
        m = (1.0 + 1j * sum_inv) * kH * c2
        return m

    @property
    def mass_real(self) -> Tensor:
        """Real part of the mass (inertial mass)."""
        return self.mass.real

    @property
    def mass_imag(self) -> Tensor:
        """Imaginary part of the mass (vibrational genesis/decay rate)."""
        return self.mass.imag

    @property
    def decay_rate(self) -> Tensor:
        """
        Effective decay rate = -κ_H · c² · Σ_f 1/Δ_f.

        Positive → field growth (genesis), negative → field decay.
        With kH > 0, c² > 0, Δ_f > 0, this is always negative
        (the default sign convention gives decay).
        """
        return -self.kappa_H * (self.c ** 2) * torch.sum(1.0 / self.folds)

    def resonance_factor(self, action_magnitude: Tensor) -> Tensor:
        """
        Compute the fractal resonance factor for a given action magnitude.

            R(S) = ∏_f 1 / (1 + (Δ_f / |S|)²)

        This factor approaches 1 when |S| is resonant with a fold Δ_f
        (i.e., |S| ≈ Δ_f), and approaches 0 when |S| is far from all folds.

        Args:
            action_magnitude: Tensor of shape (...,). The magnitude |S| of action.

        Returns:
            Tensor of same shape, resonance factor in [0, 1].
        """
        eps = 1e-8
        s = action_magnitude.abs() + eps
        # R(S) = ∏_f 1 / (1 + (Δ_f / s)²)
        #      = ∏_f s² / (s² + Δ_f²)
        factor = torch.ones_like(s)
        for f in self.folds:
            r = s ** 2 / (s ** 2 + f ** 2 + eps)
            factor = factor * r
        return factor

    def extra_repr(self) -> str:
        return (f"κ_H={self.kappa_H.item():.4f}, "
                f"Δ=[{', '.join(f'{d.item():.4f}' for d in self.folds)}], "
                f"m={self.mass.item():.4f}")


# ──────────────────────────────────────────────
#  Learned Fields — ρ_y and Φ
# ──────────────────────────────────────────────

class VibrationalDensity(nn.Module):
    """
    Learned non-negative vibrational density ρ_y(x).

    ρ_y(x) represents the "vibrational aliveness" of site x in frequency
    band y. It is parameterized as softplus(MLP(x)) to ensure non-negativity.

    This density controls the local coupling strength in the
    HijoluminicOperator via the term -β κ_H ρ_y Ψ.
    """

    def __init__(self,
                 in_features: int,
                 n_bands: int = 1,
                 hidden_dim: int = 64):
        """
        Args:
            in_features: Dimensionality of input features per site.
            n_bands: Number of vibrational frequency bands (channels y).
            hidden_dim: Hidden dimension of the MLP.
        """
        super().__init__()
        self.n_bands = n_bands
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_bands),
        )
        # Learned per-band scaling
        self.log_scale = nn.Parameter(torch.zeros(n_bands))

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute ρ_y(x).

        Args:
            x: Input features of shape (..., in_features).
               Typically position encoding or site embedding.

        Returns:
            Non-negative tensor of shape (..., n_bands).
        """
        raw = self.net(x)
        # Ensure non-negativity via softplus, then apply learned scale
        rho = F.softplus(raw) * torch.exp(self.log_scale)
        return rho


class ScalarPotential(nn.Module):
    """
    Learned scalar potential Φ(x).

    Φ(x) is a site-dependent scalar field that acts as a potential
    energy term in the HijoluminicOperator. It is parameterized as
    an MLP with tanh activations, outputting a single real scalar per site.

    The term -α ∂_p Φ in the master equation provides a driving force
    proportional to the gradient of this potential along the path.
    """

    def __init__(self,
                 in_features: int,
                 hidden_dim: int = 64,
                 n_layers: int = 3):
        """
        Args:
            in_features: Dimensionality of input per site.
            hidden_dim: Hidden dimension.
            n_layers: Number of hidden layers.
        """
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute Φ(x).

        Args:
            x: Input of shape (..., in_features).

        Returns:
            Tensor of shape (..., 1) — the scalar potential.
        """
        return self.net(x)


# ──────────────────────────────────────────────
#  Covariant Derivative
# ──────────────────────────────────────────────

class CovariantDerivative(nn.Module):
    """
    Finite-difference covariant derivative D_μ = ∂_μ + i A_μ.

    The gauge connection A_μ is a learned vector field that couples
    adjacent sites in the lattice. In the continuum interpretation,
    A_μ plays the role of a U(1) gauge field, creating a "twist"
    between neighboring sites.

    For a 1-D chain of N sites with periodic boundary conditions:
        (D_x Ψ)_i = (Ψ_{i+1} - Ψ_i) / Δx + i A_i Ψ_i

    The module supports arbitrary-dimensional inputs and uses
    central differences when possible.
    """

    def __init__(self,
                 n_sites: int,
                 site_dim: int,
                 n_spinor_components: int = 4,
                 n_spatial_dims: int = 1):
        """
        Args:
            n_sites: Number of lattice sites.
            site_dim: Feature dimension per site.
            n_spinor_components: Number of spinor components (4 for Dirac).
            n_spatial_dims: Number of spatial dimensions for derivative.
        """
        super().__init__()
        self.n_sites = n_sites
        self.site_dim = site_dim
        self.n_spinor = n_spinor_components
        self.n_dims = n_spatial_dims

        # Gauge connection A_μ: one (n_spinor, n_spinor) matrix per site per dimension
        # For efficiency, we use a block-diagonal parameterization
        self.A = nn.Parameter(
            torch.randn(n_sites, n_spatial_dims, n_spinor_components, dtype=torch.complex64) * 0.1
        )

        # Learnable lattice spacing
        self.log_dx = nn.Parameter(torch.zeros(n_spatial_dims))

    @property
    def dx(self) -> Tensor:
        """Lattice spacing for each dimension."""
        return torch.exp(self.log_dx)

    def forward(self, psi: Tensor) -> Tensor:
        """
        Apply the covariant derivative.

        For a 1-D chain of N sites:
            psi: shape (batch, N, spinor_components)
            Returns: (batch, N, spinor_components) — covariant derivative

        The gauge connection is applied as a diagonal matrix:
            (D_μ Ψ)_i = (Ψ_{i+1} - Ψ_{i-1}) / (2 dx) + i A_μ,i · Ψ_i
        """
        batch, N, S = psi.shape
        assert N == self.n_sites, f"Expected {self.n_sites} sites, got {N}"
        assert S == self.n_spinor

        dx = self.dx[0]  # 1-D case for now

        # Central difference: (Ψ_{i+1} - Ψ_{i-1}) / (2 dx)
        # With periodic boundary
        psi_forward = torch.roll(psi, shifts=-1, dims=1)   # Ψ_{i+1}
        psi_backward = torch.roll(psi, shifts=1, dims=1)   # Ψ_{i-1}
        deriv = (psi_forward - psi_backward) / (2.0 * dx)

        # Gauge term: i A_i · Ψ_i
        # A_i is diagonal along spinor dims
        gauge_term = 1j * self.A[:, 0, :] * psi  # (batch, N, S)

        return deriv + gauge_term


# ──────────────────────────────────────────────
#  Hijoluminic Operator
# ──────────────────────────────────────────────

class HijoluminicOperator(nn.Module):
    r"""
    The Hijoluminic master-equation neural operator.

    This module replaces the Transformer attention block with a
    complex-spinor operator that acts on a 4-component complex field
    Ψ(x) per site. The operator implements:

        [γ^μ (iℏ D_μ − α ∂_p Φ) − β (κ_H ρ_y + λ) c] Ψ = 0

    In the neural network context, this is used as a transformation:
        Ψ^{(\ell+1)} = Ψ^{(\ell)} − i Δt · H(Ψ^{(\ell)})

    where H is the Hijoluminic Hamiltonian derived from the operator.

    Parameters:
        Ψ: 4-component complex spinor field per site
        ρ_y: Vibrational density per site and band
        Φ: Scalar potential per site
        γ^μ: Dirac gamma matrices (fixed)
        D_μ: Covariant derivative with learned gauge connection
        α, β: Learnable coupling constants
        κ_H: Hijoluminic coupling scale
        λ: Bare mass coupling
    """

    def __init__(self,
                 n_sites: int,
                 site_dim: int,
                 n_bands: int = 1,
                 kappa_H: float = 1.0,
                 lam: float = 0.1,
                 c: float = 1.0,
                 hbar: float = 1.0,
                 init_folds: list[float] | None = None):
        """
        Args:
            n_sites: Number of lattice sites.
            site_dim: Feature dimension per site.
            n_bands: Number of vibrational bands.
            kappa_H: Hijoluminic coupling scale.
            lam: Bare mass coupling λ.
            c: Speed of light.
            hbar: Reduced Planck constant.
            init_folds: Initial fold-gap values for FractalMass.
        """
        super().__init__()
        self.n_sites = n_sites
        self.site_dim = site_dim
        self.n_bands = n_bands
        self.c = c
        self.hbar = hbar

        # Dirac gamma matrices (fixed, not learnable)
        gammas = gamma_matrices()
        self.register_buffer('γ0', gammas[0])
        self.register_buffer('γ1', gammas[1])
        self.register_buffer('γ2', gammas[2])
        self.register_buffer('γ3', gammas[3])

        # Learned fields
        self.rho = VibrationalDensity(site_dim, n_bands=n_bands)
        self.Phi = ScalarPotential(site_dim)

        # Covariant derivative
        self.D = CovariantDerivative(n_sites, site_dim)

        # Fractal mass
        self.mass = FractalMass(kappa_H=kappa_H, init_folds=init_folds)

        # Coupling constants
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lam = nn.Parameter(torch.tensor(lam, dtype=torch.float32))

        # Time step for Schrödinger evolution
        self.log_dt = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def dt(self) -> Tensor:
        """Evolution time step."""
        return torch.exp(self.log_dt)

    def forward(self,
                psi: Tensor,
                site_features: Tensor,
                return_components: bool = False) -> Tensor | tuple[Tensor, ...]:
        """
        Apply the Hijoluminic operator to a spinor field.

        Args:
            psi: Complex tensor of shape (batch, n_sites, 4) — spinor field.
            site_features: Tensor of shape (batch, n_sites, site_dim) — features
                          for computing ρ_y and Φ.
            return_components: If True, also return intermediate fields.

        Returns:
            If return_components:
                Tuple (psi_out, rho_y, Phi_val, mass, op_norm)
            Else:
                psi_out: The evolved spinor field.
        """
        batch, N, S = psi.shape
        assert N == self.n_sites
        assert S == 4

        # ── 1. Compute learned fields ──
        # ρ_y: (batch, N, n_bands)
        rho_y = self.rho(site_features)

        # Φ: (batch, N, 1)
        Phi_val = self.Phi(site_features)

        # ── 2. Compute the operator H acting on Ψ ──
        # H = γ^μ (iℏ D_μ - α ∂_p Φ) - β(κ_H ρ_y + λ)c

        # Covariant derivative term: iℏ D_μ Ψ  (in 1-D for now)
        # D_μ acts on spatial index, giving (batch, N, S)
        D_psi = self.D(psi)  # (batch, N, S)
        i_hbar_D = 1j * self.hbar * D_psi  # iℏ D_x Ψ

        # Potential gradient term: -α ∂_p Φ · Ψ
        # ∂_p Φ = finite difference of Φ along spatial dimension
        Phi_grad = torch.roll(Phi_val, shifts=-1, dims=1) - torch.roll(Phi_val, shifts=1, dims=1)
        Phi_grad = Phi_grad / 2.0  # central difference
        alpha_grad_Phi = -self.alpha * Phi_grad * psi

        # Sum: (iℏ D_μ - α ∂_p Φ) Ψ
        # This should be contracted with γ^μ. For 1-D, μ = 1 (spatial):
        K_psi = self.γ1 @ (i_hbar_D + alpha_grad_Phi).unsqueeze(-1)  # (B, N, 4, 1)
        K_psi = K_psi.squeeze(-1)  # (B, N, 4)

        # Mass term: β(κ_H ρ_y + λ)c · Ψ
        # Average ρ_y over bands for the coupling
        rho_mean = rho_y.mean(dim=-1, keepdim=True)  # (B, N, 1)
        mass_coupling = self.beta * (self.mass.kappa_H * rho_mean + self.lam) * self.c
        M_psi = mass_coupling * psi  # (B, N, 4)

        # Full operator: H Ψ = K_psi - M_psi
        H_psi = K_psi - M_psi

        # ── 3. Schrödinger evolution: Ψ' = Ψ - iΔt · HΨ ──
        # Note: this is the "Schrödinger stream" update.
        psi_out = psi - 1j * self.dt * H_psi

        if return_components:
            op_norm = H_psi.abs().mean().item()
            return psi_out, rho_y, Phi_val, self.mass.mass, op_norm

        return psi_out

    def extra_repr(self) -> str:
        return (f"sites={self.n_sites}, site_dim={self.site_dim}, "
                f"bands={self.n_bands}, α={self.alpha.item():.3f}, "
                f"β={self.beta.item():.3f}, λ={self.lam.item():.3f}")
