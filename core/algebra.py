"""
core/algebra.py — Dirac γ^μ matrices, complexified velocity, Clifford algebra,
and branch-superposition primitives.​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍

This module provides the fundamental algebraic objects that underpin the
Hijoluminic Artificial Neural System (H-ANS).

Formulations implemented:
  • Dirac gamma matrices in the chiral (Weyl) representation
  • Clifford anticommutator {γ^μ, γ^ν} = 2 η^μν
  • Complexified velocity   V = [X + j(-Y), Y + j(-X), Z + jZ]^T
  • Branch superposition primitives (coherent sums)
"""

from __future__ import annotations

import torch
from torch import Tensor


# ──────────────────────────────────────────────
#  Dirac Gamma Matrices — Chiral (Weyl) rep
# ──────────────────────────────────────────────

def pauli_matrices() -> tuple[Tensor, Tensor, Tensor]:
    """Return the three Pauli matrices as complex64 tensors (2×2)."""
    σ_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    σ_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    σ_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    return σ_x, σ_y, σ_z


def gamma_matrices(metric_sign: str = '+---') -> tuple[Tensor, ...]:
    """
    Return the four Dirac gamma matrices in the Weyl (chiral) representation.

    Args:
        metric_sign: Metric signature convention — '+---' for particle physics,
                     '-+++' for GR. This affects γ^0 sign convention.

    Returns:
        Tuple of 4 gamma matrices, each (4, 4) complex64.

    Weyl representation:
        γ^0 = [[0,  I],
               [I,  0]]

        γ^i = [[0,   σ^i],
               [-σ^i, 0]]
    """
    I2 = torch.eye(2, dtype=torch.complex64)
    O2 = torch.zeros(2, 2, dtype=torch.complex64)
    σ_x, σ_y, σ_z = pauli_matrices()

    def _block(top_left, top_right, bottom_left, bottom_right):
        top = torch.cat([top_left, top_right], dim=1)
        bottom = torch.cat([bottom_left, bottom_right], dim=1)
        return torch.cat([top, bottom], dim=0)

    γ0 = _block(O2, I2, I2, O2)
    γ1 = _block(O2, σ_x, -σ_x, O2)
    γ2 = _block(O2, σ_y, -σ_y, O2)
    γ3 = _block(O2, σ_z, -σ_z, O2)

    if metric_sign == '-+++':
        γ0 = -γ0

    return γ0, γ1, γ2, γ3


def gamma5() -> Tensor:
    """
    Return the γ⁵ matrix in the Weyl representation.

        γ⁵ = [[-I, 0],
              [ 0, I]]
    """
    I2 = torch.eye(2, dtype=torch.complex64)
    O2 = torch.zeros(2, 2, dtype=torch.complex64)
    top = torch.cat([-I2, O2], dim=1)
    bottom = torch.cat([O2, I2], dim=1)
    return torch.cat([top, bottom], dim=0)


def clifford_anticommutator(γμ: Tensor, γν: Tensor) -> Tensor:
    """
    Compute {γ^μ, γ^ν} = γ^μ γ^ν + γ^ν γ^μ.

    Returns a (4, 4) complex tensor.
    For γμ and γν from the standard gamma matrices, this should equal 2 η^μν I_4.
    """
    return γμ @ γν + γν @ γμ


def verify_clifford_algebra(gammas: tuple[Tensor, ...],
                            η: Tensor | None = None,
                            atol: float = 1e-6) -> dict[str, bool | Tensor]:
    """
    Verify that {γ^μ, γ^ν} = 2 η^μν I_4 for all pairs μ, ν.

    Args:
        gammas: Tuple of 4 gamma matrices.
        η: Metric tensor. Defaults to diag(1, -1, -1, -1).
        atol: Absolute tolerance for equality.

    Returns:
        Dict with keys:
          - 'passed': True if all pairs satisfy the algebra.
          - 'results': (4, 4) tensor of max deviations per pair.
          - 'max_deviation': maximum absolute deviation across all pairs.
    """
    if η is None:
        η = torch.diag(torch.tensor([1, -1, -1, -1], dtype=torch.complex64))

    I4 = torch.eye(4, dtype=torch.complex64)
    results = torch.zeros(4, 4, dtype=torch.float64)

    for μ in range(4):
        for ν in range(4):
            anticom = clifford_anticommutator(gammas[μ], gammas[ν])
            expected = 2 * η[μ, ν] * I4
            diff = (anticom - expected).abs().max().item()
            results[μ, ν] = diff

    max_dev = results.max().item()
    return {
        'passed': max_dev < atol,
        'results': results,
        'max_deviation': max_dev,
    }


# ──────────────────────────────────────────────
#  Complexified Velocity
# ──────────────────────────────────────────────

def complexified_velocity(X: Tensor, Y: Tensor, Z: Tensor) -> Tensor:
    """
    Compute the ℂ³ complexified velocity map.

    The mapping is:
        V₁ = X + j · (-Y)   = X - jY
        V₂ = Y + j · (-X)   = Y - jX
        V₃ = Z + j · Z      = Z(1 + j)

    This couples X↔Y antisymmetrically and leaves Z self-paired,
    producing the non-isotropic ℂ³ structure that is the source of
    "divergence sensitivity" in the H-ANS architecture.

    Args:
        X: Tensor of shape (...,), real-valued.
        Y: Tensor of shape (...,), real-valued.
        Z: Tensor of shape (...,), real-valued.

    Returns:
        Complex tensor of shape (..., 3) with V₁, V₂, V₃ in the last dim.
    """
    # Promote to complex
    X_c = X.to(dtype=torch.complex64)
    Y_c = Y.to(dtype=torch.complex64)
    Z_c = Z.to(dtype=torch.complex64)

    V1 = X_c - 1j * Y_c   # X + j(-Y)
    V2 = Y_c - 1j * X_c   # Y + j(-X)
    V3 = Z_c * (1 + 1j)   # Z + jZ

    return torch.stack([V1, V2, V3], dim=-1)


def complexified_velocity_inverse(V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Recover (X, Y, Z) from complexified velocity.

    Given V₁, V₂ ∈ ℂ and V₃ ∈ ℂ, the inverse map exists as a linear solve:
        X = (j·V₂ - j·V₁) / (j² - 1)  ... simplified:
        X = Re( (V₁ + V₂) / (1 - j) )  ... checks out actually:
        
    Solving the linear system:
        V₁ = X - jY
        V₂ = Y - jX
        
    In matrix form:
        [V₁]   [1  -j] [X]
        [V₂] = [-j  1] [Y]
        
    det = 1 - (-j)(-j) = 1 - j² = 1 + 1 = 2
        
        [X]   (1/2) [ 1   j] [V₁]
        [Y] = (1/2) [ j   1] [V₂]
        
    And V₃ = Z(1+j) → Z = V₃ / (1+j)

    Returns:
        Tuple (X, Y, Z) of real-valued tensors with same shape as V[..., 0].
    """
    V1 = V[..., 0]
    V2 = V[..., 1]
    V3 = V[..., 2]

    # Invert the 2×2 system for X, Y
    X = 0.5 * (V1 + 1j * V2).real
    Y = 0.5 * (1j * V1 + V2).real
    Z = (V3 / (1 + 1j)).real

    return X, Y, Z


# ──────────────────────────────────────────────
#  Branch Superposition Primitives
# ──────────────────────────────────────────────

def coherent_superposition(amplitudes: Tensor, phases: Tensor) -> Tensor:
    """
    Coherent sum of branch amplitudes: Σᵢ αᵢ exp(i·θᵢ).

    Args:
        amplitudes: Real tensor of shape (..., B) — branch magnitudes |αᵢ|.
        phases: Real tensor of shape (..., B) — branch phases θᵢ in radians.

    Returns:
        Complex tensor of shape (...,) — the coherent sum Σᵢ αᵢ exp(iθᵢ).
    """
    return torch.sum(amplitudes * torch.exp(1j * phases), dim=-1)


def interference_pattern(amplitudes: Tensor, phases: Tensor) -> Tensor:
    """
    Compute the interference |Σᵢ αᵢ exp(iθᵢ)|².

    Args:
        amplitudes: Real tensor of shape (..., B).
        phases: Real tensor of shape (..., B).

    Returns:
        Real tensor of shape (...,) — the squared magnitude of the sum.
    """
    summed = coherent_superposition(amplitudes, phases)
    return (summed * summed.conj()).real


def born_probability(psi: Tensor) -> Tensor:
    """
    Born rule: probability = |ψ|².

    Args:
        psi: Complex tensor of arbitrary shape.

    Returns:
        Real non-negative tensor, same shape as psi.
    """
    return (psi * psi.conj()).real


# ──────────────────────────────────────────────
#  Metric helpers
# ──────────────────────────────────────────────

def minkowski_metric(signature: str = '+---') -> Tensor:
    """Return the Minkowski metric tensor η_μν as a (4, 4) tensor."""
    if signature == '+---':
        return torch.diag(torch.tensor([1, -1, -1, -1], dtype=torch.complex64))
    elif signature == '-+++':
        return torch.diag(torch.tensor([-1, 1, 1, 1], dtype=torch.complex64))
    else:
        raise ValueError(f"Unknown signature: {signature}")


def sl2c_generator() -> list[Tensor]:
    """
    Return the six generators of SL(2, ℂ) — three rotations + three boosts.

    These generate the spinor representation of the Lorentz group and are
    useful for constructing gauge connections in the HijoluminicOperator.

    Returns:
        List of 6 tensors, each (2, 2) complex64.
        Indices 0-2: rotations (σ_x/2, σ_y/2, σ_z/2)
        Indices 3-5: boosts (iσ_x/2, iσ_y/2, iσ_z/2)
    """
    σ_x, σ_y, σ_z = pauli_matrices()
    rotations = [σ_x / 2, σ_y / 2, σ_z / 2]
    boosts = [1j * σ_x / 2, 1j * σ_y / 2, 1j * σ_z / 2]
    return rotations + boosts
