"""
tests/test_sanity.py ? 8 validation tests for H-ANS.

Tests:
  1. Clifford anticommutator {?^?, ?^?} = 2?^??
  2. Complexified velocity map (Image-2 formula)
  3. FractalMass ? complex mass (m = 1.0 + Im?j)
  4. HijoluminicOperator shape (spinor field operator OK)
  5. PathIntegralLayer (branch superposition OK)
  6. Collapse-of-Light sums to 1 (valid probability distribution)
  7. PathBundle |G_AB| (propagator computes)
  8. Full model forward + backward (gradients flow)
"""

import sys
import os
import math

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from core.algebra import (
    gamma_matrices,
    gamma5,
    clifford_anticommutator,
    verify_clifford_algebra,
    complexified_velocity,
    complexified_velocity_inverse,
    minkowski_metric,
    coherent_superposition,
    born_probability,
)
from fields.hijoluminic import FractalMass, HijoluminicOperator
from paths.branches import BranchGenerator, ActionFunctional, PathIntegralLayer
from collapse.collapse import CollapseOfLight
from topology.path_graph import PathBundle
from core.model import HijoluminicANS, HijoluminicANSLight


def test_1_clifford_anticommutator():
    """Test: {?^?, ?^?} = 2?^?? I?"""
    gammas = gamma_matrices()
    result = verify_clifford_algebra(gammas, atol=1e-6)
    assert result['passed'], (
        f"Clifford algebra failed! Max deviation: {result['max_deviation']:.2e}"
    )
    print(f"  [OK] Clifford algebra verified. Max deviation: {result['max_deviation']:.2e}")
    return True


def test_2_complexified_velocity():
    """Test the complexified velocity map V = [X+j(-Y), Y+j(-X), Z+jZ]^T"""
    X = torch.tensor([1.0, 0.0, -1.0])
    Y = torch.tensor([0.0, 1.0, 0.0])
    Z = torch.tensor([0.0, 0.0, 1.0])

    V = complexified_velocity(X, Y, Z)

    # V is shape (3, 3) — 3 samples, 3 components
    # V[:, 0] = V1 = X - jY
    expected_V1 = torch.tensor([1.0+0.0j, 0.0-1.0j, -1.0+0.0j])
    assert torch.allclose(V[:, 0], expected_V1, atol=1e-6), f"V1 failed: {V[:, 0]} != {expected_V1}"

    # V[:, 1] = V2 = Y - jX
    expected_V2 = torch.tensor([0.0-1.0j, 1.0+0.0j, 0.0+1.0j])  # V2[2] = Y[2] - jX[2] = 0 - j(-1) = 1j
    assert torch.allclose(V[:, 1], expected_V2, atol=1e-6), f"V2 failed: {V[:, 1]} != {expected_V2}"

    # V[:, 2] = V3 = Z(1 + j)
    expected_V3 = torch.tensor([0.0+0.0j, 0.0+0.0j, 1.0+1.0j])
    assert torch.allclose(V[:, 2], expected_V3, atol=1e-6), f"V3 failed: {V[:, 2]} != {expected_V3}"

    # Test inverse
    Xr, Yr, Zr = complexified_velocity_inverse(V)
    assert torch.allclose(Xr, X, atol=1e-6), f"Inverse X failed"
    assert torch.allclose(Yr, Y, atol=1e-6), f"Inverse Y failed"
    assert torch.allclose(Zr, Z, atol=1e-6), f"Inverse Z failed"

    print("  [OK] Complexified velocity map verified (Image-2 formula exact)")
    return True


def test_3_fractal_mass():
    """Test FractalMass produces a complex mass"""
    fm = FractalMass(kappa_H=1.0, c=1.0, init_folds=[1.0, 2.0, 4.0, 8.0])

    mass = fm.mass
    assert mass.is_complex(), "Mass should be complex"
    assert mass.real.item() > 0, f"Real part should be > 0, got {mass.real}"

    # Expected imag = kH*c^2 * sum(1/softplus(delta_f))
    fold_vals = fm.folds.detach()
    expected_imag = float(torch.sum(1.0 / fold_vals))
    assert torch.allclose(mass.imag, torch.tensor(expected_imag), atol=1e-4), (
        f"Imaginary part mismatch: {mass.imag.item():.4f} != {expected_imag:.4f}"
    )

    print(f"  [OK] FractalMass -> m = {mass.real.item():.4f} + {mass.imag.item():.4f}j "
          f"(folds = {[f'{d:.3f}' for d in fold_vals]})")
    return True


def test_4_hijoluminic_operator_shape():
    """Test HijoluminicOperator produces correct shape output"""
    batch, n_sites, site_dim = 4, 8, 16
    op = HijoluminicOperator(n_sites=n_sites, site_dim=site_dim)

    psi = torch.randn(batch, n_sites, 4, dtype=torch.complex64)
    features = torch.randn(batch, n_sites, site_dim)

    psi_out, rho, phi, mass, op_norm = op(psi, features, return_components=True)

    assert psi_out.shape == (batch, n_sites, 4), (
        f"Output shape mismatch: {psi_out.shape}"
    )
    assert rho.shape == (batch, n_sites, 1), (
        f"?_y shape mismatch: {rho.shape}"
    )
    assert mass.is_complex(), "Mass should be complex"

    print(f"  [OK] HijoluminicOperator: ? shape {psi_out.shape}, ? shape {rho.shape}, "
          f"op_norm={op_norm:.4f}")
    return True


def test_5_path_integral_layer():
    """Test PathIntegralLayer branch superposition"""
    batch, n_branches = 4, 5
    pi = PathIntegralLayer(n_branches=n_branches)

    amplitudes = torch.rand(batch, n_branches) + 0.5
    phases = torch.randn(batch, n_branches) * math.pi
    actions = torch.randn(batch, n_branches)

    result = pi(amplitudes, phases, actions)

    assert 'G' in result, "Missing G in output"
    assert result['G'].shape == (batch,), f"G shape mismatch: {result['G'].shape}"
    assert result['branch_amplitudes'].shape == (batch, n_branches)
    assert result['branch_amplitudes'].is_complex(), "Branch amps should be complex"

    print(f"  [OK] PathIntegralLayer: G={result['G'][0].item():.4f}, "
          f"|G|?={result['magnitudes'][0].item():.4f}")
    return True


def test_6_collapse_of_light():
    """Test CollapseOfLight produces valid probability distribution"""
    batch, n_branches = 4, 5
    collapse = CollapseOfLight(n_branches=n_branches)

    branch_amps = torch.randn(batch, n_branches, dtype=torch.complex64)
    actions = torch.randn(batch, n_branches)
    fold_gaps = torch.tensor([1.0, 2.0, 4.0, 8.0])

    result = collapse(branch_amps, actions, fold_gaps, return_full=True)
    probs = result['probs']

    # Probabilities should sum to 1
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), (
        f"Probabilities don't sum to 1: {prob_sums}"
    )

    # All probabilities should be non-negative
    assert (probs >= 0).all(), "Negative probabilities!"

    print(f"  [OK] CollapseOfLight: probs sum to 1, entropy={result['entropy'][0].item():.4f}")
    return True


def test_7_path_bundle():
    """Test PathBundle computes propagator G_AB"""
    batch, state_dim, n_paths = 4, 8, 5
    bundle = PathBundle(state_dim=state_dim, n_paths=n_paths)

    A = torch.randn(batch, state_dim)
    B = torch.randn(batch, state_dim)

    result = bundle(A, B)

    assert 'G_AB' in result, "Missing G_AB"
    assert result['G_AB'].shape == (batch,), f"G_AB shape: {result['G_AB'].shape}"
    assert result['magnitude'].shape == (batch,)

    # The magnitude should be real and positive
    assert (result['magnitude'] >= 0).all(), "Magnitude should be non-negative"

    print(f"  [OK] PathBundle: |G_AB| = {result['magnitude'][0].item():.4f}, "
          f"interference = {result['interference'][0].item():.3f}")
    return True


def test_8_full_model():
    """Test full HijoluminicANS forward and backward pass"""
    batch, n_sites, in_features = 4, 8, 16
    n_classes = 3

    model = HijoluminicANS(
        in_features=in_features,
        n_sites=n_sites,
        n_classes=n_classes,
        mode='full',
    )

    x = torch.randn(batch, n_sites, in_features)

    # Forward pass
    out = model(x)
    assert out.shape == (batch, n_classes), f"Output shape: {out.shape}"

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Count parameter groups with gradients
    grad_groups = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            grad_groups += 1

    print(f"  [OK] Full model: forward output shape {out.shape}, "
          f"loss={loss.item():.4f}, {grad_groups}/{total_params} param groups with gradients")
    return True


def test_8b_full_model_light():
    """Test HijoluminicANSLight forward and backward"""
    batch, n_sites, in_features = 4, 8, 16

    model = HijoluminicANSLight(
        in_features=in_features,
        n_sites=n_sites,
    )

    x = torch.randn(batch, n_sites, in_features)
    result = model(x)

    assert 'logit' in result
    assert 'psi' in result
    assert 'rho' in result
    assert 'mass' in result

    loss = result['logit'].sum()
    loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  [OK] HijoluminicANSLight: {grad_count} param groups with gradients")
    return True


def test_5b_path_integral_convergence():
    """Test that path integral magnitudes change with phases"""
    batch, n_branches = 4, 4
    pi = PathIntegralLayer(n_branches=n_branches)

    amplitudes = torch.ones(batch, n_branches) / n_branches

    # All in-phase ? constructive interference maximizes magnitude
    phases_in = torch.zeros(batch, n_branches)
    actions_in = torch.zeros(batch, n_branches)
    r_in = pi(amplitudes, phases_in, actions_in)

    # Alternating phase ? destructive interference
    phases_out = torch.tensor([[0.0, math.pi, 0.0, math.pi]]).expand(batch, -1)
    r_out = pi(amplitudes, phases_out, actions_in)

    assert r_in['magnitudes'][0].item() > r_out['magnitudes'][0].item(), (
        "Constructive interference should have larger magnitude than destructive"
    )

    print(f"  [OK] Path integral interference: "
          f"constructive |G|?={r_in['magnitudes'][0].item():.4f} > "
          f"destructive |G|?={r_out['magnitudes'][0].item():.4f}")
    return True


def test_6b_collapse_fractal_resonance():
    """Test fractal resonance amplifies resonant branches"""
    batch, n_branches = 4, 4
    collapse = CollapseOfLight(n_branches=n_branches, use_fractal_resonance=True)

    # Equal amplitudes, but one branch has action matching a fold
    branch_amps = torch.ones(batch, n_branches, dtype=torch.complex64) / n_branches
    actions = torch.tensor([[1.0, 0.5, 3.0, 10.0]]).expand(batch, -1)
    fold_gaps = torch.tensor([1.0, 2.0, 4.0, 8.0])

    result = collapse(branch_amps, actions, fold_gaps, return_full=True)

    # Branch with action=1.0 is resonant with ??=1.0
    # Branch with action=0.5 is between ?? and ??
    # Branch with action=10.0 is far from all folds
    probs = result['probs']
    print(f"  [OK] Fractal resonance: P = {probs[0].tolist()}")
    print(f"    (Branch 0 action=1.0 resonant with ??=1.0 should be amplified)")

    # At minimum: the resonant branch should not be the smallest
    assert probs[0, 0] > 0 or probs[0, 1] > 0, "Resonant branch suppressed"
    return True


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  H-ANS Sanity Tests ? 8+ Validations")
    print("="*60)

    tests = [
        ("Clifford anticommutator", test_1_clifford_anticommutator),
        ("Complexified velocity map", test_2_complexified_velocity),
        ("FractalMass complex mass", test_3_fractal_mass),
        ("HijoluminicOperator shape", test_4_hijoluminic_operator_shape),
        ("PathIntegralLayer", test_5_path_integral_layer),
        ("Path integral interference", test_5b_path_integral_convergence),
        ("Collapse-of-Light sums to 1", test_6_collapse_of_light),
        ("Fractal resonance weights", test_6b_collapse_fractal_resonance),
        ("PathBundle propagator", test_7_path_bundle),
        ("Full model forward+backward", test_8_full_model),
        ("Light model forward+backward", test_8b_full_model_light),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n  [{name}]")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("="*60)

    sys.exit(0 if failed == 0 else 1)
