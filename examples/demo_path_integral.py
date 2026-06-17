"""
examples/demo_path_integral.py ? A?B path-integral constructive interference.

Demonstrates how the path-integral layer learns to align branch phases
for constructive interference at the target (B).

Shows the propagator magnitude |G_AB| growing over training as paths
phase-align.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn

from topology.path_graph import PathBundle, BundleCollapse


def main():
    print("=" * 60)
    print("  Demo: A?B Path-Integral Constructive Interference")
    print("=" * 60)

    # Setup
    state_dim = 4
    n_paths = 5
    n_knots = 3
    batch = 8

    # Fixed A and B points
    A = torch.randn(batch, state_dim)
    B = torch.zeros(batch, state_dim)  # Target: collapse to origin

    # Build bundle
    bundle = PathBundle(
        state_dim=state_dim,
        n_paths=n_paths,
        n_knots_per_path=n_knots,
    )
    collapser = BundleCollapse(n_paths=n_paths)

    # Optimizer
    optimizer = torch.optim.Adam(bundle.parameters(), lr=0.02)

    print(f"\n  State dim: {state_dim}, Paths: {n_paths}, Knots per path: {n_knots}")
    print(f"  Trainable params: {sum(p.numel() for p in bundle.parameters())}")

    # Training: maximize |G_AB|? (constructive interference toward B)
    n_epochs = 200
    mags = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        result = bundle(A, B, return_detailed=False)

        # Loss = -|G_AB|? + regularization on path actions
        mag = result['magnitude']
        actions = result['per_path_actions']

        # Primary loss: negative magnitude (want constructive interference)
        loss_mag = -mag.mean()

        # Regularization: keep actions reasonable
        loss_action = actions.abs().mean() * 0.01

        # Encourage diverse paths (differing actions)
        loss_diversity = -actions.std(dim=-1).mean() * 0.01

        loss = loss_mag + loss_action + loss_diversity
        loss.backward()
        optimizer.step()

        mags.append(mag.mean().item())

        if epoch % 40 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: |G_AB|={mag.mean().item():.4f}, "
                  f"loss={loss.item():.4f}, "
                  f"actions=[{actions[0,0].item():.2f}, {actions[0,1].item():.2f}, ...]")

    # Evaluate
    with torch.no_grad():
        result = bundle(A, B, return_detailed=True)
        final_mag = result['magnitude'].mean().item()
        interference = result['interference'].mean().item()
        collapse = collapser(result, fold_gaps=torch.tensor([1.0, 2.0, 4.0, 8.0]))

    print(f"\n  Final |G_AB| = {final_mag:.4f} "
          f"(start ~ {mags[0]:.4f}, growth: {final_mag - mags[0]:.4f})")
    print(f"  Interference efficiency: {interference:.4f} "
          f"(>1.0 = constructive)")
    print(f"  Collapse entropy: {collapse['entropy'].mean().item():.4f}")
    print("=" * 60)
    print("  Demo complete ? Path interference learning ?")
    print("=" * 60)


if __name__ == '__main__':
    main()
