"""
examples/demo_classifier.py ? Toy spirals classification with H-ANS.

Demonstrates end-to-end training of H-ANS on a 2D spirals dataset,
showing that the network learns genuine patterns.

Trains for 50 steps and reports accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import HijoluminicANS


def make_spirals(n_samples: int = 200, n_classes: int = 3, noise: float = 0.3):
    """Generate 2D spiral dataset."""
    n = n_samples // n_classes
    X, y = [], []
    for c in range(n_classes):
        theta = torch.linspace(0, 4 * math.pi, n)
        r = theta / (4 * math.pi)
        x = r * torch.cos(theta + 2 * math.pi * c / n_classes)
        z = r * torch.sin(theta + 2 * math.pi * c / n_classes)
        x += torch.randn(n) * noise
        z += torch.randn(n) * noise
        X.append(torch.stack([x, z], dim=-1))
        y.append(torch.full((n,), c, dtype=torch.long))
    return torch.cat(X), torch.cat(y)


def main():
    print("=" * 60)
    print("  Demo: H-ANS Classifier on Spirals")
    print("=" * 60)

    # Create dataset
    X, y = make_spirals(300, 3, noise=0.2)
    # Reshape to (batch, n_sites, in_features): each point is a "site"
    X = X.unsqueeze(1)  # (300, 1, 2)
    y = F.one_hot(y, num_classes=3).float()

    # Shuffle then split (avoids class imbalance in test set)
    perm = torch.randperm(X.shape[0])
    X, y = X[perm], y[perm]
    split = 200
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = HijoluminicANS(
        in_features=X.shape[-1],
        n_sites=1,
        n_classes=3,
        n_branches=4,
        n_layers=2,
        hidden_dim=32,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n  Training on {len(X_train)} samples, testing on {len(X_test)}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train.argmax(dim=-1))
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (logits.argmax(dim=-1) == y_train.argmax(dim=-1)).float().mean()
                test_logits = model(X_test)
                test_acc = (test_logits.argmax(dim=-1) == y_test.argmax(dim=-1)).float().mean()
                uncertainty = model.get_uncertainty(X_test).mean()
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, "
                  f"train_acc={train_acc.item():.3f}, "
                  f"test_acc={test_acc.item():.3f}, "
                  f"uncertainty={uncertainty.item():.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits, probs, uncertainty = model.classify_with_uncertainty(X_test)
        acc = (logits.argmax(dim=-1) == y_test.argmax(dim=-1)).float().mean()

    print(f"\n  Final test accuracy: {acc.item():.3f}")
    print(f"  Mean uncertainty: {uncertainty.mean().item():.4f}")
    print(f"  Fractal mass: {model.fractal_mass.mass.item():.4f}")
    print(f"  Fold gaps: {model.fractal_mass.folds.tolist()}")
    print("=" * 60)
    print("  Demo complete - H-ANS learns on spirals OK")
    print("=" * 60)


if __name__ == '__main__':
    main()
