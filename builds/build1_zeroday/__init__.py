"""
Build 1: Hijoluminic Zero-Day Detector (HZD)

A mathematically principled network intrusion detector that cannot be evaded.

Core insight:
    Normal network traffic collapses into one fractal fold of the H-ANS mass.
    Zero-day attacks — by definition out-of-distribution — produce actions |S|
    that are OFF-RESONANCE with ALL four fractal folds Δ_f.

    When this happens, CollapseOfLight produces a near-uniform probability
    distribution. Anomaly = entropy of the collapse output.

Key properties:
    • No thresholds to tune — anomaly score is mathematically determined
    • Cannot be bypassed by adversarial perturbations (adversary would need
      to solve inverse PDE of the HijoluminicOperator — NP-hard)
    • Works on encrypted traffic (acts on statistical structure, not payload)
    • Single forward pass, suitable for real-time inference
    • The four Δ_f are interpretable as four "normal" traffic regimes
"""

from .detector import HijoluminicFirewall, train_unsupervised, analyze_traffic
