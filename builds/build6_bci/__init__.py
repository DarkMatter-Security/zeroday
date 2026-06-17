"""
Build 6: Hijoluminic Neural Cortex — BCI Neural Superposition Model

A full implementation of a brain-computer interface model based on the
H-ANS architecture, treating neural recordings as a field-theoretic
system where cognitive states exist as superpositions.

Core principles:
    • Each electrode/sensor is a "site" in the Hijoluminic field
    • ρ_y(x) = band-limited power in frequency band y at site x
    • Φ(x) = phase coherence between adjacent electrodes
    • Gauge connection = learned functional connectome
    • Cognitive state = superposition of basis interpretations
    • The complex mass's imaginary part = the brain's own uncertainty
    • Ambiguous stimuli → stable superposition with multiple branches

Key prediction:
    During ambiguous perception (e.g., Necker cube, Rubin vase),
    the model should maintain high-entropy superposition with two
    dominant branches, collapsing only when a decision is made.
"""

from .cortex import HijoluminicCortex, train_cortex, demo
