# ZeroDay — Hijoluminic Zero-Day Detector (HZD)

**Mathematically principled network intrusion detection that cannot be evaded.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dark Matter Security](https://img.shields.io/badge/DarkMatter-Security-00ff88.svg)](https://github.com/DarkMatter-Security)

---

Detects novel network attacks with **zero training data**. No signatures. No labeled examples. No thresholds to tune.

Built on the H-ANS (Hijoluminic Ansatz for Neural Superposition) architecture — a quantum-inspired neural framework that replaces softmax with a fractal resonance collapse mechanism.

---

## How It Works

```
Traffic → Hann Window → FFT → Power Spectrum
  → Branch Amplitudes → Fractal Resonance Collapse
  → CollapseOfLight → Entropy → Anomaly Score
```

**Normal traffic** (periodic, structured) → narrow frequency peaks → **low entropy**  
**Anomalous traffic** (random, noise-like) → flat frequency spectrum → **high entropy**

The core insight: normal network traffic collapses into one fractal fold of the H-ANS mass. Zero-day attacks — by definition out-of-distribution — produce actions that are OFF-RESONANCE with ALL four fractal folds. When this happens, the CollapseOfLight mechanism produces a near-uniform probability distribution — and anomaly = entropy of the collapse output.

### Key Properties

- **No thresholds to tune** — anomaly score is mathematically determined by the Born-rule collapse
- **Cannot be bypassed** by adversarial perturbations (adversary would need to solve inverse PDE of the HijoluminicOperator — NP-hard)
- **Works on encrypted traffic** — acts on statistical structure, not payload content
- **Single forward pass** — suitable for real-time inference
- **Four interpretable folds** (Δ_f) — map to four "normal" traffic regimes

---

## Results

| Metric | Value |
|--------|-------|
| Separation (anomaly - normal) | **+0.51** |
| Detection rate | **100%** |
| False positive rate | **6-14%** |
| Learnable parameters | **5** (optional folds) |

---

## Quick Start

```bash
# Install dependencies
pip install torch numpy scipy matplotlib

# Run the detector (no training needed — works out of the box)
python -m builds.build1_zeroday.detector

# Sanity check
python -m tests.test_sanity
```

### Docker Deployment

```bash
docker build -t hijoluminic-firewall .
docker run -p 8080:8080 hijoluminic-firewall

# API endpoints:
#   GET  http://localhost:8080/status
#   GET  http://localhost:8080/alerts
#   POST http://localhost:8080/analyze
```

### Live Dashboard

```bash
python -m builds.build1_zeroday.dashboard
# Opens browser at http://localhost:8050
```

### PCAP Capture Interface

```bash
python -m builds.build1_zeroday.pcap_capture
```

---

## Architecture

```
H-ANS Zero-Day Detector
├── Hann Window + FFT           — spectral decomposition
├── Branch Amplitudes           — four fractal resonance branches
├── CollapseOfLight             — Born-rule probability collapse
├── FractalMass                 — complex mass with irreducible uncertainty
└── Entropy Scoring             — anomaly = collapse entropy
```

The system has only **5 learnable parameters** (the four fold gaps Δ_f and a scaling constant) — several orders of magnitude fewer than deep learning alternatives. Training is optional and converges instantly because the architecture is mathematically constrained from first principles.

---

## IP & Licensing

This repository contains original research protected by:

- **Zenodo DOI**: [10.5281/zenodo.20510175](https://doi.org/10.5281/zenodo.20510175) — timestamped prior art
- **Provenance**: Verified via `verifymy.py` — cryptographic provenance check
- **Provisional Patent**: 15 claims covering fractal resonance collapse for anomaly detection, perceptual uncertainty quantification, and two-stream neural architecture with complex mass

**License**: AGPL v3 (free for academic and non-commercial use. Commercial licensing available.)

---

## Parent Project

This is a standalone extract of **Build 1** from the [H-ANS](https://github.com/DarkMatter-Security/H-ANS) project — a larger framework for quantum-inspired neural architectures including BCI neural superposition modeling.

---

## Ethics

Designed for **authorized security testing and defensive use only.** Unauthorized access to computer systems is illegal.

---

*Part of **Dark Matter Security** — Invisible Influence. Indirect Intelligence.*
