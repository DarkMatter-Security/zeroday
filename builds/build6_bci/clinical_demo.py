"""
build6_bci/clinical_demo.py -- Clinical BCI Perceptual Uncertainty Demo.

Demonstrates the H-ANS BCI model for clinical/neuroscience applications.

Features:
  - Works with synthetic data (no hardware needed)
  - Can load PhysioNet EEG Motor Movement/Imagery Dataset (free public dataset)
  - Produces quantitative perceptual uncertainty metrics
  - Generates clinical-style report

Requirements:
  pip install numpy torch scipy
  (optional) pip install wfdb  # for PhysioNet EEG loading

Usage:
  python -m builds.build6_bci.clinical_demo
  python -m builds.build6_bci.clinical_demo --use-eeg  # try real EEG
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from builds.build6_bci.cortex import (
    HijoluminicCortex, NeuralCortexConfig, generate_neural_response, STIMULI
)
NeuralConfig = NeuralCortexConfig  # alias for readability


# -----------------------------------------------------------------------
#  Public EEG Dataset Loader
# -----------------------------------------------------------------------

class PhysioNetEEGLoader:
    """
    Loads the PhysioNet EEG Motor Movement/Imagery Dataset.

    This dataset is freely available at:
      https://physionet.org/content/eegmmidb/1.0.0/

    The dataset contains 64-channel EEG recordings from 109 subjects
    performing motor/imagery tasks.

    This loader provides a simplified interface that returns data
    in the format expected by the BCI model.

    If the dataset is not installed, falls back gracefully.
    """

    DATASET_URL = "https://physionet.org/content/eegmmidb/1.0.0/"
    RECORDS = [
        'S001/S001R01', 'S001/S001R02', 'S001/S001R03', 'S001/S001R04',
    ]

    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir
        self.available = False
        self._check_availability()

    def _check_availability(self):
        """Check if the PhysioNet dataset is available locally."""
        try:
            import wfdb
            self.wfdb_available = True
        except ImportError:
            self.wfdb_available = False

        if self.data_dir and Path(self.data_dir).exists():
            self.available = True
        else:
            self.available = False

    def load_sample(self, n_electrodes: int = 16,
                    n_time_windows: int = 128) -> torch.Tensor | None:
        """
        Load a sample from the EEG dataset.

        Returns tensor of shape (1, n_electrodes, n_time_windows) or None.
        """
        if not self.available or not self.wfdb_available:
            return None

        try:
            import wfdb
            # Try loading first record
            record_path = os.path.join(self.data_dir, self.RECORDS[0])
            record = wfdb.rdrecord(record_path)
            data = record.p_signal  # (time, channels)

            # Select subset of channels
            n_chan = min(data.shape[1], n_electrodes)
            data = data[:n_time_windows, :n_chan]  # (time, channels)

            # Normalize
            data = (data - data.mean(axis=0, keepdims=True)) / \
                   (data.std(axis=0, keepdims=True) + 1e-8)

            # Transpose to (1, channels, time)
            return torch.from_numpy(data.T.astype(np.float32)).unsqueeze(0)

        except Exception as e:
            print(f"    [EEG loader] Warning: {e}")
            return None

    @staticmethod
    def download_instructions() -> str:
        return f"""
    To download the PhysioNet EEG dataset:

    1. Visit: {PhysioNetEEGLoader.DATASET_URL}
    2. Download the dataset files (free, requires registration)
    3. Extract to a folder (e.g., ./eeg_data/)
    4. Run: pip install wfdb
    5. Run this script with: --eeg-dir ./eeg_data --use-eeg
        """


# -----------------------------------------------------------------------
#  Clinical Analysis Functions
# -----------------------------------------------------------------------

class ClinicalPerceptualAnalyzer:
    """
    Analyzes perceptual uncertainty patterns for clinical applications.

    Provides metrics useful for:
      - Schizophrenia research (altered perceptual bistability)
      - Autism spectrum (atypical sensory processing)
      - PTSD (hypervigilance -> altered uncertainty thresholds)
      - BCI safety (confidence-weighted decoding)
    """

    def __init__(self, model: HijoluminicCortex):
        self.model = model

    def analyze_stimulus(self,
                         stimulus: torch.Tensor,
                         label: str = "stimulus") -> dict:
        """
        Run a single stimulus through the model and return clinical metrics.

        Returns:
            Dict with entropy, confidence, superposition_ratio, etc.
        """
        with torch.no_grad():
            output = self.model(stimulus, return_all=True)

        probs = output['collapse_probs'].squeeze(0)
        entropy = output['collapse_entropy'].item()
        n_branches = probs.shape[-1]
        max_entropy = math.log(n_branches)

        # Clinical metrics
        confidence = probs.max().item()
        superposition_ratio = entropy / max_entropy  # 0 = collapsed, 1 = max superposition
        prob_gap = probs.max().item() - probs.min().item()
        dominant_branch = probs.argmax().item()

        # Bistability metric: how many branches have non-negligible probability
        significant_branches = (probs > 0.1).sum().item()

        return {
            'label': label,
            'entropy': round(entropy, 4),
            'max_entropy': round(max_entropy, 4),
            'normalized_entropy': round(superposition_ratio, 4),
            'confidence': round(confidence, 4),
            'prob_gap': round(prob_gap, 4),
            'dominant_branch': dominant_branch,
            'significant_branches': significant_branches,
            'is_ambiguous': superposition_ratio > 0.3,
            'is_collapsed': confidence > 0.9,
        }

    def generate_clinical_report(self, results: list[dict]) -> str:
        """Generate a human-readable clinical report."""
        lines = []
        lines.append("=" * 65)
        lines.append("  H-ANS BCI Perceptual Uncertainty -- Clinical Report")
        lines.append("=" * 65)
        lines.append("")

        for r in results:
            status = "[!] AMBIGUOUS" if r['is_ambiguous'] else "[OK] CLEAR"
            lines.append(f"  {r['label']:25s}  |  "
                         f"Entropy: {r['entropy']:.4f}  |  "
                         f"Confidence: {r['confidence']:.4f}  |  "
                         f"{status}")

        lines.append("")
        lines.append("-" * 65)

        # Summary statistics
        collapsed = [r for r in results if r['is_collapsed']]
        ambiguous = [r for r in results if r['is_ambiguous']]
        lines.append(f"  Clear interpretations: {len(collapsed)}/{len(results)}")
        lines.append(f"  Ambiguous interpretations: {len(ambiguous)}/{len(results)}")

        # Entropy gap analysis
        if len(results) >= 2:
            entropies = [r['entropy'] for r in results]
            lines.append(f"  Entropy range: [{min(entropies):.4f}, {max(entropies):.4f}]")
            lines.append(f"  Entropy gap (max - min): {max(entropies) - min(entropies):.4f}")
            lines.append(f"  Entropy gap is a biomarker of perceptual discrimination.")

        lines.append("")
        lines.append("  Clinical Interpretation:")
        lines.append("    - Low entropy (< 0.1) on clear stimuli = normal perceptual collapse")
        lines.append("    - High entropy (> 0.5) on ambiguous = healthy bistability")
        lines.append("    - Abnormally low entropy on all stimuli may indicate:")
        lines.append("        . Perceptual rigidity (e.g., anxiety, PTSD hypervigilance)")
        lines.append("        . Reduced cognitive flexibility")
        lines.append("    - Abnormally high entropy on clear stimuli may indicate:")
        lines.append("        . Perceptual integration deficits")
        lines.append("        . Possible neurological screening indicator")
        lines.append("")
        lines.append("  NOTE: This is a research tool. Not cleared for clinical diagnosis.")
        lines.append("=" * 65)

        return "\n".join(lines)


# -----------------------------------------------------------------------
#  Demo
# -----------------------------------------------------------------------

def demo(use_eeg: bool = False, eeg_dir: str | None = None):
    print("=" * 65)
    print("  Build 6: BCI Perceptual Uncertainty -- Clinical Demo")
    print("=" * 65)

    # Initialize model
    config = NeuralConfig(n_electrodes=16, n_frequency_bands=3, n_time_windows=64)
    model = HijoluminicCortex(config=config)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params}")
    print(f"  Neural config: {config.n_electrodes} electrodes, "
          f"{config.n_frequency_bands} bands, {config.n_time_windows} timepts")

    # Load EEG if requested
    eeg_sample = None
    if use_eeg:
        print("\n  Attempting to load PhysioNet EEG data...")
        loader = PhysioNetEEGLoader(data_dir=eeg_dir)
        eeg_sample = loader.load_sample(
            n_electrodes=config.n_electrodes,
            n_time_windows=config.n_time_windows,
        )
        if eeg_sample is not None:
            print(f"  [OK] EEG data loaded: shape {eeg_sample.shape}")
        else:
            print(f"  [!] EEG data not available.")
            if eeg_dir:
                print(PhysioNetEEGLoader.download_instructions())
            print("  Falling back to synthetic data.\n")

    # Generate test stimuli
    print("\n  Generating test stimuli...")
    eeg_shape = (1, config.n_electrodes, config.n_time_windows)
    stimuli = []

    # Stimulus types
    stimulus_specs = ["clear_image", "rubin_vase", "necker_cube", "noise"]

    for name in stimulus_specs:
        stim = generate_neural_response(
            eeg_shape, stimulus=name, noise_level=0.1
        )
        stimuli.append((name, stim))

    # If we have real EEG data, add it as a "clinical" sample
    if eeg_sample is not None:
        # Pad/trim to match expected dimensions
        B, E, T = eeg_sample.shape
        target_e = config.n_electrodes
        target_t = config.n_time_windows

        eeg_resized = torch.zeros(1, target_e, target_t)
        eeg_resized[0, :min(E, target_e), :min(T, target_t)] = \
            eeg_sample[0, :min(E, target_e), :min(T, target_t)]
        stimuli.append(("real_eeg_physionet", eeg_resized))

    # Analyze with pre-training model
    print("\n  Phase 1: Pre-training perceptual analysis...")
    analyzer = ClinicalPerceptualAnalyzer(model)
    pre_results = []

    for name, stim in stimuli:
        result = analyzer.analyze_stimulus(stim, label=name)
        pre_results.append(result)
        status = "[!] AMBIGUOUS" if result['is_ambiguous'] else "[OK] CLEAR"
        print(f"    {name:25s}  entropy={result['entropy']:.4f}  "
              f"confidence={result['confidence']:.4f}  {status}")
    print(f"    (Model not yet trained -- expect both types to be uncertain)")

    # Train on perceptual decision task
    print("\n  Phase 2: Training perceptual discrimination...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_steps = 100

    for step in range(n_steps):
        optimizer.zero_grad()

        # Clear stimulus (target: collapsed, dominant class 0)
        clear = generate_neural_response(eeg_shape, stimulus='clear_image', noise_level=0.1)
        out_clear = model(clear, return_all=True)
        loss_clear = F.cross_entropy(out_clear['interpretation_logits'],
                                      torch.zeros(clear.shape[0], dtype=torch.long))

        # Ambiguous stimulus (target: collapsed, dominant class 1)
        ambig = generate_neural_response(eeg_shape, stimulus='rubin_vase', noise_level=0.1)
        out_ambig = model(ambig, return_all=True)
        loss_ambig = F.cross_entropy(out_ambig['interpretation_logits'],
                                      torch.ones(ambig.shape[0], dtype=torch.long))

        # Uncertainty loss: maximize entropy gap
        clear_entropy = out_clear['collapse_entropy'].mean()
        ambig_entropy = out_ambig['collapse_entropy'].mean()

        loss = loss_clear + loss_ambig

        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == n_steps - 1:
            with torch.no_grad():
                ce = out_clear['collapse_entropy'].mean().item()
                ae = out_ambig['collapse_entropy'].mean().item()
            print(f"    Step {step:3d}: loss={loss.item():.4f}, "
                  f"clear_entropy={ce:.4f}, ambig_entropy={ae:.4f}")

    # Post-training analysis
    print("\n  Phase 3: Post-training perceptual analysis...")
    post_results = []
    for name, stim in stimuli:
        result = analyzer.analyze_stimulus(stim, label=name)
        post_results.append(result)

    # Print results table
    print()
    print(f"  {'Stimulus':25s}  {'Entropy':>8s}  {'Confidence':>10s}  {'Status':>12s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*12}")
    for r in post_results:
        status = "AMBIGUOUS" if r['is_ambiguous'] else "CLEAR"
        print(f"  {r['label']:25s}  {r['entropy']:8.4f}  "
              f"{r['confidence']:10.4f}  {status:>12s}")

    # Clinical report
    print(f"\n{analyzer.generate_clinical_report(post_results)}")

    # Key scientific result
    print("\n  Key scientific result:")
    clear_ent = [r['entropy'] for r in post_results if not r['is_ambiguous']]
    ambig_ent = [r['entropy'] for r in post_results if r['is_ambiguous']]
    if clear_ent and ambig_ent:
        gap = sum(ambig_ent)/len(ambig_ent) - sum(clear_ent)/len(clear_ent)
        print(f"    Mean clear entropy:    {sum(clear_ent)/len(clear_ent):.4f}")
        print(f"    Mean ambiguous entropy: {sum(ambig_ent)/len(ambig_ent):.4f}")
        print(f"    Entropy gap:            {gap:.4f}")
        if gap > 0.3:
            print(f"    [OK] Perceptual discrimination learned successfully.")
        else:
            print(f"    [-] Perceptual discrimination needs more training.")

    print("=" * 65)
    print("  BCI Clinical Demo Complete OK")
    print("=" * 65)

    return post_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="H-ANS BCI Perceptual Uncertainty Clinical Demo"
    )
    parser.add_argument('--use-eeg', action='store_true',
                        help='Attempt to load PhysioNet EEG data')
    parser.add_argument('--eeg-dir', type=str, default=None,
                        help='Path to PhysioNet EEG dataset directory')
    args = parser.parse_args()

    demo(use_eeg=args.use_eeg, eeg_dir=args.eeg_dir)
