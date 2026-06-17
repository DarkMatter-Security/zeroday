"""
build1_zeroday/dashboard.py — Live Entropy Dashboard for the Hijoluminic Firewall.

Requires: streamlit (pip install streamlit)

Run: streamlit run builds/build1_zeroday/dashboard.py
     # or from project root: python -m streamlit run builds/build1_zeroday/dashboard.py

Opens a browser dashboard showing:
    - Real-time entropy graph
    - Anomaly score distribution
    - Frequency spectrum visualization
    - Alert history
    - Traffic statistics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import math
from collections import deque

import torch
import numpy as np

from builds.build1_zeroday.detector import (
    HijoluminicFirewall, TrafficConfig, FrequencyEncoder
)

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# ──────────────────────────────────────────────
#  Dashboard State
# ──────────────────────────────────────────────

class DashboardState:
    """Maintains rolling window of detection results for live display."""

    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.scores = deque(maxlen=max_history)
        self.entropies = deque(maxlen=max_history)
        self.anomaly_entropies = deque(maxlen=max_history)
        self.normal_entropies = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.start_time = time.time()
        self.total_samples = 0
        self.alert_count = 0

    def add(self, score: float, entropy: float, is_anomaly: bool):
        self.scores.append(score)
        self.entropies.append(entropy)
        self.timestamps.append(time.time() - self.start_time)
        self.total_samples += 1
        if is_anomaly:
            self.alert_count += 1
            self.anomaly_entropies.append(entropy)
        else:
            self.normal_entropies.append(entropy)

    def get_history(self):
        return {
            'scores': list(self.scores),
            'entropies': list(self.entropies),
            'timestamps': list(self.timestamps),
            'total': self.total_samples,
            'alerts': self.alert_count,
        }


# ──────────────────────────────────────────────
#  Traffic Generator (for live demo without hardware)
# ──────────────────────────────────────────────

def generate_live_traffic(n_sites: int = 64, n_features: int = 8):
    """Alternate between normal and anomalous traffic for demo display."""
    while True:
        # 4 normal batches, then 1 anomaly
        for _ in range(4):
            yield FrequencyEncoder.synthetic_traffic(
                1, n_sites, n_features, normal=True
            ), False
        yield FrequencyEncoder.synthetic_traffic(
            1, n_sites, n_features, normal=False
        ), True


# ──────────────────────────────────────────────
#  Dashboard UI
# ──────────────────────────────────────────────

def render_dashboard():
    st.set_page_config(
        page_title="H-ANS Zero-Day Detector",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ H-ANS Zero-Day Intrusion Detector")
    st.markdown("**Real-time anomaly detection via fractal resonance collapse**")

    # Sidebar config
    st.sidebar.header("Configuration")
    n_sites = st.sidebar.slider("Window size (sites)", 16, 128, 64, step=8)
    threshold = st.sidebar.slider("Anomaly threshold", 0.0, 1.0, 0.5, 0.05)
    update_interval = st.sidebar.slider("Update interval (ms)", 100, 2000, 500, step=100)

    # Initialize model
    config = TrafficConfig(n_features=8, n_sites=n_sites)
    model = HijoluminicFirewall(config=config, kappa_H=1.0)

    n_params = sum(p.numel() for p in model.parameters())
    st.sidebar.info(
        f"**Model**\n"
        f"- Architecture: FFT → {config.n_freqs} freq bins → Collapse → Entropy\n"
        f"- Parameters: {n_params}\n"
        f"- Branches: {config.n_branches}\n"
        f"- Max entropy: {math.log(config.n_branches):.3f}"
    )

    # Initialize state
    state = DashboardState(max_history=200)
    traffic_gen = generate_live_traffic(n_sites=n_sites, n_features=config.n_features)

    # Layout: 3 columns
    col1, col2, col3 = st.columns(3)

    # Metrics
    metrics_placeholder = col1.empty()
    alert_placeholder = col2.empty()
    status_placeholder = col3.empty()

    # Plots
    entropy_plot = st.empty()
    spectrum_plot = st.empty()
    history_plot = st.empty()

    # Main loop
    running = st.sidebar.checkbox("Running", value=True)

    while running:
        traffic, is_anomaly = next(traffic_gen)

        with torch.no_grad():
            out = model(traffic, return_all=True)
            score = out['scores'].item()
            entropy = out['entropy'].item()

            # Get frequency spectrum for visualization
            amps = out['branch_amps'].squeeze(0).numpy()
            probs = out['probs'].squeeze(0).numpy()

        state.add(score, entropy, is_anomaly)

        # ── Metrics ──
        max_entropy = math.log(config.n_branches)
        metrics_placeholder.metric(
            "Anomaly Score",
            f"{score:.3f}",
            f"{'ANOMALOUS' if score > threshold else 'NORMAL'}",
            delta_color="inverse",
        )
        alert_placeholder.metric(
            "Entropy",
            f"{entropy:.3f} / {max_entropy:.3f}",
            f"{entropy/max_entropy*100:.1f}% of max",
        )
        status_placeholder.metric(
            "Alerts / Total",
            f"{state.alert_count} / {state.total_samples}",
            f"{state.alert_count/max(1, state.total_samples)*100:.1f}% alert rate",
        )

        # ── Entropy Gauge ──
        fig_entropy = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=entropy,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Collapse Entropy"},
            delta={'reference': 0.5 * max_entropy},
            gauge={
                'axis': {'range': [None, max_entropy]},
                'bar': {'color': "red" if score > threshold else "green"},
                'steps': [
                    {'range': [0, 0.3*max_entropy], 'color': "lightgreen"},
                    {'range': [0.3*max_entropy, 0.7*max_entropy], 'color': "yellow"},
                    {'range': [0.7*max_entropy, max_entropy], 'color': "salmon"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * max_entropy,
                },
            }
        ))
        fig_entropy.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        entropy_plot.plotly_chart(fig_entropy, use_container_width=True)

        # ── Frequency Spectrum ──
        fig_spec = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Frequency Spectrum (Branch Amplitudes)",
                           "Collapse Probability Distribution"),
            vertical_spacing=0.15,
        )
        freqs = list(range(len(amps)))
        fig_spec.add_trace(
            go.Bar(x=freqs, y=amps, name="Amplitude",
                   marker_color='royalblue'),
            row=1, col=1,
        )
        fig_spec.add_trace(
            go.Bar(x=list(range(len(probs))), y=probs, name="Probability",
                   marker_color='coral'),
            row=2, col=1,
        )
        fig_spec.update_layout(height=400, showlegend=False,
                               margin=dict(l=20, r=20, t=40, b=20))
        fig_spec.update_xaxes(title_text="Frequency Bin", row=1, col=1)
        fig_spec.update_yaxes(title_text="Power", row=1, col=1)
        fig_spec.update_xaxes(title_text="Branch Index", row=2, col=1)
        fig_spec.update_yaxes(title_text="Probability", row=2, col=1)
        spectrum_plot.plotly_chart(fig_spec, use_container_width=True)

        # ── Score History ──
        hist = state.get_history()
        if len(hist['timestamps']) > 1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=hist['timestamps'],
                y=hist['scores'],
                mode='lines+markers',
                name='Anomaly Score',
                marker=dict(
                    color=['red' if s > threshold else 'green' for s in hist['scores']],
                    size=4,
                ),
                line=dict(color='gray', width=1),
            ))
            fig_hist.add_hline(y=threshold, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold ({threshold})")
            fig_hist.update_layout(
                title="Score History",
                xaxis_title="Time (s)",
                yaxis_title="Anomaly Score",
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis=dict(range=[0, 1]),
            )
            history_plot.plotly_chart(fig_hist, use_container_width=True)

        time.sleep(update_interval / 1000)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    if not STREAMLIT_AVAILABLE:
        print("ERROR: streamlit not installed.")
        print("Install with: pip install streamlit")
        print("\nRun with: streamlit run builds/build1_zeroday/dashboard.py")
        return

    render_dashboard()


if __name__ == '__main__':
    main()
