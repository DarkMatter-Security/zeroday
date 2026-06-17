"""
build1_zeroday/api_server.py — REST API for the Hijoluminic Firewall.

Provides a Docker-ready HTTP API for:
  - Health/status checks
  - Traffic analysis (POST raw data or PCAP)
  - Alert retrieval
  - Configuration

Run:  python -m builds.build1_zeroday.api_server
      # or via Docker: docker run -p 8080:8080 hijoluminic-firewall

Test: curl http://localhost:8080/status
      curl -X POST http://localhost:8080/analyze -H "Content-Type: application/json" -d '{...}'
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import math
from datetime import datetime
from threading import Lock

import torch
import numpy as np

from builds.build1_zeroday.detector import (
    HijoluminicFirewall, TrafficConfig, FrequencyEncoder,
    train_unsupervised, analyze_traffic
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# ──────────────────────────────────────────────
#  Alert Storage
# ──────────────────────────────────────────────

class AlertStore:
    """In-memory alert storage (replace with SQLite/Redis for production)."""

    def __init__(self, max_alerts: int = 1000):
        self.alerts = []
        self.max_alerts = max_alerts
        self.lock = Lock()

    def add(self, score: float, entropy: float, traffic_summary: str):
        with self.lock:
            self.alerts.append({
                'timestamp': datetime.utcnow().isoformat(),
                'score': round(score, 4),
                'entropy': round(entropy, 4),
                'traffic_summary': traffic_summary,
                'threshold_exceeded': score > 0.5,
            })
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]

    def get_recent(self, n: int = 10) -> list:
        with self.lock:
            return self.alerts[-n:]

    def get_stats(self) -> dict:
        with self.lock:
            if not self.alerts:
                return {'total': 0, 'alerts': 0, 'avg_score': 0.0}
            scores = [a['score'] for a in self.alerts]
            return {
                'total': len(self.alerts),
                'alerts': sum(1 for a in self.alerts if a['threshold_exceeded']),
                'avg_score': round(sum(scores) / len(scores), 4),
                'max_score': round(max(scores), 4),
                'min_score': round(min(scores), 4),
            }


# ──────────────────────────────────────────────
#  API Models
# ──────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Hijoluminic Firewall API",
        description="Zero-Day Network Intrusion Detection via Fractal Resonance Collapse",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class AnalyzeRequest(BaseModel):
        traffic: list  # list of lists: [[samples], ...]

    class ConfigUpdate(BaseModel):
        threshold: float | None = None
        n_sites: int | None = None
        n_features: int | None = None


# ──────────────────────────────────────────────
#  Global Model Instance
# ──────────────────────────────────────────────

config = TrafficConfig(n_features=8, n_sites=64)
firewall = HijoluminicFirewall(config=config, kappa_H=1.0)
alert_store = AlertStore()
start_time = time.time()


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    @app.get("/status")
    def get_status():
        """Health check and model status."""
        uptime = time.time() - start_time
        return {
            "status": "operational",
            "version": "1.0.0",
            "model": "HijoluminicFirewall",
            "architecture": "FFT → Frequency Bins → CollapseOfLight → Entropy → Score",
            "n_parameters": sum(p.numel() for p in firewall.parameters()),
            "n_sites": config.n_sites,
            "n_features": config.n_features,
            "n_frequency_bins": config.n_freqs,
            "uptime_seconds": round(uptime, 1),
            "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "alerts": alert_store.get_stats(),
        }

    @app.get("/alerts")
    def get_alerts(n: int = 10):
        """Get recent alerts."""
        return {"alerts": alert_store.get_recent(n)}

    @app.post("/analyze")
    def analyze_traffic_endpoint(request: AnalyzeRequest):
        """
        Analyze traffic data for anomalies.

        Request body: {"traffic": [[...], [...], ...]}
        Shape: (batch, n_sites, n_features) as nested lists.

        Returns per-sample anomaly scores.
        """
        try:
            data = torch.tensor(request.traffic, dtype=torch.float32)
            if data.dim() == 2:
                data = data.unsqueeze(0)  # Add batch dim
            if data.dim() != 3:
                raise ValueError(f"Expected 3D tensor (batch, sites, features), got shape {data.shape}")

            # Pad or truncate to expected dimensions
            B, S, F = data.shape
            if S != config.n_sites or F != config.n_features:
                # Resample to expected dimensions
                data = data[:, :config.n_sites, :config.n_features]
                if data.shape[1] < config.n_sites:
                    pad = config.n_sites - data.shape[1]
                    data = torch.cat([data, torch.zeros(B, pad, config.n_features)], dim=1)
                if data.shape[2] < config.n_features:
                    pad = config.n_features - data.shape[2]
                    data = torch.cat([data, torch.zeros(B, config.n_sites, pad)], dim=2)

            with torch.no_grad():
                out = firewall(data, return_all=True)
                scores = out['scores'].tolist()
                entropies = out['entropy'].tolist()

            # Store high-score alerts
            for i, (score, ent) in enumerate(zip(scores, entropies)):
                if score > 0.5:
                    alert_store.add(score, ent, f"synthetic_traffic_sample_{i}")

            return {
                "scores": scores,
                "entropies": entropies,
                "max_entropy": round(math.log(config.n_branches), 4),
                "anomaly_count": sum(1 for s in scores if s > 0.5),
                "total_samples": len(scores),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/retrain")
    def retrain_model(n_steps: int = 50):
        """Retrain on synthetic normal traffic to update baseline."""
        losses = train_unsupervised(firewall, n_steps=n_steps, verbose=False)
        return {
            "status": "retrained",
            "steps": n_steps,
            "final_loss": round(losses[-1], 4),
        }


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    if not FASTAPI_AVAILABLE:
        print("ERROR: fastapi/uvicorn not installed.")
        print("Install with: pip install fastapi uvicorn")
        print("\nFalling back to terminal demo...")
        from builds.build1_zeroday.detector import demo
        demo()
        return

    print("=" * 60)
    print("  Hijoluminic Firewall API Server")
    print("=" * 60)
    print(f"  Model: FFT → {config.n_freqs} freq bins → Collapse → Entropy → Score")
    print(f"  Parameters: {sum(p.numel() for p in firewall.parameters())}")
    print(f"  Listening on: http://0.0.0.0:8080")
    print(f"  Endpoints:")
    print(f"    GET  /status    — Health check")
    print(f"    GET  /alerts    — Recent alerts")
    print(f"    POST /analyze   — Analyze traffic")
    print(f"    POST /retrain   — Retrain baseline")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == '__main__':
    main()
