"""
build1_zeroday/pcap_capture.py — Real-world packet capture interface.

Captures live network traffic, extracts features, and feeds them
into the Hijoluminic Firewall for zero-day detection.

Requires: scapy (pip install scapy)
Optionally: npcap or libpcap for packet capture.

Usage:
    # Live capture on interface
    python -m builds.build1_zeroday.pcap_capture --interface Ethernet

    # Read from PCAP file
    python -m builds.build1_zeroday.pcap_capture --file traffic.pcap
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import math
import argparse
from collections import deque
from datetime import datetime

import torch
import numpy as np

from builds.build1_zeroday.detector import HijoluminicFirewall, TrafficConfig

try:
    from scapy.all import sniff, rdpcap, IP, TCP, UDP, Ether
    from scapy.packet import Packet
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


# ──────────────────────────────────────────────
#  Traffic Feature Extraction
# ──────────────────────────────────────────────

class PacketFeatureExtractor:
    """
    Extracts numerical features from raw packets for H-ANS analysis.

    For each packet, extracts:
      - Packet length (bytes)
      - Inter-arrival time (seconds since last packet)
      - TCP window size (if TCP)
      - Protocol type (TCP=1.0, UDP=2.0, ICMP=3.0, other=0.0)
      - Source port entropy (lower 8 bits of port / 255)
      - Destination port entropy (lower 8 bits of port / 255)
      - SYN flag (0 or 1)
      - ACK flag (0 or 1)
    """

    def __init__(self, n_features: int = 8):
        self.n_features = n_features
        self.last_time = None

    def extract(self, packet: Packet) -> np.ndarray | None:
        """Extract feature vector from a single packet."""
        if not packet.haslayer(IP):
            return None

        now = packet.time
        iat = (now - self.last_time) if self.last_time else 0.0
        self.last_time = now

        features = np.zeros(self.n_features, dtype=np.float32)

        # [0] Packet length (normalized)
        features[0] = min(len(packet) / 1500.0, 1.0)

        # [1] Inter-arrival time (clamped)
        features[1] = min(iat / 1.0, 1.0)

        # [2] Protocol type
        if packet.haslayer(TCP):
            features[2] = 1.0
            # [3] TCP window size
            features[3] = min(packet[TCP].window / 65535.0, 1.0)
            # [4] Source port feature
            features[4] = (packet[TCP].sport % 256) / 255.0
            # [5] Dest port feature
            features[5] = (packet[TCP].dport % 256) / 255.0
            # [6] SYN flag
            features[6] = 1.0 if packet[TCP].flags & 0x02 else 0.0
            # [7] ACK flag
            features[7] = 1.0 if packet[TCP].flags & 0x10 else 0.0
        elif packet.haslayer(UDP):
            features[2] = 2.0
            features[4] = (packet[UDP].sport % 256) / 255.0
            features[5] = (packet[UDP].dport % 256) / 255.0
        else:
            features[2] = 3.0  # other

        return features


class TrafficBuffer:
    """
    Stores a sliding window of packet features for H-ANS analysis.
    Maintains a fixed-size window of n_sites packets.
    """

    def __init__(self, n_sites: int = 64, n_features: int = 8):
        self.n_sites = n_sites
        self.n_features = n_features
        self.buffer = deque(maxlen=n_sites)

    def add(self, features: np.ndarray):
        self.buffer.append(features)

    def get_tensor(self) -> torch.Tensor:
        """
        Returns the current buffer as a (1, n_sites, n_features) tensor.
        Pads with zeros if buffer is not full.
        """
        arr = np.zeros((self.n_sites, self.n_features), dtype=np.float32)
        n = len(self.buffer)
        if n > 0:
            arr[-n:] = np.array(list(self.buffer))
        return torch.from_numpy(arr).unsqueeze(0)

    @property
    def is_full(self) -> bool:
        return len(self.buffer) == self.n_sites

    @property
    def fill_percent(self) -> float:
        return len(self.buffer) / self.n_sites * 100


# ──────────────────────────────────────────────
#  Live Capture Mode
# ──────────────────────────────────────────────

class LiveCapture:
    """Captures packets live and runs H-ANS detection."""

    def __init__(self,
                 interface: str = "Ethernet",
                 model: HijoluminicFirewall | None = None,
                 config: TrafficConfig | None = None):
        self.interface = interface
        self.config = config or TrafficConfig(n_features=8, n_sites=64)
        self.model = model or HijoluminicFirewall(config=self.config)
        self.extractor = PacketFeatureExtractor(n_features=self.config.n_features)
        self.buffer = TrafficBuffer(n_sites=self.config.n_sites,
                                    n_features=self.config.n_features)
        self.running = False
        self.total_packets = 0
        self.alert_count = 0

    def process_packet(self, packet: Packet):
        """Callback for each captured packet."""
        features = self.extractor.extract(packet)
        if features is None:
            return

        self.buffer.add(features)
        self.total_packets += 1

        if self.buffer.is_full:
            traffic_tensor = self.buffer.get_tensor()
            with torch.no_grad():
                out = self.model(traffic_tensor, return_all=True)
                score = out['scores'].item()
                entropy = out['entropy'].item()

            if score > 0.5:
                self.alert_count += 1
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"⚠ ANOMALY  score={score:.3f}  entropy={entropy:.3f}  "
                      f"packets={self.total_packets}")

    def start(self):
        """Start live packet capture."""
        print(f"\n  Starting live capture on '{self.interface}'...")
        print(f"  Buffer: {self.config.n_sites} packets, "
              f"{self.config.n_features} features")
        print(f"  Threshold: score > 0.5 = ANOMALY")
        print(f"  Press Ctrl+C to stop.\n")

        self.running = True
        try:
            sniff(iface=self.interface,
                  prn=self.process_packet,
                  store=False,
                  stopped_callback=lambda: not self.running)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"\n  Capture error: {e}")
            print("  Tip: Try running as administrator, or specify a different interface.")
            self.stop()

    def stop(self):
        """Stop capture and print summary."""
        self.running = False
        print(f"\n  Capture stopped.")
        print(f"  Total packets: {self.total_packets}")
        print(f"  Alerts raised: {self.alert_count}")


# ──────────────────────────────────────────────
#  PCAP File Analysis
# ──────────────────────────────────────────────

def analyze_pcap(filepath: str, model: HijoluminicFirewall):
    """Analyze a PCAP file for anomalies."""
    print(f"\n  Analyzing PCAP: {filepath}")

    packets = rdpcap(filepath)
    print(f"  Packets: {len(packets)}")

    extractor = PacketFeatureExtractor(n_features=model.config.n_features)
    buffer = TrafficBuffer(n_sites=model.config.n_sites)

    scores = []
    alert_count = 0

    for i, packet in enumerate(packets):
        features = extractor.extract(packet)
        if features is None:
            continue

        buffer.add(features)

        if buffer.is_full:
            traffic_tensor = buffer.get_tensor()
            with torch.no_grad():
                out = model(traffic_tensor, return_all=True)
                score = out['scores'].item()
                entropy = out['entropy'].item()

            scores.append(score)
            if score > 0.5:
                alert_count += 1

        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{len(packets)} packets...")

    if scores:
        scores_t = torch.tensor(scores)
        print(f"\n  Results:")
        print(f"    Windows analyzed: {len(scores)}")
        print(f"    Mean score: {scores_t.mean().item():.4f}")
        print(f"    Max score: {scores_t.max().item():.4f}")
        print(f"    Anomaly alerts: {alert_count} ({alert_count/len(scores)*100:.1f}%)")
        print(f"    Threshold: score > 0.5")
    else:
        print(f"  No complete windows analyzed (need buffer={model.config.n_sites} packets)")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="H-ANS Zero-Day Detector — PCAP Capture Interface"
    )
    parser.add_argument('--interface', '-i', type=str, default='Ethernet',
                        help='Network interface for live capture')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='PCAP file to analyze (instead of live capture)')
    parser.add_argument('--n-sites', type=int, default=64,
                        help='Number of packets per analysis window')
    parser.add_argument('--n-features', type=int, default=8,
                        help='Number of features per packet')
    args = parser.parse_args()

    if not SCAPY_AVAILABLE:
        print("ERROR: scapy not installed.")
        print("Install with: pip install scapy")
        print("Also install npcap from https://npcap.com for Windows live capture.")
        return

    config = TrafficConfig(n_features=args.n_features, n_sites=args.n_sites)
    model = HijoluminicFirewall(config=config, kappa_H=1.0)

    print("=" * 60)
    print("  H-ANS Zero-Day Detector — PCAP Interface")
    print("=" * 60)
    print(f"  Features per packet: {args.n_features}")
    print(f"  Window size: {args.n_sites} packets")
    print(f"  Frequency bins: {config.n_freqs}")
    print("=" * 60)

    if args.file:
        analyze_pcap(args.file, model)
    else:
        capture = LiveCapture(interface=args.interface, model=model, config=config)
        try:
            capture.start()
        except PermissionError:
            print("\n  Need administrator privileges for packet capture.")
            print("  Run as admin or use --file to analyze a PCAP file.")


if __name__ == '__main__':
    main()
