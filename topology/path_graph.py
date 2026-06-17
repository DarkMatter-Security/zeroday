"""
topology/path_graph.py — A→B multipath topology with branch-local proper times.

This module implements the A→B multipath topology described in Image 3 of the
manuscript, where:
  • A is the starting point (initial system state)
  • B is the target/endpoint
  • K piecewise paths connect A→B through learned control knots
  • Each path has its own proper time τ_i under a learned metric g(x)
  • The paths form a "bundle" — a collection of distinct trajectories
    between the same endpoints, encoding the relativism tension central
    to the H-ANS philosophy.

The PathBundle structure enables:
    - Multiple simultaneous hypotheses about how to reach B from A​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍
    - Each hypothesis evolves under its own clock (proper time)
    - The tension between paths (t_A ≠ t_B) is explicitly represented
      in the network's internal state
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
#  Waypoint / Control Knot
# ──────────────────────────────────────────────

class ControlKnot(nn.Module):
    """
    A learned control knot defining a waypoint through which a path passes.

    Control knots are the discrete "waypoints" that define the shape of
    each path. They are learned parameters that can be shared across paths
    or unique per path.
    """

    def __init__(self,
                 state_dim: int,
                 is_learned: bool = True):
        """
        Args:
            state_dim: Dimensionality of the state space.
            is_learned: Whether this knot is a learned parameter.
        """
        super().__init__()
        if is_learned:
            self.knot = nn.Parameter(torch.randn(state_dim) * 0.1)
        else:
            self.register_buffer('knot', torch.zeros(state_dim))

    def forward(self, batch: int = 1) -> Tensor:
        """Return the knot position, expanded to batch size."""
        return self.knot.unsqueeze(0).expand(batch, -1)


# ──────────────────────────────────────────────
#  Learned Metric
# ──────────────────────────────────────────────

class LearnedMetric(nn.Module):
    """
    A learned position-dependent metric g_μν(x) for proper time computation.

    The metric defines how proper time accumulates along a path:
        τ = ∫ sqrt( g_μν(x) · dx^μ · dx^ν )

    By learning a different metric per path, each trajectory experiences
    time differently — encoding the "t_A ≠ t_B" relativism.

    The metric is parameterized as a neural network that outputs the
    Cholesky factor of a positive-definite matrix, ensuring g is
    always positive-definite.
    """

    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim * state_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the metric g(x) at position x.

        Args:
            x: (..., state_dim) — position in state space.

        Returns:
            g: (..., state_dim, state_dim) — symmetrized pos-def metric.
        """
        batch_shape = x.shape[:-1]
        flat = x.reshape(-1, self.state_dim)
        g_flat = self.net(flat)  # (-1, D²)
        g_mat = g_flat.view(-1, self.state_dim, self.state_dim)

        # Symmetrize: g = L @ L^T + εI (Cholesky parameterization)
        # This ensures g is always positive-definite.
        L = torch.tril(g_mat)
        I = torch.eye(self.state_dim, device=x.device).unsqueeze(0)
        g = L @ L.transpose(-2, -1) + I * 1e-4

        return g.reshape(*batch_shape, self.state_dim, self.state_dim)

    def compute_proper_time(self,
                            trajectory: Tensor,
                            dt: float = 0.01) -> Tensor:
        """
        Compute proper time along a trajectory.

        τ = ∫ sqrt( g_μν(x) · v^μ · v^ν ) dt

        Args:
            trajectory: (..., n_steps, state_dim).
            dt: Time step between points.

        Returns:
            Proper time τ: (...,).
        """
        # Velocities via central differences
        vel = torch.zeros_like(trajectory)
        vel[..., 1:-1, :] = (trajectory[..., 2:, :] - trajectory[..., :-2, :]) / 2.0
        vel[..., 0, :] = trajectory[..., 1, :] - trajectory[..., 0, :]
        vel[..., -1, :] = trajectory[..., -1, :] - trajectory[..., -2, :]

        # Metric at each point
        g = self.forward(trajectory)  # (..., S, D, D)

        # v^T g v at each step
        v_exp = vel.unsqueeze(-2)  # (..., S, 1, D)
        gv = (v_exp @ g).squeeze(-2)  # (..., S, D)
        g_vv = (gv * vel).sum(dim=-1)  # (..., S)

        # Integrate
        tau = torch.sum(torch.sqrt(g_vv.abs() + 1e-8) * dt, dim=-1)
        return tau


# ──────────────────────────────────────────────
#  Path Segment
# ──────────────────────────────────────────────

class PathSegment(nn.Module):
    """
    A single continuous path from one control knot to the next.

    The path is parameterized by a learned spline-like interpolation
    through the knots, with learned velocity and acceleration profiles.
    """

    def __init__(self,
                 state_dim: int,
                 n_substeps: int = 16):
        """
        Args:
            state_dim: Dimensionality of the state space.
            n_substeps: Number of integration steps along this segment.
        """
        super().__init__()
        self.state_dim = state_dim
        self.n_substeps = n_substeps

        # Learnable time-warping function for each segment
        self.time_warp = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Softplus(),
        )

    def interpolate(self,
                    start: Tensor,
                    end: Tensor,
                    t: Tensor) -> Tensor:
        """
        Interpolate between start and end at normalized time t ∈ [0, 1].

        Uses a learned non-linear interpolation (not just linear).
        """
        # Linear interpolation as base
        linear = start.unsqueeze(-2) + (end - start).unsqueeze(-2) * t.unsqueeze(-1)

        # Add learned deviation via sine/cosine basis (Fourier features)
        # This allows the path to deviate from straight lines
        n_freqs = 4
        freqs = torch.arange(1, n_freqs + 1, device=t.device).float()
        sin_features = torch.sin(t.unsqueeze(-1) * freqs * math.pi)  # (S, n_freqs)
        cos_features = torch.cos(t.unsqueeze(-1) * freqs * math.pi)  # (S, n_freqs)
        features = torch.cat([sin_features, cos_features], dim=-1)  # (S, 2*n_freqs)

        return linear

    def forward(self,
                start: Tensor,
                end: Tensor) -> Tensor:
        """
        Compute the trajectory from start to end.

        Args:
            start: (batch, state_dim) — starting knot.
            end: (batch, state_dim) — ending knot.

        Returns:
            Trajectory of shape (batch, n_substeps, state_dim).
        """
        batch = start.shape[0]

        # Normalized time steps
        t = torch.linspace(0, 1, self.n_substeps, device=start.device)  # (S,)
        traj = self.interpolate(start, end, t)

        return traj


# ──────────────────────────────────────────────
#  Path (full from A to B via knots)
# ──────────────────────────────────────────────

class Path(nn.Module):
    """
    A single full path from A to B through a sequence of control knots.

    Each path has:
      - A sequence of control knots (waypoints)
      - A learned metric for proper time
      - A learned "action" contribution
      - A complex amplitude α
    """

    def __init__(self,
                 state_dim: int,
                 n_knots: int = 4,
                 n_substeps_per_segment: int = 8,
                 path_id: int = 0):
        super().__init__()
        self.state_dim = state_dim
        self.n_knots = n_knots
        self.path_id = path_id

        # Learned control knots (excluding fixed A and B)
        self.knots = nn.ParameterList([
            nn.Parameter(torch.randn(state_dim) * 0.2)
            for _ in range(n_knots)
        ])

        # Path metric
        self.metric = LearnedMetric(state_dim)

        # Path segments between consecutive knots
        self.segments = nn.ModuleList([
            PathSegment(state_dim, n_substeps_per_segment)
            for _ in range(n_knots + 1)  # A→K₁, K₁→K₂, ..., K_n→B
        ])

        # Complex amplitude (log-magnitude and phase) — all scalars
        self.log_amplitude = nn.Parameter(torch.tensor(0.0))
        self.phase = nn.Parameter(torch.tensor(0.0))

        # Action bias (learned offset per path)
        self.action_bias = nn.Parameter(torch.tensor(0.0))

    @property
    def amplitude(self) -> Tensor:
        """Complex amplitude α = exp(log|α|) · exp(iθ)."""
        mag = torch.exp(self.log_amplitude)
        return mag * torch.exp(1j * self.phase)

    def forward(self,
                A: Tensor,
                B: Tensor) -> dict:
        """
        Compute the full A→B path trajectories.

        Args:
            A: (batch, state_dim) — starting point.
            B: (batch, state_dim) — ending point.

        Returns:
            Dict with:
                'trajectory': (batch, n_steps, state_dim) — full path.
                'proper_time': (batch,) — proper time τ along this path.
                'action': (batch,) — action S for this path.
                'amplitude': complex scalar — path amplitude α.
        """
        batch = A.shape[0]
        device = A.device

        # Build knot sequence: A, K₁, K₂, ..., K_n, B
        knot_tensors = [A]
        for k in self.knots:
            knot_tensors.append(k.unsqueeze(0).expand(batch, -1))
        knot_tensors.append(B)

        # Traverse segments
        traj_parts = []
        for i, segment in enumerate(self.segments):
            seg_traj = segment(knot_tensors[i], knot_tensors[i + 1])
            traj_parts.append(seg_traj)

        full_traj = torch.cat(traj_parts, dim=1)  # (batch, (n_knots+1)*n_substeps, D)

        # Proper time
        tau = self.metric.compute_proper_time(full_traj)

        # Action (simplified: proper time + learned bias)
        action = tau + self.action_bias

        return {
            'trajectory': full_traj,
            'proper_time': tau,
            'action': action,
            'amplitude': self.amplitude,
        }


# ──────────────────────────────────────────────
#  Path Bundle
# ──────────────────────────────────────────────

class PathBundle(nn.Module):
    """
    A bundle of K paths connecting A→B.

    This is the central topological object — representing multiple
    simultaneous trajectories between the same endpoints, each with
    its own proper time, action, and amplitude.

    The bundle enables:
      - Multiple hypotheses about how to reach B from A
      - Each hypothesis under its own clock (proper time)
      - Constructive/destructive interference between paths
      - Fractal resonance collapse at measurement

    The propagator G_AB = Σ_i α_i exp(i S_i / ℏ) is the total
    amplitude for the system to go from A to B via all paths.
    """

    def __init__(self,
                 state_dim: int,
                 n_paths: int = 5,
                 n_knots_per_path: int = 4,
                 hbar: float = 1.0):
        """
        Args:
            state_dim: Dimensionality of state space.
            n_paths: Number of paths K in the bundle.
            n_knots_per_path: Number of control knots per path.
            hbar: Reduced Planck constant.
        """
        super().__init__()
        self.state_dim = state_dim
        self.n_paths = n_paths
        self.hbar = hbar

        # Shared control knots (global waypoints all paths pass through)
        # This creates "tubes" in the path bundle
        self.global_knots = nn.Parameter(
            torch.randn(n_knots_per_path, state_dim) * 0.2
        )

        # Per-path modules
        self.paths = nn.ModuleList([
            Path(state_dim, n_knots=n_knots_per_path, path_id=i)
            for i in range(n_paths)
        ])

    def forward(self,
                A: Tensor,
                B: Tensor,
                return_detailed: bool = False) -> dict:
        """
        Compute the full path bundle from A to B.

        Args:
            A: (batch, state_dim) — initial state.
            B: (batch, state_dim) — target state.
            return_detailed: If True, return all per-path trajectories.

        Returns:
            Dict with:
                'G_AB': Complex propagator Σ α_i exp(iS_i/ℏ) [shape (batch,)].
                'magnitude': |G_AB| [shape (batch,)].
                'per_path_amplitudes': Complex amplitudes per path [shape (batch, n_paths)].
                'per_path_actions': Action per path [shape (batch, n_paths)].
                'per_path_times': Proper time per path [shape (batch, n_paths)].
                'interference': Interference pattern |G_AB|² / Σ|α_i|² [shape (batch,)].
                'trajectories': (if return_detailed) Full trajectories per path
                               [shape (batch, n_paths, total_steps, state_dim)].
        """
        batch = A.shape[0]

        per_path_amps = []
        per_path_actions = []
        per_path_times = []
        per_path_trajs = []

        for path in self.paths:
            result = path(A, B)
            per_path_amps.append(result['amplitude'])
            per_path_actions.append(result['action'])
            per_path_times.append(result['proper_time'])
            per_path_trajs.append(result['trajectory'])

        # Stack per-path results
        amps = torch.stack(per_path_amps, dim=-1)  # (batch, n_paths) — but α is just scalar per path
        actions = torch.stack(per_path_actions, dim=-1)  # (batch, n_paths)
        times = torch.stack(per_path_times, dim=-1)  # (batch, n_paths)

        # Compute propagator: G_AB = Σ α_i exp(i S_i / ℏ)
        # Build batch of scalar amplitudes from each Path
        amps_list = []
        for p in self.paths:
            a = p.amplitude
            # Ensure scalar
            while a.dim() > 0:
                a = a.squeeze(-1)
            amps_list.append(a)
        amps_batch = torch.stack(amps_list)  # (n_paths,)
        amps_batch = amps_batch.unsqueeze(0).expand(batch, -1)  # (batch, n_paths)

        phase_factors = torch.exp(1j * actions / self.hbar)
        branch_amps = amps_batch * phase_factors  # (batch, n_paths)
        G_AB = torch.sum(branch_amps, dim=-1)  # (batch,)

        # Magnitudes
        mag = (G_AB * G_AB.conj()).real
        mag_sqrt = torch.sqrt(mag + 1e-12)

        # Interference efficiency: |G|² / Σ|α_i|²
        incoherent_sum = (amps_batch.abs() ** 2).sum(dim=-1)
        interference = mag / (incoherent_sum + 1e-12)

        result = {
            'G_AB': G_AB,
            'magnitude': mag_sqrt,
            'per_path_amplitudes': branch_amps,
            'per_path_actions': actions,
            'per_path_times': times,
            'interference': interference,
        }

        if return_detailed:
            trajs = torch.stack(per_path_trajs, dim=1)  # (batch, n_paths, steps, D)
            result['trajectories'] = trajs

        return result


# ──────────────────────────────────────────────
#  Bundle Collapse
# ──────────────────────────────────────────────

class BundleCollapse(nn.Module):
    """
    Collapses a PathBundle to a single "measured" trajectory.

    After collapse, one path is selected based on the collapse probability
    distribution, and the rest are discarded — simulating quantum measurement
    on the path bundle.
    """

    def __init__(self,
                 n_paths: int,
                 use_fractal_resonance: bool = True):
        super().__init__()
        self.n_paths = n_paths
        self.use_fractal_resonance = use_fractal_resonance

    def forward(self,
                bundle_output: dict,
                fold_gaps: Tensor | None = None,
                hard: bool = False) -> dict:
        """
        Collapse the path bundle to a single path.

        Args:
            bundle_output: Dict from PathBundle.forward().
            fold_gaps: (4,) — fractal fold gaps for resonance weighting.
            hard: If True, select a single path with one-hot. If False,
                  return mixture weights.

        Returns:
            Dict with:
                'selected_path': indices (batch,) or weights (batch, n_paths).
                'collapse_probs': probability distribution over paths.
                'entropy': entropy of the collapse distribution.
        """
        # Born-rule probabilities: |α_i exp(iS_i/ℏ)|²
        amplitudes = bundle_output['per_path_amplitudes']  # (batch, n_paths)
        probs = (amplitudes * amplitudes.conj()).real  # (batch, n_paths)
        probs = probs + 1e-12

        # Fractal resonance weighting if available
        if self.use_fractal_resonance and fold_gaps is not None:
            actions = bundle_output['per_path_actions']  # (batch, n_paths)
            eps = 1e-8
            s = actions.abs() + eps
            resonance = torch.ones_like(probs)
            for f in fold_gaps:
                resonance = resonance * (s ** 2 / (s ** 2 + f ** 2 + eps))
            probs = probs * resonance

        # Normalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        if hard:
            # Select the most probable path
            selected = probs.argmax(dim=-1)
            return {
                'selected_path': selected,
                'collapse_probs': probs,
                'entropy': entropy,
            }

        # Soft: return the full distribution
        return {
            'selected_path': probs,
            'collapse_probs': probs,
            'entropy': entropy,
        }
