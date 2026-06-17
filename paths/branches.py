"""
paths/branches.py — Trajectory synthesis, branch generation, and path-integral layer.

This module implements the path-integral branch superposition mechanism:

Formulations:
    • Trajectory synthesis: S_c(t) = Σᵢ αᵢ [P_d + Vᵢ ; Θ_d + Θᵢ]
    • Branch generation: Spawns B complex-amplitude branches from divergence events
    • Path-integral layer: G = Σᵢ αᵢ exp(i Sᵢ / ℏ)

Each branch carries a complex amplitude and a geometric phase arising from
its action Sᵢ. The coherent superposition of branches implements the
"second quantization" stream of the H-ANS architecture.​​​​​​​​​‌​​​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌‌‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​​‌‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​​​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​​‌‌‍​​​​​​​​​‌‌​​​‌‌‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​‌‌​​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌‌​​‌​​‍​​​​​​​​​‌‌​​‌‌​‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​​‌‌‌​​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​​‌‌​​‌​‍​​​​​​​​​‌‌​​​‌​‍​​​​​​​​​​‌‌​​​‌‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​‌​‌​​‍​​​​​​​​​‌​​‌‌‌‌‍​​​​​​​​​‌​​‌​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​​‌‌‌​‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​‌‌‌​‍​​​​​​​​​‌​‌​​‌‌‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​‌​​‌​​​‍​​​​​​​​​‌​​‌​​‌‍​​​​​​​​​‌​​‌​‌​‍​​​​​​​​​​‌​‌‌​‌‍​​​​​​​​​​‌‌​‌​‌‍​​​​​​​​​​‌‌​‌​​‍​​​​​​​​​‌​​​​‌‌‍​​​​​​​​​‌​​​‌​‌‍​​​​​​​​​​‌‌​‌‌​‍​​​​​​​​​‌​​​​​‌‍​​​​​​​​​‌​​​‌​​‍​​​​​​​​​​‌‌​‌​​‍
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
#  Branch Generator
# ──────────────────────────────────────────────

class BranchGenerator(nn.Module):
    """
    Spawns B complex-amplitude branches from a divergence event.

    At each "divergence event" (decision point, fork, or measurement),
    this module generates B candidate continuations, each with:
      - A complex amplitude α_i (learnable via the amplitude network)
      - An initial phase θ_i (learnable offset)
    
    The branches represent distinct possible futures in superposition.
    """

    def __init__(self,
                 in_features: int,
                 n_branches: int = 4,
                 hidden_dim: int = 64,
                 learnable_phases: bool = True):
        """
        Args:
            in_features: Dimensionality of input features at divergence point.
            n_branches: Number of branches B to generate.
            hidden_dim: Hidden dimension of the amplitude network.
            learnable_phases: Whether branch phases are learned or random.
        """
        super().__init__()
        self.n_branches = n_branches
        self.in_features = in_features

        # Amplitude network: maps divergence context to branch amplitudes
        self.amplitude_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_branches),
        )

        # Log-softplus pre-activation ensures amplitudes > 0
        self.log_amplitude_scale = nn.Parameter(torch.zeros(1))

        # Branch phases: either learned or fixed random
        if learnable_phases:
            self.phases = nn.Parameter(torch.randn(n_branches) * math.pi)
        else:
            self.register_buffer('phases', torch.randn(n_branches) * math.pi)

        # Learnable branch bias vectors (offsets for each branch in feature space)
        self.branch_biases = nn.Parameter(
            torch.randn(n_branches, in_features) * 0.1
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generate branches from a divergence event.

        Args:
            x: Context tensor of shape (batch, in_features) — the divergence
               point representation.

        Returns:
            Tuple (amplitudes, phases, branch_offsets):
                amplitudes: Tensor of shape (batch, n_branches) — |α_i| > 0
                phases: Tensor of shape (n_branches,) or (batch, n_branches) — θ_i
                branch_offsets: Tensor of shape (batch, n_branches, in_features) —
                                offsets to add to the divergence point for each branch
        """
        # Compute raw amplitudes
        raw_amps = self.amplitude_net(x)  # (batch, n_branches)

        # Ensure non-negative via softplus, then scale
        amplitudes = F.softplus(raw_amps) * torch.exp(self.log_amplitude_scale) + 1e-8

        # Phase: broadcast over batch
        phases = self.phases  # (n_branches,)
        if x.dim() > 1:
            phases = phases.unsqueeze(0).expand(x.shape[0], -1)  # (batch, n_branches)

        # Branch offsets
        branch_offsets = self.branch_biases.unsqueeze(0)  # (1, B, in_features)

        return amplitudes, phases, branch_offsets


# ──────────────────────────────────────────────
#  Action Functional
# ──────────────────────────────────────────────

class ActionFunctional(nn.Module):
    r"""
    Computes the action S along a trajectory.

    The trajectory synthesis formula:
        S_c(t) = Σ_i α_i [P_d + V_i ; Θ_d + Θ_i]

    where:
        P_d: Base position at divergence point
        V_i: Branch velocity (learned per branch)
        Θ_d: Base direction at divergence point
        Θ_i: Branch deflection (learned per branch)

    In the path-integral context, the action is the time integral of the
    Lagrangian along a path γ(t):
        S[γ] = ∫_{t_A}^{t_B} L(γ(t), γ̇(t), t) dt

    We discretize this as a sum over waypoints, computing the Lagrangian
    from learned potentials and kinetic terms.
    """

    def __init__(self,
                 state_dim: int,
                 n_branches: int = 4,
                 n_control_knots: int = 8):
        """
        Args:
            state_dim: Dimensionality of the state space.
            n_branches: Number of branches (for per-branch velocities).
            n_control_knots: Number of discretization points for the action integral.
        """
        super().__init__()
        self.state_dim = state_dim
        self.n_branches = n_branches
        self.n_knots = n_control_knots

        # Per-branch learned velocity vectors V_i
        self.branch_velocities = nn.Parameter(
            torch.randn(n_branches, state_dim) * 0.1
        )

        # Per-branch learned deflection angles Θ_i (in state space)
        self.branch_deflections = nn.Parameter(
            torch.randn(n_branches, state_dim) * 0.1
        )

        # Learned Lagrangian parameters
        # L = T - V where T = ½ m v² and V = potential
        self.log_mass = nn.Parameter(torch.zeros(1))
        self.potential_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # Metric g(x) — learned position-dependent metric for proper time
        self.metric_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, state_dim * state_dim),
        )

    @property
    def mass(self) -> Tensor:
        """Kinetic mass parameter."""
        return torch.exp(self.log_mass)

    def compute_trajectory(self,
                           divergence_point: Tensor,
                           amplitudes: Tensor,
                           branch_offsets: Tensor,
                           n_steps: int | None = None) -> Tensor:
        """
        Compute the full trajectory for each branch.

        Args:
            divergence_point: (batch, state_dim) — starting state P_d, Θ_d.
            amplitudes: (batch, n_branches) — branch amplitudes (unused here).
            branch_offsets: (batch, n_branches, state_dim) — branch-specific offsets.
            n_steps: Number of integration steps (defaults to self.n_knots).

        Returns:
            Trajectory tensor of shape (batch, n_branches, n_steps, state_dim).
        """
        steps = n_steps or self.n_knots
        batch = divergence_point.shape[0]
        device = divergence_point.device

        # Base state: P_d + V_i (position) and Θ_d + Θ_i (direction)
        # We combine them as the initial state per branch
        base = divergence_point.unsqueeze(1) + branch_offsets  # (batch, B, D)

        # Generate trajectory via linear interpolation with learned velocities
        # Trajectory γ_i(t) = base_i + V_i * t + noise
        t = torch.linspace(0, 1, steps, device=device)  # (steps,)
        t = t.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, steps, 1)
        velocity = self.branch_velocities.unsqueeze(0).unsqueeze(2)  # (1, B, 1, D)

        trajectory = base.unsqueeze(2) + velocity * t  # (batch, B, steps, D)
        return trajectory

    def compute_lagrangian(self,
                           state: Tensor,
                           velocity: Tensor) -> Tensor:
        """
        Compute the Lagrangian L = T - V at a state-velocity pair.

        Args:
            state: (..., state_dim) — position in state space.
            velocity: (..., state_dim) — velocity in state space.

        Returns:
            Lagrangian scalar per input point: (...,).
        """
        # Kinetic term: T = ½ m v²
        T = 0.5 * self.mass * (velocity ** 2).sum(dim=-1)

        # Potential term: V = V_net(state)
        V = self.potential_net(state).squeeze(-1)

        return T - V

    def compute_metric(self, x: Tensor) -> Tensor:
        """
        Compute the position-dependent metric g(x) for proper time.

        Args:
            x: (..., state_dim) — position.

        Returns:
            Metric matrix of shape (..., state_dim, state_dim).
        """
        batch_shape = x.shape[:-1]
        flat = x.reshape(-1, self.state_dim)
        g_flat = self.metric_net(flat)  # (-1, D²)
        g_flat = g_flat.view(-1, self.state_dim, self.state_dim)
        # Symmetrize to ensure g is a valid metric
        g_flat = (g_flat + g_flat.transpose(-2, -1)) / 2.0
        # Ensure positive definiteness: g = I + g' @ g'^T
        identity = torch.eye(self.state_dim, device=x.device).unsqueeze(0)
        g_flat = identity + g_flat @ g_flat.transpose(-2, -1)
        g = g_flat.reshape(*batch_shape, self.state_dim, self.state_dim)
        return g

    def compute_action(self,
                       trajectory: Tensor,
                       return_detailed: bool = False) -> Tensor | tuple[Tensor, ...]:
        """
        Compute the action S_i for each branch trajectory.

        S = ∫ L(γ(t), γ̇(t), t) dt  (discretized as Riemann sum)

        Also computes proper time τ_i = ∫ sqrt(g(γ) · γ̇, γ̇) dt

        Args:
            trajectory: (batch, n_branches, n_steps, state_dim).
            return_detailed: If True, also return Lagrangian and proper time.

        Returns:
            If return_detailed:
                Tuple (action, lagrangian, proper_time)
            Else:
                action tensor of shape (batch, n_branches).
        """
        # Positions and velocities (central differences)
        states = trajectory  # (B, B, S, D)
        vel = torch.zeros_like(states)
        vel[..., 1:-1, :] = (states[..., 2:, :] - states[..., :-2, :]) / 2.0
        vel[..., 0, :] = states[..., 1, :] - states[..., 0, :]
        vel[..., -1, :] = states[..., -1, :] - states[..., -2, :]

        # Lagrangian at each step
        L = self.compute_lagrangian(states, vel)  # (B, B, S)

        # Action as Riemann sum: S = Σ L_i * Δt
        dt = 1.0 / (states.shape[-2] - 1)
        action = torch.sum(L * dt, dim=-1)  # (B, B)

        # Proper time τ = ∫ sqrt(g(v, v)) dt
        # g(γ) at each point: (B, B, S, D, D)
        g_mat = self.compute_metric(states)
        # g(v, v) for each step: v^T g v
        v_expanded = vel.unsqueeze(-2)  # (B, B, S, 1, D)
        gv = (v_expanded @ g_mat).squeeze(-2)  # (B, B, S, D)
        g_vv = (gv * vel).sum(dim=-1)  # (B, B, S)
        proper_time = torch.sum(torch.sqrt(g_vv.abs() + 1e-8) * dt, dim=-1)

        if return_detailed:
            return action, L, proper_time

        return action


# ──────────────────────────────────────────────
#  Path Integral Layer
# ──────────────────────────────────────────────

class PathIntegralLayer(nn.Module):
    """
    Path-integral superposition layer: G = Σ_i α_i exp(i S_i / ℏ)

    This is the "second quantization" stream of H-ANS. It receives
    branch amplitudes and actions from the BranchGenerator and
    ActionFunctional, and computes the coherent superposition of paths.

    The propagator G_AB = Σ_i α_i exp(i S_i / ℏ) represents the
    amplitude for a particle at A to propagate to B, summed over all
    possible paths.

    This module also implements the "collapse" preparation by computing
    branch probabilities via the Born rule applied to the path amplitudes.
    """

    def __init__(self,
                 n_branches: int = 4,
                 hbar: float = 1.0,
                 use_learned_hbar: bool = False):
        """
        Args:
            n_branches: Number of branches B in the superposition.
            hbar: Reduced Planck constant (default 1.0 for natural units).
            use_learned_hbar: If True, ℏ is a learnable parameter.
        """
        super().__init__()
        self.n_branches = n_branches

        if use_learned_hbar:
            self.log_hbar = nn.Parameter(torch.tensor(math.log(hbar)))
        else:
            self.register_buffer('_hbar_val', torch.tensor(hbar, dtype=torch.float32))

    @property
    def hbar(self) -> Tensor:
        if hasattr(self, 'log_hbar'):
            return torch.exp(self.log_hbar)
        if hasattr(self, '_hbar_val'):
            return self._hbar_val
        return torch.tensor(1.0, dtype=torch.float32)

    def forward(self,
                amplitudes: Tensor,
                phases: Tensor,
                actions: Tensor) -> dict[str, Tensor]:
        """
        Compute the path-integral superposition.

        Args:
            amplitudes: Tensor of shape (batch, n_branches) — |α_i|.
            phases: Tensor of shape (batch, n_branches) — θ_i (initial phases).
            actions: Tensor of shape (batch, n_branches) — S_i (action per branch).

        Returns:
            Dict with keys:
                'G': Complex propagator sum Σ_i α_i exp(i S_i/ℏ) [shape (batch,)].
                'branch_amplitudes': Complex per-branch amplitudes α_i exp(i S_i/ℏ + iθ_i)
                                     [shape (batch, n_branches)].
                'magnitudes': Squared magnitudes |G|² [shape (batch,)].
                'interference_terms': Cross terms for interpretability [shape (batch, n_branches, n_branches)].
        """
        batch = amplitudes.shape[0]

        # Complex branch amplitudes: α_i exp(i (θ_i + S_i / ℏ))
        total_phase = phases + actions / self.hbar
        branch_amps = amplitudes * torch.exp(1j * total_phase)

        # Coherent sum: G = Σ_i α_i exp(i(θ_i + S_i/ℏ))
        G = torch.sum(branch_amps, dim=-1)  # (batch,)

        # Magnitudes
        mag = (G * G.conj()).real  # |G|² — probability

        # Interference matrix: (α_i α_j exp(i(θ_i-θ_j + (S_i-S_j)/ℏ)))
        # For diagnosing constructive/destructive interference
        i_minus_j = branch_amps.unsqueeze(-1) * branch_amps.conj().unsqueeze(-2)
        interference = i_minus_j  # (batch, B, B)

        return {
            'G': G,
            'branch_amplitudes': branch_amps,
            'magnitudes': mag,
            'interference_terms': interference,
        }

    def path_probabilities(self,
                           amplitudes: Tensor,
                           phases: Tensor,
                           actions: Tensor) -> Tensor:
        """
        Compute the Born-rule probability of each branch individually.

        P_i = |α_i|² — the probability associated with path i
        before interference with other paths.

        Note that Σ_i P_i may not equal 1; the total probability is
        |Σ_i α_i exp(iS_i/ℏ)|² which includes interference.

        Args:
            amplitudes: (batch, n_branches).
            phases: (batch, n_branches).
            actions: (batch, n_branches).

        Returns:
            Probabilities P_i = |α_i|², shape (batch, n_branches).
        """
        return amplitudes ** 2


# ──────────────────────────────────────────────
#  Divergence Detector
# ──────────────────────────────────────────────

class DivergenceDetector(nn.Module):
    """
    Detects when to trigger a branch divergence.

    A divergence event occurs when the system reaches a decision point
    where multiple futures are possible. This module learns a "branch
    probability" p_div(x) that goes to 1 near decision points.

    In practice, this can be:
      - An entropy-based trigger: high entropy → divergence
      - A learned threshold: p_div > 0.5 → generate branches
      - A fixed schedule: every K steps

    For maximum flexibility, this module outputs a continuous gate
    that controls how much to mix the Schrödinger and path-integral streams.
    """

    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: Tensor) -> Tensor:
        """
        Compute divergence probability p_div ∈ [0, 1].

        Args:
            state: (batch, state_dim).

        Returns:
            Divergence probability (batch, 1).
        """
        return self.net(state)
