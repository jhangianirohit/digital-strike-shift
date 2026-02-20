"""
Geometric Brownian Motion (GBM) simulation engine.

Implements the SDE: dS/S = (r_d - r_f) dt + sigma dW

Structured to be extended for jump-diffusion by adding jump component.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .base import SpotSimulator, SimulationConfig, SimulationResult


@dataclass
class GBMParams:
    """Parameters for GBM dynamics."""
    vol: float              # Annualized volatility
    rate_dom: float = 0.0   # Domestic risk-free rate
    rate_for: float = 0.0   # Foreign risk-free rate

    @property
    def drift_rate(self) -> float:
        """Net drift under risk-neutral measure."""
        return self.rate_dom - self.rate_for


class GBMSimulator(SpotSimulator):
    """
    Geometric Brownian Motion simulator for FX spot prices.

    Uses exact simulation (not Euler discretization) for efficiency and accuracy.
    The log-spot follows: d(ln S) = (r - 0.5*sigma^2) dt + sigma dW
    """

    def __init__(self, params: GBMParams):
        self.params = params

    def drift(self, spot: float, t: float) -> float:
        """Drift coefficient: (r_d - r_f) * S."""
        return self.params.drift_rate * spot

    def diffusion(self, spot: float, t: float) -> float:
        """Diffusion coefficient: sigma * S."""
        return self.params.vol * spot

    def simulate(
        self,
        spot: float,
        time_to_expiry: float,
        config: SimulationConfig
    ) -> SimulationResult:
        """
        Simulate GBM paths using exact log-normal simulation.

        ln(S_t) = ln(S_0) + (r - 0.5*sigma^2)*t + sigma*sqrt(t)*Z
        where Z ~ N(0,1)
        """
        if config.seed is not None:
            np.random.seed(config.seed)

        n_paths = config.n_paths
        n_steps = config.n_steps
        dt = time_to_expiry / n_steps

        # Time grid
        times = np.linspace(0, time_to_expiry, n_steps + 1)

        # Pre-compute constants
        drift_per_step = (self.params.drift_rate - 0.5 * self.params.vol**2) * dt
        vol_sqrt_dt = self.params.vol * np.sqrt(dt)

        # Initialize paths with starting spot
        log_paths = np.zeros((n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(spot)

        # Generate all random increments at once
        dW = np.random.randn(n_paths, n_steps)

        # Simulate log-spot evolution
        for step in range(n_steps):
            log_paths[:, step + 1] = (
                log_paths[:, step] +
                drift_per_step +
                vol_sqrt_dt * dW[:, step]
            )

        # Convert to spot prices
        paths = np.exp(log_paths)

        return SimulationResult(paths=paths, times=times, dt=dt)

    def simulate_with_antithetic(
        self,
        spot: float,
        time_to_expiry: float,
        config: SimulationConfig
    ) -> SimulationResult:
        """
        Simulate with antithetic variates for variance reduction.
        For each random path, also generate the path with negated shocks.
        """
        if config.seed is not None:
            np.random.seed(config.seed)

        n_paths = config.n_paths // 2  # Half paths, doubled by antithetic
        n_steps = config.n_steps
        dt = time_to_expiry / n_steps

        times = np.linspace(0, time_to_expiry, n_steps + 1)

        drift_per_step = (self.params.drift_rate - 0.5 * self.params.vol**2) * dt
        vol_sqrt_dt = self.params.vol * np.sqrt(dt)

        # Initialize paths
        log_paths = np.zeros((n_paths * 2, n_steps + 1))
        log_paths[:, 0] = np.log(spot)

        # Generate random increments
        dW = np.random.randn(n_paths, n_steps)

        # Original paths
        for step in range(n_steps):
            log_paths[:n_paths, step + 1] = (
                log_paths[:n_paths, step] +
                drift_per_step +
                vol_sqrt_dt * dW[:, step]
            )
            # Antithetic paths (negated shocks)
            log_paths[n_paths:, step + 1] = (
                log_paths[n_paths:, step] +
                drift_per_step -
                vol_sqrt_dt * dW[:, step]
            )

        paths = np.exp(log_paths)

        return SimulationResult(paths=paths, times=times, dt=dt)


class GBMSimulatorBridge(GBMSimulator):
    """
    GBM simulator with Brownian bridge for conditional simulation.
    Useful for barrier options and path-dependent structures.
    """

    def simulate_bridge(
        self,
        spot: float,
        terminal_spot: float,
        time_to_expiry: float,
        config: SimulationConfig
    ) -> SimulationResult:
        """
        Simulate paths conditioned on a terminal value.
        Uses Brownian bridge construction.
        """
        if config.seed is not None:
            np.random.seed(config.seed)

        n_paths = config.n_paths
        n_steps = config.n_steps
        dt = time_to_expiry / n_steps

        times = np.linspace(0, time_to_expiry, n_steps + 1)

        log_start = np.log(spot)
        log_end = np.log(terminal_spot)

        # Initialize paths
        log_paths = np.zeros((n_paths, n_steps + 1))
        log_paths[:, 0] = log_start
        log_paths[:, -1] = log_end

        # Build bridge: for each intermediate point
        for step in range(1, n_steps):
            t_curr = times[step]
            t_prev = times[step - 1]
            t_end = time_to_expiry

            # Bridge mean and variance
            ratio = (t_end - t_curr) / (t_end - t_prev)
            mean = log_paths[:, step - 1] * ratio + log_end * (1 - ratio)
            var = (t_curr - t_prev) * (t_end - t_curr) / (t_end - t_prev)

            log_paths[:, step] = mean + np.sqrt(var) * self.params.vol * np.random.randn(n_paths)

        paths = np.exp(log_paths)

        return SimulationResult(paths=paths, times=times, dt=dt)
