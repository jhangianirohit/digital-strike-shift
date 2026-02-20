"""
Jump-Diffusion simulation engine (Merton model).

Extends GBM with Poisson jumps:
dS/S = (r - lambda*k) dt + sigma dW + (J-1) dN

where:
- N is Poisson process with intensity lambda
- J is the jump multiplier (log-normal)
- k = E[J-1] is the expected jump size

PLACEHOLDER: This module provides the structure for future implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .base import SpotSimulator, SimulationConfig, SimulationResult
from .gbm import GBMParams, GBMSimulator


@dataclass
class JumpDiffusionParams:
    """Parameters for Merton jump-diffusion."""
    vol: float                  # Diffusion volatility
    rate_dom: float = 0.0       # Domestic rate
    rate_for: float = 0.0       # Foreign rate
    jump_intensity: float = 0.0  # Poisson intensity (jumps per year)
    jump_mean: float = 0.0      # Mean of log-jump size
    jump_vol: float = 0.0       # Volatility of log-jump size

    @property
    def expected_jump(self) -> float:
        """Expected value of J-1 where J = exp(jump_mean + 0.5*jump_vol^2)."""
        return np.exp(self.jump_mean + 0.5 * self.jump_vol**2) - 1

    @property
    def compensated_drift(self) -> float:
        """Risk-neutral drift including jump compensation."""
        return self.rate_dom - self.rate_for - self.jump_intensity * self.expected_jump


class JumpDiffusionSimulator(SpotSimulator):
    """
    Merton jump-diffusion simulator.

    The log-spot evolves as:
    d(ln S) = (r - 0.5*sigma^2 - lambda*k) dt + sigma dW + ln(J) dN
    """

    def __init__(self, params: JumpDiffusionParams):
        self.params = params

    def drift(self, spot: float, t: float) -> float:
        """Drift including jump compensation."""
        return self.params.compensated_drift * spot

    def diffusion(self, spot: float, t: float) -> float:
        """Diffusion coefficient (same as GBM)."""
        return self.params.vol * spot

    def simulate(
        self,
        spot: float,
        time_to_expiry: float,
        config: SimulationConfig
    ) -> SimulationResult:
        """
        Simulate jump-diffusion paths.

        For each time step:
        1. Draw number of jumps from Poisson(lambda * dt)
        2. Draw jump sizes from LogNormal
        3. Add diffusion component
        """
        if config.seed is not None:
            np.random.seed(config.seed)

        n_paths = config.n_paths
        n_steps = config.n_steps
        dt = time_to_expiry / n_steps

        times = np.linspace(0, time_to_expiry, n_steps + 1)

        # Drift components
        diffusion_drift = (
            self.params.compensated_drift - 0.5 * self.params.vol**2
        ) * dt
        vol_sqrt_dt = self.params.vol * np.sqrt(dt)

        # Initialize log-paths
        log_paths = np.zeros((n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(spot)

        # Poisson parameter for number of jumps per step
        jump_rate = self.params.jump_intensity * dt

        for step in range(n_steps):
            # Diffusion component
            dW = np.random.randn(n_paths)
            diffusion_increment = diffusion_drift + vol_sqrt_dt * dW

            # Jump component
            n_jumps = np.random.poisson(jump_rate, n_paths)
            jump_increment = np.zeros(n_paths)

            # For paths with jumps, accumulate log-jump sizes
            has_jump = n_jumps > 0
            if np.any(has_jump):
                for i in np.where(has_jump)[0]:
                    jump_sizes = np.random.normal(
                        self.params.jump_mean,
                        self.params.jump_vol,
                        n_jumps[i]
                    )
                    jump_increment[i] = np.sum(jump_sizes)

            log_paths[:, step + 1] = (
                log_paths[:, step] +
                diffusion_increment +
                jump_increment
            )

        paths = np.exp(log_paths)

        return SimulationResult(paths=paths, times=times, dt=dt)
