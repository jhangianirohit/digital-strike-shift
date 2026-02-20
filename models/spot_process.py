"""
Spot price simulation models.
Modular design: start with GBM, easy to add jumps later.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationResult:
    """Result of spot price simulation."""
    paths: np.ndarray      # Shape: (n_paths, n_steps + 1)
    times: np.ndarray      # Shape: (n_steps + 1,) in years

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def n_steps(self) -> int:
        return self.paths.shape[1] - 1

    @property
    def dt(self) -> float:
        """Time step in years."""
        return self.times[1] - self.times[0] if len(self.times) > 1 else 0


class SpotProcess(ABC):
    """Abstract base class for spot price processes."""

    @abstractmethod
    def simulate(
        self,
        spot: float,
        horizon: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Simulate spot price paths.

        Args:
            spot: Initial spot price
            horizon: Time horizon in years
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            seed: Random seed for reproducibility

        Returns:
            SimulationResult with paths and time grid
        """
        pass


class GBM(SpotProcess):
    """
    Geometric Brownian Motion.

    dS = (r - q) * S * dt + sigma * S * dW

    For FX: r = domestic rate, q = foreign rate
    For equities: r = risk-free rate, q = dividend yield
    """

    def __init__(
        self,
        sigma: float,
        r: float = 0.0,
        q: float = 0.0
    ):
        """
        Args:
            sigma: Annualized volatility
            r: Domestic interest rate (or risk-free rate)
            q: Foreign interest rate (or dividend yield)
        """
        self.sigma = sigma
        self.r = r
        self.q = q
        self.drift = r - q

    def simulate(
        self,
        spot: float,
        horizon: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """Simulate GBM paths using log-normal exact solution."""
        if seed is not None:
            np.random.seed(seed)

        dt = horizon / n_steps
        times = np.linspace(0, horizon, n_steps + 1)

        # Generate random increments
        # Z ~ N(0, 1), shape: (n_paths, n_steps)
        Z = np.random.standard_normal((n_paths, n_steps))

        # Log returns: ln(S_{t+dt}/S_t) = (drift - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
        log_returns = (self.drift - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z

        # Cumulative log returns
        cum_log_returns = np.cumsum(log_returns, axis=1)

        # Build paths: S_t = S_0 * exp(cumulative log returns)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot
        paths[:, 1:] = spot * np.exp(cum_log_returns)

        return SimulationResult(paths=paths, times=times)


# Placeholder for future extension
class JumpDiffusion(SpotProcess):
    """
    Jump-diffusion model (Merton).
    To be implemented when needed.

    dS = (r - q - lambda*k) * S * dt + sigma * S * dW + S * dJ
    where dJ is a compound Poisson process.
    """

    def __init__(
        self,
        sigma: float,
        jump_intensity: float,
        jump_mean: float,
        jump_std: float,
        r: float = 0.0,
        q: float = 0.0
    ):
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.r = r
        self.q = q
        raise NotImplementedError("Jump-diffusion not yet implemented")

    def simulate(
        self,
        spot: float,
        horizon: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None
    ) -> SimulationResult:
        raise NotImplementedError("Jump-diffusion not yet implemented")
