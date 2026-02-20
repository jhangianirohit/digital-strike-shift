"""
Abstract base class for spot price simulation engines.
Designed to be extended for different dynamics (GBM, jump-diffusion, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for path simulation."""
    n_paths: int = 10000        # Number of Monte Carlo paths
    n_steps: int = 100          # Number of time steps
    seed: Optional[int] = None  # Random seed for reproducibility


@dataclass
class SimulationResult:
    """Result of a path simulation."""
    paths: np.ndarray           # Shape: (n_paths, n_steps + 1)
    times: np.ndarray           # Shape: (n_steps + 1,)
    dt: float                   # Time step size

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def n_steps(self) -> int:
        return self.paths.shape[1] - 1

    @property
    def terminal_values(self) -> np.ndarray:
        """Final spot values across all paths."""
        return self.paths[:, -1]

    def spot_at_step(self, step: int) -> np.ndarray:
        """Spot values at a given time step across all paths."""
        return self.paths[:, step]

    def path(self, idx: int) -> np.ndarray:
        """Get a single path by index."""
        return self.paths[idx, :]


class SpotSimulator(ABC):
    """
    Abstract base class for spot price simulators.

    Subclasses implement specific dynamics:
    - GBM (Geometric Brownian Motion)
    - Merton Jump-Diffusion
    - Heston Stochastic Volatility
    - etc.
    """

    @abstractmethod
    def simulate(
        self,
        spot: float,
        time_to_expiry: float,
        config: SimulationConfig
    ) -> SimulationResult:
        """
        Simulate spot price paths.

        Args:
            spot: Initial spot price
            time_to_expiry: Total simulation time in years
            config: Simulation configuration

        Returns:
            SimulationResult containing paths and time grid
        """
        pass

    @abstractmethod
    def drift(self, spot: float, t: float) -> float:
        """
        Drift coefficient at given spot and time.
        For GBM: (r_dom - r_for) * S
        """
        pass

    @abstractmethod
    def diffusion(self, spot: float, t: float) -> float:
        """
        Diffusion coefficient at given spot and time.
        For GBM: sigma * S
        """
        pass
