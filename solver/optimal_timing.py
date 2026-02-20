"""
Optimal timing solver for digital option strike shifts.

Uses certainty equivalent (CE) framework to make risk-adjusted decisions.

Two modes:
1. Symmetric: CE = E[Cost] + (γ/2) * Var[Cost] / E[Cost]
2. Asymmetric (default): Only penalize scenarios where cost increases
   CE = E[Cost] + γ * E[(Cost - Cost_now)²⁺] / E[Cost]

Decision rule: Shift now if Cost_now < CE_wait
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from models import BSParams, gamma, GBM, SimulationResult


class Decision(Enum):
    """Shift timing recommendation."""
    SHIFT_NOW = "SHIFT_NOW"
    WAIT = "WAIT"


@dataclass
class SolverResult:
    """Results from the optimal timing solver."""
    decision: Decision
    cost_now: float                     # Cost if we shift immediately
    expected_cost_wait: float           # E[Cost] if we wait
    variance_cost_wait: float           # Var[Cost] if we wait (full variance)
    upside_variance: float              # Variance of costs > cost_now only
    ce_wait: float                      # Certainty equivalent of waiting
    gamma_now: float                    # Current gamma at strike
    days_remaining: int                 # Days until deadline
    risk_aversion: float                # Risk aversion parameter used
    pct_paths_higher: float             # % of paths where cost > cost_now

    # For visualization
    gamma_paths: Optional[np.ndarray] = None  # Shape: (n_paths, n_steps)
    cost_paths: Optional[np.ndarray] = None   # Shape: (n_paths, n_steps)
    time_grid: Optional[np.ndarray] = None    # Shape: (n_steps,) in days


class ShiftTimingSolver:
    """
    Solver for optimal strike shift timing.

    The cost of applying a strike shift is:
        Cost = Gamma(S, K, tau) * shift_size_bps

    where Gamma is the vanilla option gamma at the strike.

    We simulate spot paths and compute the distribution of costs
    at the deadline, then use certainty equivalent to decide.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        vol: float,
        expiry_days: float,
        shift_bps: float,
        risk_aversion: float = 0.1,
        deadline_days: int = 14,
        rate_dom: float = 0.0,
        rate_for: float = 0.0,
        n_paths: int = 10000,
        seed: Optional[int] = None,
        asymmetric: bool = True,
    ):
        """
        Args:
            spot: Current spot price
            strike: Digital option strike
            vol: Annualized volatility (e.g., 0.10 for 10%)
            expiry_days: Days to option expiry
            shift_bps: Strike shift size in basis points
            risk_aversion: Risk aversion parameter gamma (default 0.5)
            deadline_days: Must shift by this many days before expiry (default 14)
            rate_dom: Domestic interest rate
            rate_for: Foreign interest rate
            n_paths: Number of Monte Carlo paths
            seed: Random seed for reproducibility
            asymmetric: If True (default), only penalize upside cost variance
        """
        self.spot = spot
        self.strike = strike
        self.vol = vol
        self.expiry_days = expiry_days
        self.shift_bps = shift_bps
        self.risk_aversion = risk_aversion
        self.deadline_days = deadline_days
        self.rate_dom = rate_dom
        self.rate_for = rate_for
        self.n_paths = n_paths
        self.seed = seed
        self.asymmetric = asymmetric

        # Derived values
        self.days_remaining = max(0, int(expiry_days - deadline_days))
        self.horizon_years = self.days_remaining / 365.0

    def _compute_gamma(self, spot: float, tau_years: float) -> float:
        """Compute vanilla gamma at the strike."""
        if tau_years <= 0:
            return 0.0

        params = BSParams(
            spot=spot,
            strike=self.strike,
            vol=self.vol,
            rate_dom=self.rate_dom,
            rate_for=self.rate_for,
            time_to_expiry=tau_years
        )
        return gamma(params)

    def _compute_cost(self, spot: float, tau_years: float) -> float:
        """
        Compute cost of shift at given spot and time.

        Cost = Gamma * shift_bps
        """
        g = self._compute_gamma(spot, tau_years)
        return g * self.shift_bps

    def solve(self) -> SolverResult:
        """
        Run the optimal timing analysis.

        Returns SolverResult with recommendation and supporting data.
        """
        # Time to deadline (when we're forced to shift)
        tau_at_deadline = self.deadline_days / 365.0

        # Current cost if we shift now
        tau_now = self.expiry_days / 365.0
        gamma_now = self._compute_gamma(self.spot, tau_now)
        cost_now = gamma_now * self.shift_bps

        # If no time left to wait, must shift now
        if self.days_remaining <= 0:
            return SolverResult(
                decision=Decision.SHIFT_NOW,
                cost_now=cost_now,
                expected_cost_wait=cost_now,
                variance_cost_wait=0.0,
                upside_variance=0.0,
                ce_wait=cost_now,
                gamma_now=gamma_now,
                days_remaining=0,
                risk_aversion=self.risk_aversion,
                pct_paths_higher=0.0,
            )

        # Simulate spot paths forward to the deadline
        process = GBM(sigma=self.vol, r=self.rate_dom, q=self.rate_for)
        n_steps = self.days_remaining  # Daily steps

        sim = process.simulate(
            spot=self.spot,
            horizon=self.horizon_years,
            n_steps=n_steps,
            n_paths=self.n_paths,
            seed=self.seed
        )

        # Compute gamma and cost at each point along each path
        gamma_paths = np.zeros((self.n_paths, n_steps + 1))
        cost_paths = np.zeros((self.n_paths, n_steps + 1))

        for t_idx in range(n_steps + 1):
            # Time remaining to expiry at this simulation step
            days_elapsed = t_idx * (self.days_remaining / n_steps)
            tau_years = (self.expiry_days - days_elapsed) / 365.0

            for p_idx in range(self.n_paths):
                spot_t = sim.paths[p_idx, t_idx]
                gamma_paths[p_idx, t_idx] = self._compute_gamma(spot_t, tau_years)
                cost_paths[p_idx, t_idx] = gamma_paths[p_idx, t_idx] * self.shift_bps

        # Cost at deadline (forced shift) - last column
        costs_at_deadline = cost_paths[:, -1]

        # Statistics
        expected_cost_wait = np.mean(costs_at_deadline)
        variance_cost_wait = np.var(costs_at_deadline)

        # Upside variance: only penalize scenarios where cost > cost_now
        # E[(max(0, Cost - Cost_now))^2]
        upside_deviations = np.maximum(0, costs_at_deadline - cost_now)
        upside_variance = np.mean(upside_deviations ** 2)
        pct_paths_higher = np.mean(costs_at_deadline > cost_now)

        # Certainty equivalent calculation
        if self.asymmetric:
            # Asymmetric: only penalize upside (cost increases)
            # CE = E[Cost] + γ * E[(Cost - Cost_now)²⁺] / E[Cost]
            if expected_cost_wait > 1e-10:
                ce_wait = expected_cost_wait + self.risk_aversion * upside_variance / expected_cost_wait
            else:
                ce_wait = expected_cost_wait
        else:
            # Symmetric: penalize all variance
            # CE = E[Cost] + (γ/2) * Var[Cost] / E[Cost]
            if expected_cost_wait > 1e-10:
                ce_wait = expected_cost_wait + (self.risk_aversion / 2) * variance_cost_wait / expected_cost_wait
            else:
                ce_wait = expected_cost_wait

        # Decision: shift now if cost_now < CE_wait
        decision = Decision.SHIFT_NOW if cost_now < ce_wait else Decision.WAIT

        # Time grid in days for visualization
        time_grid = np.linspace(0, self.days_remaining, n_steps + 1)

        return SolverResult(
            decision=decision,
            cost_now=cost_now,
            expected_cost_wait=expected_cost_wait,
            variance_cost_wait=variance_cost_wait,
            upside_variance=upside_variance,
            ce_wait=ce_wait,
            gamma_now=gamma_now,
            days_remaining=self.days_remaining,
            risk_aversion=self.risk_aversion,
            pct_paths_higher=pct_paths_higher,
            gamma_paths=gamma_paths,
            cost_paths=cost_paths,
            time_grid=time_grid,
        )


def quick_decision(
    spot: float,
    strike: float,
    vol: float,
    expiry_days: float,
    shift_bps: float,
    risk_aversion: float = 0.1,
    deadline_days: int = 14,
    asymmetric: bool = True,
) -> Decision:
    """
    Quick helper to get just the decision.

    Returns Decision.SHIFT_NOW or Decision.WAIT
    """
    solver = ShiftTimingSolver(
        spot=spot,
        strike=strike,
        vol=vol,
        expiry_days=expiry_days,
        shift_bps=shift_bps,
        risk_aversion=risk_aversion,
        deadline_days=deadline_days,
        n_paths=5000,  # Faster for quick checks
        asymmetric=asymmetric,
    )
    return solver.solve().decision
