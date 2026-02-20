"""
Optimal stopping solver for FX digital option strike shifting.

The problem: Given a digital option, determine the optimal time/moneyness
boundary at which to apply a strike shift.

Approach: Backward induction using Longstaff-Schwartz style regression
to estimate continuation values.

At each point, compare:
1. Immediate exercise value: Cost of shifting now
2. Continuation value: Expected cost of waiting

Stop (shift) when immediate exercise is better than continuation.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional

from digital.digital_option import (
    DigitalOption,
    DigitalType,
    digital_price,
    call_spread_approx_gamma,
)
from digital.strike_shift import shifted_digital_cost
from simulation import GBMSimulator, GBMParams, SimulationConfig
from models.black_scholes import BSParams, gamma


class StoppingDecision(Enum):
    """Decision output from the solver."""
    APPLY_NOW = "apply_now"    # Apply strike shift immediately
    WAIT = "wait"              # Continue hedging, wait for better opportunity


@dataclass
class StoppingBoundary:
    """
    The optimal stopping boundary in (time, moneyness) space.

    For each time point, gives the moneyness thresholds:
    - If |moneyness| > boundary, WAIT (option is safely ITM or OTM)
    - If |moneyness| < boundary, APPLY_NOW (too risky, shift the strike)
    """
    times: np.ndarray           # Time to expiry values
    upper_boundary: np.ndarray  # Upper moneyness boundary (OTM side)
    lower_boundary: np.ndarray  # Lower moneyness boundary (ITM side)

    def should_stop(self, time_to_expiry: float, moneyness: float) -> bool:
        """Check if we should stop (apply shift) at given state."""
        if time_to_expiry <= 0:
            return False  # At expiry, no point shifting

        upper = np.interp(time_to_expiry, self.times[::-1], self.upper_boundary[::-1])
        lower = np.interp(time_to_expiry, self.times[::-1], self.lower_boundary[::-1])

        # Stop if moneyness is within the risky zone
        return lower < moneyness < upper


@dataclass
class DecisionResult:
    """Complete result from optimal stopping analysis."""
    decision: StoppingDecision
    expected_cost_if_shift_now: float
    expected_cost_optimal: float  # Optimal cost (may include shifting later)
    cost_savings: float  # Savings from following optimal strategy vs shift now
    boundary: Optional[StoppingBoundary]
    confidence: float  # 0-1, based on simulation variance
    details: dict


class OptimalStoppingSolver:
    """
    Solves the optimal stopping problem for digital strike shifting.

    Uses backward induction:
    1. Simulate many spot paths
    2. At each time step (backward), estimate continuation value
    3. Compare with immediate shift cost
    4. Build optimal stopping boundary
    """

    def __init__(
        self,
        option: DigitalOption,
        spread_width: float,
        n_paths: int = 10000,
        n_time_steps: int = 50,
        seed: Optional[int] = None,
        must_shift_days: int = 14,  # Must shift by X days before expiry
        daily_carry_cost: float = 0.0  # Opportunity cost per day of not shifting
    ):
        self.option = option
        self.spread_width = spread_width
        self.n_paths = n_paths
        self.n_time_steps = n_time_steps
        self.seed = seed
        self.must_shift_days = must_shift_days
        self.daily_carry_cost = daily_carry_cost

        # Decision horizon: from now until must-shift deadline
        # If less than must_shift_days to expiry, must shift immediately
        self.deadline = must_shift_days / 252  # Convert to years
        self.decision_horizon = max(0, option.time_to_expiry - self.deadline)

        # Setup simulator
        self.gbm_params = GBMParams(
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for
        )
        self.simulator = GBMSimulator(self.gbm_params)

    def _compute_gamma_grid_with_tte(self, paths: np.ndarray, option_tte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-compute gamma values for all paths and times (vectorized).

        Args:
            paths: Simulated spot paths (n_paths x n_steps+1)
            option_tte: Option time-to-expiry at each step (n_steps+1,)

        Returns:
            - unshifted_gamma: gamma for digital (approximated)
            - shifted_gamma: gamma after applying call spread
        """
        from scipy.stats import norm

        n_paths, n_steps_plus_1 = paths.shape
        K = self.option.strike
        vol = self.option.vol
        r_d = self.option.rate_dom
        r_f = self.option.rate_for
        notional = self.option.notional
        half_w = self.spread_width / 2
        T_total = self.option.time_to_expiry

        unshifted_gamma = np.zeros((n_paths, n_steps_plus_1))
        shifted_gamma = np.zeros((n_paths, n_steps_plus_1))

        for j in range(n_steps_plus_1):
            tau = option_tte[j]  # Time to expiry at this step
            if tau <= 1e-10:
                continue

            S = paths[:, j]
            sqrt_tau = np.sqrt(tau)
            vol_sqrt_tau = vol * sqrt_tau

            # Vanilla gamma
            d1 = (np.log(S / K) + (r_d - r_f + 0.5 * vol**2) * tau) / vol_sqrt_tau
            discount = np.exp(-r_f * tau)
            vanilla_gamma = discount * norm.pdf(d1) / (S * vol_sqrt_tau)

            # Unshifted gamma - scales with time remaining until deadline
            time_until_deadline = tau - self.deadline
            if time_until_deadline > 0:
                # More time = more flexibility = effectively wider spread
                dynamic_spread = self.spread_width * (1 + np.sqrt(time_until_deadline / T_total))
            else:
                dynamic_spread = self.spread_width
            unshifted_gamma[:, j] = np.abs(vanilla_gamma / dynamic_spread * notional)

            # Shifted gamma (call spread)
            d1_lower = (np.log(S / (K - half_w)) + (r_d - r_f + 0.5 * vol**2) * tau) / vol_sqrt_tau
            d1_upper = (np.log(S / (K + half_w)) + (r_d - r_f + 0.5 * vol**2) * tau) / vol_sqrt_tau
            gamma_lower = discount * norm.pdf(d1_lower) / (S * vol_sqrt_tau)
            gamma_upper = discount * norm.pdf(d1_upper) / (S * vol_sqrt_tau)
            shifted_gamma[:, j] = np.abs((gamma_lower - gamma_upper) / self.spread_width * notional)

        return unshifted_gamma, shifted_gamma

    def _compute_shift_costs_with_tte(self, paths: np.ndarray, option_tte: np.ndarray) -> np.ndarray:
        """Pre-compute immediate shift costs for all states (vectorized).

        Uses CONSERVATIVE spread direction:
        - For LONG call: spread is Long Call(K) - Short Call(K + width)
        - For LONG put: spread is Long Put(K) - Short Put(K - width)

        Shift cost = Digital Price - Conservative Spread Price (always positive)

        Args:
            paths: Simulated spot paths (n_paths x n_steps+1)
            option_tte: Option time-to-expiry at each step (n_steps+1,)
        """
        from scipy.stats import norm

        n_paths, n_steps_plus_1 = paths.shape
        K = self.option.strike
        vol = self.option.vol
        r_d = self.option.rate_dom
        r_f = self.option.rate_for
        notional = self.option.notional
        w = self.spread_width
        is_call = self.option.option_type == DigitalType.CALL

        shift_costs = np.zeros((n_paths, n_steps_plus_1))

        for j in range(n_steps_plus_1):
            tau = option_tte[j]  # Time to expiry at this step
            if tau <= 1e-10:
                continue

            S = paths[:, j]
            sqrt_tau = np.sqrt(tau)
            vol_sqrt_tau = vol * sqrt_tau
            discount = np.exp(-r_d * tau)
            discount_for = np.exp(-r_f * tau)

            # Digital price
            d2 = (np.log(S / K) + (r_d - r_f - 0.5 * vol**2) * tau) / vol_sqrt_tau
            if is_call:
                digital_price = notional * discount * norm.cdf(d2)
            else:
                digital_price = notional * discount * norm.cdf(-d2)

            # Conservative spread price
            if is_call:
                # Long Call(K) - Short Call(K + width)
                K1, K2 = K, K + w
                d2_1 = (np.log(S / K1) + (r_d - r_f - 0.5 * vol**2) * tau) / vol_sqrt_tau
                d2_2 = (np.log(S / K2) + (r_d - r_f - 0.5 * vol**2) * tau) / vol_sqrt_tau
                d1_1 = d2_1 + vol_sqrt_tau
                d1_2 = d2_2 + vol_sqrt_tau
                call_1 = S * discount_for * norm.cdf(d1_1) - K1 * discount * norm.cdf(d2_1)
                call_2 = S * discount_for * norm.cdf(d1_2) - K2 * discount * norm.cdf(d2_2)
                spread_price = notional * (call_1 - call_2) / w
            else:
                # Long Put(K) - Short Put(K - width)
                K1, K2 = K, K - w
                d2_1 = (np.log(S / K1) + (r_d - r_f - 0.5 * vol**2) * tau) / vol_sqrt_tau
                d2_2 = (np.log(S / K2) + (r_d - r_f - 0.5 * vol**2) * tau) / vol_sqrt_tau
                d1_1 = d2_1 + vol_sqrt_tau
                d1_2 = d2_2 + vol_sqrt_tau
                put_1 = K1 * discount * norm.cdf(-d2_1) - S * discount_for * norm.cdf(-d1_1)
                put_2 = K2 * discount * norm.cdf(-d2_2) - S * discount_for * norm.cdf(-d1_2)
                spread_price = notional * (put_1 - put_2) / w

            # Shift cost is always positive: digital pays more than conservative spread
            shift_costs[:, j] = digital_price - spread_price

        return shift_costs

    def solve(self) -> Tuple[StoppingBoundary, np.ndarray, float, float]:
        """
        Solve for optimal stopping boundary using backward induction.

        The decision horizon runs from now until the must-shift deadline.
        At the deadline, you MUST shift (no choice), so terminal cost = shift cost at that point.

        Returns:
            - StoppingBoundary object
            - Optimal value function at t=0 for each path
            - Cost of shifting immediately at t=0 (averaged across paths)
            - Fraction of paths where shifting at t=0 is optimal
        """
        # If already past deadline, must shift immediately
        if self.decision_horizon <= 0:
            # No optionality - must shift now
            opt = DigitalOption(
                spot=self.option.spot,
                strike=self.option.strike,
                vol=self.option.vol,
                rate_dom=self.option.rate_dom,
                rate_for=self.option.rate_for,
                time_to_expiry=self.option.time_to_expiry,
                notional=self.option.notional,
                option_type=self.option.option_type
            )
            shift_cost = shifted_digital_cost(opt, self.spread_width).immediate_cost
            empty_boundary = StoppingBoundary(
                times=np.array([self.option.time_to_expiry]),
                upper_boundary=np.array([np.inf]),
                lower_boundary=np.array([-np.inf])
            )
            return empty_boundary, np.array([shift_cost]), shift_cost, 1.0

        config = SimulationConfig(
            n_paths=self.n_paths,
            n_steps=self.n_time_steps,
            seed=self.seed
        )

        # Simulate paths over the DECISION HORIZON (not full time to expiry)
        result = self.simulator.simulate(
            self.option.spot,
            self.decision_horizon,  # Only simulate until must-shift deadline
            config
        )

        n_steps = result.n_steps
        paths = result.paths
        times = result.times

        # Option time-to-expiry at each simulation step
        option_tte = self.option.time_to_expiry - times

        # Compute shift costs at each point
        shift_costs = self._compute_shift_costs_with_tte(paths, option_tte)

        # Value function: V[i, j] = optimal expected shift cost from state (i, j)
        V = np.zeros((self.n_paths, n_steps + 1))

        # Terminal condition: at deadline, MUST shift - cost is shift cost at that spot
        V[:, -1] = shift_costs[:, -1]

        # Track stopping decisions
        stopped = np.zeros((self.n_paths, n_steps + 1), dtype=bool)
        upper_boundary = np.zeros(n_steps + 1)
        lower_boundary = np.zeros(n_steps + 1)

        # Carry cost per time step (opportunity cost of waiting)
        dt_years = self.decision_horizon / n_steps if n_steps > 0 else 0
        dt_days = dt_years * 252
        carry_cost_per_step = self.daily_carry_cost * dt_days

        # Backward induction
        # At each step: shift now (pay shift_cost) vs wait (pay carry cost + expected future cost)
        # Key: continuation value should be EXPECTED value at next step for paths at similar spots
        for j in range(n_steps - 1, -1, -1):
            cost_if_shift = shift_costs[:, j]

            if j == 0:
                # At t=0, all paths start at same spot
                # Continuation value is the EXPECTED future optimal cost + carry cost
                continuation_value = np.mean(V[:, j + 1])
                cost_if_wait = np.full(self.n_paths, continuation_value + carry_cost_per_step)
            else:
                # For intermediate steps, use path-specific continuation + carry cost
                # (This is simplified; proper LSM would use regression)
                cost_if_wait = V[:, j + 1] + carry_cost_per_step

            # Optimal decision
            shift_better = cost_if_shift < cost_if_wait
            stopped[:, j] = shift_better

            V[:, j] = np.where(shift_better, cost_if_shift, cost_if_wait)

            # Estimate boundary
            moneyness_vals = np.log(paths[:, j] / self.option.strike)
            stopped_moneyness = moneyness_vals[shift_better]

            if len(stopped_moneyness) > 0:
                upper_boundary[j] = np.percentile(stopped_moneyness, 95)
                lower_boundary[j] = np.percentile(stopped_moneyness, 5)
            else:
                if j < n_steps - 1:
                    upper_boundary[j] = upper_boundary[j + 1]
                    lower_boundary[j] = lower_boundary[j + 1]

        boundary = StoppingBoundary(
            times=self.option.time_to_expiry - times,
            upper_boundary=upper_boundary,
            lower_boundary=lower_boundary
        )

        cost_shift_now_t0 = np.mean(shift_costs[:, 0])
        # Compare shift now vs wait (which includes carry cost to get to next step)
        expected_continuation_with_carry = (np.mean(V[:, 1]) + carry_cost_per_step) if n_steps > 0 else cost_shift_now_t0
        pct_shift_now = 1.0 if cost_shift_now_t0 <= expected_continuation_with_carry else 0.0

        return boundary, V[:, 0], cost_shift_now_t0, pct_shift_now

    def decide(self) -> DecisionResult:
        """
        Make a decision for the current option state.

        Returns recommendation: APPLY_NOW or WAIT, with expected costs.

        The decision is based on backward induction: if the majority of simulated
        paths find it optimal to shift at t=0, recommend APPLY_NOW.
        """
        # Solve the full problem
        boundary, optimal_values, total_cost_shift_now, pct_shift_now = self.solve()

        # Current state
        current_moneyness = np.log(self.option.spot / self.option.strike)

        # Optimal expected cost (from simulation)
        expected_cost_optimal = np.mean(optimal_values)
        std_cost = np.std(optimal_values)

        # Decision based on what fraction of paths should shift at t=0
        # If >50% of paths find shifting optimal at t=0, recommend shift
        should_stop = pct_shift_now > 0.5

        # Cost savings (positive = waiting/optimal strategy saves money)
        cost_savings = total_cost_shift_now - expected_cost_optimal

        decision = StoppingDecision.APPLY_NOW if should_stop else StoppingDecision.WAIT

        # Confidence based on how decisive the recommendation is
        # pct_shift_now near 0 or 1 = high confidence, near 0.5 = low confidence
        confidence = abs(pct_shift_now - 0.5) * 2  # 0 to 1 scale

        # Get immediate shift cost for reporting
        opt = DigitalOption(
            spot=self.option.spot,
            strike=self.option.strike,
            vol=self.option.vol,
            rate_dom=self.option.rate_dom,
            rate_for=self.option.rate_for,
            time_to_expiry=self.option.time_to_expiry,
            notional=self.option.notional,
            option_type=self.option.option_type
        )
        immediate_shift_cost = shifted_digital_cost(opt, self.spread_width).immediate_cost

        return DecisionResult(
            decision=decision,
            expected_cost_if_shift_now=total_cost_shift_now,
            expected_cost_optimal=expected_cost_optimal,
            cost_savings=cost_savings,
            boundary=boundary,
            confidence=confidence,
            details={
                "current_moneyness": current_moneyness,
                "moneyness_pct": current_moneyness * 100,
                "immediate_shift_cost": immediate_shift_cost,
                "future_hedge_cost": total_cost_shift_now - immediate_shift_cost,
                "pct_paths_shift_now": pct_shift_now,
                "std_cost": std_cost,
                "n_paths": self.n_paths,
            }
        )


def quick_decision(
    spot: float,
    strike: float,
    vol: float,
    time_to_expiry: float,
    spread_width_pct: float = 0.01,
    rate_dom: float = 0.0,
    rate_for: float = 0.0,
    option_type: str = "call",
    n_paths: int = 5000,
    daily_carry_cost: float = 0.0,
) -> DecisionResult:
    """
    Quick wrapper for making a strike shift decision.

    Args:
        spot: Current spot price
        strike: Digital strike
        vol: Annualized volatility
        time_to_expiry: Time to expiry in years
        spread_width_pct: Spread width as % of strike
        rate_dom: Domestic rate
        rate_for: Foreign rate
        option_type: "call" or "put"
        n_paths: Number of Monte Carlo paths
        daily_carry_cost: Opportunity cost per day of not shifting

    Returns:
        DecisionResult with recommendation
    """
    opt_type = DigitalType.CALL if option_type == "call" else DigitalType.PUT

    option = DigitalOption(
        spot=spot,
        strike=strike,
        vol=vol,
        rate_dom=rate_dom,
        rate_for=rate_for,
        time_to_expiry=time_to_expiry,
        option_type=opt_type
    )

    spread_width = strike * spread_width_pct

    solver = OptimalStoppingSolver(
        option=option,
        spread_width=spread_width,
        n_paths=n_paths,
        n_time_steps=max(10, int(time_to_expiry * 252)),
        daily_carry_cost=daily_carry_cost,
    )

    return solver.decide()
