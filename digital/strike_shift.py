"""
Strike shifting strategies for digital options.

The core problem: Digital options have infinite gamma at the strike as
expiry approaches. In practice, you either:
1. Accept the P&L variance from delta-hedging
2. "Shift the strike" - pay a fixed cost to move the effective strike

Strike shifting is essentially buying insurance: you pay a premium now
to avoid potential large hedging costs later.

The optimal stopping problem: When should you apply the shift?
- If spot moves far from strike, you don't need to shift (digital is clear ITM/OTM)
- If spot stays near strike close to expiry, shifting becomes expensive
- There's an optimal boundary in (spot, time) space
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

from .digital_option import (
    DigitalOption,
    DigitalType,
    digital_price,
    digital_delta,
    call_spread_approx_price,
    call_spread_approx_gamma,
    conservative_spread_price,
    conservative_spread_gamma,
)
from models.black_scholes import BSParams, gamma, expected_hedging_cost


class StrikeShiftStrategy(Enum):
    """Available strike shifting strategies."""
    FIXED_SPREAD = "fixed_spread"  # Shift to a call spread with fixed width
    PROPORTIONAL = "proportional"  # Shift proportional to remaining time
    VEGA_MATCHED = "vega_matched"  # Shift to match vega exposure


@dataclass
class ShiftCost:
    """Cost breakdown for a strike shift."""
    immediate_cost: float      # Cost to execute shift now
    spread_width: float        # Width of the resulting spread
    effective_delta: float     # Delta after shift
    effective_gamma: float     # Gamma after shift
    breakeven_move: float      # Spot move needed to make shift worthwhile


def shifted_digital_cost(
    option: DigitalOption,
    spread_width: float,
) -> ShiftCost:
    """
    Calculate cost of shifting a digital to a conservative call/put spread.

    Conservative shift direction:
    - LONG digital call: shift to call spread Long Call(K) - Short Call(K + width)
    - LONG digital put: shift to put spread Long Put(K) - Short Put(K - width)

    The shift is always in the conservative direction, meaning the spread
    always pays out LESS than the digital in the "danger zone" near the strike.
    Therefore, the shift cost is always positive (you're giving up value).

    Shift cost = Digital Price - Conservative Spread Price (always > 0)
    """
    digital_val = digital_price(option)
    spread_val = conservative_spread_price(option, spread_width)

    # Shift cost is always positive: digital pays more than conservative spread
    immediate_cost = digital_val - spread_val

    # Effective Greeks after shift (using conservative spread)
    effective_gamma = conservative_spread_gamma(option, spread_width)
    effective_delta = effective_gamma * option.spot  # Approximate

    # Breakeven: how far must spot move for the shift to be worthwhile?
    # Rough approximation based on gamma reduction
    old_gamma = abs(digital_delta(option) / option.spot)  # Approximate
    new_gamma = abs(effective_gamma)
    gamma_reduction = old_gamma - new_gamma if old_gamma > new_gamma else 0

    if gamma_reduction > 0:
        # Cost = 0.5 * gamma * spot^2 * move^2
        # Solving: immediate_cost = 0.5 * gamma_reduction * spot^2 * move^2
        breakeven_move = np.sqrt(2 * immediate_cost / (gamma_reduction * option.spot**2 + 1e-10))
    else:
        breakeven_move = np.inf

    return ShiftCost(
        immediate_cost=immediate_cost,
        spread_width=spread_width,
        effective_delta=effective_delta,
        effective_gamma=effective_gamma,
        breakeven_move=breakeven_move
    )


def strike_shift_cost(
    option: DigitalOption,
    shift_pct: float = 0.01,  # Shift as percentage of strike
) -> float:
    """
    Quick calculation of shift cost.

    shift_pct: the strike shift as a fraction of the strike price
    """
    spread_width = option.strike * shift_pct
    cost = shifted_digital_cost(option, spread_width)
    return cost.immediate_cost


def optimal_shift_width(
    option: DigitalOption,
    max_gamma_tolerance: float = 1.0,
) -> float:
    """
    Find minimum spread width to achieve target gamma.

    This is a simple heuristic - the optimal stopping solver does this
    more rigorously across the time dimension.
    """
    # Binary search for spread width
    low = option.strike * 0.001  # 0.1% of strike
    high = option.strike * 0.20  # 20% of strike

    for _ in range(50):
        mid = (low + high) / 2
        gamma_val = abs(call_spread_approx_gamma(option, mid))

        if gamma_val > max_gamma_tolerance:
            low = mid
        else:
            high = mid

    return high


def cumulative_hedging_cost(
    option: DigitalOption,
    spots: np.ndarray,
    times: np.ndarray,
    spread_width: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate cumulative hedging cost along a path.

    If spread_width is provided, use conservative spread gamma.
    Otherwise, use theoretical digital gamma.
    """
    n_steps = len(spots) - 1
    costs = np.zeros(n_steps + 1)

    for i in range(n_steps):
        dt = times[i + 1] - times[i] if i < len(times) - 1 else times[i] - times[i - 1]

        opt_i = option.with_spot(spots[i]).with_time(option.time_to_expiry - times[i])

        if spread_width is not None:
            gamma_val = conservative_spread_gamma(opt_i, spread_width)
        else:
            # Use vanilla gamma as proxy (digital gamma is unstable)
            params = opt_i.to_bs_params()
            gamma_val = gamma(params)

        # Expected hedging cost: 0.5 * gamma * S^2 * sigma^2 * dt
        costs[i + 1] = costs[i] + 0.5 * abs(gamma_val) * spots[i]**2 * option.vol**2 * dt

    return costs


def expected_future_hedging_cost(
    option: DigitalOption,
    spread_width: Optional[float] = None,
    n_simulations: int = 1000,
) -> float:
    """
    Monte Carlo estimate of expected future hedging cost.

    This is used in the optimal stopping calculation to compare:
    - Cost of shifting now
    - Expected cost of continuing to hedge
    """
    from simulation import GBMSimulator, GBMParams, SimulationConfig

    params = GBMParams(
        vol=option.vol,
        rate_dom=option.rate_dom,
        rate_for=option.rate_for
    )
    simulator = GBMSimulator(params)

    config = SimulationConfig(
        n_paths=n_simulations,
        n_steps=max(10, int(option.time_to_expiry * 252))  # Daily steps
    )

    result = simulator.simulate(option.spot, option.time_to_expiry, config)

    total_cost = 0.0
    for i in range(result.n_paths):
        path_cost = cumulative_hedging_cost(
            option,
            result.path(i),
            result.times,
            spread_width
        )
        total_cost += path_cost[-1]

    return total_cost / result.n_paths
