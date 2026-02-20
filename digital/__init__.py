"""Digital option pricing and strike shifting logic."""

from .digital_option import (
    DigitalOption,
    DigitalType,
    digital_price,
    digital_delta,
    digital_gamma,
    call_spread_approx_price,
    call_spread_approx_delta,
    call_spread_approx_gamma,
    conservative_spread_price,
    conservative_spread_delta,
    conservative_spread_gamma,
)
from .strike_shift import (
    StrikeShiftStrategy,
    shifted_digital_cost,
    strike_shift_cost,
    optimal_shift_width,
)

__all__ = [
    "DigitalOption",
    "DigitalType",
    "digital_price",
    "digital_delta",
    "digital_gamma",
    "call_spread_approx_price",
    "call_spread_approx_delta",
    "call_spread_approx_gamma",
    "conservative_spread_price",
    "conservative_spread_delta",
    "conservative_spread_gamma",
    "StrikeShiftStrategy",
    "shifted_digital_cost",
    "strike_shift_cost",
    "optimal_shift_width",
]
