"""
Black-Scholes model for vanilla option pricing and Greeks.
Focused on gamma calculation for hedging cost estimation.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


@dataclass
class BSParams:
    """Black-Scholes model parameters."""
    spot: float          # Current spot price
    strike: float        # Option strike
    vol: float           # Annualized volatility
    rate_dom: float      # Domestic risk-free rate
    rate_for: float      # Foreign risk-free rate (for FX options)
    time_to_expiry: float  # Time to expiry in years

    @property
    def forward(self) -> float:
        """Forward price."""
        return self.spot * np.exp((self.rate_dom - self.rate_for) * self.time_to_expiry)

    @property
    def moneyness(self) -> float:
        """Log-moneyness: ln(S/K)."""
        return np.log(self.spot / self.strike)

    @property
    def forward_moneyness(self) -> float:
        """Forward log-moneyness: ln(F/K)."""
        return np.log(self.forward / self.strike)


def d1(params: BSParams) -> float:
    """Calculate d1 in Black-Scholes formula."""
    if params.time_to_expiry <= 0:
        return np.inf if params.spot > params.strike else -np.inf

    numerator = (
        np.log(params.spot / params.strike) +
        (params.rate_dom - params.rate_for + 0.5 * params.vol**2) * params.time_to_expiry
    )
    denominator = params.vol * np.sqrt(params.time_to_expiry)
    return numerator / denominator


def d2(params: BSParams) -> float:
    """Calculate d2 in Black-Scholes formula."""
    return d1(params) - params.vol * np.sqrt(params.time_to_expiry)


def call_price(params: BSParams) -> float:
    """European call option price."""
    if params.time_to_expiry <= 0:
        return max(params.spot - params.strike, 0)

    d1_val = d1(params)
    d2_val = d2(params)

    return (
        params.spot * np.exp(-params.rate_for * params.time_to_expiry) * norm.cdf(d1_val) -
        params.strike * np.exp(-params.rate_dom * params.time_to_expiry) * norm.cdf(d2_val)
    )


def put_price(params: BSParams) -> float:
    """European put option price."""
    if params.time_to_expiry <= 0:
        return max(params.strike - params.spot, 0)

    d1_val = d1(params)
    d2_val = d2(params)

    return (
        params.strike * np.exp(-params.rate_dom * params.time_to_expiry) * norm.cdf(-d2_val) -
        params.spot * np.exp(-params.rate_for * params.time_to_expiry) * norm.cdf(-d1_val)
    )


def delta(params: BSParams, option_type: Literal["call", "put"] = "call") -> float:
    """Option delta (sensitivity to spot)."""
    if params.time_to_expiry <= 0:
        if option_type == "call":
            return 1.0 if params.spot > params.strike else 0.0
        else:
            return -1.0 if params.spot < params.strike else 0.0

    d1_val = d1(params)
    discount = np.exp(-params.rate_for * params.time_to_expiry)

    if option_type == "call":
        return discount * norm.cdf(d1_val)
    else:
        return discount * (norm.cdf(d1_val) - 1)


def gamma(params: BSParams) -> float:
    """
    Option gamma (second derivative w.r.t. spot).
    Same for calls and puts.

    Gamma = e^(-rf*T) * N'(d1) / (S * sigma * sqrt(T))
    """
    if params.time_to_expiry <= 0:
        return 0.0

    d1_val = d1(params)
    discount = np.exp(-params.rate_for * params.time_to_expiry)

    return (
        discount * norm.pdf(d1_val) /
        (params.spot * params.vol * np.sqrt(params.time_to_expiry))
    )


def vega(params: BSParams) -> float:
    """Option vega (sensitivity to volatility)."""
    if params.time_to_expiry <= 0:
        return 0.0

    d1_val = d1(params)
    discount = np.exp(-params.rate_for * params.time_to_expiry)

    return params.spot * discount * norm.pdf(d1_val) * np.sqrt(params.time_to_expiry)


def theta(params: BSParams, option_type: Literal["call", "put"] = "call") -> float:
    """Option theta (time decay, per year)."""
    if params.time_to_expiry <= 0:
        return 0.0

    d1_val = d1(params)
    d2_val = d2(params)
    sqrt_t = np.sqrt(params.time_to_expiry)

    # First term: time decay of gamma
    term1 = -(
        params.spot * np.exp(-params.rate_for * params.time_to_expiry) *
        norm.pdf(d1_val) * params.vol / (2 * sqrt_t)
    )

    if option_type == "call":
        term2 = params.rate_for * params.spot * np.exp(-params.rate_for * params.time_to_expiry) * norm.cdf(d1_val)
        term3 = -params.rate_dom * params.strike * np.exp(-params.rate_dom * params.time_to_expiry) * norm.cdf(d2_val)
    else:
        term2 = params.rate_for * params.spot * np.exp(-params.rate_for * params.time_to_expiry) * norm.cdf(-d1_val)
        term3 = -params.rate_dom * params.strike * np.exp(-params.rate_dom * params.time_to_expiry) * norm.cdf(-d2_val)

    return term1 - term2 - term3


def gamma_dollar(params: BSParams) -> float:
    """
    Dollar gamma: 0.5 * Gamma * S^2 * sigma^2 * dt
    This represents the expected hedging cost per unit time.

    Returns gamma contribution to P&L for a 1% move.
    """
    return 0.5 * gamma(params) * params.spot**2 * 0.01**2


def expected_hedging_cost(params: BSParams, dt: float = 1/252) -> float:
    """
    Expected hedging cost over time interval dt.
    Based on gamma and realized variance.

    Cost = 0.5 * Gamma * S^2 * sigma^2 * dt
    """
    return 0.5 * gamma(params) * params.spot**2 * params.vol**2 * dt
