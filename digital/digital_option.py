"""
Digital (binary) option pricing and Greeks.

Digital options have discontinuous payoffs at the strike, which creates
hedging challenges:
- Infinite gamma at strike as expiry approaches
- Practical hedging requires approximation (call spread)

The strike shifting problem: when should you lock in a hedge by
shifting the strike vs. waiting and hoping spot moves away?
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from models.black_scholes import BSParams, d1, d2


class DigitalType(Enum):
    """Type of digital option."""
    CALL = "call"  # Pays if S > K at expiry
    PUT = "put"    # Pays if S < K at expiry


@dataclass
class DigitalOption:
    """Specification of a digital option."""
    spot: float
    strike: float
    vol: float
    rate_dom: float
    rate_for: float
    time_to_expiry: float
    notional: float = 1.0  # Payout amount
    option_type: DigitalType = DigitalType.CALL

    def to_bs_params(self) -> BSParams:
        """Convert to BSParams for vanilla calculations."""
        return BSParams(
            spot=self.spot,
            strike=self.strike,
            vol=self.vol,
            rate_dom=self.rate_dom,
            rate_for=self.rate_for,
            time_to_expiry=self.time_to_expiry
        )

    @property
    def moneyness(self) -> float:
        """Log-moneyness ln(S/K)."""
        return np.log(self.spot / self.strike)

    @property
    def normalized_moneyness(self) -> float:
        """Moneyness in vol units: ln(S/K) / (sigma * sqrt(T))."""
        if self.time_to_expiry <= 0:
            return np.inf if self.spot > self.strike else -np.inf
        return self.moneyness / (self.vol * np.sqrt(self.time_to_expiry))

    def with_spot(self, new_spot: float) -> "DigitalOption":
        """Return copy with updated spot."""
        return DigitalOption(
            spot=new_spot,
            strike=self.strike,
            vol=self.vol,
            rate_dom=self.rate_dom,
            rate_for=self.rate_for,
            time_to_expiry=self.time_to_expiry,
            notional=self.notional,
            option_type=self.option_type
        )

    def with_time(self, new_time: float) -> "DigitalOption":
        """Return copy with updated time to expiry."""
        return DigitalOption(
            spot=self.spot,
            strike=self.strike,
            vol=self.vol,
            rate_dom=self.rate_dom,
            rate_for=self.rate_for,
            time_to_expiry=new_time,
            notional=self.notional,
            option_type=self.option_type
        )


def digital_price(option: DigitalOption) -> float:
    """
    Price a digital option under Black-Scholes.

    Digital call: e^(-r_d * T) * N(d2)
    Digital put:  e^(-r_d * T) * N(-d2)
    """
    if option.time_to_expiry <= 0:
        if option.option_type == DigitalType.CALL:
            return option.notional if option.spot > option.strike else 0.0
        else:
            return option.notional if option.spot < option.strike else 0.0

    params = option.to_bs_params()
    d2_val = d2(params)
    discount = np.exp(-option.rate_dom * option.time_to_expiry)

    if option.option_type == DigitalType.CALL:
        return option.notional * discount * norm.cdf(d2_val)
    else:
        return option.notional * discount * norm.cdf(-d2_val)


def digital_delta(option: DigitalOption) -> float:
    """
    Delta of a digital option.

    d(Digital)/dS = e^(-r_d*T) * N'(d2) * (1 / (S * sigma * sqrt(T)))
    """
    if option.time_to_expiry <= 0:
        return 0.0

    params = option.to_bs_params()
    d2_val = d2(params)
    discount = np.exp(-option.rate_dom * option.time_to_expiry)

    delta_val = (
        option.notional * discount * norm.pdf(d2_val) /
        (option.spot * option.vol * np.sqrt(option.time_to_expiry))
    )

    if option.option_type == DigitalType.PUT:
        delta_val = -delta_val

    return delta_val


def digital_gamma(option: DigitalOption) -> float:
    """
    Gamma of a digital option.

    This explodes near the strike as time approaches expiry,
    which is the core hedging problem for digitals.
    """
    if option.time_to_expiry <= 0:
        return 0.0

    params = option.to_bs_params()
    d2_val = d2(params)
    discount = np.exp(-option.rate_dom * option.time_to_expiry)
    vol_sqrt_t = option.vol * np.sqrt(option.time_to_expiry)

    # d(delta)/dS
    gamma_val = (
        -option.notional * discount * norm.pdf(d2_val) * d2_val /
        (option.spot**2 * vol_sqrt_t**2)
    )

    # Note: gamma sign depends on moneyness and option type
    # For calls: negative gamma if ITM, positive if OTM
    if option.option_type == DigitalType.PUT:
        gamma_val = -gamma_val

    return gamma_val


def call_spread_approx_price(option: DigitalOption, spread_width: float) -> float:
    """
    Approximate digital using a call spread.

    Digital call ≈ (1/spread_width) * (Call(K - spread_width/2) - Call(K + spread_width/2))

    This is the standard hedging approach for digitals.
    """
    from models.black_scholes import call_price, put_price

    half_width = spread_width / 2

    lower_params = BSParams(
        spot=option.spot,
        strike=option.strike - half_width,
        vol=option.vol,
        rate_dom=option.rate_dom,
        rate_for=option.rate_for,
        time_to_expiry=option.time_to_expiry
    )

    upper_params = BSParams(
        spot=option.spot,
        strike=option.strike + half_width,
        vol=option.vol,
        rate_dom=option.rate_dom,
        rate_for=option.rate_for,
        time_to_expiry=option.time_to_expiry
    )

    if option.option_type == DigitalType.CALL:
        return option.notional * (call_price(lower_params) - call_price(upper_params)) / spread_width
    else:
        return option.notional * (put_price(upper_params) - put_price(lower_params)) / spread_width


def call_spread_approx_delta(option: DigitalOption, spread_width: float) -> float:
    """Delta of the call-spread approximation."""
    from models.black_scholes import delta as bs_delta

    half_width = spread_width / 2

    lower_params = BSParams(
        spot=option.spot,
        strike=option.strike - half_width,
        vol=option.vol,
        rate_dom=option.rate_dom,
        rate_for=option.rate_for,
        time_to_expiry=option.time_to_expiry
    )

    upper_params = BSParams(
        spot=option.spot,
        strike=option.strike + half_width,
        vol=option.vol,
        rate_dom=option.rate_dom,
        rate_for=option.rate_for,
        time_to_expiry=option.time_to_expiry
    )

    if option.option_type == DigitalType.CALL:
        return option.notional * (bs_delta(lower_params, "call") - bs_delta(upper_params, "call")) / spread_width
    else:
        return option.notional * (bs_delta(upper_params, "put") - bs_delta(lower_params, "put")) / spread_width


def call_spread_approx_gamma(option: DigitalOption, spread_width: float) -> float:
    """Gamma of the call-spread approximation."""
    from models.black_scholes import gamma as bs_gamma

    half_width = spread_width / 2

    lower_params = BSParams(
        spot=option.spot,
        strike=option.strike - half_width,
        vol=option.vol,
        rate_dom=option.rate_dom,
        rate_for=option.rate_for,
        time_to_expiry=option.time_to_expiry
    )

    upper_params = BSParams(
        spot=option.spot,
        strike=option.strike + half_width,
        vol=option.vol,
        rate_dom=option.rate_dom,
        rate_for=option.rate_for,
        time_to_expiry=option.time_to_expiry
    )

    return option.notional * (bs_gamma(lower_params) - bs_gamma(upper_params)) / spread_width


def conservative_spread_price(option: DigitalOption, spread_width: float) -> float:
    """
    Price the conservative call/put spread for strike shifting.

    For LONG digital call:
        Shift to call spread: Long Call(K) - Short Call(K + width)
        Payoff: 0 if S <= K, ramp from 0 to 1 if K < S < K+width, 1 if S >= K+width
        This is always <= digital call payoff, so shift always costs money.

    For LONG digital put:
        Shift to put spread: Long Put(K) - Short Put(K - width)
        Payoff: 0 if S >= K, ramp from 0 to 1 if K-width < S < K, 1 if S <= K-width
        This is always <= digital put payoff, so shift always costs money.

    The shift cost = Digital Price - Conservative Spread Price (always positive).
    """
    from models.black_scholes import call_price, put_price

    if option.option_type == DigitalType.CALL:
        # Long Call(K), Short Call(K + width)
        lower_params = BSParams(
            spot=option.spot,
            strike=option.strike,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        upper_params = BSParams(
            spot=option.spot,
            strike=option.strike + spread_width,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        # Normalize by spread width so max payoff = 1
        return option.notional * (call_price(lower_params) - call_price(upper_params)) / spread_width
    else:
        # Long Put(K), Short Put(K - width)
        upper_params = BSParams(
            spot=option.spot,
            strike=option.strike,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        lower_params = BSParams(
            spot=option.spot,
            strike=option.strike - spread_width,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        return option.notional * (put_price(upper_params) - put_price(lower_params)) / spread_width


def conservative_spread_delta(option: DigitalOption, spread_width: float) -> float:
    """Delta of the conservative spread."""
    from models.black_scholes import delta as bs_delta

    if option.option_type == DigitalType.CALL:
        lower_params = BSParams(
            spot=option.spot,
            strike=option.strike,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        upper_params = BSParams(
            spot=option.spot,
            strike=option.strike + spread_width,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        return option.notional * (bs_delta(lower_params, "call") - bs_delta(upper_params, "call")) / spread_width
    else:
        upper_params = BSParams(
            spot=option.spot,
            strike=option.strike,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        lower_params = BSParams(
            spot=option.spot,
            strike=option.strike - spread_width,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        return option.notional * (bs_delta(upper_params, "put") - bs_delta(lower_params, "put")) / spread_width


def conservative_spread_gamma(option: DigitalOption, spread_width: float) -> float:
    """Gamma of the conservative spread."""
    from models.black_scholes import gamma as bs_gamma

    if option.option_type == DigitalType.CALL:
        lower_params = BSParams(
            spot=option.spot,
            strike=option.strike,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        upper_params = BSParams(
            spot=option.spot,
            strike=option.strike + spread_width,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        return option.notional * (bs_gamma(lower_params) - bs_gamma(upper_params)) / spread_width
    else:
        upper_params = BSParams(
            spot=option.spot,
            strike=option.strike,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        lower_params = BSParams(
            spot=option.spot,
            strike=option.strike - spread_width,
            vol=option.vol,
            rate_dom=option.rate_dom,
            rate_for=option.rate_for,
            time_to_expiry=option.time_to_expiry
        )
        return option.notional * (bs_gamma(upper_params) - bs_gamma(lower_params)) / spread_width
