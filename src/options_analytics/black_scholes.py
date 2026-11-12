"""Black-Scholes-Merton European option prices, Greeks, and implied volatility."""

from math import erf, exp, isfinite, log, pi, sqrt


def _validate(spot: float, strike: float, maturity: float, rate: float, volatility: float,
              option_type: str) -> None:
    if spot <= 0 or strike <= 0 or maturity < 0 or volatility < 0:
        raise ValueError("spot and strike must be positive; maturity and volatility non-negative")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    if not all(isfinite(value) for value in (spot, strike, maturity, rate, volatility)):
        raise ValueError("inputs must be finite")


def _cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Price a European call or put with continuously compounded rate."""
    _validate(spot, strike, maturity, rate, volatility, option_type)
    discounted_strike = strike * exp(-rate * maturity)
    if maturity == 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if volatility == 0:
        return max(spot - discounted_strike, 0.0) if option_type == "call" else max(discounted_strike - spot, 0.0)
    root_t = sqrt(maturity)
    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    if option_type == "call":
        return spot * _cdf(d1) - discounted_strike * _cdf(d2)
    return discounted_strike * _cdf(-d2) - spot * _cdf(-d1)


def greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Return delta, gamma, vega, theta/year, and rho per unit rate."""
    _validate(spot, strike, maturity, rate, volatility, option_type)
    if maturity == 0:
        intrinsic_delta = float(spot > strike) if option_type == "call" else -float(spot < strike)
        return {"delta": intrinsic_delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    discounted_strike = strike * exp(-rate * maturity)
    if volatility == 0:
        in_the_money = spot > discounted_strike if option_type == "call" else spot < discounted_strike
        delta = float(in_the_money) if option_type == "call" else -float(in_the_money)
        rho = (maturity * discounted_strike if option_type == "call" else -maturity * discounted_strike) if in_the_money else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": rho}
    root_t = sqrt(maturity)
    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    density = _pdf(d1)
    gamma = density / (spot * volatility * root_t)
    vega = spot * density * root_t
    common_theta = -spot * density * volatility / (2.0 * root_t)
    if option_type == "call":
        return {
            "delta": _cdf(d1),
            "gamma": gamma,
            "vega": vega,
            "theta": common_theta - rate * discounted_strike * _cdf(d2),
            "rho": maturity * discounted_strike * _cdf(d2),
        }
    return {
        "delta": _cdf(d1) - 1.0,
        "gamma": gamma,
        "vega": vega,
        "theta": common_theta + rate * discounted_strike * _cdf(-d2),
        "rho": -maturity * discounted_strike * _cdf(-d2),
    }


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str = "call",
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
    upper_volatility: float = 5.0,
) -> float:
    """Invert a European option price by bisection over volatility."""
    _validate(spot, strike, maturity, rate, 0.0, option_type)
    if (
        not all(isfinite(value) for value in (market_price, tolerance, upper_volatility))
        or market_price < 0
        or tolerance <= 0
        or max_iterations < 1
        or upper_volatility <= 0
    ):
        raise ValueError("market_price, tolerance, iterations, and upper volatility are invalid")
    lower_bound = max(spot - strike * exp(-rate * maturity), 0.0) if option_type == "call" else max(
        strike * exp(-rate * maturity) - spot, 0.0
    )
    upper_bound = spot if option_type == "call" else strike * exp(-rate * maturity)
    if market_price < lower_bound - tolerance or market_price > upper_bound + tolerance:
        raise ValueError("market price violates European no-arbitrage bounds")
    if abs(market_price - lower_bound) <= tolerance:
        return 0.0
    low, high = 0.0, upper_volatility
    if price(spot, strike, maturity, rate, high, option_type) < market_price:
        raise ValueError("implied volatility exceeds the configured search bound")
    for _ in range(max_iterations):
        middle = (low + high) / 2.0
        estimate = price(spot, strike, maturity, rate, middle, option_type)
        if abs(estimate - market_price) <= tolerance:
            return middle
        if estimate < market_price:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0

