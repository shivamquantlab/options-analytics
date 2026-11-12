"""Black-Scholes-Merton European option prices, Greeks, and implied volatility."""

from math import erfc, exp, isfinite, isnan, log, pi, sqrt


def _validate(spot: float, strike: float, maturity: float, rate: float, volatility: float,
              option_type: str) -> tuple[float, float, float, float, float]:
    if not isinstance(option_type, str) or option_type not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    if any(isinstance(value, bool) for value in (spot, strike, maturity, rate, volatility)):
        raise ValueError("numeric model inputs must not be booleans")
    try:
        spot, strike, maturity, rate, volatility = (
            float(value) for value in (spot, strike, maturity, rate, volatility)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("numeric model inputs must be finite real numbers") from error
    if spot <= 0 or strike <= 0 or maturity < 0 or volatility < 0:
        raise ValueError("spot and strike must be positive; maturity and volatility non-negative")
    if not all(isfinite(value) for value in (spot, strike, maturity, rate, volatility)):
        raise ValueError("inputs must be finite")
    return spot, strike, maturity, rate, volatility


def _discounted_strike(strike: float, rate: float, maturity: float) -> float:
    try:
        discounted = strike * exp(-rate * maturity)
    except OverflowError as error:
        raise ValueError("discounted strike exceeds the finite float range") from error
    if not isfinite(discounted):
        raise ValueError("discounted strike exceeds the finite float range")
    return discounted


def _d_values(spot: float, strike: float, maturity: float, rate: float,
              volatility: float) -> tuple[float, float]:
    root_t = sqrt(maturity)
    try:
        sigma_root_t = volatility * root_t
        d1 = (log(spot) - log(strike) + (rate + 0.5 * volatility**2) * maturity) / sigma_root_t
        d2 = d1 - sigma_root_t
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("Black-Scholes d-values exceed the finite float range") from error
    if isnan(d1) or isnan(d2):
        raise ValueError("Black-Scholes d-values are undefined for these inputs")
    return d1, d2


def _cdf(value: float) -> float:
    return 0.5 * erfc(-value / sqrt(2.0))


def _pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _checked_greeks(values: dict[str, float]) -> dict[str, float]:
    if not all(isfinite(value) for value in values.values()):
        raise ValueError("Greek values exceed the finite float range")
    return values


def price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Price a European call or put with continuously compounded rate."""
    spot, strike, maturity, rate, volatility = _validate(
        spot, strike, maturity, rate, volatility, option_type
    )
    discounted_strike = _discounted_strike(strike, rate, maturity)
    if maturity == 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if volatility == 0:
        return max(spot - discounted_strike, 0.0) if option_type == "call" else max(discounted_strike - spot, 0.0)
    d1, d2 = _d_values(spot, strike, maturity, rate, volatility)
    if option_type == "call":
        value = spot * _cdf(d1) - discounted_strike * _cdf(d2)
        return min(spot, max(value, 0.0))
    value = discounted_strike * _cdf(-d2) - spot * _cdf(-d1)
    return min(discounted_strike, max(value, 0.0))


def greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Return delta, gamma, vega, theta/year, and rho per unit rate."""
    spot, strike, maturity, rate, volatility = _validate(
        spot, strike, maturity, rate, volatility, option_type
    )
    if maturity == 0:
        intrinsic_delta = float(spot > strike) if option_type == "call" else -float(spot < strike)
        return _checked_greeks(
            {"delta": intrinsic_delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        )
    discounted_strike = _discounted_strike(strike, rate, maturity)
    if volatility == 0:
        in_the_money = spot > discounted_strike if option_type == "call" else spot < discounted_strike
        delta = float(in_the_money) if option_type == "call" else -float(in_the_money)
        rho = (maturity * discounted_strike if option_type == "call" else -maturity * discounted_strike) if in_the_money else 0.0
        return _checked_greeks({"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": rho})
    root_t = sqrt(maturity)
    d1, d2 = _d_values(spot, strike, maturity, rate, volatility)
    density = _pdf(d1)
    gamma = density / (spot * volatility * root_t)
    vega = spot * density * root_t
    common_theta = -spot * density * volatility / (2.0 * root_t)
    if option_type == "call":
        return _checked_greeks({
            "delta": _cdf(d1),
            "gamma": gamma,
            "vega": vega,
            "theta": common_theta - rate * discounted_strike * _cdf(d2),
            "rho": maturity * discounted_strike * _cdf(d2),
        })
    return _checked_greeks({
        "delta": _cdf(d1) - 1.0,
        "gamma": gamma,
        "vega": vega,
        "theta": common_theta + rate * discounted_strike * _cdf(-d2),
        "rho": -maturity * discounted_strike * _cdf(-d2),
    })


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
    spot, strike, maturity, rate, _ = _validate(
        spot, strike, maturity, rate, 0.0, option_type
    )
    if any(isinstance(value, bool) for value in (market_price, tolerance, upper_volatility)):
        raise ValueError("market_price, tolerance, and upper volatility must be numeric")
    try:
        market_price, tolerance, upper_volatility = (
            float(value) for value in (market_price, tolerance, upper_volatility)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("market_price, tolerance, and upper volatility must be numeric") from error
    if (
        not all(isfinite(value) for value in (market_price, tolerance, upper_volatility))
        or market_price < 0
        or tolerance <= 0
        or isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
        or upper_volatility <= 0
    ):
        raise ValueError("market_price, tolerance, iterations, and upper volatility are invalid")
    discounted_strike = _discounted_strike(strike, rate, maturity)
    lower_bound = max(spot - discounted_strike, 0.0) if option_type == "call" else max(
        discounted_strike - spot, 0.0
    )
    upper_bound = lower_bound if maturity == 0 else (
        spot if option_type == "call" else discounted_strike
    )
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
    raise RuntimeError("implied-volatility solver did not converge within max_iterations")

