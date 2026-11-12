import pytest

from options_analytics.black_scholes import greeks, implied_volatility, price


def test_black_scholes_reference_prices_and_put_call_parity():
    call = price(100, 100, 1, 0.05, 0.2, "call")
    put = price(100, 100, 1, 0.05, 0.2, "put")
    assert call == pytest.approx(10.45058357, abs=1e-7)
    assert put == pytest.approx(5.57352602, abs=1e-7)
    assert call - put == pytest.approx(100 - 100 * 2.718281828459045 ** -0.05)


def test_greeks_are_finite_and_vega_is_positive():
    values = greeks(100, 100, 1, 0.05, 0.2, "call")
    assert 0 < values["delta"] < 1
    assert values["gamma"] > 0
    assert values["vega"] > 0


def test_implied_volatility_recovers_input():
    observed = price(100, 105, 0.75, 0.03, 0.27, "put")
    assert implied_volatility(observed, 100, 105, 0.75, 0.03, "put") == pytest.approx(0.27, abs=1e-8)


def test_invalid_option_price_is_rejected():
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(200, 100, 100, 1, 0.05, "call")

