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


def test_atm_greeks_match_standard_reference_values():
    values = greeks(100, 100, 1, 0.05, 0.2, "call")
    assert values["delta"] == pytest.approx(0.6368306512, abs=1e-9)
    assert values["gamma"] == pytest.approx(0.01876201735, abs=1e-10)
    assert values["vega"] == pytest.approx(37.52403469, abs=1e-7)
    assert values["theta"] == pytest.approx(-6.414027546, abs=1e-8)
    assert values["rho"] == pytest.approx(53.23248155, abs=1e-7)


def test_implied_volatility_recovers_input():
    observed = price(100, 105, 0.75, 0.03, 0.27, "put")
    assert implied_volatility(observed, 100, 105, 0.75, 0.03, "put") == pytest.approx(0.27, abs=1e-8)


def test_invalid_option_price_is_rejected():
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(200, 100, 100, 1, 0.05, "call")


def test_deep_out_of_the_money_prices_remain_finite_and_nonnegative():
    call = price(1.0, 100.0, 1.0, 0.05, 0.01, "call")
    put = price(1.0, 100.0, 1.0, 0.05, 0.01, "put")
    assert call >= 0.0 and put >= 0.0
    assert call < 1e-12
    assert put == pytest.approx(100.0 * 2.718281828459045 ** -0.05 - 1.0)


def test_expiry_implied_volatility_requires_intrinsic_price():
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(50.0, 100.0, 100.0, 0.0, 0.05, "call")


def test_implied_volatility_rejects_fractional_iteration_limit_and_nonconvergence():
    with pytest.raises(ValueError, match="iterations"):
        implied_volatility(10.0, 100.0, 100.0, 1.0, 0.05, "call", max_iterations=2.5)
    with pytest.raises(RuntimeError, match="did not converge"):
        implied_volatility(
            10.0, 100.0, 100.0, 1.0, 0.05, "call", max_iterations=1, tolerance=1e-16
        )

