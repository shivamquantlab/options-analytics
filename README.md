# Options Analytics

## Research question
How do European option model prices and local sensitivities change with volatility, and can volatility be recovered from a valid option price?

## Methodology and formulas
The package implements Black-Scholes-Merton with continuous compounding: d1=[ln(S/K)+(r+sigma^2/2)T]/(sigma*sqrt(T)), d2=d1-sigma*sqrt(T). Call value is S*N(d1)-K*exp(-rT)*N(d2); put value is K*exp(-rT)*N(-d2)-S*N(-d1). It reports delta, gamma, vega, theta per year, rho per unit rate, and bisection implied volatility.

## Assumptions and limitations
- European exercise, constant volatility, constant risk-free rate, frictionless trading, no dividends, and lognormal underlying price dynamics.
- The report is a deterministic model scenario (S=K=100, T=1, r=5%, volatility=20%); it contains no market quote and is not a fair-value recommendation.
- Greeks are local model derivatives; at expiry and zero volatility the package uses explicit boundary conventions.
- American exercise, discrete dividends, volatility smiles, borrow costs, and transaction costs are not modeled.

## Reproducibility
With Python 3.11+, run python -m pip install -e ".[test]", then python -m pytest and options-demo. The command regenerates reports/call_volatility_sensitivity.png and reports/summary.json; it uses no external data, APIs, or credentials.

