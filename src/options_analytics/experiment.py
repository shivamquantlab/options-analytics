"""Deterministic option-price curve and scenario summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .black_scholes import greeks, implied_volatility, price


def run(output_dir: Path) -> dict[str, float]:
    spot, strike, maturity, rate, volatility = 100.0, 100.0, 1.0, 0.05, 0.20
    call = price(spot, strike, maturity, rate, volatility, "call")
    put = price(spot, strike, maturity, rate, volatility, "put")
    summary = {
        "spot": spot,
        "strike": strike,
        "maturity_years": maturity,
        "continuously_compounded_rate": rate,
        "input_volatility": volatility,
        "call_price": call,
        "put_price": put,
        "call_implied_volatility": implied_volatility(call, spot, strike, maturity, rate, "call"),
        "call_delta": greeks(spot, strike, maturity, rate, volatility, "call")["delta"],
    }
    vols = np.linspace(0.05, 0.8, 100)
    call_prices = [price(spot, strike, maturity, rate, float(vol), "call") for vol in vols]
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(vols, call_prices, color="#6d28d9")
    ax.axvline(volatility, color="#64748b", ls="--", lw=1)
    ax.set(title="Black-Scholes European call sensitivity", xlabel="Annualized volatility input",
           ylabel="Model price per unit")
    ax.text(0.01, 0.98, "MODEL SCENARIO • no market quotes used",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "call_volatility_sensitivity.png", dpi=150)
    plt.close(fig)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.output_dir)
    print(f"Wrote option model report to {args.output_dir}")


if __name__ == "__main__":
    main()

