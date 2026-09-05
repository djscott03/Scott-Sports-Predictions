"""
Hold-out check of the market/model blend — the only honest way to tune it.

  python run.py nfl backtest 2019 2025            # writes backtest_nfl.csv
  python holdout.py backtest_nfl.csv 2019-2022 2023-2025

Fits the blend weight on the first season range, reports what that weight does on
the second, and buckets the held-out games by |edge| to show whether bigger model
disagreements win more often. If the best weight is ~1.0 and the buckets sit at
50%, the ratings add nothing at the close (the 2026-09-04 finding for both leagues).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

WEIGHTS = np.round(np.arange(0.5, 1.001, 0.05), 2)
BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]


def blend_mae(d: pd.DataFrame, w: float) -> float:
    return float((d.margin - (w * d.market_margin + (1 - w) * d.model_margin)).abs().mean())


def fit_weight(fit: pd.DataFrame) -> tuple[float, dict]:
    maes = {w: blend_mae(fit, w) for w in WEIGHTS}
    return min(maes, key=maes.get), maes


def edge_buckets(test: pd.DataFrame, w: float) -> pd.DataFrame:
    """ATS record of the model's side, by size of its disagreement with the line at weight w."""
    t = test.copy()
    t["edge"] = (w * t.market_margin + (1 - w) * t.model_margin) - t.market_margin
    t["side_won"] = np.sign(t.edge) * np.sign(t.margin - t.market_margin)   # +1 model side covered
    rows = []
    for lo, hi in BUCKETS:
        s = t[(t.edge.abs() >= lo) & (t.edge.abs() < hi) & (t.side_won != 0)]
        if len(s):
            rows.append(dict(edge_lo=lo, edge_hi=hi, n=len(s), ats=float((s.side_won > 0).mean())))
    return pd.DataFrame(rows)


def holdout(bt: pd.DataFrame, fit_seasons, test_seasons, current_w: float) -> dict:
    b = bt.dropna(subset=["market_margin", "model_margin", "margin"])
    fit, test = b[b.season.isin(fit_seasons)], b[b.season.isin(test_seasons)]
    best, maes = fit_weight(fit)
    return dict(n_fit=len(fit), n_test=len(test), best_w=best,
                fit_mae_best=maes[best], fit_mae_market=maes[1.0],
                test_mae_best=blend_mae(test, best), test_mae_market=blend_mae(test, 1.0),
                test_mae_current=blend_mae(test, current_w), buckets=edge_buckets(test, current_w))


def _span(s: str) -> list[int]:
    a, b = s.split("-"); return list(range(int(a), int(b) + 1))


if __name__ == "__main__":
    path, fit_s, test_s = sys.argv[1], _span(sys.argv[2]), _span(sys.argv[3])
    w_now = float(sys.argv[4]) if len(sys.argv) > 4 else (0.70 if "nfl" in path else 0.65)
    r = holdout(pd.read_csv(path), fit_s, test_s, w_now)
    print(f"fit {fit_s[0]}-{fit_s[-1]} (n={r['n_fit']}): best w={r['best_w']:.2f} "
          f"MAE {r['fit_mae_best']:.3f} vs market-only {r['fit_mae_market']:.3f}")
    print(f"test {test_s[0]}-{test_s[-1]} (n={r['n_test']}): w={r['best_w']:.2f} MAE {r['test_mae_best']:.3f} | "
          f"market-only {r['test_mae_market']:.3f} | current w={w_now:.2f} {r['test_mae_current']:.3f}")
    print(f"\nheld-out ATS by |edge| at w={w_now:.2f} (breakeven .524):")
    print(r["buckets"].to_string(index=False))
