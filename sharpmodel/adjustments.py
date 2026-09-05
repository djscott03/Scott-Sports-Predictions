"""
Situational adjustments — the layer where actual edge comes from.

Power ratings describe a team on average. The market is near-perfect at that.
Edge lives in game-specific information the ratings don't know yet:
  * QB change   (the single largest source of stale ratings in the NFL)
  * Wind/weather (mostly totals; wind is under-priced more often than cold)
  * Rest        (short week, bye, travel)
  * Injuries    (plug your own numbers in via `manual`)
  * Motivation  (week 18 resting starters, bowl opt-outs in CFB)

Every function returns (margin_adj, total_adj) in points to add to the MODEL
number before blending with the market. Magnitudes below are conservative
literature/industry priors; calibrate them with your own backtests.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def rest_adjustment(home_rest: float, away_rest: float) -> float:
    """Rest differential. Bye (13-14 days) ~ +0.5, Thursday short week ~ -0.5."""
    def val(r):
        if not np.isfinite(r): return 0.0
        if r <= 5: return -0.5
        if r >= 13: return 0.5
        return 0.0
    return val(home_rest) - val(away_rest)


def weather_total_adjustment(wind: float, temp: float, roof: str) -> float:
    """Outdoor only. ~-0.3 pts of total per mph of wind above 10; cold is minor."""
    if roof in ("dome", "closed") or not np.isfinite(wind):
        return 0.0
    adj = -0.30 * max(wind - 10, 0)
    if np.isfinite(temp) and temp < 25:
        adj -= 0.5
    return adj


def qb_change_flags(hist: pd.DataFrame, games: pd.DataFrame, lookback: int = 6) -> pd.DataFrame:
    """
    Flag games where a team's listed starter differs from its recent regular starter.
    We do NOT auto-adjust (backup quality varies from -2 to -10 pts); we flag so you
    either skip the game or apply a manual number. Requires home_qb/away_qb columns.
    """
    if "home_qb" not in hist:
        return games.assign(home_qb_change=False, away_qb_change=False)
    long = pd.concat([
        hist[["date", "home", "home_qb"]].rename(columns={"home": "team", "home_qb": "qb"}),
        hist[["date", "away", "away_qb"]].rename(columns={"away": "team", "away_qb": "qb"}),
    ]).dropna().sort_values("date")
    out = games.copy()
    for side in ("home", "away"):
        flags = []
        for _, g in out.iterrows():
            prev = long[(long.team == g[side]) & (long.date < g.date)].tail(lookback)
            usual = prev.qb.mode().iloc[0] if len(prev) else None
            flags.append(bool(usual and pd.notna(g.get(f"{side}_qb")) and g[f"{side}_qb"] != usual))
        out[f"{side}_qb_change"] = flags
    return out


def apply_adjustments(games: pd.DataFrame, manual: dict | None = None) -> pd.DataFrame:
    """
    Adds `margin_adj` and `total_adj` columns.
    manual: {game_id: {"margin": +2.0, "total": -3.0}} for injuries / news you price yourself.
    """
    g = games.copy()
    g["margin_adj"] = [rest_adjustment(r.get("home_rest", np.nan), r.get("away_rest", np.nan))
                       for _, r in g.iterrows()] if "home_rest" in g else 0.0
    g["total_adj"] = [weather_total_adjustment(r.get("wind", np.nan), r.get("temp", np.nan), r.get("roof", ""))
                      for _, r in g.iterrows()] if "wind" in g else 0.0
    for gid, adj in (manual or {}).items():
        m = g.game_id == gid
        g.loc[m, "margin_adj"] += adj.get("margin", 0.0)
        g.loc[m, "total_adj"] += adj.get("total", 0.0)
    return g
