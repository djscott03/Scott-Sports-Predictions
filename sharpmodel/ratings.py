"""
Rating engines.

Everything here is a regularized least-squares ("ridge") team-strength model,
which is the same family sharps use as a baseline (Massey / SRS with priors).

Key ideas:
  * Recency weighting   - old games matter less (half-life in weeks).
  * Margin capping      - blowout points past ~28 are noise.
  * Bayesian priors     - preseason ratings enter as pseudo-games, so week 1
                          isn't garbage and week 10 isn't over-fitted.
  * Market ratings      - fit the same model on closing lines to get the
                          market's implied power ratings.
  * Off/Def split       - separate offense/defense ratings power the totals model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


def _weighted_ridge(X, y, w, lam, penalty):
    """Solve (X'WX + lam*diag(penalty)) b = X'Wy."""
    A = X.T @ (X * w[:, None]) + lam * np.diag(penalty)
    b = X.T @ (y * w)
    return np.linalg.solve(A, b)


def recency_weights(week_index: np.ndarray, as_of_index: float, half_life: float) -> np.ndarray:
    age = np.clip(as_of_index - week_index, 0, None)
    return 0.5 ** (age / half_life)


@dataclass
class MarginModel:
    """Team ratings in points: predicted home margin = r_home - r_away + HFA."""
    lam: float = 6.0
    half_life: float = 8.0          # weeks
    margin_cap: float = 28.0
    hfa_prior: float = 1.6          # NFL ~1.5-2.0, CFB ~2.5-3.0
    hfa_prior_weight: float = 25.0
    prior_weight: float = 4.0       # a preseason prior counts as this many games
    ratings: pd.Series = field(default_factory=pd.Series)
    hfa: float = 0.0
    teams: list = field(default_factory=list)

    def fit(self, games: pd.DataFrame, target: str, as_of_index: float,
            priors: dict | None = None):
        """
        games: rows with home, away, neutral (0/1), week_index, and `target`
               (a home-minus-away quantity, e.g. margin or market_margin).
        """
        g = games.dropna(subset=[target]).copy()
        teams = sorted(set(g.home) | set(g.away) | set((priors or {}).keys()))
        idx = {t: i for i, t in enumerate(teams)}
        n, T = len(g), len(teams)

        X = np.zeros((n, T + 1))
        X[np.arange(n), g.home.map(idx).values] = 1.0
        X[np.arange(n), g.away.map(idx).values] = -1.0
        X[:, T] = 1.0 - g.neutral.values
        y = np.clip(g[target].values.astype(float), -self.margin_cap, self.margin_cap)
        w = recency_weights(g.week_index.values, as_of_index, self.half_life)

        # Pseudo-observations: preseason priors and HFA prior
        rows, ys, ws = [], [], []
        if priors:
            for t, r in priors.items():
                if t in idx and np.isfinite(r):
                    row = np.zeros(T + 1); row[idx[t]] = 1.0
                    rows.append(row); ys.append(r); ws.append(self.prior_weight)
        row = np.zeros(T + 1); row[T] = 1.0
        rows.append(row); ys.append(self.hfa_prior); ws.append(self.hfa_prior_weight)

        X = np.vstack([X, np.array(rows)])
        y = np.concatenate([y, ys]); w = np.concatenate([w, ws])

        penalty = np.ones(T + 1); penalty[T] = 0.0     # don't shrink HFA to 0 (prior handles it)
        beta = _weighted_ridge(X, y, w, self.lam, penalty)

        r = beta[:T] - beta[:T].mean()                 # center ratings at 0
        self.ratings = pd.Series(r, index=teams).sort_values(ascending=False)
        self.hfa = float(beta[T])
        self.teams = teams
        return self

    def predict(self, home, away, neutral=0):
        rh = self.ratings.get(home, 0.0); ra = self.ratings.get(away, 0.0)
        return rh - ra + self.hfa * (1 - np.asarray(neutral))


@dataclass
class PointsModel:
    """Offense/defense ratings for totals: pts(team i vs j) = mu + off_i + def_j + hfa*home."""
    lam: float = 6.0
    half_life: float = 8.0
    prior_weight: float = 4.0
    off: pd.Series = field(default_factory=pd.Series)
    dfn: pd.Series = field(default_factory=pd.Series)
    mu: float = 22.0
    hfa: float = 0.8

    def fit(self, games: pd.DataFrame, as_of_index: float,
            off_priors: dict | None = None, def_priors: dict | None = None):
        g = games.dropna(subset=["home_pts", "away_pts"]).copy()
        teams = sorted(set(g.home) | set(g.away))
        idx = {t: i for i, t in enumerate(teams)}
        T, n = len(teams), len(g)
        w_game = recency_weights(g.week_index.values, as_of_index, self.half_life)

        # two rows per game: home offense row, away offense row
        X = np.zeros((2 * n, 2 * T + 2)); y = np.zeros(2 * n); w = np.zeros(2 * n)
        hi, ai = g.home.map(idx).values, g.away.map(idx).values
        home_flag = 1.0 - g.neutral.values
        r = np.arange(n)
        X[r, hi] = 1; X[r, T + ai] = 1; X[r, 2 * T] = 1; X[r, 2 * T + 1] = home_flag
        y[r] = g.home_pts.values; w[r] = w_game
        X[n + r, ai] = 1; X[n + r, T + hi] = 1; X[n + r, 2 * T] = 1; X[n + r, 2 * T + 1] = -home_flag
        y[n + r] = g.away_pts.values; w[n + r] = w_game

        rows, ys, ws = [], [], []
        for prior_map, offset in ((off_priors, 0), (def_priors, T)):
            for t, v in (prior_map or {}).items():
                if t in idx and np.isfinite(v):
                    row = np.zeros(2 * T + 2); row[offset + idx[t]] = 1
                    rows.append(row); ys.append(v); ws.append(self.prior_weight)
        if rows:
            X = np.vstack([X, rows]); y = np.concatenate([y, ys]); w = np.concatenate([w, ws])

        penalty = np.ones(2 * T + 2); penalty[2 * T] = 0.0; penalty[2 * T + 1] = 0.2
        beta = _weighted_ridge(X, y, w, self.lam, penalty)
        off, dfn = beta[:T], beta[T:2 * T]
        # identifiability: center both, push the means into mu
        self.mu = float(beta[2 * T] + off.mean() + dfn.mean())
        self.off = pd.Series(off - off.mean(), index=teams)
        self.dfn = pd.Series(dfn - dfn.mean(), index=teams)   # positive = allows more points
        self.hfa = float(beta[2 * T + 1])
        return self

    def predict(self, home, away, neutral=0):
        h = self.mu + self.off.get(home, 0) + self.dfn.get(away, 0) + self.hfa * (1 - neutral)
        a = self.mu + self.off.get(away, 0) + self.dfn.get(home, 0) - self.hfa * (1 - neutral)
        return h, a


def regress_to_mean(ratings: pd.Series, factor: float) -> dict:
    """Preseason prior = factor * last season's final rating (NFL ~0.5, CFB ~0.65)."""
    return (ratings * factor).to_dict()
