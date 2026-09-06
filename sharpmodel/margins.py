"""
Empirical NFL margin distribution: a discretized Normal re-weighted at the key numbers.

NFL margins are not Normal. On nflverse 1999-2025 regular season (6,967 games with a
closing spread) P(|margin| = 3) is .150 vs .058 for a Normal(0, 13.4), 7 is .091 vs .052,
while 1, 2 and 8 come in *below* the Normal and ties are ~0.2% instead of ~3%. Anything
that lives on a single integer -- a push, a half-point buy, the value in a middle -- is
priced wrong off the plain Normal.

Method (fit_key_weights): for every game the discretized Normal(mu = closing spread,
sd) gives an expected probability for each integer margin; summed over all games that is
an expected count per absolute margin k, compared with the observed count and shrunk
toward 1 with a pseudo-count:  w_k = (obs_k + pseudo) / (exp_k + pseudo).  margin_pmf
then multiplies the Normal's mass at each integer by w_|k| and renormalizes. The bumps
are unconditional on the spread (one table for all games), which understates the 3 a
little when the spread *is* 3 but is far closer than the Normal.

This module is OPT-IN. engine.py / pricing.py still price the weekly card off the plain
Normal: switching them would re-price the frozen record in predictions/ and mix two
distributions in one graded history. middles.py prices middles and half-point values
off this pmf. Use weights=None for CFB and for totals (not fitted here).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

NFL_GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
NFL_SD = 13.2          # sd of (margin - closing spread), 1999-2025 REG: 13.19

# Fitted 2026-09-05 on nflverse games.csv, game_type == 'REG', seasons 1999-2025,
# n = 6967 games with result and spread_line; sd=13.2, pseudo=50 (pseudo_tie=5 at k=0), max_k=30.
# Reproduce:
#   python -c "import pandas as pd; from sharpmodel.margins import *; g = pd.read_csv(NFL_GAMES_URL);
#              print(fit_key_weights(g[g.season.between(1999, 2025)]))"
# Ties: 15 observed vs 189 expected (raw ratio 0.079). The shared pseudo=50 pulled that to 0.272 and
# the pmf gave ~0.8% to a tie at a pick'em where reality is ~0.2%; k=0 gets its own pseudo_tie=5,
# (15+5)/(189+5) = 0.103, so a pick'em tie is now ~0.3%.
NFL_KEY_WEIGHTS = {
    0: 0.103, 1: 0.802, 2: 0.793, 3: 2.605, 4: 0.943, 5: 0.732, 6: 1.167, 7: 1.763,
    8: 0.834, 9: 0.448, 10: 1.256, 11: 0.625, 12: 0.507, 13: 0.787, 14: 1.322, 15: 0.567,
    16: 0.764, 17: 1.201, 18: 0.913, 19: 0.612, 20: 0.991, 21: 1.272, 22: 0.698, 23: 0.795,
    24: 1.351, 25: 0.922, 26: 0.830, 27: 1.169, 28: 1.518, 29: 0.858, 30: 0.847,
}


def _grid(lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer margins lo..hi and the bin edges (k +- .5); the end bins run to +-inf so
    the discretized Normal sums to exactly 1."""
    k = np.arange(int(lo), int(hi) + 1)
    edges = np.concatenate(([-np.inf], k[:-1] + 0.5, [np.inf]))
    return k, edges


def fit_key_weights(games: pd.DataFrame, sd: float = NFL_SD, pseudo: float = 50.0,
                    max_k: int = 30, pseudo_tie: float = 5.0) -> dict:
    """
    {abs_margin: weight} from a games table with `result` (home margin) and `spread_line`
    (home-favoured positive) columns; regular season only if `game_type` is present.
    weight_k = (observed_k + pseudo) / (expected_k + pseudo), expected from the
    discretized Normal(spread, sd) per game, +k and -k folded together. k = 0 is ties and
    uses pseudo_tie instead: ties are so rare (15 in 6,967 games vs 189 expected) that the
    shared pseudo-count would drag the raw 0.08 up to 0.27 and quadruple the pmf's tie mass.
    """
    g = games
    if "game_type" in g.columns:
        g = g[g["game_type"] == "REG"]
    g = g.dropna(subset=["result", "spread_line"])
    mus, counts = np.unique(g["spread_line"].astype(float).values, return_counts=True)
    k, edges = _grid(-75, 75)
    mass = np.diff(norm.cdf(edges[None, :], mus[:, None], sd), axis=1)   # (n_unique, n_k)
    expected = pd.Series(counts @ mass, index=k)
    observed = g["result"].round().astype(int).abs().value_counts()
    out = {}
    for a in range(0, max_k + 1):
        exp_a = expected[a] + (expected[-a] if a > 0 else 0.0)
        obs_a = float(observed.get(a, 0))
        ps = pseudo_tie if a == 0 else pseudo
        out[a] = float((obs_a + ps) / (exp_a + ps))
    return out


def margin_pmf(mu: float, sd: float = NFL_SD, weights: dict | None = NFL_KEY_WEIGHTS,
               lo: int = -75, hi: int = 75) -> pd.Series:
    """P(margin = k) for integer k in lo..hi: discretized Normal(mu, sd) times
    weights[|k|] (1.0 if missing), renormalized. weights=None -> plain Normal."""
    k, edges = _grid(lo, hi)
    mass = np.diff(norm.cdf(edges, mu, sd))
    if weights:
        mass = mass * np.array([weights.get(int(abs(i)), 1.0) for i in k])
    return pd.Series(mass / mass.sum(), index=pd.Index(k, name="margin"), name="p")


def prob_between(pmf: pd.Series, lo: float, hi: float,
                 inclusive: tuple[bool, bool] = (True, True)) -> float:
    """P(lo <= margin <= hi) with each edge inclusive or strict; +-inf are fine."""
    k = pmf.index.values
    left = (k >= lo) if inclusive[0] else (k > lo)
    right = (k <= hi) if inclusive[1] else (k < hi)
    return float(pmf.values[left & right].sum())


def cover_probs_emp(mu: float, line: float, sd: float = NFL_SD,
                    weights: dict | None = NFL_KEY_WEIGHTS) -> dict:
    """pricing.cover_probs on the empirical pmf: {'win','push','loss'} for the HOME side
    of spread `line` (negative = home favoured). Home covers if margin > -line; an
    integer line pushes with the pmf's mass on that number."""
    pmf = margin_pmf(mu, sd, weights)
    need = -line
    push = float(pmf.get(int(round(need)), 0.0)) if abs(need - round(need)) < 1e-9 else 0.0
    win = prob_between(pmf, need, np.inf, (False, True))
    return {"win": win, "push": push, "loss": 1 - win - push}


def moneyline_prob_emp(mu: float, sd: float = NFL_SD,
                       weights: dict | None = NFL_KEY_WEIGHTS) -> float:
    """P(home wins outright) = P(margin > 0); ties are not a win."""
    return prob_between(margin_pmf(mu, sd, weights), 0, np.inf, (False, True))
