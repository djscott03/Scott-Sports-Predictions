"""
Turning a predicted margin into prices, and prices into bets.

Margin distributions (empirical, from closing-line residuals):
  NFL spread sd ~13.4   NFL total sd ~10.3
  CFB spread sd ~16.5   CFB total sd ~13.0
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

LEAGUE_SD = {"nfl": {"spread": 13.4, "total": 10.3},
             "cfb": {"spread": 16.5, "total": 13.0}}


# ---------- odds conversions ----------
def american_to_prob(odds: float) -> float:
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def prob_to_american(p: float) -> float:
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def decimal_from_american(odds: float) -> float:
    odds = float(odds)
    return 1 + odds / 100 if odds > 0 else 1 + 100 / -odds


def devig(odds_a: float, odds_b: float, method: str = "power") -> tuple[float, float]:
    """Remove the vig from a two-way market. 'power' is closest to how sharps price
    favorite/longshot bias; 'multiplicative' is the simple normalization."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    if method == "multiplicative":
        s = pa + pb
        return pa / s, pb / s
    # power method: find k so that pa^k + pb^k = 1
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if pa ** k + pb ** k > 1: lo = k
        else: hi = k
    return pa ** k, pb ** k


# ---------- cover probabilities ----------
def cover_probs(mu: float, line: float, sd: float) -> dict:
    """
    P(home covers) for home spread `line` (negative = home favored).
    mu = predicted home margin. Uses a discretized normal so that integer
    key numbers (3, 7) produce proper push probability.
    Returns {'win','push','loss'} from the HOME side's perspective.
    """
    need = -line  # home covers if margin > need
    if abs(need - round(need)) < 1e-9:      # integer line -> pushes possible
        win = 1 - norm.cdf(need + 0.5, mu, sd)
        push = norm.cdf(need + 0.5, mu, sd) - norm.cdf(need - 0.5, mu, sd)
    else:
        win = 1 - norm.cdf(need, mu, sd); push = 0.0
    return {"win": win, "push": push, "loss": 1 - win - push}


def total_probs(mu_total: float, line: float, sd: float) -> dict:
    """P(over) with push handling."""
    if abs(line - round(line)) < 1e-9:
        over = 1 - norm.cdf(line + 0.5, mu_total, sd)
        push = norm.cdf(line + 0.5, mu_total, sd) - norm.cdf(line - 0.5, mu_total, sd)
    else:
        over = 1 - norm.cdf(line, mu_total, sd); push = 0.0
    return {"over": over, "push": push, "under": 1 - over - push}


def moneyline_prob(mu: float, sd: float) -> float:
    """P(home wins outright); ties are negligible."""
    return 1 - norm.cdf(0, mu, sd)


# ---------- edge & staking ----------
def edge_and_kelly(p_win: float, p_push: float, odds: float,
                   kelly_fraction: float = 0.25, max_stake: float = 0.03) -> dict:
    """
    p_win/p_push: our probabilities for the side we're betting.
    odds: the American price we can actually get.
    Kelly with pushes: bet resolves on non-push outcomes.
    """
    b = decimal_from_american(odds) - 1
    p_loss = 1 - p_win - p_push
    ev = p_win * b - p_loss                       # per unit staked
    breakeven = american_to_prob(odds) * (1 - p_push) if p_push else american_to_prob(odds)
    kelly = (p_win * b - p_loss) / b if b > 0 else 0.0
    stake = float(np.clip(kelly * kelly_fraction, 0, max_stake))
    return {"ev": ev, "edge_prob": p_win - (1 - p_push) * american_to_prob(odds),
            "kelly_full": kelly, "stake_frac": stake, "breakeven": breakeven}


def line_to_prob_edge(model_margin: float, market_margin: float, sd: float) -> float:
    """Convert a point edge into a probability edge vs a fair 50/50 line."""
    return norm.cdf((model_margin - market_margin) / sd) - 0.5
