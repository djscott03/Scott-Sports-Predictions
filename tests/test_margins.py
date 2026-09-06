"""Offline tests for the empirical key-number margin pmf. Uses the hard-coded table; no network."""
import numpy as np
import pandas as pd
import pytest

from sharpmodel.pricing import cover_probs
from sharpmodel.margins import (NFL_KEY_WEIGHTS, NFL_SD, fit_key_weights, margin_pmf, prob_between,
                                cover_probs_emp, moneyline_prob_emp)


def test_table_shape_and_key_numbers():
    assert sorted(NFL_KEY_WEIGHTS) == list(range(0, 31))
    assert NFL_KEY_WEIGHTS[3] > 2.0 and NFL_KEY_WEIGHTS[7] > 1.5     # the bumps
    assert NFL_KEY_WEIGHTS[0] < 0.5                                   # ties are rare
    assert NFL_KEY_WEIGHTS[1] < 1.0 and NFL_KEY_WEIGHTS[2] < 1.0      # 1 and 2 are thin


def test_pmf_sums_to_one_and_is_integer_indexed():
    for mu in (-10.5, -3.0, 0.0, 2.5, 3.0, 7.0, 21.0):
        for w in (NFL_KEY_WEIGHTS, None):
            p = margin_pmf(mu, weights=w)
            assert p.sum() == pytest.approx(1.0, abs=1e-12)
            assert (p >= 0).all()
            assert p.index.dtype.kind == "i" and p.index.min() == -75 and p.index.max() == 75


def test_three_is_a_key_number_and_ties_are_rare():
    emp = margin_pmf(3.0)[3]
    normal = margin_pmf(3.0, weights=None)[3]
    assert emp == pytest.approx(0.092, abs=0.02)      # P(fav wins by exactly 3 | spread 3) = .092 empirically
    assert emp > 1.8 * normal
    assert margin_pmf(0.0)[0] < 0.01                  # worst case for ties is a pick'em
    assert margin_pmf(3.0)[0] < 0.01
    assert margin_pmf(7.0)[7] > 1.5 * margin_pmf(7.0, weights=None)[7]


def test_weights_none_reproduces_pricing_cover_probs():
    for sd in (13.2, 13.4):
        for mu in (-4.0, 0.0, 2.5, 3.0, 10.0):
            for line in (-7, -3, -2.5, 0, 3.5, 6):
                a = cover_probs_emp(mu, line, sd=sd, weights=None)
                b = cover_probs(mu, line, sd)
                for key in ("win", "push", "loss"):
                    assert a[key] == pytest.approx(b[key], abs=1e-9), (sd, mu, line, key)


def test_cover_probs_emp_push_is_pmf_mass():
    pmf = margin_pmf(3.0)
    cp = cover_probs_emp(3.0, -3)                     # home -3, mean 3 -> push on exactly 3
    assert cp["push"] == pytest.approx(pmf[3])
    assert cp["win"] == pytest.approx(prob_between(pmf, 3, np.inf, (False, True)))
    assert cp["win"] + cp["push"] + cp["loss"] == pytest.approx(1.0)
    assert cp["push"] > cover_probs(3.0, -3, NFL_SD)["push"] * 1.8   # far more push risk than the Normal says
    cp7 = cover_probs_emp(-7.0, 7)                    # away favourite by 7, home +7 -> push on -7
    assert cp7["push"] == pytest.approx(margin_pmf(-7.0)[-7])
    half = cover_probs_emp(3.0, -3.5)
    assert half["push"] == 0.0 and half["win"] == pytest.approx(prob_between(pmf, 4, np.inf))
    assert half["win"] == pytest.approx(cp["win"])    # -3 and -3.5 win on the same margins (>= 4) ...
    assert half["loss"] == pytest.approx(cp["loss"] + cp["push"])   # ... the half-point turns the push into a loss


def test_prob_between_edges():
    pmf = margin_pmf(0.0)
    inner = sum(pmf[k] for k in range(3, 8))
    assert prob_between(pmf, 3, 7) == pytest.approx(inner)
    assert prob_between(pmf, 3, 7, (False, False)) == pytest.approx(inner - pmf[3] - pmf[7])
    assert prob_between(pmf, 3, 7, (True, False)) == pytest.approx(inner - pmf[7])
    assert prob_between(pmf, 3, 7, (False, True)) == pytest.approx(inner - pmf[3])
    assert prob_between(pmf, 2.5, 7.5) == pytest.approx(inner)        # half-point edges: inclusivity is moot
    assert prob_between(pmf, 2.5, 7.5, (False, False)) == pytest.approx(inner)
    assert prob_between(pmf, -np.inf, np.inf) == pytest.approx(1.0)
    assert prob_between(pmf, 7, 3) == 0.0                              # empty interval


def test_moneyline_prob_emp():
    p0 = margin_pmf(0.0)[0]
    assert moneyline_prob_emp(0.0) == pytest.approx((1 - p0) / 2)     # symmetric, ties are not wins
    assert moneyline_prob_emp(0.0, weights=None) == pytest.approx((1 - margin_pmf(0.0, weights=None)[0]) / 2)
    assert moneyline_prob_emp(7.0) > 0.65
    assert moneyline_prob_emp(-7.0) == pytest.approx(1 - moneyline_prob_emp(7.0) - margin_pmf(7.0)[0])


def test_tie_weight_uses_its_own_pseudo_count():
    """Ties: 15 observed vs 189 expected. The shared pseudo=50 pulled the raw 0.079 to 0.272 (a pick'em tie ~0.8%
    vs ~0.2% empirical); k = 0 gets pseudo_tie=5 -> (15+5)/(189+5) = 0.103."""
    assert NFL_KEY_WEIGHTS[0] == pytest.approx(0.103, abs=1e-3)
    assert margin_pmf(0.0)[0] < 0.004 and margin_pmf(3.0)[0] < 0.004
    assert margin_pmf(0.0)[0] > 0.5 * 15 / 6967                                   # but not zeroed: ties happen
    # Normal data with 95% of the ties knocked out: only k = 0 moves, and it moves much further with pseudo_tie=5
    rng = np.random.default_rng(3)
    n = 30000
    spread = np.round(rng.normal(0, 6, n) * 2) / 2
    result = np.round(spread + rng.normal(0, NFL_SD, n))
    ties = np.flatnonzero(result == 0)
    redo = ties[rng.random(len(ties)) >= 0.05]
    result[redo] = np.where(rng.random(len(redo)) < 0.5, 1, -1) * np.maximum(np.round(np.abs(rng.normal(0, NFL_SD, len(redo)))), 1)
    games = pd.DataFrame({"season": 2020, "game_type": "REG", "result": result, "spread_line": spread})
    w, w50 = fit_key_weights(games), fit_key_weights(games, pseudo_tie=50.0)
    assert w[0] < 0.08 < w50[0] and w[0] < 0.7 * w50[0]
    assert all(w[k] == pytest.approx(w50[k]) for k in range(1, 31))


def test_fit_key_weights_flat_on_normal_data():
    rng = np.random.default_rng(7)
    n = 30000
    spread = np.round(rng.normal(0, 6, n) * 2) / 2                     # half-point spreads like real lines
    result = np.round(spread + rng.normal(0, NFL_SD, n))
    games = pd.DataFrame({"season": 2020, "game_type": "REG", "result": result, "spread_line": spread})
    # playoff rows all landing on 3 must be ignored, and an unplayed game (NaN result) dropped
    junk = pd.DataFrame({"season": 2020, "game_type": "POST", "result": 3.0, "spread_line": 3.0}, index=range(500))
    unplayed = pd.DataFrame({"season": 2020, "game_type": ["REG"], "result": [np.nan], "spread_line": [3.0]})
    w = fit_key_weights(pd.concat([games, junk, unplayed], ignore_index=True))
    assert sorted(w) == list(range(0, 31))
    assert all(abs(v - 1.0) < 0.15 for v in w.values()), w
    # and the same generator with a genuine bump shows up
    bumped = games.copy()
    bumped.loc[rng.random(n) < 0.08, "result"] = 3.0
    assert fit_key_weights(bumped)[3] > 1.8
