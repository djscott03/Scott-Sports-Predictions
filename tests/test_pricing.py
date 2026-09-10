"""Offline unit tests for the pricing math. No network, runs in <1s."""
import math
import numpy as np
import pandas as pd
import pytest

from sharpmodel.pricing import (american_to_prob, prob_to_american, decimal_from_american,
                                devig, cover_probs, total_probs, moneyline_prob, edge_and_kelly)
from sharpmodel.margins import cover_probs_emp
from sharpmodel.odds import sharp_fair, find_ev, find_arbs, best_lines, load_odds_csv, parse_odds_json


def test_american_conversions_round_trip():
    for odds in (-110, -105, +150, -300, +270):
        p = american_to_prob(odds)
        assert 0 < p < 1
        assert prob_to_american(p) == pytest.approx(odds, abs=1e-6)
    assert american_to_prob(+100) == american_to_prob(-100) == 0.5   # same price, two spellings
    assert decimal_from_american(-110) == pytest.approx(1 + 100 / 110)
    assert decimal_from_american(+150) == pytest.approx(2.5)


def test_devig_sums_to_one_and_keeps_favourite():
    for method in ("power", "multiplicative"):
        a, b = devig(-330, +270, method)
        assert a + b == pytest.approx(1.0, abs=1e-9)
        assert a > b
    # symmetric price -> exactly 50/50
    a, b = devig(-105, -105)
    assert a == pytest.approx(0.5, abs=1e-9)


def test_cover_probs_integer_line_has_push_mass():
    cp = cover_probs(mu=3.0, line=-3, sd=13.4)
    assert cp["push"] > 0
    assert cp["win"] + cp["push"] + cp["loss"] == pytest.approx(1.0)
    # centred exactly on the number -> symmetric win/loss
    assert cp["win"] == pytest.approx(cp["loss"], abs=1e-9)
    # half-point line -> no push
    cp2 = cover_probs(mu=3.0, line=-3.5, sd=13.4)
    assert cp2["push"] == 0.0
    assert cp2["win"] < 0.5   # need to win by 4+, mean is 3


def test_total_probs_and_moneyline():
    tp = total_probs(47.0, 47.5, 10.3)
    assert tp["push"] == 0.0 and tp["over"] < 0.5
    assert moneyline_prob(0.0, 13.4) == pytest.approx(0.5)
    assert moneyline_prob(7.0, 13.4) > 0.65


def test_edge_and_kelly_signs_and_cap():
    fair = edge_and_kelly(p_win=0.5, p_push=0.0, odds=-110)
    assert fair["ev"] < 0 and fair["stake_frac"] == 0.0        # coin flip at -110 is -EV
    good = edge_and_kelly(p_win=0.60, p_push=0.0, odds=-110, kelly_fraction=1.0, max_stake=0.03)
    assert good["ev"] > 0 and good["stake_frac"] == 0.03       # full Kelly is big; cap binds
    assert good["edge_prob"] == pytest.approx(0.60 - american_to_prob(-110))


def test_sharp_fair_inverts_symmetric_line_exactly():
    odds = pd.DataFrame([
        dict(event_id="e", home="PHI", away="DAL", book="pinnacle", market="spreads", side="home", line=-7.0, price=-105),
        dict(event_id="e", home="PHI", away="DAL", book="pinnacle", market="spreads", side="away", line=7.0, price=-105),
        dict(event_id="e", home="PHI", away="DAL", book="pinnacle", market="totals", side="over", line=47.5, price=-110),
        dict(event_id="e", home="PHI", away="DAL", book="pinnacle", market="totals", side="under", line=47.5, price=-110),
    ])
    f = sharp_fair(odds, "nfl")
    assert f["ref_book"] == "pinnacle" and f["ref_n"] == 1
    # NFL spreads invert on the key-number pmf: 'exactly' now means the mu whose P(cover -7 | no push) is 1/2. That
    # is not 7.0 -- the fat 3 below 7 holds more mass than 8..10 above it, so the mean sits at ~8.3 -- but pricing
    # the same -7 on the same pmf gives back a coin flip. The plain Normal (CFB) still lands on 7.0 exactly.
    cp = cover_probs_emp(f["mu_margin"], -7.0)
    assert cp["win"] == pytest.approx(cp["loss"], abs=1e-6) and 7.0 < f["mu_margin"] < 9.0
    assert sharp_fair(odds, "cfb")["mu_margin"] == pytest.approx(7.0, abs=1e-6)
    assert f["mu_total"] == pytest.approx(47.5, abs=1e-6)                         # totals stay Normal


def test_find_ev_on_template_flags_stale_fanduel_total():
    odds = load_odds_csv("lines_template.csv")
    ev = find_ev(odds, "nfl", model_fair=None, min_ev=0.015)
    assert len(ev) == 1
    row = ev.iloc[0]
    assert (row.book, row.market, row.side, row.line) == ("fanduel", "totals", "under", 48.5)
    assert row.ev_pct > 0.02 and 0 < row.kelly_stake <= 0.03
    # pinnacle's own lines must never show as +EV against themselves
    assert not (ev.book == "pinnacle").any()


def test_find_ev_model_blend_moves_fair():
    odds = load_odds_csv("lines_template.csv")
    model = pd.DataFrame([dict(home="PHI", away="DAL", model_margin=7.0, model_total=40.0)])
    ev = find_ev(odds, "nfl", model_fair=model, model_weight=0.5, min_ev=-1.0)
    under = ev[(ev.market == "totals") & (ev.side == "under") & (ev.book == "fanduel")].iloc[0]
    sharp_mu = sharp_fair(odds, "nfl")["mu_total"]          # ~47.44: pinnacle is -104/-106, not symmetric
    assert under.fair_mu == pytest.approx(0.5 * sharp_mu + 0.5 * 40.0)
    assert under.fair_mu < sharp_mu                          # model pulled it down


def test_arbs_and_best_lines():
    odds = pd.DataFrame([
        dict(event_id="e", home="A", away="B", book="x", market="ml", side="home", line=np.nan, price=+120),
        dict(event_id="e", home="A", away="B", book="y", market="ml", side="away", line=np.nan, price=+110),
        dict(event_id="e", home="A", away="B", book="z", market="ml", side="away", line=np.nan, price=-130),
    ])
    arbs = find_arbs(odds)
    assert len(arbs) == 1 and arbs.iloc[0].profit_pct > 0
    bl = best_lines(odds)
    assert bl[bl.side == "away"].iloc[0].book == "y"


def test_parse_odds_json_v4_shape():
    events = [{
        "id": "abc", "commence_time": "2026-09-06T17:00:00Z",
        "home_team": "Philadelphia Eagles", "away_team": "Dallas Cowboys",
        "bookmakers": [{"key": "pinnacle", "markets": [
            {"key": "h2h", "last_update": "t", "outcomes": [
                {"name": "Philadelphia Eagles", "price": -330}, {"name": "Dallas Cowboys", "price": 270}]},
            {"key": "spreads", "last_update": "t", "outcomes": [
                {"name": "Philadelphia Eagles", "price": -105, "point": -7.0},
                {"name": "Dallas Cowboys", "price": -105, "point": 7.0}]},
            {"key": "totals", "last_update": "t", "outcomes": [
                {"name": "Over", "price": -104, "point": 47.5}, {"name": "Under", "price": -106, "point": 47.5}]},
        ]}]}]
    df = parse_odds_json(events, "nfl")
    assert set(df.home) == {"PHI"} and set(df.away) == {"DAL"}
    assert set(df.market) == {"ml", "spreads", "totals"}
    assert set(df.side) == {"home", "away", "over", "under"}
    assert len(df) == 6
