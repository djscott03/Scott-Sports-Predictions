"""Freshness: minutes since a book last touched a price, carried through +EV rows, top picks and middles."""
import os
import numpy as np
import pandas as pd

from sharpmodel.odds import add_age, fresh, find_ev, top_picks, load_odds_csv, STALE_MIN, parse_odds_json
from sharpmodel.middles import find_middles, game_fairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = pd.Timestamp("2026-09-13T16:00:00Z")
EV = dict(event_id="e", commence="2026-09-13T17:00:00Z", home="PHI", away="DAL")


def _rows(*rows):
    return pd.DataFrame([dict(EV, **r) for r in rows])


def _board():
    """pinnacle fresh at -7 -105/-105; draftkings 50 minutes stale at -5.5 -110/-110 (the +EV home side)."""
    return _rows(dict(book="pinnacle", market="spreads", side="home", line=-7.0, price=-105, updated="2026-09-13T15:58:00Z"),
                 dict(book="pinnacle", market="spreads", side="away", line=7.0, price=-105, updated="2026-09-13T15:58:00Z"),
                 dict(book="draftkings", market="spreads", side="home", line=-5.5, price=-110, updated="2026-09-13T15:10:00Z"),
                 dict(book="draftkings", market="spreads", side="away", line=5.5, price=-110, updated="2026-09-13T15:10:00Z"))


def test_add_age_and_fresh():
    o = add_age(_board(), now=NOW)
    assert o.age_min.round(1).tolist() == [2.0, 2.0, 50.0, 50.0]
    assert add_age(_board().drop(columns="updated"), now=NOW).age_min.isna().all()        # no timestamp -> NaN
    csv = add_age(load_odds_csv(os.path.join(ROOT, "lines_template.csv")))
    assert csv.age_min.isna().all()
    assert len(fresh(o, 45)) == 2 and set(fresh(o, 45).book) == {"pinnacle"}
    assert len(fresh(o, 0)) == 4 and len(fresh(o, 60)) == 4                                # 0 = keep everything
    assert len(fresh(csv, 45)) == len(csv)                                                # NaN rows are kept
    assert STALE_MIN == 45


def test_find_ev_carries_age_and_lag_vs_the_sharp_book():
    o = add_age(_board(), now=NOW)
    ev = find_ev(o, "nfl", min_ev=-1.0)
    dk = ev[(ev.book == "draftkings") & (ev.side == "home")].iloc[0]
    assert round(dk.age_min) == 50 and round(dk.lag_min) == 48                             # 48 min behind pinnacle
    assert dk.updated == "2026-09-13T15:10:00Z" and dk.ev_pct > 0                          # -5.5 vs a -7 fair: the stale +EV line
    pin = ev[(ev.book == "pinnacle") & (ev.side == "home")].iloc[0]
    assert round(pin.age_min) == 2 and round(pin.lag_min) == 0
    assert "age_min" in top_picks(ev).columns and round(top_picks(ev).iloc[0].age_min) == 50
    ev2 = find_ev(_board(), "nfl", min_ev=-1.0)                                            # age computed on demand
    assert ev2.age_min.notna().all()
    kept = fresh(ev, 45)
    assert set(kept.book) == {"pinnacle"}


def test_middles_carry_the_older_legs_age():
    o = add_age(_rows(dict(book="A", market="spreads", side="home", line=-2.5, price=-110, updated="2026-09-13T15:55:00Z"),
                      dict(book="B", market="spreads", side="away", line=3.5, price=-110, updated="2026-09-13T15:20:00Z")),
                now=NOW)
    fairs = pd.DataFrame([dict(event_id="e", market="spreads", player=np.nan, mu=3.0, sd=np.nan, dist="nfl_margin")])
    m = find_middles(o, fairs)
    assert len(m) == 1 and round(m.iloc[0].age_min) == 40                                  # B is the older leg
    assert fresh(m, 45).shape[0] == 1 and fresh(m, 30).empty
    m2 = find_middles(_rows(dict(book="A", market="spreads", side="home", line=-2.5, price=-110),
                            dict(book="B", market="spreads", side="away", line=3.5, price=-110)), fairs)
    assert m2.age_min.isna().all()                                                          # CSV legs: unknown age


def test_parse_keeps_last_update():
    ev = [dict(id="x", commence_time="2026-09-13T17:00:00Z", home_team="Philadelphia Eagles", away_team="Dallas Cowboys",
               bookmakers=[{"key": "fanduel", "markets": [{"key": "h2h", "last_update": "2026-09-13T15:30:00Z", "outcomes": [
                   {"name": "Philadelphia Eagles", "price": -150}, {"name": "Dallas Cowboys", "price": 130}]}]}])]
    o = add_age(parse_odds_json(ev, "nfl"), now=NOW)
    assert o.updated.tolist() == ["2026-09-13T15:30:00Z"] * 2 and o.age_min.round().tolist() == [30.0, 30.0]
