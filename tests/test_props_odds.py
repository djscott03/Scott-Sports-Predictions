"""Offline tests for player-prop ingestion in odds.py. No network; requests.get is faked."""
import numpy as np
import pandas as pd
import pytest
import requests

from sharpmodel.odds import (PROP_COLS, PROP_MARKETS_DEFAULT, PROP_MARKETS_ALL, EVENTS_API, PROPS_API,
                             fetch_events, fetch_props, parse_props_json, load_props_csv, estimate_prop_credits)


def _event(bookmakers):
    return {"id": "ev1", "commence_time": "2026-09-06T17:00:00Z",
            "home_team": "Philadelphia Eagles", "away_team": "Dallas Cowboys", "bookmakers": bookmakers}


V4_EVENT = _event([
    {"key": "draftkings", "markets": [
        {"key": "player_pass_yds", "last_update": "t1", "outcomes": [
            {"name": "Over", "description": "Jalen Hurts", "price": -110, "point": 274.5},
            {"name": "Under", "description": "Jalen Hurts", "price": -110, "point": 274.5}]},
        {"key": "player_anytime_td", "last_update": "t2", "outcomes": [
            {"name": "Yes", "description": "Saquon Barkley", "price": -175}]}]},
    {"key": "fanduel", "markets": [
        {"key": "player_receptions", "last_update": "t3", "outcomes": [
            {"name": "over", "description": "A.J. Brown", "price": -120, "point": 5.5},
            {"name": "UNDER", "description": "A.J. Brown", "price": 100, "point": 5.5}]}]},
])


class _Resp:
    """Minimal stand-in for requests.Response."""
    def __init__(self, payload, headers=None): self._payload, self.headers = payload, headers or {}
    def json(self): return self._payload
    def raise_for_status(self): pass


def test_market_lists():
    assert len(PROP_MARKETS_DEFAULT) == 4 and len(PROP_MARKETS_ALL) == 6
    assert set(PROP_MARKETS_DEFAULT) < set(PROP_MARKETS_ALL)
    assert {"player_pass_tds", "player_anytime_td"} < set(PROP_MARKETS_ALL)


def test_parse_props_json_v4_shape():
    df = parse_props_json(V4_EVENT, "nfl")
    assert list(df.columns) == PROP_COLS
    assert len(df) == 5
    assert set(df.home) == {"PHI"} and set(df.away) == {"DAL"} and set(df.event_id) == {"ev1"}
    assert set(df.side) == {"over", "under", "yes"}                  # case-insensitive mapping
    hurts = df[df.player == "Jalen Hurts"]
    assert (hurts.line == 274.5).all() and (hurts.book == "draftkings").all()
    td = df[df.market == "player_anytime_td"].iloc[0]
    assert td.side == "yes" and np.isnan(td.line) and td.price == -175 and td.updated == "t2"
    assert (df[df.book == "fanduel"].market == "player_receptions").all()


def test_parse_props_json_skips_bad_outcomes():
    ev = _event([{"key": "draftkings", "markets": [{"key": "player_rush_yds", "last_update": "t", "outcomes": [
        {"name": "Over", "price": -110, "point": 60.5},                                  # no description
        {"name": "Over", "description": "", "price": -110, "point": 60.5},               # empty description
        {"name": "Home", "description": "Saquon Barkley", "price": -110, "point": 60.5}, # not a prop side
        {"name": "Under", "description": "Saquon Barkley", "price": -110, "point": 60.5}]}]}])
    df = parse_props_json(ev, "nfl")
    assert len(df) == 1 and df.iloc[0].side == "under"


def test_parse_props_json_no_bookmakers():
    for ev in (_event([]), {k: v for k, v in _event([]).items() if k != "bookmakers"}):
        df = parse_props_json(ev, "nfl")
        assert df.empty and list(df.columns) == PROP_COLS


def test_load_props_csv_template():
    df = load_props_csv("props_template.csv")
    assert list(df.columns) == PROP_COLS
    assert len(df) == 14
    assert set(df.event_id) == {"PHI@DAL"}
    assert df.commence.isna().all() and df.updated.isna().all()
    assert set(df.side) == {"over", "under", "yes"}
    assert df[df.side == "yes"].line.isna().all() and df[df.side != "yes"].line.notna().all()
    assert set(df.market) == {"player_pass_yds", "player_reception_yds", "player_receptions", "player_anytime_td"}
    # the stale-book setup the scanner is meant to catch: same QB, two different numbers
    hurts = df[(df.player == "Jalen Hurts") & (df.side == "over")].set_index("book").line
    assert hurts["draftkings"] == 274.5 and hurts["fanduel"] == 264.5


def test_load_props_csv_requires_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("home,away,book,market,side,line,price\nPHI,DAL,dk,player_pass_yds,over,270.5,-110\n")
    with pytest.raises(ValueError, match="player"):
        load_props_csv(str(p))


def test_estimate_prop_credits():
    assert estimate_prop_credits(16, PROP_MARKETS_DEFAULT) == 64
    assert estimate_prop_credits(16, PROP_MARKETS_ALL, "us,us2") == 192
    assert estimate_prop_credits(1, ["player_anytime_td"], "us,us2,eu") == 3
    assert estimate_prop_credits(0, PROP_MARKETS_ALL) == 0
    assert estimate_prop_credits(2, "player_pass_yds,player_rush_yds") == 4       # comma string tolerated


def test_fetch_props_refuses_before_any_network_call(monkeypatch):
    def boom(*a, **k): raise AssertionError("requests.get must not be called when over budget")
    monkeypatch.setattr(requests, "get", boom)
    ids = [f"e{i}" for i in range(16)]                                           # 16 x 4 x 1 = 64 > 60
    with pytest.raises(RuntimeError, match="64 credits") as ei:
        fetch_props("nfl", ids, api_key="k")
    assert "15 events" in str(ei.value) and "3 markets" in str(ei.value)         # tells you what to drop
    # exactly at the cap is allowed (would call the network, so still guarded by boom)
    with pytest.raises(AssertionError):
        fetch_props("nfl", ids[:15], api_key="k")
    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        monkeypatch.delenv("ODDS_API_KEY", raising=False); fetch_props("nfl", ids[:1])


def test_fetch_props_stops_when_remaining_below_reserve(monkeypatch, capsys):
    remaining = iter(["100", "40", "30"])
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        return _Resp(V4_EVENT, {"x-requests-last": "4", "x-requests-remaining": next(remaining)})
    monkeypatch.setattr(requests, "get", fake_get)
    df = fetch_props("nfl", ["e1", "e2", "e3"], api_key="k", max_credits=60, reserve=50)
    assert len(calls) == 2                                                       # third event never fetched
    assert calls[0][0] == PROPS_API.format(sport="americanfootball_nfl", event_id="e1")
    assert calls[0][2] == 30 and calls[0][1]["markets"] == ",".join(PROP_MARKETS_DEFAULT)
    assert calls[0][1]["regions"] == "us" and calls[0][1]["oddsFormat"] == "american"
    assert list(df.columns) == PROP_COLS and len(df) == 10
    out = capsys.readouterr().out
    assert "[props] DAL@PHI: cost 4, remaining 100" in out and "WARNING" in out


def test_fetch_props_empty_when_no_books(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_event([]), {"x-requests-remaining": "400"}))
    df = fetch_props("nfl", ["e1"], api_key="k")
    assert df.empty and list(df.columns) == PROP_COLS


def test_fetch_events_free_endpoint(monkeypatch, capsys):
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        return _Resp([{"id": "e1", "commence_time": "2026-09-06T17:00:00Z",
                       "home_team": "Philadelphia Eagles", "away_team": "Dallas Cowboys"},
                      {"id": "e2", "commence_time": "2026-09-07T00:20:00Z",
                       "home_team": "Kansas City Chiefs", "away_team": "Los Angeles Chargers"}])
    monkeypatch.setattr(requests, "get", fake_get)
    df = fetch_events("nfl", api_key="k")
    assert calls[0][0] == EVENTS_API.format(sport="americanfootball_nfl") and calls[0][2] == 30
    assert list(df.columns) == ["event_id", "commence", "home", "away"]
    assert list(df.home) == ["PHI", "KC"] and list(df.away) == ["DAL", "LAC"]
    assert str(df.commence.dtype) == "datetime64[ns, UTC]" and df.commence.iloc[0].hour == 17
    assert capsys.readouterr().out == ""                                         # free call: no quota chatter
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp([]))
    empty = fetch_events("nfl", api_key="k")
    assert empty.empty and list(empty.columns) == ["event_id", "commence", "home", "away"]
