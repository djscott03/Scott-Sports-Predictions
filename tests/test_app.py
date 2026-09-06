"""Dashboard smoke tests via streamlit's AppTest. No network: requests.get is faked or forbidden."""
import os
import pandas as pd
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

import sharpmodel
from sharpmodel import props
from sharpmodel.odds import ODDS_API, EVENTS_API, PROPS_API

TABS = ["+EV plays", "Arbs & middles", "Best lines", "Model card", "Props"]
NFL = "americanfootball_nfl"
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")   # streamlit >= 1.5x resolves
                                                                                             # relative paths against this file


class _Resp:
    def __init__(self, payload, headers=None): self._payload, self.headers = payload, headers or {}
    def json(self): return self._payload
    def raise_for_status(self): pass


def _upcoming():
    now = pd.Timestamp.now(tz="UTC")
    iso = lambda h: (now + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [dict(id="e1", commence_time=iso(30), home_team="Philadelphia Eagles", away_team="Dallas Cowboys"),
            dict(id="e2", commence_time=iso(100), home_team="Kansas City Chiefs", away_team="Los Angeles Chargers")]


def _prop_event(events):
    """Per-event payload: the same QB line at two books, one of them 20 yards stale."""
    o = lambda side, line: {"name": side, "description": "Jalen Hurts", "price": -110, "point": line}
    return dict(events[0], bookmakers=[
        {"key": "draftkings", "markets": [{"key": "player_pass_yds", "last_update": "t", "outcomes": [o("Over", 274.5), o("Under", 274.5)]}]},
        {"key": "fanduel", "markets": [{"key": "player_pass_yds", "last_update": "t", "outcomes": [o("Over", 254.5), o("Under", 254.5)]}]}])


def _run(monkeypatch, key):
    st.cache_data.clear()                                   # st.cache_data is process-wide across AppTest runs
    if key: monkeypatch.setenv("ODDS_API_KEY", key)
    else: monkeypatch.delenv("ODDS_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_no_key_shows_warning_on_both_boards(monkeypatch):
    def boom(*a, **k): raise AssertionError("no network without a key")
    monkeypatch.setattr(requests, "get", boom)
    at = _run(monkeypatch, None)
    assert [t.label for t in at.tabs] == TABS
    assert len(at.warning) == 2 and all("ODDS_API_KEY" in w.value for w in at.warning)


def _game_event(events):
    """Game-line payload: pinnacle -2.5 / fanduel +3.5 at -110 -- a cross-book middle on the 3."""
    ev = dict(events[0])
    return [dict(ev, bookmakers=[
        {"key": "pinnacle", "markets": [{"key": "spreads", "last_update": "t", "outcomes": [
            {"name": "Philadelphia Eagles", "price": -110, "point": -2.5}, {"name": "Dallas Cowboys", "price": -110, "point": 2.5}]}]},
        {"key": "fanduel", "markets": [{"key": "spreads", "last_update": "t", "outcomes": [
            {"name": "Philadelphia Eagles", "price": -110, "point": -3.5}, {"name": "Dallas Cowboys", "price": -110, "point": 3.5}]}]}])]


def test_arbs_and_middles_tab_shows_the_cross_book_middle(monkeypatch):
    events = _upcoming()
    def fake_get(url, params=None, timeout=None):
        if url == ODDS_API.format(sport=NFL): return _Resp(_game_event(events), {"x-requests-remaining": "450"})
        if url == EVENTS_API.format(sport=NFL): return _Resp(events, {"x-requests-remaining": "450"})
        raise AssertionError(f"unexpected call: {url}")
    monkeypatch.setattr(requests, "get", fake_get)
    def no_nflverse(*a, **k): raise RuntimeError("offline test")
    monkeypatch.setattr(sharpmodel, "load_nfl", no_nflverse)          # model card unavailable -> market-only fairs
    at = _run(monkeypatch, "k")
    assert any("Model unavailable" in i.value for i in at.info)
    tab = at.tabs[1]
    # sidebar default 'nonus': pinnacle still anchors the fair but cannot be a leg -> nothing to show
    assert tab.label == "Arbs & middles" and len(tab.dataframe) == 0
    assert any("No cross-book arbs" in s.value for s in tab.success) and at.metric[3].value == "0"
    box = [t for t in at.text_input if t.label.startswith("Exclude books")][0]
    assert box.value == "nonus"
    box.set_value("").run()                                            # allow every book as a leg
    assert not at.exception, at.exception
    tab = at.tabs[1]
    assert len(tab.dataframe) == 1
    view = tab.dataframe[0].value
    assert list(view["leg A"]) == ["PHI -2.5 -110 @ pinnacle"] and list(view["leg B"]) == ["DAL +3.5 -110 @ fanduel"]
    assert list(view["window"]) == ["3"] and list(view["type"]) == ["middle"] and "player" not in view.columns
    assert float(view["miss cost %"].iloc[0]) == 4.55 and 5 < float(view["middle %"].iloc[0]) < 10   # key-number pmf
    assert float(view["EV %"].iloc[0]) > 0 and any("worse-priced leg first" in c.value for c in tab.caption)
    assert [m.label for m in at.metric][:4] == ["Games", "Books", "+EV lines", "Arbs & middles"]
    assert at.metric[3].value == "1"
    # kickoff window: only e1 has lines (30 h out); a 20 h window empties the board without a new odds pull
    assert at.metric[0].value == "1"
    [n for n in at.number_input if n.label.startswith("Kickoff within")][0].set_value(20).run()
    assert not at.exception, at.exception
    assert at.metric[0].value == "0" and at.metric[3].value == "0"
    assert any("No cross-book arbs" in s.value for s in at.tabs[1].success)


def test_props_tab_never_fetches_on_load(monkeypatch):
    """Loading the page may hit the game-line board (1 credit) and the FREE events list, never the per-event endpoint."""
    events, calls = _upcoming(), []
    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url == ODDS_API.format(sport=NFL): return _Resp([])
        if url == EVENTS_API.format(sport=NFL): return _Resp(events, {"x-requests-remaining": "450"})
        raise AssertionError(f"unexpected (billable) call: {url}")
    monkeypatch.setattr(requests, "get", fake_get)
    at = _run(monkeypatch, "k")
    assert calls == [ODDS_API.format(sport=NFL), EVENTS_API.format(sport=NFL)]
    btn = [b for b in at.button if b.label.startswith("Scan props")]
    assert len(btn) == 1 and btn[0].label == "Scan props (≈4 credits)"   # 1 game inside 72h x 4 default markets
    assert "props" not in at.session_state and "props_odds" not in at.session_state and not at.dataframe
    assert any("450 left this month" in c.value for c in at.caption)


def test_scan_button_bills_once_and_the_board_survives_reruns(monkeypatch):
    """One click = one per-event call; the raw prices land in session_state; reruns re-use them (no new call)."""
    events, calls = _upcoming(), []
    billed = PROPS_API.format(sport=NFL, event_id="e1")
    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url == ODDS_API.format(sport=NFL): return _Resp([])
        if url == EVENTS_API.format(sport=NFL): return _Resp(events, {"x-requests-remaining": "450"})
        if url == billed: return _Resp(_prop_event(events), {"x-requests-last": "4", "x-requests-remaining": "446"})
        raise AssertionError(f"unexpected call: {url}")
    monkeypatch.setattr(requests, "get", fake_get)
    def no_nflverse(*a, **k): raise RuntimeError("offline test")
    monkeypatch.setattr(props, "load_player_weeks", no_nflverse)     # projections fail -> market-only, data still kept
    at = _run(monkeypatch, "k")
    [btn] = [b for b in at.button if b.label.startswith("Scan props")]
    btn.click().run()
    assert not at.exception, at.exception
    assert calls.count(billed) == 1 and not at.error
    raw = at.session_state["props_odds"]
    assert len(raw["odds"]) == 4 and raw["failed"] == [] and list(raw["odds"].columns)[:4] == ["event_id", "commence", "home", "away"]
    board = at.session_state["props"]["ev"]
    assert set(zip(board.book, board.side)) == {("fanduel", "over"), ("draftkings", "under")}   # both beat the 264.5 consensus
    assert (board.ev_pct >= 0.015).all() and (board["flags"] == "no_proj").all() and len(at.dataframe) == 2
    mid = at.session_state["props"]["mid"]                            # the same two books make a 20-yard middle
    assert len(mid) == 1 and mid.iloc[0].window == "255-274" and mid.iloc[0].type == "middle" and mid.iloc[0].ev_pct > 0
    assert mid.iloc[0].bet_a == "J. Hurts O 254.5 -110 @ fanduel" and mid.iloc[0].bet_b == "J. Hurts U 274.5 -110 @ draftkings"
    assert "leg A" in at.dataframe[1].value.columns and "middle %" in at.dataframe[1].value.columns
    assert any("Projections unavailable" in i.value for i in at.info)
    n = len(calls)
    at.run()                                                          # rerun (auto-refresh, slider move, ...)
    assert not at.exception, at.exception
    assert len(calls) == n and calls.count(billed) == 1               # nothing re-fetched; events/odds are cached
    assert len(at.session_state["props_odds"]["odds"]) == 4 and len(at.dataframe) == 2


def test_scan_remaining_beats_the_cached_events_count(monkeypatch):
    """The scan's own x-requests-remaining is fresher than the 30-min cached events call: it drives the 'left this
    month' caption and the reserve pre-check, so a second click under the reserve is refused before any call."""
    events, calls = _upcoming(), []
    billed = PROPS_API.format(sport=NFL, event_id="e1")
    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url == ODDS_API.format(sport=NFL): return _Resp([])
        if url == EVENTS_API.format(sport=NFL): return _Resp(events, {"x-requests-remaining": "450"})
        if url == billed: return _Resp(_prop_event(events), {"x-requests-last": "4", "x-requests-remaining": "52"})
        raise AssertionError(f"unexpected call: {url}")
    monkeypatch.setattr(requests, "get", fake_get)
    def no_nflverse(*a, **k): raise RuntimeError("offline test")
    monkeypatch.setattr(props, "load_player_weeks", no_nflverse)
    at = _run(monkeypatch, "k")
    assert any("450 left this month" in c.value for c in at.caption)
    [btn] = [b for b in at.button if b.label.startswith("Scan props")]
    btn.click().run()
    assert not at.exception, at.exception
    assert calls.count(billed) == 1 and not at.error and at.session_state["props_odds"]["remaining"] == 52
    at.run()
    assert not at.exception, at.exception
    caps = [c.value for c in at.caption]
    assert any("52 left this month" in c for c in caps) and not any("450 left" in c for c in caps)
    [btn] = [b for b in at.button if b.label.startswith("Scan props")]
    btn.click().run()                                                 # 52 - 4 < the 50 reserve: refused up front
    assert not at.exception, at.exception
    assert calls.count(billed) == 1 and any("52 credits remaining - 4 per call < reserve=50" in e.value for e in at.error)
    assert len(at.session_state["props_odds"]["odds"]) == 4           # the first scan's board is untouched
