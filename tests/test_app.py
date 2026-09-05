"""Dashboard smoke tests via streamlit's AppTest. No network: requests.get is faked or forbidden."""
import pandas as pd
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from sharpmodel.odds import ODDS_API, EVENTS_API

TABS = ["+EV plays", "Arbs", "Best lines", "Model card", "Props"]


class _Resp:
    def __init__(self, payload): self._payload, self.headers = payload, {}
    def json(self): return self._payload
    def raise_for_status(self): pass


def _run(monkeypatch, key):
    st.cache_data.clear()                                   # st.cache_data is process-wide across AppTest runs
    if key: monkeypatch.setenv("ODDS_API_KEY", key)
    else: monkeypatch.delenv("ODDS_API_KEY", raising=False)
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_no_key_shows_warning_on_both_boards(monkeypatch):
    def boom(*a, **k): raise AssertionError("no network without a key")
    monkeypatch.setattr(requests, "get", boom)
    at = _run(monkeypatch, None)
    assert [t.label for t in at.tabs] == TABS
    assert len(at.warning) == 2 and all("ODDS_API_KEY" in w.value for w in at.warning)


def test_props_tab_never_fetches_on_load(monkeypatch):
    """Loading the page may hit the game-line board (1 credit) and the FREE events list, never the per-event endpoint."""
    now = pd.Timestamp.now(tz="UTC")
    iso = lambda h: (now + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = [dict(id="e1", commence_time=iso(30), home_team="Philadelphia Eagles", away_team="Dallas Cowboys"),
              dict(id="e2", commence_time=iso(100), home_team="Kansas City Chiefs", away_team="Los Angeles Chargers")]
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url == ODDS_API.format(sport="americanfootball_nfl"): return _Resp([])
        if url == EVENTS_API.format(sport="americanfootball_nfl"): return _Resp(events)
        raise AssertionError(f"unexpected (billable) call: {url}")
    monkeypatch.setattr(requests, "get", fake_get)
    at = _run(monkeypatch, "k")
    assert calls == [ODDS_API.format(sport="americanfootball_nfl"), EVENTS_API.format(sport="americanfootball_nfl")]
    btn = [b for b in at.button if b.label.startswith("Scan props")]
    assert len(btn) == 1 and btn[0].label == "Scan props (≈4 credits)"   # 1 game inside 72h x 4 default markets
    assert "props" not in at.session_state and not at.dataframe
