"""Offline tests for player-prop ingestion in odds.py. No network; requests.get is faked."""
import os, subprocess, sys
import numpy as np
import pandas as pd
import pytest
import requests

from sharpmodel.odds import (PROP_COLS, PROP_MARKETS_DEFAULT, PROP_MARKETS_ALL, EVENTS_API, PROPS_API,
                             OddsAPIError, fetch_odds, fetch_events, fetch_props, parse_props_json, load_props_csv,
                             estimate_prop_credits)

KEY = "SECRET-KEY-123"


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


class _Fail:
    """A response whose raise_for_status raises the way requests does: the message embeds the full URL (key included)."""
    def __init__(self, status, body='{"message":"Invalid API key"}'):
        self.status_code, self.text, self.headers = status, body, {}
    def raise_for_status(self):
        err = requests.HTTPError(f"{self.status_code} Client Error for url: {EVENTS_API}?apiKey={KEY}", response=self)
        raise err


class _BadBody:
    """A 200 whose body is not JSON (a proxy's HTML page, say): requests raises ValueError from .json()."""
    def __init__(self, headers=None): self.headers = headers or {}
    def json(self): raise ValueError("Expecting value: line 1 column 1 (char 0)")
    def raise_for_status(self): pass


def _timeout(url, params=None, timeout=None):
    raise requests.Timeout(f"HTTPSConnectionPool: read timed out for {url}?apiKey={params['apiKey']}")


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


def test_parse_props_json_null_point_is_nan():
    ev = _event([{"key": "draftkings", "markets": [{"key": "player_rush_yds", "last_update": "t", "outcomes": [
        {"name": "Over", "description": "Saquon Barkley", "price": -110, "point": None},      # explicit null
        {"name": "Under", "description": "Saquon Barkley", "price": -110}]}]}])              # absent
    df = parse_props_json(ev, "nfl")
    assert len(df) == 2 and df.line.isna().all() and str(df.line.dtype) == "float64"


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


def test_load_props_csv_validates_sides(tmp_path):
    head = "home,away,book,market,player,side,line,price\n"
    ok = tmp_path / "ok.csv"
    ok.write_text(head + "PHI,DAL,dk,player_pass_yds,Jalen Hurts, Over ,270.5,-110\n"
                         "PHI,DAL,dk,player_pass_yds,Jalen Hurts,UNDER,270.5,-110\n"
                         "PHI,DAL,dk,player_anytime_td,Saquon Barkley,Yes,,-175\n")
    assert list(load_props_csv(str(ok)).side) == ["over", "under", "yes"]         # stripped + lower-cased
    bad = tmp_path / "bad.csv"
    bad.write_text(head + "PHI,DAL,dk,player_pass_yds,Jalen Hurts,ovr,270.5,-110\n"
                          "PHI,DAL,dk,player_pass_yds,Jalen Hurts,home,270.5,-110\n")
    with pytest.raises(ValueError, match=r"unknown side\(s\) \['home', 'ovr'\].*over/under/yes/no"):
        load_props_csv(str(bad))


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
    assert list(df.columns) == PROP_COLS and len(df) == 10 and df.attrs["remaining"] == 40   # the last response's header
    out = capsys.readouterr().out
    assert "[props] DAL@PHI: cost 4, remaining 100" in out and "WARNING" in out


def test_fetch_props_empty_when_no_books(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_event([]), {"x-requests-remaining": "400"}))
    df = fetch_props("nfl", ["e1"], api_key="k")
    assert df.empty and list(df.columns) == PROP_COLS and df.attrs["failed"] == []


def test_fetch_props_refuses_when_reserve_would_be_breached(monkeypatch):
    def boom(*a, **k): raise AssertionError("requests.get must not be called under the reserve")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(RuntimeError, match="52 credits remaining - 4 per call < reserve=50"):
        fetch_props("nfl", ["e1", "e2"], api_key="k", remaining=52)                # 52 - 4 = 48 < 50
    with pytest.raises(AssertionError):                                            # 54 - 4 = 50: allowed -> network
        fetch_props("nfl", ["e1"], api_key="k", remaining=54)
    with pytest.raises(AssertionError):                                            # unknown remaining: allowed
        fetch_props("nfl", ["e1"], api_key="k", remaining=None)
    assert fetch_props("nfl", [], api_key="k", remaining=10).empty                 # nothing to fetch, nothing billed


def _events_df():
    return pd.DataFrame([dict(event_id="e1", commence=pd.Timestamp("2026-09-06T17:00Z"), home="PHI", away="DAL"),
                         dict(event_id="e2", commence=pd.Timestamp("2026-09-06T20:00Z"), home="KC", away="LAC"),
                         dict(event_id="e3", commence=pd.Timestamp("2026-09-07T00:20Z"), home="SF", away="SEA")])


def test_fetch_props_keeps_billed_events_when_a_later_call_fails(monkeypatch, capsys):
    responses = {"e1": _Resp(V4_EVENT, {"x-requests-last": "4", "x-requests-remaining": "100"}),
                 "e2": _Fail(500, "upstream error"),
                 "e3": _Resp(V4_EVENT, {"x-requests-last": "4", "x-requests-remaining": "92"})}
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append(url); return responses[url.rsplit("/", 2)[-2]]
    monkeypatch.setattr(requests, "get", fake_get)
    df = fetch_props("nfl", _events_df(), api_key=KEY)
    assert len(calls) == 3 and len(df) == 10 and df.attrs["failed"] == ["e2"]     # e1 + e3 kept, e2 skipped
    out = capsys.readouterr().out
    assert "[props] skip LAC@KC: 500" in out and KEY not in out


def test_fetch_props_stops_on_auth_or_quota_status(monkeypatch, capsys):
    for status in (401, 402, 429):
        responses = {"e1": _Resp(V4_EVENT, {"x-requests-last": "4", "x-requests-remaining": "100"}),
                     "e2": _Fail(status), "e3": _Resp(V4_EVENT, {"x-requests-remaining": "92"})}
        calls = []
        def fake_get(url, params=None, timeout=None):
            calls.append(url); return responses[url.rsplit("/", 2)[-2]]
        monkeypatch.setattr(requests, "get", fake_get)
        df = fetch_props("nfl", ["e1", "e2", "e3"], api_key=KEY)
        assert len(calls) == 2 and len(df) == 5 and df.attrs["failed"] == ["e2", "e3"]   # e3 never attempted
        out = capsys.readouterr().out
        assert f"[props] skip e2: {status}" in out and "stopping" in out and KEY not in out
    # every call failed -> nothing billed, so the error is raised instead of an empty board
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Fail(401))
    with pytest.raises(OddsAPIError, match="Odds API 401 error: .*Invalid API key") as ei:
        fetch_props("nfl", ["e1"], api_key=KEY)
    assert ei.value.status == 401 and KEY not in str(ei.value)


def test_fetch_props_skips_a_200_with_a_bad_body_and_keeps_the_rest(monkeypatch, capsys):
    """A 200 whose body is not the documented shape used to escape the per-event try and discard the billed frames."""
    ok = lambda rem: _Resp(V4_EVENT, {"x-requests-last": "4", "x-requests-remaining": rem})
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append(url); return responses[url.rsplit("/", 2)[-2]]
    monkeypatch.setattr(requests, "get", fake_get)
    for bad in (_BadBody({"x-requests-last": "4", "x-requests-remaining": "96"}),   # not JSON -> ValueError
                _Resp({"id": "e2", "bookmakers": []}, {"x-requests-remaining": "96"}),  # no home_team -> KeyError
                _Resp([V4_EVENT], {"x-requests-remaining": "96"})):                    # a list, not an event -> TypeError
        responses, calls[:] = {"e1": ok("100"), "e2": bad, "e3": ok("92")}, []
        df = fetch_props("nfl", _events_df(), api_key=KEY)
        assert len(calls) == 3 and len(df) == 10 and df.attrs["failed"] == ["e2"] and df.attrs["remaining"] == 92
        out = capsys.readouterr().out
        assert "[props] skip LAC@KC: bad response body" in out and "SEA@SF: cost 4" in out and KEY not in out
    # every body bad -> nothing usable was billed: raised, key-safe, status None like a transport error
    monkeypatch.setattr(requests, "get", lambda *a, **k: _BadBody())
    with pytest.raises(OddsAPIError, match=r"^Odds API request error: bad response body \(ValueError\)$") as ei:
        fetch_props("nfl", ["e1"], api_key=KEY)
    assert ei.value.status is None and KEY not in str(ei.value)


def test_fetch_props_stops_after_two_consecutive_transport_failures(monkeypatch, capsys):
    """Timeouts / connection errors carry no status, so they never hit STOP_STATUSES; two in a row end the loop
    (the rest are marked failed, never attempted) instead of timing out once per remaining event."""
    ok = lambda rem: _Resp(V4_EVENT, {"x-requests-last": "4", "x-requests-remaining": rem})
    calls = []
    def fake_get(url, params=None, timeout=None):
        eid = url.rsplit("/", 2)[-2]; calls.append(eid)
        r = responses[eid]
        return r(url, params, timeout) if callable(r) else r
    monkeypatch.setattr(requests, "get", fake_get)
    responses = {"e1": ok("100"), "e2": _timeout, "e3": _timeout, "e4": ok("92")}
    df = fetch_props("nfl", ["e1", "e2", "e3", "e4"], api_key=KEY)
    assert calls == ["e1", "e2", "e3"] and len(df) == 5 and df.attrs["failed"] == ["e2", "e3", "e4"]
    assert df.attrs["remaining"] == 100                                          # last response that came back
    out = capsys.readouterr().out
    assert "[props] skip e2: request error" in out and "stopping: 2 consecutive request errors" in out and KEY not in out
    # a success in between resets the count: every event is attempted
    responses, calls[:] = {"e1": _timeout, "e2": ok("100"), "e3": _timeout, "e4": ok("96")}, []
    df = fetch_props("nfl", ["e1", "e2", "e3", "e4"], api_key=KEY)
    assert calls == ["e1", "e2", "e3", "e4"] and len(df) == 10 and df.attrs["failed"] == ["e1", "e3"]
    # a failure WITH a status (500) is skipped as before and does not count toward the transport limit
    responses, calls[:] = {"e1": _timeout, "e2": _Fail(500, "upstream"), "e3": _timeout, "e4": ok("96")}, []
    df = fetch_props("nfl", ["e1", "e2", "e3", "e4"], api_key=KEY)
    assert calls == ["e1", "e2", "e3", "e4"] and len(df) == 5 and df.attrs["failed"] == ["e1", "e2", "e3"]
    assert "stopping" not in capsys.readouterr().out
    # nothing ever came back -> raised, key-safe
    monkeypatch.setattr(requests, "get", _timeout)
    with pytest.raises(OddsAPIError, match="^Odds API request error: Timeout$") as ei:
        fetch_props("nfl", ["e1", "e2", "e3"], api_key=KEY)
    assert ei.value.status is None and KEY not in str(ei.value) and "stopping: 2 consecutive" in capsys.readouterr().out


def test_fetch_props_defaults_remaining_from_the_events_frame_and_returns_the_last_header(monkeypatch):
    def boom(*a, **k): raise AssertionError("requests.get must not be called under the reserve")
    monkeypatch.setattr(requests, "get", boom)
    ev = _events_df(); ev.attrs["remaining"] = 52                                # what fetch_events attaches
    with pytest.raises(RuntimeError, match="52 credits remaining - 4 per call < reserve=50"):
        fetch_props("nfl", ev, api_key="k")                                       # remaining=None -> the frame's
    with pytest.raises(RuntimeError, match="52 credits remaining"):
        fetch_props("nfl", ev[ev.event_id != "e3"], api_key="k")                 # attrs survive the kickoff filter
    with pytest.raises(RuntimeError, match="51 credits remaining"):
        fetch_props("nfl", ev, api_key="k", remaining=51)                         # an explicit value wins
    with pytest.raises(AssertionError):                                           # plain ids carry no attrs: allowed
        fetch_props("nfl", list(ev.event_id), api_key="k")
    # attrs['remaining'] on the result is the LAST response's x-requests-remaining
    remaining = iter(["100", "96", "92"])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(V4_EVENT, {"x-requests-remaining": next(remaining)}))
    ev.attrs["remaining"] = 500
    df = fetch_props("nfl", ev, api_key="k")
    assert df.attrs["remaining"] == 92 and df[df.book == "fanduel"].attrs["remaining"] == 92
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(V4_EVENT))       # header absent -> None
    assert fetch_props("nfl", ["e1"], api_key="k").attrs["remaining"] is None
    assert fetch_props("nfl", [], api_key="k", remaining=300).attrs["remaining"] is None   # no call -> None


def test_odds_api_errors_never_carry_the_key(monkeypatch):
    """requests' own messages embed the full URL, apiKey included; ours carry only status + body."""
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Fail(401, '{"message":"key ' + KEY + ' is invalid"}'))
    for call in (lambda: fetch_events("nfl", api_key=KEY), lambda: fetch_odds("nfl", api_key=KEY),
                 lambda: fetch_props("nfl", ["e1"], api_key=KEY)):
        with pytest.raises(RuntimeError, match="^Odds API 401 error: ") as ei:
            call()
        msg = str(ei.value)
        assert KEY not in msg and "apiKey" not in msg and "http" not in msg and "***" in msg   # body scrubbed too
        assert isinstance(ei.value, OddsAPIError) and ei.value.status == 401
    # no response at all (timeout / connection error): status 'request', exception class only
    def timeout(url, params=None, timeout=None):
        raise requests.Timeout(f"HTTPSConnectionPool: read timed out for {url}?apiKey={params['apiKey']}")
    monkeypatch.setattr(requests, "get", timeout)
    with pytest.raises(RuntimeError, match="^Odds API request error: Timeout$") as ei:
        fetch_events("nfl", api_key=KEY)
    assert KEY not in str(ei.value) and ei.value.status is None


def test_fetch_events_free_endpoint(monkeypatch, capsys):
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        return _Resp([{"id": "e1", "commence_time": "2026-09-06T17:00:00Z",
                       "home_team": "Philadelphia Eagles", "away_team": "Dallas Cowboys"},
                      {"id": "e2", "commence_time": "2026-09-07T00:20:00Z",
                       "home_team": "Kansas City Chiefs", "away_team": "Los Angeles Chargers"}],
                     {"x-requests-remaining": "480"})
    monkeypatch.setattr(requests, "get", fake_get)
    df = fetch_events("nfl", api_key="k")
    assert calls[0][0] == EVENTS_API.format(sport="americanfootball_nfl") and calls[0][2] == 30
    assert list(df.columns) == ["event_id", "commence", "home", "away"]
    assert list(df.home) == ["PHI", "KC"] and list(df.away) == ["DAL", "LAC"]
    assert str(df.commence.dt.tz) == "UTC" and df.commence.iloc[0].hour == 17      # tz-aware; ns or us resolution (pandas 3)
    assert df.attrs["remaining"] == 480 and df[df.home == "KC"].attrs["remaining"] == 480   # survives filtering
    assert capsys.readouterr().out == ""                                         # free call: no quota chatter
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp([]))
    empty = fetch_events("nfl", api_key="k")
    assert empty.empty and list(empty.columns) == ["event_id", "commence", "home", "away"]
    assert empty.attrs["remaining"] is None                                      # header absent


# ---------------- run.py props mode (subprocess; a dead proxy makes any network attempt fail loudly) ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE = dict(os.environ, HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="", ODDS_API_KEY="")


def _run(args, cwd):
    return subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "nfl", "props", "2026", "1"] + args,
                          cwd=cwd, env=OFFLINE, capture_output=True, text=True, timeout=120)


def test_run_props_validates_markets(tmp_path):
    r = _run(["--markets", "player_pass_yds, bogus_market,", "--csv", os.path.join(ROOT, "props_template.csv")], tmp_path)
    assert r.returncode != 0 and "unknown --markets ['bogus_market']" in r.stderr
    assert "valid keys: " + ", ".join(PROP_MARKETS_ALL) in r.stderr
    assert not list(tmp_path.iterdir())                                          # nothing written, nothing fetched


def test_run_props_csv_nomodel_is_fully_offline(tmp_path):
    """--csv + --nomodel must touch neither the Odds API nor nflverse (no schedule/stats download)."""
    csv = tmp_path / "mine.csv"
    csv.write_text("home,away,book,market,player,side,line,price\n"
                   "PHI,DAL,draftkings,player_pass_yds,Jalen Hurts,Over,274.5,-110\n"
                   "PHI,DAL,draftkings,player_pass_yds,Jalen Hurts,Under,274.5,-110\n"
                   "PHI,DAL,fanduel,player_pass_yds,Jalen Hurts,over,244.5,-110\n"        # 30 yards stale
                   "PHI,DAL,fanduel,player_pass_yds,Jalen Hurts,under,244.5,-110\n")
    r = _run(["--csv", str(csv), "--nomodel", "--markets", " player_pass_yds ,player_receptions", "--weight", "0.5"],
             tmp_path)
    assert r.returncode == 0, r.stderr
    assert "1 games, 2 books, 4 prop prices" in r.stdout and "+EV PROPS (vs devigged market consensus, sorted" in r.stdout
    board = pd.read_csv(tmp_path / "props_nfl_2026_w1.csv")
    assert set(zip(board.book, board.side)) == {("fanduel", "over"), ("draftkings", "under")}
    assert (board["flags"] == "no_proj").all() and (board.ev_pct > 0.03).all()   # market-only: no projections loaded
    assert (board.dist == "normal").all() and (board.fair_sd > 0).all()
    assert (tmp_path / "props_odds_nfl_2026_w1.csv").exists()
