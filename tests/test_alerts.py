"""Offline tests for alerts.py (Discord / Telegram alerts on new arbs, middles and +EV picks) and alerts.yml."""
import json, os, subprocess, sys
import numpy as np
import pandas as pd
import pytest
import yaml

import alerts
from alerts import (build_items, chunks, compose, deliver, load_state, save_state, middle_line, pick_line, pick_key,
                    middle_key, pretty_bet, pretty_also, kickoff, header, scan, parse_exclude, wiring_message, new_items,
                    credits_per_run, CHUNK, STATE_TTL_H)
from sharpmodel.middles import find_middles
from sharpmodel.odds import NON_US_BOOKS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = pd.Timestamp("2026-09-06T22:24:00Z")
EV = dict(event_id="e", commence="2026-09-13T17:00:00Z", home="PHI", away="DAL")
FAIRS = pd.DataFrame([dict(event_id="e", market="spreads", player=np.nan, mu=3.0, sd=np.nan, dist="nfl_margin"),
                      dict(event_id="e", market="ml", player=np.nan, mu=3.0, sd=np.nan, dist="nfl_margin"),
                      dict(event_id="e", market="totals", player=np.nan, mu=45.5, sd=10.3, dist="normal")])


def _odds(*rows):
    return pd.DataFrame([dict(EV, **r) for r in rows])


def _mids():
    """One of each type off the key-number pmf: ML arb, even-money free middle, -2.5/+3.5 middle, -2.5/+3 half middle."""
    m = find_middles(_odds(dict(book="fanduel", market="ml", side="home", line=np.nan, price=100),
                           dict(book="draftkings", market="ml", side="away", line=np.nan, price=110),
                           dict(book="fanduel", market="totals", side="over", line=44.5, price=100),
                           dict(book="betmgm", market="totals", side="under", line=46.5, price=100),
                           dict(book="betus", market="spreads", side="home", line=-2.5, price=-110),
                           dict(book="draftkings", market="spreads", side="away", line=3.5, price=-110)), FAIRS)
    half = find_middles(_odds(dict(book="betus", market="spreads", side="home", line=-2.5, price=-105),
                              dict(book="draftkings", market="spreads", side="away", line=3.0, price=-105)), FAIRS)
    return pd.concat([m, half], ignore_index=True)


def _pick(**kw):
    r = dict(commence="2026-09-13T17:00:00Z", matchup="ATL @ PIT", market="spreads", side="home", team="PIT", line=-3.5,
             price=105, book="hardrockbet_oh", also="bovada +100; betus +100", fair_price=-107.0, ev_pct=0.062)
    r.update(kw)
    return r


def _board(prices):
    """One game per price: pinnacle -3 -105/-105 anchors it, fanduel posts the away side at +3.5 for `price` -- one
    +EV pick per game (+100 = +3.0%, +105 = +5.6%, ... ranked by price), kicking off tomorrow, no timestamps."""
    rows, kick = [], (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=24)).isoformat()
    for i, p in enumerate(prices):
        ev = dict(event_id=f"e{i}", commence=kick, home=f"H{i}", away=f"A{i}")
        rows += [dict(ev, book="pinnacle", market="spreads", side="home", line=-3.0, price=-105),
                 dict(ev, book="pinnacle", market="spreads", side="away", line=3.0, price=-105),
                 dict(ev, book="fanduel", market="spreads", side="away", line=3.5, price=p)]
    return pd.DataFrame(rows)


def _feed(monkeypatch, *boards):
    """main() reads its board through load_odds_csv: serve these frames in order (any --csv path)."""
    it = iter(boards)
    monkeypatch.setattr(alerts, "load_odds_csv", lambda path: next(it))


# ---------------- formatting ----------------
def test_friendly_names_and_kickoff():
    assert pretty_bet("PHI -2.5 -110 @ draftkings") == "PHI -2.5 -110 @ DraftKings"
    assert pretty_bet("O 44.5 +100 @ some_new_book") == "O 44.5 +100 @ some_new_book"     # unknown key passes through
    assert pretty_also("espnbet +100; betmgm -102") == "ESPN BET +100, BetMGM -102"
    assert pretty_also("a +1; b +2; c +3; d +4; e +5") == "a +1, b +2, c +3 +2 more" and pretty_also("") == ""
    assert pretty_also(np.nan) == "" and pretty_also(None) == ""
    assert kickoff("2026-09-13T17:00:00Z") == "Sun 1:00PM" and kickoff("2026-09-14T00:20:00Z") == "Sun 8:20PM"
    assert kickoff(None) == "" and kickoff(np.nan) == "" and kickoff("garbage") == ""
    assert header("nfl", NOW) == "🏈 SharpModel · NFL · Sun Sep 6 6:24 PM ET"


def test_message_line_for_each_item_type():
    m = _mids()
    assert list(m.type) == ["arb", "free_middle", "middle", "half_middle"]
    lines = [middle_line(r) for r in m.to_dict("records")]
    assert lines[0] == "🔒 ARB +2.44% locked · PHI ML +100 @ FanDuel × DAL ML +110 @ DraftKings · split 51/49"
    assert lines[1].startswith("🟢 FREE MIDDLE +7.")                                   # a totals leg names no team: the game leads
    assert "· DAL @ PHI · O 44.5 +100 @ FanDuel × U 46.5 +100 @ BetMGM · window 45-46 · hits 8%" in lines[1]
    assert lines[1].endswith("hits 8%")                                            # a free middle has no miss cost
    assert lines[2] == "🎯 MIDDLE +3.0% EV · PHI -2.5 -110 @ BetUS × DAL +3.5 -110 @ DraftKings · window 3 · hits 8%, miss costs 4.5%"
    assert lines[3].startswith("½ HALF MIDDLE +1.") and "DAL +3 -105 @ DraftKings · window 3p · hits 8%, miss costs 2.4%" in lines[3]
    assert pick_line(_pick()) == ("💰 +EV 6.2% · PIT -3.5 +105 @ Hard Rock OH (fair -107) · also Bovada +100, BetUS +100 "
                                  "· Sun 1:00PM")
    assert pick_line(_pick(market="ml", team="DEN", line=np.nan, price=130, book="bovada", also="", fair_price=125,
                           ev_pct=0.021, matchup="DEN @ KC", commence=None)) == "💰 +EV 2.1% · DEN ML +130 @ Bovada (fair +125)"
    tot = pick_line(_pick(market="totals", side="under", team="under", line=48.5, price=-110, book="fanduel", also="",
                          fair_price=-117.9, ev_pct=0.033, matchup="DAL @ PHI", commence=np.nan))
    assert tot == "💰 +EV 3.3% · DAL @ PHI U 48.5 -110 @ FanDuel (fair -118)"      # totals name the game
    assert pick_line(_pick(line=0.0, price=-105)).startswith("💰 +EV 6.2% · PIT PK -105 @ Hard Rock OH")
    # an arb that also has a window, and a free half middle, are still 'locked' / 'free' by their guaranteed number
    r = m.iloc[0].to_dict(); r.update(window="3", p_middle=0.08)
    assert middle_line(r).endswith("split 51/49 · window 3 · hits 8%")
    r = m.iloc[3].to_dict(); r.update(guaranteed_pct=0.0)
    assert middle_line(r).startswith("🟢 FREE MIDDLE") and "miss" not in middle_line(r)


def test_items_order_thresholds_and_keys():
    m = _mids()
    picks = pd.DataFrame([_pick(), _pick(ev_pct=0.015, book="fanduel", price=100)])
    items = build_items(picks, m, min_ev=2.0, min_middle_ev=1.0)
    assert [i["kind"] for i in items] == ["lock", "lock", "mid", "mid", "pick"]      # cannot-lose first, then middles, then picks
    assert items[0]["key"] == "mid|DAL @ PHI|PHI ML +100 @ fanduel|DAL ML +110 @ draftkings"   # game + raw book keys
    assert items[-1]["key"] == "pick|ATL @ PIT|spreads|PIT|-3.5|hardrockbet_oh|+105"
    assert [i["kind"] for i in build_items(picks, m, 2.0, 2.0)] == ["lock", "lock", "mid", "pick"]   # half middle +1.6 < 2
    assert [i["kind"] for i in build_items(picks, m, 2.0, 50.0)] == ["lock", "lock", "pick"]         # locks survive any bar
    assert [i["kind"] for i in build_items(picks, m, 1.0, 1.0)][-2:] == ["pick", "pick"]
    assert build_items(None, None) == [] and build_items(pd.DataFrame(), pd.DataFrame()) == []
    assert pick_key(_pick(market="ml", line=np.nan)) == "pick|ATL @ PIT|ml|PIT||hardrockbet_oh|+105"
    assert middle_key(m.iloc[2].to_dict()) == "mid|DAL @ PHI|PHI -2.5 -110 @ betus|DAL +3.5 -110 @ draftkings"
    text = compose(items, "nfl", 471, NOW)
    assert text.startswith("🏈 SharpModel · NFL · Sun Sep 6 6:24 PM ET\n🔒 ARB") and text.endswith("\n5 new · credits left 471")
    assert compose(items[:1], "cfb", None, NOW).endswith("\n1 new")
    assert wiring_message(picks, "nfl", 2.0, 471, NOW).split("\n") == [header("nfl", NOW), pick_line(_pick()),
                                                                     "test message · credits left 471"]
    assert wiring_message(pd.DataFrame(), "nfl", 2.5, None, NOW).split("\n")[1:] == ["no plays above +2.5% EV right now",
                                                                                   "test message"]


def test_totals_middles_name_the_game_and_dedupe_per_game():
    """'O 44.5 +100 @ FanDuel x U 46.5 +100 @ BetMGM' says nothing about which game: the matchup is in the line and in
    the key, so two games posting the same numbers at the same two books are two alerts, not one."""
    free = _mids().iloc[1].to_dict()
    assert middle_key(free) == "mid|DAL @ PHI|O 44.5 +100 @ fanduel|U 46.5 +100 @ betmgm"
    assert middle_line(free).split(" · ")[1:3] == ["DAL @ PHI", "O 44.5 +100 @ FanDuel × U 46.5 +100 @ BetMGM"]
    tot = [dict(book="fanduel", market="totals", side="over", line=44.5, price=100),
           dict(book="betmgm", market="totals", side="under", line=46.5, price=100)]
    kc = dict(event_id="e2", commence="2026-09-13T20:25:00Z", home="KC", away="DEN")
    odds = pd.concat([_odds(*tot), pd.DataFrame([dict(kc, **r) for r in tot])], ignore_index=True)
    fairs = pd.concat([FAIRS, pd.DataFrame([dict(event_id="e2", market="totals", player=np.nan, mu=45.5, sd=10.3,
                                                 dist="normal")])], ignore_index=True)
    mids = find_middles(odds, fairs)
    assert len(mids) == 2 and set(mids.matchup) == {"DAL @ PHI", "DEN @ KC"}
    items = build_items(None, mids)
    assert len({i["key"] for i in items}) == 2 and sorted(i["text"].split(" · ")[1] for i in items) == ["DAL @ PHI", "DEN @ KC"]
    assert new_items(items, {items[0]["key"]: NOW.isoformat()}, 5) == [items[1]]         # the first game alerted: the second still goes


# ---------------- dedupe ----------------
def test_dedupe_new_keys_changed_price_and_expiry(tmp_path):
    path = str(tmp_path / "state.json")
    assert load_state(path) == {}                                                   # no file: everything is new
    items = build_items(pd.DataFrame([_pick()]), _mids())
    state = load_state(path, NOW)
    new = [i for i in items if i["key"] not in state]
    assert len(new) == len(items) == 5
    for i in new: state[i["key"]] = NOW.isoformat()
    save_state(path, state)
    state2 = load_state(path, NOW + pd.Timedelta(hours=1))                          # second run: nothing new
    assert [i for i in items if i["key"] not in state2] == []
    repriced = build_items(pd.DataFrame([_pick(price=110)]), _mids())              # a re-price is a new key
    assert [i["key"] for i in repriced if i["key"] not in state2] == ["pick|ATL @ PIT|spreads|PIT|-3.5|hardrockbet_oh|+110"]
    assert len(load_state(path, NOW + pd.Timedelta(hours=STATE_TTL_H - 1))) == 5
    assert load_state(path, NOW + pd.Timedelta(hours=STATE_TTL_H + 1)) == {}       # expired: alert again
    (tmp_path / "bad.json").write_text("not json")
    assert load_state(str(tmp_path / "bad.json")) == {}
    (tmp_path / "list.json").write_text("[1, 2]")
    assert load_state(str(tmp_path / "list.json")) == {}
    (tmp_path / "junk.json").write_text(json.dumps({"k": "not a date", "ok": NOW.isoformat()}))
    assert list(load_state(str(tmp_path / "junk.json"), NOW)) == ["ok"]


def test_top_cap_applies_after_the_dedupe(tmp_path, monkeypatch, capsys):
    """Five picks alerted this morning keep sitting on the board; a sixth, weaker one appears: it is the alert.
    (--top used to cut the board to five BEFORE the state was consulted, so rank 6 was never sent.)"""
    picks = [dict(kind="pick", key=f"k{i}", text=f"p{i}") for i in range(6)]
    state = {f"k{i}": NOW.isoformat() for i in range(5)}
    assert new_items(picks, state, 5) == [picks[5]] and new_items(picks, state, 1) == [picks[5]]
    lock, mid = dict(kind="lock", key="L", text="l"), dict(kind="mid", key="M", text="m")
    assert new_items([lock, mid] + picks, {}, 2) == [lock, mid, picks[0], picks[1]]   # locks / middles never count
    assert new_items([lock] + picks, {"L": NOW.isoformat()}, 0) == []
    # end to end through main(): run 1 alerts the five strongest, run 2 sees them plus a new +3.0% game
    monkeypatch.chdir(tmp_path)
    _feed(monkeypatch, _board([125, 120, 115, 110, 105]), _board([125, 120, 115, 110, 105, 100]))
    args = ["nfl", "--csv", "x.csv", "--dry-run", "--state", "s.json", "--top", "5"]
    assert alerts.main(args) == 0
    out = capsys.readouterr().out
    assert out.count("💰") == 5 and "A0 +3.5 +125 @ FanDuel" in out and "5 new" in out
    assert alerts.main(args) == 0
    out = capsys.readouterr().out
    assert out.count("💰") == 1 and "A5 +3.5 +100 @ FanDuel" in out and "5 already alerted" in out
    assert len(json.loads((tmp_path / "s.json").read_text())) == 6
    # a tighter cap: the picks past it are neither sent nor remembered, and the summary says so
    _feed(monkeypatch, _board([125, 120, 115, 110, 105]), _board([125, 120, 115, 110, 105]))
    args = ["nfl", "--csv", "x.csv", "--dry-run", "--state", "s2.json", "--top", "2"]
    assert alerts.main(args) == 0
    out = capsys.readouterr().out
    assert out.count("💰") == 2 and "A0 +3.5 +125" in out and "A1 +3.5 +120" in out and "3 more new picks held" in out
    assert len(json.loads((tmp_path / "s2.json").read_text())) == 2
    assert "3 more picks held for the next run (--top 2)" in (tmp_path / "alerts_summary.md").read_text()
    assert alerts.main(args) == 0
    out = capsys.readouterr().out
    assert out.count("💰") == 2 and "A2 +3.5 +115" in out and "A3 +3.5 +110" in out and "1 more new picks held" in out


# ---------------- chunking ----------------
def test_chunks_respect_discord_limit():
    lines = [f"line {i} " + "x" * 120 for i in range(60)]
    text = "\n".join(lines)
    parts = chunks(text)
    assert all(len(p) <= CHUNK for p in parts) and len(parts) > 1
    assert "\n".join(parts).split("\n") == lines                                   # nothing lost, no line split
    assert chunks("short") == ["short"] and chunks("") == []
    long = "y" * 2500
    assert chunks(long) == ["y" * (CHUNK - 1) + "…"]                                # a single oversize line is cut
    assert chunks("a\n" + long)[0] == "a"


# ---------------- delivery ----------------
class _Resp:
    def __init__(self, status, text=""): self.status_code, self.text = status, text


def test_discord_and_telegram_payloads_and_secret_hygiene(monkeypatch):
    import requests
    calls = []
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: (calls.append((url, json, timeout)) or _Resp(204)))
    text = "\n".join(["🏈 header"] + ["z" * 150] * 20)                              # > 1900 chars -> 2 chunks
    hook = "https://discord.com/api/webhooks/123/SECRET-TOKEN"
    env = {"DISCORD_WEBHOOK": hook, "TELEGRAM_BOT_TOKEN": "bot-secret", "TELEGRAM_CHAT_ID": "42"}
    assert deliver(text, env) == (["Discord", "Telegram"], [])
    assert len(calls) == 4 and all(t == 15 for _, _, t in calls)
    d = [c for c in calls if c[0] == hook]
    assert len(d) == 2 and list(d[0][1]) == ["content"] and "\n".join(p["content"] for _, p, _ in d) == text
    t = [c for c in calls if c[0] != hook]
    assert t[0][0] == "https://api.telegram.org/botbot-secret/sendMessage"
    assert t[0][1] == {"chat_id": "42", "text": d[0][1]["content"], "disable_web_page_preview": True}
    assert deliver(text, {}) == ([], []) and deliver(text, {"DISCORD_WEBHOOK": "", "TELEGRAM_BOT_TOKEN": "x"}) == ([], [])
    assert len(calls) == 4                                                          # nothing configured -> no HTTP
    # a non-2xx is reported without the URL / token in the message (even when the body echoes them)
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: _Resp(429, f"rate limited at {url}"))
    sent, failed = deliver("x", {"DISCORD_WEBHOOK": hook})
    assert sent == [] and len(failed) == 1 and "Discord returned HTTP 429" in failed[0]
    assert "SECRET-TOKEN" not in failed[0] and hook not in failed[0]
    sent, failed = deliver("x", {"TELEGRAM_BOT_TOKEN": "bot-secret", "TELEGRAM_CHAT_ID": "42"})
    assert sent == [] and "Telegram returned HTTP 429" in failed[0] and "bot-secret" not in failed[0]

    def boom(url, json=None, timeout=None):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")
    monkeypatch.setattr(requests, "post", boom)
    assert deliver("x", {"DISCORD_WEBHOOK": hook}) == ([], ["Discord: ConnectionError (request failed)"])
    # one channel down does not cost the other: Discord gets it, Telegram's failure is reported alongside
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: _Resp(204) if url == hook else _Resp(500, "down"))
    assert deliver("x", env) == (["Discord"], ["Telegram returned HTTP 500: down"])
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: _Resp(500, "down") if url == hook else _Resp(200))
    assert deliver("x", env) == (["Telegram"], ["Discord returned HTTP 500: down"])


def test_partial_delivery_remembers_and_exits_0_all_failed_exits_1(tmp_path, monkeypatch, capsys):
    """Discord got the message and Telegram did not: exit 0, state saved (no re-send next run), a warning in the
    summary, no token in any output. Both down: exit 1, nothing remembered."""
    import requests
    monkeypatch.chdir(tmp_path)
    hook = "https://discord.com/api/webhooks/1/HOOK-SECRET"
    for k, v in dict(DISCORD_WEBHOOK=hook, TELEGRAM_BOT_TOKEN="tg-secret-xyz", TELEGRAM_CHAT_ID="7").items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None:
                        _Resp(204) if url == hook else _Resp(500, f"nope {url}"))
    _feed(monkeypatch, _board([110]), _board([110]))
    args = ["nfl", "--csv", "x.csv", "--state", "s.json"]
    assert alerts.main(args) == 0
    out = capsys.readouterr().out
    assert "A0 +3.5 +110 @ FanDuel" in out and "delivery failed: Telegram returned HTTP 500" in out
    assert "secret-xyz" not in out and "HOOK-SECRET" not in out
    assert list(json.loads((tmp_path / "s.json").read_text())) == ["pick|A0 @ H0|spreads|A0|3.5|fanduel|+110"]
    md = (tmp_path / "alerts_summary.md").read_text()
    assert "**1 new · sent to Discord · ⚠️ Telegram returned HTTP 500" in md and "- 💰 +EV 8.1%" in md and "secret-xyz" not in md
    assert alerts.main(args) == 0 and "nothing new (1 already alerted" in capsys.readouterr().out
    # every channel down: exit 1, the state file never appears, the error (scrubbed) in the summary
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: _Resp(500, f"nope {url}"))
    _feed(monkeypatch, _board([110]))
    assert alerts.main(["nfl", "--csv", "x.csv", "--state", "s2.json"]) == 1
    out = capsys.readouterr().out
    assert "delivery failed: Discord returned HTTP 500" in out and "delivery failed: Telegram returned HTTP 500" in out
    assert not (tmp_path / "s2.json").exists()
    md = (tmp_path / "alerts_summary.md").read_text()
    assert "**1 new · delivery failed**" in md and "❌ Discord returned HTTP 500" in md and "HOOK-SECRET" not in md


# ---------------- freshness + exclusions ----------------
def test_stale_rows_dropped_at_max_age_kept_at_zero():
    now = pd.Timestamp.now(tz="UTC")
    old, new = (now - pd.Timedelta(minutes=90)).isoformat(), (now - pd.Timedelta(minutes=5)).isoformat()
    rows = [dict(book="pinnacle", market="spreads", side="home", line=-3.0, price=-105, updated=new),
            dict(book="pinnacle", market="spreads", side="away", line=3.0, price=-105, updated=new),
            dict(book="fanduel", market="spreads", side="away", line=3.5, price=100, updated=old),       # stale +EV
            dict(book="betus", market="spreads", side="home", line=-2.5, price=-110, updated=old)]      # stale leg
    odds = _odds(*rows)
    picks, mids = scan(odds, "nfl", min_ev=1.0, min_middle_ev=0.0, max_age=45, exclude=NON_US_BOOKS)
    assert picks.empty and mids.empty
    picks, mids = scan(odds, "nfl", min_ev=1.0, min_middle_ev=0.0, max_age=0, exclude=NON_US_BOOKS)
    assert list(picks.book) == ["fanduel"] and picks.iloc[0].age_min == pytest.approx(90, abs=1)
    assert len(mids) == 1 and mids.iloc[0].age_min == pytest.approx(90, abs=1) and mids.iloc[0].window == "3"
    picks, mids = scan(odds, "nfl", min_ev=1.0, max_age=0, exclude=[])                # pinnacle allowed as a leg / bet
    assert "pinnacle" in set(mids.book_a) | set(mids.book_b) or len(mids) >= 1
    for empty in (None, pd.DataFrame(), odds.iloc[0:0]):                              # off-season / nothing inside --hours
        picks, mids = scan(empty, "nfl")
        assert picks.empty and mids.empty and build_items(picks, mids) == []
    picks, _ = scan(_board([125, 120, 115, 110, 105, 100]), "nfl", exclude=NON_US_BOOKS)
    assert len(picks) == 6 and list(picks["rank"]) == [1, 2, 3, 4, 5, 6]               # every pick: the cap is main's
    assert parse_exclude("nonus") == NON_US_BOOKS and parse_exclude("nonus,betus") == NON_US_BOOKS + ["betus"]
    assert parse_exclude(" a , b ") == ["a", "b"] and parse_exclude("") == [] and parse_exclude(None) == []


def test_credits_per_run_is_markets_times_book_groups():
    assert credits_per_run("core") == 3 and credits_per_run("wide") == 6 and alerts.DEFAULT_BOOKS == "core"
    assert credits_per_run("pinnacle,fanduel") == 3 and credits_per_run(",".join(f"b{i}" for i in range(11))) == 6
    assert credits_per_run("") == 9                                                  # no books -> the regions default
    assert alerts.MARKETS.count(",") + 1 == alerts.N_MARKETS == 3


# ---------------- the CLI (subprocess; a dead proxy makes any network attempt fail loudly) ----------------
OFFLINE = dict(os.environ, HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="",
               ODDS_API_KEY="", DISCORD_WEBHOOK="", TELEGRAM_BOT_TOKEN="", TELEGRAM_CHAT_ID="")


def _run(args, cwd, env=OFFLINE):
    return subprocess.run([sys.executable, os.path.join(ROOT, "alerts.py")] + args, cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=120)


def test_dry_run_on_the_template_writes_summary_and_state(tmp_path):
    csv = os.path.join(ROOT, "lines_template.csv")
    r = _run(["nfl", "--csv", csv, "--dry-run"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "🏈 SharpModel · NFL ·" in r.stdout and "💰 +EV 3.3% · DAL @ PHI U 48.5 -110 @ FanDuel (fair -118)" in r.stdout
    assert r.stdout.rstrip().endswith("1 new") and "@ pinnacle" not in r.stdout and "Pinnacle" not in r.stdout
    assert "2 books" in r.stdout and "credits" not in r.stdout                        # a CSV bills nothing: no cost line
    state = json.loads((tmp_path / "alerts_state.json").read_text())
    assert list(state) == ["pick|DAL @ PHI|totals|under|48.5|fanduel|-110"]
    md = (tmp_path / "alerts_summary.md").read_text()
    assert md.startswith("## 🏈 SharpModel · NFL") and "**1 new · dry run (not sent)**" in md and "- 💰 +EV 3.3%" in md
    r = _run(["nfl", "--csv", csv, "--dry-run"], tmp_path)                          # second run: dedupe
    assert r.returncode == 0 and "nothing new (1 already alerted within 24 h)" in r.stdout and "💰" not in r.stdout
    assert "**nothing new · dry run (not sent) · 1 already alerted" in (tmp_path / "alerts_summary.md").read_text()
    # no exclusions + middle bar 0: the template's pinnacle / fanduel total middle joins the message; own state file
    r = _run(["nfl", "--csv", csv, "--dry-run", "--exclude", "", "--min-middle-ev", "0", "--state", "s2.json"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "🎯 MIDDLE +0.5% EV · DAL @ PHI · O 47.5 -104 @ Pinnacle × U 48.5 -110 @ FanDuel · window 48" in r.stdout
    assert r.stdout.rstrip().endswith("2 new") and len(json.loads((tmp_path / "s2.json").read_text())) == 2
    assert "mid|DAL @ PHI|O 47.5 -104 @ pinnacle|U 48.5 -110 @ fanduel" in json.loads((tmp_path / "s2.json").read_text())
    # nothing configured and not a dry run: printed only, exit 0, and NOT remembered (a run before the webhook
    # exists must not eat the 24 h window) -- so the next run prints the same pick again
    r = _run(["nfl", "--csv", csv, "--state", "s3.json"], tmp_path)
    assert r.returncode == 0 and "printed only, not remembered" in r.stdout and not (tmp_path / "s3.json").exists()
    assert "printed only (no webhook configured; not remembered)" in (tmp_path / "alerts_summary.md").read_text()
    r = _run(["nfl", "--csv", csv, "--state", "s3.json"], tmp_path)
    assert r.returncode == 0 and "1 new" in r.stdout and not (tmp_path / "s3.json").exists()


def test_test_flag_sends_top_pick_and_never_touches_state(tmp_path):
    csv = os.path.join(ROOT, "lines_template.csv")
    r = _run(["nfl", "--csv", csv, "--dry-run", "--test"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "💰 +EV 3.3% · DAL @ PHI U 48.5 -110 @ FanDuel (fair -118)" in r.stdout and "test message" in r.stdout
    assert not (tmp_path / "alerts_state.json").exists()
    assert "**test message · dry run (not sent)**" in (tmp_path / "alerts_summary.md").read_text()
    r = _run(["nfl", "--csv", csv, "--dry-run", "--test", "--min-ev", "50"], tmp_path)
    assert r.returncode == 0 and "no plays above +50% EV right now" in r.stdout
    # a delivery failure exits 1 and leaves the state untouched (the next run re-sends); no URL in the output
    r = _run(["nfl", "--csv", csv, "--state", "s.json"], tmp_path,
             env=dict(OFFLINE, DISCORD_WEBHOOK="https://discord.com/api/webhooks/1/SECRET"))
    assert r.returncode == 1 and "delivery failed: Discord:" in r.stdout and "SECRET" not in r.stdout + r.stderr
    assert not (tmp_path / "s.json").exists() and "delivery failed" in (tmp_path / "alerts_summary.md").read_text()
    # no key and no csv: clean failure, key-safe, summary written, nothing billed
    r = _run(["nfl", "--state", "s.json"], tmp_path)
    assert r.returncode == 1 and "board fetch failed" in r.stdout and "Traceback" not in r.stderr
    assert "no scan" in (tmp_path / "alerts_summary.md").read_text()
    assert "--books" in _run(["--help"], tmp_path).stdout                          # the cost knob is on the CLI


# ---------------- the workflow ----------------
def _expand(field, lo, hi):
    """Cron field -> set of ints ('*', 'a', 'a-b', 'a,b', 'a-b/2' not needed here)."""
    if field == "*": return set(range(lo, hi + 1))
    out = set()
    for part in field.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def test_alerts_workflow_parses_and_schedule_is_12_runs_a_week():
    with open(os.path.join(ROOT, ".github", "workflows", "alerts.yml"), encoding="utf-8") as f:
        raw = f.read()
    wf = yaml.safe_load(raw)
    on = wf.get("on") or wf.get(True)                                               # PyYAML reads the key `on` as True
    assert wf["name"] == "alerts" and wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"] == {"group": "alerts", "cancel-in-progress": False}    # overlapping runs queue, never double-send
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["league"]["default"] == "nfl" and inputs["test"]["default"] == "false"
    assert inputs["min_ev"]["default"] == "2.0" and inputs["hours"]["default"] == "240"
    assert inputs["books"]["default"] == alerts.DEFAULT_BOOKS == "core"           # 3 credits a run, not odds.py's 9
    crons = [c["cron"] for c in on["schedule"]]
    runs = 0
    for c in crons:
        f = c.split()
        assert len(f) == 5, c
        assert "23,0" not in f[1] and "," not in f[4], f"{c}: a window across midnight must be two lines"
        minute, hour, dom, mon, dow = f
        assert minute == "0" and dom == "*" and mon == "*"
        hours, days = _expand(hour, 0, 23), _expand(dow, 0, 6)
        assert max(hours) <= 23 and max(days) <= 6
        runs += len(hours) * len(days)
    assert runs == 12                                                               # 9+1 Sun/SNF + 1 TNF + 1 MNF pregame
    assert runs * credits_per_run("core") * 52 / 12 < 220                             # ~210 credits/month of the free 500
    assert "0 0 * * 1" in crons and "0 13-21 * * 0" in crons                         # SNF wraps into Monday UTC
    steps = wf["jobs"]["alerts"]["steps"]
    uses = [s.get("uses", "") for s in steps]
    assert any(u.startswith("actions/cache/restore@v4") for u in uses) and any(u.startswith("actions/cache/save@v4") for u in uses)
    save = next(s for s in steps if s.get("uses", "").startswith("actions/cache/save"))
    assert save["if"] == "always()" and save["with"]["path"] == "alerts_state.json"
    scan_step = next(s for s in steps if "alerts.py" in s.get("run", ""))
    assert set(scan_step["env"]) >= {"ODDS_API_KEY", "DISCORD_WEBHOOK", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "BOOKS"}
    assert "--test" in scan_step["run"] and "inputs.test" in raw and '--books "$BOOKS"' in scan_step["run"]
    # a scheduled run with no channel to deliver to stops before the pull (0 credits); manual runs always pull
    guard = scan_step["run"].split("python alerts.py")[0]
    assert '"$GITHUB_EVENT_NAME" = "schedule"' in guard and "DISCORD_WEBHOOK" in guard and "exit 0" in guard
    assert "GITHUB_STEP_SUMMARY" in raw and "git push" not in raw and "git commit" not in raw
    assert "3 credits" in raw and "12 runs/week" in raw and "markets x regions" in raw.lower()   # the credit math stays documented
    assert "1 credit" not in raw.replace("1 credit per market", "")                   # the old "one credit per run" claim is gone
