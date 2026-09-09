"""
Alerts: one board pull -> what is NEW since the last run -> Discord / Telegram.

  python alerts.py nfl                            # 3 credits (--books core): fetch, scan, push anything new
  python alerts.py nfl --csv odds.csv --dry-run   # offline: print + write alerts_summary.md, no HTTP, 0 credits
  python alerts.py nfl --test                     # push the current top pick regardless of state (wiring check)
  python alerts.py nfl --min-ev 3 --min-middle-ev 2 --max-age 30 --exclude nonus,betus --top 3 --books wide

Alertable, in this order:
  (a) every pair that cannot lose (guaranteed_pct >= 0: arbs, free middles)      always
  (b) middles / half middles with ev_pct >= --min-middle-ev (percent of the total stake)
  (c) +EV picks (one row per pick at its best book) with EV >= --min-ev (percent): at most --top NEW ones per run
Everything is priced market-only (model weight 0) and books in --exclude are dropped as bets and as legs while
they still anchor the fair numbers, exactly like the ev-scan Action. Rows whose price its book has not touched
in --max-age minutes are dropped first (odds.fresh): the +EV signal *is* a lagging book, but a 90-minute-old
number is usually gone by the time you click. 0 keeps everything; hand CSVs have no timestamps and are kept.

Dedupe: --state (alerts_state.json) maps key -> first_seen (UTC iso). A key already in the state is not
re-sent; keys expire after STATE_TTL_H so a number that survives a day gets one reminder. A pick's key carries
its book AND price ('pick|DAL @ PHI|totals|under|48.5|fanduel|-110'), so a re-price is a new alert; a
middle's key is its game and two legs ('mid|DAL @ PHI|PHI -2.5 -110 @ draftkings|DAL +3.5 -110 @ fanduel').
--top is applied AFTER the dedupe (new_items): five picks alerted this morning do not hold the five slots
against a weaker new one this afternoon; picks past the cap are neither sent nor remembered, so they come
next run. The state is saved after a dry run, after a run with nothing new, and after a live run once at
least one channel got the message; nothing configured, or every channel failed, = nothing remembered, so the
next run re-sends. --test never touches it. No state file = first run = everything is new.

Delivery: DISCORD_WEBHOOK (POST {"content": chunk}), TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (sendMessage);
whichever are set, each tried on its own; neither -> stdout only. Messages are chunked at CHUNK chars
(Discord's limit is 2000). The message is always printed and alerts_summary.md written (the Action appends it
to the job summary). Exit 1 only when there was a message and NO channel took it; a channel that failed while
another succeeded is a warning in the summary (the state is saved: you did get the message).
Secrets never reach a log: requests' own exception text embeds the URL (webhook token / bot token), so every
failure is re-raised as a RuntimeError naming the service and the HTTP status only; the Odds API key is
handled by odds._get the same way.

Credits: the game-line endpoint bills MARKETS x REGIONS per call, and "every group of 10 bookmakers is the
equivalent of 1 region" when the call names its books instead (the-odds-api.com quota docs; measured on the
2026-09-06 ev-scan Action runs: 500 -> 491 -> 474 with the old us,us2,eu default = 9 a pull). alerts.py asks for
the three markets (h2h, spreads, totals) at --books core = odds.CORE_BOOKS, ten books INCLUDING Pinnacle (the
sharp anchor) plus the big US apps and two soft offshore books = 3 credits a run; --books wide (20) = 6. The
cron in .github/workflows/alerts.yml is 16 runs/week x 3 = ~210 credits/month; the math lives in that file's
header. The [alerts] line prints what the pull actually cost (x-requests-last) and 'credits left'.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sharpmodel.odds import (fetch_odds, load_odds_csv, within_hours, add_age, fresh, find_ev, top_picks,
                             NON_US_BOOKS, STALE_MIN, OddsAPIError, odds_credits, book_list)
from sharpmodel.middles import find_middles, game_fairs

# Odds API book keys -> the names on the apps. Same table as app.py's BOOK_NAMES (kept separate on purpose:
# app.py imports streamlit, and this script runs on a bare runner).
BOOK_NAMES = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
    "espnbet": "ESPN BET", "fanatics": "Fanatics", "betrivers": "BetRivers", "hardrockbet": "Hard Rock",
    "hardrockbet_fl": "Hard Rock FL", "hardrockbet_oh": "Hard Rock OH", "betparx": "betPARX", "ballybet": "Bally Bet",
    "fliff": "Fliff", "bovada": "Bovada", "betonlineag": "BetOnline", "betus": "BetUS", "lowvig": "LowVig",
    "mybookieag": "MyBookie", "superbook": "SuperBook", "unibet_us": "Unibet", "twinspires": "TwinSpires",
    "wynnbet": "WynnBET", "pointsbetus": "PointsBet", "circasports": "Circa", "bookmaker": "Bookmaker",
    "betanysports": "BetAnySports", "everygame": "Everygame", "gtbets": "GTBets", "pinnacle": "Pinnacle",
    "marathonbet": "Marathonbet", "matchbook": "Matchbook", "betfair_ex_eu": "Betfair Ex", "betfair_ex_uk": "Betfair Ex UK",
    "unibet_eu": "Unibet EU", "unibet_nl": "Unibet NL", "unibet_se": "Unibet SE", "leovegas_se": "LeoVegas", "pmu_fr": "PMU",
    "coolbet": "Coolbet", "tipico_de": "Tipico", "onexbet": "1xBet", "williamhill": "William Hill",
}
STATE_FILE, SUMMARY_FILE = "alerts_state.json", "alerts_summary.md"
STATE_TTL_H = 24            # a key older than this expires: a line that survives a day gets one reminder
CHUNK = 1900                # Discord caps a message at 2000 chars; Telegram at 4096
TZ = "America/New_York"     # Cleveland time in every timestamp
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
EPS = 1e-9
MARKETS = "h2h,spreads,totals"   # the three game-line markets; the call bills one credit per market per region
N_MARKETS = 3
DEFAULT_BOOKS = "core"           # odds.CORE_BOOKS: 10 books incl. Pinnacle = 1 region-equivalent = 3 credits a run


def credits_per_run(books: str = DEFAULT_BOOKS) -> int:
    """What one board pull bills: 3 markets x ceil(books / 10) ('core' = 3; 'wide' = 6; a 25-book list = 9)."""
    return odds_credits(MARKETS, books=books)


# ---------------- text ----------------
def book_name(b) -> str:
    return BOOK_NAMES.get(str(b), str(b))


def pretty_bet(s) -> str:
    """'PHI -2.5 -110 @ draftkings' -> 'PHI -2.5 -110 @ DraftKings' (middles' bet_a / bet_b)."""
    head, sep, bk = str(s).rpartition(" @ ")
    return f"{head} @ {book_name(bk)}" if sep else str(s)


def pretty_also(s, n: int = 3) -> str:
    """top_picks' 'espnbet +100; betmgm -102' -> 'ESPN BET +100, BetMGM -102' (first n, then '+k more')."""
    parts = [p.strip() for p in str(s if s is not None and not (isinstance(s, float) and np.isnan(s)) else "").split(";")
             if p.strip()]
    out = []
    for p in parts[:n]:
        bk, _, price = p.rpartition(" ")
        out.append(f"{book_name(bk)} {price}" if bk else p)
    return ", ".join(out) + (f" +{len(parts) - n} more" if len(parts) > n else "")


def _am(x) -> str:
    return f"{int(round(float(x))):+d}"


def _num(x) -> str:
    return "" if x is None or pd.isna(x) else f"{float(x):g}"


def _utc(now=None) -> pd.Timestamp:
    t = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def kickoff(commence) -> str:
    """'Sun 1:00PM' in Cleveland time; '' when the row has no kickoff (hand CSV)."""
    if commence is None or (isinstance(commence, float) and np.isnan(commence)) or commence == "": return ""
    t = pd.to_datetime(commence, utc=True, errors="coerce")
    if t is None or pd.isna(t): return ""
    t = t.tz_convert(TZ)
    return f"{t:%a} {t.hour % 12 or 12}:{t:%M%p}"


def header(league: str, now=None) -> str:
    t = _utc(now).tz_convert(TZ)
    return f"🏈 SharpModel · {league.upper()} · {t:%a %b} {t.day} {t.hour % 12 or 12}:{t:%M %p} ET"


def pick_text(r: dict) -> str:
    """'PIT -3.5 +105 @ Hard Rock OH' / 'DEN ML +130 @ Bovada' / 'DAL @ PHI U 48.5 -110 @ FanDuel' (totals name the game)."""
    mk, price = r["market"], _am(r["price"])
    if mk == "spreads":
        line = float(r["line"])
        pick = f"{r['team']} {'PK' if line == 0 else format(line, '+g')} {price}"
    elif mk == "ml":
        pick = f"{r['team']} ML {price}"
    else:
        pick = f"{r['matchup']} {'O' if r['side'] == 'over' else 'U'} {float(r['line']):g} {price}"
    return f"{pick} @ {book_name(r['book'])}"


def pick_line(r: dict) -> str:
    """💰 +EV 6.2% · PIT -3.5 +105 @ Hard Rock OH (fair -107) · also Bovada +100, BetUS +100 · Sun 1:00PM"""
    parts = [f"💰 +EV {100 * float(r['ev_pct']):.1f}%", f"{pick_text(r)} (fair {_am(r['fair_price'])})"]
    also, kick = pretty_also(r.get("also", "")), kickoff(r.get("commence"))
    if also: parts.append(f"also {also}")
    if kick: parts.append(kick)
    return " · ".join(parts)


def middle_line(r: dict) -> str:
    """🔒 ARB +2.44% locked · SEA ML +100 @ FanDuel × LA ML +110 @ DraftKings · split 52/48
       🟢 FREE MIDDLE +7.9% EV · DAL @ PHI · O 44.5 +100 @ FanDuel × U 46.5 +100 @ BetMGM · window 45-46 · hits 8%
       🎯 MIDDLE +3.0% EV · BAL -2.5 -110 @ BetUS × LAC +3.5 -110 @ DraftKings · window 3 · hits 8%, miss costs 4.5%
       ½ HALF MIDDLE +1.2% EV · ... · window 3p · hits 8%, miss costs 2.4%   (on the 'p' number one leg pushes)
    A totals leg ('O 44.5 +100 @ FanDuel') names no team, so totals carry the matchup, like pick_text."""
    legs = f"{pretty_bet(r['bet_a'])} × {pretty_bet(r['bet_b'])}"
    if str(r.get("market")) == "totals": legs = f"{r['matchup']} · {legs}"
    win = f"window {r['window']} · hits {100 * float(r['p_middle']):.0f}%" if str(r.get("window") or "") else ""
    g, ev = float(r["guaranteed_pct"]), float(r["ev_pct"])
    if g > EPS:                                                   # arb / arb+middle: cannot lose, pays either way
        split = f"split {float(r['stake_a_pct']):.0f}/{float(r['stake_b_pct']):.0f}"
        return " · ".join(x for x in (f"🔒 ARB {g:+.2f}% locked", legs, split, win) if x)
    if g >= -EPS:                                                 # free (half) middle: worst case zero
        return " · ".join(x for x in (f"🟢 FREE MIDDLE {ev:+.1f}% EV", legs, win) if x)
    label = "🎯 MIDDLE" if r["type"] == "middle" else "½ HALF MIDDLE"
    return f"{label} {ev:+.1f}% EV · {legs} · {win}, miss costs {float(r['miss_cost_pct']):.1f}%"


def pick_key(r: dict) -> str:
    return "|".join(["pick", str(r["matchup"]), str(r["market"]), str(r["team"]), _num(r.get("line")),
                     str(r["book"]), _am(r["price"])])


def middle_key(r: dict) -> str:
    """The game is part of the key: two games can post the same total legs at the same two books."""
    return f"mid|{r['matchup']}|{r['bet_a']}|{r['bet_b']}"


# ---------------- the scan ----------------
def parse_exclude(s) -> list:
    """'nonus' -> odds.NON_US_BOOKS; 'a,b' -> ['a', 'b']; 'nonus,betus' -> both; '' -> []."""
    out = []
    for b in str(s or "").split(","):
        b = b.strip()
        if b == "nonus": out += NON_US_BOOKS
        elif b: out.append(b)
    return out


def scan(odds: pd.DataFrame, league: str, min_ev: float = 2.0, min_middle_ev: float = 1.0,
         max_age: float = STALE_MIN, exclude=None):
    """Board -> (picks, mids). +EV rows priced market-only (weight 0) at >= min_ev PERCENT, excluded books dropped
    as bets and as legs, stale rows dropped (odds.fresh at max_age minutes; 0 keeps all), one row per pick
    (top_picks, EVERY pick: --top is applied in new_items after the dedupe); middles from find_middles on the same
    fairs. build_items turns them into alerts."""
    excl = list(exclude or [])
    if odds is None or len(odds) == 0 or "event_id" not in odds:           # off-season: the API posts nothing at all
        return top_picks(None), find_middles(None, None)
    if "updated" not in odds: odds = odds.assign(updated=np.nan)           # hand CSV: no timestamps, age NaN
    odds = add_age(odds)                                                    # one clock for the +EV rows and the legs
    ev = find_ev(odds, league, None, 0.0, min_ev=min_ev / 100)
    if excl and len(ev): ev = ev[~ev.book.isin(excl)].reset_index(drop=True)
    ev = fresh(ev, max_age)
    picks = top_picks(ev, max(len(ev), 1))
    mids = find_middles(odds, game_fairs(odds, league, None, 0), league, exclude=excl)
    mids = fresh(mids, max_age)
    return picks, mids


def build_items(picks, mids, min_ev: float = 2.0, min_middle_ev: float = 1.0) -> list:
    """Alertable rows -> [{kind, key, text}]: (a) pairs that cannot lose, (b) middles at >= min_middle_ev %,
    (c) picks at >= min_ev % -- in that order."""
    locks, middles, plays = [], [], []
    if mids is not None and len(mids):
        for r in mids.to_dict("records"):
            if float(r["guaranteed_pct"]) >= -EPS:
                locks.append(dict(kind="lock", key=middle_key(r), text=middle_line(r)))
            elif r["type"] in ("middle", "half_middle") and float(r["ev_pct"]) >= min_middle_ev:
                middles.append(dict(kind="mid", key=middle_key(r), text=middle_line(r)))
    if picks is not None and len(picks):
        for r in picks.to_dict("records"):
            if float(r["ev_pct"]) >= min_ev / 100 - EPS:
                plays.append(dict(kind="pick", key=pick_key(r), text=pick_line(r)))
    return locks + middles + plays


def new_items(items: list, state: dict, top: int) -> list:
    """Items whose key is not in the state, the picks capped at `top` AFTER that check: a pick alerted this morning
    never holds a slot against a new, weaker one this afternoon. Locks and middles are never capped. Picks past
    the cap are not returned, so main() neither sends nor remembers them and they come next run."""
    new = [i for i in items if i["key"] not in state]
    picks = [i for i in new if i["kind"] == "pick"]
    return [i for i in new if i["kind"] != "pick"] + picks[:max(int(top), 0)]


# ---------------- state ----------------
def load_state(path: str, now=None) -> dict:
    """{key: first_seen_iso} with keys older than STATE_TTL_H dropped; {} when the file is missing or unreadable."""
    try:
        with open(path) as f: raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict): return {}
    cutoff = _utc(now) - pd.Timedelta(hours=STATE_TTL_H)
    out = {}
    for k, v in raw.items():
        t = pd.to_datetime(v, utc=True, errors="coerce")
        if t is not None and not pd.isna(t) and t > cutoff: out[k] = v
    return out


def save_state(path: str, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=0, sort_keys=True)


# ---------------- message ----------------
def compose(items: list, league: str, remaining=None, now=None) -> str:
    foot = f"{len(items)} new" + (f" · credits left {remaining}" if remaining is not None else "")
    return "\n".join([header(league, now)] + [i["text"] for i in items] + [foot])


def wiring_message(picks, league: str, min_ev: float, remaining=None, now=None) -> str:
    """--test: the current top pick (or a 'no plays' line) regardless of state."""
    top = pick_line(picks.iloc[0].to_dict()) if picks is not None and len(picks) else \
        f"no plays above +{min_ev:g}% EV right now"
    foot = "test message" + (f" · credits left {remaining}" if remaining is not None else "")
    return "\n".join([header(league, now), top, foot])


def chunks(text: str, limit: int = CHUNK) -> list:
    """Split on newlines into pieces of <= limit chars (a single longer line is cut with an ellipsis)."""
    out, cur = [], ""
    for line in text.split("\n"):
        if len(line) > limit: line = line[:limit - 1] + "…"
        if cur and len(cur) + 1 + len(line) > limit:
            out.append(cur); cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur: out.append(cur)
    return out


# ---------------- delivery ----------------
def _post(name: str, url: str, payload: dict, secrets: list) -> None:
    """POST json; any failure becomes RuntimeError('<name> ...') carrying the status and a scrubbed body -- never the
    URL, which holds the webhook token / bot token (requests puts the URL in its own exception messages)."""
    import requests
    try:
        r = requests.post(url, json=payload, timeout=15)
    except requests.RequestException as e:
        raise RuntimeError(f"{name}: {type(e).__name__} (request failed)") from None
    if not 200 <= r.status_code < 300:
        body = str(r.text or "")[:200]
        for s in secrets:
            if s: body = body.replace(s, "***")
        raise RuntimeError(f"{name} returned HTTP {r.status_code}: {body}")


def send_discord(text: str, webhook: str) -> int:
    parts = chunks(text)
    for c in parts: _post("Discord", webhook, {"content": c}, [webhook])
    return len(parts)


def send_telegram(text: str, token: str, chat_id: str) -> int:
    parts = chunks(text)
    for c in parts:
        _post("Telegram", TELEGRAM_API.format(token=token),
              {"chat_id": chat_id, "text": c, "disable_web_page_preview": True}, [token])
    return len(parts)


def deliver(text: str, env=None) -> tuple:
    """Push to every configured channel, each tried on its own -> (sent, failed): the channel names that got the
    message and the secret-free '<name> ...' error strings of those that did not. ([], []) = nothing configured,
    printed only. A Telegram outage never costs the Discord message, and vice versa."""
    env = os.environ if env is None else env
    channels = []
    if env.get("DISCORD_WEBHOOK"):
        channels.append(("Discord", lambda: send_discord(text, env["DISCORD_WEBHOOK"])))
    if env.get("TELEGRAM_BOT_TOKEN") and env.get("TELEGRAM_CHAT_ID"):
        channels.append(("Telegram", lambda: send_telegram(text, env["TELEGRAM_BOT_TOKEN"], env["TELEGRAM_CHAT_ID"])))
    sent, failed = [], []
    for name, push in channels:
        try:
            push(); sent.append(name)
        except RuntimeError as e:
            failed.append(str(e))
    return sent, failed


# ---------------- summary ----------------
def summary_md(league: str, now, status: str, items=None, error: str = "") -> str:
    """Markdown for the Action's job summary: header, a status line, then one bullet per alert (or the error)."""
    lines = [f"## {header(league, now)}", "", f"**{status}**"]
    if error: lines += ["", f"❌ {error}"]
    elif items: lines += [""] + [f"- {i['text']}" for i in items]
    return "\n".join(lines) + "\n"


def write_summary(text: str, path: str = SUMMARY_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------- CLI ----------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scan the board once and push what is new to Discord / Telegram.")
    p.add_argument("league", choices=["nfl", "cfb"])
    p.add_argument("--csv", help="lines you captured yourself (0 credits); default: fetch_odds (markets x regions credits)")
    p.add_argument("--books", default=DEFAULT_BOOKS,
                   help=f"which books to pull: 'core' (10 incl. Pinnacle = {credits_per_run('core')} credits), 'wide' "
                        f"(20 = {credits_per_run('wide')}), or a comma list of Odds API keys (3 credits per 10 books)")
    p.add_argument("--hours", type=float, default=240, help="games kicking off within this many hours (240 = Tue-Mon)")
    p.add_argument("--min-ev", type=float, default=2.0, help="min EV %% for a pick (default 2.0)")
    p.add_argument("--min-middle-ev", type=float, default=1.0, help="min EV %% of the total stake for a middle (1.0)")
    p.add_argument("--max-age", type=float, default=STALE_MIN,
                   help=f"drop prices a book has not touched in this many minutes (default {STALE_MIN}; 0 = keep all)")
    p.add_argument("--exclude", default="nonus", help="books dropped as bets/legs: 'nonus' or a comma list (nonus)")
    p.add_argument("--state", default=STATE_FILE, help="dedupe file (default alerts_state.json)")
    p.add_argument("--top", type=int, default=5, help="at most this many NEW picks per run, applied after the dedupe (5)")
    p.add_argument("--test", action="store_true", help="send the current top pick regardless of state; no state write")
    p.add_argument("--dry-run", action="store_true", help="no HTTP: print + summary only (state is still saved)")
    p.add_argument("--snapshot", metavar="DIR",
                   help="also write the raw board to DIR/odds_<league>.csv + meta_<league>.json (the dashboard's "
                        "static snapshot: the Action publishes it so viewers never spend a credit)")
    return p.parse_args(argv)


def write_snapshot(odds: pd.DataFrame, league: str, folder: str, now, cost=None, books: str = "") -> str:
    """The board as the dashboard reads it: every row (incl. `updated`), plus meta with the pull time (epoch) and
    the credits picture. Written before the scan so a scan crash never loses a paid-for pull."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"odds_{league}.csv")
    odds.to_csv(path, index=False)
    n_events, n_books = (odds.event_id.nunique(), odds.book.nunique()) if len(odds) else (0, 0)
    meta = dict(league=league, fetched_at=float(_utc(now).timestamp()), fetched_iso=_utc(now).isoformat(),
                remaining=odds.attrs.get("remaining"), cost=cost, n_events=int(n_events), n_books=int(n_books),
                books=books, rows=int(len(odds)))
    with open(os.path.join(folder, f"meta_{league}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return path


def main(argv=None) -> int:
    a = parse_args(argv)
    now = _utc()
    excl = parse_exclude(a.exclude)
    cost = None if a.csv else credits_per_run(a.books)                    # the estimate; attrs['cost'] is the bill
    try:
        odds = load_odds_csv(a.csv) if a.csv else fetch_odds(a.league, markets=MARKETS, books=a.books)
        if not a.csv and odds.attrs.get("cost") is not None: cost = odds.attrs["cost"]
    except (OddsAPIError, RuntimeError, OSError) as e:                 # key-safe messages (odds._get); no key; no file
        msg = f"board fetch failed: {e}"
        print(f"alerts: {msg}")
        write_summary(summary_md(a.league, now, "no scan", error=msg))
        return 1
    remaining = odds.attrs.get("remaining")                            # credits left, from the API headers (None: CSV)
    if a.snapshot:
        print(f"[alerts] snapshot -> {write_snapshot(odds, a.league, a.snapshot, now, cost, a.books)}")
    n_all, n_books = (odds.event_id.nunique(), odds.book.nunique()) if len(odds) else (0, 0)
    odds = within_hours(odds, a.hours)
    picks, mids = scan(odds, a.league, a.min_ev, a.min_middle_ev, a.max_age, excl)
    items = build_items(picks, mids, a.min_ev, a.min_middle_ev)
    n_games = odds.event_id.nunique() if len(odds) else 0
    print(f"[alerts] {n_games} games within {a.hours:.0f}h (of {n_all}), {n_books} books, {len(mids)} arbs/middles on the "
          f"board, {len(picks)} picks >= +{a.min_ev:g}% EV, {len(items)} alertable"
          + (f" · this pull {cost} credits ({N_MARKETS} markets x {len(book_list(a.books) or [])} books '{a.books}')"
             if cost is not None else "")
          + (f", {remaining} credits left" if remaining is not None else ""))

    held = 0
    if a.test:
        text, new, state, known = wiring_message(picks, a.league, a.min_ev, remaining, now), [], {}, 0
    else:
        state = load_state(a.state, now)
        unseen = [i for i in items if i["key"] not in state]
        new = new_items(items, state, a.top)
        known, held = len(items) - len(unseen), len(unseen) - len(new)   # already alerted / picks past --top (next run)
        text = compose(new, a.league, remaining, now) if new else ""
    if text:
        print(text)
        if known: print(f"({known} already alerted within {STATE_TTL_H} h)")
    else:
        print("nothing new" + (f" ({known} already alerted within {STATE_TTL_H} h)" if known else ""))
    if held: print(f"({held} more new picks held for the next run: --top {a.top})")

    sent, failed = [], []
    if text and not a.dry_run:
        sent, failed = deliver(text)
        for f in failed: print(f"alerts: delivery failed: {f}")
        if failed and not sent:                                         # nobody got it: state untouched, next run re-sends
            write_summary(summary_md(a.league, now, f"{len(new)} new · delivery failed", new, error="; ".join(failed)))
            return 1
        if not sent: print("(no DISCORD_WEBHOOK / TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID set: printed only, not remembered)")
    if not a.test and (sent or a.dry_run or not text):                 # remembered only once a channel got it
        for i in new: state[i["key"]] = now.isoformat()
        save_state(a.state, state)

    status = ("test message" if a.test else f"{len(new)} new" if new else "nothing new")
    if sent: status += " · sent to " + " + ".join(sent)
    elif a.dry_run: status += " · dry run (not sent)"
    elif text: status += " · printed only (no webhook configured; not remembered)"
    if failed: status += " · ⚠️ " + "; ".join(failed)
    if known: status += f" · {known} already alerted within {STATE_TTL_H} h"
    if held: status += f" · {held} more picks held for the next run (--top {a.top})"
    if cost is not None: status += f" · {cost} credits this pull"
    if remaining is not None: status += f" · credits left {remaining}"
    write_summary(summary_md(a.league, now, status, [dict(text=text.split("\n")[1])] if a.test else new))
    return 0


if __name__ == "__main__":
    sys.exit(main())
