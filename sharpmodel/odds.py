"""
Multi-book odds -> fair prices -> +EV lines.

Pipeline
  1. Pull every book's spread / total / moneyline (The Odds API, or CSV).
  2. Devig every sharp book's two-way price (SHARP_BOOKS) and invert each into a fair margin/total (mu).
     e.g. Pinnacle PHI -6.5 (-108 / -102) -> P(PHI covers)=0.513 -> the mu whose P(margin > 6.5) is 0.513.
     NFL spreads and moneylines invert through the empirical key-number pmf (margins.py: a -3 pushes ~9% of
     the time, not the Normal's 3%); totals and CFB through the Normal. margin_dist() holds that choice.
  3. Average the sharp mus, weighted by SHARP_WEIGHTS x freshness (exp(-age_min / 60)): a stale Pinnacle
     number yields to a live BetOnline one. Median over every book when no sharp book posts the market.
  4. Optionally blend mu with your model number.
  5. Price EVERY book's line (including off-market alternate numbers) off that mu, on the same distribution.
  6. Rank by EV, size with fractional Kelly, flag arbs.

Why invert to mu instead of comparing prices directly: books post different
numbers (-6.5 vs -7 vs -7.5). You can only compare a -7 (+100) against a
-6.5 (-108) once both are probabilities from the same distribution.

Player props live on a per-event endpoint that bills (markets x regions) per call, so
fetch_events (free) -> estimate_prop_credits -> fetch_props (never automatic) -> long df
with a `player` column. Pricing them is props.py's job; this file only ingests.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from .pricing import (LEAGUE_SD, devig, american_to_prob, decimal_from_american,
                      cover_probs, total_probs, moneyline_prob, edge_and_kelly)
from .margins import (NFL_SD, NFL_KEY_WEIGHTS, cover_probs_emp, moneyline_prob_emp,
                      implied_margin_emp, implied_margin_ml_emp)

ODDS_API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
EVENTS_API = "https://api.the-odds-api.com/v4/sports/{sport}/events"                 # free (0 credits)
PROPS_API = "https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"  # markets x regions credits
SPORT_KEY = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf"}
SHARP_BOOKS = ["pinnacle", "circasports", "betonlineag", "bookmaker", "lowvig"]
# Prior weight of each sharp book in the consensus fair; multiplied by freshness = exp(-age_min / FRESH_TAU_MIN),
# so a Pinnacle number an hour old counts 3.0 x 0.37 = 1.1 against a live BetOnline's 1.5. Unknown age (CSV) = 1.0.
SHARP_WEIGHTS = {"pinnacle": 3.0, "circasports": 2.0, "betonlineag": 1.5, "bookmaker": 1.0, "lowvig": 1.0}
FRESH_TAU_MIN = 120.0     # e-fold of a sharp book's weight per two hours BEHIND the freshest sharp book on that market
FRESH_FLOOR = 0.2         # a confident, unchanged Pinnacle never drops below a fifth of its prior weight
# Books a US bettor cannot get down at (EU/UK/AU region keys, exchanges, Pinnacle). They still anchor the fair
# number; `--exclude nonus` keeps them out of the +EV rows and the arb/middle legs.
NON_US_BOOKS = ["pinnacle", "marathonbet", "matchbook", "smarkets", "betfair_ex_eu", "betfair_ex_uk", "betfair_ex_au",
                "unibet_eu", "unibet_nl", "unibet_se", "unibet_uk", "unibet", "leovegas", "leovegas_se", "pmu_fr",
                "coolbet", "tipico_de", "onexbet", "williamhill", "betsson", "nordicbet", "suprabets", "winamax_fr",
                "winamax_de", "betclic_fr", "parionssport_fr", "sport888", "mrgreen", "paddypower", "skybet",
                "ladbrokes_uk", "ladbrokes_au", "coral", "betvictor", "boylesports", "grosvenor", "virginbet",
                "livescorebet", "casumo", "betway", "sportsbet", "tab", "neds", "playup", "pointsbetau", "betr_au",
                "bluebet", "topsport", "gtbets", "everygame"]


STALE_MIN = 45   # a price its book has not touched in this long, while the sharp books moved, is probably already gone


def add_age(odds: pd.DataFrame, now=None) -> pd.DataFrame:
    """age_min: minutes since the book last changed this price (the API's per-market last_update, kept as `updated`).
    NaN for hand-captured CSVs. The +EV signal *is* a book lagging the sharps, so an old price is both the
    opportunity and the warning that it may not be there when you click -- see find_ev's lag_min."""
    o = odds.copy()
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None: now = now.tz_localize("UTC")
    upd = (pd.to_datetime(o["updated"], utc=True, errors="coerce") if "updated" in o
           else pd.Series(pd.NaT, index=o.index, dtype="datetime64[ns, UTC]"))   # tz-aware NaT: naive - aware raises
    o["age_min"] = (now - upd).dt.total_seconds() / 60
    return o


def within_hours(odds: pd.DataFrame, hours: float) -> pd.DataFrame:
    """Lines whose game kicks off within `hours` from now. The API posts the whole season (272 NFL games on the
    first live pull) and far-future numbers are stale by nature; 240 h covers a Tue-Mon slate. Rows without a
    kickoff (a hand-captured CSV) are kept."""
    if "commence" not in odds or not odds.commence.notna().any(): return odds
    c, now = pd.to_datetime(odds.commence, utc=True, errors="coerce"), pd.Timestamp.now(tz="UTC")
    return odds[c.isna() | ((c > now) & (c <= now + pd.Timedelta(hours=hours)))]

# Odds API player-prop market keys. Also exist (out of v1 scope): player_pass_attempts,
# player_pass_completions, player_rush_attempts and *_alternate variants -- extend here.
PROP_MARKETS_DEFAULT = ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions"]
PROP_MARKETS_ALL = PROP_MARKETS_DEFAULT + ["player_pass_tds", "player_anytime_td"]
PROP_SIDES = {"over": "over", "under": "under", "yes": "yes", "no": "no"}
PROP_COLS = ["event_id", "commence", "home", "away", "book", "market", "player", "side", "line", "price", "updated"]

NFL_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def normalize_team(name: str, league: str, known: list[str] | None = None) -> str:
    if league == "nfl":
        return NFL_NAMES.get(name, name)
    # CFB: Odds API says "Alabama Crimson Tide", CFBD says "Alabama" -> longest known prefix
    if known:
        hits = [k for k in known if name.startswith(k)]
        if hits: return max(hits, key=len)
    return name


# ---------------- ingestion ----------------
class OddsAPIError(RuntimeError):
    """A failed Odds API call. `status` is the HTTP status (None for timeouts / connection errors).
    The message carries the status and the response body only -- never the URL, which holds the key."""
    def __init__(self, status, body: str = ""):
        self.status = status
        super().__init__(f"Odds API {status or 'request'} error: {body}")


def _key(api_key: str | None) -> str:
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Set ODDS_API_KEY (free tier at the-odds-api.com)")
    return key


def _get(url: str, params: dict, timeout: int = 30):
    """requests.get + raise_for_status. Any requests failure is re-raised as OddsAPIError so the key
    (a query param, hence in every requests exception message) can never reach a log or the UI."""
    import requests
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        resp = getattr(e, "response", None)
        body = str(getattr(resp, "text", "") or "")[:200] if resp is not None else type(e).__name__
        key = params.get("apiKey")
        if key: body = body.replace(key, "***")
        raise OddsAPIError(getattr(resp, "status_code", None), body) from None


def _remaining(r) -> int | None:
    rem = r.headers.get("x-requests-remaining")
    try:
        return int(float(rem)) if rem is not None else None
    except (TypeError, ValueError):
        return None


# The game-line endpoint bills MARKETS x REGIONS per call, and "every group of 10 bookmakers is the equivalent of 1
# region" when `bookmakers` is given instead of `regions` (the-odds-api.com v4 quota docs; measured 2026-09-06: the
# old us,us2,eu default cost 9 a pull). So a hand-picked 10-book list costs 3 credits AND keeps Pinnacle, the sharp
# anchor, which regions=us would drop. CORE = the anchor + the big US apps + two soft offshore books; WIDE adds the
# next ten (6 credits). Pass a comma list or a list of keys for anything else.
CORE_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics", "betrivers",
              "bovada", "betonlineag"]
WIDE_BOOKS = CORE_BOOKS + ["hardrockbet", "betus", "lowvig", "mybookieag", "betanysports", "superbook", "betparx",
                           "ballybet", "fliff", "unibet_us"]
BOOK_SETS = {"core": CORE_BOOKS, "wide": WIDE_BOOKS}


def book_list(books) -> list | None:
    """'core' / 'wide' / 'a,b,c' / a list -> list of Odds API book keys; None or '' -> None (use regions)."""
    if books is None: return None
    if isinstance(books, str):
        s = books.strip()
        if not s: return None
        if s in BOOK_SETS: return list(BOOK_SETS[s])
        return [b.strip() for b in s.split(",") if b.strip()]
    return list(books)


def odds_credits(markets="h2h,spreads,totals", regions="us,us2,eu", books="core") -> int:
    """What one fetch_odds call bills: markets x (ceil(len(books) / 10) if books else regions)."""
    n_m = len([m for m in str(markets).split(",") if m.strip()])
    bl = book_list(books)
    if bl: return n_m * ((len(bl) + 9) // 10)
    return n_m * len([r for r in str(regions).split(",") if r.strip()])


def fetch_odds(league: str, api_key: str | None = None, regions="us,us2,eu",
               markets="h2h,spreads,totals", known_teams: list[str] | None = None, books="core") -> pd.DataFrame:
    """Long format: event_id, commence, home, away, book, market, side, line, price. books='core' (default, 3 credits:
    CORE_BOOKS incl. Pinnacle) / 'wide' (6) / a comma list or list of keys / None -> `regions` (us,us2,eu = 9).
    df.attrs: remaining (credits left this month), cost (this call's x-requests-last), both ints or None."""
    params = {"apiKey": _key(api_key), "markets": markets, "oddsFormat": "american"}
    bl = book_list(books)
    if bl: params["bookmakers"] = ",".join(bl)
    else: params["regions"] = regions
    r = _get(ODDS_API.format(sport=SPORT_KEY[league]), params)
    print(f"[odds] this pull cost {r.headers.get('x-requests-last')} credits; "
          f"{r.headers.get('x-requests-remaining')} remaining this month")
    df = parse_odds_json(r.json(), league, known_teams)
    df.attrs["remaining"] = _remaining(r)                       # an int, never a frame (pandas 3 compares attrs)
    df.attrs["cost"] = _header_int(r, "x-requests-last")
    return df


def _header_int(r, name) -> int | None:
    try: return int(float(r.headers.get(name)))
    except (TypeError, ValueError): return None


def parse_odds_json(events: list, league: str, known_teams=None) -> pd.DataFrame:
    rows = []
    for ev in events:
        home = normalize_team(ev["home_team"], league, known_teams)
        away = normalize_team(ev["away_team"], league, known_teams)
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    nm = o["name"]
                    if mk["key"] == "totals":
                        side = nm.lower()                      # over / under
                    else:
                        side = "home" if normalize_team(nm, league, known_teams) == home else "away"
                    rows.append(dict(event_id=ev["id"], commence=ev["commence_time"], home=home, away=away,
                                     book=bk["key"], market={"h2h": "ml"}.get(mk["key"], mk["key"]),
                                     side=side, line=o.get("point", np.nan), price=o["price"],
                                     updated=mk.get("last_update")))
    return pd.DataFrame(rows)


def load_odds_csv(path: str) -> pd.DataFrame:
    """Same long schema, for lines you capture by hand (or export from an odds screen).
    Required columns: home, away, book, market(spreads|totals|ml), side(home|away|over|under), line, price"""
    df = pd.read_csv(path)
    if "event_id" not in df: df["event_id"] = df.home + "@" + df.away
    return df


# ---------------- player props ----------------
def fetch_events(league: str, api_key: str | None = None, known_teams: list[str] | None = None) -> pd.DataFrame:
    """Upcoming events from the FREE endpoint (0 credits): event_id, commence (UTC), home, away.
    df.attrs['remaining'] = x-requests-remaining (None if the header is absent) -- pass it to fetch_props."""
    r = _get(EVENTS_API.format(sport=SPORT_KEY[league]), {"apiKey": _key(api_key)})
    rows = [dict(event_id=ev["id"], commence=ev["commence_time"],
                 home=normalize_team(ev["home_team"], league, known_teams),
                 away=normalize_team(ev["away_team"], league, known_teams)) for ev in r.json()]
    df = pd.DataFrame(rows, columns=["event_id", "commence", "home", "away"])
    df["commence"] = pd.to_datetime(df.commence, utc=True)
    df.attrs["remaining"] = _remaining(r)
    return df


def estimate_prop_credits(n_events: int, markets, regions: str = "us") -> int:
    """Upper bound on credits for fetch_props: each per-event call bills (markets returned x regions)."""
    if isinstance(markets, str): markets = markets.split(",")
    return int(n_events) * len(markets) * len(regions.split(","))


STOP_STATUSES = (401, 402, 429)     # bad key / out of credits / rate-limited: retrying the next event is pointless
MAX_TRANSPORT_FAILURES = 2          # consecutive timeouts / connection errors (status None) before giving up


def fetch_props(league: str, events, markets=PROP_MARKETS_DEFAULT, regions: str = "us",
                api_key: str | None = None, max_credits: int = 60, reserve: int = 50,
                known_teams: list[str] | None = None, remaining: int | None = None) -> pd.DataFrame:
    """
    Player props for the given events, one per-event call each. `events` is the fetch_events frame
    (its home/away label the log lines; its attrs['remaining'] is the default `remaining`) or a plain
    iterable of event ids. Long format: PROP_COLS.
    Quota-aware: refuses up front if the estimate exceeds max_credits or if `remaining` minus one call's
    cost would already breach `reserve`, and stops early once x-requests-remaining drops below `reserve`.
    A failed per-event call is skipped (printed as '[props] skip away@home: status') so the events
    already billed are kept; 401/402/429 stop the loop instead, and so do MAX_TRANSPORT_FAILURES
    consecutive timeouts / connection errors. A 200 whose body is not the documented shape is skipped
    too ('bad response body'). Skipped and never-attempted ids are returned in df.attrs['failed'];
    df.attrs['remaining'] is x-requests-remaining from the last response (None if absent / no call).
    If no event succeeded at all the last error is raised (nothing usable was billed).
    """
    key = _key(api_key)
    if isinstance(events, pd.DataFrame):
        labels = {r.event_id: f"{r.away}@{r.home}" for r in events.itertuples()}
        event_ids = list(events.event_id)
        if remaining is None: remaining = events.attrs.get("remaining")
    else:
        labels, event_ids = {}, list(events)
    markets = list(markets)
    n_reg = len(regions.split(","))
    per_call = len(markets) * n_reg
    cost = estimate_prop_credits(len(event_ids), markets, regions)
    if cost > max_credits:
        raise RuntimeError(
            f"[props] {len(event_ids)} events x {len(markets)} markets x {n_reg} region(s) = {cost} credits "
            f"> max_credits={max_credits}. Drop to <= {max_credits // per_call} events, "
            f"or <= {max_credits // (len(event_ids) * n_reg)} markets, or raise max_credits.")
    if remaining is not None and event_ids and remaining - per_call < reserve:
        raise RuntimeError(f"[props] {remaining} credits remaining - {per_call} per call < reserve={reserve}; "
                           "not scanning (wait for the monthly reset or lower reserve)")
    frames, failed, last_err, rem, n_transport = [], [], None, None, 0
    for i, eid in enumerate(event_ids):
        try:
            r = _get(PROPS_API.format(sport=SPORT_KEY[league], event_id=eid),
                     {"apiKey": key, "regions": regions, "markets": ",".join(markets), "oddsFormat": "american"})
            ev = r.json()
            frame = parse_props_json(ev, league, known_teams)
        except OddsAPIError as e:
            failed.append(eid); last_err = e
            print(f"[props] skip {labels.get(eid, eid)}: {e.status or 'request error'}")
            n_transport = n_transport + 1 if e.status is None else 0
            if e.status in STOP_STATUSES or n_transport >= MAX_TRANSPORT_FAILURES:
                print(f"[props] stopping: {e}" if e.status in STOP_STATUSES
                      else f"[props] stopping: {n_transport} consecutive request errors")
                failed += event_ids[i + 1:]                       # never attempted, so not on the board either
                break
            continue
        except (ValueError, KeyError, TypeError) as e:            # a 200 whose body is not the documented shape
            failed.append(eid); last_err = OddsAPIError(None, f"bad response body ({type(e).__name__})")
            print(f"[props] skip {labels.get(eid, eid)}: bad response body")
            continue
        n_transport = 0
        frames.append(frame)
        last, rem = r.headers.get("x-requests-last"), _remaining(r)
        label = labels.get(eid) or (f"{normalize_team(ev['away_team'], league, known_teams)}@"
                                    f"{normalize_team(ev['home_team'], league, known_teams)}")
        print(f"[props] {label}: cost {last}, remaining {rem}")
        if rem is not None and rem < reserve:
            print(f"[props] WARNING: {rem} credits left < reserve={reserve}; stopping after "
                  f"{len(frames)}/{len(event_ids)} events")
            break
    if not frames and last_err is not None:
        raise last_err
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PROP_COLS)
    df.attrs["failed"] = failed
    df.attrs["remaining"] = rem
    return df


def parse_props_json(event: dict, league: str, known_teams=None) -> pd.DataFrame:
    """One per-event response -> long rows (PROP_COLS). Yes/No markets (anytime TD) carry no line."""
    home = normalize_team(event["home_team"], league, known_teams)
    away = normalize_team(event["away_team"], league, known_teams)
    rows = []
    for bk in event.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            for o in mk.get("outcomes") or []:
                player, side = o.get("description"), PROP_SIDES.get(str(o.get("name", "")).lower())
                if not player or side is None: continue
                line = o.get("point")
                rows.append(dict(event_id=event["id"], commence=event.get("commence_time"), home=home, away=away,
                                 book=bk["key"], market=mk["key"], player=player, side=side,
                                 line=np.nan if line is None else line,          # explicit "point": null -> NaN
                                 price=o["price"], updated=mk.get("last_update")))
    return pd.DataFrame(rows, columns=PROP_COLS)


def load_props_csv(path: str) -> pd.DataFrame:
    """Same long schema, for props you capture by hand (see props_template.csv).
    Required columns: home, away, book, market, player, side(over|under|yes|no), line, price.
    Sides are stripped and lower-cased; anything outside PROP_SIDES raises ValueError."""
    df = pd.read_csv(path)
    need = ["home", "away", "book", "market", "player", "side", "line", "price"]
    missing = [c for c in need if c not in df]
    if missing: raise ValueError(f"props csv missing columns: {missing}")
    df["side"] = df.side.astype(str).str.strip().str.lower()
    unknown = sorted(set(df.side) - set(PROP_SIDES))
    if unknown: raise ValueError(f"props csv: unknown side(s) {unknown}; allowed: {'/'.join(PROP_SIDES)}")
    if "event_id" not in df: df["event_id"] = df.home + "@" + df.away
    for c in ("commence", "updated"):
        if c not in df: df[c] = np.nan
    return df[PROP_COLS]


# ---------------- fair pricing ----------------
def margin_dist(league: str) -> dict:
    """The margin distribution behind every spread and moneyline price, {kind, sd, weights}: 'nfl_margin' for the
    NFL (margins.margin_pmf at NFL_SD with the key-number weights: a -3 pushes ~9% of the time, not the Normal's
    3%), 'normal' at LEAGUE_SD for CFB. sharp_fair inverts prices with it and find_ev prices with it, so the two
    can never disagree; middles.game_fairs names the same dist. Totals are Normal at LEAGUE_SD everywhere."""
    if league == "nfl":
        return {"kind": "nfl_margin", "sd": NFL_SD, "weights": NFL_KEY_WEIGHTS}
    return {"kind": "normal", "sd": LEAGUE_SD[league]["spread"], "weights": None}


def _cover(dist: dict, mu: float, home_line: float) -> dict:
    """{win, push, loss} for the HOME side of `home_line` at fair mu on `dist`."""
    if dist["kind"] == "nfl_margin": return cover_probs_emp(mu, home_line, dist["sd"], dist["weights"])
    return cover_probs(mu, home_line, dist["sd"])


def _ml_prob(dist: dict, mu: float) -> float:
    """P(home wins outright) at fair mu on `dist`."""
    if dist["kind"] == "nfl_margin": return moneyline_prob_emp(mu, dist["sd"], dist["weights"])
    return moneyline_prob(mu, dist["sd"])


def _mu_from_spread(dist: dict, home_line: float, q_home: float) -> float:
    """Devigged P(home covers `home_line`) -> the fair mu on `dist` (home covers if margin > -line)."""
    if dist["kind"] == "nfl_margin": return implied_margin_emp(home_line, q_home, dist["sd"], dist["weights"])
    return -home_line + dist["sd"] * norm.ppf(q_home)


def _mu_from_ml(dist: dict, q_home: float) -> float:
    """Devigged P(home wins) -> the fair mu on `dist`."""
    if dist["kind"] == "nfl_margin": return implied_margin_ml_emp(q_home, dist["sd"], dist["weights"])
    return dist["sd"] * norm.ppf(q_home)


def _freshness(age_min, newest=0.0) -> float:
    """Weight multiplier for a sharp book's price by how far BEHIND the freshest sharp book on the same market it
    is (rel = age - newest): max(exp(-rel / FRESH_TAU_MIN), FRESH_FLOOR). The API's last_update advances when a
    book changes its price, so an absolute age cannot tell 'stale' from 'confident and unchanged' -- only lagging
    the peers can; the freshest book always carries its full prior weight. 1.0 when the age is unknown (a CSV)."""
    if age_min is None or pd.isna(age_min): return 1.0
    base = 0.0 if newest is None or pd.isna(newest) else float(newest)
    rel = max(float(age_min) - base, 0.0)
    return float(max(np.exp(-rel / FRESH_TAU_MIN), FRESH_FLOOR))


def _two_way(df: pd.DataFrame, a: str, b: str, market: str):
    """Matched (line, price_a, price_b, age_min) for a two-way market at one book, or None. age_min is the older
    side's (the API stamps both sides of a market together), NaN when neither row carries one."""
    da, db = df[df.side == a], df[df.side == b]
    for _, ra in da.iterrows():
        if market == "ml": m = db
        elif market == "spreads": m = db[np.isclose(db.line.astype(float), -float(ra.line))]
        else: m = db[np.isclose(db.line.astype(float), float(ra.line))]
        if len(m):
            rb = m.iloc[0]
            ages = [x for x in (ra.get("age_min", np.nan), rb.get("age_min", np.nan)) if x is not None and pd.notna(x)]
            return ra.line, ra.price, rb.price, (max(ages) if ages else np.nan)
    return None


def _implied(book_odds: pd.DataFrame, dist: dict, sd_t: float) -> dict:
    """One book's devigged prices -> {market: (mu, age_min)} for whichever of spreads / ml / totals it posts two-way."""
    out = {}
    ok = lambda *xs: all(x is not None and np.isfinite(float(x)) for x in xs)   # a blank price cannot anchor anything
    sp = _two_way(book_odds[book_odds.market == "spreads"], "home", "away", "spreads")
    if sp and ok(sp[0], sp[1], sp[2]):
        line, ph, pa, age = sp
        out["spreads"] = (_mu_from_spread(dist, float(line), devig(ph, pa)[0]), age)
    ml = _two_way(book_odds[book_odds.market == "ml"], "home", "away", "ml")
    if ml and ok(ml[1], ml[2]):
        _, ph, pa, age = ml
        out["ml"] = (_mu_from_ml(dist, devig(ph, pa)[0]), age)
    tt = _two_way(book_odds[book_odds.market == "totals"], "over", "under", "totals")
    if tt and ok(tt[0], tt[1], tt[2]):
        line, po, pu, age = tt
        out["totals"] = (float(line) + sd_t * norm.ppf(devig(po, pu)[0]), age)
    return {m: v for m, v in out.items() if np.isfinite(v[0])}


MARKETS = ("spreads", "ml", "totals")


def sharp_fair(event_odds: pd.DataFrame, league: str) -> dict:
    """
    Fair mu_margin / mu_ml / mu_total for one event from the sharp books, each market on its own.
    Every SHARP_BOOKS book posting a market is devigged and inverted on margin_dist(league) (spreads, moneylines) or
    the Normal (totals), then averaged with weight SHARP_WEIGHTS[book] x _freshness(age, newest) -- a book's weight
    decays by how far its last price change lags the freshest sharp book on that market (e-fold FRESH_TAU_MIN,
    floor FRESH_FLOOR; 1.0 when ages are unknown); age_min is add_age'd here if the frame lacks it. A Pinnacle
    spread untouched for six hours next to a BetOnline one changed two minutes ago lands mostly on BetOnline
    (3 x 0.2 vs 1.5 x 1); two equally fresh land 2:1 toward Pinnacle; a Pinnacle that is the freshest keeps full weight
    however old, because 'unchanged' is not 'stale'.
    A book that posts only a spread contributes only to mu_margin. A market no sharp book posts falls back to the
    median over every book; a margin with no spread anywhere comes from the moneylines and vice versa.
    ref_book: the heaviest contributor to the spread (to the ML, then the total, when nobody posts a spread);
    ref_n: distinct sharp books used (0 on the all-book fallback); ref_age: ref_book's age_min on that market.
    """
    dist, sd_t = margin_dist(league), LEAGUE_SD[league]["total"]
    out = {"ref_book": None, "mu_margin": np.nan, "mu_total": np.nan, "mu_ml": np.nan, "ref_n": 0, "ref_age": np.nan}
    if event_odds is None or len(event_odds) == 0: return out
    e = event_odds if "age_min" in event_odds else add_age(event_odds)
    present = set(e.book)
    contrib = {m: [] for m in MARKETS}                         # market -> [(mu, prior weight, book, age_min)]
    for bk in [b for b in SHARP_BOOKS if b in present]:
        for m, (mu, age) in _implied(e[e.book == bk], dist, sd_t).items():
            contrib[m].append((mu, SHARP_WEIGHTS.get(bk, 1.0), bk, age))
    out["ref_n"] = len({c[2] for m in MARKETS for c in contrib[m]})
    everyone = None                                             # every book's implied mus, built only for a fallback
    mu, ref = {}, {}
    for m in MARKETS:
        c = contrib[m]
        if not c:                                               # no sharp book posts it: median over all books, as before
            if everyone is None:
                everyone = {bk: _implied(e[e.book == bk], dist, sd_t) for bk in sorted(present)}
            c = [(v[m][0], 1.0, bk, v[m][1]) for bk, v in everyone.items() if m in v]
            if not c: continue
            mu[m] = float(np.median([x[0] for x in c]))
            ref[m] = c[0]                                       # first book key posting it, as the old fallback did
        else:
            ages = [x[3] for x in c if x[3] is not None and pd.notna(x[3])]
            newest = min(ages) if ages else 0.0                    # the freshest sharp book on THIS market
            w = np.array([x[1] * _freshness(x[3], newest) for x in c]); mus = np.array([x[0] for x in c])
            mu[m] = float((w * mus).sum() / w.sum())
            ref[m] = c[int(np.argmax(w))]
    out["mu_margin"] = mu.get("spreads", mu.get("ml", np.nan))
    out["mu_ml"] = mu.get("ml", out["mu_margin"])
    out["mu_total"] = mu.get("totals", np.nan)
    for m in MARKETS:
        if m in ref:
            out["ref_book"], out["ref_age"] = ref[m][2], ref[m][3]
            break
    return out


def blended_fair(event_odds: pd.DataFrame, league: str, model_fair: pd.DataFrame | None = None,
                 model_weight: float = 0.25) -> dict:
    """sharp_fair for one event, nudged toward the model: mu = (1-w)*sharp + w*model for the margin and
    the total, the ML mu shifted by the same margin nudge. model_fair: df with home, away, model_margin,
    model_total (or that df already indexed by [home, away]); None or a missing game -> pure sharp fair.
    ref_book / ref_n / ref_age pass through. find_ev and middles.game_fairs both price off exactly this dict."""
    f = sharp_fair(event_odds, league)
    if model_fair is None: return f
    mf = model_fair if isinstance(model_fair.index, pd.MultiIndex) else model_fair.set_index(["home", "away"])
    home, away = event_odds.home.iloc[0], event_odds.away.iloc[0]
    if (home, away) not in mf.index: return f
    mm, mt = mf.loc[(home, away), ["model_margin", "model_total"]]
    out = dict(f)
    if np.isfinite(f["mu_margin"]) and np.isfinite(mm):
        # mu is the distribution's LOCATION parameter, which under the key-number pmf sits above the number for a
        # favourite (a -7 coin flip inverts to ~8.3). The model's number is an expected margin, so convert it into the
        # same space first -- "the model thinks -mm is a coin flip" -- or a model that agrees with the market would
        # still tilt every favourite's fair toward the dog.
        mm_loc = _mu_from_spread(margin_dist(league), -float(mm), 0.5)
        out["mu_margin"] = (1 - model_weight) * f["mu_margin"] + model_weight * mm_loc
        out["mu_ml"] = f["mu_ml"] + (out["mu_margin"] - f["mu_margin"])     # shift ML mu by the same model nudge
    if np.isfinite(f["mu_total"]) and np.isfinite(mt):
        out["mu_total"] = (1 - model_weight) * f["mu_total"] + model_weight * mt
    return out


def find_ev(odds: pd.DataFrame, league: str, model_fair: pd.DataFrame | None = None,
            model_weight: float = 0.25, min_ev: float = 0.02, kelly_fraction: float = 0.25,
            max_stake: float = 0.03) -> pd.DataFrame:
    """
    Price every posted line against the fair mu and return +EV plays.
    model_fair: optional df with home, away, model_margin, model_total from SharpModel;
                blended into the sharp mu with weight `model_weight` (blended_fair).
    Spreads and moneylines are priced on margin_dist(league) -- the same pmf sharp_fair inverted, so a sharp book's
    own number is never +EV against itself; an integer line carries the pmf's push mass. Totals are Normal. A
    moneyline tie (~0.3% at a pick'em) is folded into the away side as before. Rows carry ref_book / ref_n and
    lag_min = age_min - the reference book's age on its anchoring market (> 0: this book is behind the sharps).
    """
    dist, sd_t = margin_dist(league), LEAGUE_SD[league]["total"]
    mf = model_fair.set_index(["home", "away"]) if model_fair is not None else None
    if len(odds) and "age_min" not in odds: odds = add_age(odds)
    rows = []
    for eid, e in odds.groupby("event_id"):
        home, away = e.home.iloc[0], e.away.iloc[0]
        f = blended_fair(e, league, mf, model_weight)
        mu_m, mu_t, mu_ml, ref_age = f["mu_margin"], f["mu_total"], f["mu_ml"], f["ref_age"]
        for _, o in e.iterrows():
            if o.market == "spreads" and np.isfinite(mu_m):
                home_line = o.line if o.side == "home" else -o.line
                cp = _cover(dist, mu_m, home_line)
                p_win = cp["win"] if o.side == "home" else cp["loss"]; p_push = cp["push"]
            elif o.market == "totals" and np.isfinite(mu_t):
                tp = total_probs(mu_t, o.line, sd_t)
                p_win, p_push = tp[o.side], tp["push"]
            elif o.market == "ml" and np.isfinite(mu_ml):
                ph = _ml_prob(dist, mu_ml)
                p_win, p_push = (ph if o.side == "home" else 1 - ph), 0.0
            else:
                continue
            ek = edge_and_kelly(p_win, p_push, o.price, kelly_fraction, max_stake)
            rows.append(dict(commence=o.get("commence"), matchup=f"{away} @ {home}", market=o.market,
                             side=o.side, team=(home if o.side == "home" else away if o.side == "away" else o.side),
                             line=o.line, price=o.price, book=o.book, ref_book=f["ref_book"], ref_n=f["ref_n"],
                             fair_mu={"spreads": mu_m, "totals": mu_t, "ml": mu_ml}[o.market],
                             p_win=p_win, p_push=p_push, fair_price=_fair_american(p_win, p_push),
                             ev_pct=ek["ev"], kelly_stake=ek["stake_frac"],
                             updated=o.get("updated"), age_min=o.get("age_min", np.nan),
                             lag_min=o.get("age_min", np.nan) - ref_age))    # > 0: this book is behind the sharp book
    res = pd.DataFrame(rows)
    if res.empty: return res
    return (res[res.ev_pct >= min_ev].sort_values("ev_pct", ascending=False).reset_index(drop=True))


def fresh(df: pd.DataFrame, max_age: float = STALE_MIN) -> pd.DataFrame:
    """Rows whose price was touched within max_age minutes (rows without a timestamp are kept); 0 = keep everything."""
    if not max_age or df is None or len(df) == 0 or "age_min" not in df: return df
    return df[df.age_min.isna() | (df.age_min <= max_age)].reset_index(drop=True)


def _fair_american(p_win, p_push):
    p = p_win / (1 - p_push) if p_push < 1 else 0.5
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def best_lines(odds: pd.DataFrame) -> pd.DataFrame:
    """Best available price per matchup / market / side / line (the line-shopping board).
    Rows without a price (a blank in a hand-captured CSV) are dropped: idxmax cannot pick one."""
    o = odds.reset_index(drop=True); o["dec"] = pd.to_numeric(o.price, errors="coerce").map(decimal_from_american)
    o = o[o.dec.notna()]
    idx = o.groupby(["event_id", "market", "side", "line"], dropna=False).dec.idxmax()
    return o.loc[idx, ["home", "away", "market", "side", "line", "price", "book"]].reset_index(drop=True)


# ---------------- top picks ----------------
PICK_COLS = ["rank", "commence", "matchup", "market", "player", "side", "team", "line", "price", "book", "also",
             "n_books", "fair_price", "p_win", "ev_pct", "kelly_stake", "ref_book", "ref_n", "flags", "updated",
             "age_min", "lag_min"]


def _am(p) -> str:
    return f"{int(round(float(p))):+d}"


def top_picks(ev: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    The +EV board deduped to ONE row per pick -- (matchup, market[, player], side, line) -- at its best price, ranked
    by EV: 'CHI -3 +100' is one pick even when five books post it. Works on the props board too (player column).
    `also` lists the other books on the same number, best price first ('betus +100; betmgm -102'), n_books counts
    them all. Empty in -> empty out (PICK_COLS, only the columns present).
    """
    if ev is None or len(ev) == 0: return pd.DataFrame(columns=PICK_COLS)
    e = ev.copy()
    e["_dec"] = pd.to_numeric(e.price, errors="coerce").map(decimal_from_american)
    e = e.sort_values(["ev_pct", "_dec"], ascending=[False, False], kind="stable")
    keys = ["matchup", "market"] + (["player"] if "player" in e else []) + ["side", "line"]
    rows = []
    for _, g in e.groupby(keys, dropna=False, sort=False):
        d = g.iloc[0].to_dict()
        d["n_books"] = len(g)
        d["also"] = "; ".join(f"{r.book} {_am(r.price)}" for r in g.iloc[1:].itertuples())
        rows.append(d)
    out = pd.DataFrame(rows).sort_values("ev_pct", ascending=False, kind="stable").head(n).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[[c for c in PICK_COLS if c in out]]


# ---------------- odds screen ----------------
US_BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics", "betrivers", "hardrockbet",
            "hardrockbet_fl", "hardrockbet_oh", "betparx", "ballybet", "fliff", "bovada", "betonlineag", "betus",
            "lowvig", "mybookieag", "superbook", "unibet_us", "twinspires", "wynnbet", "pointsbetus", "circasports",
            "bookmaker"]                                              # column order on the screen; the rest alphabetical
MARKET_ORDER = ["spreads", "totals", "ml"]


def _cell(market, line, price) -> str:
    p = f"{int(round(float(price))):+d}"
    if market == "ml" or pd.isna(line): return p
    if market == "spreads": return f"{'PK' if line == 0 else format(line, '+g')} {p}"
    return f"{line:g} {p}"                                             # totals / props: over-under is the row


def odds_grid(odds: pd.DataFrame, exclude=None, masks: bool = False):
    """
    Odds screen: one row per matchup x market x pick (x player for props), one column per book, cell = 'line price'
    ('-2.5 -110', '47.5 -105', '+130'). Per (row, book) the entry nearest the row's modal number is kept (best price
    on ties) so alternate lines do not duplicate rows. Books in US_BOOKS order first, then the rest alphabetically;
    trailing columns best ('-105 @ pinnacle': best price among books AT the modal number) and books (count).
    masks=True also returns two boolean frames (same rows, book columns) for highlighting: the best price at the
    number, and a book posting a DIFFERENT number (look in Arbs & middles). They are returned, not stored in
    DataFrame.attrs: pandas 3 compares attrs on every concat and a DataFrame in there is an ambiguous truth value.
    Empty in -> empty out.
    """
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()) if masks else pd.DataFrame()
    if odds is None or len(odds) == 0 or "book" not in odds: return empty
    o = odds.copy()
    if exclude is not None and len(exclude): o = o[~o.book.isin(set(exclude))]
    o["line"] = pd.to_numeric(o["line"], errors="coerce") if "line" in o else np.nan
    o["price"] = pd.to_numeric(o["price"], errors="coerce")
    o = o[o.price.notna()]
    if o.empty: return empty
    o["commence"] = o["commence"].fillna("").astype(str) if "commence" in o else ""
    o["matchup"] = o.away.astype(str) + " @ " + o.home.astype(str)
    o["pick"] = np.where(o.market.isin(["spreads", "ml"]), np.where(o.side == "home", o.home, o.away), o.side)
    keys = ["commence", "matchup", "market"] + (["player"] if "player" in o else []) + ["pick"]
    if "player" in o: o["player"] = o["player"].fillna("").astype(str)
    o["_dec"] = o.price.map(decimal_from_american)
    # a book's main number is the line it prices nearest -110 (alternates sit away from it); the market's number is the
    # most common main line, ties -> nearest the mean of the main lines, then the lower one
    o["_off110"] = (o._dec - decimal_from_american(-110)).abs()
    main = o.sort_values("_off110").drop_duplicates(keys + ["book"])
    def _modal(s):
        s = s.dropna()
        if s.empty: return np.nan
        m = s.mode()
        return m.iloc[int((m - s.mean()).abs().argmin())] if len(m) > 1 else m.iloc[0]
    modal = main.groupby(keys).line.agg(_modal)
    o = o.merge(modal.rename("_modal"), left_on=keys, right_index=True)
    o["_gap"] = (o.line - o._modal).abs().fillna(0.0)
    o = o.sort_values(["_gap", "_dec"], ascending=[True, False]).drop_duplicates(keys + ["book"])
    o["_mo"] = o.market.map({m: i for i, m in enumerate(MARKET_ORDER)}).fillna(len(MARKET_ORDER))
    o["_po"] = np.where(o.side.isin(["home", "over", "yes"]), 0, 1)
    o = o.sort_values(["commence", "matchup", "_mo", "market"] + (["player"] if "player" in o else []) + ["_po"],
                      kind="stable")
    books = [b for b in US_BOOKS if b in set(o.book)] + sorted(set(o.book) - set(US_BOOKS))
    txt, best, off = [], [], []
    for key, g in o.groupby(keys, sort=False):
        d = dict(zip(keys, key)); mk = d["market"]
        at = g[g._gap == 0]
        b = at.loc[at._dec.idxmax()] if len(at) else None
        d.update({bk: "" for bk in books})
        for r in g.itertuples(): d[r.book] = _cell(mk, r.line, r.price)
        d["best"] = f"{int(round(float(b.price))):+d} @ {b.book}" if b is not None else ""
        d["books"] = len(g)
        txt.append(d)
        best.append({bk: bool(b is not None and bk == b.book) for bk in books})
        offb = set(g[g._gap > 0].book)
        off.append({bk: bk in offb for bk in books})
    grid = pd.DataFrame(txt, columns=keys + books + ["best", "books"])
    if masks: return grid, pd.DataFrame(best, columns=books), pd.DataFrame(off, columns=books)
    return grid


def find_arbs(odds: pd.DataFrame) -> pd.DataFrame:
    """Two-way arbs across books at the SAME number (spread/total) or moneyline. Kept for compatibility:
    middles.find_middles supersedes it (different numbers too, middles priced off the fair pmf, stakes)."""
    out = []
    for (eid, mk), e in odds.groupby(["event_id", "market"]):
        a, b = ("over", "under") if mk == "totals" else ("home", "away")
        A, B = e[e.side == a], e[e.side == b]
        for _, ra in A.iterrows():
            match_line = -ra.line if mk == "spreads" else ra.line
            cand = B if mk == "ml" else B[np.isclose(B.line, match_line)]
            if cand.empty: continue
            rb = cand.loc[cand.price.apply(decimal_from_american).idxmax()]
            hold = american_to_prob(ra.price) + american_to_prob(rb.price)
            if hold < 1.0:
                out.append(dict(matchup=f"{ra.away} @ {ra.home}", market=mk, line_a=ra.line, book_a=ra.book,
                                price_a=ra.price, book_b=rb.book, price_b=rb.price, profit_pct=1 / hold - 1))
    return pd.DataFrame(out).sort_values("profit_pct", ascending=False) if out else pd.DataFrame()
