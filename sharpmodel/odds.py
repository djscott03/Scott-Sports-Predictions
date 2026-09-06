"""
Multi-book odds -> fair prices -> +EV lines.

Pipeline
  1. Pull every book's spread / total / moneyline (The Odds API, or CSV).
  2. Pick a sharp reference (Pinnacle > Circa > BetOnline > Bookmaker > consensus).
  3. Devig the sharp two-way price and invert it into a fair margin/total (mu).
     e.g. Pinnacle PHI -6.5 (-108 / -102) -> P(PHI covers)=0.513 -> mu_PHI = 6.5 + 13.4*z(0.513)
  4. Optionally blend mu with your model number.
  5. Price EVERY book's line (including off-market alternate numbers) off that mu.
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
                      cover_probs, total_probs, edge_and_kelly)

ODDS_API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
EVENTS_API = "https://api.the-odds-api.com/v4/sports/{sport}/events"                 # free (0 credits)
PROPS_API = "https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"  # markets x regions credits
SPORT_KEY = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf"}
SHARP_BOOKS = ["pinnacle", "circasports", "betonlineag", "bookmaker", "lowvig"]
# Books a US bettor cannot get down at (EU/UK/AU region keys, exchanges, Pinnacle). They still anchor the fair
# number; `--exclude nonus` keeps them out of the +EV rows and the arb/middle legs.
NON_US_BOOKS = ["pinnacle", "marathonbet", "matchbook", "smarkets", "betfair_ex_eu", "betfair_ex_uk", "betfair_ex_au",
                "unibet_eu", "unibet_nl", "unibet_se", "unibet_uk", "unibet", "leovegas", "leovegas_se", "pmu_fr",
                "coolbet", "tipico_de", "onexbet", "williamhill", "betsson", "nordicbet", "suprabets", "winamax_fr",
                "winamax_de", "betclic_fr", "parionssport_fr", "sport888", "mrgreen", "paddypower", "skybet",
                "ladbrokes_uk", "ladbrokes_au", "coral", "betvictor", "boylesports", "grosvenor", "virginbet",
                "livescorebet", "casumo", "betway", "sportsbet", "tab", "neds", "playup", "pointsbetau", "betr_au",
                "bluebet", "topsport", "gtbets", "everygame"]


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


def fetch_odds(league: str, api_key: str | None = None, regions="us,us2,eu",
               markets="h2h,spreads,totals", known_teams: list[str] | None = None) -> pd.DataFrame:
    """Long format: event_id, commence, home, away, book, market, side, line, price."""
    r = _get(ODDS_API.format(sport=SPORT_KEY[league]),
             {"apiKey": _key(api_key), "regions": regions, "markets": markets, "oddsFormat": "american"})
    print(f"[odds] requests remaining this month: {r.headers.get('x-requests-remaining')}")
    df = parse_odds_json(r.json(), league, known_teams)
    df.attrs["remaining"] = _remaining(r)                       # an int, never a frame (pandas 3 compares attrs)
    return df


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
def _two_way(df: pd.DataFrame, a: str, b: str, market: str):
    """Matched (line, price_a, price_b) for a two-way market at one book, or None."""
    da, db = df[df.side == a], df[df.side == b]
    for _, ra in da.iterrows():
        if market == "ml": m = db
        elif market == "spreads": m = db[np.isclose(db.line.astype(float), -float(ra.line))]
        else: m = db[np.isclose(db.line.astype(float), float(ra.line))]
        if len(m): return ra.line, ra.price, m.iloc[0].price
    return None


def sharp_fair(event_odds: pd.DataFrame, league: str) -> dict:
    """Invert the sharpest available book into fair mu_margin / mu_total.
    Falls back to the median over all books if no sharp book is posted."""
    sd_s, sd_t = LEAGUE_SD[league]["spread"], LEAGUE_SD[league]["total"]
    out = {"ref_book": None, "mu_margin": np.nan, "mu_total": np.nan, "mu_ml": np.nan}

    books = [b for b in SHARP_BOOKS if b in set(event_odds.book)] or sorted(set(event_odds.book))
    mus_m, mus_t, mus_ml, ref = [], [], [], None
    for bk in books:
        e = event_odds[event_odds.book == bk]
        sp = _two_way(e[e.market == "spreads"], "home", "away", "spreads")
        if sp:
            line, ph, pa = sp
            qh, _ = devig(ph, pa)
            mus_m.append(-line + sd_s * norm.ppf(qh))      # home covers if margin > -line
        ml = _two_way(e[e.market == "ml"], "home", "away", "ml")
        if ml:
            _, ph, pa = ml
            qh, _ = devig(ph, pa)
            mus_ml.append(sd_s * norm.ppf(qh))           # moneyline-implied margin, kept separate
            if not sp: mus_m.append(mus_ml[-1])
        tt = _two_way(e[e.market == "totals"], "over", "under", "totals")
        if tt:
            line, po, pu = tt
            qo, _ = devig(po, pu)
            mus_t.append(line + sd_t * norm.ppf(qo))
        if (sp or ml or tt) and ref is None:
            ref = bk
        if bk in SHARP_BOOKS and (sp or ml):
            break                                           # sharpest book found; stop
    out["ref_book"] = ref
    if mus_m: out["mu_margin"] = float(np.median(mus_m))
    if mus_t: out["mu_total"] = float(np.median(mus_t))
    out["mu_ml"] = float(np.median(mus_ml)) if mus_ml else out["mu_margin"]
    return out


def blended_fair(event_odds: pd.DataFrame, league: str, model_fair: pd.DataFrame | None = None,
                 model_weight: float = 0.25) -> dict:
    """sharp_fair for one event, nudged toward the model: mu = (1-w)*sharp + w*model for the margin and
    the total, the ML mu shifted by the same margin nudge. model_fair: df with home, away, model_margin,
    model_total (or that df already indexed by [home, away]); None or a missing game -> pure sharp fair.
    find_ev and middles.game_fairs both price off exactly this dict."""
    f = sharp_fair(event_odds, league)
    if model_fair is None: return f
    mf = model_fair if isinstance(model_fair.index, pd.MultiIndex) else model_fair.set_index(["home", "away"])
    home, away = event_odds.home.iloc[0], event_odds.away.iloc[0]
    if (home, away) not in mf.index: return f
    mm, mt = mf.loc[(home, away), ["model_margin", "model_total"]]
    out = dict(f)
    if np.isfinite(f["mu_margin"]) and np.isfinite(mm):
        out["mu_margin"] = (1 - model_weight) * f["mu_margin"] + model_weight * mm
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
    """
    sd_s, sd_t = LEAGUE_SD[league]["spread"], LEAGUE_SD[league]["total"]
    mf = model_fair.set_index(["home", "away"]) if model_fair is not None else None
    rows = []
    for eid, e in odds.groupby("event_id"):
        home, away = e.home.iloc[0], e.away.iloc[0]
        f = blended_fair(e, league, mf, model_weight)
        mu_m, mu_t, mu_ml = f["mu_margin"], f["mu_total"], f["mu_ml"]
        for _, o in e.iterrows():
            if o.market == "spreads" and np.isfinite(mu_m):
                home_line = o.line if o.side == "home" else -o.line
                cp = cover_probs(mu_m, home_line, sd_s)
                p_win = cp["win"] if o.side == "home" else cp["loss"]; p_push = cp["push"]
            elif o.market == "totals" and np.isfinite(mu_t):
                tp = total_probs(mu_t, o.line, sd_t)
                p_win, p_push = tp[o.side], tp["push"]
            elif o.market == "ml" and np.isfinite(mu_ml):
                ph = 1 - norm.cdf(0, mu_ml, sd_s)
                p_win, p_push = (ph if o.side == "home" else 1 - ph), 0.0
            else:
                continue
            ek = edge_and_kelly(p_win, p_push, o.price, kelly_fraction, max_stake)
            rows.append(dict(commence=o.get("commence"), matchup=f"{away} @ {home}", market=o.market,
                             side=o.side, team=(home if o.side == "home" else away if o.side == "away" else o.side),
                             line=o.line, price=o.price, book=o.book, ref_book=f["ref_book"],
                             fair_mu={"spreads": mu_m, "totals": mu_t, "ml": mu_ml}[o.market],
                             p_win=p_win, p_push=p_push, fair_price=_fair_american(p_win, p_push),
                             ev_pct=ek["ev"], kelly_stake=ek["stake_frac"]))
    res = pd.DataFrame(rows)
    if res.empty: return res
    return (res[res.ev_pct >= min_ev].sort_values("ev_pct", ascending=False).reset_index(drop=True))


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
