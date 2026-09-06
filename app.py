"""
Live +EV dashboard (tabs: +EV plays, arbs & middles, odds screen, model card, props).
Run locally:  streamlit run app.py
Deploy free:  push repo to GitHub -> share.streamlit.io -> New app -> add secrets.

Refresh strategy: odds are cached for REFRESH_MIN minutes and re-fetched on the
next page load/auto-refresh after that. This spends 1 API request per refresh,
not per viewer, so the free 500/month Odds API tier survives a season if you
keep REFRESH_MIN >= 20 and only leave the tab open on game days.

Player props (Props tab) are never fetched automatically: the per-event endpoint bills
games x markets per scan, so the tab shows the estimate on the button and only spends
credits on click. The raw (billed) prices land in session_state the moment the call
returns; the board is priced from there with the sidebar's blend weight and EV threshold,
so a rerun or a failed projection can never discard paid-for data.

Demo mode: set SHARPMODEL_DEMO_ODDS (secret or env var) to an odds_*.csv written by
`run.py ev` and the board runs off that file -- no key, no credits (screenshots, styling).
"""
import os, time
import numpy as np
import pandas as pd
import streamlit as st

from sharpmodel import SharpModel, load_nfl, load_cfb, props
from sharpmodel.odds import (fetch_odds, find_ev, best_lines, parse_odds_json, fetch_events, fetch_props, within_hours,
                             odds_grid, load_odds_csv, NON_US_BOOKS, top_picks,
                             estimate_prop_credits, PROP_MARKETS_DEFAULT, PROP_MARKETS_ALL)
from sharpmodel.props import load_player_weeks, project_players, fit_dispersion, price_props
from sharpmodel.middles import find_middles, game_fairs, prop_fairs

st.set_page_config(page_title="SharpModel Live", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")

# -------- config / secrets --------
def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:              # no secrets.toml -> fall back to env vars
        return os.environ.get(name, default)

ODDS_KEY = secret("ODDS_API_KEY")
CFBD_KEY = secret("CFBD_API_KEY")
REFRESH_MIN = int(secret("REFRESH_MIN", 20))
DEMO_ODDS = secret("SHARPMODEL_DEMO_ODDS")
PIN = secret("SCAN_PIN")                              # set before sharing the link: gates props scans + force refresh
MAX_PULLS = int(secret("MAX_PULLS_PER_DAY", 24))      # board pulls per day (1 credit each); then the odds freeze till midnight

# -------- look --------
# Cleveland palette: Browns orange #FF3C00 / brown #311D00, Cavs wine #860038 / gold #FDBB30. Icon fonts are left
# alone on purpose (a global font-family override turns Material icons into their literal names).
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600&display=swap');
.block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1440px; }
.sm-hero { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap; margin:0 0 1.1rem 0; }
.sm-eyebrow { font-size:.74rem; letter-spacing:.16em; text-transform:uppercase; color:#b59a86; font-weight:700; }
.sm-title { font-size:2.5rem; font-weight:800; line-height:1.05; margin:.2rem 0 0 0; letter-spacing:-.02em;
            background:linear-gradient(90deg,#fff7f2 0%,#FF3C00 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.sm-pills { display:flex; gap:.45rem; flex-wrap:wrap; }
.sm-pill { border:1px solid #3b2a1f; background:#1a110c; color:#e7d9cf; border-radius:999px; padding:.32rem .75rem; font-size:.76rem; font-weight:600; white-space:nowrap; }
.sm-pill.live { border-color:#FF3C00; color:#ffb597; background:#3a1305; }
.sm-pill.warn { border-color:#860038; color:#ff9ec4; background:#3d0019; }
.sm-pill.demo { border-color:#FDBB30; color:#FDBB30; background:#3a2a00; }
div[data-testid="stMetric"] { background:#1a110c; border:1px solid #3b2a1f; border-radius:14px; padding:.85rem 1rem .7rem 1rem; }
div[data-testid="stMetric"] label p { color:#b59a86 !important; font-size:.74rem !important; letter-spacing:.08em; text-transform:uppercase; font-weight:700; }
div[data-testid="stMetricValue"] { font-size:2.1rem; font-weight:800; letter-spacing:-.02em; }
.sm-card { background:linear-gradient(180deg,#1a110c 0%,#120b07 100%); border:1px solid #3b2a1f; border-radius:16px; padding:1rem 1.15rem; min-height:148px; }
.sm-card.ev { border-color:#FF3C00; box-shadow: inset 0 0 0 1px rgba(255,60,0,.18); }
.sm-card.arb { border-color:#FDBB30; box-shadow: inset 0 0 0 1px rgba(253,187,48,.18); }
.sm-card .k { font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:#b59a86; font-weight:700; }
.sm-card .big { font-size:1.95rem; font-weight:800; margin:.15rem 0 .1rem 0; letter-spacing:-.02em; }
.sm-card .big.ev { color:#FF6A33; }  .sm-card .big.arb { color:#FDBB30; }
.sm-card .big.info { font-size:1.5rem; color:#f5efe9; }  .sm-card .big.dim { color:#b59a86; font-size:1.3rem; }
.sm-card .bet { font-size:1.02rem; font-weight:700; color:#fff7f2; }
.sm-card .sub { font-size:.8rem; color:#b59a86; margin-top:.4rem; line-height:1.4; }
.sm-card .mono { font-family:'JetBrains Mono', ui-monospace, SFMono-Regular, monospace; }
.sm-section { font-size:.74rem; letter-spacing:.16em; text-transform:uppercase; color:#b59a86; font-weight:700; margin:1.1rem 0 .4rem 0; }
button[data-baseweb="tab"] { font-weight:700; font-size:.95rem; padding:.65rem 1.05rem; }
section[data-testid="stSidebar"] { border-right:1px solid #3b2a1f; }
.sm-brand { font-size:1.45rem; font-weight:800; letter-spacing:-.02em; margin-bottom:.4rem; }
.sm-brand span { color:#FF3C00; }
.sm-brand small { display:block; font-size:.7rem; color:#b59a86; letter-spacing:.16em; text-transform:uppercase; font-weight:700; margin-top:.1rem; }
.sm-foot { color:#8a7462; font-size:.76rem; margin-top:1.5rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

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
MARKET_NAMES = {"spreads": "Spread", "totals": "Total", "ml": "Moneyline",
                "player_pass_yds": "Pass yds", "player_pass_tds": "Pass TDs", "player_rush_yds": "Rush yds",
                "player_reception_yds": "Rec yds", "player_receptions": "Receptions", "player_anytime_td": "Anytime TD",
                "player_pass_attempts": "Pass att", "player_pass_completions": "Completions", "player_rush_attempts": "Rush att"}
TYPE_NAMES = {"arb": "🔒 arb", "arb+middle": "🔒 arb + middle", "free_middle": "🟢 free middle", "middle": "🎯 middle",
              "half_middle": "½ half middle"}

def book(b): return BOOK_NAMES.get(b, b)
def market(m): return MARKET_NAMES.get(m, m)
def am(p):
    try: return f"{int(round(float(p))):+d}"
    except (TypeError, ValueError): return ""
def pretty_bets(s):
    """'PHI -2.5 -110 @ draftkings' -> '... @ DraftKings' (bet strings and the alt column)."""
    out = s.astype(str)
    for k, v in BOOK_NAMES.items():
        out = out.str.replace(f"@ {k}", f"@ {v}", regex=False).str.replace(f" {k} ", f" {v} ", regex=False)
    return out

# -------- sidebar --------
st.sidebar.markdown('<div class="sm-brand">🏈 Sharp<span>Model</span><small>live betting board</small></div>',
                    unsafe_allow_html=True)
league = st.sidebar.radio("League", ["nfl", "cfb"], horizontal=True)
sc1, sc2 = st.sidebar.columns(2)
season = sc1.number_input("Season", 2020, 2030, 2026)
week = sc2.number_input("Week", 1, 22, 1)
filters = st.sidebar.expander("Filters", expanded=True)        # the Teams picker is appended once the board is loaded
with filters:
    min_ev = st.slider("Min EV %", 0.0, 8.0, 1.5, 0.5) / 100
    hours = st.number_input("Kickoff within (hours)", 1, 2000, 240,
                            help="The API posts the whole season; 240 h = a full Tue-Mon slate.")
    markets = st.multiselect("Markets", ["spreads", "totals", "ml"], ["spreads", "totals", "ml"], format_func=market)
    books_excl = st.text_input("Exclude books (comma sep)", "nonus",
                               help="Books you cannot bet at are dropped as +EV rows and as arb/middle legs but still "
                                    "anchor the fair numbers. 'nonus' = every EU/UK/AU/exchange key + pinnacle.")
with st.sidebar.expander("Model & refresh", expanded=False):
    model_w = st.slider("Model blend weight", 0.0, 0.6, 0.0, 0.05,
                        help="Share of the fair number that comes from the ratings model. 0 = sharp-book market only, "
                             "i.e. stale lines and middles. Above 0 the board fills with the model disagreeing with "
                             "every book at once, which is not a stale line.")
    auto = st.toggle("Auto-refresh", True)
    pin = st.text_input("Owner PIN", type="password", help="Set SCAN_PIN in the app secrets before sharing the link: "
                        "only the PIN can scan props or force a refresh (both spend credits).") if PIN else ""
    owner = (not PIN) or pin == PIN
    if owner and st.button("Force refresh now"):
        st.cache_data.clear()
excl = [b.strip() for b in books_excl.split(",") if b.strip()]
if "nonus" in excl:
    excl = [b for b in excl if b != "nonus"] + NON_US_BOOKS

# -------- cached loaders --------
@st.cache_data(ttl=60 * 60 * 6, show_spinner="Loading schedule & results…")
def load_hist(league, season):
    if league == "nfl":
        return load_nfl([season - 1, season])
    os.environ["CFBD_API_KEY"] = CFBD_KEY or ""
    return load_cfb([season - 1, season])

@st.cache_data(ttl=60 * 60 * 6, show_spinner="Fitting ratings…")
def model_card(league, season, week):
    hist = load_hist(league, season)
    wp = SharpModel(league).predict_week(hist, season, week)
    return wp.preds, wp.ratings, wp.market_ratings

@st.cache_resource
def pull_budget():
    """Process-wide: pulls today and the last board. Shared by every viewer -- the point is that a link passed
    around cannot spend more than MAX_PULLS credits a day on the game-line board."""
    return {"day": None, "n": 0, "last": None}

@st.cache_data(ttl=60 * REFRESH_MIN, show_spinner="Pulling odds from every book…")
def load_odds(league, season):
    """-> (odds, fetched_at, credits_remaining, paused). Demo file, or the live board (1 credit), or empty without a
    key. Past the daily budget the last board is served unchanged (paused=True) until the New York date changes."""
    if DEMO_ODDS:
        return load_odds_csv(DEMO_ODDS), time.time(), None, False
    if not ODDS_KEY:
        return pd.DataFrame(), time.time(), None, False
    b, today = pull_budget(), pd.Timestamp.now(tz="America/New_York").date()
    if b["day"] != today:
        b["day"], b["n"] = today, 0
    if b["n"] >= MAX_PULLS:
        if b["last"] is not None:
            df, ts, rem = b["last"]
            return df, ts, rem, True
        return pd.DataFrame(), time.time(), None, True
    known = None
    if league == "cfb":
        h = load_hist(league, season); known = sorted(set(h.home) | set(h.away))
    df = fetch_odds(league, api_key=ODDS_KEY, known_teams=known)
    b["n"] += 1
    b["last"] = (df, time.time(), df.attrs.get("remaining"))
    return df, time.time(), df.attrs.get("remaining"), False

@st.cache_data(ttl=60 * REFRESH_MIN, show_spinner="Pricing the board…")
def price_board(league, season, week, model_w, min_ev, markets, excl, hours, teams, fetched_at, model_fair):
    """+EV rows and arbs/middles for the sidebar settings. Keyed on fetched_at so a fresh odds pull re-prices;
    otherwise a slider change is a cache hit, not a 30 s recompute. Excluded books still anchor the fairs."""
    odds, _, _, _ = load_odds(league, season)
    odds = within_hours(odds[odds.market.isin(list(markets))], hours)
    if teams: odds = odds[odds.home.isin(teams) | odds.away.isin(teams)]
    ev = find_ev(odds, league, model_fair=model_fair, model_weight=model_w, min_ev=min_ev)
    if excl and len(ev): ev = ev[~ev.book.isin(excl)].reset_index(drop=True)
    mids = find_middles(odds, game_fairs(odds, league, model_fair, model_w), league, exclude=list(excl))
    return odds, ev, mids

@st.cache_data(ttl=60 * 30, show_spinner="Listing upcoming games (free endpoint)…")
def load_events(league):
    ev = fetch_events(league, api_key=ODDS_KEY)
    return ev, ev.attrs.get("remaining"), time.time()          # credits left (for fetch_props' reserve check) + when

@st.cache_data(ttl=60 * 60 * 6, show_spinner="Projecting players…")
def props_projections(season, week):
    stats = load_player_weeks([season - 2, season - 1, season])
    cv = fit_dispersion(stats, as_of=(season, week))          # returned too: the cache hit skips this side effect
    return project_players(stats, load_hist("nfl", season), as_of=(season, week)), cv

def kickoff(s):
    """'Sun 1:00PM' in Cleveland time."""
    try:
        t = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert("America/New_York")
        return t.dt.strftime("%a ") + t.dt.hour.mod(12).replace(0, 12).astype("Int64").astype(str) + t.dt.strftime(":%M%p")
    except Exception:
        return pd.to_datetime(s, errors="coerce", utc=True).dt.strftime("%a %H:%M UTC")

def age_text(minutes):
    return "just now" if minutes < 1 else f"{minutes:.0f} min ago"

def pinned(cfg_cls, label, **kw):
    """column_config with pinned=True where this Streamlit supports it (1.5x+); plain otherwise."""
    try: return cfg_cls(label, pinned=True, **kw)
    except TypeError: return cfg_cls(label, **kw)

# -------- views --------
NO_KEY = "No odds loaded. Add ODDS_API_KEY in Streamlit secrets (free at the-odds-api.com)."
MID_COLS = ["commence", "matchup", "market", "player", "type", "bet_a", "bet_b", "window", "p_middle", "miss_cost_pct",
            "win_both_pct", "ev_pct", "guaranteed_pct", "breakeven_p", "stake_a_pct", "stake_b_pct", "alt"]
MID_NAMES = {"bet_a": "leg A", "bet_b": "leg B", "p_middle": "middle %", "miss_cost_pct": "miss cost %",
             "win_both_pct": "win both %", "ev_pct": "EV %", "guaranteed_pct": "guaranteed %",
             "breakeven_p": "breakeven %", "stake_a_pct": "stake A %", "stake_b_pct": "stake B %",
             "alt": "same numbers also at"}
MID_NOTE = ("Two legs at two books, stakes split so a miss (the sides split) costs the same either way; every % is of "
            "the total stake. The middle lands when the number finishes inside the window and both legs cash "
            "(win both %); miss cost % is what a miss loses; breakeven % is the middle probability that covers it and "
            "middle % is the fair distribution's probability (NFL spreads and moneylines use the empirical key-number "
            "margin distribution, so a 3 counts ~3x what a Normal says). arb / free_middle cannot lose; half_middle "
            "wins one leg on the 'Np' number while the other pushes. Lines move within minutes: place the "
            "worse-priced leg first, confirm it is accepted, then the other. Same-book pairs are excluded.")
SCREEN_NOTE = ("One row per pick, one column per book, cell = 'number price'. Green = the best price among the books "
               "on the market's number; red = that book is on a DIFFERENT number (a middle candidate: see Arbs & "
               "middles). 'Best' repeats the green cell; excluded books are hidden.")

def ev_cfg(max_ev):
    return {"commence": st.column_config.TextColumn("Kickoff", width="small"),
            "matchup": st.column_config.TextColumn("Game", width="small"),
            "market": st.column_config.TextColumn("Market", width="small"),
            "team": st.column_config.TextColumn("Pick", width="small"),
            "player": st.column_config.TextColumn("Player"),
            "side": st.column_config.TextColumn("Side", width="small"),
            "line": st.column_config.TextColumn("Line", width="small"),
            "price": st.column_config.TextColumn("Price", width="small"),
            "book": st.column_config.TextColumn("Book"),
            "fair_price": st.column_config.TextColumn("Fair", width="small", help="the sharp-book fair price for this line"),
            "p_win": st.column_config.NumberColumn("Win %", format="%.1f%%", width="small"),
            "ev_pct": st.column_config.ProgressColumn("EV %", format="%.2f%%", min_value=0.0, max_value=float(max(3.0, max_ev))),
            "kelly_stake": st.column_config.NumberColumn("Stake %", format="%.2f%%", width="small",
                                                         help="quarter-Kelly, % of bankroll"),
            "ref_book": st.column_config.TextColumn("vs", width="small", help="the sharp reference book"),
            "n_books": st.column_config.NumberColumn("Books", width="small"),
            "market_mu": st.column_config.NumberColumn("Market", format="%.1f", width="small"),
            "proj_mean": st.column_config.NumberColumn("Proj", format="%.1f", width="small"),
            "fair_mu": st.column_config.NumberColumn("Fair #", format="%.1f", width="small"),
            "flags": st.column_config.TextColumn("Flags", width="small")}

def ev_view(ev, cols):
    show = ev.copy()
    show["ev_pct"] = (show.ev_pct * 100).round(2)
    show["kelly_stake"] = (show.kelly_stake * 100).round(2)
    show["p_win"] = (show.p_win * 100).round(1)
    show["fair_price"] = show.fair_price.map(am)
    show["price"] = show.price.map(am)
    spread = show.market.eq("spreads")
    show["line"] = [("" if pd.isna(x) else f"{x:+g}" if s else f"{x:g}") for x, s in zip(show.line, spread)]
    show["commence"] = kickoff(show.commence)
    show["market"] = show.market.map(market)
    show["book"] = show.book.map(book)
    if "ref_book" in show: show["ref_book"] = show.ref_book.map(book)
    cols = [c for c in cols if c in show]
    st.dataframe(show[cols], width="stretch", hide_index=True, column_config=ev_cfg(show.ev_pct.max()))

def pick_text(r):
    if r.market == "spreads": return f"{r.team} {r.line:+g}"
    if r.market == "ml": return f"{r.team} ML"
    if r.market == "totals": return f"{str(r.team).capitalize()} {r.line:g}"
    line = "" if pd.isna(r.line) else f" {r.line:g}"
    return f"{getattr(r, 'player', '')} {str(r.side).capitalize()}{line}"          # props: 'Jalen Hurts Over 264.5'

def pretty_also(s):
    """'espnbet +100; betmgm -102' -> 'ESPN BET +100; BetMGM -102'."""
    def one(x):
        parts = x.split(" ", 1)
        return f"{book(parts[0])} {parts[1]}" if len(parts) == 2 else x
    return s.fillna("").astype(str).map(lambda v: "; ".join(one(x) for x in v.split("; ") if x))

def picks_view(p):
    show = p.copy()
    if "team" not in show: show["team"] = ""
    show["pick"] = [pick_text(r) for r in show.itertuples()]
    show["commence"] = kickoff(show.commence)
    show["market"] = show.market.map(market)
    show["book"] = show.book.map(book)
    show["price"] = show.price.map(am)
    show["fair_price"] = show.fair_price.map(am)
    show["also"] = pretty_also(show.also)
    show["ev_pct"] = (show.ev_pct * 100).round(2)
    show["p_win"] = (show.p_win * 100).round(1)
    show["kelly_stake"] = (show.kelly_stake * 100).round(2)
    cols = ["rank", "commence", "matchup", "market", "pick", "price", "book", "also", "fair_price", "p_win", "ev_pct",
            "kelly_stake"]
    cfg = ev_cfg(show.ev_pct.max())
    cfg.update({"rank": st.column_config.NumberColumn("#", width="small"),
                "pick": st.column_config.TextColumn("Pick", width="small"),
                "book": st.column_config.TextColumn("Best book"),
                "also": st.column_config.TextColumn("Also at", width="large")})
    st.dataframe(show[cols], width="stretch", hide_index=True, column_config=cfg,
                 height=int(min(80 + 35 * (len(show) + 1), 480)))

def mid_cfg(max_ev):
    return {"commence": st.column_config.TextColumn("Kickoff", width="small"),
            "matchup": st.column_config.TextColumn("Game", width="small"),
            "market": st.column_config.TextColumn("Market", width="small"),
            "player": st.column_config.TextColumn("Player"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "leg A": st.column_config.TextColumn("Leg A", width="medium"),
            "leg B": st.column_config.TextColumn("Leg B", width="medium"),
            "window": st.column_config.TextColumn("Window", width="small", help="numbers where BOTH legs cash"),
            "middle %": st.column_config.NumberColumn("Middle %", format="%.1f%%", width="small", help="chance the number lands inside the window"),
            "miss cost %": st.column_config.NumberColumn("Miss cost", format="%.2f%%", width="small", help="what you lose when the sides split"),
            "win both %": st.column_config.NumberColumn("Win both", format="%.1f%%", width="small"),
            "EV %": st.column_config.ProgressColumn("EV %", format="%.2f%%", min_value=0.0, max_value=float(max(3.0, max_ev))),
            "guaranteed %": st.column_config.NumberColumn("Locked", format="%.2f%%", width="small", help="worst case, as % of the total stake; > 0 = arb"),
            "breakeven %": st.column_config.NumberColumn("Breakeven", format="%.1f%%", width="small"),
            "stake A %": st.column_config.NumberColumn("Stake A", format="%.0f%%", width="small"),
            "stake B %": st.column_config.NumberColumn("Stake B", format="%.0f%%", width="small"),
            "same numbers also at": st.column_config.TextColumn("Also at", width="medium")}

def middles_view(m):
    show = m.copy()
    show["commence"] = kickoff(show.commence)
    for c in ("p_middle", "breakeven_p"): show[c] = (show[c] * 100).round(1)
    for c in ("miss_cost_pct", "win_both_pct", "ev_pct", "guaranteed_pct", "stake_a_pct", "stake_b_pct"):
        show[c] = show[c].round(2)
    show["market"] = show.market.map(market)
    show["type"] = show.type.map(lambda t: TYPE_NAMES.get(t, t))
    for c in ("bet_a", "bet_b", "alt"): show[c] = pretty_bets(show[c])
    cols = [c for c in MID_COLS if c != "player" or show.player.notna().any()]
    return show[cols].rename(columns=MID_NAMES)

def middles_table(m):
    v = middles_view(m)
    st.dataframe(v, width="stretch", hide_index=True, column_config=mid_cfg(v["EV %"].max()))

def screen_view(grid, best, off):
    """The odds screen (odds_grid(..., masks=True)): best price at the number in green, off-number books in amber."""
    if grid is None or grid.empty:
        st.success("No lines in the window."); return
    show = grid.copy()
    if show.commence.astype(bool).any(): show["commence"] = kickoff(show.commence)
    show["market"] = show.market.map(market)
    show["best"] = pretty_bets(show.best)
    books = list(best.columns)
    pretty = [book(b) for b in books]
    show = show.rename(columns=dict(zip(books, pretty)))
    b, o = best.to_numpy(dtype=bool), off.to_numpy(dtype=bool)          # plain arrays: no pandas truthiness anywhere
    def paint(col):
        j = pretty.index(col.name)
        return np.where(b[:, j], "background-color:#166534;color:#dcfce7;font-weight:700",       # green: best price on the number
                        np.where(o[:, j], "background-color:#7f1d1d;color:#fee2e2", "")).tolist()  # red: a different number
    cfg = {"commence": st.column_config.TextColumn("Kickoff", width="small"),
           "matchup": pinned(st.column_config.TextColumn, "Game", width="small"),
           "market": st.column_config.TextColumn("Market", width="small"),
           "pick": pinned(st.column_config.TextColumn, "Pick", width="small"),
           "player": pinned(st.column_config.TextColumn, "Player"),
           "best": st.column_config.TextColumn("Best", width="medium"),
           "books": st.column_config.NumberColumn("#", width="small")}
    height = int(min(80 + 35 * (len(show) + 1), 720))
    try:
        st.dataframe(show.style.apply(paint, axis=0, subset=pretty), width="stretch", hide_index=True, column_config=cfg,
                     height=height)
    except Exception:                                                     # a pandas/streamlit Styler mismatch never takes the page down
        marked = show.copy()
        for j, bk in enumerate(pretty):
            marked[bk] = np.where(b[:, j], "★ ", np.where(o[:, j], "≠ ", "")) + marked[bk].astype(str)
        st.dataframe(marked, width="stretch", hide_index=True, column_config=cfg, height=height)
        st.caption("★ = best price on the number, ≠ = a different number (colour styling unavailable in this pandas build).")
    st.caption(SCREEN_NOTE)

def card(kind, k, big, bet, sub):
    st.markdown(f'<div class="sm-card {kind if kind != "info" else ""}"><div class="k">{k}</div><div class="big {kind}">{big}</div>'
                f'<div class="bet">{bet}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

def ev_bet(r):
    if r.market == "spreads": return f"{r.team} {r.line:+g} <span class='mono'>{am(r.price)}</span>"
    if r.market == "ml": return f"{r.team} ML <span class='mono'>{am(r.price)}</span>"
    if r.market == "totals": return f"{r.team.capitalize()} {r.line:g} <span class='mono'>{am(r.price)}</span>"
    return f"{r.get('player', '')} {r.side} {r.line:g} <span class='mono'>{am(r.price)}</span>"

def top_cards(ev, mids, min_ev, age, remaining, n_games, n_books):
    c1, c2, c3 = st.columns(3)
    with c1:
        if len(ev):
            r = ev.iloc[0]
            card("ev", "Best +EV line", f"+{r.ev_pct * 100:.1f}% EV", ev_bet(r),
                 f"{book(r.book)} · {r.matchup} · fair <span class='mono'>{am(r.fair_price)}</span> at {book(r.ref_book)} · "
                 f"win {r.p_win:.0%} · stake {r.kelly_stake:.1%}")
        else:
            card("ev", "Best +EV line", "<span class='dim'>nothing above %.1f%%</span>" % (min_ev * 100),
                 "Efficient board right now", "That's normal between line moves. Lower Min EV % or check back after the sharp books move.")
    with c2:
        if len(mids):
            r = mids.iloc[0]
            locked = r.guaranteed_pct > 1e-9
            big = f"+{r.guaranteed_pct:.2f}% locked" if locked else f"+{r.ev_pct:.2f}% EV"
            sub = (f"{TYPE_NAMES.get(r.type, r.type)} · {r.matchup}" + (f" · window {r.window}" if r.window else "")
                   + (f" · hits {r.p_middle:.0%}, miss costs {r.miss_cost_pct:.1f}%" if not locked else " · cannot lose"))
            card("arb", "Best arb / middle", big,
                 f"{pretty_bets(pd.Series([r.bet_a])).iloc[0]} &nbsp;<span style='color:#64748b'>×</span>&nbsp; "
                 f"{pretty_bets(pd.Series([r.bet_b])).iloc[0]}", sub)
        else:
            card("arb", "Best arb / middle", "<span class='dim'>none on the board</span>", "No cross-book gaps right now",
                 "Middles appear when books disagree on the number — most often in the hours after a sharp move.")
    with c3:
        card("info", "Board", f"{n_games} games · {n_books} books",
             f"Odds pulled {age_text(age)}" + (f" · {remaining} credits left this month" if remaining is not None else ""),
             f"Refreshes every {REFRESH_MIN} min while this tab is open (1 credit each). Non-US books are hidden as bets "
             "but still set the fair numbers.")

# -------- main --------
odds, fetched_at, remaining, paused = load_odds(league, season)
has_odds = not odds.empty                                          # a key and a live pull; the window may still empty the board
age = (time.time() - fetched_at) / 60
pills = []
if DEMO_ODDS: pills.append('<span class="sm-pill demo">demo file</span>')
if paused:
    pills.append(f'<span class="sm-pill warn">paused · {MAX_PULLS} pulls today, odds frozen until midnight</span>')
else:
    pills.append(f'<span class="sm-pill live">● live · odds {age_text(age)}</span>' if has_odds else
                 '<span class="sm-pill warn">no odds key</span>')
if remaining is not None: pills.append(f'<span class="sm-pill">{remaining} credits left</span>')
pills.append(f'<span class="sm-pill">{"market only" if model_w == 0 else f"model {model_w:.0%}"}</span>')
st.markdown(f'<div class="sm-hero"><div><div class="sm-eyebrow">{league.upper()} · {season} · week {week}</div>'
            f'<div class="sm-title">{league.upper()} +EV board</div></div><div class="sm-pills">{"".join(pills)}</div></div>',
            unsafe_allow_html=True)
if odds.empty:
    st.warning(f"Daily credit budget reached ({MAX_PULLS} pulls); the board will pull again after midnight New York time."
               if paused else NO_KEY)
else:
    try:
        preds, ratings, mkt_ratings = model_card(league, season, week)
        model_fair = preds[["home", "away", "model_margin", "model_total"]]
    except Exception as e:
        preds, ratings, mkt_ratings, model_fair = None, None, None, None
        st.info(f"Model unavailable ({e}); showing pure market-vs-market EV.")

    n_posted = odds.event_id.nunique()
    with filters:
        teams = st.multiselect("Teams", sorted(set(odds.home) | set(odds.away)), [], placeholder="Type to search…",
                               help="Only games involving these teams; leave empty for the whole slate.")
        present = [b for b in odds.book.unique() if b not in excl]
        keys_by_name = {book(b): b for b in sorted(present, key=lambda b: (b not in BOOK_NAMES, book(b)))}
        my_books = st.multiselect("Books", list(keys_by_name), [], placeholder="Type to search…",
                                  help="Only show prices and legs at these books (your accounts); leave empty for all. "
                                       "Every book still helps set the fair numbers.")
        if my_books:                                                   # hide the rest as bets, keep them in the fairs
            keep = {keys_by_name[n] for n in my_books}
            excl = excl + [b for b in present if b not in keep]
    odds, ev, mids = price_board(league, season, week, model_w, min_ev, tuple(markets), tuple(excl), int(hours),
                                 tuple(teams), fetched_at, model_fair)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games", odds.event_id.nunique(), help=f"kicking off within {hours} h, of {n_posted} posted")
    c2.metric("Books", odds.book.nunique())
    c3.metric("+EV lines", len(ev))
    c4.metric("Arbs & middles", len(mids), help="cross-book pairs that cannot lose or are +EV to middle")
    c5.metric("Odds age", "now" if age < 1 else f"{age:.0f} min", help=f"Refreshes every {REFRESH_MIN} min")
    st.markdown('<div class="sm-section">Right now</div>', unsafe_allow_html=True)
    top_cards(ev, mids, min_ev, age, remaining, odds.event_id.nunique(), odds.book.nunique())
    st.markdown('<div class="sm-section">The board</div>', unsafe_allow_html=True)

labels = ["Top picks", "+EV plays", "Arbs & middles", "Odds screen", "Model card", "Props"]
if has_odds:
    picks = top_picks(ev, 10)
    labels[0], labels[1], labels[2] = f"Top picks · {len(picks)}", f"+EV plays · {len(ev)}", f"Arbs & middles · {len(mids)}"
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(labels)
if has_odds:
    with tab0:
        with st.expander("How to read this"):
            st.markdown("- The same list as **+EV plays**, but one row per pick at its **best book** — 'CHI −3 +100' "
                        "is one pick even when five books post it. **Also at** shows the other books on the same number, "
                        "best price first, so you can go where you have an account.\n"
                        "- Ranked by EV %. **Stake %** is a quarter-Kelly size; spread it across books rather than "
                        "hammering one.\n"
                        "- Lines move within minutes of a sharp move — check the price is still there before you bet.")
        if len(picks):
            picks_view(picks)
            st.caption(f"Top {len(picks)} of {len(ev)} +EV prices on the board, deduped to one row per pick. "
                       "Fair prices from the sharp reference book; stakes quarter-Kelly, capped at 3% of bankroll.")
        else:
            st.success("Nothing above your EV threshold right now — that's normal on an efficient board.")
    with tab1:
        with st.expander("How to read this"):
            st.markdown("- One row = one book's price that is better than the sharp fair price for that exact line.\n"
                        "- **EV %** is your edge per dollar; **Stake %** is a quarter-Kelly bet size as a share of bankroll.\n"
                        "- **Fair** is what the sharp reference book (the *vs* column) says the line is worth. "
                        "Bet the row's book at the row's price — lines move within minutes.")
        if len(ev):
            ev_view(ev, ["commence", "matchup", "market", "team", "line", "price", "book", "fair_price", "p_win",
                         "ev_pct", "kelly_stake", "ref_book"])
            st.caption("One row per stale price. Fair prices come from the sharp reference book (vs), nudged toward the "
                       "model by the blend weight; stakes are quarter-Kelly, capped at 3% of bankroll.")
        else:
            st.success("Nothing above your EV threshold right now — that's normal on an efficient board.")
    with tab2:
        with st.expander("How to read this"):
            st.markdown("- Two bets at two books on opposite sides. **Leg A** and **Leg B** are the exact tickets; "
                        "**Stake A/B** is how to split the money so a miss costs the same either way.\n"
                        "- **🔒 arb**: you profit whatever happens (the *Locked* %). **🟢 free middle**: worst case zero, "
                        "and you win both if the number lands inside the **Window**. **🎯 middle**: a miss costs the "
                        "*Miss cost* %, a hit pays *Win both* %; it's +EV because the window hits more often than "
                        "the breakeven %.\n"
                        "- Place the worse-priced leg first, confirm it's accepted, then the other.")
        if len(mids):
            middles_table(mids)
        else:
            st.success("No cross-book arbs or +EV middles on the board right now.")
        st.caption(MID_NOTE)
    with tab3:
        with st.expander("How to read this"):
            st.markdown("- Every sportsbook's number and price for every pick, side by side. Scroll right for more books.\n"
                        "- **Green** = the best price on the market's number (the *Best* column repeats it). "
                        "**Red** = that book is on a *different* number — compare it with the Arbs & middles tab.\n"
                        "- Hidden books (the sidebar's *Exclude books*) still set the fair numbers, they just can't be bet.")
        screen_view(*odds_grid(odds, exclude=excl, masks=True))
    with tab4:
        if preds is not None:
            cols = ["away", "home", "market_margin", "model_margin", "fair_margin", "spread_side",
                    "spread_edge_pts", "market_total", "model_total", "total_side", "total_edge_pts", "qb_flag"]
            st.dataframe(preds[[c for c in cols if c in preds]].round(2), width="stretch", hide_index=True,
                         column_config={"market_margin": "market", "model_margin": "model", "fair_margin": "fair",
                                        "spread_side": "side", "spread_edge_pts": "edge pts", "market_total": "mkt total",
                                        "model_total": "model total", "total_side": "O/U", "total_edge_pts": "edge pts ",
                                        "qb_flag": "QB flag"})
            st.markdown('<div class="sm-section">Power ratings · model vs market-implied</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"model": ratings, "market": mkt_ratings}).round(1)
                         .sort_values("model", ascending=False), width="stretch")
            st.caption("The ratings model alone does not beat the closing line (51.1% ATS, 2019–2025). It is here for "
                       "context; the edge is in the other tabs.")

# -------- props (manual scan only; never auto-fetched) --------
with tab5:
    st.markdown('<div class="sm-section">Player props · manual scan</div>', unsafe_allow_html=True)
    if league != "nfl":
        st.info("Props are NFL-only (projections come from nflverse player stats).")
    elif not ODDS_KEY:
        st.warning(NO_KEY)
    else:
        try:
            events, ev_remaining, events_at = load_events(league)
        except Exception as e:
            st.error(f"Events list failed: {e}")
            events, ev_remaining, events_at = pd.DataFrame(columns=["event_id", "commence", "home", "away"]), None, 0.0
        scanned = st.session_state.get("props_odds")
        if scanned and scanned.get("remaining") is not None and scanned["at"] > events_at:
            ev_remaining = scanned["remaining"]                # the last billed call is fresher than the cached events call
        p_markets = st.multiselect("Prop markets", PROP_MARKETS_ALL, PROP_MARKETS_DEFAULT, format_func=market)
        pc1, pc2 = st.columns(2)
        p_hours = pc1.number_input("Hours ahead", 1, 240, 72, 12)
        credits = pc2.number_input("Max credits this scan", 1, 500, 60, 10)
        now = pd.Timestamp.now(tz="UTC")
        window = events[(events.commence > now) & (events.commence <= now + pd.Timedelta(hours=int(p_hours)))]
        cost = estimate_prop_credits(len(window), p_markets)
        st.caption(f"{len(window)} game(s) kick off within {int(p_hours)}h"
                   + (": " + ", ".join(f"{r.away}@{r.home}" for r in window.itertuples()) if len(window) else "")
                   + f" — {len(window)} x {len(p_markets)} markets x 1 region = {cost} credits"
                   + (f"; {ev_remaining} left this month." if ev_remaining is not None else "."))
        if not owner:
            st.caption("🔒 Scanning spends credits — enter the owner PIN in the sidebar (Model & refresh) to enable it.")
        if st.button(f"Scan props (≈{cost} credits)", disabled=(cost == 0 or not owner), type="primary"):
            try:
                with st.spinner("Pulling props (one call per game)…"):
                    po = fetch_props(league, window, markets=p_markets, api_key=ODDS_KEY, max_credits=int(credits),
                                     remaining=ev_remaining)
                labels = dict(zip(window.event_id, window.away + "@" + window.home))
                # billed data lands in session_state first; projections/pricing below can fail without losing it
                st.session_state["props_odds"] = dict(odds=po, at=time.time(),
                                                      failed=[labels.get(i, i) for i in po.attrs.get("failed", [])],
                                                      remaining=po.attrs.get("remaining"))   # last response's header
                if "props" in st.session_state: del st.session_state["props"]
            except Exception as e:
                st.error(str(e))
        raw = st.session_state.get("props_odds")
        if raw is None:
            st.info("Nothing scanned yet. Pick markets and a window, check the credit estimate on the button, then scan.")
        else:
            po = raw["odds"]
            if raw["failed"]:
                st.warning(f"{len(raw['failed'])} game(s) missing after Odds API errors (see server log): "
                           + ", ".join(raw["failed"]))
            key = (int(season), int(week), float(model_w), float(min_ev))
            board = st.session_state.get("props")
            if board is None or board["key"] != key:                 # price from session_state; re-price on slider change
                proj, note = None, None
                try:
                    proj, cv = props_projections(season, week); props._CV.update(cv)
                except Exception as e:
                    note = f"Projections unavailable ({e}); pricing market-only."
                board = dict(ev=price_props(po, proj, model_weight=model_w, min_ev=min_ev), key=key, note=note)
                full = price_props(po, proj, model_weight=model_w, min_ev=-1.0)   # every two-way market -> fairs
                board["mid"] = find_middles(po, prop_fairs(full), "nfl", exclude=excl)
                st.session_state["props"] = board
            if board["note"]: st.info(board["note"])
            pev = board["ev"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games", po.event_id.nunique()); m2.metric("Books", po.book.nunique()); m3.metric("Prop prices", len(po))
            m4.metric("+EV props", len(pev), help="above the sidebar Min EV %")
            if len(pev):
                st.markdown('<div class="sm-section">Top prop picks · best book per prop</div>', unsafe_allow_html=True)
                picks_view(top_picks(pev, 10))
                st.markdown('<div class="sm-section">Every +EV prop price</div>', unsafe_allow_html=True)
                ev_view(pev, ["commence", "matchup", "market", "player", "side", "line", "price", "book", "n_books",
                              "market_mu", "proj_mean", "fair_mu", "fair_price", "p_win", "ev_pct", "kelly_stake", "flags"])
            else:
                st.success("Nothing above your EV threshold in the last scan.")
            st.markdown('<div class="sm-section">Prop arbs & middles</div>', unsafe_allow_html=True)
            pm = board.get("mid")
            if pm is not None and len(pm):
                middles_table(pm)
                st.caption(MID_NOTE + " Prop limits are low.")
            else:
                st.caption("No cross-book prop arbs or +EV middles in the last scan. " + MID_NOTE)
            st.markdown('<div class="sm-section">Prop odds screen</div>', unsafe_allow_html=True)
            screen_view(*odds_grid(raw["odds"], exclude=excl, masks=True))
            st.caption(f"Scanned {pd.Timestamp.fromtimestamp(raw['at']).strftime('%H:%M:%S')} server time; the board stays "
                       "until you scan again. Quota: each scan bills games x markets x regions credits from the free "
                       "500/month Odds API tier (the ≈N on the button), on top of the main board's 1 per refresh. "
                       f"Fair = {1 - model_w:.0%}/{model_w:.0%} devigged market consensus / projection (sidebar blend "
                       "weight); stakes are quarter-Kelly capped at 2%. 'no_market' rows have no two-way market behind "
                       "them: no EV, never staked, listed last. 'team_mismatch' rows are staked 0; 'no_proj' rows are "
                       "priced market-only. Props limits are low and books limit winners.")

st.markdown(f'<div class="sm-foot">Last odds pull {pd.Timestamp.fromtimestamp(fetched_at).strftime("%Y-%m-%d %H:%M:%S")} server time · '
            'the ratings model does not beat closing lines — the edge is stale prices, arbs and middles · '
            'educational tooling; bet responsibly and only where legal.</div>', unsafe_allow_html=True)

# -------- auto refresh --------
if auto:
    from streamlit_autorefresh import st_autorefresh
    secs_left = max(REFRESH_MIN * 60 - (time.time() - fetched_at), 15)
    st_autorefresh(interval=int(secs_left * 1000), key="odds_refresh")
    st.sidebar.caption(f"Next refresh in ~{secs_left/60:.0f} min")
