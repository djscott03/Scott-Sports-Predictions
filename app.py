"""
Live +EV dashboard (tabs: +EV plays, arbs & middles, best lines, model card, props).
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
"""
import os, time
import pandas as pd
import streamlit as st

from sharpmodel import SharpModel, load_nfl, load_cfb, props
from sharpmodel.odds import (fetch_odds, find_ev, best_lines, parse_odds_json, fetch_events, fetch_props, within_hours,
                             estimate_prop_credits, PROP_MARKETS_DEFAULT, PROP_MARKETS_ALL)
from sharpmodel.props import load_player_weeks, project_players, fit_dispersion, price_props
from sharpmodel.middles import find_middles, game_fairs, prop_fairs

st.set_page_config(page_title="SharpModel Live", page_icon="🏈", layout="wide")

# -------- config / secrets --------
def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:              # no secrets.toml -> fall back to env vars
        return os.environ.get(name, default)

ODDS_KEY = secret("ODDS_API_KEY")
CFBD_KEY = secret("CFBD_API_KEY")
REFRESH_MIN = int(secret("REFRESH_MIN", 20))

# -------- sidebar --------
st.sidebar.title("SharpModel Live")
league = st.sidebar.radio("League", ["nfl", "cfb"], horizontal=True)
season = st.sidebar.number_input("Season", 2020, 2030, 2026)
week = st.sidebar.number_input("Week", 1, 22, 1)
min_ev = st.sidebar.slider("Min EV %", 0.0, 8.0, 1.5, 0.5) / 100
model_w = st.sidebar.slider("Model blend weight", 0.0, 0.6, 0.0, 0.05,
                            help="Share of the fair number that comes from the ratings model. 0 = sharp-book market "
                                 "only, i.e. stale lines and middles. Above 0 the board fills with the model "
                                 "disagreeing with every book at once, which is not a stale line.")
hours = st.sidebar.number_input("Kickoff within (hours)", 1, 2000, 240,
                                help="The API posts the whole season; 240 h = a full Tue-Mon slate.")
markets = st.sidebar.multiselect("Markets", ["spreads", "totals", "ml"], ["spreads", "totals", "ml"])
books_excl = st.sidebar.text_input("Exclude books (comma sep)", "nonus",
                                   help="Books you cannot bet at are dropped as +EV rows and as arb/middle legs but still "
                                        "anchor the fair numbers. 'nonus' = every EU/UK/AU/exchange key + pinnacle.")
excl = [b.strip() for b in books_excl.split(",") if b.strip()]
if "nonus" in excl:
    from sharpmodel.odds import NON_US_BOOKS
    excl = [b for b in excl if b != "nonus"] + NON_US_BOOKS
auto = st.sidebar.toggle("Auto-refresh", True)
if st.sidebar.button("Force refresh now"):
    st.cache_data.clear()

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

@st.cache_data(ttl=60 * REFRESH_MIN, show_spinner="Pulling odds from every book…")
def load_odds(league, season):
    known = None
    if league == "cfb":
        h = load_hist(league, season); known = sorted(set(h.home) | set(h.away))
    if not ODDS_KEY:
        return pd.DataFrame(), time.time()
    return fetch_odds(league, api_key=ODDS_KEY, known_teams=known), time.time()

@st.cache_data(ttl=60 * REFRESH_MIN, show_spinner="Pricing the board…")
def price_board(league, season, week, model_w, min_ev, markets, excl, hours, fetched_at, model_fair):
    """+EV rows and arbs/middles for the sidebar settings. Keyed on fetched_at so a fresh odds pull re-prices;
    otherwise a slider change is a cache hit, not a 30 s recompute. Excluded books still anchor the fairs."""
    odds, _ = load_odds(league, season)
    odds = within_hours(odds[odds.market.isin(list(markets))], hours)
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
    try:
        return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.strftime("%a %I:%M%p")
    except Exception:
        return pd.to_datetime(s, errors="coerce", utc=True).dt.strftime("%a %H:%M UTC")

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


def middles_view(m):
    show = m.copy()
    show["commence"] = kickoff(show.commence)
    for c in ("p_middle", "breakeven_p"): show[c] = (show[c] * 100).round(1)
    for c in ("miss_cost_pct", "win_both_pct", "ev_pct", "guaranteed_pct", "stake_a_pct", "stake_b_pct"):
        show[c] = show[c].round(2)
    cols = [c for c in MID_COLS if c != "player" or show.player.notna().any()]
    return show[cols].rename(columns=MID_NAMES)

# -------- main --------
odds, fetched_at = load_odds(league, season)
has_odds = not odds.empty                                          # a key and a live pull; the window may still empty the board
age = (time.time() - fetched_at) / 60
st.title(f"{league.upper()} +EV board")
if odds.empty:
    st.warning(NO_KEY)
else:
    try:
        preds, ratings, mkt_ratings = model_card(league, season, week)
        model_fair = preds[["home", "away", "model_margin", "model_total"]]
    except Exception as e:
        preds, ratings, mkt_ratings, model_fair = None, None, None, None
        st.info(f"Model unavailable ({e}); showing pure market-vs-market EV.")

    n_posted = odds.event_id.nunique()
    odds, ev, mids = price_board(league, season, week, model_w, min_ev, tuple(markets), tuple(excl), int(hours),
                                 fetched_at, model_fair)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games", odds.event_id.nunique(), help=f"kicking off within {hours} h, of {n_posted} posted")
    c2.metric("Books", odds.book.nunique())
    c3.metric("+EV lines", len(ev))
    c4.metric("Arbs & middles", len(mids), help="cross-book pairs that cannot lose or are +EV to middle")
    c5.metric("Odds age", f"{age:.0f} min", help=f"Refreshes every {REFRESH_MIN} min")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["+EV plays", "Arbs & middles", "Best lines", "Model card", "Props"])
if has_odds:
    with tab1:
        if len(ev):
            show = ev.copy()
            show["ev_pct"] = (show.ev_pct * 100).round(2)
            show["kelly_stake"] = (show.kelly_stake * 100).round(2)
            show["p_win"] = (show.p_win * 100).round(1)
            show["fair_price"] = show.fair_price.round(0)
            show["commence"] = kickoff(show.commence)
            st.dataframe(show[["commence", "matchup", "market", "team", "line", "price", "book",
                               "fair_price", "p_win", "ev_pct", "kelly_stake", "ref_book"]]
                         .rename(columns={"ev_pct": "EV %", "kelly_stake": "stake % bankroll", "p_win": "win %"}),
                         width="stretch", hide_index=True)
            st.caption("Stakes are quarter-Kelly, capped at 3% of bankroll. Fair prices derive from the sharp reference book "
                       "(ref_book), nudged toward the model by the blend weight.")
        else:
            st.success("Nothing above your EV threshold right now — that's normal on an efficient board.")
    with tab2:
        if len(mids):
            st.dataframe(middles_view(mids), width="stretch", hide_index=True)
        else:
            st.success("No cross-book arbs or +EV middles on the board right now.")
        st.caption(MID_NOTE)
    with tab3:
        st.dataframe(best_lines(odds), width="stretch", hide_index=True)
    with tab4:
        if preds is not None:
            cols = ["away", "home", "market_margin", "model_margin", "fair_margin", "spread_side",
                    "spread_edge_pts", "market_total", "model_total", "total_side", "total_edge_pts", "qb_flag"]
            st.dataframe(preds[[c for c in cols if c in preds]].round(2), width="stretch", hide_index=True)
            st.subheader("Power ratings (model vs market-implied)")
            st.dataframe(pd.DataFrame({"model": ratings, "market": mkt_ratings}).round(1)
                         .sort_values("model", ascending=False), width="stretch")

# -------- props (manual scan only; never auto-fetched) --------
with tab5:
    st.subheader("Player props — manual scan")
    if league != "nfl":
        st.info("Props are NFL-only (projections come from nflverse player stats).")
    elif not ODDS_KEY:
        st.warning(NO_KEY)
    else:
        try:
            events, remaining, events_at = load_events(league)
        except Exception as e:
            st.error(f"Events list failed: {e}")
            events, remaining, events_at = pd.DataFrame(columns=["event_id", "commence", "home", "away"]), None, 0.0
        scanned = st.session_state.get("props_odds")
        if scanned and scanned.get("remaining") is not None and scanned["at"] > events_at:
            remaining = scanned["remaining"]                   # the last billed call is fresher than the cached events call
        p_markets = st.multiselect("Prop markets", PROP_MARKETS_ALL, PROP_MARKETS_DEFAULT)
        pc1, pc2 = st.columns(2)
        hours = pc1.number_input("Hours ahead", 1, 240, 72, 12)
        credits = pc2.number_input("Max credits this scan", 1, 500, 60, 10)
        now = pd.Timestamp.now(tz="UTC")
        window = events[(events.commence > now) & (events.commence <= now + pd.Timedelta(hours=int(hours)))]
        cost = estimate_prop_credits(len(window), p_markets)
        st.caption(f"{len(window)} game(s) kick off within {int(hours)}h"
                   + (": " + ", ".join(f"{r.away}@{r.home}" for r in window.itertuples()) if len(window) else "")
                   + f" — {len(window)} x {len(p_markets)} markets x 1 region = {cost} credits"
                   + (f"; {remaining} left this month." if remaining is not None else "."))
        if st.button(f"Scan props (≈{cost} credits)", disabled=cost == 0):
            try:
                with st.spinner("Pulling props (one call per game)…"):
                    po = fetch_props(league, window, markets=p_markets, api_key=ODDS_KEY, max_credits=int(credits),
                                     remaining=remaining)
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
                show = pev.copy()
                show["ev_pct"] = (show.ev_pct * 100).round(2)
                show["kelly_stake"] = (show.kelly_stake * 100).round(2)
                show["p_win"] = (show.p_win * 100).round(1)
                show["fair_price"] = show.fair_price.round(0)
                show[["market_mu", "proj_mean", "fair_mu"]] = show[["market_mu", "proj_mean", "fair_mu"]].round(1)
                show["commence"] = kickoff(show.commence)
                st.dataframe(show[["commence", "matchup", "market", "player", "side", "line", "price", "book", "n_books",
                                   "market_mu", "proj_mean", "fair_mu", "fair_price", "p_win", "ev_pct", "kelly_stake",
                                   "flags"]]
                             .rename(columns={"ev_pct": "EV %", "kelly_stake": "stake % bankroll", "p_win": "win %"}),
                             width="stretch", hide_index=True)
            else:
                st.success("Nothing above your EV threshold in the last scan.")
            st.subheader("Prop arbs & middles")
            pm = board.get("mid")
            if pm is not None and len(pm):
                st.dataframe(middles_view(pm), width="stretch", hide_index=True)
                st.caption(MID_NOTE + " Prop limits are low.")
            else:
                st.caption("No cross-book prop arbs or +EV middles in the last scan. " + MID_NOTE)
            st.caption(f"Scanned {pd.Timestamp.fromtimestamp(raw['at']).strftime('%H:%M:%S')} server time; the board stays "
                       "until you scan again. Quota: each scan bills games x markets x regions credits from the free "
                       "500/month Odds API tier (the ≈N on the button), on top of the main board's 1 per refresh. "
                       f"Fair = {1 - model_w:.0%}/{model_w:.0%} devigged market consensus / projection (sidebar blend "
                       "weight); stakes are quarter-Kelly capped at 2%. 'no_market' rows have no two-way market behind "
                       "them: no EV, never staked, listed last. 'team_mismatch' rows are staked 0; 'no_proj' rows are "
                       "priced market-only. Props limits are low and books limit winners.")

st.caption(f"Last odds pull: {pd.Timestamp.fromtimestamp(fetched_at).strftime('%Y-%m-%d %H:%M:%S')} server time. "
           "Educational tooling; bet responsibly and only where legal.")

# -------- auto refresh --------
if auto:
    from streamlit_autorefresh import st_autorefresh
    remaining = max(REFRESH_MIN * 60 - (time.time() - fetched_at), 15)
    st_autorefresh(interval=int(remaining * 1000), key="odds_refresh")
    st.sidebar.caption(f"Next refresh in ~{remaining/60:.0f} min")
