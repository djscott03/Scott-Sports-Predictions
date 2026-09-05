"""
Live +EV dashboard.  Run locally:  streamlit run app.py
Deploy free:  push repo to GitHub -> share.streamlit.io -> New app -> add secrets.

Refresh strategy: odds are cached for REFRESH_MIN minutes and re-fetched on the
next page load/auto-refresh after that. This spends 1 API request per refresh,
not per viewer, so the free 500/month Odds API tier survives a season if you
keep REFRESH_MIN >= 20 and only leave the tab open on game days.

Player props (Props tab) are never fetched automatically: the per-event endpoint bills
games x markets per scan, so the tab shows the estimate on the button and only spends
credits on click. The result lives in session_state until the next scan.
"""
import os, time
import pandas as pd
import streamlit as st

from sharpmodel import SharpModel, load_nfl, load_cfb, props
from sharpmodel.odds import (fetch_odds, find_ev, best_lines, find_arbs, parse_odds_json, fetch_events, fetch_props,
                             estimate_prop_credits, PROP_MARKETS_DEFAULT, PROP_MARKETS_ALL)
from sharpmodel.props import load_player_weeks, project_players, fit_dispersion, price_props

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
model_w = st.sidebar.slider("Model blend weight", 0.0, 0.6, 0.25, 0.05)
markets = st.sidebar.multiselect("Markets", ["spreads", "totals", "ml"], ["spreads", "totals", "ml"])
books_excl = st.sidebar.text_input("Exclude books (comma sep)", "")
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

@st.cache_data(ttl=60 * 30, show_spinner="Listing upcoming games (free endpoint)…")
def load_events(league):
    return fetch_events(league, api_key=ODDS_KEY)

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

# -------- main --------
odds, fetched_at = load_odds(league, season)
age = (time.time() - fetched_at) / 60
st.title(f"{league.upper()} +EV board")
if odds.empty:
    st.warning(NO_KEY)
else:
    excl = [b.strip() for b in books_excl.split(",") if b.strip()]
    odds = odds[odds.market.isin(markets) & ~odds.book.isin(excl)]
    try:
        preds, ratings, mkt_ratings = model_card(league, season, week)
        model_fair = preds[["home", "away", "model_margin", "model_total"]]
    except Exception as e:
        preds, ratings, mkt_ratings, model_fair = None, None, None, None
        st.info(f"Model unavailable ({e}); showing pure market-vs-market EV.")

    ev = find_ev(odds, league, model_fair=model_fair, model_weight=model_w, min_ev=min_ev)
    arbs = find_arbs(odds)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games", odds.event_id.nunique())
    c2.metric("Books", odds.book.nunique())
    c3.metric("+EV lines", len(ev))
    c4.metric("Odds age", f"{age:.0f} min", help=f"Refreshes every {REFRESH_MIN} min")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["+EV plays", "Arbs", "Best lines", "Model card", "Props"])
if not odds.empty:
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
        st.dataframe(arbs.assign(profit_pct=lambda d: (d.profit_pct * 100).round(2)) if len(arbs) else
                     pd.DataFrame({"status": ["no arbs"]}), width="stretch", hide_index=True)
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
            events = load_events(league)
        except Exception as e:
            st.error(f"Events list failed: {e}"); events = pd.DataFrame(columns=["event_id", "commence", "home", "away"])
        p_markets = st.multiselect("Prop markets", PROP_MARKETS_ALL, PROP_MARKETS_DEFAULT)
        pc1, pc2 = st.columns(2)
        hours = pc1.number_input("Hours ahead", 1, 240, 72, 12)
        credits = pc2.number_input("Max credits this scan", 1, 500, 60, 10)
        now = pd.Timestamp.now(tz="UTC")
        window = events[(events.commence > now) & (events.commence <= now + pd.Timedelta(hours=int(hours)))]
        cost = estimate_prop_credits(len(window), p_markets)
        st.caption(f"{len(window)} game(s) kick off within {int(hours)}h"
                   + (": " + ", ".join(f"{r.away}@{r.home}" for r in window.itertuples()) if len(window) else "")
                   + f" — {len(window)} x {len(p_markets)} markets x 1 region = {cost} credits.")
        if st.button(f"Scan props (≈{cost} credits)", disabled=cost == 0):
            try:
                with st.spinner("Pulling props (one call per game)…"):
                    po = fetch_props(league, window.event_id, markets=p_markets, api_key=ODDS_KEY, max_credits=int(credits))
                proj = None
                try:
                    proj, cv = props_projections(season, week); props._CV.update(cv)
                except Exception as e:
                    st.info(f"Projections unavailable ({e}); pricing market-only.")
                st.session_state["props"] = dict(ev=price_props(po, proj, model_weight=0.30, min_ev=-1.0), at=time.time(),
                                                 games=po.event_id.nunique(), books=po.book.nunique(), prices=len(po))
            except Exception as e:
                st.error(str(e))
        res = st.session_state.get("props")
        if res is None:
            st.info("Nothing scanned yet. Pick markets and a window, check the credit estimate on the button, then scan.")
        else:
            pev = res["ev"]
            pev = pev[pev.ev_pct >= min_ev] if len(pev) else pev
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games", res["games"]); m2.metric("Books", res["books"]); m3.metric("Prop prices", res["prices"])
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
            st.caption(f"Scanned {pd.Timestamp.fromtimestamp(res['at']).strftime('%H:%M:%S')} server time; the board stays "
                       "until you scan again. Quota: each scan bills games x markets x regions credits from the free "
                       "500/month Odds API tier (the ≈N on the button), on top of the main board's 1 per refresh. "
                       "Fair = 70/30 devigged market consensus / projection; stakes are quarter-Kelly capped at 2%. "
                       "'no_market' rows have no two-way market behind them and are never staked; 'no_proj' rows are "
                       "priced market-only. Props limits are low and books limit winners.")

st.caption(f"Last odds pull: {pd.Timestamp.fromtimestamp(fetched_at).strftime('%Y-%m-%d %H:%M:%S')} server time. "
           "Educational tooling; bet responsibly and only where legal.")

# -------- auto refresh --------
if auto:
    from streamlit_autorefresh import st_autorefresh
    remaining = max(REFRESH_MIN * 60 - (time.time() - fetched_at), 15)
    st_autorefresh(interval=int(remaining * 1000), key="odds_refresh")
    st.sidebar.caption(f"Next refresh in ~{remaining/60:.0f} min")
