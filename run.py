"""
Usage:
  python run.py nfl backtest 2019 2025          # walk-forward backtest
  python run.py nfl predict 2026 1              # price this week's card
  python run.py cfb predict 2026 2              # needs CFBD_API_KEY
  python run.py nfl ev 2026 1                   # scan every book for +EV lines (needs ODDS_API_KEY)
  python run.py nfl ev 2026 1 --csv lines.csv   # ...or from lines you captured yourself
  python run.py nfl ev 2026 1 --nomodel         # pure sharp-book devig, no model blend
  python run.py nfl props 2026 1                # player props: free events list -> credit estimate -> per-event fetch
  python run.py nfl props 2026 1 --markets player_pass_yds,player_anytime_td --hours 48 --credits 30
  python run.py nfl props 2026 1 --csv props_template.csv   # ...or props you captured yourself (0 credits)
  python run.py nfl props 2026 1 --nomodel      # market consensus only, no projection blend
Add --epa to use play-by-play EPA (NFL only, slower first download).
"""
import sys, json
import pandas as pd
from sharpmodel import SharpModel, summarize_backtest, load_nfl, load_cfb
from sharpmodel.odds import fetch_odds, load_odds_csv, find_ev, best_lines, find_arbs

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

OPTS = ("--csv", "--markets", "--hours", "--credits")       # flags that take a value (their value is not a season/week)
opt = lambda flag, default=None: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

league, mode = sys.argv[1], sys.argv[2]
epa = "--epa" in sys.argv
args = [int(a) for i, a in enumerate(sys.argv) if i > 2 and a.isdigit() and sys.argv[i - 1] not in OPTS]

if mode == "backtest":
    s0, s1 = args
    seasons = list(range(s0 - 1, s1 + 1))         # extra season for priors
    hist = load_nfl(seasons, with_epa=epa) if league == "nfl" else load_cfb(seasons)
    m = SharpModel(league)
    bt = m.backtest(hist, list(range(s0, s1 + 1)), start_week=1)
    bt.to_csv(f"backtest_{league}.csv", index=False)
    print(json.dumps(summarize_backtest(bt, m.sd["spread"]), indent=2, default=float))

elif mode == "predict":
    season, week = args
    hist = load_nfl([season - 1, season], with_epa=epa) if league == "nfl" else load_cfb([season - 1, season])
    m = SharpModel(league)
    wp = m.predict_week(hist, season, week)
    p = wp.preds
    print("\n=== POWER RATINGS (model | market-implied) ===")
    print(pd.DataFrame({"model": wp.ratings, "market": wp.market_ratings}).round(1).sort_values("model", ascending=False).head(40))
    print(f"\nHFA: {m.margin_model.hfa:.2f}")
    cols = ["away", "home", "market_margin", "model_margin", "fair_margin",
            "spread_side", "spread_line", "spread_p", "spread_edge_pts", "spread_stake",
            "market_total", "model_total", "total_side", "total_edge_pts", "total_stake"]
    print("\n=== CARD ===")
    print(p[[c for c in cols if c in p]].round(2).to_string(index=False))
    bets = p[(p.get("spread_stake", 0) > 0) | (p.get("total_stake", 0) > 0)]
    print(f"\n{len(bets)} flagged plays. Stakes are fraction of bankroll (quarter-Kelly, 3% cap).")
    p.to_csv(f"card_{league}_{season}_w{week}.csv", index=False)

elif mode == "ev":
    season, week = args
    csv = opt("--csv")
    model_fair = None
    known = None
    if league == "cfb":
        hist = load_cfb([season - 1, season]); known = sorted(set(hist.home) | set(hist.away))
    if "--nomodel" not in sys.argv:
        hist = load_nfl([season - 1, season], with_epa=epa) if league == "nfl" else hist
        wp = SharpModel(league).predict_week(hist, season, week)
        model_fair = wp.preds[["home", "away", "model_margin", "model_total"]]
    odds = load_odds_csv(csv) if csv else fetch_odds(league, known_teams=known)
    odds.to_csv(f"odds_{league}_{season}_w{week}.csv", index=False)
    print(f"{odds.event_id.nunique()} games, {odds.book.nunique()} books, {len(odds)} prices")
    ev = find_ev(odds, league, model_fair=model_fair, model_weight=0.25, min_ev=0.015)
    cols = ["matchup", "market", "team", "line", "price", "book", "ref_book", "fair_price", "p_win", "ev_pct", "kelly_stake"]
    print("\n=== +EV LINES (vs sharp-book fair, sorted by EV) ===")
    print(ev[cols].round(3).to_string(index=False) if len(ev) else "none above threshold")
    arbs = find_arbs(odds)
    if len(arbs): print("\n=== ARBS ===\n", arbs.round(4).to_string(index=False))
    ev.to_csv(f"ev_{league}_{season}_w{week}.csv", index=False)
    best_lines(odds).to_csv(f"bestlines_{league}_{season}_w{week}.csv", index=False)

elif mode == "props":
    from sharpmodel.odds import fetch_events, fetch_props, load_props_csv, estimate_prop_credits, PROP_MARKETS_DEFAULT
    from sharpmodel.props import load_player_weeks, project_players, fit_dispersion, price_props
    if league != "nfl": sys.exit("props: NFL only (projections come from nflverse player stats)")
    season, week = args
    csv, markets = opt("--csv"), opt("--markets", ",".join(PROP_MARKETS_DEFAULT)).split(",")
    hours, credits = float(opt("--hours", 72)), int(opt("--credits", 60))
    hist = load_nfl([season - 1, season])
    if csv:
        odds = load_props_csv(csv)
    else:
        now = pd.Timestamp.now(tz="UTC")
        events = fetch_events(league)                                          # free endpoint
        events = events[(events.commence > now) & (events.commence <= now + pd.Timedelta(hours=hours))]
        cost = estimate_prop_credits(len(events), markets)
        print(f"{len(events)} games kick off within {hours:.0f}h x {len(markets)} markets x 1 region "
              f"= {cost} credits (cap --credits {credits}; free tier 500/month)")
        for r in events.itertuples(): print(f"  {r.away} @ {r.home}  {r.commence:%a %m-%d %H:%M} UTC")
        odds = fetch_props(league, events.event_id, markets=markets, max_credits=credits)
    odds.to_csv(f"props_odds_{league}_{season}_w{week}.csv", index=False)
    print(f"{odds.event_id.nunique()} games, {odds.book.nunique()} books, {len(odds)} prop prices")
    projections = None
    if "--nomodel" not in sys.argv:
        stats = load_player_weeks([season - 2, season - 1, season])           # current season 404s until it starts
        fit_dispersion(stats, as_of=(season, week))                           # walk-forward cv per yardage market
        projections = project_players(stats, hist, as_of=(season, week))     # full hist -> env_factor gets team ppg
    ev = price_props(odds, projections, model_weight=0.30, min_ev=0.03)
    cols = ["matchup", "market", "player", "side", "line", "price", "book", "n_books", "market_mu", "proj_mean",
            "fair_mu", "fair_price", "ev_pct", "kelly_stake", "flags"]
    blend = ", 70/30 market/projection" if projections is not None else ""
    print(f"\n=== +EV PROPS (vs devigged market consensus{blend}, sorted by EV) ===")
    print(ev[cols].round(3).to_string(index=False) if len(ev) else "none above threshold")
    if len(ev): print("\nkelly_stake is fraction of bankroll (quarter-Kelly, 2% cap); 'no_market' rows are never staked.")
    (ev if len(ev) else pd.DataFrame(columns=cols)).to_csv(f"props_{league}_{season}_w{week}.csv", index=False)
