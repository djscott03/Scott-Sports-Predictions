# Scott Sports Predictions

NFL & college football betting model with a frozen, auto-graded weekly prediction
record. Opponent-adjusted ratings, market-anchored blending, calibrated cover
probabilities, devigging, fractional Kelly, walk-forward backtesting, a multi-book
+EV scanner, and Closing Line Value tracking.

**This week's card → [`predictions/nfl/`](predictions/nfl/README.md)**
(published automatically every Tuesday and Thursday morning; each week is frozen at
publish time and graded once scores are final).

## Read this first (the honest part)

Walk-forward NFL backtest, 2019–2025, 1,960 games, betting only at the close:

| | MAE vs actual margin |
|---|---|
| Closing line | 9.82 |
| Ratings model alone | 10.29 |
| 70/30 market/model blend | 9.86 |

Spread bets flagged at ≥1.5 pt edge: **51.1% ATS, −2.0% ROI** (breakeven 52.4%).
Edge size shows no monotonic signal. **A ratings model does not beat NFL closing lines.**
Anyone selling you one that "does" is selling backtest overfitting.

That is not a failure of this code — it is the finding. Sharps win 2–4% ROI from:

1. **Beating the close, not the result.** Bet Sunday-night openers / soft books, track CLV.
2. **Information the ratings don't have yet** — QB changes, injuries, weather, rest,
   motivation. `adjustments.py` is where this lives; the model *flags* QB changes
   and refuses to stake them until you price the backup yourself.
3. **Softer markets** — CFB (especially G5/mid-week), totals, first-half, derivatives.
   The wind adjustment alone flipped totals from −17% to +6% ROI (36 bets — small
   sample, but the mechanism is real and well documented).
4. **Line shopping** — a half-point on 3 or 7 is worth ~2–3% of win probability.

Use this as the disciplined skeleton that stops you from betting noise, and plug
your information edge into the adjustment layer.

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -q tests                     # 15 offline tests, ~1s

python run.py nfl backtest 2019 2025          # walk-forward, prints metrics (~5s)
python run.py nfl predict 2026 1              # price a week's card from live nflverse lines
python run.py nfl predict 2026 1 --epa        # use play-by-play EPA (better ratings, slower)
python publish_card.py                        # auto-detect current week -> predictions/nfl/
export CFBD_API_KEY=...                       # free at collegefootballdata.com
python run.py cfb predict 2026 2
```

Data: NFL pulls live from nflverse (schedules, closing lines, moneylines, weather,
rest, listed QBs, play-by-play). CFB pulls games + consensus lines from CFBD. No keys
needed for NFL.

## Live dashboard
`streamlit run app.py` locally, or deploy free to Streamlit Community Cloud — see **DEPLOY.md**.

## Multi-book +EV scanner (`odds.py`)

```bash
export ODDS_API_KEY=...        # free tier at the-odds-api.com (500 req/mo; each scan = 1 request)
python run.py nfl ev 2026 1                 # every US/EU book, spreads + totals + ML
python run.py nfl ev 2026 1 --csv lines.csv # lines you captured yourself (see lines_template.csv)
python run.py nfl ev 2026 1 --nomodel       # pure market-vs-market, no model blend
```

How it prices a line:
1. Take the sharpest posted book (Pinnacle > Circa > BetOnline > Bookmaker; else median of all).
2. Devig its two-way price (power method) and invert into a fair number:
   `PHI -7 (-105/-105)` -> P(cover)=.50 -> fair margin 7.0. Moneylines get their own
   fair from the sharp ML, not from the spread.
3. Optionally nudge that fair toward your model (default weight 0.25).
4. Price **every** book's line — including alternate numbers like -6.5 or -7.5 — off
   that same distribution, so a `-6.5 -110` and a `-7 +100` are compared as probabilities.
5. Rank by EV%, size with quarter-Kelly, and separately flag true arbs.

Outputs: `ev_*.csv` (plays), `bestlines_*.csv` (line-shopping board), `odds_*.csv` (raw).

Realistic expectations: on a mature NFL Sunday you'll see 0–5 lines above +1.5% EV,
mostly stale numbers at recreational books after a sharp move. That is the whole
game — get down before they move. CFB mid-week and totals surface far more.
Books limit winners; spread action across accounts and don't hammer one book.

## Architecture

```
sharpmodel/
  data.py         nflverse + CFBD loaders -> unified schema; EPA per game
  ratings.py      MarginModel  (weighted ridge, recency half-life, margin cap,
                                preseason priors as pseudo-games, HFA prior)
                  PointsModel  (offense/defense split for totals)
  pricing.py      normal-with-key-number-pushes cover probs, devig (power method),
                  American odds math, EV, fractional Kelly with push handling
  adjustments.py  rest, wind/temp/roof, QB-change flags, manual injury hook
  odds.py         The Odds API / CSV ingestion, sharp-book devig + inversion,
                  +EV scan across books, best-line board, arbitrage finder
  engine.py       SharpModel: fit_as_of / price_games / predict_week / backtest
                  summarize_backtest, BetTracker (CLV report)
run.py            CLI: backtest | predict | ev
publish_card.py   freeze a week into predictions/<league>/ and grade past weeks
app.py            Streamlit dashboard
tests/            offline pytest suite (pricing math, ratings, walk-forward, adjustments)
.github/workflows/
  ci.yml          runs the tests on every push
  weekly-card.yml Tue/Thu cron: publish_card.py -> commits predictions/
```

### The number pipeline for one game
```
ratings margin  + situational adj  = model_margin
fair_margin     = 0.70 * market_margin + 0.30 * model_margin
P(cover)        = discretized Normal(fair_margin, sd=13.4) vs the line
stake           = ¼ Kelly on (P(cover) − breakeven), capped 3%, only if edge ≥ 1.5 pts
```

## Tunables (`LEAGUE_CFG` in engine.py)

| key | NFL | CFB | meaning |
|---|---|---|---|
| lam | 6 | 8 | ridge shrinkage; higher = more regression to mean |
| half_life | 9 | 10 | weeks until a game's weight halves |
| hfa_prior | 1.5 | 2.6 | home-field points prior |
| regress | 0.45 | 0.65 | preseason carryover of last year's rating |
| epa_weight | 0.5 | 0 | blend of EPA-implied margin into game results |
| market_weight | 0.70 | 0.65 | how much you trust the line vs your model |
| min_edge_pts | 1.5 | 2.5 | minimum point discrepancy to bet |

Re-tune with `backtest`, but hold out seasons: fit parameters on 2019–2022,
confirm on 2023–2025. If it only works in-sample, it doesn't work.

## Tracking real bets (the part that matters)

```python
from sharpmodel import BetTracker
t = BetTracker("bets.csv")
t.log("2026_01_DAL_NYG", "spread", "home", 2.5, -110, stake=0.02, model_fair=1.0)
# after kickoff:
t.set_close("2026_01_DAL_NYG", "spread", close_line=1.5)
print(t.clv_report())    # avg_clv_pts > 0 and pct_beat_close > 55% => you are sharp
```

If your CLV is positive over 300+ bets you will win long-run even through losing months.
If it is negative, no amount of "the model says" will save the bankroll.

## Roadmap to actual edge (in rough order of value)

1. **CFB first.** Softer lines, 130 teams, mid-week MACtion. Run the same backtest.
2. **QB pricing table** — starter vs backup point values; auto-apply instead of flagging.
3. **Injury feeds** — OL/CB/EDGE losses worth 0.5–2 pts; scrape official reports Wed–Fri.
4. **Openers vs closers** — capture opening lines (The Odds API), bet the open, measure CLV.
5. **Derivatives** — 1H/team totals priced off `PointsModel`; books are lazier there.
6. **Empirical margin distribution** — replace the Normal with a key-number histogram
   conditional on the spread (matters most for buying/selling half-points).
7. **Drive-level EPA model** for CFB via CFBD `/plays` (same code path as NFL EPA).

## Disclaimer
Educational tooling. Sports betting carries real financial risk; most bettors lose.
Bet only what you can afford to lose and only where it is legal for you.
