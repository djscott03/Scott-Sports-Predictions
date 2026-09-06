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

College is *worse*, not better. Walk-forward CFB backtest, 2021–2025, 3,944 FBS games
(run 2026-09-05 via the `backtest` Action with a CFBD key):

| | MAE vs actual margin |
|---|---|
| Closing line | 12.11 |
| Ratings model alone | 14.22 |
| 65/35 market/model blend | 12.41 |

Flagged spread bets (≥2.5 pt edge): **50.4% ATS, −3.7% ROI** over 1,600 bets; totals
51.2%, −2.3% over 219. The CFB ratings sit further from the market than the NFL ones and
the edge threshold flags ~40% of games — a Week 1 card "liked" 32 of 51 games, which is
noise, not signal. That is why CFB cards are manual-dispatch only and `LEAGUE_CFG` was not
re-tuned on this result (that would be in-sample fitting). CFB edge has to come from the
odds screen — stale lines, arbs, middles — not from ratings.

That is not a failure of this code — it is the finding. Sharps win 2–4% ROI from:

1. **Beating the close, not the result.** Bet Sunday-night openers / soft books, track CLV.
2. **Information the ratings don't have yet** — QB changes, injuries, weather, rest,
   motivation. `adjustments.py` is where this lives; the model *flags* QB changes
   and refuses to stake them until you price the backup yourself.
3. **Softer markets** — CFB (especially G5/mid-week), totals, first-half, derivatives.
   The wind adjustment alone flipped totals from −17% to +6% ROI (36 bets — small
   sample, but the mechanism is real and well documented). "Softer" means more stale
   numbers across books, not that a ratings model beats the CFB close — see above.
4. **Line shopping** — a half-point on 3 or 7 is worth ~2–3% of win probability.

Use this as the disciplined skeleton that stops you from betting noise, and plug
your information edge into the adjustment layer.

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -q tests                     # 79 offline tests, ~6s

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
5. Rank by EV%, size with quarter-Kelly. Arbs and middles are a separate pass (next section).

Outputs: `ev_*.csv` (plays), `bestlines_*.csv` (line-shopping board), `middles_*.csv`
(arbs & middles), `odds_*.csv` (raw).

No laptop needed: the **`ev scan`** Action (GitHub → Actions → ev scan → Run workflow;
pick league/season/week) runs the same command on GitHub's runner with the repo's
`ODDS_API_KEY` secret and writes the +EV table and the arbs & middles table into the run's
job summary, with the CSVs as a downloadable artifact. Manual only — every run is 1 credit.

Realistic expectations: on a mature NFL Sunday you'll see 0–5 lines above +1.5% EV,
mostly stale numbers at recreational books after a sharp move. That is the whole
game — get down before they move. CFB mid-week and totals surface far more.
Books limit winners; spread action across accounts and don't hammer one book.

## Arbs & middles (cross-book)

The owner's example: one book has Hurts passing yards **O 280.5 +100**, another still has
**U 315.5 +100**. Bet both, half the stake on each:

| Hurts throws for | O 280.5 | U 315.5 | net |
|---|---|---|---|
| 281–315 (the window) | wins +50 | wins +50 | **+100** |
| 316 or more | wins +50 | loses −50 | 0 |
| 280 or fewer | loses −50 | wins +50 | 0 |

You cannot lose (`free_middle`), and you double up whenever he lands inside. At the usual
−110/−110 the two split outcomes cost about 4.5% of the total stake (`miss_cost_pct`) and the
window pays ~91% (`win_both_pct`), so the middle has to land more than
4.5 / (90.9 + 4.5) ≈ 4.8% of the time to be +EV (`breakeven_p`). A 35-yard window on a
QB whose game-to-game sd is ~80 yards lands roughly one time in six; `middles.py` prices
it off the same fair distribution the props board used and reports the exact EV.

Why the empirical margin distribution matters: a `PHI −2.5 / DAL +3.5` pair wins both legs
only on a 3-point home win. A Normal(3, 13) says that is ~3% — a *losing* middle at
−110/−110 (−1.7%). NFL games actually land on the favourite's 3 about 9% of the time
(`margins.py`, fitted on 1999–2025 nflverse results, ~8% conditional on the spread), which
makes the same pair worth about +3%. NFL spreads and moneylines are priced off that
key-number pmf, totals and CFB off a discretized Normal, props off the Normal / Poisson each
board row was priced with. The market is still the prior: game fairs are the sharp book's
number (nudged toward the model exactly as the +EV scanner does), prop fairs the devigged
multi-book consensus; a projection alone never prices a middle.

```bash
python run.py nfl ev 2026 1          # prints === ARBS & MIDDLES === after the +EV table -> middles_nfl_2026_w1.csv
python run.py nfl props 2026 1       # prints === PROP ARBS & MIDDLES === after the board -> props_middles_nfl_2026_w1.csv
```

The dashboard's **Arbs & middles** tab shows the game-line table; the **Props** tab lists prop
middles under the board (priced from the last scan — no extra credits).

Columns: `type` — `arb` (cannot lose, no window), `arb+middle` (cannot lose *and* a window),
`free_middle` (worst case zero, both win inside the window), `middle` (+EV; a miss costs
`miss_cost_pct`), `half_middle` (no both-win number, but one — shown as `3p` — where one leg
wins and the other pushes, e.g. `−2.5 / +3`); `bet_a` / `bet_b` — the two legs, book included;
`window` — the integers where both legs win; `p_middle` — probability of landing inside;
`win_both_pct`, `miss_cost_pct`, `guaranteed_pct` (the worst outcome), `ev_pct` (exact
expectation over every outcome, pushes included), `breakeven_p`, `stake_a_pct` / `stake_b_pct`
(split so a miss costs the same whichever side wins). Every `_pct` is percent of the *total*
stake across both legs. Sorted: pairs that cannot lose first, then middles by EV; scalps
(no window, can lose) are dropped.

Caveats (this is where the money leaks):
- Lines move within minutes of a sharp move. Place the worse-priced leg (usually the stale
  one) first, confirm it is accepted, then the other. If the second number is gone you are
  holding one bet you did not pick for its own EV — check the +EV board before you do.
- Books void obvious errors (palps) and limit accounts that only ever bet stale numbers.
- Same-book pairs are excluded by default (`include_same_book=True` shows them, flagged): a
  book that posts both sides of a middle is not going to let you keep it.
- Prop limits are low, so a prop middle is beer money, not a bankroll plan, and the sd behind
  `p_middle` still carries projection error (see the props caveats).
- Integer numbers push and NFL moneyline ties refund both legs; the outcome table handles both
  (a refund is never counted as a loss); the pmf gives a tie ~0.3% (reality ~0.2%).

## Player props (NFL, free tier)

Same philosophy, smaller market: the devigged multi-book consensus is the prior, a usage
projection nudges it, and a stale number at one book is the edge.

```bash
export ODDS_API_KEY=...
python run.py nfl props 2026 1                    # free events list -> credit estimate -> one call per game -> board
python run.py nfl props 2026 1 --markets player_pass_yds,player_anytime_td --hours 48 --credits 30
python run.py nfl props 2026 1 --csv props_template.csv   # props you captured yourself: 0 credits
python run.py nfl props 2026 1 --nomodel          # market consensus only, no projection
python run.py nfl props 2026 1 --weight 0.15      # projection blend weight (default 0.30)
```

How it prices a prop (`props.py`):
1. **Projection** = recency-weighted per-game usage (nflverse weekly stats, half-life 4 games,
   shrunk toward the position's *starter* mean) × opponent factor (stat allowed to that
   position vs league, half-shrunk to 1) × game environment (√ of market-implied team total
   over the team's own scoring). Yardage is Normal with a walk-forward-fitted cv — fitted on
   established starters only (≥ 8 prior games, inside the starter population), currently
   ≈0.35 passing (the floor), ≈0.6 rushing, ≈0.65 receiving — receptions and TD passes Poisson,
   anytime TD Bernoulli.
2. **Market consensus**: every book posting both sides of the same number is devigged and
   inverted to an implied mean; the median across books is `market_mu` (player names are
   normalised, so `A.J. Brown` at one book and `AJ Brown` at another are one consensus).
   Yes-only anytime-TD prices are power-devigged against a synthetic `no` priced at a 7% hold,
   so longshots take most of the vig the way they do in a real two-way market.
3. `fair_mu = (1 − w) × market_mu + w × proj_mean`, `w` = `--weight` / the sidebar blend
   slider, default 0.30. A player with a projection but no two-way market is shown flagged
   `no_market` with **no EV and stake 0**, listed last — **a projection alone is never a bet**.
   `team_mismatch` rows (the name matched a projection on another team) keep their EV but are
   staked 0: the opponent and environment factors belong to the wrong team.
4. Every posted line at every book is priced off that distribution (integer lines push),
   ranked by EV, sized quarter-Kelly capped at 2%. Each row carries the distribution it was
   priced with (`dist`, `fair_mu`, `fair_sd`) so it can be rebuilt downstream.

Outputs: `props_nfl_<season>_w<week>.csv` (plays), `props_middles_nfl_<season>_w<week>.csv`
(cross-book prop arbs & middles), `props_odds_nfl_<season>_w<week>.csv` (raw).
The dashboard's **Props** tab does the same thing behind a button that shows the credit
estimate; the `props scan` Action (manual dispatch) writes the board to the job summary.

**Quota math.** Props live on a per-event endpoint that bills `markets × regions` credits per
game (the game-line scan above is 1 credit total). Cost = games × markets × regions; the free
tier is 500/month. Defaults — 4 markets, US region, games in the next 72 h — run ≈ 4 × ~6
games ≈ 24 credits per scan (a full 16-game window would be 64). `fetch_props` refuses up
front when the estimate exceeds `--credits`, or when the credits the free events call reports
minus one game's cost would already dip under the 50-credit reserve, and stops early if the
API reports fewer than 50 left mid-scan. A game whose call fails is skipped and reported
(`[props] skip DAL@PHI: 500`); the games already billed stay on the board, and a 401/402/429
(bad key, out of credits, rate-limited) stops the scan instead of burning through the rest.
That is why the scan is **manual only**: no cron, no auto-refresh, the Action is
`workflow_dispatch`-only, and the Props tab spends nothing until you click — and what it
spends lands in `session_state` before anything else runs, so a rerun cannot lose it.

Caveats:
- Prop limits are low and books cut winners off fast. This is a supplement, not a bankroll plan.
- The projection regresses stars toward a starter mean, so it tends to sit *below* the
  market on big passing/receiving lines. A board that is all unders on stars is telling you
  about the projection, not the market — lower `--weight` (the sidebar blend slider in the
  dashboard) or use `--nomodel`.
- The fitted sd still includes the projection's own error on top of true game-to-game
  variance, so the tails are a little fat around a market line and EV% understates a stale
  number. Treat it as a conservative *ranking*, not a calibrated edge.
- Projection-only rows (`no_market`) are never staked; fuzzy name matches are flagged
  (`fuzzy_name`, `team_mismatch`) — check them before betting.
- Not backtested against closing prop lines (no free history exists).

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
  odds.py         The Odds API / CSV ingestion, sharp-book devig + inversion (blended_fair),
                  +EV scan across books, best-line board, legacy same-number arb finder;
                  player-prop ingestion (free events list, credit estimate, per-event fetch)
  margins.py      empirical NFL margin pmf: discretized Normal re-weighted at the key numbers
                  (fitted on 1999-2025 results; opt-in, the weekly card keeps the Normal)
  middles.py      cross-book arbs & middles: game_fairs / prop_fairs -> find_middles prices
                  every opposite-side pair at two books on the fair pmf's integer support
  props.py        nflverse weekly player stats -> usage projections (EW x opponent x
                  environment), Normal/Poisson/Bernoulli prop distributions, devigged
                  multi-book consensus, 70/30 blend, +EV prop board
  engine.py       SharpModel: fit_as_of / price_games / predict_week / backtest
                  summarize_backtest, BetTracker (CLV report)
run.py            CLI: backtest | predict | ev (+EV, arbs & middles) | props (board, prop arbs & middles)
holdout.py        fit the blend weight on early seasons, confirm on held-out ones
publish_card.py   freeze a week into predictions/<league>/ and grade past weeks
app.py            Streamlit dashboard (+EV, arbs & middles, best lines, model card, props)
lines_template.csv / props_template.csv   schemas for --csv (hand-captured lines / props)
tests/            offline pytest suite (pricing math, ratings, walk-forward, adjustments,
                  props projections + pricing, prop ingestion, key-number pmf, arbs & middles,
                  grading, holdout, dashboard AppTest)
.github/workflows/
  ci.yml          runs the tests on every push
  weekly-card.yml Tue/Thu cron: publish_card.py -> commits predictions/
  backtest.yml    manual walk-forward backtest -> job summary
  props-scan.yml  manual-only prop scan -> job summary + CSV artifact (never scheduled: quota)
  ev-scan.yml     manual-only +EV / arbs & middles scan (1 credit) -> job summary + CSV artifact
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
6. **Empirical margin distribution** — done for middles (`margins.py`, unconditional key-number
   weights); next: condition on the spread and use it for half-point buys/sells on the card.
7. **Drive-level EPA model** for CFB via CFBD `/plays` (same code path as NFL EPA).

## Disclaimer
Educational tooling. Sports betting carries real financial risk; most bettors lose.
Bet only what you can afford to lose and only where it is legal for you.
