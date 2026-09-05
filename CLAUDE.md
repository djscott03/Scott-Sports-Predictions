# CLAUDE.md — project context for Claude Code

Repo: `djscott03/Scott-Sports-Predictions` (public). Local checkout lives at
`~/Desktop/Claude/Sports Bets/`. Originally designed in a claude.ai conversation on
2026-09-03/04, verified, tested and pushed from Claude Code on 2026-09-04.

## What this is
`sharpmodel` — an NFL / college football betting model with a multi-book +EV scanner,
a Streamlit dashboard, and a frozen weekly prediction record in `predictions/` that a
GitHub Action publishes and grades automatically.

## Layout
```
sharpmodel/data.py         nflverse (no key) + CollegeFootballData (CFBD_API_KEY) loaders
sharpmodel/ratings.py      weighted ridge power ratings (MarginModel) + off/def PointsModel
sharpmodel/pricing.py      cover probs w/ push handling, devig, EV, fractional Kelly
sharpmodel/adjustments.py  rest, wind/temp, QB-change flags, manual injury hook
sharpmodel/engine.py       SharpModel.fit_as_of / price_games / predict_week / backtest, BetTracker (CLV)
sharpmodel/odds.py         The Odds API (ODDS_API_KEY) + CSV ingest, sharp-book fair, find_ev, best_lines, find_arbs
run.py                     CLI: backtest | predict | ev
publish_card.py            auto-detect week -> predictions/<league>/<season>_wNN.{md,csv} + graded README index
app.py                     Streamlit dashboard (tabs: +EV, arbs, best lines, model card)
tests/                     offline pytest (15 tests, ~1s); conftest chdir's to repo root
.github/workflows/ci.yml           pytest on push/PR (python 3.12)
.github/workflows/weekly-card.yml  cron Tue+Thu 13:00 UTC + manual dispatch; commits predictions/
```

## Key design decisions (don't undo these)
- **Market is the prior.** `fair = 0.70*market + 0.30*model` for NFL. Walk-forward
  backtest 2019–2025 (1,960 games) showed the ratings model alone is 51.1% ATS at the
  close = does NOT beat the market. Edge comes from odds.py (stale lines across books)
  and adjustments.py, not from the ratings. Do not "improve" the backtest by tuning
  in-sample; hold out 2023–2025.
- Moneylines are priced from the sharp book's own ML, not derived from the spread.
- Games with an unpriced QB change get stake forced to 0 (`qb_flag`).
- `predictions/` is a frozen record. `publish_card.py` will re-price a week (fresher
  lines) only while *none* of its games have kicked off; once any game has a result the
  existing card is kept and only the graded index is regenerated. Never commit a week
  that was priced *after* its games were played (that's a backtest, not a prediction).
- Odds API free tier = 500 req/month → `REFRESH_MIN >= 20` in the dashboard; each
  refresh is one request regardless of viewers.
- Odds API team names → nflverse abbreviations via `NFL_NAMES` in odds.py; CFB uses
  longest-prefix match against CFBD school names.

## Verified state (2026-09-04, local .venv on python 3.9; CI uses 3.12)
- `python -m pytest -q tests` → 15 passed.
- `python run.py nfl backtest 2019 2025` → n=1960, MAE market 9.82 / model 10.29 / blend 9.86,
  323 spread bets 51.1% ATS −2.0% ROI, 36 totals 55.6% +6.1%.
- `python run.py nfl predict 2026 1` prices the real Week 1 card from live nflverse lines.
- `python publish_card.py` wrote `predictions/nfl/2026_w01.{md,csv}` (16 games, 3 plays).
- Grading path verified by publishing a finished 2025 week and confirming W-L-P/units, then deleted.
- `app.py` passes `streamlit.testing.v1.AppTest` with no key (shows the warning, no exceptions).
- `fetch_odds` has NOT been hit against the live Odds API yet (no key). First real call:
  sanity-check `parse_odds_json` field names against the v4 response (a unit test covers the
  documented shape).

## Backlog (owner's roadmap, rough priority)
1. Add `ODDS_API_KEY` + `CFBD_API_KEY` as repo secrets; deploy app.py to Streamlit Cloud (DEPLOY.md)
2. Telegram/Discord alert on new +EV line (Option B in DEPLOY.md)
3. Run the same backtest on CFB (needs CFBD_API_KEY)
4. QB starter-vs-backup point-value table; auto-apply instead of flag
5. Opening-line capture → real CLV tracking via BetTracker
6. Empirical key-number margin distribution to replace the Normal
7. Derivatives (1H, team totals) off PointsModel

## Conventions
- pandas/numpy/scipy only in the core; no heavy ML deps. Must import on 3.9+ (all modules
  use `from __future__ import annotations`).
- Keep `sharpmodel/` importable without Streamlit; `app.py` is the only UI dependency.
- All odds in American format; all margins are home-minus-away.
- Owner has done GitHub projects with Claude before; comfortable with git basics, not a
  professional developer.
