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
sharpmodel/odds.py         The Odds API (ODDS_API_KEY) + CSV ingest, sharp_fair -> blended_fair (model nudge),
                           find_ev, best_lines, find_arbs (legacy same-number arbs; middles.py supersedes it),
                           within_hours, NON_US_BOOKS / US_BOOKS, odds_grid (the odds screen; masks=True also
                           returns the best / off-number boolean frames -- never put a DataFrame in .attrs:
                           pandas 3 compares attrs on every concat and raises 'truth value is ambiguous');
                           every HTTP call goes through _get -> OddsAPIError (status + body, never the URL/key);
                           player-prop ingest: fetch_events (free; attrs['remaining']) -> estimate_prop_credits ->
                           fetch_props (per event; skips failed games, attrs['failed'], stops on 401/402/429)
sharpmodel/margins.py      empirical NFL margin pmf: discretized Normal x key-number weights (NFL_KEY_WEIGHTS,
                           fitted 1999-2025), margin_pmf / cover_probs_emp / moneyline_prob_emp; OPT-IN
sharpmodel/middles.py      cross-book arbs & middles: game_fairs (blended_fair -> nfl_margin/normal) and
                           prop_fairs (price_props board -> its dist/fair_mu/fair_sd) -> find_middles: every
                           opposite-side pair at two books priced on the fair pmf's integer support; stakes
                           split so a miss costs the same either way; types arb / arb+middle / free_middle /
                           middle / half_middle, scalps dropped; *_pct columns are % of the TOTAL stake
sharpmodel/props.py        nflverse weekly player stats -> project_players (EW usage x opp x env), fit_dispersion
                           (starters only), prop_probs / implied_mean (normal/poisson/bernoulli), price_props
                           (devigged consensus by normalised name, blend weight, rows carry event_id + dist + fair_sd)
sharpmodel/history.py      line history + steam: append_snapshot(odds, league, folder, pulled_at) -> folder/<league>/
                           <YYYY-MM-DD>.parquet (NY date; read + concat + rewrite, COLS = pulled_at epoch + the board
                           columns), load_history(folder, league, days) -> one frame, moves(prev, curr, books=
                           SHARP_BOOKS, min_spread=1, min_total=1.5, min_ml_prob=0.03) -> one row per event/book/
                           market (home spread, over, home ML implied prob; delta positive = toward home / the over),
                           steam_lines(mv, names) -> '📈 STEAM · DAL @ PHI · Pinnacle PHI -2.5 -> -3.5 (+1.0) in 62 min'
run.py                     CLI: backtest | predict | ev (--csv --hours --exclude --weight/--nomodel; +EV table, then
                           ARBS & MIDDLES -> middles_*.csv) | props (--csv --markets --hours --credits --weight
                           --exclude --nomodel; board, then PROP ARBS & MIDDLES -> props_middles_*.csv);
                           --exclude takes book keys or 'nonus' (odds.NON_US_BOOKS)
alerts.py                  CLI: alerts.py <league> [--csv] [--hours 240] [--min-ev 2.0] [--min-middle-ev 1.0]
                           [--max-age 45] [--exclude nonus] [--state alerts_state.json] [--top 5] [--test] [--dry-run]
                           [--snapshot DIR] [--history DIR] [--prev PATH_OR_URL] [--top-steam 5].
                           One board pull -> snapshot (odds_<league>.csv now carries pulled_at) + history.append_snapshot
                           of the whole board -> scan (market-only, odds.fresh at --max-age) -> alertable items in
                           order: guaranteed_pct >= 0 pairs (always), middles >= --min-middle-ev %, the sharp books'
                           moves vs --prev (history.moves, informational, at most --top-steam), top picks >= --min-ev %
                           -> dedupe against the state JSON {key: first_seen_iso} (24 h TTL; key = pick|matchup|
                           market|team|line|book|price, mid|bet_a|bet_b or steam|matchup|market|book|to_line) ->
                           Discord (DISCORD_WEBHOOK) / Telegram (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID), chunked at
                           1900 chars, friendly BOOK_NAMES (copied from app.py: never import app.py, it imports
                           streamlit). Always prints the message and writes alerts_summary.md; state saved after
                           delivery succeeds (a failed webhook exits 1 with the state untouched); --test sends the
                           top pick and never writes. A history / prev failure prints '[alerts] history skipped: ..'
                           and never fails the run (a missing or 404 --prev is the first run).
holdout.py                 fit blend weight on early seasons, confirm on held-out ones
publish_card.py            auto-detect week -> predictions/<league>/<season>_wNN.{md,csv} + graded README index
app.py                     Streamlit dashboard, Cleveland theme (.streamlit/config.toml + CSS block). Tabs: top picks
                           (odds.top_picks: one row per pick at its best book, 'also' = other books on the number),
                           +EV, arbs & middles, odds screen [odds.odds_grid: row per pick x column per book, best
                           price GREEN / off-number RED (signal colours stay universal; the Cleveland palette is
                           chrome only), pinned Game/Pick], model card, props [button-gated;
                           prop middles + prop odds screen under the board]. Hero pills + 'Right now' cards (best
                           +EV line, best arb/middle, board status); price_board cached on sidebar settings +
                           fetched_at. SHARPMODEL_DEMO_ODDS=<odds_*.csv> runs it off a file (no key, no credits).
                           Sharing guards: SCAN_PIN (gates props scans + force refresh), MAX_CREDITS_PER_DAY (default
                           60; st.cache_resource budget counting each pull's x-requests-last, 3 at ODDS_BOOKS=core;
                           then the last board is served 'paused' till midnight NY).
                           Sidebar Filters: min EV, window, markets, Teams, Books (pretty names -> extra exclusions),
                           raw exclude box. Never override font-family globally: Material icons become their names.
lines_template.csv         --csv schema for game lines;  props_template.csv  --csv schema for props (both tracked)
tests/                     offline pytest (136 tests, ~20s; incl. AppTest smoke + subprocess runs of run.py ev/props
                           and alerts.py --dry-run / --test; test_alerts.py also parses alerts.yml and counts its
                           cron runs; test_history.py covers the day parquets, moves and steam text); conftest
                           chdir's to repo root
.github/workflows/ci.yml           pytest on push/PR (python 3.12)
.github/workflows/weekly-card.yml  cron Tue+Thu 13:00 UTC + manual dispatch; commits predictions/
.github/workflows/backtest.yml     manual dispatch; backtest -> job summary + csv artifact
.github/workflows/props-scan.yml   manual dispatch ONLY (quota); run.py nfl props -> job summary + csv artifact
.github/workflows/ev-scan.yml      manual dispatch ONLY (3 credits); run.py <league> ev -> +EV + arbs & middles
                                   in the job summary + csv artifact; never commits
.github/workflows/alerts.yml       board + alerts on a CONSERVATIVE UTC cron (Sun hourly 13-21 + Mon 00 for SNF,
                                   Thu 23 / Mon 23 for TNF/MNF pregame, Tue-Sat 16 = 17 runs x 3 credits = 51/week,
                                   ~220/month; a window past midnight is a separate cron line because day-of-week
                                   flips at 00:00 UTC) + manual dispatch (league, test, min_ev, hours, books). Every
                                   run: curl the previous board/odds_<league>.csv off the `board` branch (-> --prev)
                                   + git archive its history/ -> alerts.py --snapshot board --history history --prev
                                   prev_board.csv -> board/ + history/ committed ON TOP of the `board` branch (fetched,
                                   plain push, commit only when changed, orphan only on the very first run; the
                                   dashboard's free snapshot; permissions contents: write; never main) -> alerts to
                                   whichever channels have secrets; alerts_state.json rides actions/cache
                                   (restore-keys prefix alerts-state-, save if: always()); summary appended to the
                                   job summary; concurrency group
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
- **Credits = markets × regions, or markets × ceil(named books / 10).** The Odds API bills a
  game-line pull that way (docs re-read 2026-09-09; the first live pulls with the old us,us2,eu
  default cost 9 each: 500 → 491 → 474). `fetch_odds(books="core")` names `odds.CORE_BOOKS`
  (10 incl. Pinnacle, the sharp anchor) = 3 credits and records `attrs["cost"]` from
  x-requests-last; 'wide' = 20 books = 6. Never call it with regions unless you mean 9. Free
  tier = 500/month → `REFRESH_MIN >= 20` in the dashboard; each refresh is one pull (3 credits)
  regardless of viewers, capped by MAX_CREDITS_PER_DAY.
- Odds API team names → nflverse abbreviations via `NFL_NAMES` in odds.py; CFB uses
  longest-prefix match against CFBD school names.
- **Props are never fetched automatically.** The per-event endpoint bills games × markets ×
  regions per scan (vs 3 credits for the whole game-line board), so `fetch_props` refuses
  above `max_credits`, refuses up front when `remaining` (from the free events call) minus one
  call would breach the 50-credit reserve, and stops under the reserve mid-scan; run.py prints
  the estimate first, the Props tab shows it on the button, and props-scan.yml is
  `workflow_dispatch` only. Do not add a cron or an auto-refresh.
- **Billed data is never thrown away.** `fetch_props` keeps the games already fetched when a
  later call fails (`attrs['failed']`, 401/402/429 stop the loop), and the Props tab stores the
  raw frame in `session_state["props_odds"]` *before* projections/pricing run; the board is
  priced from there (re-priced only when the sidebar weight / EV threshold / week change).
- **The API key never reaches a log or the UI.** requests' exception messages embed the full
  URL (`?apiKey=...`), so every call goes through `odds._get`, which re-raises as
  `OddsAPIError("Odds API <status> error: <body>")`. Never print a raw requests exception.
- Props pricing keeps the same prior: `fair_mu = (1-w)*market_mu + w*proj_mean`, `w` = 0.30
  by default (`--weight`, sidebar slider). `no_market` rows (projection, no two-way market)
  get ev_pct NaN and stake 0 and sort last; `team_mismatch` rows keep EV but stake 0.
  Yes-only anytime-TD prices are power-devigged against a synthetic 'no' at a 7% hold
  (capped at 98%: the steepest 'no' a book posts, else the power method zeroes longshots).
- `fit_dispersion` fits cv on established starters only (>= 8 prior games, in the STARTERS
  population). The sd still carries projection error, so EV% is a conservative ranking.
- `price_props` rows carry `event_id`, `dist` and `fair_sd` (NaN for poisson/bernoulli) so
  `middles.prop_fairs` can rebuild the distribution — keep them. Build the fairs from a board
  priced with `min_ev=-1` (every two-way market), never from the +EV-filtered board alone.
- **Two margin distributions, on purpose.** engine.py / pricing.py / find_ev still price the weekly
  card and the +EV board off the plain Normal (sd 13.4): switching would re-price the frozen
  `predictions/` record. `middles.py` prices NFL spreads/ML off `margins.margin_pmf` (empirical
  key-number weights, sd 13.2) because a middle lives on single integers — 3 holds ~8% of the mass
  where the Normal says ~3%. Totals, CFB and props stay Normal / Poisson. Don't "unify" them.
- Middles: the market is still the prior (`game_fairs` = `blended_fair`, `prop_fairs` drops
  `no_market` rows). Same-book pairs are excluded by default; scalps (no window, can lose) are
  dropped; anything with `guaranteed_pct >= 0` (arbs, free middles) is always kept and sorted first.
  `ev_pct` is the exact sum over every outcome (pushes included); the `window` / `p_middle` /
  `win_both_pct` columns are the headline both-win numbers (a `half_middle` shows its push-win
  number as `3p` and that outcome's payoff instead). `find_middles(collapse=True)` (default)
  keeps one row per pair of numbers — the best-EV pairing — with the other book combinations in
  `n_alt` / `alt`; `exclude=` removes books as legs only (fairs are built by the caller from the
  full board, on purpose).
- **Actions and the dashboard scan market-only by default** (`weight` input 0 / sidebar blend 0,
  `exclude` `nonus`, kickoff window 240 h); the CLI keeps the documented 0.25 / 0.30 blend defaults.
  The live board with the model on was model-vs-market, which is not what the scanner is for.
- **Dashboard performance.** `price_board` is `st.cache_data`-keyed on the sidebar settings plus
  `fetched_at`, so slider changes are cache hits and only a new odds pull re-prices. `find_middles`
  with `collapse=True` pairs the best-priced book per NUMBER (top 3 per side so a same-book clash falls
  through), not every book x book: the full 272-game board prices in ~2 s instead of ~33 s. The first
  Streamlit Cloud deploy (2026-09-06) sat on "running" for that half-minute — that is what this fixes.
- In pandas use `df["flags"]`, never `df.flags` (built-in attribute shadows the column).
- **The site is a snapshot; the cron is the spend.** `BOARD_SOURCE=snapshot` (default): app.py reads
  `BOARD_URL` (raw.githubusercontent.com/<repo>/board/board/odds_<league>.csv + meta_<league>.json),
  which `alerts.py --snapshot board` writes and alerts.yml commits to the `board` branch (never main:
  Streamlit Cloud would redeploy). Viewers never pull; the owner PIN's "Pull fresh odds now" makes one
  budgeted live pull that wins while it is newer than the snapshot; `BOARD_SOURCE=live` is the old
  per-viewer mode. 17 cron runs/week x 3 credits ≈ 220/month.
- **Every pull is kept; steam is read from the last two.** A pull is billed, so `alerts.py --history DIR`
  appends the whole board to `DIR/<league>/<NY date>.parquet` (`history.append_snapshot`) and alerts.yml
  keeps `history/` on the `board` branch: fetched first (`git archive board history`) so the day file is
  appended to, then committed on top with a plain push -- the branch is no longer an orphan force-push, it
  grows by one small parquet a day (~17 x 2.5k rows a week), fine for years. `--prev` (the previous
  `board/odds_<league>.csv`, which carries `pulled_at`) feeds `history.moves` for `odds.SHARP_BOOKS`
  regardless of `--exclude` (the sharps are excluded as bets, not as a signal), thresholds 1 pt / 1.5 pt /
  3% implied, delta positive = toward home / the over (the margin convention), one move per event x book x
  market; the 📈 STEAM block is informational, sits between the middles and the picks, is capped by
  `--top-steam` after the dedupe and keyed on the number moved TO. Nothing in history may fail the alerts:
  every failure is one '[alerts] history skipped: ...' line. `load_history` is the opening-line / CLV feed
  (backlog item 5) -- never re-price the frozen card from it.
- **Alerts are one 3-credit pull per run and dedupe by content, not by time.** `alerts.py` keys a pick on
  matchup|market|team|line|book|price and a middle on its two `bet_a|bet_b` strings (raw book keys), so
  a re-price is a new alert and the same number is never sent twice within 24 h; the state is saved
  only after delivery succeeded (a dead webhook = exit 1, nothing remembered, re-sent next run) and
  `--test` never writes it. The state file lives in the Action's cache (`alerts-state-<run_id>`,
  restored by prefix), never in git. The cron is 17 runs x 3 = 51 credits/week on purpose: the owner will not pay
  for the API, so every scheduled line in alerts.yml has to justify itself in the header comment; do
  not add a `*/5` schedule. Book names come from `alerts.BOOK_NAMES`, a copy of app.py's table --
  importing app.py would pull streamlit onto the runner.

## Verified state (2026-09-05, local .venv on python 3.9; CI uses 3.12)
- `python -m pytest -q tests` → 95 passed in ~8s (15 original + props projections/pricing,
  prop ingestion + run.py subprocess runs, dashboard AppTest, publish grading, holdout,
  `tests/test_margins.py`, `tests/test_middles.py` incl. a subprocess run of run.py ev + props;
  the 2026-09-05 second review pass added 9: yes-only longshot bound, bad-body / transport-failure
  handling in fetch_props, attrs['remaining'], duplicate-index and blank-price safety in
  find_middles, empty-frame contract, tie mass).
- CFB walk-forward backtest (GitHub Action `backtest.yml`, 2026-09-05, CFBD key, 2021–2025,
  3,944 FBS games): MAE market 12.11 / model 14.22 / 65-35 blend 12.41; 1,600 flagged spread bets
  **50.4% ATS, −3.7% flat ROI**; 219 totals 51.2%, −2.3%. The CFB ratings model is further from
  the market than NFL's, and the min_edge threshold flags ~40% of games (a week-1 card had 32
  plays in 51 games). CFB cards therefore stay manual-dispatch only; CFB edge must come from
  odds.py / middles.py. LEAGUE_CFG was NOT re-tuned on this (in-sample).
- 2026-09-05 arbs & middles (`middles.py`, all offline-tested): spread −2.5/+3.5 at −110 with fair 3
  → window "3", p_middle = margin_pmf(3)[3] = .079, miss cost 4.55%, EV +3.0% ('middle'; the Normal
  says −1.7%); +100/+100 → 'free_middle'; ML +120/−105 → 'arb' +3.4%; totals 44.5/46.5 → "45-46";
  poisson receptions 4.5/6.5 → "5-6"; gap and same-book pairs skipped; −2.5/+3 → 'half_middle' "3p".
  `run.py ev --csv lines_template.csv --nomodel` prints one 'middle' (pinnacle O 47.5 / fanduel
  U 48.5, window 48, EV +0.5%) and writes `middles_nfl_2026_w1.csv`; `run.py props --csv
  props_template.csv --nomodel` prints the Hurts 264.5/274.5 middle and writes
  `props_middles_nfl_2026_w1.csv`. The Arbs & middles tab and the prop middles under the Props
  board are exercised by AppTest with faked feeds.
- 2026-09-05 props review fixes (all tested): key-safe `OddsAPIError`; per-event failure handling
  with `attrs['failed']`; consensus grouped by `normalize_player`; `load_props_csv` validates sides;
  Props tab stores the raw fetch in `session_state` before pricing and prices with the sidebar
  weight; yes-only TD devig vs synthetic 'no'; `point: null` -> NaN and NaN lines skipped;
  workflow inputs passed via env, `--markets` validated against `PROP_MARKETS_ALL`; `--weight`;
  `--csv`/`--nomodel` load nothing from nflverse; `team_mismatch` staked 0; `no_market` ev NaN,
  sorted last; reserve check before the first billed call; `dist` + `fair_sd` columns;
  `fit_dispersion` on starters only.
- `python run.py nfl backtest 2019 2025` → n=1960, MAE market 9.82 / model 10.29 / blend 9.86,
  323 spread bets 51.1% ATS −2.0% ROI, 36 totals 55.6% +6.1%.
- `python run.py nfl predict 2026 1` prices the real Week 1 card from live nflverse lines.
- `python publish_card.py` wrote `predictions/nfl/2026_w01.{md,csv}` (16 games, 3 plays).
- Grading path verified by publishing a finished 2025 week and confirming W-L-P/units, then deleted.
- `app.py` passes `streamlit.testing.v1.AppTest` with no key (shows the warning, no exceptions).
- **Live Odds API verified 2026-09-05** via the `ev scan` and `props scan` Actions (secrets set
  by the owner): `fetch_odds` -> 272 events / 35 books / 5,394 prices — 9 credits with the all-regions
  default of the time (since 2026-09-09 it names CORE_BOOKS: 3 credits, verify with attrs['cost']); `fetch_events`
  + `fetch_props` on 2 games x 4 markets cost exactly 4 per call (`x-requests-last`), 483 credits
  left afterwards. `parse_odds_json` / `parse_props_json` matched the real v4 shape. Findings that
  drove the same-day changes: (1) with `--weight 0.25` the +EV board was the model disagreeing
  with every book at once, so the Actions default to `--weight 0`; (2) one stale marathonbet ML
  produced 28 "arbs" -> `find_middles` now collapses to the best pairing per pair of numbers
  (`n_alt` / `alt`) and `--exclude nonus` (`odds.NON_US_BOOKS`) keeps EU/exchange books out of
  the legs and +EV rows while they still anchor the fairs; (3) the API returns the whole season,
  so `ev` has `--hours` (default 240 = a Tue-Mon slate; 168 from a Saturday missed Sunday). Real prop middle seen live: T. Ferguson rec yds O 19.5 /
  U 22.5 (window 20-22, +1.1%).
- Props: `python run.py nfl props 2026 1 --csv props_template.csv` runs end to end
  (nflverse stats 2024–2025 load in ~1s, 2026 404s and is skipped, 2 plays on the template's
  stale lines). Fitted cv as of 2026 w1 with the starters-only fit: pass 0.35 (floor) / rush 0.62 /
  rec 0.67, vs 0.42 / 0.77 / 0.81 when every player with >= 3 games was included (literature
  ≈0.30 / 0.55 / 0.65). `fetch_events` and the live per-event props endpoint have NOT been hit
  yet (no key); `parse_props_json` is verified only against the documented v4 shape. First real
  scan: use `--credits 10` on one game and check the `x-requests-last` cost printed per call
  matches markets × regions.
- Props tab verified with `AppTest` + a faked `requests.get` (tests/test_app.py): page load makes
  no per-event call, the button label carries the estimate, one click bills exactly one call,
  the raw frame and the board land in `session_state`, and a second `at.run()` makes no call.
- 2026-09-06 alerts (`alerts.py`, `alerts.yml`, `tests/test_alerts.py`, 10 offline tests): every item
  type rendered with friendly names (ML arb `🔒 ARB +2.44% locked … split 51/49`, even-money total
  `🟢 FREE MIDDLE +7.7% EV … window 45-46 · hits 8%`, −2.5/+3.5 `🎯 MIDDLE +3.0% EV … hits 8%, miss
  costs 4.5%`, −2.5/+3 `½ HALF MIDDLE … window 3p`, `💰 +EV 6.2% · PIT -3.5 +105 @ Hard Rock OH (fair
  -107) · also Bovada +100, BetUS +100 · Sun 1:00PM`); dedupe (second run nothing, re-price new key,
  24 h expiry, bad JSON tolerated); chunking at 1900; Discord / Telegram payloads with a monkeypatched
  `requests.post` and no URL / token in any raised message; `alerts.py nfl --csv lines_template.csv
  --dry-run` via subprocess sends the FanDuel U 48.5 pick (+3.3%), writes the state + summary, exits 0,
  and says `nothing new` on the second run; `--test` never writes state; a dead webhook exits 1 with
  no state; the YAML parses, 7 cron lines x 5 fields, no `23,0` wraps, exactly 28 runs/week. The
  Action itself has NOT run yet: the owner adds `DISCORD_WEBHOOK` (README → Alerts), then Actions →
  alerts → Run workflow → test=true.

## Backlog (owner's roadmap, rough priority)
1. ~~Add secrets; deploy app.py to Streamlit Cloud~~ — DONE 2026-09-05/06: secrets set by the owner,
   app live at scott-sports-predictions-dg3viypucz4nyba8clzscc.streamlit.app (auto-redeploys on push to main)
2. ~~Telegram/Discord alert on new +EV line~~ — DONE 2026-09-06/09: `alerts.py` + `alerts.yml` (17 runs/week
   x 3 credits ≈ 220/month, publishes the board snapshot too, dedupe state in the Action cache); the owner still has to add the
   `DISCORD_WEBHOOK` secret and fire the test run (README → Alerts, DEPLOY.md → Option B)
3. ~~Run the same backtest on CFB~~ — DONE 2026-09-05 (50.4% ATS, see verified state); next
   is a CFB `ev`/middles scan, not ratings work
4. QB starter-vs-backup point-value table; auto-apply instead of flag
5. Opening-line capture → real CLV tracking via BetTracker
6. Empirical key-number margin distribution — DONE for middles (`margins.py`, unconditional
   weights); still open: condition on the spread, and half-point buy/sell pricing on the card
7. Derivatives (1H, team totals) off PointsModel

## Conventions
- pandas/numpy/scipy only in the core; no heavy ML deps. Must import on 3.9+ (all modules
  use `from __future__ import annotations`).
- Keep `sharpmodel/` importable without Streamlit; `app.py` is the only UI dependency.
- All odds in American format; all margins are home-minus-away.
- Owner has done GitHub projects with Claude before; comfortable with git basics, not a
  professional developer.
