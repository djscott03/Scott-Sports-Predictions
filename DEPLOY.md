# Deploy as a live site (free, ~15 minutes)

## Option A — Streamlit Community Cloud (recommended)

1. **Get keys**
   - Odds: https://the-odds-api.com → free tier (500 requests/month)
   - CFB (optional): https://collegefootballdata.com → free key

2. **The code is already on GitHub**: https://github.com/djscott03/Scott-Sports-Predictions
   Do NOT commit `.streamlit/secrets.toml` (it's gitignored). Keys go in step 4.
   For the weekly-card Action to price CFB, also add `CFBD_API_KEY` under
   repo → Settings → Secrets and variables → Actions (NFL needs no key).

3. **Create the app**: https://share.streamlit.io → *New app* → pick the repo,
   branch `main`, main file `app.py` → *Deploy*.

4. **Add secrets**: App → ⋮ → *Settings* → *Secrets*, paste:
   ```toml
   ODDS_API_KEY = "..."
   CFBD_API_KEY = "..."
   REFRESH_MIN = "20"
   ```
   Save → the app reboots and starts pulling odds.

5. Open the URL on your phone, add to home screen. Done.

### How "live" works
- Odds are cached for `REFRESH_MIN` minutes; the page auto-reloads when the cache
  expires and pulls fresh lines. One API request per refresh, regardless of viewers.
- Model ratings re-fit every 6 hours (schedules/results from nflverse, no key needed).
- **Quota math**: 500 free requests ÷ 30 days ≈ 16 pulls/day. At 20-min refresh
  that's ~5 hours of open tab per day. Close the tab when not looking, or upgrade to
  the $30/mo tier (20k requests) and set `REFRESH_MIN = "5"` for true near-real-time.
- Community Cloud sleeps the app after ~12h without visitors and wakes on the next
  visit (~30s). That's fine for a personal board.
- The **Props** tab is the exception to "1 request per refresh": each click bills
  games × markets credits (the estimate is on the button, ≈24 with the defaults). It
  never fetches on its own and keeps the last scan until you click again.

### Sharing the link with friends (credits)
- The game-line board is **shared**: one pull per `REFRESH_MIN` minutes for everyone
  combined, not per viewer. Ten people watching costs the same as one. What costs
  more is *coverage*: a pull is 3 credits (3 markets × 10 named books), so a tab left
  open all day keeps pulling — 9 credits/hour at the 20-minute default.
- `MAX_CREDITS_PER_DAY` (default 60 = 20 pulls ≈ 6½ hours of continuous viewing) caps
  that: past the budget the last board is served unchanged with a "paused" pill until
  midnight New York time. The cap is a backstop, not a plan — close the tab when you're
  not looking. Set 30 if you share widely, or raise `REFRESH_MIN` to 30.
- **Set `SCAN_PIN` before sharing.** Props scans bill games × markets credits to whoever
  clicks, and Force refresh is 3 credits too. With the PIN set, both need it (sidebar →
  Model & refresh → Owner PIN); viewers can still see the last props scan.
  ```toml
  SCAN_PIN = "pick-something"
  MAX_CREDITS_PER_DAY = "60"
  ```

### Run it locally instead
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add keys
streamlit run app.py          # http://localhost:8501
```

## Option B — Alerts to your phone (built in, free)

Community Cloud only refreshes while someone is looking. The **`alerts`** GitHub Action
(`.github/workflows/alerts.yml` → `alerts.py`) covers the gaps: on game days it pulls the
board hourly and pushes anything **new** — arbs, free middles, +EV middles, the top +EV
picks — to Discord and/or Telegram, remembering what it already sent for 24 h so you get
one ping per line. Nothing runs on your laptop and nothing is committed.

Setup (the secrets are yours to add; click-by-click in README → *Alerts*):

1. Discord: channel gear → **Integrations** → **Webhooks** → **New Webhook** → **Copy
   Webhook URL**.
2. `cd "~/Desktop/Claude/Sports Bets" && gh secret set DISCORD_WEBHOOK` and paste (or
   GitHub → repo → Settings → Secrets and variables → Actions → New repository secret).
   Telegram: `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` the same way.
3. GitHub → **Actions** → **alerts** → **Run workflow** → `test` = `true` → your channel
   gets the current top pick. Done; the schedule takes over from there.

Quota: every run is 3 credits (3 markets × 10 named books, Pinnacle included). The schedule
is Sunday hourly 9am–5pm ET plus 8pm (SNF), Thursday and Monday at 7pm and 8pm ET, and noon
ET Tuesday and Friday = 16 runs, 48 credits/week (~210/month), which fits under the free 500
next to the dashboard's `MAX_CREDITS_PER_DAY`. A scheduled run with no webhook secret exits
before pulling (0 credits). To pause,
comment out the `schedule:` block in `alerts.yml`; to tune, edit the input defaults there
(`min_ev`, `hours`) or the `python alerts.py` flags (`--min-middle-ev`, `--max-age`,
`--top`, `--exclude`).

What this is *not*: a 2-minute scanner. The +EV window at a soft book is often under 10
minutes, and catching that needs a paid Odds API tier (20k requests at $30/mo) plus a
process polling every few minutes — the same `alerts.py` on a `*/5` cron would do it, but
not on 500 free credits. Hourly on game days is what the free tier buys.

## Option C — Fully custom site
FastAPI backend + scheduler (APScheduler) + React/HTMX front end on Railway/Render.
Same `sharpmodel` package underneath; only worth it if you're sharing it with others.
