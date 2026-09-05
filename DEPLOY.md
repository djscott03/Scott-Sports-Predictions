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

### Run it locally instead
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add keys
streamlit run app.py          # http://localhost:8501
```

## Option B — Always-on scanner with alerts (when you're serious)

Community Cloud only refreshes while someone is looking. To catch stale lines the
moment they appear you need a process running 24/7 that pushes alerts:

- Run `run.py nfl ev` on a cron every 2–5 min on a $5/mo VPS (Hetzner, DigitalOcean)
  or a free GitHub Actions schedule (min interval 5 min, quota-limited).
- Post new +EV lines to Telegram/Discord via webhook (a 10-line addition).
- Paid Odds API tier is mandatory at this cadence.

Sharps care about speed here, not dashboards: the +EV window at a soft book is often
under 10 minutes.

## Option C — Fully custom site
FastAPI backend + scheduler (APScheduler) + React/HTMX front end on Railway/Render.
Same `sharpmodel` package underneath; only worth it if you're sharing it with others.
