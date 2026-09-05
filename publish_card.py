"""
Price the current (or a given) week and write it into predictions/ as a
permanent, timestamped record — the point of a *predictions* repo is that the
numbers are frozen before kickoff and graded afterwards.

  python publish_card.py                 # NFL, auto-detect current season/week
  python publish_card.py nfl 2026 3      # explicit
  python publish_card.py cfb             # needs CFBD_API_KEY

Writes predictions/<league>/<season>_w<week>.csv and .md, and regenerates
predictions/<league>/README.md (index + grading of finished weeks).
"""
from __future__ import annotations
import glob, os, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sharpmodel import SharpModel, load_nfl, load_cfb

OUT = "predictions"


# ---------------- week detection ----------------
def current_season(today: pd.Timestamp) -> int:
    return today.year if today.month >= 8 else today.year - 1


def current_week(hist: pd.DataFrame, season: int, today: pd.Timestamp) -> int:
    """First week of `season` that still has an unplayed game; else the last week."""
    s = hist[hist.season == season]
    if s.empty:
        raise ValueError(f"no schedule rows for {season}")
    open_weeks = s[s.margin.isna() & (s.date >= today.normalize() - pd.Timedelta(days=1))].week
    return int(open_weeks.min()) if len(open_weeks) else int(s.week.max())


# ---------------- formatting ----------------
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if not np.isfinite(v) else f"{v:.2f}".rstrip("0").rstrip("."))
            else:
                cells.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pick(preds: pd.DataFrame) -> pd.DataFrame:
    """Human-readable pick column: 'PHI -7' / 'OVER 47.5'."""
    p = preds.copy()
    team = np.where(p.spread_side == "home", p.home, p.away)
    p["spread_pick"] = [f"{t} {l:+.1f}" if np.isfinite(l) else "" for t, l in zip(team, p.spread_line)]
    p["total_pick"] = [f"{s.upper()} {t:.1f}" if isinstance(s, str) and np.isfinite(t) else ""
                       for s, t in zip(p.get("total_side", [None] * len(p)), p.market_total)]
    return p


def write_week(league: str, season: int, week: int, wp, hfa: float) -> str:
    os.makedirs(f"{OUT}/{league}", exist_ok=True)
    stem = f"{OUT}/{league}/{season}_w{week:02d}"
    p = _pick(wp.preds)
    p["published_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    p.to_csv(f"{stem}.csv", index=False)

    card_cols = ["date", "away", "home", "market_margin", "model_margin", "fair_margin",
                 "spread_pick", "spread_p", "spread_edge_pts", "spread_stake",
                 "market_total", "model_total", "total_pick", "total_edge_pts", "total_stake", "qb_flag"]
    card = p[[c for c in card_cols if c in p]].copy()
    card["date"] = pd.to_datetime(card.date).dt.strftime("%a %m/%d")
    for c in ("spread_p",):
        if c in card: card[c] = (card[c] * 100).round(1)
    for c in ("spread_stake", "total_stake"):
        if c in card: card[c] = (card[c] * 100).round(2)
    card = card.rename(columns={"spread_p": "cover %", "spread_stake": "spread stake %",
                                "total_stake": "total stake %", "spread_edge_pts": "spread edge",
                                "total_edge_pts": "total edge"})

    plays = p[(p.get("spread_stake", 0) > 0) | (p.get("total_stake", 0) > 0)]
    play_lines = []
    for _, r in plays.iterrows():
        if r.get("spread_stake", 0) > 0:
            play_lines.append(f"- **{r.spread_pick}** ({r.spread_odds:+.0f}) — {r.away} @ {r.home}, "
                              f"fair {r.fair_margin:+.1f} vs line {r.market_margin:+.1f} "
                              f"({r.spread_edge_pts:.1f} pt edge), stake {r.spread_stake*100:.1f}% of bankroll")
        if r.get("total_stake", 0) > 0:
            play_lines.append(f"- **{r.total_pick}** ({r.total_odds:+.0f}) — {r.away} @ {r.home}, "
                              f"fair {r.fair_total:.1f} vs line {r.market_total:.1f} "
                              f"({r.total_edge_pts:.1f} pt edge), stake {r.total_stake*100:.1f}% of bankroll")
    flagged = p[p.qb_flag]
    ratings = pd.DataFrame({"team": wp.ratings.index, "model": wp.ratings.values.round(1)})
    if len(wp.market_ratings):
        ratings["market"] = wp.market_ratings.reindex(ratings.team).values.round(1)

    md = [f"# {league.upper()} {season} — Week {week}",
          f"_Published {p.published_utc.iloc[0]} UTC. Margins are home-minus-away; "
          f"fair = {SharpModel(league).cfg['market_weight']:.0%} market + model. HFA {hfa:.2f}._",
          "", "## Plays", *(play_lines or ["_No edges above the minimum threshold this week._"]),
          ""]
    if len(flagged):
        md += ["> ⚠️ QB change flagged (stake forced to 0 until priced manually): " +
               ", ".join(f"{r.away} @ {r.home}" for _, r in flagged.iterrows()), ""]
    md += ["## Full card", _md_table(card), "",
           "## Power ratings", _md_table(ratings), ""]
    with open(f"{stem}.md", "w") as f:
        f.write("\n".join(md))
    return stem


# ---------------- grading ----------------
def grade(league: str, hist: pd.DataFrame) -> pd.DataFrame:
    """Join every published week against final scores; return per-week ATS record + ROI."""
    rows = []
    res = hist[["game_id", "margin", "home_pts", "away_pts"]].dropna()
    for path in sorted(glob.glob(f"{OUT}/{league}/*_w*.csv")):
        pred = pd.read_csv(path)
        season, wk = os.path.basename(path)[:-4].split("_w")
        g = pred.merge(res, on="game_id", how="inner", suffixes=("_pred", ""))
        row = dict(season=int(season), week=int(wk), games=len(pred), graded=len(g))
        if len(g):
            sp = g[g.spread_stake > 0]
            if len(sp):
                cov = np.sign(sp.margin + np.where(sp.spread_side == "home", sp.spread_line, -sp.spread_line))
                won = np.where(sp.spread_side == "home", cov, -cov)
                dec = sp.spread_odds.apply(lambda o: o / 100 if o > 0 else 100 / -o)
                pnl = np.where(won > 0, dec, np.where(won < 0, -1.0, 0.0))
                row.update(spread_bets=len(sp), spread_W=int((won > 0).sum()), spread_L=int((won < 0).sum()),
                           spread_P=int((won == 0).sum()), spread_units=round(float(pnl.sum()), 2))
            tt = g[g.total_stake > 0]
            if len(tt):
                ov = np.sign(tt.home_pts + tt.away_pts - tt.market_total)
                won = np.where(tt.total_side == "over", ov, -ov)
                pnl = np.where(won > 0, 100 / 110, np.where(won < 0, -1.0, 0.0))
                row.update(total_bets=len(tt), total_W=int((won > 0).sum()), total_L=int((won < 0).sum()),
                           total_P=int((won == 0).sum()), total_units=round(float(pnl.sum()), 2))
            row["blend_MAE"] = round(float((g.margin - g.fair_margin).abs().mean()), 2)
            row["market_MAE"] = round(float((g.margin - g.market_margin).abs().mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def write_index(league: str, hist: pd.DataFrame):
    g = grade(league, hist)
    md = [f"# {league.upper()} predictions", "",
          "Each week is frozen at publish time (see `published_utc` in the CSV) and graded here "
          "once results are in. Units are flat 1-unit stakes; `spread_units` counts wins at the "
          "posted price, losses at −1.", ""]
    if len(g):
        cols = [c for c in ["season", "week", "games", "graded", "spread_bets", "spread_W", "spread_L", "spread_P",
                            "spread_units", "total_bets", "total_W", "total_L", "total_P", "total_units",
                            "blend_MAE", "market_MAE"] if c in g]
        md.append(_md_table(g[cols].fillna("")))
        tot = g.sum(numeric_only=True)
        if "spread_W" in g:
            md += ["", f"**Season to date — spreads:** {int(tot.get('spread_W', 0))}-{int(tot.get('spread_L', 0))}"
                   f"-{int(tot.get('spread_P', 0))}, {tot.get('spread_units', 0):+.2f} u"]
        if "total_W" in g:
            md += [f"**Season to date — totals:** {int(tot.get('total_W', 0))}-{int(tot.get('total_L', 0))}"
                   f"-{int(tot.get('total_P', 0))}, {tot.get('total_units', 0):+.2f} u"]
    md += ["", "## Weeks", *(f"- [{os.path.basename(f)[:-3]}]({os.path.basename(f)})"
                              for f in sorted(glob.glob(f"{OUT}/{league}/*_w*.md")))]
    with open(f"{OUT}/{league}/README.md", "w") as f:
        f.write("\n".join(md) + "\n")


# ---------------- main ----------------
if __name__ == "__main__":
    league = sys.argv[1] if len(sys.argv) > 1 else "nfl"
    today = pd.Timestamp.now(tz="UTC").tz_localize(None)
    season = int(sys.argv[2]) if len(sys.argv) > 2 else current_season(today)
    hist = load_nfl([season - 1, season]) if league == "nfl" else load_cfb([season - 1, season])
    week = int(sys.argv[3]) if len(sys.argv) > 3 else current_week(hist, season, today)

    stem = f"{OUT}/{league}/{season}_w{week:02d}"
    wk = hist[(hist.season == season) & (hist.week == week)]
    if os.path.exists(f"{stem}.csv") and wk.margin.notna().any():
        # A card is frozen once kickoff has happened: re-pricing now would be a backtest.
        print(f"{stem} already published and week {week} has started — keeping the frozen card")
    else:
        m = SharpModel(league)
        wp = m.predict_week(hist, season, week)
        write_week(league, season, week, wp, m.margin_model.hfa)
        n = int(((wp.preds.get("spread_stake", 0) > 0) | (wp.preds.get("total_stake", 0) > 0)).sum())
        print(f"wrote {stem}.md / .csv — {len(wp.preds)} games, {n} plays")
    write_index(league, hist)
