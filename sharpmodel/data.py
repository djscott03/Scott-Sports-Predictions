"""
Data loaders -> unified game schema.

Columns:
  game_id, league, season, week, date, home, away, neutral,
  home_pts, away_pts, margin (home - away),
  market_margin (market's expected home margin; = -home_spread),
  total_line, home_spread_odds, away_spread_odds, over_odds, under_odds,
  home_ml, away_ml, week_index (continuous time index for recency weighting)
Optional efficiency columns (NFL, from play-by-play):
  home_epa_pp, away_epa_pp  (offensive EPA per play in that game)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

NFL_GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
NFL_PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
CFBD_BASE = "https://api.collegefootballdata.com"

WEEKS_PER_SEASON = 25  # gap between seasons so recency weights decay across the offseason


def _week_index(season: pd.Series, week: pd.Series, base_season: int) -> pd.Series:
    return (season - base_season) * WEEKS_PER_SEASON + week


# ---------------- NFL ----------------
def load_nfl(seasons: list[int], with_epa: bool = False) -> pd.DataFrame:
    raw = pd.read_csv(NFL_GAMES_URL)
    g = raw[raw.season.isin(seasons)].copy()
    # Fold playoffs into a continuing week index
    g["week"] = g["week"].astype(int)
    df = pd.DataFrame({
        "game_id": g.game_id, "league": "nfl", "season": g.season, "week": g.week,
        "date": pd.to_datetime(g.gameday), "home": g.home_team, "away": g.away_team,
        "neutral": (g.location == "Neutral").astype(int),
        "home_pts": g.home_score, "away_pts": g.away_score, "margin": g.result,
        "market_margin": g.spread_line,          # nflverse: + means home favored
        "total_line": g.total_line,
        "home_spread_odds": g.home_spread_odds, "away_spread_odds": g.away_spread_odds,
        "over_odds": g.over_odds, "under_odds": g.under_odds,
        "home_ml": g.home_moneyline, "away_ml": g.away_moneyline,
        "game_type": g.game_type, "roof": g.roof, "temp": g.temp, "wind": g.wind,
        "home_qb": g.home_qb_name, "away_qb": g.away_qb_name,
    })
    df["week_index"] = _week_index(df.season, df.week, min(seasons))
    if with_epa:
        df = df.merge(load_nfl_epa(seasons), on="game_id", how="left")
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def load_nfl_epa(seasons: list[int]) -> pd.DataFrame:
    """Per-game offensive EPA/play and success rate for each team (from nflverse pbp)."""
    out = []
    for y in seasons:
        try:
            pbp = pd.read_parquet(NFL_PBP_URL.format(y=y),
                                  columns=["game_id", "posteam", "home_team", "away_team",
                                           "epa", "success", "play_type", "qtr", "score_differential"])
        except Exception as e:  # season not published yet
            print(f"[epa] skip {y}: {e}"); continue
        p = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna()]
        # drop garbage time: 4th quarter with 3+ score lead
        p = p[~((p.qtr == 4) & (p.score_differential.abs() > 16))]
        agg = p.groupby(["game_id", "posteam"]).agg(epa_pp=("epa", "mean"),
                                                    sr=("success", "mean"),
                                                    plays=("epa", "size")).reset_index()
        meta = pbp.groupby("game_id")[["home_team", "away_team"]].first().reset_index()
        agg = agg.merge(meta, on="game_id")
        h = agg[agg.posteam == agg.home_team].set_index("game_id")[["epa_pp", "sr", "plays"]].add_prefix("home_")
        a = agg[agg.posteam == agg.away_team].set_index("game_id")[["epa_pp", "sr", "plays"]].add_prefix("away_")
        out.append(h.join(a, how="inner").reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["game_id"])


# ---------------- CFB ----------------
def load_cfb(seasons: list[int], api_key: str | None = None, fbs_only: bool = True,
             provider_pref=("consensus", "Bovada", "DraftKings", "ESPN Bet")) -> pd.DataFrame:
    """Requires a free key from collegefootballdata.com (env CFBD_API_KEY)."""
    import requests
    key = api_key or os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("Set CFBD_API_KEY (free at collegefootballdata.com)")
    H = {"Authorization": f"Bearer {key}"}
    frames = []
    for y in seasons:
        games = pd.DataFrame(requests.get(f"{CFBD_BASE}/games", headers=H,
                                          params={"year": y, "division": "fbs" if fbs_only else None}).json())
        lines = requests.get(f"{CFBD_BASE}/lines", headers=H, params={"year": y}).json()
        lrows = []
        for gm in lines:
            ls = {l["provider"]: l for l in gm.get("lines", [])}
            pick = next((ls[p] for p in provider_pref if p in ls), next(iter(ls.values()), None))
            if pick is None: continue
            lrows.append({"id": gm["id"], "home_spread": pick.get("spread"), "total_line": pick.get("overUnder"),
                          "home_ml": pick.get("homeMoneyline"), "away_ml": pick.get("awayMoneyline")})
        games = games.merge(pd.DataFrame(lrows), on="id", how="left")
        # CFBD field names changed to camelCase in 2024+; handle both
        c = lambda a, b: games[a] if a in games else games[b]
        df = pd.DataFrame({
            "game_id": games.id.astype(str), "league": "cfb", "season": y,
            "week": c("week", "week").astype(int) + (c("seasonType", "season_type") == "postseason") * 16,
            "date": pd.to_datetime(c("startDate", "start_date"), utc=True).dt.tz_localize(None),
            "home": c("homeTeam", "home_team"), "away": c("awayTeam", "away_team"),
            "neutral": c("neutralSite", "neutral_site").fillna(False).astype(int),
            "home_pts": c("homePoints", "home_points"), "away_pts": c("awayPoints", "away_points"),
            "total_line": pd.to_numeric(games.total_line, errors="coerce"),
            "home_ml": games.home_ml, "away_ml": games.away_ml,
        })
        df["margin"] = df.home_pts - df.away_pts
        df["market_margin"] = -pd.to_numeric(games.home_spread, errors="coerce")
        for col in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds"):
            df[col] = -110.0
        if fbs_only:
            div = c("homeClassification", "home_division"), c("awayClassification", "away_division")
            df = df[(div[0] == "fbs") & (div[1] == "fbs")]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["week_index"] = _week_index(out.season, out.week, min(seasons))
    return out.sort_values(["date", "game_id"]).reset_index(drop=True)


# ---------------- adjusted margin ----------------
def adjusted_margin(df: pd.DataFrame, epa_weight: float = 0.5, plays: float = 63.0) -> pd.Series:
    """
    De-noised game result: blend actual margin with an EPA-implied margin.
    EPA strips out turnover luck, special-teams randomness and garbage time,
    and is a better predictor of *future* margin than points alone.
    """
    m = df["margin"].astype(float)
    if "home_epa_pp" in df and df["home_epa_pp"].notna().any():
        epa_m = (df["home_epa_pp"] - df["away_epa_pp"]) * plays
        return np.where(epa_m.notna(), (1 - epa_weight) * m + epa_weight * epa_m, m)
    return m.values
