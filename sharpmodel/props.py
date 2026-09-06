"""
Player props: nflverse weekly stats -> projections -> distributions -> prices.

Pipeline
  1. load_player_weeks: one row per player-game (nflverse stats_player_week parquet).
  2. project_players: recency-weighted per-game mean shrunk to the position mean, times an
     opponent factor (stat allowed to that position) and a game-environment factor
     (market implied team total vs the team's own scoring).
  3. prop_probs / implied_mean: the distribution per market (Normal for yardage with
     integer-line pushes, Poisson for counts, Poisson-lambda Bernoulli for anytime TD)
     and its inverse -- the props analogue of odds.sharp_fair.
  4. price_props: devig every two-way book, invert to a market mean, blend the projection
     in at model_weight, price every posted line off that mean, rank by EV.

Same rule as odds.py: the market is the prior. A projection with no two-way market
behind it is shown with stake 0, no EV and flagged 'no_market'.

Dispersion caveat: the yardage sd is a cv fitted from walk-forward residuals of the projection
(established starters only, see fit_dispersion), so it still carries projection error on top of
true game-to-game variance. Priced around a market line that makes the tails a little fat and
EV% a conservative *ranking* of stale numbers, not a calibrated edge.
"""
from __future__ import annotations
import re
from difflib import get_close_matches
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm, poisson
from .data import WEEKS_PER_SEASON
from .pricing import american_to_prob, prob_to_american, devig, edge_and_kelly

STATS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/stats_player/"
             "stats_player_week_{season}.parquet")
STAT_COLS = ["completions", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards",
             "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds"]
META_COLS = ["player_id", "player_display_name", "position", "position_group", "season", "week",
             "season_type", "team", "opponent_team"]
SKILL = ["QB", "RB", "WR", "TE"]

# Odds API market key -> stat (nflverse column or callable on the stats df), distribution, positions.
# Adding a market is one line, e.g. "player_rush_attempts": dict(stat="carries", dist="poisson", positions=["QB","RB","WR"])
MARKETS = {
    "player_pass_yds":      dict(stat="passing_yards",   dist="normal",  positions=["QB"], sd_floor=15.0),
    "player_pass_tds":      dict(stat="passing_tds",     dist="poisson", positions=["QB"]),
    "player_rush_yds":      dict(stat="rushing_yards",   dist="normal",  positions=["QB", "RB", "WR"], sd_floor=10.0),
    "player_reception_yds": dict(stat="receiving_yards", dist="normal",  positions=["RB", "WR", "TE"], sd_floor=10.0),
    "player_receptions":    dict(stat="receptions",      dist="poisson", positions=["RB", "WR", "TE"]),
    "player_anytime_td":    dict(stat=lambda d: d.rushing_tds + d.receiving_tds, dist="bernoulli", positions=SKILL),
}
DEFAULT_CV = {"player_pass_yds": 0.30, "player_rush_yds": 0.55, "player_reception_yds": 0.65}
_CV: dict = {}      # filled by fit_dispersion; DEFAULT_CV until then (deterministic tests)
OPP_CLIP = (0.7, 1.4)   # low-count stats (WR rush yds, QB TDs) swing wildly even after the 50% shrink
STARTERS = {"QB": 32, "RB": 64, "WR": 96, "TE": 32}   # shrink target = mean of this many per position (books price starters)


def _game_index(df: pd.DataFrame) -> pd.Series:
    return df.season * WEEKS_PER_SEASON + df.week      # continuous, so decay crosses the offseason


def _stat(df: pd.DataFrame, stat) -> np.ndarray:
    return np.asarray(stat(df) if callable(stat) else df[stat], float)


def _sd(market: str, ref):
    """Normal-market sd = cv * mean with an absolute floor; None for count/bernoulli markets."""
    spec = MARKETS[market]
    if spec["dist"] != "normal": return None
    return np.maximum(_CV.get(market, DEFAULT_CV.get(market, 0.5)) * np.abs(ref), spec.get("sd_floor", 0.0))


# ---------------- data ----------------
def load_player_weeks(seasons: list[int]) -> pd.DataFrame:
    """Long df: season, week, season_type, player_id, player, team, opponent, position, <stats>, game_index.
    Skill positions only; a season that 404s (not published yet) is skipped with a note."""
    frames = []
    for y in seasons:
        try:
            frames.append(pd.read_parquet(STATS_URL.format(season=y), columns=META_COLS + STAT_COLS))
        except Exception as e:
            print(f"[props] skip {y}: {e}"); continue
    if not frames: raise RuntimeError("no player stats loaded")
    df = pd.concat(frames, ignore_index=True).rename(columns={"player_display_name": "player",
                                                              "opponent_team": "opponent"})
    df["position"] = df.position_group.fillna(df.position)
    df = df[df.position.isin(SKILL)].drop(columns="position_group").copy()
    df[STAT_COLS] = df[STAT_COLS].fillna(0.0)
    df["game_index"] = _game_index(df)
    return df.sort_values(["game_index", "player_id"]).reset_index(drop=True)


# ---------------- projections ----------------
def _ew_project(sub: pd.DataFrame, x: np.ndarray, t: float, half_life: float,
                prior_games: float, min_games: int) -> pd.DataFrame:
    """Per-player recency-weighted mean of x (rows before t), shrunk toward the position's starter mean
    (top STARTERS[pos] players by that mean, flagged `starter`) with prior_games pseudo-games. Kish
    effective n, so the decay sharpens memory without draining the sample."""
    w = 0.5 ** ((t - sub.game_index.values) / half_life)
    d = pd.DataFrame({"pid": sub.player_id.values, "pos": sub.position.values, "w": w, "wx": w * x, "w2": w * w})
    g = d.groupby("pid").agg(pos=("pos", "last"), w=("w", "sum"), wx=("wx", "sum"), w2=("w2", "sum"),
                             n_games=("w", "size"))
    g = g[g.n_games >= min_games]
    if g.empty: return g
    ew, n_eff = g.wx / g.w, g.w ** 2 / g.w2
    rank = ew.groupby(g.pos).rank(ascending=False, method="first")
    starter = rank <= g.pos.map(STARTERS).fillna(np.inf)
    reg = g[starter].groupby("pos")[["wx", "w"]].sum()
    mu = (n_eff * ew + prior_games * g.pos.map(reg.wx / reg.w)) / (n_eff + prior_games)
    return pd.DataFrame({"ew": ew, "n_eff": n_eff, "n_games": g.n_games, "mu": mu, "starter": starter})


def _opp_factors(sub: pd.DataFrame, x: np.ndarray, t: float, half_life: float) -> dict:
    """{(defense, position): factor}: stat allowed per game to that position vs league, shrunk 50% to 1."""
    w = 0.5 ** ((t - sub.game_index.values) / half_life)
    d = pd.DataFrame({"opp": sub.opponent.values, "pos": sub.position.values, "gi": sub.game_index.values,
                      "wx": w * x, "w": w})
    pg = d.groupby(["opp", "pos", "gi"]).agg(wx=("wx", "sum"), w=("w", "first"))
    tm, lg = pg.groupby(["opp", "pos"]).sum(), pg.groupby("pos").sum()
    allowed = tm.wx / tm.w
    league = (lg.wx / lg.w).reindex(allowed.index.get_level_values("pos")).values
    ratio = np.divide(allowed.values, league, out=np.ones(len(league)), where=league > 0)
    return dict(zip(allowed.index, 0.5 + 0.5 * ratio))


def _split_games(g: pd.DataFrame, season: int, week: int, t: float):
    """This week's slate, plus played history (for team ppg) when the df carries earlier weeks."""
    if g is None or g.empty: return pd.DataFrame(columns=["home", "away"]), pd.DataFrame()
    if "season" in g and "week" in g:
        slate = g[(g.season == season) & (g.week == week)]
        hist = g[(_game_index(g) < t) & g.home_pts.notna()] if "home_pts" in g else g.iloc[0:0]
        return slate, hist
    return g, g.iloc[0:0]


def _env_factors(slate: pd.DataFrame, hist: pd.DataFrame, t: float, half_life: float) -> dict:
    """{team: sqrt(market implied team total / recency-weighted ppg)} clipped to [0.8, 1.25]."""
    if slate.empty or hist.empty or not {"total_line", "market_margin"} <= set(slate.columns): return {}
    imp = {}
    for r in slate.itertuples():
        if pd.notna(r.total_line) and pd.notna(r.market_margin):
            imp[r.home] = (r.total_line + r.market_margin) / 2; imp[r.away] = (r.total_line - r.market_margin) / 2
    w = 0.5 ** ((t - _game_index(hist)) / half_life)
    pts = pd.DataFrame({"team": pd.concat([hist.home, hist.away]), "w": pd.concat([w, w]),
                        "wx": pd.concat([w * hist.home_pts, w * hist.away_pts])})
    ppg = pts.groupby("team").wx.sum() / pts.groupby("team").w.sum()
    return {tm: float(np.clip((v / ppg[tm]) ** 0.5, 0.8, 1.25)) for tm, v in imp.items() if ppg.get(tm, 0) > 0}


def project_players(stats: pd.DataFrame, games_this_week: pd.DataFrame, as_of: tuple,
                    half_life: float = 4.0, prior_games: float = 3.0, min_games: int = 3) -> pd.DataFrame:
    """
    Per player x market: proj_mean = shrunk EW mean * opp_factor * env_factor, proj_sd per MARKETS.
    stats: load_player_weeks output. games_this_week: this week's games (home, away, market_margin,
    total_line); pass the full load_nfl df and earlier played weeks feed the team ppg denominator of
    env_factor (1.0 without them). as_of=(season, week): only rows strictly before count.
    Team = most recent row's team; players with < min_games rows are dropped.
    """
    season, week = as_of; t = season * WEEKS_PER_SEASON + week
    st = stats if "game_index" in stats else stats.assign(game_index=_game_index(stats))
    hist = st[st.game_index < t].sort_values("game_index")
    last = hist.groupby("player_id")[["player", "team", "position"]].last()
    slate, ghist = _split_games(games_this_week, season, week, t)
    opp = {**dict(zip(slate.home, slate.away)), **dict(zip(slate.away, slate.home))}
    env = _env_factors(slate, ghist, t, half_life)
    out = []
    for mk, spec in MARKETS.items():
        sub = hist[hist.position.isin(spec["positions"])]
        if sub.empty: continue
        x = _stat(sub, spec["stat"])
        p = _ew_project(sub, x, t, half_life, prior_games, min_games)
        if p.empty: continue
        ofac = _opp_factors(sub, x, t, half_life)
        p = p.join(last)
        p["opponent"] = p.team.map(opp)
        p["opp_factor"] = np.clip([ofac.get((o, ps), 1.0) for o, ps in zip(p.opponent, p.position)], *OPP_CLIP)
        p["env_factor"] = p.team.map(env).fillna(1.0)
        mean = p.mu * p.opp_factor * p.env_factor
        out.append(pd.DataFrame({"player_id": p.index, "player": p.player.values, "team": p.team.values,
                                 "opponent": p.opponent.values, "position": p.position.values, "market": mk,
                                 "proj_mean": mean.values,
                                 "proj_sd": _sd(mk, mean.values) if spec["dist"] == "normal" else mean.values,
                                 "n_games": p.n_games.values, "opp_factor": p.opp_factor.values,
                                 "env_factor": p.env_factor.values}))
    cols = ["player_id", "player", "team", "opponent", "position", "market", "proj_mean", "proj_sd",
            "n_games", "opp_factor", "env_factor"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def fit_dispersion(stats: pd.DataFrame, as_of: tuple | None = None, weeks: int = 17, half_life: float = 4.0,
                   prior_games: float = 3.0, min_games: int = 3, cv_floor: float = 0.35,
                   min_fit_games: int = 8) -> dict:
    """Coefficient of variation per normal market from walk-forward residuals of the same EW projection
    (cv^2 = sum resid^2 / sum proj^2 over the last `weeks` game-weeks before as_of). Fitted on established
    starters only -- the STARTERS population with >= min_fit_games prior games -- because shrunk backups
    and rookies add projection error that is not game-to-game variance and inflates the sd around a
    market line (books price starters). Still includes the starters' own projection error, so it is an
    upper bound on the true cv. Cached in _CV."""
    st = stats if "game_index" in stats else stats.assign(game_index=_game_index(stats))
    t_max = as_of[0] * WEEKS_PER_SEASON + as_of[1] if as_of else st.game_index.max() + 1
    acc = {mk: [0.0, 0.0] for mk, sp in MARKETS.items() if sp["dist"] == "normal"}
    for t in sorted(i for i in st.game_index.unique() if i < t_max)[-weeks:]:
        hist, cur = st[st.game_index < t], st[st.game_index == t]
        for mk in acc:
            sp = MARKETS[mk]
            sub, c = hist[hist.position.isin(sp["positions"])], cur[cur.position.isin(sp["positions"])]
            if sub.empty or c.empty: continue
            p = _ew_project(sub, _stat(sub, sp["stat"]), t, half_life, prior_games, min_games)
            if p.empty: continue
            p = p[p.starter & (p.n_games >= min_fit_games)]
            actual = pd.Series(_stat(c, sp["stat"]), index=c.player_id.values)
            both = actual.index.intersection(p.index)
            acc[mk][0] += float(((actual[both] - p.mu[both]) ** 2).sum()); acc[mk][1] += float((p.mu[both] ** 2).sum())
    fitted = {mk: max(np.sqrt(n / d), cv_floor) if d > 0 else DEFAULT_CV.get(mk, 0.5) for mk, (n, d) in acc.items()}
    _CV.update(fitted)
    return fitted


# ---------------- distributions ----------------
def prop_probs(dist: str, mean: float, sd: float | None, line: float, side: str) -> dict:
    """{'win','push','loss'} for the bettor's side. Normal is discretized like pricing.cover_probs
    (integer lines push); Poisson pushes at integer lines; bernoulli ignores the line, P(yes)=1-exp(-mean)."""
    side = str(side).lower()
    if dist == "bernoulli":
        p = 1 - np.exp(-max(mean, 0.0))
        win = p if side in ("yes", "over") else 1 - p
        return {"win": win, "push": 0.0, "loss": 1 - win}
    integer = abs(line - round(line)) < 1e-9
    if dist == "normal":
        hi = norm.cdf(line + 0.5, mean, sd) if integer else norm.cdf(line, mean, sd)
        over, push = 1 - hi, (hi - norm.cdf(line - 0.5, mean, sd)) if integer else 0.0
    else:                                                   # poisson: P(X > line)
        k, lam = int(np.floor(line + 1e-9)), max(mean, 1e-9)
        over, push = 1 - poisson.cdf(k, lam), (poisson.pmf(k, lam) if integer else 0.0)
    win = over if side == "over" else 1 - over - push
    return {"win": float(win), "push": float(push), "loss": float(1 - win - push)}


def implied_mean(dist: str, line: float, p_over: float, sd: float | None = None) -> float:
    """Mean at which P(over line | no push) = p_over. Normal closed form off integer lines,
    otherwise brentq through prop_probs (so it round-trips exactly, pushes included)."""
    p = float(np.clip(p_over, 1e-4, 1 - 1e-4))
    if dist == "bernoulli": return float(-np.log(1 - p))
    if dist == "normal" and abs(line - round(line)) > 1e-9: return float(line + sd * norm.ppf(p))
    f = lambda m: (lambda q: q["win"] / (1 - q["push"]))(prop_probs(dist, m, sd, line, "over")) - p
    lo, hi = (line - 8 * sd, line + 8 * sd) if dist == "normal" else (1e-6, 4 * max(line, 1.0) + 20)
    return float(brentq(f, lo, hi))


# ---------------- pricing ----------------
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b")


def normalize_player(name: str) -> str:
    """'A.J. Brown' -> 'aj brown', 'Kenneth Walker III' -> 'kenneth walker'."""
    s = re.sub(r"[^a-z0-9 ]", "", str(name).lower().replace(".", "").replace("-", " "))
    return " ".join(_SUFFIX.sub("", s).split())


def _proj_index(projections: pd.DataFrame | None) -> dict:
    if projections is None or len(projections) == 0: return {}
    p = projections.assign(key=projections.player.map(normalize_player)).sort_values("n_games", ascending=False)
    return {mk: d for mk, d in p.groupby("market")}


def _match(pdf: pd.DataFrame | None, name: str, home: str, away: str):
    """Exact normalized name on the event's two teams first; else difflib over every projected player."""
    if pdf is None: return None, ["no_proj"]
    key = normalize_player(name)
    hit = pdf[(pdf.key == key) & pdf.team.isin([home, away])]
    if len(hit): return hit.iloc[0], []
    close = get_close_matches(key, pdf.key.unique().tolist(), n=1, cutoff=0.85)
    if not close: return None, ["no_proj"]
    hit = pdf[pdf.key == close[0]].iloc[0]
    return hit, (["team_mismatch"] if hit.team not in (home, away) else ["fuzzy_name"])


def _fair_american(p_win, p_push):
    p = float(np.clip(p_win / (1 - p_push) if p_push < 1 else 0.5, 1e-4, 1 - 1e-4))
    return prob_to_american(p)


MAX_SYNTH_NO = 0.98     # steepest 'no' a book posts (~ -4900); past it the synthetic pair would devig longshots to ~0


def _yes_only_prob(yes_price: float, one_sided_hold: float) -> float:
    """P(yes) from a yes-only price: power-devig against a synthetic 'no' priced so the pair holds
    one_sided_hold (capped at MAX_SYNTH_NO). Longshots take most of the vig, as in a real two-way market.
    Bounded by implied(yes) * (1 - hold): past ~+4900 the cap turns the pair's hold negative and the power
    devig would hand a yes-only price MORE than its own implied probability, i.e. +EV against itself."""
    p_yes = american_to_prob(yes_price)
    p_no = float(np.clip(1 + one_sided_hold - p_yes, 1e-3, MAX_SYNTH_NO))
    return min(devig(yes_price, prob_to_american(p_no))[0], p_yes * (1 - one_sided_hold))


def price_props(props_odds: pd.DataFrame, projections: pd.DataFrame | None = None, model_weight: float = 0.30,
                min_ev: float = 0.03, kelly_fraction: float = 0.25, max_stake: float = 0.02,
                one_sided_hold: float = 0.07) -> pd.DataFrame:
    """
    Price every posted prop line and return +EV plays, ranked.
    props_odds (long, from odds.parse_props_json): event_id, commence, home, away, book, market, player,
    side (over|under|yes|no), line, price.
    Per (event, market, normalize_player(player)) -- so 'A.J. Brown' and 'AJ Brown' are one player, shown
    under the first spelling seen: every book with a two-way pair at the same number is devigged and
    inverted -> market_mu is the median (n_books); over/under rows without a line are ignored. Yes-only
    prices are devigged against a synthetic 'no' at one_sided_hold (_yes_only_prob).
    fair_mu = (1-model_weight)*market_mu + model_weight*proj_mean, priced with fair_sd = cv * fair_mu
    (normal markets; NaN for poisson/bernoulli) -- `event_id`, `dist` and `fair_sd` are returned per row so
    the distribution can be rebuilt (middles.prop_fairs). No market_mu at all -> fair from the projection, flagged 'no_market',
    ev_pct NaN and stake 0 (a projection alone is never a bet); such rows are kept when the price beats
    the projection by min_ev and always sort last. 'team_mismatch' rows keep their EV but are staked 0:
    the opponent/environment factors belong to the stale team.
    """
    proj, rows = _proj_index(projections), []
    po = props_odds.assign(_key=props_odds.player.map(normalize_player))
    for (eid, mk, _k), e in po.groupby(["event_id", "market", "_key"]):
        if mk not in MARKETS: continue
        dist = MARKETS[mk]["dist"]
        home, away, name = e.home.iloc[0], e.away.iloc[0], e.player.iloc[0]
        pr, flags = _match(proj.get(mk), name, home, away)
        mus, books = [], set()
        for bk, b in e.groupby("book"):
            if dist == "bernoulli":
                yes, no = b[b.side.isin(["yes", "over"])], b[b.side.isin(["no", "under"])]
                if yes.empty: continue
                mus.append(devig(yes.price.iloc[0], no.price.iloc[0])[0] if len(no)
                           else _yes_only_prob(yes.price.iloc[0], one_sided_hold))
                books.add(bk)
                continue
            for _, o in b[(b.side == "over") & b.line.notna()].iterrows():
                u = b[(b.side == "under") & np.isclose(b.line.astype(float), float(o.line))]
                if len(u):
                    q, _ = devig(o.price, u.iloc[0].price)
                    mus.append(implied_mean(dist, float(o.line), q, _sd(mk, float(o.line)))); books.add(bk)
        market_mu = np.nan
        if mus: market_mu = float(-np.log(1 - np.median(mus)) if dist == "bernoulli" else np.median(mus))
        proj_mean = float(pr["proj_mean"]) if pr is not None else np.nan
        if np.isfinite(market_mu):
            fair = (1 - model_weight) * market_mu + model_weight * proj_mean if np.isfinite(proj_mean) else market_mu
        elif np.isfinite(proj_mean):
            fair = proj_mean; flags = flags + ["no_market"]
        else:
            continue
        no_market, unstaked = "no_market" in flags, ("no_market" in flags or "team_mismatch" in flags)
        sd = _sd(mk, fair)
        for _, o in e.iterrows():
            if dist != "bernoulli" and pd.isna(o.line): continue
            pp = prop_probs(dist, fair, sd, float(o.line) if pd.notna(o.line) else np.nan, o.side)
            ek = edge_and_kelly(pp["win"], pp["push"], o.price, kelly_fraction, max_stake)
            if ek["ev"] < min_ev: continue                          # no_market rows: filtered on the projection's EV
            rows.append(dict(event_id=eid, commence=o.get("commence"), matchup=f"{away} @ {home}", market=mk, player=name,
                             team=pr["team"] if pr is not None else np.nan, side=o.side, line=o.line,
                             price=o.price, book=o.book, n_books=len(books), market_mu=market_mu,
                             proj_mean=proj_mean, fair_mu=fair, fair_sd=float(sd) if sd is not None else np.nan,
                             dist=dist, p_win=pp["win"], fair_price=_fair_american(pp["win"], pp["push"]),
                             ev_pct=np.nan if no_market else ek["ev"],
                             kelly_stake=0.0 if unstaked else ek["stake_frac"], flags=";".join(flags)))
    res = pd.DataFrame(rows)
    if res.empty: return res
    return res.sort_values("ev_pct", ascending=False, na_position="last").reset_index(drop=True)
