"""
The full pipeline: fit ratings as of a point in time, blend with the market,
price every game, and evaluate walk-forward.

The single most important design choice: the *market is the prior*.
A closing NFL line is the best public predictor of a game (MAE ~10.3 pts).
Your model does not replace it; it finds the residual the market mispriced.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .ratings import MarginModel, PointsModel, regress_to_mean
from .pricing import (LEAGUE_SD, cover_probs, total_probs, moneyline_prob,
                      edge_and_kelly, devig, american_to_prob)
from .data import adjusted_margin
from .adjustments import apply_adjustments, qb_change_flags

LEAGUE_CFG = {
    "nfl": dict(lam=6.0, half_life=9.0, hfa_prior=1.5, regress=0.45, epa_weight=0.5,
                market_weight=0.70, min_edge_pts=1.5, min_edge_total=2.0),
    "cfb": dict(lam=8.0, half_life=10.0, hfa_prior=2.6, regress=0.65, epa_weight=0.0,
                market_weight=0.65, min_edge_pts=2.5, min_edge_total=3.0),
}


@dataclass
class WeekPrediction:
    ratings: pd.Series
    market_ratings: pd.Series
    preds: pd.DataFrame


class SharpModel:
    def __init__(self, league: str, **overrides):
        self.league = league
        self.cfg = {**LEAGUE_CFG[league], **overrides}
        self.sd = LEAGUE_SD[league]

    # ---------- fitting ----------
    def _priors(self, hist: pd.DataFrame, season: int) -> dict:
        """Preseason prior = regressed blend of last season's model + market ratings."""
        last = hist[hist.season == season - 1]
        if last.empty:
            return {}
        c = self.cfg
        mm = MarginModel(lam=c["lam"], half_life=99, hfa_prior=c["hfa_prior"])
        last = last.assign(adj=adjusted_margin(last, c["epa_weight"]))
        model_r = mm.fit(last, "adj", last.week_index.max()).ratings
        if last.market_margin.notna().sum() > 20:
            mkt_r = MarginModel(lam=c["lam"], half_life=99, hfa_prior=c["hfa_prior"]).fit(
                last, "market_margin", last.week_index.max()).ratings
            blend = 0.5 * model_r.add(mkt_r, fill_value=0) if len(mkt_r) else model_r
        else:
            blend = model_r
        return regress_to_mean(blend, c["regress"])

    def fit_as_of(self, hist: pd.DataFrame, as_of_index: float, season: int):
        """Fit on every completed game strictly before `as_of_index`."""
        c = self.cfg
        h = hist[(hist.week_index < as_of_index) & hist.margin.notna()].copy()
        # Use last ~2 seasons of data; recency weights + priors handle the rest
        h = h[h.season >= season - 1]
        h["adj"] = adjusted_margin(h, c["epa_weight"])
        priors = self._priors(hist, season)

        self.margin_model = MarginModel(lam=c["lam"], half_life=c["half_life"],
                                        hfa_prior=c["hfa_prior"]).fit(h, "adj", as_of_index, priors)
        self.points_model = PointsModel(lam=c["lam"], half_life=c["half_life"]).fit(h, as_of_index)
        hm = h[h.market_margin.notna()]
        self.market_model = (MarginModel(lam=c["lam"], half_life=c["half_life"], hfa_prior=c["hfa_prior"])
                             .fit(hm, "market_margin", as_of_index, priors) if len(hm) > 10 else None)
        return self

    # ---------- pricing ----------
    def price_games(self, games: pd.DataFrame, kelly_fraction=0.25, max_stake=0.03) -> pd.DataFrame:
        c, sd_s, sd_t = self.cfg, self.sd["spread"], self.sd["total"]
        rows = []
        for _, g in games.iterrows():
            model_m = float(self.margin_model.predict(g.home, g.away, g.neutral)) + float(g.get("margin_adj", 0.0) or 0.0)
            hp, ap = self.points_model.predict(g.home, g.away, g.neutral)
            model_t = float(hp + ap) + float(g.get("total_adj", 0.0) or 0.0)
            mkt_m, mkt_t = g.get("market_margin", np.nan), g.get("total_line", np.nan)

            # Blend: market anchors, model nudges. If no line, model stands alone.
            fair_m = c["market_weight"] * mkt_m + (1 - c["market_weight"]) * model_m if np.isfinite(mkt_m) else model_m
            fair_t = c["market_weight"] * mkt_t + (1 - c["market_weight"]) * model_t if np.isfinite(mkt_t) else model_t

            row = dict(game_id=g.game_id, season=g.season, week=g.week, date=g.date,
                       home=g.home, away=g.away, neutral=g.neutral,
                       qb_flag=bool(g.get("home_qb_change", False) or g.get("away_qb_change", False)),
                       model_margin=model_m, market_margin=mkt_m, fair_margin=fair_m,
                       model_total=model_t, market_total=mkt_t, fair_total=fair_t,
                       p_home_ml=moneyline_prob(fair_m, sd_s))

            # ---- spread ----
            if np.isfinite(mkt_m):
                line = -mkt_m
                cp = cover_probs(fair_m, line, sd_s)
                side = "home" if fair_m > mkt_m else "away"
                p_win = cp["win"] if side == "home" else cp["loss"]
                odds = g.get(f"{side}_spread_odds", -110)
                odds = -110 if not np.isfinite(odds) else odds
                ek = edge_and_kelly(p_win, cp["push"], odds, kelly_fraction, max_stake)
                edge_pts = abs(fair_m - mkt_m)
                row.update(spread_side=side, spread_line=(line if side == "home" else -line),
                           spread_odds=odds, spread_p=p_win, spread_edge_pts=edge_pts,
                           spread_ev=ek["ev"], spread_stake=ek["stake_frac"] if edge_pts >= c["min_edge_pts"] else 0.0)
            # ---- total ----
            if np.isfinite(mkt_t):
                tp = total_probs(fair_t, mkt_t, sd_t)
                side = "over" if fair_t > mkt_t else "under"
                p_win = tp[side]
                odds = g.get(f"{side}_odds", -110)
                odds = -110 if not np.isfinite(odds) else odds
                ek = edge_and_kelly(p_win, tp["push"], odds, kelly_fraction, max_stake)
                edge_pts = abs(fair_t - mkt_t)
                row.update(total_side=side, total_odds=odds, total_p=p_win, total_edge_pts=edge_pts,
                           total_ev=ek["ev"], total_stake=ek["stake_frac"] if edge_pts >= c["min_edge_total"] else 0.0)
            # ---- moneyline ----
            hml, aml = g.get("home_ml", np.nan), g.get("away_ml", np.nan)
            if np.isfinite(hml) and np.isfinite(aml):
                fh, fa = devig(hml, aml)
                ph = row["p_home_ml"]
                side = "home" if ph > fh else "away"
                p_win = ph if side == "home" else 1 - ph
                ek = edge_and_kelly(p_win, 0.0, hml if side == "home" else aml, kelly_fraction, max_stake)
                row.update(ml_side=side, ml_odds=hml if side == "home" else aml, ml_p=p_win,
                           ml_fair_p=fh if side == "home" else fa, ml_ev=ek["ev"],
                           ml_stake=ek["stake_frac"] if ek["edge_prob"] >= 0.03 else 0.0)
            rows.append(row)
        return pd.DataFrame(rows)

    def predict_week(self, hist: pd.DataFrame, season: int, week: int, **kw) -> WeekPrediction:
        games = hist[(hist.season == season) & (hist.week == week)]
        if games.empty:
            raise ValueError(f"No games for {season} week {week}")
        as_of = games.week_index.min()
        self.fit_as_of(hist, as_of, season)
        games = apply_adjustments(qb_change_flags(hist, games), manual=kw.pop("manual", None))
        preds = self.price_games(games, **kw)
        # never stake a game with an unpriced QB change
        for c in ("spread_stake", "total_stake", "ml_stake"):
            if c in preds: preds.loc[preds.qb_flag, c] = 0.0
        mkt = self.market_model.ratings if self.market_model is not None else pd.Series(dtype=float)
        return WeekPrediction(self.margin_model.ratings, mkt, preds)

    # ---------- evaluation ----------
    def backtest(self, hist: pd.DataFrame, seasons: list[int], start_week: int = 1,
                 verbose=True) -> pd.DataFrame:
        out = []
        for s in seasons:
            weeks = sorted(hist[(hist.season == s) & hist.margin.notna()].week.unique())
            for w in weeks:
                if w < start_week: continue
                games = hist[(hist.season == s) & (hist.week == w) & hist.margin.notna()]
                if games.empty: continue
                self.fit_as_of(hist, games.week_index.min(), s)
                p = self.price_games(apply_adjustments(games))
                p = p.merge(games[["game_id", "margin", "home_pts", "away_pts"]], on="game_id")
                out.append(p)
            if verbose: print(f"  backtested {s}")
        bt = pd.concat(out, ignore_index=True)
        bt["total"] = bt.home_pts + bt.away_pts
        return bt


def summarize_backtest(bt: pd.DataFrame, sd_spread: float) -> dict:
    """Metrics that matter: does the blend beat the market, and does betting only
    flagged edges make money after vig?"""
    b = bt.dropna(subset=["market_margin"]).copy()
    res = {"n_games": len(b),
           "MAE_market": (b.margin - b.market_margin).abs().mean(),
           "MAE_model": (b.margin - b.model_margin).abs().mean(),
           "MAE_blend": (b.margin - b.fair_margin).abs().mean()}
    # spread bets
    sp = b[b.spread_stake > 0].copy()
    if len(sp):
        home_cov = np.sign(sp.margin + np.where(sp.spread_side == "home", sp.spread_line, -sp.spread_line))
        won = np.where(sp.spread_side == "home", home_cov, -home_cov)
        sp = sp.assign(result=won)
        dec = sp.spread_odds.apply(lambda o: 1 + o / 100 if o > 0 else 1 + 100 / -o)
        pnl = np.where(won > 0, dec - 1, np.where(won < 0, -1.0, 0.0))
        res.update(spread_bets=len(sp), spread_win_pct=(won > 0).sum() / max((won != 0).sum(), 1),
                   spread_roi_flat=pnl.mean(), spread_units_flat=pnl.sum(),
                   spread_roi_kelly=(pnl * sp.spread_stake).sum() / sp.spread_stake.sum())
        # calibration: predicted p vs realised
        res["spread_brier"] = float(np.mean((sp.spread_p - (won > 0)) ** 2))
    # totals
    tt = b[b.total_stake > 0].copy()
    if len(tt):
        ov = np.sign(tt.total - tt.market_total)
        won = np.where(tt.total_side == "over", ov, -ov)
        pnl = np.where(won > 0, 100 / 110, np.where(won < 0, -1.0, 0.0))
        res.update(total_bets=len(tt), total_win_pct=(won > 0).sum() / max((won != 0).sum(), 1),
                   total_roi_flat=pnl.mean(), total_units_flat=pnl.sum())
    # breakeven at -110 is 52.38%
    res["breakeven_pct_-110"] = 0.5238
    return res


class BetTracker:
    """Log real bets and compute Closing Line Value — the sharp's true scorecard.
    If you consistently beat the close, you will win long-run regardless of short-term results."""
    def __init__(self, path="bets.csv"):
        self.path = path
        try: self.df = pd.read_csv(path)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=["ts", "game_id", "market", "side", "line", "odds",
                                            "stake", "model_fair", "close_line", "close_odds", "result"])

    def log(self, game_id, market, side, line, odds, stake, model_fair):
        self.df.loc[len(self.df)] = [pd.Timestamp.now(), game_id, market, side, line, odds,
                                     stake, model_fair, np.nan, np.nan, np.nan]
        self.df.to_csv(self.path, index=False)

    def set_close(self, game_id, market, close_line, close_odds=-110):
        m = (self.df.game_id == game_id) & (self.df.market == market)
        self.df.loc[m, ["close_line", "close_odds"]] = [close_line, close_odds]
        self.df.to_csv(self.path, index=False)

    def clv_report(self, sd=13.4):
        d = self.df.dropna(subset=["close_line"]).copy()
        # points of CLV: positive when the line moved toward your side after you bet
        d["clv_pts"] = np.where(d.side.isin(["home", "over"]),
                                d.close_line - d.line,   # you took e.g. -3, closed -4.5 => +1.5
                                d.line - d.close_line)
        d["clv_prob"] = d.clv_pts.apply(lambda x: __import__("scipy.stats").stats.norm.cdf(x / sd) - 0.5)
        return {"bets": len(d), "avg_clv_pts": d.clv_pts.mean(), "pct_beat_close": (d.clv_pts > 0).mean(),
                "avg_clv_prob": d.clv_prob.mean()}
