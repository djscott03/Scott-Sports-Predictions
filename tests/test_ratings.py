"""Ratings + engine tests on a synthetic league — no network."""
import numpy as np
import pandas as pd
import pytest

from sharpmodel.ratings import MarginModel, PointsModel
from sharpmodel.engine import SharpModel, summarize_backtest
from sharpmodel.adjustments import rest_adjustment, weather_total_adjustment, apply_adjustments


TRUE = {"A": 9.0, "B": 3.0, "C": -3.0, "D": -9.0}
HFA = 2.0


def synthetic_league(seasons=(2023, 2024), weeks=12, seed=0):
    """Round-robin league whose true ratings are TRUE; noise sd 10; market knows the truth."""
    rng = np.random.default_rng(seed)
    teams = list(TRUE)
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(teams)
            for i in range(0, len(order), 2):
                h, a = order[i], order[i + 1]
                exp = TRUE[h] - TRUE[a] + HFA
                margin = round(exp + rng.normal(0, 10))
                hp = 24 + round(margin / 2); ap = hp - margin
                rows.append(dict(game_id=f"{s}_{w:02d}_{a}_{h}", league="nfl", season=s, week=w,
                                 date=pd.Timestamp(f"{s}-09-01") + pd.Timedelta(weeks=w),
                                 home=h, away=a, neutral=0, home_pts=hp, away_pts=ap, margin=margin,
                                 market_margin=exp + rng.normal(0, 0.5), total_line=48.0,
                                 home_spread_odds=-110, away_spread_odds=-110, over_odds=-110, under_odds=-110,
                                 home_ml=np.nan, away_ml=np.nan))
    df = pd.DataFrame(rows)
    df["week_index"] = (df.season - min(seasons)) * 25 + df.week
    return df


def test_margin_model_recovers_ordering_and_hfa():
    g = synthetic_league()
    mm = MarginModel(lam=2.0, half_life=99, hfa_prior=2.0).fit(g, "margin", g.week_index.max() + 1)
    r = mm.ratings
    assert list(r.index) == ["A", "B", "C", "D"]              # sorted best -> worst
    assert r["A"] - r["D"] > 10                                # spread of ~18 shrunk by ridge, still large
    assert abs(r.mean()) < 1e-9                                # centred
    assert 0.5 < mm.hfa < 4.0
    assert mm.predict("A", "D", neutral=0) > mm.predict("A", "D", neutral=1)


def test_points_model_totals_are_sane():
    g = synthetic_league()
    pm = PointsModel(lam=2.0, half_life=99).fit(g, g.week_index.max() + 1)
    h, a = pm.predict("A", "D")
    assert 35 < h + a < 65
    assert h > a                                               # better team scores more


def test_engine_walk_forward_no_lookahead():
    g = synthetic_league()
    m = SharpModel("nfl", lam=2.0)
    bt = m.backtest(g, [2024], start_week=2, verbose=False)
    assert len(bt) == 22                                       # 2 games/week x 11 weeks
    assert bt.fair_margin.notna().all()
    # blend can never be worse than the worse of its two parts on average by much
    res = summarize_backtest(bt, 13.4)
    assert res["n_games"] == 22
    assert res["MAE_blend"] <= max(res["MAE_market"], res["MAE_model"]) + 0.5


def test_predict_week_zero_stakes_qb_change():
    g = synthetic_league()
    g["home_qb"] = "QB_" + g.home; g["away_qb"] = "QB_" + g.away
    # change A's listed starter for the final week only
    last = g[(g.season == 2024) & (g.week == 12)]
    g.loc[last.index[last.home == "A"], "home_qb"] = "BACKUP"
    g.loc[last.index[last.away == "A"], "away_qb"] = "BACKUP"
    wp = SharpModel("nfl", lam=2.0).predict_week(g, 2024, 12)
    flagged = wp.preds[wp.preds.qb_flag]
    assert len(flagged) == 1 and set(flagged[["home", "away"]].iloc[0]) & {"A"}
    assert (flagged.spread_stake == 0).all()


def test_adjustments():
    assert rest_adjustment(14, 7) == 0.5
    assert rest_adjustment(4, 7) == -0.5
    assert rest_adjustment(np.nan, np.nan) == 0.0
    assert weather_total_adjustment(20, 60, "outdoors") == pytest.approx(-3.0)
    assert weather_total_adjustment(25, 60, "dome") == 0.0
    g = pd.DataFrame([dict(game_id="x", wind=15, temp=20, roof="outdoors", home_rest=7, away_rest=14)])
    out = apply_adjustments(g, manual={"x": {"margin": 2.0, "total": -1.0}})
    assert out.margin_adj.iloc[0] == pytest.approx(-0.5 + 2.0)
    assert out.total_adj.iloc[0] == pytest.approx(-1.5 - 0.5 - 1.0)
