"""publish_card grading/indexing on a temp predictions dir — no network."""
import numpy as np
import pandas as pd

import publish_card as pc


def _hist(game_ids):
    return pd.DataFrame(dict(game_id=game_ids, season=2025, week=1, margin=[7.0, -3.0],
                             home_pts=[24, 20], away_pts=[17, 23], date=pd.Timestamp("2025-09-06")))


def _pred_csv(path, game_ids):
    pd.DataFrame(dict(game_id=game_ids, home=["A", "C"], away=["B", "D"],
                      market_margin=[3.0, -1.0], fair_margin=[5.0, -3.0],
                      spread_side=["home", "away"], spread_line=[-3.0, 1.0], spread_odds=[-110, -105],
                      spread_stake=[0.01, 0.02], total_side=["over", "under"], market_total=[40.0, 44.0],
                      total_stake=[0.01, 0.0])).to_csv(path, index=False)


def test_grade_handles_numeric_game_ids(tmp_path, monkeypatch):
    """CFBD ids are ints on disk but str in the schedule; grading must still join."""
    monkeypatch.setattr(pc, "OUT", str(tmp_path))
    (tmp_path / "cfb").mkdir()
    _pred_csv(tmp_path / "cfb" / "2025_w01.csv", [401520281, 401520282])
    g = pc.grade("cfb", _hist(["401520281", "401520282"]))
    assert g.graded.iloc[0] == 2
    # A -3 won by 7 -> cover; D +1 lost by... home C lost by 3 so away D +1 covers
    assert (g.spread_W.iloc[0], g.spread_L.iloc[0]) == (2, 0)
    assert g.spread_units.iloc[0] > 1.8
    assert g.total_bets.iloc[0] == 1 and g.total_W.iloc[0] == 1       # 41 > 40 over hits


def test_write_index_lists_weeks(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "OUT", str(tmp_path))
    (tmp_path / "nfl").mkdir()
    _pred_csv(tmp_path / "nfl" / "2025_w01.csv", ["2025_01_B_A", "2025_01_D_C"])
    (tmp_path / "nfl" / "2025_w01.md").write_text("# x")
    pc.write_index("nfl", _hist(["2025_01_B_A", "2025_01_D_C"]))
    txt = (tmp_path / "nfl" / "README.md").read_text()
    assert "2025_w01" in txt and "Season to date" in txt


def test_current_week_picks_first_unplayed():
    today = pd.Timestamp("2026-09-10")
    h = pd.DataFrame(dict(season=2026, week=[1, 1, 2, 2], margin=[3.0, np.nan, np.nan, np.nan],
                          date=pd.to_datetime(["2026-09-09", "2026-09-13", "2026-09-17", "2026-09-20"])))
    assert pc.current_week(h, 2026, today) == 1
    assert pc.current_week(h, 2026, pd.Timestamp("2026-09-15")) == 2
    assert pc.current_season(pd.Timestamp("2026-03-01")) == 2025
