"""Top picks: the +EV board deduped to one row per pick at its best price."""
import numpy as np
import pandas as pd

from sharpmodel.odds import top_picks, PICK_COLS, find_ev, load_odds_csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ev(matchup, market, side, team, line, price, book, ev):
    return dict(commence="2026-09-13T17:00:00Z", matchup=matchup, market=market, side=side, team=team, line=line,
                price=price, book=book, ref_book="pinnacle", fair_mu=3.0, p_win=0.52, p_push=0.0, fair_price=-110.0,
                ev_pct=ev, kelly_stake=ev / 4)


def test_one_row_per_pick_best_price_first_and_also_column():
    ev = pd.DataFrame([
        _ev("CAR @ CHI", "spreads", "away", "CHI", -3.0, 100, "betus", 0.0466),
        _ev("CAR @ CHI", "spreads", "away", "CHI", -3.0, 100, "espnbet", 0.0466),      # same pick, same price
        _ev("CAR @ CHI", "spreads", "away", "CHI", -3.0, -102, "betmgm", 0.0367),      # same pick, worse price
        _ev("ATL @ PIT", "spreads", "home", "PIT", -3.5, 105, "hardrockbet", 0.0615),  # a better pick
        _ev("DEN @ KC", "ml", "away", "DEN", np.nan, 130, "bovada", 0.03),             # moneyline: NaN line groups too
        _ev("CAR @ CHI", "spreads", "away", "CHI", -3.5, 110, "fanduel", 0.02),        # different number = different pick
    ])
    p = top_picks(ev, 10)
    assert list(p.columns) == PICK_COLS and list(p["rank"]) == [1, 2, 3, 4]
    assert list(p.matchup) == ["ATL @ PIT", "CAR @ CHI", "DEN @ KC", "CAR @ CHI"]      # by EV
    chi = p.iloc[1]
    assert chi.book == "betus" and chi.price == 100 and chi.n_books == 3 and chi.also == "espnbet +100; betmgm -102"
    assert p.iloc[0].also == "" and p.iloc[0].n_books == 1
    assert pd.isna(p.iloc[2].line) and p.iloc[2].book == "bovada"
    assert len(top_picks(ev, 2)) == 2 and list(top_picks(ev, 2)["rank"]) == [1, 2]
    assert top_picks(pd.DataFrame()).empty and list(top_picks(None).columns) == PICK_COLS


def test_top_picks_on_the_template_board():
    ev = find_ev(load_odds_csv(os.path.join(ROOT, "lines_template.csv")), "nfl", min_ev=-1.0)
    p = top_picks(ev, 5)
    assert 0 < len(p) <= 5 and p.ev_pct.is_monotonic_decreasing and p["rank"].tolist() == list(range(1, len(p) + 1))
    assert not p.duplicated(["matchup", "market", "side", "line"]).any()
