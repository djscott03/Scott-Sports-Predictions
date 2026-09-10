"""Offline tests for sharpmodel/history.py: every pull kept as one parquet per New York day, and the sharp books'
moves between two snapshots (steam)."""
import os
import numpy as np
import pandas as pd
import pytest

from sharpmodel.history import (append_snapshot, load_history, moves, steam_lines, steam_line, day_file, COLS,
                                MOVE_COLS, BOOK_NAMES)
from sharpmodel.odds import SHARP_BOOKS

EV = dict(event_id="e1", commence="2026-09-13T17:00:00Z", home="PHI", away="DAL")
T0 = pd.Timestamp("2026-09-13T15:00:00Z")            # 11:00am ET Sunday


def _board(sp=-2.5, tot=44.5, ml=-150, book="pinnacle", ev=EV, updated=None, extra=()):
    """One game at one book: the home spread (both sides), the total (both sides), the home ML, plus `extra` rows."""
    rows = [dict(ev, book=book, market="spreads", side="home", line=sp, price=-105, updated=updated),
            dict(ev, book=book, market="spreads", side="away", line=-sp, price=-105, updated=updated),
            dict(ev, book=book, market="totals", side="over", line=tot, price=-110, updated=updated),
            dict(ev, book=book, market="totals", side="under", line=tot, price=-110, updated=updated),
            dict(ev, book=book, market="ml", side="home", line=np.nan, price=ml, updated=updated)]
    return pd.DataFrame(rows + [dict(ev, **r) for r in extra])


def test_append_snapshot_twice_on_one_day_is_one_file_with_both_pulls(tmp_path):
    p1 = append_snapshot(_board(), "nfl", str(tmp_path), T0)
    p2 = append_snapshot(_board(-3.5, 46, -175, updated="2026-09-13T15:30:00Z"), "nfl", str(tmp_path), T0.timestamp() + 3720)
    assert p1 == p2 == str(tmp_path / "nfl" / "2026-09-13.parquet") == day_file(str(tmp_path), "nfl", T0)
    h = pd.read_parquet(p1)
    assert list(h.columns) == COLS and len(h) == 10                                   # rows summed, schema fixed
    assert h.pulled_at.nunique() == 2 and h.pulled_at.max() - h.pulled_at.min() == pytest.approx(3720)
    assert h.pulled_at.iloc[0] == pytest.approx(T0.timestamp())
    assert h.updated.iloc[0] is None and h.updated.iloc[-1] == "2026-09-13T15:30:00Z"   # hand CSV vs API timestamps mix
    assert h.line.dtype == float and np.isnan(h.line.iloc[4]) and h.price.iloc[-1] == -175
    # the day is New York's: 11:30pm ET is still Sunday's file, 12:30am ET starts Monday's
    assert append_snapshot(_board(), "nfl", str(tmp_path), "2026-09-14T03:30:00Z") == p1
    assert append_snapshot(_board(), "nfl", str(tmp_path), "2026-09-14T04:30:00Z").endswith("2026-09-14.parquet")
    assert len(pd.read_parquet(p1)) == 15
    # a hand CSV without commence / updated columns still lands; an empty board writes an empty day file
    p = append_snapshot(_board().drop(columns=["commence", "updated"]), "cfb", str(tmp_path), T0)
    assert list(pd.read_parquet(p).columns) == COLS and pd.read_parquet(p).commence.isna().all()
    p = append_snapshot(pd.DataFrame(), "cfb", str(tmp_path), "2026-09-14T16:00:00Z")
    assert os.path.exists(p) and len(pd.read_parquet(p)) == 0


def test_load_history_over_day_files(tmp_path):
    assert load_history(str(tmp_path), "nfl").empty and list(load_history(str(tmp_path), "nfl").columns) == COLS
    append_snapshot(_board(), "nfl", str(tmp_path), "2026-09-12T16:00:00Z")
    append_snapshot(_board(-3), "nfl", str(tmp_path), "2026-09-13T15:00:00Z")
    append_snapshot(_board(-3.5), "nfl", str(tmp_path), "2026-09-13T16:00:00Z")
    h = load_history(str(tmp_path), "nfl")
    assert len(h) == 15 and list(h.columns) == COLS and h.pulled_at.nunique() == 3
    assert h.pulled_at.is_monotonic_increasing                                           # oldest day first
    assert load_history(str(tmp_path), "nfl", days=1).pulled_at.nunique() == 2         # the last day file only
    assert load_history(str(tmp_path), "cfb").empty                                     # another league: nothing


def test_moves_between_two_snapshots():
    prev = _board(-2.5, 44.5, -150, extra=[dict(book="fanduel", market="spreads", side="home", line=-2.5, price=-110)])
    curr = _board(-3.5, 46.0, -175, extra=[dict(book="fanduel", market="spreads", side="home", line=-4.5, price=-110)])
    prev, curr = prev.assign(pulled_at=T0.timestamp()), curr.assign(pulled_at=T0.timestamp() + 62 * 60)
    mv = moves(prev, curr)
    assert list(mv.columns) == MOVE_COLS and set(mv.book) == {"pinnacle"}               # fanduel is not a sharp book
    assert sorted(mv.market) == ["ml", "spreads", "totals"] and (mv.minutes == 62).all()
    sp = mv[mv.market == "spreads"].iloc[0]
    assert sp.side == "home" and (sp.from_line, sp.to_line, sp.delta) == (-2.5, -3.5, 1.0)   # once, the home side, +1 of margin
    assert sp.matchup == "DAL @ PHI" and sp.event_id == "e1"
    tt = mv[mv.market == "totals"].iloc[0]
    assert tt.side == "over" and (tt.from_line, tt.to_line, tt.delta) == (44.5, 46.0, 1.5)
    ml = mv[mv.market == "ml"].iloc[0]
    assert (ml.from_price, ml.to_price) == (-150, -175) and ml.delta == pytest.approx(0.6364 - 0.6, abs=1e-3)
    assert np.isnan(ml.from_line) and np.isnan(ml.to_line)
    assert mv.iloc[0].market == "ml"                                                    # biggest relative to its bar first
    # below the bars: a half-point spread, a one-point total, a -150 -> -160 moneyline (+1.5% implied)
    assert moves(_board(-2.5, 44.5, -150), _board(-3.0, 45.5, -160)).empty
    assert len(moves(_board(-2.5, 44.5, -150), _board(-3.0, 45.5, -160), min_spread=0.5)) == 1
    # a listed book is reported; the same pull twice moves nothing; missing pulled_at = NaN minutes
    mv = moves(prev, curr, books=["pinnacle", "fanduel"])
    assert set(mv.book) == {"pinnacle", "fanduel"} and len(mv[mv.book == "fanduel"]) == 1
    assert mv[mv.book == "fanduel"].iloc[0].delta == 2.0
    assert moves(curr, curr).empty and moves(prev.drop(columns="pulled_at"), curr).minutes.isna().all()
    for bad in (None, pd.DataFrame(), prev.iloc[0:0], prev.drop(columns=["line"])):
        assert moves(bad, curr).empty and moves(curr, bad).empty and list(moves(bad, curr).columns) == MOVE_COLS
    # the away-side / under rows never produce a second move; a game only in one snapshot is not a move
    other = _board(-7, 50, -300, ev=dict(EV, event_id="e2", home="KC", away="DEN"))
    mv = moves(prev, pd.concat([curr, other], ignore_index=True))
    assert len(mv) == 3 and set(mv.event_id) == {"e1"}
    assert SHARP_BOOKS[0] == "pinnacle" and set(BOOK_NAMES) == set(SHARP_BOOKS)


def test_steam_lines_formatting():
    prev = _board(-2.5, 44.5, -150).assign(pulled_at=T0.timestamp())
    curr = _board(-3.5, 46.0, -175).assign(pulled_at=T0.timestamp() + 62 * 60)
    lines = steam_lines(moves(prev, curr))
    assert lines == ["📈 STEAM · DAL @ PHI · Pinnacle PHI ML -150 -> -175 (+3.6%) in 62 min",
                     "📈 STEAM · DAL @ PHI · Pinnacle PHI -2.5 -> -3.5 (+1.0) in 62 min",
                     "📈 STEAM · DAL @ PHI · Pinnacle total 44.5 -> 46 (+1.5) in 62 min"]
    assert steam_lines(moves(prev, curr, books=[])) == [] and steam_lines(None) == [] and steam_lines(pd.DataFrame()) == []
    # no pulled_at: no "in N min"; PK spreads; a book outside the small table passes through unless a table is given
    r = dict(matchup="DAL @ PHI", market="spreads", book="fanduel", from_line=0.0, to_line=-1.0, from_price=-110,
             to_price=-110, delta=1.0, minutes=np.nan)
    assert steam_line(r) == "📈 STEAM · DAL @ PHI · fanduel PHI PK -> -1 (+1.0)"
    assert steam_line(r, {"fanduel": "FanDuel"}) == "📈 STEAM · DAL @ PHI · FanDuel PHI PK -> -1 (+1.0)"
    r.update(market="totals", from_line=47.5, to_line=45.5, delta=-2.0, minutes=30.4)
    assert steam_line(r) == "📈 STEAM · DAL @ PHI · fanduel total 47.5 -> 45.5 (-2.0) in 30 min"
    r.update(market="ml", from_price=120, to_price=-105, delta=0.0668, minutes=None)
    assert steam_line(r) == "📈 STEAM · DAL @ PHI · fanduel PHI ML +120 -> -105 (+6.7%)"
