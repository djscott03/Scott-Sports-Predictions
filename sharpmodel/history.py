"""
Line history: keep every board pull, and read the moves between two pulls (steam).

  append_snapshot(odds, league, folder, pulled_at)  -> folder/<league>/<YYYY-MM-DD>.parquet (New York date), one row
                                                       per price per pull; the day file is read, appended, rewritten
  load_history(folder, league, days=7)              -> the last N day files as one frame (empty frame when none)
  moves(prev, curr, books=SHARP_BOOKS)              -> the sharp books' line moves between two snapshots
  steam_lines(mv)                                   -> "📈 STEAM · DAL @ PHI · Pinnacle PHI -2.5 -> -3.5 (+1.0) in 62 min"

A pull is ~2.5k rows (272 games x 10 books x 3 markets x 2 sides, fewer inside the window), so a day of the alerts
cron (at most 10 pulls) is a few hundred KB of parquet and a season is under 100 MB. The alerts Action writes
--history history and the `board` branch carries it (alerts.yml); nothing here talks to the network.

`delta` is in the model's own units, so a positive number always means "toward the home side / the over":
a spread is the HOME line and delta = from_line - to_line (PHI -2.5 -> -3.5 is +1.0 of home margin), a total is
the over number and delta = to_line - from_line, a moneyline is the home price and delta is the change in its
implied probability (vig included; the same book both times, so the hold cancels). One move per event / book /
market: the home spread, the over, the home ML -- the other side is the same move.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd
from .odds import SHARP_BOOKS
from .pricing import american_to_prob

TZ = "America/New_York"
COLS = ["pulled_at", "event_id", "commence", "home", "away", "book", "market", "side", "line", "price", "updated"]
TEXT = ["event_id", "commence", "home", "away", "book", "market", "side", "updated"]
MOVE_COLS = ["event_id", "matchup", "market", "book", "side", "from_line", "to_line", "from_price", "to_price",
             "delta", "minutes"]
# the sharp books' names on the apps (alerts.BOOK_NAMES has every book; pass it as `names` for the full table)
BOOK_NAMES = {"pinnacle": "Pinnacle", "circasports": "Circa", "betonlineag": "BetOnline", "bookmaker": "Bookmaker",
              "lowvig": "LowVig"}
_SIDE = {"spreads": "home", "totals": "over", "ml": "home"}     # the side a move is reported on


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp.now(tz="UTC") if t is None else pd.Timestamp(t, unit="s") if isinstance(t, (int, float)) \
        else pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def day_file(folder: str, league: str, pulled_at=None) -> str:
    """folder/<league>/<YYYY-MM-DD>.parquet, the date in New York time (a Sunday's pulls stay one file past 8pm ET)."""
    return os.path.join(folder, league, f"{_utc(pulled_at).tz_convert(TZ):%Y-%m-%d}.parquet")


def _rows(odds: pd.DataFrame, pulled_at) -> pd.DataFrame:
    """The pull in the history schema: COLS in order, text columns as object/None, numbers as float."""
    out = pd.DataFrame(index=range(len(odds)))
    out["pulled_at"] = float(_utc(pulled_at).timestamp())
    for c in COLS[1:]:
        s = odds[c].reset_index(drop=True) if c in odds else pd.Series([np.nan] * len(odds))
        if c in TEXT:
            out[c] = s.astype(object).where(s.notna(), None)
        else:
            out[c] = pd.to_numeric(s, errors="coerce").astype(float)
    return out


def append_snapshot(odds: pd.DataFrame, league: str, folder: str, pulled_at=None) -> str:
    """Append one pull to its day file (read + concat + rewrite: the file is small) and return the path. An empty
    board (off-season) adds no rows but the day file exists. pulled_at: Timestamp / iso string / epoch s; None = now."""
    path = day_file(folder, league, pulled_at)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = _rows(odds if odds is not None else pd.DataFrame(), pulled_at)
    if os.path.exists(path):
        parts = [f for f in (pd.read_parquet(path).reindex(columns=COLS), new) if len(f)]   # no all-empty concat
        new = pd.concat(parts, ignore_index=True) if len(parts) > 1 else (parts[0] if parts else new)
    new.to_parquet(path, index=False)
    return path


def load_history(folder: str, league: str, days: int = 7) -> pd.DataFrame:
    """The last `days` day files as one frame (COLS), oldest first; an empty frame with the columns when there are none."""
    files = sorted(glob.glob(os.path.join(folder, league, "*.parquet")))[-max(int(days), 0):]
    frames = [pd.read_parquet(f).reindex(columns=COLS) for f in files]
    frames = [f for f in frames if len(f)]
    if not frames: return pd.DataFrame(columns=COLS)
    return pd.concat(frames, ignore_index=True)


def _side(df: pd.DataFrame, market: str, books: list) -> pd.DataFrame:
    """One row per event / book for the reported side of `market` (the first one when a book posts several)."""
    d = df[(df.market == market) & (df.side == _SIDE[market]) & df.book.isin(books)]
    d = d.drop_duplicates(["event_id", "book"])
    return d[["event_id", "home", "away", "book", "line", "price"]]


def moves(prev: pd.DataFrame, curr: pd.DataFrame, books=None, min_spread: float = 1.0, min_total: float = 1.5,
          min_ml_prob: float = 0.03) -> pd.DataFrame:
    """Line moves at `books` (default SHARP_BOOKS) between two snapshots -> MOVE_COLS, the biggest move (relative to
    its threshold) first. A spread is reported once per event / book on the home side, a total on the over, a
    moneyline on the home price by implied probability; smaller moves than the thresholds are not moves.
    `minutes` is the gap between the snapshots' pulled_at (NaN when either lacks the column, e.g. a hand CSV)."""
    books = list(SHARP_BOOKS if books is None else books)
    need = {"event_id", "home", "away", "book", "market", "side", "line", "price"}
    if prev is None or curr is None or not len(prev) or not len(curr) or not need <= set(prev) or not need <= set(curr):
        return pd.DataFrame(columns=MOVE_COLS)
    gap = np.nan
    if "pulled_at" in prev and "pulled_at" in curr:
        t0, t1 = pd.to_numeric(prev.pulled_at, errors="coerce").max(), pd.to_numeric(curr.pulled_at, errors="coerce").max()
        if pd.notna(t0) and pd.notna(t1): gap = (float(t1) - float(t0)) / 60
    out = []
    for market, floor in (("spreads", min_spread), ("totals", min_total), ("ml", min_ml_prob)):
        a, b = _side(prev, market, books), _side(curr, market, books)
        if not len(a) or not len(b): continue
        m = a.merge(b, on=["event_id", "book"], suffixes=("_from", "_to"))
        if not len(m): continue
        fl, tl = m.line_from.astype(float), m.line_to.astype(float)
        fp, tp = m.price_from.astype(float), m.price_to.astype(float)
        if market == "spreads": delta = fl - tl                                 # home margin: -2.5 -> -3.5 is +1.0
        elif market == "totals": delta = tl - fl
        else: delta = tp.map(american_to_prob) - fp.map(american_to_prob)
        keep = delta.abs() >= floor - 1e-9
        if not keep.any(): continue
        m = m[keep]
        out.append(pd.DataFrame(dict(event_id=m.event_id.values,
                                     matchup=(m.away_to.astype(str) + " @ " + m.home_to.astype(str)).values,
                                     market=market, book=m.book.values, side=_SIDE[market],
                                     from_line=fl[keep].values, to_line=tl[keep].values,
                                     from_price=fp[keep].values, to_price=tp[keep].values,
                                     delta=delta[keep].values, minutes=gap, _rank=(delta[keep].abs() / floor).values)))
    if not out: return pd.DataFrame(columns=MOVE_COLS)
    df = pd.concat(out, ignore_index=True).sort_values(["_rank", "event_id", "book"], ascending=[False, True, True])
    return df[MOVE_COLS].reset_index(drop=True)


def book_name(b, names=None) -> str:
    return (names or BOOK_NAMES).get(str(b), str(b))


def _spread(x) -> str:
    x = float(x)
    return "PK" if x == 0 else format(x, "+g")


def steam_line(r: dict, names=None) -> str:
    """📈 STEAM · DAL @ PHI · Pinnacle PHI -2.5 -> -3.5 (+1.0) in 62 min
       📈 STEAM · DAL @ PHI · Pinnacle total 44.5 -> 46 (+1.5) in 62 min
       📈 STEAM · DAL @ PHI · Pinnacle PHI ML -150 -> -175 (+3.5%) in 62 min"""
    mk, home = r["market"], str(r["matchup"]).rpartition(" @ ")[2]
    bk, d = book_name(r["book"], names), float(r["delta"])
    if mk == "spreads": move = f"{bk} {home} {_spread(r['from_line'])} -> {_spread(r['to_line'])} ({d:+.1f})"
    elif mk == "totals": move = f"{bk} total {float(r['from_line']):g} -> {float(r['to_line']):g} ({d:+.1f})"
    else: move = f"{bk} {home} ML {int(round(float(r['from_price']))):+d} -> {int(round(float(r['to_price']))):+d} ({100 * d:+.1f}%)"
    mins = r.get("minutes")
    if mins is not None and pd.notna(mins) and float(mins) > 0: move += f" in {int(round(float(mins)))} min"
    return f"📈 STEAM · {r['matchup']} · {move}"


def steam_lines(mv: pd.DataFrame, names=None) -> list:
    """One line per row of moves(); `names` = a fuller book-name table (alerts.BOOK_NAMES)."""
    if mv is None or not len(mv): return []
    return [steam_line(r, names) for r in mv.to_dict("records")]
