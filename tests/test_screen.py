"""The odds screen: one row per game x market x pick (or player prop), one column per book."""
import os
import numpy as np
import pandas as pd

from sharpmodel.odds import odds_grid, load_props_csv, load_odds_csv, US_BOOKS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = dict(event_id="e", commence="2026-09-13T17:00:00Z", home="PHI", away="DAL")


def _row(book, market, side, line, price):
    return dict(EV, book=book, market=market, side=side, line=line, price=price)


def _board():
    return pd.DataFrame([
        _row("pinnacle", "spreads", "home", -2.5, -105), _row("pinnacle", "spreads", "away", 2.5, -115),
        _row("draftkings", "spreads", "home", -2.5, -110), _row("draftkings", "spreads", "away", 2.5, -110),
        _row("fanduel", "spreads", "home", -3.5, -110), _row("fanduel", "spreads", "away", 3.5, -110),   # off the number
        _row("draftkings", "spreads", "home", -3.0, 100),                                                 # alternate line
        _row("draftkings", "totals", "over", 47.5, -110), _row("draftkings", "totals", "under", 47.5, -110),
        _row("fanduel", "totals", "over", 47.5, -105), _row("fanduel", "totals", "under", 47.5, -115),
        _row("draftkings", "ml", "home", np.nan, -140), _row("draftkings", "ml", "away", np.nan, 120),
        _row("fanduel", "ml", "home", np.nan, -135), _row("fanduel", "ml", "away", np.nan, 115),
    ])


def test_grid_layout_cells_and_flags():
    g, best, off = odds_grid(_board(), masks=True)
    assert list(g.columns) == ["commence", "matchup", "market", "pick", "draftkings", "fanduel", "pinnacle", "best", "books"]
    assert list(g.market.drop_duplicates()) == ["spreads", "totals", "ml"]           # MARKET_ORDER
    assert list(g[g.market == "spreads"].pick) == ["PHI", "DAL"]                    # home first
    r = g[(g.market == "spreads") & (g.pick == "PHI")].iloc[0]
    assert r.draftkings == "-2.5 -110" and r.pinnacle == "-2.5 -105" and r.fanduel == "-3.5 -110"   # alt -3 not shown
    assert r.best == "-105 @ pinnacle" and r.books == 3 and r.matchup == "DAL @ PHI"
    i = r.name
    assert best.loc[i, "pinnacle"] and not best.loc[i, "draftkings"]
    assert off.loc[i, "fanduel"] and not off.loc[i, "draftkings"]
    t = g[(g.market == "totals") & (g.pick == "over")].iloc[0]
    assert t.draftkings == "47.5 -110" and t.fanduel == "47.5 -105" and t.best == "-105 @ fanduel" and t.pinnacle == ""
    m = g[(g.market == "ml") & (g.pick == "DAL")].iloc[0]
    assert m.draftkings == "+120" and m.fanduel == "+115" and m.best == "+120 @ draftkings"
    assert not off.loc[m.name].any()                                                # no numbers on a moneyline
    assert list(best.columns) == ["draftkings", "fanduel", "pinnacle"] and len(best) == len(g) == len(off)
    assert not g.attrs                                                              # pandas 3 compares attrs on concat
    pd.testing.assert_frame_equal(odds_grid(_board()), g)                          # masks=False: the grid alone


def test_grid_exclude_empty_and_props():
    ex = odds_grid(_board(), exclude=["pinnacle"])
    assert "pinnacle" not in ex.columns
    r = ex[(ex.market == "spreads") & (ex.pick == "PHI")].iloc[0]
    assert r.books == 2 and r.best.endswith("@ fanduel")        # 1-1 tie on the number -> nearest the mean, then lower
    # an alternate line never sets the market number: draftkings' -3 +100 is an alternate, its main line is -2.5 -110
    alt, _, alt_off = odds_grid(_board()[lambda d: d.book != "fanduel"], masks=True)
    r = alt[(alt.market == "spreads") & (alt.pick == "PHI")].iloc[0]
    assert r.draftkings == "-2.5 -110" and r.best == "-105 @ pinnacle" and not alt_off.loc[r.name].any()
    assert odds_grid(pd.DataFrame()).empty and odds_grid(None).empty
    assert all(x.empty for x in odds_grid(None, masks=True)) and len(odds_grid(None, masks=True)) == 3
    assert odds_grid(_board().assign(price=np.nan)).empty
    p, _, p_off = odds_grid(load_props_csv(os.path.join(ROOT, "props_template.csv")), masks=True)
    assert list(p.columns[:5]) == ["commence", "matchup", "market", "player", "pick"]
    hurts = p[(p.player == "Jalen Hurts") & (p.pick == "over")].iloc[0]
    assert hurts.books == 2 and hurts.draftkings.startswith("274.5") and hurts.fanduel.startswith("264.5")
    assert p_off.loc[hurts.name].any()                                             # the stale book is off the number
    lines = odds_grid(load_odds_csv(os.path.join(ROOT, "lines_template.csv")))
    assert {"pinnacle", "fanduel"} <= set(lines.columns) and (lines.books >= 1).all()
    assert US_BOOKS[0] == "draftkings"
