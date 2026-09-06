"""Offline tests for the cross-book arb / middle finder. No network (the run.py tests sit behind a dead proxy)."""
import os, subprocess, sys
import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from sharpmodel.margins import margin_pmf
from sharpmodel.middles import find_middles, game_fairs, prop_fairs, support_pmf, price_pair, COLS, FAIR_COLS, TYPES
from sharpmodel.odds import load_odds_csv, load_props_csv, find_ev, find_arbs, best_lines
from sharpmodel.props import price_props

EV = dict(event_id="e", commence="2026-09-06T17:00:00Z", home="PHI", away="DAL")
FAIRS = pd.DataFrame([dict(event_id="e", market="spreads", player=np.nan, mu=3.0, sd=np.nan, dist="nfl_margin"),
                      dict(event_id="e", market="ml", player=np.nan, mu=3.0, sd=np.nan, dist="nfl_margin"),
                      dict(event_id="e", market="totals", player=np.nan, mu=45.5, sd=10.3, dist="normal")])


def _odds(*rows):
    return pd.DataFrame([dict(EV, **r) for r in rows])


def _spread(pa=-110, pb=-110, book_b="B", la=-2.5, lb=3.5):
    return _odds(dict(book="A", market="spreads", side="home", line=la, price=pa),
                 dict(book=book_b, market="spreads", side="away", line=lb, price=pb))


def test_spread_middle_on_three_uses_the_key_number_pmf():                       # (a)
    m = find_middles(_spread(), FAIRS)
    assert list(m.columns) == COLS and len(m) == 1
    r = m.iloc[0]
    assert r.type == "middle" and r.window == "3" and r.market == "spreads" and pd.isna(r.player)
    assert r.p_middle == pytest.approx(margin_pmf(3.0)[3], abs=0.02)
    assert r.p_middle > 2 * margin_pmf(3.0, weights=None)[3]                     # the Normal calls this a loser
    assert r.miss_cost_pct == pytest.approx(4.5, abs=0.1) and r.win_both_pct == pytest.approx(90.9, abs=0.1)
    assert r.ev_pct > 0 and r.guaranteed_pct == pytest.approx(-r.miss_cost_pct)
    assert r.breakeven_p == pytest.approx(r.miss_cost_pct / (r.win_both_pct + r.miss_cost_pct))
    assert r.stake_a_pct == pytest.approx(50) and r.stake_b_pct == pytest.approx(50)
    assert r.bet_a == "PHI -2.5 -110 @ A" and r.bet_b == "DAL +3.5 -110 @ B" and r.matchup == "DAL @ PHI"
    assert not r.same_book and r.commence == EV["commence"]
    assert r.ev_pct == pytest.approx(r.p_middle * r.win_both_pct - (1 - r.p_middle) * r.miss_cost_pct)
    # the Normal says the same pair loses money
    normal = FAIRS.assign(dist="normal", sd=13.2)
    assert find_middles(_spread(), normal, min_ev=-100).iloc[0].ev_pct < 0


def test_even_money_pair_is_a_free_middle():                                     # (b)
    r = find_middles(_spread(100, 100), FAIRS).iloc[0]
    assert r.type == "free_middle" and r.guaranteed_pct == 0 and r.miss_cost_pct == 0 and r.ev_pct > 0
    assert r.ev_pct == pytest.approx(100 * r.p_middle) and r.win_both_pct == pytest.approx(100)
    assert r.breakeven_p == 0


def test_moneyline_arb():                                                        # (c)
    odds = _odds(dict(book="A", market="ml", side="home", line=np.nan, price=120),
                 dict(book="B", market="ml", side="away", line=np.nan, price=-105))
    m = find_middles(odds, FAIRS)
    r = m.iloc[0]
    assert len(m) == 1 and r.type == "arb" and r.guaranteed_pct > 0 and r.window == "" and r.p_middle == 0
    assert r.bet_a == "PHI ML +120 @ A" and r.bet_b == "DAL ML -105 @ B"
    hold = 100 / 220 + 105 / 205
    assert r.guaranteed_pct == pytest.approx(100 * (1 / hold - 1)) and r.miss_cost_pct == pytest.approx(-r.guaranteed_pct)
    assert r.stake_a_pct < r.stake_b_pct and r.stake_a_pct + r.stake_b_pct == pytest.approx(100)
    assert 0 < r.ev_pct <= r.guaranteed_pct + 1e-9                                # a tie refunds both legs
    assert len(find_arbs(odds)) == 1                                              # the old finder still agrees
    odds.loc[0, "price"] = -110                                                   # now a scalp: dropped
    assert find_middles(odds, FAIRS, min_ev=-100).empty


def test_totals_window_has_two_integers():                                       # (d)
    odds = _odds(dict(book="A", market="totals", side="over", line=44.5, price=-110),
                 dict(book="B", market="totals", side="under", line=46.5, price=-110))
    r = find_middles(odds, FAIRS).iloc[0]
    assert r.window == "45-46" and r.type == "middle" and r.ev_pct > 0
    k, p = support_pmf("normal", 45.5, 10.3)
    assert r.p_middle == pytest.approx(p[(k == 45) | (k == 46)].sum())
    assert r.bet_a == "O 44.5 -110 @ A" and r.bet_b == "U 46.5 -110 @ B"


def test_poisson_prop_window_and_name_normalisation():                           # (e)
    odds = _odds(dict(book="A", market="player_receptions", player="A.J. Brown", side="over", line=4.5, price=-110),
                 dict(book="B", market="player_receptions", player="AJ Brown", side="under", line=6.5, price=-110))
    fairs = pd.DataFrame([dict(event_id="e", market="player_receptions", player="A.J. Brown", mu=5.5, sd=np.nan,
                               dist="poisson")])
    m = find_middles(odds, fairs)
    r = m.iloc[0]
    assert len(m) == 1 and r.window == "5-6" and r.player == "A.J. Brown" and r.market == "player_receptions"
    assert r.p_middle == pytest.approx(poisson.pmf(5, 5.5) + poisson.pmf(6, 5.5))
    assert r.bet_a == "A. Brown O 4.5 -110 @ A" and r.bet_b == "A. Brown U 6.5 -110 @ B"
    assert find_middles(odds, fairs.assign(player="Someone Else")).empty         # no fair -> no pair


def test_gap_pairs_are_skipped():                                                # (f)
    odds = _odds(dict(book="A", market="totals", side="over", line=46.5, price=-110),
                 dict(book="B", market="totals", side="under", line=44.5, price=-110))
    assert find_middles(odds, FAIRS, min_ev=-100).empty
    assert find_middles(_spread(la=-3.5, lb=2.5), FAIRS, min_ev=-100).empty      # both lose on a 3
    assert find_middles(_spread(la=-2.5, lb=2.5), FAIRS, min_ev=-100).empty      # same half number: pure scalp


def test_same_book_excluded_by_default_and_flagged():                            # (g)
    odds = _spread(book_b="A")
    assert find_middles(odds, FAIRS).empty
    m = find_middles(odds, FAIRS, include_same_book=True)
    assert len(m) == 1 and bool(m.iloc[0].same_book) and m.iloc[0].type == "middle"


def test_integer_numbers_push_and_half_middles():
    # -3 / +3 at +105 both: no window, a 3 refunds both legs, every other number pays 1/hold - 1 -> arb
    r = find_middles(_spread(105, 105, la=-3.0, lb=3.0), FAIRS).iloc[0]
    assert r.type == "arb" and r.window == "" and r.guaranteed_pct > 0 and r.ev_pct < r.guaranteed_pct
    assert find_middles(_spread(-110, -110, la=-3.0, lb=3.0), FAIRS, min_ev=-100).empty   # same number, juice: scalp
    # -2.5 / +3: no both-win number, but a 3 wins A while B pushes
    r = find_middles(_spread(-105, -105, la=-2.5, lb=3.0), FAIRS).iloc[0]
    assert r.type == "half_middle" and r.window == "3p" and r.p_middle == pytest.approx(margin_pmf(3.0)[3])
    assert 0 < r.win_both_pct < 50 and r.ev_pct > 0 and r.guaranteed_pct < 0
    assert r.bet_b == "DAL +3 -105 @ B"
    assert find_middles(_spread(-110, -110, la=-2.5, lb=3.0), FAIRS).empty      # at -110 the push side loses money
    # -2 / +3.5: window is the 3 (both win); the push-win on 2 is inside ev_pct but not the headline window
    r = find_middles(_spread(la=-2.0, lb=3.5), FAIRS).iloc[0]
    assert r.type == "middle" and r.window == "3"
    k, p = support_pmf("nfl_margin", 3.0)
    pp = price_pair(k, p, 2.0, 3.5, -110, -110)
    assert list(pp["half"]) == [2] and pp["ev"] > pp["p_middle"] * pp["win_both"] - (1 - pp["p_middle"]) * pp["miss_cost"]


def test_bernoulli_yes_no_arb_only():
    odds = _odds(dict(book="A", market="player_anytime_td", player="Saquon Barkley", side="yes", line=np.nan, price=110),
                 dict(book="B", market="player_anytime_td", player="Saquon Barkley", side="no", line=np.nan, price=100))
    fairs = pd.DataFrame([dict(event_id="e", market="player_anytime_td", player="Saquon Barkley", mu=0.7, sd=np.nan,
                               dist="bernoulli")])
    r = find_middles(odds, fairs).iloc[0]
    assert r.type == "arb" and r.window == "" and r.guaranteed_pct > 0 and r.ev_pct == pytest.approx(r.guaranteed_pct)
    assert r.bet_a == "S. Barkley YES +110 @ A" and r.bet_b == "S. Barkley NO +100 @ B"


def test_best_price_per_book_and_ranking():
    odds = _odds(dict(book="A", market="spreads", side="home", line=-2.5, price=-115),
                 dict(book="A", market="spreads", side="home", line=-2.5, price=-105),   # stale duplicate: best kept
                 dict(book="B", market="spreads", side="away", line=3.5, price=-110),
                 dict(book="C", market="spreads", side="away", line=3.5, price=100),
                 dict(book="A", market="ml", side="home", line=np.nan, price=120),
                 dict(book="B", market="ml", side="away", line=np.nan, price=-105),
                 dict(book="A", market="totals", side="over", line=44.5, price=100),
                 dict(book="B", market="totals", side="under", line=46.5, price=100))
    m = find_middles(odds, FAIRS)
    assert list(m.type) == ["arb", "free_middle", "middle"]                      # cannot-lose first, then by EV
    assert m.guaranteed_pct.iloc[0] > 0 and m.guaranteed_pct.iloc[1] == 0
    r = m.iloc[2]                                                                # -2.5 / +3.5: best pairing is A vs C
    assert r.bet_b == "DAL +3.5 +100 @ C" and "-105 @ A" in r.bet_a and not any("-115" in b for b in m.bet_a)
    assert r.n_alt == 1 and r.alt == "B B -110"                                  # the B pairing folded into alt
    assert (m.n_alt.iloc[:2] == 0).all() and (m.alt.iloc[:2] == "").all()
    every = find_middles(odds, FAIRS, collapse=False)                            # collapse=False lists every pairing
    assert list(every.type) == ["arb", "free_middle", "middle", "middle"] and (every.n_alt == 0).all()
    assert every.ev_pct.iloc[2] > every.ev_pct.iloc[3]
    high = find_middles(odds, FAIRS, min_ev=50)                                    # arbs / free middles survive any threshold
    assert list(high.type) == ["arb", "free_middle"]
    assert set(m.type) <= set(TYPES)


def test_collapse_keeps_one_row_per_pair_of_numbers_and_exclude_drops_legs():
    """One stale price against many books is one opportunity: the live scan showed the same marathonbet ML paired
    with 28 books as 28 'arbs'. Excluded books cannot be a leg but still shape the fairs (built by the caller)."""
    rows = [dict(book="stale", market="ml", side="home", line=np.nan, price=-159)]
    rows += [dict(book=b, market="ml", side="away", line=np.nan, price=p)
             for b, p in (("x", 170), ("y", 167), ("z", 165), ("w", 165), ("v", 160), ("u", 160))]
    m = find_middles(_odds(*rows), FAIRS)
    assert len(m) == 1 and m.iloc[0].type == "arb" and m.iloc[0].book_b == "x" and m.iloc[0].n_alt == 5
    assert m.iloc[0].alt == "B y +167; B w +165; B z +165; B u +160 ..."          # EV order, ties by book key
    ex = find_middles(_odds(*rows), FAIRS, exclude=["stale"])
    assert ex.empty and list(ex.columns) == COLS
    ex = find_middles(_odds(*rows), FAIRS, exclude=["x", "y"])
    assert len(ex) == 1 and ex.iloc[0].book_b == "w" and ex.iloc[0].n_alt == 3          # +165 tie -> first book key
    # a leg at a different NUMBER is a different row, not an alternative
    two = find_middles(_odds(dict(book="A", market="spreads", side="home", line=-2.5, price=-110),
                             dict(book="B", market="spreads", side="away", line=3.5, price=-110),
                             dict(book="C", market="spreads", side="away", line=3.0, price=100)), FAIRS, min_ev=-100)
    assert len(two) == 2 and (two.n_alt == 0).all()


def test_non_unique_index_from_a_concat_is_reset():
    """pd.concat of two CSVs repeats index labels; the label-based best-price reduction then re-admitted the worse
    duplicate and emitted every pair several times (8 rows for what should be 2)."""
    a = _odds(dict(book="A", market="spreads", side="home", line=-2.5, price=-115),
              dict(book="B", market="spreads", side="away", line=3.5, price=-110))
    b = _odds(dict(book="A", market="spreads", side="home", line=-2.5, price=-105),
              dict(book="C", market="spreads", side="away", line=3.5, price=100))
    cat = pd.concat([a, b])                                                      # index 0, 1, 0, 1
    assert not cat.index.is_unique
    clean = find_middles(pd.concat([a, b], ignore_index=True), FAIRS, collapse=False)
    m = find_middles(cat, FAIRS, collapse=False)
    pd.testing.assert_frame_equal(m, clean)
    assert len(m) == 2 and all("-105 @ A" in x for x in m.bet_a) and set(m.bet_b) == {"DAL +3.5 -110 @ B", "DAL +3.5 +100 @ C"}
    pd.testing.assert_frame_equal(find_middles(pd.concat([cat, cat]), FAIRS, collapse=False), clean)   # the same CSV loaded twice
    one = find_middles(cat, FAIRS)                                               # collapsed: one row, B folded in
    assert len(one) == 1 and one.iloc[0].bet_b == "DAL +3.5 +100 @ C" and one.iloc[0].n_alt == 1
    assert cat.index.tolist() == [0, 1, 0, 1]                                    # the caller's frame is untouched


def test_blank_price_rows_are_dropped_not_fatal():
    """A blank price (hand-captured CSV) is NaN: idxmax on an all-NaN group used to raise KeyError."""
    odds = pd.concat([_spread(), _odds(dict(book="C", market="spreads", side="away", line=4.5, price=np.nan))],
                     ignore_index=True)
    pd.testing.assert_frame_equal(find_middles(odds, FAIRS), find_middles(_spread(), FAIRS))
    bl = best_lines(odds)
    assert len(bl) == 2 and bl.price.notna().all() and "C" not in set(bl.book)
    assert find_middles(odds.assign(price=np.nan), FAIRS, min_ev=-100).empty and best_lines(odds.assign(price=np.nan)).empty


def test_empty_or_column_less_frames_return_empty_tables():
    """A column-less pd.DataFrame() (no lines at all) raised AttributeError / KeyError instead of an empty table."""
    named = pd.DataFrame(columns=["event_id", "commence", "home", "away", "book", "market", "side", "line", "price"])
    for empty in (None, pd.DataFrame(), named):
        m = find_middles(empty, FAIRS)
        assert m.empty and list(m.columns) == COLS
        f = game_fairs(empty, "nfl")
        assert f.empty and list(f.columns) == FAIR_COLS
    assert find_middles(pd.DataFrame(), game_fairs(pd.DataFrame())).empty         # run.py's path on an empty board
    for no_fair in (None, pd.DataFrame(), pd.DataFrame(columns=FAIR_COLS)):
        m = find_middles(_spread(), no_fair)
        assert m.empty and list(m.columns) == COLS


def test_support_pmf_sums_to_one():
    for dist, mu, sd in (("nfl_margin", 3.0, None), ("nfl_margin", -6.5, 13.4), ("normal", 45.5, 10.3),
                         ("normal", 264.5, 80.0), ("poisson", 5.5, None), ("poisson", 0.3, None), ("bernoulli", 0.7, None)):
        k, p = support_pmf(dist, mu, sd)
        assert p.sum() == pytest.approx(1.0, abs=1e-9) and (p >= 0).all() and k.dtype.kind == "i"
    k, p = support_pmf("nfl_margin", 3.0)
    assert p[k == 3][0] == pytest.approx(margin_pmf(3.0)[3])
    with pytest.raises(ValueError):
        support_pmf("normal", 45.5, np.nan)
    with pytest.raises(ValueError):
        support_pmf("gamma", 1.0, 1.0)


def test_game_fairs_match_find_ev():
    odds = load_odds_csv("lines_template.csv")
    model = pd.DataFrame([dict(home="PHI", away="DAL", model_margin=7.0, model_total=40.0)])
    for mf, w in ((None, 0.25), (model, 0.5)):
        gf = game_fairs(odds, "nfl", mf, w)
        assert list(gf.columns) == FAIR_COLS and gf.player.isna().all() and len(gf) == 3
        f = gf.set_index("market")
        ev = find_ev(odds, "nfl", model_fair=mf, model_weight=w, min_ev=-1.0)
        for mk in ("spreads", "totals", "ml"):
            assert f.loc[mk, "mu"] == pytest.approx(ev[ev.market == mk].fair_mu.iloc[0])
        assert f.loc["spreads", "dist"] == f.loc["ml", "dist"] == "nfl_margin" and f.loc["totals", "dist"] == "normal"
    cfb = game_fairs(odds, "cfb")
    assert (cfb.dist == "normal").all() and cfb.set_index("market").loc["spreads", "sd"] == 16.5
    # the template's only pair: pinnacle O 47.5 / fanduel U 48.5 middles on 48
    m = find_middles(odds, game_fairs(odds, "nfl"), min_ev=-100)
    assert len(m) == 1 and m.iloc[0].window == "48" and m.iloc[0].type == "middle"


def test_prop_fairs_from_board_drop_projection_only_rows():
    po = load_props_csv("props_template.csv")
    proj = pd.DataFrame([dict(player_id="x", player="Dallas Goedert", team="PHI", opponent="DAL", position="TE",
                              market="player_reception_yds", proj_mean=45.0, proj_sd=30.0, n_games=20,
                              opp_factor=1.0, env_factor=1.0)])
    board = price_props(pd.concat([po, pd.DataFrame([dict(EV, event_id="PHI@DAL", book="dk", market="player_reception_yds",
                                                          player="Dallas Goedert", side="over", line=30.5, price=-110)])]),
                        proj, min_ev=-1.0)
    assert (board[board.player == "Dallas Goedert"]["flags"] == "no_market").all()
    f = prop_fairs(board)
    assert list(f.columns) == FAIR_COLS and "Dallas Goedert" not in set(f.player)   # projection alone never prices a middle
    assert len(f) == 4 and f.set_index("market").loc["player_pass_yds", "dist"] == "normal"
    assert f.set_index("market").loc["player_receptions", "dist"] == "poisson"
    assert prop_fairs(None).empty and prop_fairs(pd.DataFrame()).empty
    m = find_middles(po, f)
    hurts = m[m.market == "player_pass_yds"].iloc[0]
    assert hurts.window == "265-274" and hurts.type == "middle" and "fanduel" in hurts.bet_a and "draftkings" in hurts.bet_b


# ---------------- run.py (subprocess; a dead proxy makes any network attempt fail loudly) ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE = dict(os.environ, HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="", ODDS_API_KEY="")


def _run(mode, args, cwd):
    return subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "nfl", mode, "2026", "1"] + args,
                          cwd=cwd, env=OFFLINE, capture_output=True, text=True, timeout=120)


def test_run_ev_and_props_write_middles_csvs(tmp_path):                          # (h)
    r = _run("ev", ["--csv", os.path.join(ROOT, "lines_template.csv"), "--nomodel"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "=== +EV LINES" in r.stdout and "=== ARBS & MIDDLES" in r.stdout and "O 47.5 -104 @ pinnacle" in r.stdout
    m = pd.read_csv(tmp_path / "middles_nfl_2026_w1.csv")
    assert list(m.columns) == COLS and len(m) == 1 and m.iloc[0].window == 48 and m.iloc[0].type == "middle"
    # --exclude drops a leg's book (pinnacle is leg A of the only middle); --weight 0 == --nomodel; --hours keeps
    # rows without a kickoff
    r = _run("ev", ["--csv", os.path.join(ROOT, "lines_template.csv"), "--weight", "0", "--exclude", "nonus",
                    "--hours", "1"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "books excluded as legs" in r.stdout and "fair = sharp book only" in r.stdout
    assert "=== +EV LINES" in r.stdout and "@ pinnacle" not in r.stdout
    assert pd.read_csv(tmp_path / "middles_nfl_2026_w1.csv").empty
    r = _run("props", ["--csv", os.path.join(ROOT, "props_template.csv"), "--nomodel"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "=== +EV PROPS" in r.stdout and "=== PROP ARBS & MIDDLES" in r.stdout
    pm = pd.read_csv(tmp_path / "props_middles_nfl_2026_w1.csv")
    assert list(pm.columns) == COLS and set(pm.type) <= set(TYPES)
    assert (pm[pm.market == "player_pass_yds"].window == "265-274").all() and len(pm[pm.market == "player_pass_yds"]) == 1
