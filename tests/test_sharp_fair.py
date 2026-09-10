"""sharp_fair on the +EV board: NFL spreads / moneylines inverted and priced on the key-number pmf, the sharp books
averaged by prior x freshness. Offline and deterministic (ages are add_age'd at a fixed NOW; CSV rows have none)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from sharpmodel.margins import (NFL_SD, NFL_KEY_WEIGHTS, margin_pmf, cover_probs_emp, moneyline_prob_emp,
                                implied_margin_emp, implied_margin_ml_emp)
from sharpmodel.pricing import LEAGUE_SD, devig, cover_probs, prob_to_american
from sharpmodel.odds import (sharp_fair, blended_fair, find_ev, top_picks, margin_dist, add_age, load_odds_csv,
                             SHARP_BOOKS, SHARP_WEIGHTS, FRESH_TAU_MIN, FRESH_FLOOR, _freshness)
from sharpmodel.middles import game_fairs

NOW = pd.Timestamp("2026-09-13T16:00:00Z")
EV = dict(event_id="e", commence="2026-09-13T17:00:00Z", home="PHI", away="DAL")


def _at(minutes):
    return (NOW - pd.Timedelta(minutes=minutes)).isoformat()


def _spread(book, line, ph=-110, pa=-110, updated=None):
    return [dict(EV, book=book, market="spreads", side="home", line=line, price=ph, updated=updated),
            dict(EV, book=book, market="spreads", side="away", line=-line, price=pa, updated=updated)]


def _ml(book, ph, pa, updated=None):
    return [dict(EV, book=book, market="ml", side="home", line=np.nan, price=ph, updated=updated),
            dict(EV, book=book, market="ml", side="away", line=np.nan, price=pa, updated=updated)]


def _total(book, line, po=-110, pu=-110, updated=None):
    return [dict(EV, book=book, market="totals", side="over", line=line, price=po, updated=updated),
            dict(EV, book=book, market="totals", side="under", line=line, price=pu, updated=updated)]


def _frame(*groups, aged=True):
    df = pd.DataFrame([r for g in groups for r in g])
    return add_age(df, now=NOW) if aged else df


# ---------------- the inversions ----------------
def test_implied_margin_round_trips_cover_probs_emp():
    for line in (-10.0, -7.0, -3.0, -2.5, 0.0, 3.0, 6.5, 14.0):
        for p in (0.3, 0.45, 0.5, 0.6, 0.75):
            mu = implied_margin_emp(line, p)
            cp = cover_probs_emp(mu, line)
            assert cp["win"] / (1 - cp["push"]) == pytest.approx(p, abs=1e-6), (line, p)
    # sd / weights pass through: weights=None on a half-point line is the Normal's closed form
    for line, p in ((-2.5, 0.55), (6.5, 0.4)):
        assert implied_margin_emp(line, p, sd=13.4, weights=None) == pytest.approx(-line + 13.4 * norm.ppf(p), abs=1e-6)
    assert implied_margin_emp(-3.0, 0.5, sd=10.0) != pytest.approx(implied_margin_emp(-3.0, 0.5), abs=1e-3)


def test_implied_margin_ml_round_trips_moneyline_prob_emp():
    for p in (0.1, 0.25, 0.5, 0.6, 0.85, 0.95):
        mu = implied_margin_ml_emp(p)
        assert moneyline_prob_emp(mu) == pytest.approx(p, abs=1e-6)
    assert implied_margin_ml_emp(0.5) > 0 and implied_margin_ml_emp(0.5) < 0.2       # ties are not a home win
    assert implied_margin_ml_emp(0.5, weights=None) > 0
    assert implied_margin_ml_emp(0.75) > implied_margin_ml_emp(0.6) > implied_margin_ml_emp(0.5)


def test_half_point_off_three_is_worth_far_more_than_the_normal_said():
    """Fair -3 at -100: what is -2.5 worth? The half point wins exactly the push mass on 3 -- 7.9% on the pmf vs 3.0%
    on the Normal (~2.6x; ~17 cents vs ~6). At 7 the fitted bump is smaller (weight 1.76 vs 2.61): ~1.8x, 11 vs 6."""
    for line, lo, hi in ((-3.0, 2.3, 3.0), (-7.0, 1.5, 2.1)):
        mu_e, mu_n = implied_margin_emp(line, 0.5), -line
        gain_e = cover_probs_emp(mu_e, line + 0.5)["win"] - 0.5              # -2.5 is a coin flip plus the push mass
        gain_n = cover_probs(mu_n, line + 0.5, NFL_SD)["win"] - 0.5
        k = int(-line)                                                        # the half point wins exactly the pushes
        assert gain_e == pytest.approx(margin_pmf(mu_e)[k] / 2, abs=1e-9)
        assert gain_n == pytest.approx(margin_pmf(mu_n, weights=None)[k] / 2, abs=1e-9)
        assert lo < gain_e / gain_n < hi, (line, gain_e / gain_n)
    # a -3 -110/-110 lands near 3.0 either way (Normal: exactly 3; pmf: the mean sits a hair above the number)
    mu = implied_margin_emp(-3.0, devig(-110, -110)[0])
    assert 3.0 < mu < 3.15
    assert prob_to_american(cover_probs_emp(mu, -2.5)["win"]) < -114      # -2.5 fair ~ -117 (market: -115 to -120)
    assert prob_to_american(cover_probs(3.0, -2.5, NFL_SD)["win"]) > -108   # the Normal said ~ -106


def test_margin_dist_is_the_single_source_of_truth():
    nfl, cfb = margin_dist("nfl"), margin_dist("cfb")
    assert nfl == {"kind": "nfl_margin", "sd": NFL_SD, "weights": NFL_KEY_WEIGHTS}
    assert cfb == {"kind": "normal", "sd": LEAGUE_SD["cfb"]["spread"], "weights": None}
    odds = load_odds_csv("lines_template.csv")
    for league in ("nfl", "cfb"):
        gf = game_fairs(odds, league).set_index("market")
        assert gf.loc["spreads", "dist"] == gf.loc["ml", "dist"] == margin_dist(league)["kind"]
        assert gf.loc["spreads", "sd"] == margin_dist(league)["sd"] and gf.loc["totals", "dist"] == "normal"


def test_freshness_weight():
    assert _freshness(np.nan) == 1.0 and _freshness(None) == 1.0 and _freshness(0) == 1.0
    assert _freshness(FRESH_TAU_MIN) == pytest.approx(np.exp(-1)) and _freshness(-5) == 1.0    # a clock skew is not a bonus
    assert _freshness(360) == FRESH_FLOOR and _freshness(360, newest=360) == 1.0   # floored; the freshest book keeps full weight
    assert _freshness(30, newest=10) == pytest.approx(np.exp(-20 / FRESH_TAU_MIN))  # only the LAG behind the peers counts
    assert [SHARP_WEIGHTS[b] for b in SHARP_BOOKS] == [3.0, 2.0, 1.5, 1.0, 1.0]


# ---------------- sharp_fair ----------------
def test_one_sharp_book_is_the_direct_inversion():                                                  # (a)
    """A single sharp book: the weighted mean is that book's inverted price, whatever its weight."""
    q_sp, q_ml, q_t = devig(-108, -102)[0], devig(-160, 140)[0], devig(-104, -116)[0]
    for book in ("pinnacle", "betonlineag", "lowvig"):
        f = sharp_fair(_frame(_spread(book, -3.0, -108, -102), _ml(book, -160, 140), _total(book, 44.5, -104, -116),
                              aged=False), "nfl")
        assert f["ref_book"] == book and f["ref_n"] == 1 and np.isnan(f["ref_age"])
        assert f["mu_margin"] == pytest.approx(implied_margin_emp(-3.0, q_sp), abs=1e-9)
        assert f["mu_ml"] == pytest.approx(implied_margin_ml_emp(q_ml), abs=1e-9)
        assert f["mu_total"] == pytest.approx(44.5 + LEAGUE_SD["nfl"]["total"] * norm.ppf(q_t), abs=1e-9)
    # each market on its own: a spread-only book shapes only mu_margin, an ML-only book only mu_ml
    f = sharp_fair(_frame(_spread("pinnacle", -3.0, -108, -102), _ml("betonlineag", -160, 140), aged=False), "nfl")
    assert f["mu_margin"] == pytest.approx(implied_margin_emp(-3.0, q_sp), abs=1e-9)
    assert f["mu_ml"] == pytest.approx(implied_margin_ml_emp(q_ml), abs=1e-9)
    assert f["ref_book"] == "pinnacle" and f["ref_n"] == 2 and np.isnan(f["mu_total"])
    # no spread anywhere: the margin comes from the moneyline, as before
    f = sharp_fair(_frame(_ml("pinnacle", -160, 140), aged=False), "nfl")
    assert f["mu_margin"] == f["mu_ml"] == pytest.approx(implied_margin_ml_emp(q_ml), abs=1e-9)
    assert f["ref_book"] == "pinnacle" and f["ref_n"] == 1
    assert sharp_fair(pd.DataFrame(), "nfl")["ref_book"] is None


def test_two_sharp_books_weighted_by_prior_and_freshness():                                         # (b)
    mu_p, mu_b = implied_margin_emp(-3.0, 0.5), implied_margin_emp(-4.0, 0.5)
    # six-hour-old Pinnacle next to a two-minute-old BetOnline: the fair lands next to BetOnline
    f = sharp_fair(_frame(_spread("pinnacle", -3.0, updated=_at(360)), _spread("betonlineag", -4.0, updated=_at(2))), "nfl")
    w_p, w_b = 3.0 * FRESH_FLOOR, 1.5 * 1.0        # Pinnacle 358 min behind the freshest -> floored at a fifth; BetOnline full
    assert f["mu_margin"] == pytest.approx((w_p * mu_p + w_b * mu_b) / (w_p + w_b), abs=1e-9)
    assert abs(f["mu_margin"] - mu_b) < 0.3 * abs(mu_p - mu_b)
    assert f["ref_book"] == "betonlineag" and f["ref_n"] == 2 and f["ref_age"] == pytest.approx(2.0)
    # equally fresh: 3.0 : 1.5, two thirds of the way to Pinnacle
    f = sharp_fair(_frame(_spread("pinnacle", -3.0, updated=_at(2)), _spread("betonlineag", -4.0, updated=_at(2))), "nfl")
    assert f["mu_margin"] == pytest.approx((3.0 * mu_p + 1.5 * mu_b) / 4.5, abs=1e-9)
    assert f["ref_book"] == "pinnacle" and f["ref_age"] == pytest.approx(2.0)
    # no timestamps at all (a CSV): the priors alone, age_min computed inside when the column is missing
    f = sharp_fair(_frame(_spread("pinnacle", -3.0), _spread("betonlineag", -4.0), aged=False), "nfl")
    assert f["mu_margin"] == pytest.approx((3.0 * mu_p + 1.5 * mu_b) / 4.5, abs=1e-9) and np.isnan(f["ref_age"])
    # freshness is per market: a stale Pinnacle total does not drag its live spread
    f = sharp_fair(_frame(_spread("pinnacle", -3.0, updated=_at(1)), _total("pinnacle", 44.5, updated=_at(600)),
                          _spread("betonlineag", -4.0, updated=_at(1)), _total("betonlineag", 46.5, updated=_at(1))), "nfl")
    assert f["mu_margin"] == pytest.approx((3.0 * mu_p + 1.5 * mu_b) / 4.5, abs=1e-6)
    assert abs(f["mu_total"] - 46.5) < 0.3 * 2.0                                   # mostly BetOnline: Pinnacle keeps its floor (0.6 vs 1.5)
    # all five sharp books, mixed markets: mu_margin only from the spreads, ref_n counts every contributor
    f = sharp_fair(_frame(_spread("pinnacle", -3.0), _spread("circasports", -3.0), _spread("betonlineag", -4.0),
                          _ml("bookmaker", -150, 130), _ml("lowvig", -150, 130), aged=False), "nfl")
    assert f["mu_margin"] == pytest.approx((5.0 * mu_p + 1.5 * mu_b) / 6.5, abs=1e-9) and f["ref_n"] == 5
    assert f["mu_ml"] == pytest.approx(implied_margin_ml_emp(devig(-150, 130)[0]), abs=1e-9)


def test_no_sharp_book_falls_back_to_the_median_over_every_book():                                  # (c)
    soft = _frame(_spread("draftkings", -3.0), _spread("fanduel", -3.5), _spread("betmgm", -2.5),
                  _total("draftkings", 44.5), _total("fanduel", 45.5), aged=False)
    f = sharp_fair(soft, "nfl")
    assert f["mu_margin"] == pytest.approx(implied_margin_emp(-3.0, 0.5), abs=1e-9)        # the median of three
    assert f["mu_total"] == pytest.approx(45.0, abs=1e-9) and f["mu_ml"] == f["mu_margin"]
    assert f["ref_book"] == "betmgm" and f["ref_n"] == 0                                    # first book key, as before
    # a sharp book that posts only the total: the total is its, the spread is still the soft median
    f = sharp_fair(_frame(_spread("draftkings", -3.0), _spread("fanduel", -3.5), _spread("betmgm", -2.5),
                          _total("pinnacle", 47.5), _total("fanduel", 45.5), aged=False), "nfl")
    assert f["mu_total"] == pytest.approx(47.5) and f["mu_margin"] == pytest.approx(implied_margin_emp(-3.0, 0.5))
    assert f["ref_n"] == 1 and f["ref_book"] == "betmgm"


def test_totals_stay_normal_and_cfb_stays_normal():                                                  # (d) (e)
    odds = load_odds_csv("lines_template.csv")
    f = sharp_fair(odds, "nfl")
    assert f["mu_total"] == pytest.approx(47.5 + 10.3 * norm.ppf(devig(-104, -106)[0]), abs=1e-9)
    assert f["mu_total"] == pytest.approx(47.43786600859237)                                # the number before this change
    c = sharp_fair(odds, "cfb")
    assert c["mu_margin"] == pytest.approx(7.0, abs=1e-9)                                   # -7 -105/-105 on the Normal
    assert c["mu_ml"] == pytest.approx(16.5 * norm.ppf(devig(-330, 270)[0]), abs=1e-9)
    assert c["mu_total"] == pytest.approx(47.5 + 13.0 * norm.ppf(devig(-104, -106)[0]), abs=1e-9)
    assert c["ref_book"] == "pinnacle" and c["ref_n"] == 1
    ev = find_ev(odds, "cfb", min_ev=-1.0)
    sp = ev[(ev.market == "spreads") & (ev.side == "home")].iloc[0]
    assert sp.p_push == pytest.approx(cover_probs(7.0, -7.0, 16.5)["push"]) and sp.p_win == pytest.approx(cover_probs(7.0, -7.0, 16.5)["win"])
    ml = ev[(ev.market == "ml") & (ev.side == "home")].iloc[0]
    assert ml.p_win == pytest.approx(1 - norm.cdf(0, c["mu_ml"], 16.5))
    # two CFB sharp books still average 2:1 on the Normal
    two = sharp_fair(_frame(_spread("pinnacle", -3.0), _spread("betonlineag", -4.0), aged=False), "cfb")
    assert two["mu_margin"] == pytest.approx((3.0 * 3.0 + 1.5 * 4.0) / 4.5, abs=1e-9)


# ---------------- find_ev ----------------
def test_find_ev_prices_spreads_and_moneylines_on_the_pmf_with_pushes():                            # (f)
    board = _frame(_spread("pinnacle", -3.0, -105, -105), _ml("pinnacle", -160, 140),
                   _spread("draftkings", -3.0, 100, -120), _spread("fanduel", -2.5, -110, -110),
                   _ml("fanduel", -150, 130), aged=False)
    ev = find_ev(board, "nfl", min_ev=-1.0)
    f = sharp_fair(board, "nfl")
    mu = f["mu_margin"]
    assert (ev.ref_n == 1).all() and (ev.ref_book == "pinnacle").all() and (ev.fair_mu[ev.market == "spreads"] == mu).all()
    dk = ev[(ev.book == "draftkings") & (ev.side == "home")].iloc[0]
    cp = cover_probs_emp(mu, -3.0)
    assert dk.p_push == pytest.approx(margin_pmf(mu)[3]) and dk.p_push > 0.07               # the pmf's push, not the Normal's 3%
    assert dk.p_win == pytest.approx(cp["win"]) and dk.ev_pct == pytest.approx(0.0, abs=1e-9)   # +100 on a -100 number: exactly 0
    away = ev[(ev.book == "draftkings") & (ev.side == "away")].iloc[0]
    assert away.p_win == pytest.approx(cp["loss"]) and away.p_push == pytest.approx(cp["push"])
    # the sharp book's own number is a coin flip on the pmf it was inverted on: never +EV against itself
    pin = ev[(ev.book == "pinnacle") & (ev.market == "spreads")]
    assert pin.fair_price.abs().tolist() == pytest.approx([100.0, 100.0], abs=1e-6) and (pin.ev_pct < 0).all()
    assert pin.ev_pct.iloc[0] == pytest.approx(pin.ev_pct.iloc[1], abs=1e-9)
    # -2.5 at -110 off a -3 coin flip: the half point is worth ~17 cents, so this is +EV (the Normal said no)
    fd = ev[(ev.book == "fanduel") & (ev.side == "home") & (ev.market == "spreads")].iloc[0]
    assert fd.p_push == 0.0 and fd.p_win == pytest.approx(cp["win"] + cp["push"]) and fd.ev_pct > 0.02
    # moneylines through moneyline_prob_emp; pinnacle's own ML round-trips the devig
    ph = moneyline_prob_emp(f["mu_ml"])
    home = ev[(ev.market == "ml") & (ev.side == "home")].set_index("book")
    assert home.loc["pinnacle", "p_win"] == pytest.approx(devig(-160, 140)[0], abs=1e-6)
    assert home.loc["fanduel", "p_win"] == pytest.approx(ph) and home.loc["fanduel", "ev_pct"] > 0
    assert ev[(ev.market == "ml") & (ev.side == "away")].p_win.iloc[0] == pytest.approx(1 - ph) and (ev[ev.market == "ml"].p_push == 0).all()
    assert "ref_n" in top_picks(ev).columns and top_picks(ev).iloc[0].ref_n == 1


def test_stale_pinnacle_yields_to_fresh_betonline_on_the_board():
    """Pinnacle -3 six hours ago, BetOnline -4 two minutes ago, DraftKings -3 -110 now: DraftKings' -3 is the play,
    and every row names BetOnline as the reference with lag measured against it."""
    board = _frame(_spread("pinnacle", -3.0, updated=_at(360)), _spread("betonlineag", -4.0, updated=_at(2)),
                   _spread("draftkings", -3.0, updated=_at(1)))
    ev = find_ev(board, "nfl", min_ev=-1.0)
    assert (ev.ref_book == "betonlineag").all() and (ev.ref_n == 2).all()
    dk = ev[(ev.book == "draftkings") & (ev.side == "home")].iloc[0]
    mu_p, mu_b = implied_margin_emp(-3.0, 0.5), implied_margin_emp(-4.0, 0.5)
    assert dk.ev_pct > 0 and dk.lag_min == pytest.approx(-1.0) and abs(dk.fair_mu - mu_b) < 0.3 * abs(mu_p - mu_b)
    assert ev[(ev.book == "pinnacle") & (ev.side == "home")].iloc[0].lag_min == pytest.approx(358.0)
    # with Pinnacle fresh instead the fair moves two thirds of the way back and DraftKings' -3 is no longer a play
    fresh = find_ev(_frame(_spread("pinnacle", -3.0, updated=_at(2)), _spread("betonlineag", -4.0, updated=_at(2)),
                           _spread("draftkings", -3.0, updated=_at(1))), "nfl", min_ev=-1.0)
    assert (fresh.ref_book == "pinnacle").all()
    assert fresh[(fresh.book == "draftkings") & (fresh.side == "home")].iloc[0].ev_pct < dk.ev_pct
    # the model nudge rides on top and carries ref_n / ref_age through blended_fair
    model = pd.DataFrame([dict(home="PHI", away="DAL", model_margin=0.0, model_total=40.0)])
    b = blended_fair(board, "nfl", model, 0.5)
    s = sharp_fair(board, "nfl")
    assert b["mu_margin"] == pytest.approx(0.5 * s["mu_margin"]) and b["ref_n"] == 2 and b["ref_age"] == pytest.approx(2.0)


def test_game_fairs_and_find_ev_agree_with_ages_present():                                          # (g)
    """middles.game_fairs and find_ev price off the same weighted fair even when the freshness weights bite."""
    board = _frame(_spread("pinnacle", -3.0, updated=_at(90)), _spread("betonlineag", -4.0, updated=_at(3)),
                   _ml("pinnacle", -160, 140, updated=_at(90)), _ml("lowvig", -150, 130, updated=_at(3)),
                   _total("pinnacle", 44.5, updated=_at(90)), _total("circasports", 45.5, updated=_at(3)))
    gf = game_fairs(board, "nfl").set_index("market")
    ev = find_ev(board, "nfl", min_ev=-1.0)
    for mk in ("spreads", "ml", "totals"):
        assert gf.loc[mk, "mu"] == pytest.approx(ev[ev.market == mk].fair_mu.iloc[0], abs=1e-12)
    assert (ev.ref_n == 4).all()


def test_blank_price_is_unpriced_not_a_crash():
    """A hand CSV with a blank Pinnacle price: that market is skipped (mu NaN / next book), never a brentq ValueError."""
    f = sharp_fair(_frame(_spread("pinnacle", -3.0, np.nan, -102), aged=False), "nfl")
    assert np.isnan(f["mu_margin"]) and f["ref_n"] == 0
    f = sharp_fair(_frame(_spread("pinnacle", -3.0, np.nan, -102), _spread("betonlineag", -3.5, -110, -110), aged=False), "nfl")
    assert f["ref_book"] == "betonlineag" and f["mu_margin"] == pytest.approx(implied_margin_emp(-3.5, 0.5), abs=1e-9)
    ev = find_ev(_frame(_spread("pinnacle", -3.0, np.nan, -102), aged=False), "nfl", min_ev=-1.0)
    assert len(ev) == 0
    assert np.isnan(implied_margin_emp(-3.0, np.nan)) and np.isnan(implied_margin_ml_emp(1.0)) and np.isnan(implied_margin_emp(np.nan, 0.5))


def test_model_that_agrees_with_the_market_does_not_move_the_fair():
    """mu is the pmf's location parameter (a -7 coin flip sits at ~8.3); the model's expected margin is converted into
    that space before blending, so agreeing exactly with the market changes nothing at any weight."""
    for line in (-3.0, -7.0, -10.0, 2.5):
        board = _frame(_spread("pinnacle", line, -110, -110), _ml("pinnacle", -330, 270), aged=False)
        s = sharp_fair(board, "nfl")
        model = pd.DataFrame([dict(home="PHI", away="DAL", model_margin=-line, model_total=np.nan)])
        for w in (0.25, 0.5, 1.0):
            b = blended_fair(board, "nfl", model, w)
            assert b["mu_margin"] == pytest.approx(s["mu_margin"], abs=1e-6), (line, w)
            assert b["mu_ml"] == pytest.approx(s["mu_ml"], abs=1e-6), (line, w)
    # and a model that disagrees still moves it the right way
    board = _frame(_spread("pinnacle", -7.0, -110, -110), aged=False)
    s = sharp_fair(board, "nfl")
    up = blended_fair(board, "nfl", pd.DataFrame([dict(home="PHI", away="DAL", model_margin=10.0, model_total=np.nan)]), 0.5)
    assert up["mu_margin"] > s["mu_margin"]
