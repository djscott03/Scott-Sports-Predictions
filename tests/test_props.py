"""Player-prop projection + pricing tests on a synthetic league — no network, <2s."""
import numpy as np
import pandas as pd
import pytest

from sharpmodel import props
from sharpmodel.pricing import devig, prob_to_american, american_to_prob
from sharpmodel.props import (MARKETS, DEFAULT_CV, project_players, fit_dispersion, prop_probs,
                              implied_mean, price_props, normalize_player)

TEAMS = [f"T{i}" for i in range(1, 9)]
DEF_REC = {"T8": 1.5, "T7": 0.6}          # receiving yards allowed multiplier (T8 soft, T7 tough)
NOISE = 0.10


def synthetic_stats(seasons=(2023, 2024, 2025), weeks=16, seed=0):
    """8 teams, one QB/RB/WR/TE each with means rising by team index; week-17 slate has lines only."""
    rng = np.random.default_rng(seed)
    rows, games = [], []
    yds = lambda m: max(0.0, rng.normal(m, NOISE * m))
    for s in seasons:
        for w in range(1, weeks + 1):
            order = list(rng.permutation(TEAMS))
            for i in range(0, 8, 2):
                h, a = order[i], order[i + 1]
                games.append(dict(season=s, week=w, home=h, away=a, home_pts=rng.poisson(24), away_pts=rng.poisson(24),
                                  market_margin=0.0, total_line=48.0))
                for tm, opp in ((h, a), (a, h)):
                    k, m = TEAMS.index(tm), DEF_REC.get(opp, 1.0)
                    base = dict(season=s, week=w, season_type="REG", team=tm, opponent=opp)
                    rows += [
                        dict(base, player_id=f"QB{tm}", player=f"Quarterback {tm}", position="QB",
                             passing_yards=yds(180 + 20 * k), passing_tds=rng.poisson(1.0 + 0.2 * k),
                             rushing_yards=yds(15), rushing_tds=rng.poisson(0.1), receiving_yards=0.0,
                             receiving_tds=0, receptions=0),
                        dict(base, player_id=f"RB{tm}", player=f"Runner {tm}", position="RB",
                             passing_yards=0.0, passing_tds=0, rushing_yards=yds(40 + 8 * k),
                             rushing_tds=rng.poisson(0.5), receiving_yards=yds(20 * m), receiving_tds=rng.poisson(0.1),
                             receptions=rng.poisson(2.5)),
                        dict(base, player_id=f"WR{tm}", player=f"Receiver {tm}", position="WR",
                             passing_yards=0.0, passing_tds=0, rushing_yards=0.0, rushing_tds=0,
                             receiving_yards=yds((30 + 10 * k) * m), receiving_tds=rng.poisson(0.4),
                             receptions=rng.poisson(3 + 0.5 * k)),
                        dict(base, player_id=f"TE{tm}", player=f"Tightend {tm}", position="TE",
                             passing_yards=0.0, passing_tds=0, rushing_yards=0.0, rushing_tds=0,
                             receiving_yards=yds(40 * m), receiving_tds=rng.poisson(0.3), receptions=rng.poisson(4)),
                    ]
    # a rookie with only two games must get no projection
    for w in (weeks - 1, weeks):
        rows.append(dict(season=seasons[-1], week=w, season_type="REG", team="T1", opponent="T2", player_id="WRrook",
                         player="Rookie T1", position="WR", passing_yards=0.0, passing_tds=0, rushing_yards=0.0,
                         rushing_tds=0, receiving_yards=80.0, receiving_tds=1, receptions=6))
    slate = [dict(season=2025, week=17, home="T8", away="T1", market_margin=0.0, total_line=48.0),   # T1 WR faces soft T8
             dict(season=2025, week=17, home="T7", away="T2", market_margin=0.0, total_line=48.0),   # T2 WR faces tough T7
             dict(season=2025, week=17, home="T3", away="T4", market_margin=0.0, total_line=60.0),   # high total
             dict(season=2025, week=17, home="T5", away="T6", market_margin=0.0, total_line=36.0)]   # low total
    return pd.DataFrame(rows), pd.concat([pd.DataFrame(games), pd.DataFrame(slate)], ignore_index=True)


@pytest.fixture(scope="module")
def league():
    stats, games = synthetic_stats()
    return stats, games, project_players(stats, games, as_of=(2025, 17))


def test_projections_recover_ordering_and_shrink(league):
    stats, games, proj = league
    assert list(proj.columns) == ["player_id", "player", "team", "opponent", "position", "market", "proj_mean",
                                  "proj_sd", "n_games", "opp_factor", "env_factor"]
    assert "WRrook" not in set(proj.player_id)                    # < min_games
    assert (proj.n_games == 48).all()
    qb = proj[proj.market == "player_pass_yds"].set_index("team")
    base = qb.proj_mean / (qb.opp_factor * qb.env_factor)         # strip the multiplicative factors
    assert base["T1"] < base["T4"] < base["T8"]                   # true means 180 .. 320
    assert np.corrcoef(base.reindex(TEAMS), np.arange(8))[0, 1] > 0.95
    assert 180 < base["T1"] and base["T8"] < 320                  # shrunk toward the QB mean
    assert base["T8"] - base["T1"] < 140
    assert proj.proj_mean.notna().all() and proj.opp_factor.notna().all()   # all-zero stats (WR rush) -> 1.0
    assert qb.proj_sd["T8"] == pytest.approx(DEFAULT_CV["player_pass_yds"] * qb.proj_mean["T8"])
    td = proj[proj.market == "player_anytime_td"]
    assert set(td.position) == set(props.SKILL) and (td.proj_sd == td.proj_mean).all()
    assert set(proj.market) == set(MARKETS)


def test_opp_factor_moves_the_right_way(league):
    _, _, proj = league
    wr = proj[(proj.market == "player_reception_yds") & (proj.position == "WR")].set_index("team")
    assert wr.opponent["T1"] == "T8" and wr.opp_factor["T1"] > 1.15    # 1.5x allowed, shrunk 50% -> ~1.25
    assert wr.opponent["T2"] == "T7" and wr.opp_factor["T2"] < 0.85    # 0.6x -> ~0.80
    assert wr.opp_factor["T3"] == pytest.approx(1.0, abs=0.08)
    qb = proj[proj.market == "player_pass_yds"].set_index("team")      # passing has no defense effect built in
    assert abs(qb.opp_factor - 1).max() < 0.08
    assert proj.opp_factor.between(*props.OPP_CLIP).all()


def test_env_factor_from_implied_totals(league):
    stats, games, proj = league
    qb = proj[proj.market == "player_pass_yds"].set_index("team")
    assert qb.env_factor["T3"] == pytest.approx((30 / 24) ** 0.5, abs=0.06) and qb.env_factor["T4"] > 1.05
    assert qb.env_factor["T5"] == pytest.approx((18 / 24) ** 0.5, abs=0.06) and qb.env_factor["T6"] < 0.95
    assert qb.env_factor["T1"] == pytest.approx(1.0, abs=0.06)      # ppg is Poisson(24) noise around 24
    assert proj.env_factor.between(0.8, 1.25).all()
    # slate only (no played history) -> env is neutral, opponents still attached
    slate_only = project_players(stats, games[games.week == 17], as_of=(2025, 17))
    assert (slate_only.env_factor == 1.0).all() and slate_only.opponent.notna().all()


def test_fit_dispersion_floors_and_caches(league, monkeypatch):
    stats, _, _ = league
    monkeypatch.setattr(props, "_CV", {})
    cv = fit_dispersion(stats, as_of=(2025, 17), weeks=6)
    assert set(cv) == {m for m, s in MARKETS.items() if s["dist"] == "normal"}
    assert all(v >= 0.35 for v in cv.values()) and props._CV == cv
    assert props._sd("player_pass_yds", 100.0) == pytest.approx(cv["player_pass_yds"] * 100)


def _noisy_wr(n_games, hi=100.0):
    """A T1 receiver alternating hi / 0 yards over the last n_games weeks of 2025: mean ~hi/2, huge residuals."""
    return pd.DataFrame([dict(season=2025, week=w, season_type="REG", team="T1", opponent="T2", player_id="WRbk",
                              player="Backup T1", position="WR", passing_yards=0.0, passing_tds=0, rushing_yards=0.0,
                              rushing_tds=0, receiving_yards=0.0 if i % 2 else hi, receiving_tds=0, receptions=0)
                         for i, w in enumerate(range(17 - n_games, 17))])


def test_fit_dispersion_uses_established_starters_only(league, monkeypatch):
    """Backups (few prior games, or outside the STARTERS population) add projection error, not variance."""
    stats, _, _ = league
    monkeypatch.setattr(props, "_CV", {})
    fit = lambda st: fit_dispersion(st, as_of=(2025, 17), weeks=6, cv_floor=0.0)["player_reception_yds"]
    base = fit(stats)
    assert 0 < base < 0.35                                                    # 10% synthetic noise: below the floor
    # < 8 prior games: out of the fit (he still nudges the shrink target the other projections use, hence rel)
    assert fit(pd.concat([stats, _noisy_wr(5)], ignore_index=True)) == pytest.approx(base, rel=1e-3)
    with_bk = pd.concat([stats, _noisy_wr(12)], ignore_index=True)
    assert fit(with_bk) > 1.15 * base                       # 12 games and a WR96 population: counted as a starter
    monkeypatch.setitem(props.STARTERS, "WR", 4)            # ...but with 4 starters per position he ranks 5th+ by usage
    assert fit(with_bk) == pytest.approx(fit(stats))


def test_prop_probs_and_implied_mean_round_trip():
    for dist, mean, sd, line in [("normal", 250.0, 80.0, 245.5), ("normal", 250.0, 80.0, 250.0),
                                 ("poisson", 2.3, None, 1.5), ("poisson", 2.3, None, 2.0)]:
        q = prop_probs(dist, mean, sd, line, "over")
        assert q["win"] + q["push"] + q["loss"] == pytest.approx(1.0)
        assert (q["push"] > 0) == (line == round(line))                        # integer lines push
        u = prop_probs(dist, mean, sd, line, "under")
        assert u["win"] == pytest.approx(q["loss"]) and u["push"] == q["push"]
        assert implied_mean(dist, line, q["win"] / (1 - q["push"]), sd) == pytest.approx(mean, abs=1e-6)
    assert prop_probs("normal", 250.0, 80.0, 250.0, "over")["win"] == pytest.approx(
        prop_probs("normal", 250.0, 80.0, 250.0, "over")["loss"])                # centred on the number
    yes = prop_probs("bernoulli", 0.8, None, np.nan, "yes")
    assert yes["win"] == pytest.approx(1 - np.exp(-0.8)) and yes["push"] == 0.0
    assert implied_mean("bernoulli", np.nan, yes["win"]) == pytest.approx(0.8)


def test_normalize_player():
    assert normalize_player("A.J. Brown") == normalize_player("AJ Brown") == "aj brown"
    assert normalize_player("Kenneth Walker III") == normalize_player("Kenneth Walker") == "kenneth walker"
    assert normalize_player("Odell Beckham Jr.") == "odell beckham"
    assert normalize_player("  Ja'Marr   Chase ") == "jamarr chase"


def _props_book():
    ev = dict(event_id="e1", commence="2026-09-06T17:00:00Z", home="PHI", away="DAL")
    rows = []
    for book, line, po, pu in [("pinnacle", 245.5, -108, -112), ("draftkings", 245.5, -110, -110),
                               ("fanduel", 235.5, -110, -110)]:                      # fanduel is stale
        rows += [dict(ev, book=book, market="player_pass_yds", player="Jalen Hurts", side="over", line=line, price=po),
                 dict(ev, book=book, market="player_pass_yds", player="Jalen Hurts", side="under", line=line, price=pu)]
    for book, line in [("pinnacle", 5.5), ("draftkings", 5.5)]:
        rows += [dict(ev, book=book, market="player_receptions", player="A.J. Brown", side="over", line=line, price=-115),
                 dict(ev, book=book, market="player_receptions", player="A.J. Brown", side="under", line=line, price=-105)]
    rows.append(dict(ev, book="betmgm", market="player_rush_yds", player="Kenneth Walker III", side="over", line=60.5,
                     price=+150))                                                   # one book, one side
    for book, price in [("pinnacle", -150), ("draftkings", -140), ("fanduel", -160)]:
        rows.append(dict(ev, book=book, market="player_anytime_td", player="Saquon Barkley", side="yes",
                         line=np.nan, price=price))
    return pd.DataFrame(rows)


def _projections():
    cols = ["player_id", "player", "team", "opponent", "position", "market", "proj_mean", "proj_sd", "n_games",
            "opp_factor", "env_factor"]
    return pd.DataFrame([
        ("h", "Jalen Hurts", "PHI", "DAL", "QB", "player_pass_yds", 250.0, 75.0, 30, 1.0, 1.0),
        ("b", "AJ Brown", "PHI", "DAL", "WR", "player_receptions", 6.0, 6.0, 30, 1.0, 1.0),
        ("w", "Kenneth Walker", "SEA", np.nan, "RB", "player_rush_yds", 90.0, 49.5, 30, 1.0, 1.0),
        ("s", "Saquon Barkley", "PHI", "DAL", "RB", "player_anytime_td", 0.9, 0.9, 30, 1.0, 1.0),
    ], columns=cols)


def test_price_props_flags_stale_book_and_never_stakes_no_market():
    res = price_props(_props_book(), _projections(), min_ev=0.03)
    hurts = res[res.market == "player_pass_yds"]
    assert len(hurts) == 1
    row = hurts.iloc[0]
    assert (row.book, row.side, row.line) == ("fanduel", "over", 235.5)
    assert row.n_books == 3 and row.team == "PHI" and row["flags"] == ""   # df.flags is a pandas attribute
    assert 244 < row.market_mu < 246 and row.proj_mean == 250.0
    assert row.fair_mu == pytest.approx(0.7 * row.market_mu + 0.3 * 250.0)
    assert row.ev_pct > 0.03 and 0 < row.kelly_stake <= 0.02 and row.fair_price < 0
    assert row.dist == "normal" and row.fair_sd == pytest.approx(props._sd("player_pass_yds", row.fair_mu))
    assert not (res.book == "pinnacle").any()
    # projection-only line: shown with the flag (the price beats the projection), no EV, stake 0, listed last
    walker = res[res.player == "Kenneth Walker III"]
    assert len(walker) == 1 and res.index[-1] == walker.index[0]
    w = walker.iloc[0]
    assert "no_market" in w["flags"] and "team_mismatch" in w["flags"] and w.kelly_stake == 0.0 and np.isnan(w.ev_pct)
    assert np.isnan(w.market_mu) and w.fair_mu == 90.0 and w.n_books == 0 and w.team == "SEA"
    assert w.fair_sd == pytest.approx(49.5) and w.p_win > 0.7                 # sd = 0.55 * 90; over 60.5 at +150
    # A.J. Brown matched AJ Brown exactly (same team) -> receptions priced off a blended poisson mean
    everything = price_props(_props_book(), _projections(), min_ev=-1.0)
    aj = everything[everything.player == "A.J. Brown"]
    assert len(aj) == 4 and (aj["flags"] == "").all() and (aj.team == "PHI").all()
    assert aj.fair_mu.iloc[0] == pytest.approx(0.7 * aj.market_mu.iloc[0] + 0.3 * 6.0)
    assert (aj.ev_pct < 0.03).all()
    # no projections at all -> pure market pricing, no_proj flag, still finds the stale line
    nomodel = price_props(_props_book(), None, min_ev=0.03)
    assert set(nomodel.player) == {"Jalen Hurts"} and (nomodel["flags"] == "no_proj").all()
    assert nomodel.iloc[0].fair_mu == nomodel.iloc[0].market_mu


def test_one_sided_anytime_td_pricing():
    res = price_props(_props_book(), _projections(), min_ev=-1.0)
    td = res[res.market == "player_anytime_td"]
    assert len(td) == 3 and (td.n_books == 3).all() and (td.dist == "bernoulli").all() and td.fair_sd.isna().all()
    # median of -150/-140/-160 is -150: power-devigged against a synthetic 'no' so the pair holds 7%, bounded by
    # implied * (1 - hold) -- the bound is what binds for a favourite (the power method lets it keep more)
    p_med = min(devig(-150, prob_to_american(1.07 - 0.6))[0], 0.6 * 0.93)
    assert 0.55 < p_med < 0.57 and td.market_mu.iloc[0] == pytest.approx(-np.log(1 - p_med))
    assert (0 < td.p_win).all() and (td.p_win < 1).all()
    assert td.fair_mu.iloc[0] == pytest.approx(0.7 * td.market_mu.iloc[0] + 0.3 * 0.9)
    assert td.p_win.iloc[0] == pytest.approx(1 - np.exp(-td.fair_mu.iloc[0]))
    assert (td.kelly_stake <= 0.02).all()
    # longshots take most of the vig (power method) but are never devigged to nothing: the synthetic 'no' is
    # capped at MAX_SYNTH_NO, so +2000 keeps ~half its implied probability instead of ~4% of it
    for price in (+300, +1000, +2000):
        p = props._yes_only_prob(price, 0.07)
        assert 0.25 * american_to_prob(price) < p < american_to_prob(price)
    assert props._yes_only_prob(-150, 0.07) == pytest.approx(p_med)


def test_yes_only_price_never_beats_itself():
    """Past ~+4900 the MAX_SYNTH_NO cap turns the synthetic pair's hold negative and the power devig alone hands
    the price MORE than its own implied probability (+10000 -> 1.75% vs 0.99% implied: +EV against itself)."""
    ev = dict(event_id="e1", commence="2026-09-06T17:00:00Z", home="PHI", away="DAL")
    lone = lambda price: pd.DataFrame([dict(ev, book="betmgm", market="player_anytime_td", player="Long Shot",
                                            side="yes", line=np.nan, price=price)])
    for price in (150, 1000, 5000, 10000, 20000):
        implied = american_to_prob(price)
        p = props._yes_only_prob(price, 0.07)
        assert 0 < p < implied and p <= implied * 0.93 + 1e-12, price
        market_p = 1 - np.exp(-price_props(lone(price), None, min_ev=-1.0).market_mu.iloc[0])    # the board's P(yes)
        assert market_p == pytest.approx(p) and market_p < implied
    res = price_props(lone(10000), None, min_ev=-1.0)
    assert len(res) == 1 and res.iloc[0].ev_pct <= 0 and res.iloc[0].kelly_stake == 0
    assert price_props(lone(10000), None, min_ev=0.0).empty                    # never shows up as a play


def _two_way(ev, book, market, player, line, po, pu):
    return [dict(ev, book=book, market=market, player=player, side="over", line=line, price=po),
            dict(ev, book=book, market=market, player=player, side="under", line=line, price=pu)]


def test_price_props_groups_by_normalized_name():
    ev = dict(event_id="e1", commence="2026-09-06T17:00:00Z", home="PHI", away="DAL")
    rows = (_two_way(ev, "pinnacle", "player_receptions", "A.J. Brown", 5.5, -115, -105)
            + _two_way(ev, "draftkings", "player_receptions", "AJ Brown", 5.5, -110, -110))
    res = price_props(pd.DataFrame(rows), _projections(), min_ev=-1.0)
    assert len(res) == 4 and (res.n_books == 2).all() and set(res.player) == {"A.J. Brown"}   # first spelling seen
    assert (res["flags"] == "").all() and (res.team == "PHI").all()


def test_price_props_skips_lines_that_are_nan():
    ev = dict(event_id="e1", commence="2026-09-06T17:00:00Z", home="PHI", away="DAL")
    rows = (_two_way(ev, "draftkings", "player_pass_yds", "Jalen Hurts", 245.5, -110, -110)
            + _two_way(ev, "betmgm", "player_pass_yds", "Jalen Hurts", np.nan, -110, -110))     # "point": null
    res = price_props(pd.DataFrame(rows), None, min_ev=-1.0)
    assert len(res) == 2 and set(res.book) == {"draftkings"} and (res.n_books == 1).all()
    assert res.line.notna().all() and np.isfinite(res.market_mu).all()
    only_nan = price_props(pd.DataFrame(rows[2:]), None, min_ev=-1.0)
    assert only_nan.empty


def test_team_mismatch_is_shown_but_never_staked():
    ev = dict(event_id="e1", commence="2026-09-06T17:00:00Z", home="PHI", away="DAL")
    rows = (_two_way(ev, "draftkings", "player_rush_yds", "Kenneth Walker III", 60.5, -110, -110)
            + _two_way(ev, "fanduel", "player_rush_yds", "Kenneth Walker III", 50.5, -110, -110))   # stale
    res = price_props(pd.DataFrame(rows), _projections(), min_ev=0.03)
    # fair = 0.7 * 55.5 (median of the two books) + 0.3 * 90 (the SEA projection): both overs clear +3% EV
    assert len(res) == 2 and (res["flags"] == "team_mismatch").all() and (res.side == "over").all()
    r = res.iloc[0]
    assert (r.book, r.line) == ("fanduel", 50.5) and r.ev_pct > res.ev_pct.iloc[1] > 0.03
    assert (res.kelly_stake == 0.0).all() and (res.team == "SEA").all() and (res.n_books == 2).all()
    assert res.fair_mu.iloc[0] == pytest.approx(0.7 * 55.5 + 0.3 * 90.0, abs=0.01)


def test_no_market_rows_sort_last_and_filter_on_projection_ev():
    ev = dict(event_id="e1", commence="2026-09-06T17:00:00Z", home="PHI", away="DAL")
    rows = _two_way(ev, "draftkings", "player_pass_yds", "Jalen Hurts", 245.5, -110, -110)
    rows.append(dict(ev, book="betmgm", market="player_receptions", player="AJ Brown", side="over", line=5.5, price=+100))
    rows.append(dict(ev, book="betmgm", market="player_receptions", player="AJ Brown", side="under", line=9.5, price=-110))
    res = price_props(pd.DataFrame(rows), _projections(), min_ev=-1.0)
    assert list(res["flags"]) == ["", "", "no_market", "no_market"] and res.ev_pct.iloc[2:].isna().all()
    assert list(res.ev_pct.iloc[:2]) == sorted(res.ev_pct.iloc[:2], reverse=True) and (res.kelly_stake.iloc[2:] == 0).all()
    # projection 6.0 receptions: over 5.5 at +100 beats it, under 9.5 at -110 does too; under 5.5 at +100 would not
    rows.append(dict(ev, book="fanduel", market="player_receptions", player="AJ Brown", side="under", line=5.5, price=+100))
    kept = price_props(pd.DataFrame(rows), _projections(), min_ev=0.03)
    nm = kept[kept["flags"] == "no_market"]
    assert set(zip(nm.book, nm.side, nm.line)) == {("betmgm", "over", 5.5), ("betmgm", "under", 9.5)}
    assert nm.ev_pct.isna().all() and (nm.kelly_stake == 0).all()
