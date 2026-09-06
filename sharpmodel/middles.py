"""
Cross-book arbs and middles: the two sides of one market at two books whose numbers or
prices are far enough apart that both can win (a middle) or the pair cannot lose (an arb).

The owner's example: Hurts pass yds O 280.5 +100 at one book and U 315.5 +100 at another.
Bet both. Land inside 281-315 and both cash; land outside and one leg wins exactly what the
other loses, so the worst case is zero (a free middle). At -110/-110 the split costs ~4.5% of
the total stake, so the middle has to land more than ~4.5% of the time to be +EV
(breakeven_p); a 35-yard window lands far more often than that.

Every pair is priced on the integer support of a fair distribution (`fairs`):
  nfl_margin  NFL spreads and moneylines: margins.margin_pmf with the empirical key-number
              weights. A one-number window on 3 holds ~9% of the mass; the Normal says ~3%,
              which is the difference between a +EV middle and a losing one.
  normal      totals, CFB spreads, yardage props (discretized, integer lines push)
  poisson     count props (receptions, TD passes)
  bernoulli   anytime TD yes/no (arb only)
The market is still the prior: game fairs are odds.blended_fair (the sharp book, nudged toward
the model exactly as find_ev does) and prop fairs are price_props' devigged consensus blend;
a projection alone never prices a middle (prop_fairs drops `no_market` rows).

Stakes split so the two split outcomes pay the same (stake proportional to 1/decimal odds);
every *_pct column is percent of the total stake. odds.find_arbs is the old same-number-only
finder, kept for compatibility; find_middles supersedes it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson
from .pricing import LEAGUE_SD, decimal_from_american
from .margins import NFL_SD, NFL_KEY_WEIGHTS, margin_pmf
from .odds import blended_fair
from .props import normalize_player

DISTS = ("nfl_margin", "normal", "poisson", "bernoulli")
FAIR_COLS = ["event_id", "market", "player", "mu", "sd", "dist"]
COLS = ["commence", "matchup", "market", "player", "type", "bet_a", "bet_b", "window", "p_middle",
        "miss_cost_pct", "win_both_pct", "ev_pct", "guaranteed_pct", "breakeven_p",
        "stake_a_pct", "stake_b_pct", "same_book",
        "event_id", "book_a", "line_a", "price_a", "book_b", "line_b", "price_b"]      # machine columns last
SIDE_A = {"home", "over", "yes"}          # side A wins when the number goes UP
SIDE_B = {"away", "under", "no"}          # side B wins when it goes DOWN
TYPES = ("arb", "arb+middle", "free_middle", "middle", "half_middle")
EPS = 1e-9


# ---------------- fair distributions ----------------
def game_fairs(odds: pd.DataFrame, league: str = "nfl", model_fair: pd.DataFrame | None = None,
               model_weight: float = 0.25) -> pd.DataFrame:
    """One fair per (event, market) -- FAIR_COLS with player NaN -- from odds.blended_fair, i.e. the same
    sharp-book fair, nudged toward `model_fair` at `model_weight`, that find_ev prices off.
    NFL spreads and ML: dist 'nfl_margin' at margins.NFL_SD (the sd the key weights were fitted at);
    totals and CFB spreads/ML: 'normal' at LEAGUE_SD."""
    sd_s, sd_t = LEAGUE_SD[league]["spread"], LEAGUE_SD[league]["total"]
    mdist, msd = ("nfl_margin", NFL_SD) if league == "nfl" else ("normal", sd_s)
    if odds is None or len(odds) == 0: return pd.DataFrame(columns=FAIR_COLS)
    rows = []
    for eid, e in odds.groupby("event_id"):
        f = blended_fair(e, league, model_fair, model_weight)
        rows += [dict(event_id=eid, market="spreads", player=np.nan, mu=f["mu_margin"], sd=msd, dist=mdist),
                 dict(event_id=eid, market="ml", player=np.nan, mu=f["mu_ml"], sd=msd, dist=mdist),
                 dict(event_id=eid, market="totals", player=np.nan, mu=f["mu_total"], sd=sd_t, dist="normal")]
    return pd.DataFrame(rows, columns=FAIR_COLS).dropna(subset=["mu"]).reset_index(drop=True)


def prop_fairs(board: pd.DataFrame | None) -> pd.DataFrame:
    """One fair per (event, market, player) from a price_props board: its fair_mu / fair_sd / dist.
    Price the board with min_ev=-1 so every two-way market is present, not only the +EV lines.
    Rows flagged 'no_market' (projection only) are dropped: a projection alone never prices a middle."""
    if board is None or len(board) == 0: return pd.DataFrame(columns=FAIR_COLS)
    b = board[~board["flags"].astype(str).str.contains("no_market")]
    b = b.assign(_key=b.player.map(normalize_player)).drop_duplicates(["event_id", "market", "_key"])
    return b.rename(columns={"fair_mu": "mu", "fair_sd": "sd"})[FAIR_COLS].reset_index(drop=True)


def support_pmf(dist: str, mu: float, sd: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(k, p): P(x = k) on the integer support of a fair distribution; the end bins absorb the tails.
    nfl_margin: margins.margin_pmf(mu, sd or NFL_SD, key weights) on -75..75.  normal: discretized
    Normal on mu +- 8 sd.  poisson: 0..lam + 10 sqrt(lam) + 10.  bernoulli: {0, 1} with P(1) = 1 - exp(-mu)."""
    if dist == "nfl_margin":
        s = margin_pmf(mu, sd if sd is not None and np.isfinite(sd) else NFL_SD, NFL_KEY_WEIGHTS)
        return s.index.values, s.values
    if dist == "normal":
        if sd is None or not np.isfinite(sd) or sd <= 0: raise ValueError("normal fair needs a positive sd")
        s = margin_pmf(mu, sd, None, int(np.floor(mu - 8 * sd)), int(np.ceil(mu + 8 * sd)))
        return s.index.values, s.values
    if dist == "poisson":
        lam = max(float(mu), 1e-9)
        k = np.arange(0, int(np.ceil(lam + 10 * np.sqrt(lam) + 10)) + 1)
        p = poisson.pmf(k, lam); p[-1] += max(1 - p.sum(), 0.0)
        return k, p
    if dist == "bernoulli":
        p1 = 1 - np.exp(-max(float(mu), 0.0))
        return np.array([0, 1]), np.array([1 - p1, p1])
    raise ValueError(f"unknown dist {dist!r}; expected one of {DISTS}")


# ---------------- one pair ----------------
def price_pair(k: np.ndarray, p: np.ndarray, lo: float, hi: float, price_a: float, price_b: float) -> dict:
    """
    Outcome table for two legs on the same number x with pmf (k, p): A wins iff x > lo (pushes at x == lo),
    B wins iff x < hi (pushes at x == hi). Stakes sum to 1 and are proportional to 1/decimal odds, so the two
    split outcomes (A wins & B loses, B wins & A loses) pay the same: -miss_cost. Requires lo <= hi.
    Returns p_middle (P both win), miss_cost, win_both, ev, guaranteed (min payoff over outcomes with P > 0
    where any stake is settled -- both legs pushing returns everything, so it cannot be a loss), stake_a,
    stake_b, window (integers strictly inside: both win), half (integer numbers where one leg wins and the
    other pushes, e.g. the 3 of a -2.5 / +3 pair), p_half and half_pay (its probability and its smallest
    payoff), all per 1 unit total stake.
    """
    dec_a, dec_b = decimal_from_american(price_a), decimal_from_american(price_b)
    s = 1 / dec_a + 1 / dec_b
    stake_a, stake_b = (1 / dec_a) / s, (1 / dec_b) / s
    ra = np.where(k > lo, 1, np.where(k == lo, 0, -1))                  # +1 win, 0 push, -1 loss
    rb = np.where(k < hi, 1, np.where(k == hi, 0, -1))
    pay = (np.select([ra > 0, ra == 0], [dec_a - 1, 0.0], -1.0) * stake_a
           + np.select([rb > 0, rb == 0], [dec_b - 1, 0.0], -1.0) * stake_b)
    live = ((ra != 0) | (rb != 0)) & (p > 0)
    both, half = (ra > 0) & (rb > 0), ((ra > 0) & (rb == 0)) | ((ra == 0) & (rb > 0))
    return dict(p_middle=float(p[both].sum()), miss_cost=1 - 1 / s, win_both=2 / s - 1,
                ev=float((p * pay).sum()), guaranteed=float(pay[live].min()) if live.any() else 0.0,
                stake_a=stake_a, stake_b=stake_b, window=k[both],
                half=k[half], p_half=float(p[half].sum()), half_pay=float(pay[half].min()) if half.any() else 0.0)


def _pkey(player) -> str:
    return normalize_player(player) if isinstance(player, str) and player.strip() else ""


def _thresholds(market: str, dist: str, line_a: float, line_b: float):
    """(lo, hi) for price_pair, or None when a leg has no usable number."""
    if market == "ml": return 0.0, 0.0                               # margin > 0 / < 0, a tie pushes both
    if dist == "bernoulli": return 0.5, 0.5                          # yes iff x = 1
    if not (np.isfinite(line_a) and np.isfinite(line_b)): return None
    if market == "spreads": return -float(line_a), float(line_b)     # home covers if margin > -line_home
    return float(line_a), float(line_b)                              # totals / over-under props


def _price(x: float) -> str:
    return f"{int(x):+d}" if float(x).is_integer() else f"{x:+g}"


def _short(player: str) -> str:
    parts = str(player).split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else str(player)


def _bet(market: str, side: str, line: float, price: float, book: str, home: str, away: str, player) -> str:
    """'PHI -2.5 -110 @ draftkings', 'O 44.5 -110 @ fanduel', 'DAL ML +120 @ x', 'J. Hurts O 280.5 +100 @ fanduel'."""
    if market == "spreads":
        num = "PK" if line == 0 else f"{line:+g}"
        return f"{home if side == 'home' else away} {num} {_price(price)} @ {book}"
    if market == "ml":
        return f"{home if side == 'home' else away} ML {_price(price)} @ {book}"
    who = f"{_short(player)} " if isinstance(player, str) and player else ""
    if side in ("yes", "no"):
        return f"{who}{side.upper()} {_price(price)} @ {book}"
    return f"{who}{'O' if side == 'over' else 'U'} {line:g} {_price(price)} @ {book}"


def _window_str(w: np.ndarray, suffix: str = "") -> str:
    if len(w) == 0: return ""
    return f"{int(w[0])}{suffix}" if len(w) == 1 else f"{int(w[0])}{suffix}-{int(w[-1])}{suffix}"


# ---------------- the finder ----------------
def find_middles(odds: pd.DataFrame, fairs: pd.DataFrame, league: str = "nfl", min_ev: float = 0.0,
                 include_same_book: bool = False) -> pd.DataFrame:
    """
    Every cross-book pair of opposite sides on one market, priced as a middle/arb off `fairs`.
    odds: long schema (event_id, commence, home, away, book, market, side, line, price [, player]).
    fairs: FAIR_COLS -- one row per (event_id, market[, player]) with mu, sd, dist in DISTS; see
           game_fairs / prop_fairs. Markets without a fair are skipped.
    Per (event, market[, normalised player]): side A = home/over/yes, side B = away/under/no, reduced to
    the best price per (side, line, book); every (a, b) at different books (same-book pairs only with
    include_same_book, flagged same_book) becomes price_pair on the integer support. Pairs with a
    both-lose gap (B's number below A's) are skipped. The caller's index is discarded and exact duplicate
    rows dropped first (a concat of two CSVs repeats index labels, which would re-admit worse prices and
    emit every pair several times); rows without a price are dropped too. None / empty odds -> empty COLS.
    type: 'arb' (guaranteed > 0, no window), 'arb+middle' (guaranteed > 0 and a window), 'free_middle'
    (guaranteed == 0 and a window: cannot lose), 'middle' (window, kept when ev_pct >= min_ev),
    'half_middle' (no both-win window, but an integer number where one leg wins and the other pushes,
    e.g. -2.5 / +3 on a 3: window shown as '3p', p_middle = P(that number), win_both_pct = its payoff);
    scalps (no window at all, guaranteed <= 0) are dropped. min_ev, like every *_pct column, is percent
    of the total stake; pairs that cannot lose (guaranteed_pct >= 0) are always kept. Sorted: those first
    (guaranteed, then ev), then the rest by ev_pct. Columns: COLS (player NaN for game lines; p_middle /
    breakeven_p are probabilities, breakeven_p = miss / (win + miss); the *_pct columns percent of the total
    stake; ev_pct is the exact sum over every outcome, pushes included).
    """
    if odds is None or len(odds) == 0 or fairs is None or len(fairs) == 0: return pd.DataFrame(columns=COLS)
    fk = {(r.event_id, r.market, _pkey(r.player)): (float(r.mu), r.sd, r.dist) for r in fairs.itertuples()}
    o = odds.reset_index(drop=True).drop_duplicates()
    o["_pkey"] = o.player.map(_pkey) if "player" in o else ""
    o["line"] = pd.to_numeric(o.line, errors="coerce") if "line" in o else np.nan
    o["price"] = pd.to_numeric(o.price, errors="coerce").astype(float)   # stable dtype whether or not a blank is dropped
    o["_dec"] = o.price.map(decimal_from_american)
    o = o[o._dec.notna()]                                        # a blank price cannot be the best price
    rows = []
    for (eid, mk, pk), e in o.groupby(["event_id", "market", "_pkey"]):
        fair = fk.get((eid, mk, pk))
        if fair is None or not np.isfinite(fair[0]): continue
        mu, sd, dist = fair
        best = e.loc[e.groupby(["side", "line", "book"], dropna=False)["_dec"].idxmax()]
        A, B = best[best.side.isin(SIDE_A)], best[best.side.isin(SIDE_B)]
        if A.empty or B.empty: continue
        k, p = support_pmf(dist, mu, None if sd is None or pd.isna(sd) else float(sd))
        home, away = e.home.iloc[0], e.away.iloc[0]
        player = e.player.iloc[0] if "player" in e and pk else np.nan
        for a in A.itertuples():
            for b in B.itertuples():
                same = a.book == b.book
                if same and not include_same_book: continue
                th = _thresholds(mk, dist, a.line, b.line)
                if th is None or th[1] < th[0] - EPS: continue          # both-lose gap
                r = price_pair(k, p, th[0], th[1], a.price, b.price)
                window, half, miss = len(r["window"]) > 0, len(r["half"]) > 0, r["miss_cost"]
                p_mid, win, wstr = r["p_middle"], r["win_both"], _window_str(r["window"])
                if r["guaranteed"] > EPS: typ = "arb+middle" if window else "arb"
                elif window: typ = "free_middle" if abs(r["guaranteed"]) <= EPS else "middle"
                elif half:                                                # one leg wins, the other pushes
                    typ, p_mid, win, wstr = "half_middle", r["p_half"], r["half_pay"], _window_str(r["half"], "p")
                else: continue                                            # scalp: no window, can lose
                rows.append(dict(
                    commence=a.commence if "commence" in e else np.nan, matchup=f"{away} @ {home}", market=mk,
                    player=player, type=typ,
                    bet_a=_bet(mk, a.side, a.line, a.price, a.book, home, away, player),
                    bet_b=_bet(mk, b.side, b.line, b.price, b.book, home, away, player),
                    window=wstr, p_middle=p_mid, miss_cost_pct=100 * miss, win_both_pct=100 * win,
                    ev_pct=100 * r["ev"], guaranteed_pct=100 * r["guaranteed"],
                    breakeven_p=miss / (win + miss) if win + miss > 0 else np.nan,
                    stake_a_pct=100 * r["stake_a"], stake_b_pct=100 * r["stake_b"], same_book=same,
                    event_id=eid, book_a=a.book, line_a=a.line, price_a=a.price,
                    book_b=b.book, line_b=b.line, price_b=b.price))
    res = pd.DataFrame(rows, columns=COLS)
    if res.empty: return res
    res = res[(res.guaranteed_pct >= -EPS) | (res.ev_pct >= min_ev)]
    safe = res.guaranteed_pct >= -EPS                                    # cannot lose: arbs, free (half) middles
    res = res.assign(_o=np.where(safe, 0, 1), _k=np.where(safe, res.guaranteed_pct, res.ev_pct))
    return (res.sort_values(["_o", "_k", "ev_pct"], ascending=[True, False, False])
            .drop(columns=["_o", "_k"]).reset_index(drop=True))
