"""Credit accounting: a game-line pull bills markets x regions, or markets x ceil(books/10) when books are named."""
import requests

from sharpmodel.odds import (odds_credits, book_list, fetch_odds, CORE_BOOKS, WIDE_BOOKS, ODDS_API, SPORT_KEY)


class _Resp:
    def __init__(self, payload, headers): self._payload, self.headers, self.status_code, self.text = payload, headers, 200, ""
    def json(self): return self._payload
    def raise_for_status(self): pass


def test_book_sets_and_estimates():
    assert len(CORE_BOOKS) == 10 and "pinnacle" in CORE_BOOKS and "draftkings" in CORE_BOOKS
    assert len(WIDE_BOOKS) == 20 and WIDE_BOOKS[:10] == CORE_BOOKS and len(set(WIDE_BOOKS)) == 20
    assert book_list("core") == CORE_BOOKS and book_list("wide") == WIDE_BOOKS
    assert book_list("a, b ,c") == ["a", "b", "c"] and book_list(["x"]) == ["x"]
    assert book_list(None) is None and book_list("") is None
    assert odds_credits(books="core") == 3 and odds_credits(books="wide") == 6
    assert odds_credits(books=",".join(f"b{i}" for i in range(21))) == 9           # 21 books = 3 region-equivalents
    assert odds_credits(books=None) == 9 and odds_credits(regions="us", books=None) == 3   # the old regions path
    assert odds_credits(markets="spreads", books="core") == 1


def test_fetch_odds_names_its_books_and_records_the_bill(monkeypatch):
    seen = {}
    def fake_get(url, params=None, timeout=None):
        assert url == ODDS_API.format(sport=SPORT_KEY["nfl"])
        seen.update(params)
        return _Resp([], {"x-requests-remaining": "497", "x-requests-last": "3"})
    monkeypatch.setattr(requests, "get", fake_get)
    df = fetch_odds("nfl", api_key="k")                                   # default: core
    assert seen["bookmakers"] == ",".join(CORE_BOOKS) and "regions" not in seen and seen["markets"] == "h2h,spreads,totals"
    assert df.attrs["remaining"] == 497 and df.attrs["cost"] == 3
    seen.clear()
    fetch_odds("nfl", api_key="k", books=None, regions="us")               # explicit regions path still works
    assert seen["regions"] == "us" and "bookmakers" not in seen
    seen.clear()
    fetch_odds("nfl", api_key="k", books="pinnacle,fanduel")
    assert seen["bookmakers"] == "pinnacle,fanduel"
