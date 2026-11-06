"""Price store: NY dates, fetching, resumable bulk download, incremental update."""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import _alpaca, prices
from DataAnalysisPipeline2.v3.tests._fakes import FakeBarsSession, fake_client, make_bars

END = dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc)  # "previous UTC midnight"


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BDA_DATA_V3", str(tmp_path))
    return tmp_path


def _dataset(dates, tickers=("AAA", "BBB", "CCC", "DDD", "EEE")):
    out = {"all": {}, "raw": {}}
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(i)  # per ticker: series independent of the date range length
        px = 20 * (i + 1) * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        out["all"][t] = make_bars(dates, px)
        out["raw"][t] = make_bars(dates, px * (2.0 if t == "AAA" else 1.0))  # AAA split 2:1 later
    return out


DATES = pd.bdate_range("2023-11-01", "2024-02-29").strftime("%Y-%m-%d").tolist()


# --------------------------------------------------------------------- dates
def test_ny_date_conversion_handles_dst_and_late_utc():
    ts = pd.Series(["2024-03-08T05:00:00Z",   # EST session start
                    "2024-03-11T04:00:00Z",   # EDT session start (after DST switch)
                    "2024-03-08T02:00:00Z",   # still 2024-03-07 in New York
                    "2023-12-29T05:00:00Z"])  # year boundary
    assert _alpaca.to_ny_date(ts).tolist() == ["2024-03-08", "2024-03-11", "2024-03-07", "2023-12-29"]


def test_last_completed_close_is_previous_utc_midnight():
    now = dt.datetime(2024, 3, 5, 15, 30, tzinfo=dt.timezone.utc)
    assert _alpaca.last_completed_close_utc(now) == dt.datetime(2024, 3, 5, tzinfo=dt.timezone.utc)
    early = dt.datetime(2024, 3, 5, 0, 5, tzinfo=dt.timezone.utc)  # inside the "recent" window
    assert _alpaca.last_completed_close_utc(early) < early - dt.timedelta(minutes=15)


# ------------------------------------------------------------------ fetching
def test_get_bars_paginates_drops_invalid_and_survives_429():
    data = _dataset(DATES[:5])
    sess = FakeBarsSession(data, page_size=3, invalid={"BAD.WS"}, rate_limit_first=True)
    df, invalid = fake_client(sess).get_bars(["AAA", "BAD.WS", "BBB", "NOPE"], "2023-01-01", END)
    assert invalid == ["BAD.WS"]
    assert sorted(df["symbol"].unique()) == ["AAA", "BBB"]  # unknown NOPE silently absent
    assert len(df) == 10 and not df.duplicated(["symbol", "timestamp"]).any()


def test_fetch_chunk_adds_raw_close_and_ny_dates():
    data = _dataset(DATES[:5])
    df, _ = prices.fetch_chunk(fake_client(FakeBarsSession(data)), ["AAA", "BBB"], "2023-01-01", END)
    assert list(df.columns) == prices.STORE_COLUMNS
    assert df["Date"].tolist()[:5] == DATES[:5]
    a = df[df["ticker"] == "AAA"]
    np.testing.assert_allclose(a["close_raw"] / a["company_close"], 2.0)


# ---------------------------------------------------------- bulk + resuming
def test_download_is_resumable_and_writes_year_partitions():
    data = _dataset(DATES)
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE", "ZZZ"]
    crash = FakeBarsSession(data, page_size=50, fail_after=10)  # chunk 1 = 8 requests, crash inside chunk 2
    with pytest.raises(KeyboardInterrupt):
        prices.download(syms, start="2023-01-01", end=END, chunk_size=2, workers=1,
                        client=fake_client(crash))
    staged = [f for f in os.listdir(prices.prices_dir("_staging", "full_20240301")) if f.endswith(".parquet")]
    assert 0 < len(staged) < 3
    resume = FakeBarsSession(data, page_size=50)
    meta = prices.download(syms, start="2023-01-01", end=END, chunk_size=2, workers=1,
                           client=fake_client(resume))
    # chunks finished before the crash are not fetched again
    fetched = {s for c in resume.calls for s in c["symbols"].split(",")}
    assert not ({"AAA", "BBB"} & fetched)
    assert prices.stored_years() == [2023, 2024]
    assert meta["last_date"] == DATES[-1] and "ZZZ" in meta["requested"] and "ZZZ" not in meta["with_data"]
    df = prices.load_prices(clean=False)
    assert len(df) == 5 * len(DATES) and not df.duplicated(["ticker", "Date"]).any()
    assert not os.path.exists(os.path.join(prices.prices_dir("_staging"), "full_20240301"))  # cleaned up


def test_load_prices_filters_and_adj_factor():
    data = _dataset(DATES)
    prices.download(["AAA", "BBB"], start="2023-01-01", end=END, chunk_size=5, workers=1,
                    client=fake_client(FakeBarsSession(data, page_size=500)))
    df = prices.load_prices("2024-01-02", "2024-01-31", tickers=["AAA"])
    assert set(df["ticker"]) == {"AAA"} and df["Date"].min() >= "2024-01-02" and df["Date"].max() <= "2024-01-31"
    np.testing.assert_allclose(df["adj_factor"], 0.5)
    assert "close_raw" not in prices.load_prices(with_raw=False).columns


# -------------------------------------------------------------- incremental
def test_update_fetches_new_sessions_and_refetches_split_tickers():
    old_dates, new_dates = DATES[:-10], DATES
    prices.download(["AAA", "BBB", "CCC"], start="2023-01-01", end=dt.datetime(2024, 2, 16, tzinfo=dt.timezone.utc),
                    chunk_size=2, workers=1, client=fake_client(FakeBarsSession(_dataset(old_dates), page_size=500)))
    assert prices.read_meta()["last_date"] == old_dates[-1]

    data = _dataset(new_dates, tickers=("AAA", "BBB", "CCC", "NEW"))
    # BBB splits 3:1 on the 3rd-last session: the adjusted history is rescaled.
    b = data["all"]["BBB"].copy()
    b.loc[: len(b) - 3, ["o", "h", "l", "c", "vw"]] /= 3.0
    data["all"]["BBB"] = b
    sess = FakeBarsSession(data, page_size=500)
    meta = prices.update(["AAA", "BBB", "CCC", "NEW"], overlap_sessions=5, end=END, chunk_size=2,
                         workers=1, client=fake_client(sess))
    assert meta["last_date"] == new_dates[-1]
    hist = meta["history"][-1]
    assert hist["splits"] == ["BBB"] and hist["n_new"] == 1
    full_calls = [c for c in sess.calls if c["start"].startswith("2016")]
    assert {s for c in full_calls for s in c["symbols"].split(",")} == {"BBB", "NEW"}
    df = prices.load_prices(clean=False)
    assert not df.duplicated(["ticker", "Date"]).any()
    for t in ("AAA", "BBB", "CCC", "NEW"):
        got = df[df["ticker"] == t].set_index("Date")["company_close"]
        np.testing.assert_allclose(got.reindex(new_dates).values, data["all"][t]["c"].values)


def test_adjustment_changes_classifies_split_vs_dividend():
    old = pd.DataFrame({"ticker": ["A", "A", "B", "B", "C"], "Date": ["d1", "d2"] * 2 + ["d1"],
                        "company_close": [10.0, 11.0, 10.0, 11.0, 5.0]})
    new = old.copy()
    new.loc[new["ticker"] == "A", "company_close"] /= 2       # split
    new.loc[new["ticker"] == "B", "company_close"] *= 0.995   # dividend re-scale
    ch = prices.adjustment_changes(old, new).set_index("ticker")
    assert ch.loc["A", "kind"] == "split" and ch.loc["B", "kind"] == "dividend/other" and "C" not in ch.index
