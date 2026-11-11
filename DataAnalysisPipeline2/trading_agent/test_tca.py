"""TCA of paper fills (trading_agent/tca.py): offline tests on synthetic data, no network."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.trading_agent import tca
from DataAnalysisPipeline2.v3 import costs as cs

DATES = pd.bdate_range("2026-06-01", "2026-11-06").strftime("%Y-%m-%d").tolist()


def _bars(tickers=("AAA", "BBB"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k, t in enumerate(tickers):
        px = 50.0 * (k + 1) * np.cumprod(1 + rng.normal(0, 0.01 * (k + 1), len(DATES)))
        for d, p in zip(DATES, px):
            rows.append({"ticker": t, "Date": d, "Open": p * 0.999, "High": p * 1.01, "Low": p * 0.99,
                         "company_close": p, "company_volume": 1e5 * (10 ** k), "VWAP": p * 1.001})
    return pd.DataFrame(rows)


def _fills():
    return pd.DataFrame({
        "symbol": ["AAA", "BBB"], "side": ["buy", "sell"], "type": "market",
        "filled_qty": ["10", "4"], "client_order_id": ["bda-x-AAA", "other"],
        "submitted_at": ["2026-11-06T16:30:57Z", "2026-11-06T16:30:58Z"],
        "filled_at": ["2026-11-06T16:31:00Z", "2026-11-06T16:31:01Z"]})


def test_shortfall_sign_and_nan():
    assert tca.shortfall_bps("buy", 101.0, 100.0) == pytest.approx(100.0)
    assert tca.shortfall_bps("sell", 101.0, 100.0) == pytest.approx(-100.0)   # selling above ref = gain
    assert tca.shortfall_bps("sell", 99.0, 100.0) == pytest.approx(100.0)
    assert np.isnan(tca.shortfall_bps("buy", 1.0, 0.0))
    assert np.isnan(tca.shortfall_bps("buy", 1.0, None))


def test_liquidity_bucket_and_decision_date():
    assert tca.liquidity_bucket(5e6) == "<$10M"
    assert tca.liquidity_bucket(10e6) == "$10-50M"
    assert tca.liquidity_bucket(1e9) == ">$250M"
    assert tca.liquidity_bucket(float("nan")) == "unknown"
    ds = ["2026-10-05", "2026-11-05"]
    assert tca.decision_date_for("2026-11-06", ds) == "2026-11-05"
    assert tca.decision_date_for("2026-11-05", ds) == "2026-10-05"     # strictly before the trade day
    assert tca.decision_date_for("2026-10-01", ds) is None


def test_modelled_cost_matches_costs_py():
    hs, imp = tca.modelled_cost_bps(0.002, 0.02, 1e7, 1e5)
    assert hs == pytest.approx(20.0)
    assert imp == pytest.approx(cs.IMPACT_K * 0.02 * np.sqrt(0.01) * 1e4)
    # same number as the optimizer's c|dw| + d|dw|^1.5 at AUM = notional / dw
    aum, dw = 1e6, 0.1
    total = float(cs.trade_cost(dw, 0.002, 0.02, 1e7, aum)) / dw * 1e4
    assert sum(tca.modelled_cost_bps(0.002, 0.02, 1e7, aum * dw)) == pytest.approx(total)
    hs, imp = tca.modelled_cost_bps(float("nan"), float("nan"), float("nan"), 1e4)
    assert hs == pytest.approx(cs.DEFAULT_HS * 1e4) and np.isnan(imp)


def test_cost_inputs_point_in_time():
    b = _bars()
    ci = tca.cost_inputs(b, "2026-11-05")
    assert set(ci.index) == {"AAA", "BBB"}
    assert ci["cs_ok"].all()
    assert (ci["hs"] >= cs.HS_FLOOR).all() and (ci["hs"] <= cs.HS_CAP).all()
    last = b[(b.ticker == "AAA") & (b.Date <= "2026-11-05")].tail(20)
    assert ci.loc["AAA", "adv"] == pytest.approx((last.company_close * last.company_volume).mean())
    # changing the trade day (after the decision date) must not change the inputs
    b2 = b.copy()
    b2.loc[b2.Date == "2026-11-06", ["High", "Low", "company_close"]] *= 3
    pd.testing.assert_frame_equal(ci, tca.cost_inputs(b2, "2026-11-05"))


def test_mid_from_quotes_last_valid_before_t():
    q = pd.DataFrame({"timestamp": pd.to_datetime(["2026-11-06T16:30:50Z", "2026-11-06T16:30:55Z",
                                                   "2026-11-06T16:30:56Z", "2026-11-06T16:31:30Z"]),
                      "bid_price": [9.9, 10.0, 0.0, 11.0], "ask_price": [10.1, 10.2, 10.3, 11.2]})
    bid, ask, mid = tca.mid_from_quotes(q, "2026-11-06T16:30:57Z")
    assert (bid, ask) == (10.0, 10.2) and mid == pytest.approx(10.1)   # zero bid skipped, future quote ignored
    assert np.isnan(tca.mid_from_quotes(None, "2026-11-06")[2])


def test_analyse_and_aggregate():
    b = _bars()
    f = _fills()
    dc = b.set_index(["ticker", "Date"]).loc[("AAA", "2026-11-05"), "company_close"]
    f["filled_avg_price"] = [str(dc * 1.002), "100.0"]

    def quote_fn(sym, t):
        if sym == "SPY":
            return 499.0, 501.0, 500.0
        if sym == "AAA":
            return dc * 0.999, dc * 1.001, dc
        return float("nan"), float("nan"), float("nan")

    spy = pd.DataFrame({"Date": ["2026-11-05"], "company_close": [495.0]})
    df = tca.analyse(f, bars_adj=b, bars_raw=b, decision_dates=["2026-11-05"], quote_fn=quote_fn,
                     equity_by_date={"2026-11-05": 1e5}, spy_raw=spy)
    assert list(df.columns) == tca.CSV_COLUMNS
    a = df.set_index("symbol").loc["AAA"]
    assert a["decision_date"] == "2026-11-05" and a["trade_date"] == "2026-11-06"
    assert bool(a["bot_tagged"]) and not bool(df.set_index("symbol").loc["BBB", "bot_tagged"])
    assert a["is_decision_bps"] == pytest.approx(20.0)
    assert a["is_arrival_bps"] == pytest.approx(20.0)
    assert a["arrival_spread_bps"] == pytest.approx(20.0)
    spy_move = (500.0 / 495.0 - 1) * 1e4
    assert a["is_decision_mktadj_bps"] == pytest.approx(20.0 - spy_move)
    ci = tca.cost_inputs(b, "2026-11-05").loc["AAA"]
    assert a["hs_model_bps"] == pytest.approx(ci["hs"] * 1e4)
    assert a["impact_model_bps"] == pytest.approx(cs.IMPACT_K * ci["sigma"] * np.sqrt(a["notional"] / ci["adv"]) * 1e4)
    assert a["impact_100m_bps"] == pytest.approx(a["impact_model_bps"] * np.sqrt(1e8 / 1e5))
    assert np.isnan(df.set_index("symbol").loc["BBB", "is_arrival_bps"])

    agg = tca.aggregate(df)
    allr = agg[agg.bucket == "all"].iloc[0]
    assert allr["n"] == 2
    w = df["notional"]
    assert allr["model_bps"] == pytest.approx(float((df["model_bps"] * w).sum() / w.sum()))
    assert allr["is_arrival_bps"] == pytest.approx(20.0)      # NaN rows are excluded from the weighted mean
    md = tca.render_markdown(df, agg, "monthly")
    assert "paper fills are optimistic" in md and "| **all** | 2 |" in md


def test_empty_and_outputs_deterministic(tmp_path):
    empty = tca.analyse(pd.DataFrame(), bars_adj=_bars(), bars_raw=_bars(), decision_dates=[])
    assert empty.empty and list(empty.columns) == tca.CSV_COLUMNS
    assert "No filled orders yet" in tca.render_markdown(empty, tca.aggregate(empty), "monthly")
    b = _bars()
    f = _fills().assign(filled_avg_price=["50", "100"])
    df = tca.analyse(f, bars_adj=b, bars_raw=b, decision_dates=["2026-11-05"])
    p1 = [p.read_bytes() for p in tca.write_outputs(df, tca.aggregate(df), tmp_path, "monthly")]
    p2 = [p.read_bytes() for p in tca.write_outputs(df, tca.aggregate(df), tmp_path, "monthly")]
    assert p1 == p2


class _FakeApi:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, path, **params):
        self.calls.append((path, params))
        return self.pages.pop(0) if self.pages else []


def test_fetch_fills_paginates_and_keeps_filled_only():
    def o(i, filled="1"):
        return {"id": f"o{i}", "client_order_id": f"bda-{i}", "symbol": "AAA", "side": "buy", "type": "market",
                "qty": "1", "filled_qty": filled, "filled_avg_price": "10" if filled != "0" else None,
                "submitted_at": f"2026-11-06T16:{i // 60:02d}:{i % 60:02d}Z", "filled_at": "2026-11-06T17:00:00Z",
                "status": "filled" if filled != "0" else "canceled"}
    page1 = [o(i) for i in range(500)]
    page2 = [o(499)] + [o(500), o(501, "0")]
    api = _FakeApi([page1, page2])
    f = tca.fetch_fills(api, "2026-10-01")
    assert len(f) == 501 and f["id"].is_unique
    assert api.calls[1][1]["after"] == page1[-1]["submitted_at"]
    assert all(c[0] == "/v2/orders" for c in api.calls)
