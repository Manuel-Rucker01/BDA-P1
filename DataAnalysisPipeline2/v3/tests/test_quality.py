"""Quality checks: big-move labelling and integrity stats."""
import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import quality as Q
from DataAnalysisPipeline2.v3._alpaca import md_table


def _row(t, d, c, raw, v=1e6):
    return {"ticker": t, "Date": d, "company_close": c, "close_raw": raw, "company_volume": v,
            "Open": c, "High": c, "Low": c}


def test_big_moves_labels():
    dates = pd.bdate_range("2020-01-01", periods=30).strftime("%Y-%m-%d")
    rows = []
    for i, d in enumerate(dates):
        rows.append(_row("REAL", d, 10.0 if i < 25 else 1.5, 10.0 if i < 25 else 1.5, 1e6 if i != 25 else 2e7))
        rows.append(_row("UNADJ", d, 50.0 if i < 25 else 5.0, 50.0 if i < 25 else 5.0))       # 10:1, not adjusted
        rows.append(_row("GLITCH", d, 10.0 if i < 25 else 50.0, 10.0))                          # raw flat
        rows.append(_row("OK", d, 10.0 + 0.01 * i, 10.0 + 0.01 * i))
    ev = Q.big_moves(pd.DataFrame(rows)).set_index("ticker")["label"]
    assert ev["REAL"] == "real_move" and ev["UNADJ"] == "unadjusted_split?"
    assert ev["GLITCH"] == "adjustment_glitch" and "OK" not in ev.index


def test_basic_stats_counts_problems():
    df = pd.DataFrame([_row("A", "2020-01-02", 10, 10), _row("A", "2020-01-02", 10, 10),
                       _row("B", "2020-01-02", -1, np.nan, 0.0)])
    df["vwap"] = df["trade_count"] = 1.0
    st = Q.basic_stats(df).loc["2020"]
    assert st["duplicates"] == 1 and st["non_positive_close"] == 1
    assert st["missing_close_raw"] == 1 and st["zero_volume_bars"] == 1


def test_md_table():
    t = md_table(pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]}), index=False)
    assert t.splitlines()[0] == "| a | b |" and t.splitlines()[3] == "| 2 |  |"
