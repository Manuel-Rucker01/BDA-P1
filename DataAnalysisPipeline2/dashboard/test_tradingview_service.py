import csv
import io
import json

from DataAnalysisPipeline2.dashboard.tradingview_service import (
    artifact_payload,
    build_tradingview_package,
)


def _sample_result():
    return {
        "strategy": "top_k",
        "universe": "high_alpha",
        "regime": "bull",
        "gross_exposure_pct": 100.0,
        "proposals": [
            {
                "ticker": "AAPL",
                "weight": 0.25,
                "weight_pct": 25.0,
                "side": "LONG",
                "pred_rank": 98.1,
                "price": 200.0,
                "sector": "Technology",
            },
            {
                "ticker": "TSLA",
                "weight": -0.10,
                "weight_pct": -10.0,
                "side": "SHORT",
                "pred_rank": 12.0,
                "price": 180.0,
                "sector": "Consumer Discretionary",
            },
        ],
    }


def test_build_tradingview_package_contains_expected_artifacts():
    package = build_tradingview_package("job123", _sample_result())

    assert package["ok"] is True
    assert "direct retail account order API" in package["disclaimer"]
    assert "NASDAQ:AAPL" in package["artifacts"]["watchlist"]
    assert "NASDAQ:TSLA" in package["artifacts"]["watchlist"]
    assert "BDA proposal" in package["artifacts"]["pine_script"]

    rows = list(csv.DictReader(io.StringIO(package["artifacts"]["orders_csv"])))
    assert rows[0]["symbol"] == "NASDAQ:AAPL"
    assert rows[0]["action"] == "BUY"
    assert rows[1]["action"] == "SELL_SHORT"

    alerts = json.loads(package["artifacts"]["webhook_json"])
    assert alerts["job_id"] == "job123"
    payload = json.loads(alerts["alerts"][0]["message"])
    assert payload["ticker"] == "AAPL"
    assert payload["target_weight"] == 0.25


def test_artifact_payload_sets_download_metadata():
    package = build_tradingview_package("job123", _sample_result(), exchange_prefix="NYSE")

    content, mime, filename = artifact_payload(package, "orders_csv")

    assert "NYSE:AAPL" in content
    assert mime == "text/csv"
    assert filename == "tradingview_orders_csv.csv"
