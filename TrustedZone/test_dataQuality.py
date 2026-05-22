"""
Unit tests for TrustedZone Data Quality Pipeline.

Covers:
- Database preparation and verification
- Data extraction from FormattedZone (all six source tables)
- Data quality / denial-constraint application
- Data validation
- Data writing (incl. the enriched `companies` table built in the TrustedZone)
- End-to-end pipeline integration
"""

import os
import sys
import tempfile

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataQuality import (
    prepare_trusted_database,
    initialize_spark,
    extract_and_filter_data,
    validate_cleaned_data,
    write_to_trusted_zone,
    verify_trusted_zone_database,
)

# ── Shared Spark session ──────────────────────────────────────────────────────

_spark = None


def get_spark():
    global _spark
    if _spark is None:
        _spark = initialize_spark()
    return _spark


@pytest.fixture(scope="session", autouse=True)
def cleanup_spark():
    yield
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_default_frames():
    """Default valid frames for the six FormattedZone tables."""
    nasdaq = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "GOOGL"],
        "Name": ["Apple", "Microsoft", "Google"],
        "LastSale": [150.0, 300.0, 2800.0],
        "MarketCap": [2500e9, 2300e9, 1900e9],
        "IPOyear": [1980, 1986, 2004],
        "Sector": ["Tech", "Tech", "Tech"],
        "Industry": ["Computer", "Software", "Internet"],
    })
    history = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        "Company": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "Open": [380.0, 385.0, 240.0, 245.0],
        "High": [390.0, 395.0, 250.0, 255.0],
        "Low": [375.0, 380.0, 235.0, 240.0],
        "Close": [385.0, 390.0, 248.0, 252.0],
        "Volume": [1_000_000, 1_100_000, 800_000, 850_000],
    })
    exchange = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "EUR": [0.92, 0.93],
        "JPY": [133.0, 134.0],
    })
    sp500 = pd.DataFrame({
        "Ticker": ["AAPL"], "Name": ["Apple Inc."], "Sector": ["Tech"],
        "Industry": ["Computer"], "Country": ["United States"],
        "MarketCap": [2500e9], "Employees": [150_000],
    })
    forbes = pd.DataFrame({
        "company": ["Apple"], "rank": [1],
        "country_territory": ["United States"], "publish_year": [2024],
    })
    acq = pd.DataFrame({
        "ID": [1, 2], "Parent Company": ["Apple", "Microsoft"],
        "Acquired Company": ["Beats", "GitHub"],
        "Business": ["Audio", "Software"],
        "Country": ["United States", "United States"],
        "Acquisition Year": [2014, 2018],
        "Acquisition Price": [3.0e9, 7.5e9],
        "Category": ["Hardware", "Software"],
    })
    return {
        "nasdaq": nasdaq, "company_history": history, "us_exchange": exchange,
        "sp500_companies": sp500, "forbes_employers": forbes,
        "company_acquisitions": acq,
    }


def _write_formatted_zone(db_path, frames=None):
    """Create a FormattedZone DuckDB at `db_path` with the six required tables."""
    frames = frames or _make_default_frames()
    conn = duckdb.connect(db_path)
    for table, df in frames.items():
        view = f"_{table}_view"
        conn.register(view, df)
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM {view}")
    conn.close()


# ── TestDatabasePreparation ───────────────────────────────────────────────────

class TestDatabasePreparation:
    def test_prepare_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "t.duckdb")
            prepare_trusted_database(db)
            assert os.path.exists(db)
            duckdb.connect(db).close()

    def test_prepare_clears_existing_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "t.duckdb")
            conn = duckdb.connect(db)
            conn.execute("CREATE TABLE nasdaq (symbol VARCHAR)")
            conn.execute("CREATE TABLE company_history (date DATE)")
            conn.close()
            prepare_trusted_database(db)
            conn = duckdb.connect(db)
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
            conn.close()
            assert len(tables) == 0

    def test_prepare_raises_on_invalid_path(self):
        with pytest.raises(Exception):
            prepare_trusted_database("/no/such/dir/db.duckdb")


# ── TestSparkInitialization ───────────────────────────────────────────────────

class TestSparkInitialization:
    def test_initialize_spark_returns_session(self):
        spark = get_spark()
        assert spark is not None
        assert spark.sparkContext is not None

    def test_spark_can_create_dataframe(self):
        spark = get_spark()
        df = spark.createDataFrame([("test", 1)], ["name", "value"])
        assert df.schema.names == ["name", "value"]


# ── TestDataExtraction ────────────────────────────────────────────────────────

class TestDataExtraction:

    @pytest.fixture
    def formatted_zone_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "FormattedZone.duckdb")
            _write_formatted_zone(db)
            yield db

    def test_extract_returns_cleaned_data_and_metrics(self, formatted_zone_db):
        cleaned, metrics = extract_and_filter_data(formatted_zone_db, get_spark())

        assert isinstance(cleaned, dict)
        for key in ("nasdaq", "company_history", "us_exchange",
                    "sp500_companies", "forbes_employers", "company_acquisitions"):
            assert key in cleaned

        for key in ("raw_counts", "clean_counts", "rows_removed",
                    "removal_rate", "denial_constraints"):
            assert key in metrics

        assert "Symbol" in cleaned["nasdaq"].columns
        assert "Date" in cleaned["company_history"].columns
        assert "EUR" in cleaned["us_exchange"].columns

    def test_extract_raises_on_missing_database(self):
        with pytest.raises(Exception):
            extract_and_filter_data("/nonexistent/db.duckdb", get_spark())

    def test_extract_removes_invalid_nasdaq_rows(self):
        frames = _make_default_frames()
        frames["nasdaq"] = pd.DataFrame({
            "Symbol": ["AAPL", "BAD1", None],
            "Name": ["Apple", None, "BadCorp"],
            "LastSale": [150.0, -100.0, 200.0],
            "MarketCap": [2500e9, -500e9, 300e9],
            "IPOyear": [1980, 2050, 2000],
            "Sector": ["Tech", "Tech", "Tech"],
            "Industry": ["Computer", "Software", "Internet"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "FormattedZone.duckdb")
            _write_formatted_zone(db, frames)
            cleaned, metrics = extract_and_filter_data(db, get_spark())
            assert metrics["rows_removed"]["nasdaq"] >= 1
            assert len(cleaned["nasdaq"]) < 3


# ── TestDataValidation ────────────────────────────────────────────────────────

class TestDataValidation:

    def _seed_clean_trusted(self, db_path):
        prepare_trusted_database(db_path)
        cleaned = {k: v.copy() for k, v in _make_default_frames().items()}
        write_to_trusted_zone(db_path, cleaned, {"denial_constraints": {}})

    def test_validate_passes_on_clean_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trusted.duckdb")
            self._seed_clean_trusted(db)
            conn = duckdb.connect(db)
            assert validate_cleaned_data(conn) is True
            conn.close()

    def test_validate_fails_on_invalid_nasdaq(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "invalid.duckdb")
            prepare_trusted_database(db)
            cleaned = {k: v.copy() for k, v in _make_default_frames().items()}
            # Inject a null Symbol that should fail validation
            cleaned["nasdaq"] = pd.concat([
                cleaned["nasdaq"],
                pd.DataFrame([{
                    "Symbol": None, "Name": "Bad", "LastSale": 1.0,
                    "MarketCap": 1.0, "IPOyear": 2000,
                    "Sector": "X", "Industry": "Y",
                }]),
            ], ignore_index=True)
            write_to_trusted_zone(db, cleaned, {"denial_constraints": {}})
            conn = duckdb.connect(db)
            assert validate_cleaned_data(conn) is False
            conn.close()


# ── TestDataWriting ───────────────────────────────────────────────────────────

class TestDataWriting:

    def test_write_creates_all_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trusted.duckdb")
            prepare_trusted_database(db)
            cleaned = {k: v.copy() for k, v in _make_default_frames().items()}
            write_to_trusted_zone(db, cleaned, {"denial_constraints": {}})

            conn = duckdb.connect(db)
            names = {row[0] for row in conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()}
            conn.close()

            for expected in ("nasdaq", "company_history", "us_exchange",
                             "sp500_companies", "forbes_employers",
                             "company_acquisitions", "companies",
                             "data_quality_metrics"):
                assert expected in names, f"missing table: {expected}"

    def test_write_preserves_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trusted.duckdb")
            prepare_trusted_database(db)
            cleaned = {k: v.copy() for k, v in _make_default_frames().items()}
            write_to_trusted_zone(db, cleaned, {"denial_constraints": {}})

            conn = duckdb.connect(db)
            assert conn.execute("SELECT COUNT(*) FROM nasdaq").fetchone()[0] == 3
            assert conn.execute(
                "SELECT COUNT(*) FROM us_exchange").fetchone()[0] == 2
            # The enriched companies table is created downstream of the join
            assert conn.execute(
                "SELECT COUNT(*) FROM companies").fetchone()[0] == 3
            # AAPL must resolve to United States via sp500.Country
            assert conn.execute(
                "SELECT country FROM companies WHERE Symbol = 'AAPL'"
            ).fetchone()[0] == "United States"
            conn.close()


# ── TestDatabaseVerification ──────────────────────────────────────────────────

class TestDatabaseVerification:

    def test_verify_passes_on_valid_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trusted.duckdb")
            prepare_trusted_database(db)
            cleaned = {k: v.copy() for k, v in _make_default_frames().items()}
            write_to_trusted_zone(db, cleaned, {"denial_constraints": {}})
            assert verify_trusted_zone_database(db) is True

    def test_verify_fails_on_missing_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "incomplete.duckdb")
            conn = duckdb.connect(db)
            conn.execute("CREATE TABLE nasdaq (symbol VARCHAR)")
            conn.close()
            assert verify_trusted_zone_database(db) is False


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_with_valid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            formatted_db = os.path.join(tmp, "FormattedZone.duckdb")
            trusted_db = os.path.join(tmp, "TrustedZone.duckdb")
            _write_formatted_zone(formatted_db)

            prepare_trusted_database(trusted_db)
            cleaned, metrics = extract_and_filter_data(formatted_db, get_spark())
            write_to_trusted_zone(trusted_db, cleaned, metrics)

            conn = duckdb.connect(trusted_db)
            assert validate_cleaned_data(conn) is True
            conn.close()
            assert verify_trusted_zone_database(trusted_db) is True

    def test_pipeline_filters_invalid_records(self):
        frames = _make_default_frames()
        # Inject invalid rows in each main source
        frames["nasdaq"] = pd.concat([
            frames["nasdaq"],
            pd.DataFrame([{
                "Symbol": None, "Name": "Invalid", "LastSale": -10.0,
                "MarketCap": 1.0, "IPOyear": 2030,
                "Sector": "X", "Industry": "Y",
            }]),
        ], ignore_index=True)
        frames["company_history"] = pd.concat([
            frames["company_history"],
            pd.DataFrame([{
                "Date": "2023-01-03", "Company": "AAPL",
                "Open": 1.0, "High": 5.0, "Low": 10.0,  # High < Low
                "Close": 3.0, "Volume": -100,           # negative volume
            }]),
        ], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmp:
            formatted_db = os.path.join(tmp, "FormattedZone.duckdb")
            trusted_db = os.path.join(tmp, "TrustedZone.duckdb")
            _write_formatted_zone(formatted_db, frames)

            prepare_trusted_database(trusted_db)
            cleaned, metrics = extract_and_filter_data(formatted_db, get_spark())
            write_to_trusted_zone(trusted_db, cleaned, metrics)

            assert metrics["clean_counts"]["nasdaq"] < metrics["raw_counts"]["nasdaq"]
            assert metrics["clean_counts"]["company_history"] < metrics["raw_counts"]["company_history"]
