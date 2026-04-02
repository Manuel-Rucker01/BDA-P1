"""
Unit tests for TrustedZone Data Quality Pipeline

Tests all functions in dataQuality.py including:
- Database preparation and verification
- Data extraction from FormattedZone
- Data quality rules application
- Data validation
- Data writing and verification
"""

import os
import sys
import tempfile
import shutil
import json
import pytest
import pandas as pd
import duckdb
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataQuality import (
    prepare_trusted_database,
    initialize_spark,
    extract_and_filter_data,
    validate_cleaned_data,
    write_to_trusted_zone,
    verify_trusted_zone_database
)

# Global Spark session - reuse across tests to avoid crashes
_spark = None

def get_spark():
    """Get or create a single Spark session for all tests."""
    global _spark
    if _spark is None:
        _spark = initialize_spark()
    return _spark

@pytest.fixture(scope="session", autouse=True)
def cleanup_spark():
    """Cleanup Spark session after all tests."""
    yield
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None


class TestDatabasePreparation:
    """Tests for database preparation functions."""
    
    def test_prepare_trusted_database_creates_file(self):
        """Test that prepare_trusted_database creates a database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.duckdb")
            
            # Should not raise an error
            prepare_trusted_database(db_path)
            
            # Database file should exist
            assert os.path.exists(db_path)
            
            # Should be a valid DuckDB database
            conn = duckdb.connect(db_path)
            conn.close()
    
    def test_prepare_trusted_database_clears_existing_tables(self):
        """Test that prepare_trusted_database drops existing tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.duckdb")
            
            # Create database with tables
            conn = duckdb.connect(db_path)
            conn.execute("CREATE TABLE nasdaq (symbol VARCHAR)")
            conn.execute("CREATE TABLE company_history (date DATE)")
            conn.execute("CREATE TABLE us_exchange (rate FLOAT)")
            conn.close()
            
            # Verify tables exist
            conn = duckdb.connect(db_path)
            tables = conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
            conn.close()
            assert len(tables) == 3
            
            # Run prepare_trusted_database
            prepare_trusted_database(db_path)
            
            # Verify tables are dropped
            conn = duckdb.connect(db_path)
            tables = conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
            conn.close()
            assert len(tables) == 0
    
    def test_prepare_trusted_database_raises_on_invalid_path(self):
        """Test that prepare_trusted_database raises error for invalid paths."""
        invalid_path = "/invalid/path/that/does/not/exist/db.duckdb"
        
        with pytest.raises(Exception):
            prepare_trusted_database(invalid_path)


class TestSparkInitialization:
    """Tests for Spark session initialization."""
    
    def test_initialize_spark_returns_session(self):
        """Test that initialize_spark returns a valid Spark session."""
        spark = get_spark()
        
        assert spark is not None
        assert spark.sparkContext is not None
    
    def test_spark_session_can_create_dataframe(self):
        """Test that the initialized Spark session can create DataFrames."""
        spark = get_spark()
        
        # Create a simple DataFrame
        data = [("test", 1)]
        df = spark.createDataFrame(data, ["name", "value"])
        
        # Verify schema instead of count() to avoid serialization on Windows
        assert df.schema.names == ["name", "value"]
        assert len(df.schema) == 2


class TestDataExtraction:
    """Tests for data extraction from FormattedZone."""
    
    @pytest.fixture
    def formatted_zone_database(self):
        """Create a test FormattedZone database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "FormattedZone.duckdb")
            
            conn = duckdb.connect(db_path)
            
            # Create NASDAQ table
            nasdaq_data = {
                "Symbol": ["AAPL", "MSFT", "GOOGL"],
                "Name": ["Apple", "Microsoft", "Google"],
                "LastSale": [150.0, 300.0, 2800.0],
                "MarketCap": [2500e9, 2300e9, 1900e9],
                "IPOyear": [1980, 1986, 2004]
            }
            nasdaq_df = pd.DataFrame(nasdaq_data)
            conn.register('nasdaq_data', nasdaq_df)
            conn.execute("CREATE TABLE nasdaq AS SELECT * FROM nasdaq_data")
            
            # Create Company History table
            company_history_data = {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": [380.0, 385.0],
                "High": [390.0, 395.0],
                "Low": [375.0, 380.0],
                "Close": [385.0, 390.0],
                "Volume": [1000000, 1100000]
            }
            company_history_df = pd.DataFrame(company_history_data)
            conn.register('company_history_data', company_history_df)
            conn.execute("CREATE TABLE company_history AS SELECT * FROM company_history_data")
            
            # Create Exchange table
            exchange_data = {
                "Date": ["2023-01-01", "2023-01-02"],
                "EUR": [0.92, 0.93],
                "JPY": [133.0, 134.0]
            }
            exchange_df = pd.DataFrame(exchange_data)
            conn.register('exchange_data', exchange_df)
            conn.execute("CREATE TABLE us_exchange AS SELECT * FROM exchange_data")
            
            conn.close()
            
            yield db_path
    
    def test_extract_and_filter_data_returns_cleaned_data_and_metrics(self, formatted_zone_database):
        """Test data extraction and filtering from FormattedZone."""
        spark = get_spark()
        
        cleaned_data, metrics = extract_and_filter_data(formatted_zone_database, spark)
        
        # Verify structure
        assert isinstance(cleaned_data, dict)
        assert isinstance(metrics, dict)
        
        # Verify cleaned_data has expected keys
        assert "nasdaq" in cleaned_data
        assert "company_history" in cleaned_data
        assert "exchange" in cleaned_data
        
        # Verify metrics has expected keys
        assert "raw_counts" in metrics
        assert "clean_counts" in metrics
        assert "rows_removed" in metrics
        assert "removal_rate" in metrics
        assert "denial_constraints" in metrics
        
        # Verify row counts
        assert len(cleaned_data["nasdaq"]) == 3
        assert len(cleaned_data["company_history"]) == 2
        assert len(cleaned_data["exchange"]) == 2
        
        # Verify column names
        assert "Symbol" in cleaned_data["nasdaq"].columns
        assert "Date" in cleaned_data["company_history"].columns
        assert "EUR" in cleaned_data["exchange"].columns
    
    def test_extract_raises_on_missing_database(self):
        """Test that extraction raises error for missing database."""
        spark = get_spark()
        
        with pytest.raises(Exception):
            extract_and_filter_data("/nonexistent/database.duckdb", spark)
    
    def test_extract_and_filter_removes_invalid_records(self):
        """Test that invalid records are removed during extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "FormattedZone.duckdb")
            
            conn = duckdb.connect(db_path)
            
            # Create NASDAQ table with invalid data
            nasdaq_data = {
                "Symbol": ["AAPL", "BAD1", None],
                "Name": ["Apple", None, "BadCorp"],
                "LastSale": [150.0, -100.0, 200.0],
                "MarketCap": [2500e9, -500e9, 300e9],
                "IPOyear": [1980, 2050, 2000]
            }
            nasdaq_df = pd.DataFrame(nasdaq_data)
            conn.register('nasdaq_data', nasdaq_df)
            conn.execute("CREATE TABLE nasdaq AS SELECT * FROM nasdaq_data")
            
            # Create valid Company History table
            company_history_data = {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": [380.0, 385.0],
                "High": [390.0, 395.0],
                "Low": [375.0, 380.0],
                "Close": [385.0, 390.0],
                "Volume": [1000000, 1100000]
            }
            company_history_df = pd.DataFrame(company_history_data)
            conn.register('company_history_data', company_history_df)
            conn.execute("CREATE TABLE company_history AS SELECT * FROM company_history_data")
            
            # Create valid Exchange table
            exchange_data = {
                "Date": ["2023-01-01", "2023-01-02"],
                "EUR": [0.92, 0.93],
                "JPY": [133.0, 134.0]
            }
            exchange_df = pd.DataFrame(exchange_data)
            conn.register('exchange_data', exchange_df)
            conn.execute("CREATE TABLE us_exchange AS SELECT * FROM exchange_data")
            
            conn.close()
            
            spark = get_spark()
            
            cleaned_data, metrics = extract_and_filter_data(db_path, spark)
            
            # Verify invalid records were removed
            assert len(cleaned_data["nasdaq"]) <= 3
            assert metrics["rows_removed"]["nasdaq"] >= 1


class TestDataValidation:
    """Tests for validated cleaned data."""
    
    @pytest.fixture
    def clean_trusted_database(self):
        """Create a clean TrustedZone database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "TrustedZone.duckdb")
            
            conn = duckdb.connect(db_path)
            
            # Create clean tables
            nasdaq_data = {
                "Symbol": ["AAPL"],
                "Name": ["Apple"],
                "LastSale": [150.0],
                "MarketCap": [2500e9],
                "IPOyear": [1980]
            }
            nasdaq_df = pd.DataFrame(nasdaq_data)
            conn.register('nasdaq_data', nasdaq_df)
            conn.execute("CREATE TABLE nasdaq AS SELECT * FROM nasdaq_data")
            
            company_history_data = {
                "Date": ["2023-01-01"],
                "Open": [380.0],
                "High": [390.0],
                "Low": [375.0],
                "Close": [385.0],
                "Volume": [1000000]
            }
            company_history_df = pd.DataFrame(company_history_data)
            conn.register('company_history_data', company_history_df)
            conn.execute("CREATE TABLE company_history AS SELECT * FROM company_history_data")
            
            exchange_data = {
                "Date": ["2023-01-01"],
                "EUR": [0.92],
                "JPY": [133.0]
            }
            exchange_df = pd.DataFrame(exchange_data)
            conn.register('exchange_data', exchange_df)
            conn.execute("CREATE TABLE us_exchange AS SELECT * FROM exchange_data")
            
            conn.close()
            
            yield db_path
    
    def test_validate_cleaned_data_passes_valid_data(self, clean_trusted_database):
        """Test that validation passes for valid data."""
        conn = duckdb.connect(clean_trusted_database)
        is_valid = validate_cleaned_data(conn)
        conn.close()
        
        assert is_valid is True
    
    def test_validate_cleaned_data_fails_invalid_nasdaq(self):
        """Test that validation fails for invalid NASDAQ data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "invalid.duckdb")
            
            conn = duckdb.connect(db_path)
            
            # Invalid NASDAQ data (null Symbol)
            nasdaq_data = {
                "Symbol": [None],
                "Name": ["BadCorp"],
                "LastSale": [100.0],
                "MarketCap": [100e9],
                "IPOyear": [2000]
            }
            nasdaq_df = pd.DataFrame(nasdaq_data)
            conn.register('nasdaq_data', nasdaq_df)
            conn.execute("CREATE TABLE nasdaq AS SELECT * FROM nasdaq_data")
            
            # Valid Company History
            company_history_data = {
                "Date": ["2023-01-01"],
                "Open": [380.0],
                "High": [390.0],
                "Low": [375.0],
                "Close": [385.0],
                "Volume": [1000000]
            }
            company_history_df = pd.DataFrame(company_history_data)
            conn.register('company_history_data', company_history_df)
            conn.execute("CREATE TABLE company_history AS SELECT * FROM company_history_data")
            
            # Valid Exchange
            exchange_data = {
                "Date": ["2023-01-01"],
                "EUR": [0.92],
                "JPY": [133.0]
            }
            exchange_df = pd.DataFrame(exchange_data)
            conn.register('exchange_data', exchange_df)
            conn.execute("CREATE TABLE us_exchange AS SELECT * FROM exchange_data")
            
            is_valid = validate_cleaned_data(conn)
            conn.close()
            
            assert is_valid is False


class TestDataWriting:
    """Tests for writing data to TrustedZone."""
    
    def test_write_to_trusted_zone_creates_tables(self):
        """Test that write_to_trusted_zone creates necessary tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trusted.duckdb")
            
            # Create test data
            nasdaq_data = {
                "Symbol": ["AAPL"],
                "Name": ["Apple"],
                "LastSale": [150.0],
                "MarketCap": [2500e9],
                "IPOyear": [1980]
            }
            nasdaq_df = pd.DataFrame(nasdaq_data)
            
            company_history_data = {
                "Date": ["2023-01-01"],
                "Open": [380.0],
                "High": [390.0],
                "Low": [375.0],
                "Close": [385.0],
                "Volume": [1000000]
            }
            company_history_df = pd.DataFrame(company_history_data)
            
            exchange_data = {
                "Date": ["2023-01-01"],
                "EUR": [0.92],
                "JPY": [133.0]
            }
            exchange_df = pd.DataFrame(exchange_data)
            
            cleaned_data = {
                "nasdaq": nasdaq_df,
                "company_history": company_history_df,
                "exchange": exchange_df
            }
            
            metrics = {
                "raw_counts": {"nasdaq": 1, "company_history": 1, "exchange": 1},
                "clean_counts": {"nasdaq": 1, "company_history": 1, "exchange": 1},
                "rows_removed": {"nasdaq": 0, "company_history": 0, "exchange": 0},
                "removal_rate": {"nasdaq": "0%", "company_history": "0%", "exchange": "0%"},
                "denial_constraints": {}
            }
            
            prepare_trusted_database(db_path)
            write_to_trusted_zone(db_path, cleaned_data, metrics)
            
            # Verify tables exist
            conn = duckdb.connect(db_path)
            tables = conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
            table_names = [table[0] for table in tables]
            
            assert "nasdaq" in table_names
            assert "company_history" in table_names
            assert "us_exchange" in table_names
            assert "data_quality_metrics" in table_names
            
            conn.close()
    
    def test_write_to_trusted_zone_preserves_data(self):
        """Test that written data can be read back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trusted.duckdb")
            
            # Create test data
            nasdaq_data = {
                "Symbol": ["AAPL", "MSFT"],
                "Name": ["Apple", "Microsoft"],
                "LastSale": [150.0, 300.0],
                "MarketCap": [2500e9, 2300e9],
                "IPOyear": [1980, 1986]
            }
            nasdaq_df = pd.DataFrame(nasdaq_data)
            
            company_history_data = {
                "Date": ["2023-01-01"],
                "Open": [380.0],
                "High": [390.0],
                "Low": [375.0],
                "Close": [385.0],
                "Volume": [1000000]
            }
            company_history_df = pd.DataFrame(company_history_data)
            
            exchange_data = {
                "Date": ["2023-01-01"],
                "EUR": [0.92],
                "JPY": [133.0]
            }
            exchange_df = pd.DataFrame(exchange_data)
            
            cleaned_data = {
                "nasdaq": nasdaq_df,
                "company_history": company_history_df,
                "exchange": exchange_df
            }
            
            metrics = {
                "raw_counts": {"nasdaq": 2, "company_history": 1, "exchange": 1},
                "clean_counts": {"nasdaq": 2, "company_history": 1, "exchange": 1},
                "rows_removed": {"nasdaq": 0, "company_history": 0, "exchange": 0},
                "removal_rate": {"nasdaq": "0%", "company_history": "0%", "exchange": "0%"},
                "denial_constraints": {}
            }
            
            prepare_trusted_database(db_path)
            write_to_trusted_zone(db_path, cleaned_data, metrics)
            
            # Read back data
            conn = duckdb.connect(db_path)
            
            nasdaq_read = conn.execute("SELECT * FROM nasdaq").fetchall()
            assert len(nasdaq_read) == 2
            assert nasdaq_read[0][0] == "AAPL"
            assert nasdaq_read[1][0] == "MSFT"
            
            company_history_read = conn.execute("SELECT * FROM company_history").fetchall()
            assert len(company_history_read) == 1
            
            exchange_read = conn.execute("SELECT * FROM us_exchange").fetchall()
            assert len(exchange_read) == 1
            
            conn.close()


class TestDatabaseVerification:
    """Tests for TrustedZone database verification."""
    
    def test_verify_trusted_zone_database_valid(self):
        """Test verification of a valid TrustedZone database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trusted.duckdb")
            
            # Create valid database
            nasdaq_data = {
                "Symbol": ["AAPL"],
                "Name": ["Apple"],
                "LastSale": [150.0],
                "MarketCap": [2500e9],
                "IPOyear": [1980]
            }
            nasdaq_df = pd.DataFrame(nasdaq_data)
            
            company_history_data = {
                "Date": ["2023-01-01"],
                "Open": [380.0],
                "High": [390.0],
                "Low": [375.0],
                "Close": [385.0],
                "Volume": [1000000]
            }
            company_history_df = pd.DataFrame(company_history_data)
            
            exchange_data = {
                "Date": ["2023-01-01"],
                "EUR": [0.92],
                "JPY": [133.0]
            }
            exchange_df = pd.DataFrame(exchange_data)
            
            cleaned_data = {
                "nasdaq": nasdaq_df,
                "company_history": company_history_df,
                "exchange": exchange_df
            }
            
            metrics = {
                "raw_counts": {"nasdaq": 1, "company_history": 1, "exchange": 1},
                "clean_counts": {"nasdaq": 1, "company_history": 1, "exchange": 1},
                "rows_removed": {"nasdaq": 0, "company_history": 0, "exchange": 0},
                "removal_rate": {"nasdaq": "0%", "company_history": "0%", "exchange": "0%"}
            }
            
            prepare_trusted_database(db_path)
            write_to_trusted_zone(db_path, cleaned_data, metrics)
            
            # Verify database
            is_valid = verify_trusted_zone_database(db_path)
            
            assert is_valid is True
    
    def test_verify_trusted_zone_database_missing_tables(self):
        """Test verification fails for incomplete database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "incomplete.duckdb")
            
            # Create incomplete database
            conn = duckdb.connect(db_path)
            conn.execute("CREATE TABLE nasdaq (symbol VARCHAR)")
            conn.close()
            
            # Verify should fail
            is_valid = verify_trusted_zone_database(db_path)
            
            assert is_valid is False
    
    def test_verify_trusted_zone_database_empty_tables(self):
        """Test verification fails when tables are empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "empty.duckdb")
            
            # Create database with empty tables
            conn = duckdb.connect(db_path)
            conn.execute("CREATE TABLE nasdaq (symbol VARCHAR)")
            conn.execute("CREATE TABLE company_history (date DATE)")
            conn.execute("CREATE TABLE us_exchange (date DATE)")
            conn.execute("CREATE TABLE data_quality_metrics (metric_name VARCHAR, metric_value VARCHAR, timestamp TIMESTAMP)")
            conn.close()
            
            # Verify should fail
            is_valid = verify_trusted_zone_database(db_path)
            
            assert is_valid is False


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline_with_test_data(self):
        """Test the complete pipeline with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create FormattedZone database
            formatted_db_path = os.path.join(tmpdir, "FormattedZone.duckdb")
            conn = duckdb.connect(formatted_db_path)
            
            # Create tables
            nasdaq_data = {
                "Symbol": ["AAPL", "MSFT", "GOOGL"],
                "Name": ["Apple", "Microsoft", "Google"],
                "LastSale": [150.0, 300.0, 2800.0],
                "MarketCap": [2500e9, 2300e9, 1900e9],
                "IPOyear": [1980, 1986, 2004]
            }
            nasdaq_df = pd.DataFrame(nasdaq_data)
            conn.register('nasdaq_data', nasdaq_df)
            conn.execute("CREATE TABLE nasdaq AS SELECT * FROM nasdaq_data")
            
            company_history_data = {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": [380.0, 385.0],
                "High": [390.0, 395.0],
                "Low": [375.0, 380.0],
                "Close": [385.0, 390.0],
                "Volume": [1000000, 1100000]
            }
            company_history_df = pd.DataFrame(company_history_data)
            conn.register('company_history_data', company_history_df)
            conn.execute("CREATE TABLE company_history AS SELECT * FROM company_history_data")
            
            exchange_data = {
                "Date": ["2023-01-01", "2023-01-02"],
                "EUR": [0.92, 0.93],
                "JPY": [133.0, 134.0]
            }
            exchange_df = pd.DataFrame(exchange_data)
            conn.register('exchange_data', exchange_df)
            conn.execute("CREATE TABLE us_exchange AS SELECT * FROM exchange_data")
            
            conn.close()
            
            # Run pipeline
            trusted_db_path = os.path.join(tmpdir, "TrustedZone.duckdb")
            
            prepare_trusted_database(trusted_db_path)
            spark = get_spark()
            
            # Extract and filter data with Spark (gets cleaned_data dict and metrics)
            cleaned_data, metrics = extract_and_filter_data(
                formatted_db_path, spark
            )
            
            # Write cleaned data to TrustedZone
            write_to_trusted_zone(
                trusted_db_path, cleaned_data, metrics
            )
            
            # Validate cleaned data using DuckDB connection
            trusted_conn = duckdb.connect(trusted_db_path)
            is_valid = validate_cleaned_data(trusted_conn)
            trusted_conn.close()
            
            db_verified = verify_trusted_zone_database(trusted_db_path)
            
            # Assert
            assert is_valid is True
            assert db_verified is True
    
    def test_pipeline_with_invalid_data(self):
        """Test pipeline handles invalid data gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create FormattedZone database with invalid data
            formatted_db_path = os.path.join(tmpdir, "FormattedZone.duckdb")
            conn = duckdb.connect(formatted_db_path)
            
            # Create tables with invalid records
            nasdaq_data = {
                "Symbol": ["AAPL", None],  # Null symbol should be filtered
                "Name": ["Apple", "Invalid"],
                "LastSale": [150.0, -10.0],  # Negative price should be filtered
                "MarketCap": [2500e9, 1000e9],
                "IPOyear": [1980, 2030]  # Year > 2026 should be filtered
            }
            nasdaq_df = pd.DataFrame(nasdaq_data)
            conn.register('nasdaq_data', nasdaq_df)
            conn.execute("CREATE TABLE nasdaq AS SELECT * FROM nasdaq_data")
            
            company_history_data = {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": [380.0, 385.0],
                "High": [390.0, 375.0],  # High < Low should be filtered
                "Low": [375.0, 380.0],
                "Close": [385.0, 390.0],
                "Volume": [1000000, -100]  # Negative volume should be filtered
            }
            company_history_df = pd.DataFrame(company_history_data)
            conn.register('company_history_data', company_history_df)
            conn.execute("CREATE TABLE company_history AS SELECT * FROM company_history_data")
            
            exchange_data = {
                "Date": ["2023-01-01", "2023-01-02"],
                "EUR": [0.92, -0.5],  # Negative EUR should be filtered
                "JPY": [133.0, 134.0]
            }
            exchange_df = pd.DataFrame(exchange_data)
            conn.register('exchange_data', exchange_df)
            conn.execute("CREATE TABLE us_exchange AS SELECT * FROM exchange_data")
            
            conn.close()
            
            # Run pipeline
            trusted_db_path = os.path.join(tmpdir, "TrustedZone.duckdb")
            
            prepare_trusted_database(trusted_db_path)
            spark = get_spark()
            
            cleaned_data, metrics = extract_and_filter_data(
                formatted_db_path, spark
            )
            
            write_to_trusted_zone(
                trusted_db_path, cleaned_data, metrics
            )
            
            # Verify invalid records were filtered out
            assert metrics["clean_counts"]["nasdaq"] < metrics["raw_counts"]["nasdaq"]
            assert metrics["clean_counts"]["company_history"] < metrics["raw_counts"]["company_history"]
            assert metrics["clean_counts"]["exchange"] < metrics["raw_counts"]["exchange"]
            
            db_verified = verify_trusted_zone_database(trusted_db_path)
            assert db_verified is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
