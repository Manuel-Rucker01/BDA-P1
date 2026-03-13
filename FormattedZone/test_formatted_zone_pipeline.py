"""
Tests for the FormattedZone data pipeline.

This module contains tests to verify that the data pipeline correctly loads
CSV datasets into DuckDB tables with the expected structure and data.
"""

import os
import pytest
import duckdb


@pytest.fixture
def db_path():
    """Provide the path to the DuckDB database."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "FormattedZone.duckdb")


@pytest.fixture
def db_connection(db_path):
    """Create a connection to the DuckDB database."""
    connection = duckdb.connect(db_path)
    yield connection
    connection.close()


class TestDatabaseConnection:
    """Tests for database connectivity."""
    
    def test_database_file_exists(self, db_path):
        """Verify that the DuckDB database file exists."""
        assert os.path.exists(db_path), "FormattedZone.duckdb database file not found"
    
    def test_database_connection(self, db_connection):
        """Verify that we can establish a connection to the database."""
        assert db_connection is not None, "Failed to establish database connection"


class TestTableCreation:
    """Tests for table creation and existence."""
    
    def test_nasdaq_table_exists(self, db_connection):
        """Verify that the nasdaq table exists."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'nasdaq'"
        ).fetchone()
        assert result[0] == 1, "nasdaq table does not exist"
    
    def test_sp500_table_exists(self, db_connection):
        """Verify that the sp500 table exists."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'sp500'"
        ).fetchone()
        assert result[0] == 1, "sp500 table does not exist"
    
    def test_us_exchange_table_exists(self, db_connection):
        """Verify that the us_exchange table exists."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'us_exchange'"
        ).fetchone()
        assert result[0] == 1, "us_exchange table does not exist"


class TestDataIntegrity:
    """Tests for data content and integrity."""
    
    def test_nasdaq_row_count(self, db_connection):
        """Verify that the nasdaq table has the expected number of rows."""
        result = db_connection.execute("SELECT COUNT(*) FROM nasdaq").fetchone()
        row_count = result[0]
        assert row_count == 3426, f"Expected 3426 rows in nasdaq, got {row_count}"
    
    def test_sp500_row_count(self, db_connection):
        """Verify that the sp500 table has the expected number of rows."""
        result = db_connection.execute("SELECT COUNT(*) FROM sp500").fetchone()
        row_count = result[0]
        assert row_count == 1255, f"Expected 1255 rows in sp500, got {row_count}"
    
    def test_us_exchange_row_count(self, db_connection):
        """Verify that the us_exchange table has the expected number of rows."""
        result = db_connection.execute("SELECT COUNT(*) FROM us_exchange").fetchone()
        row_count = result[0]
        assert row_count == 73, f"Expected 73 rows in us_exchange, got {row_count}"


class TestTableStructure:
    """Tests for table schema and structure."""
    
    def test_nasdaq_has_columns(self, db_connection):
        """Verify that the nasdaq table has columns."""
        result = db_connection.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'nasdaq'").fetchone()
        column_count = result[0]
        assert column_count > 0, "nasdaq table has no columns"
    
    def test_sp500_has_columns(self, db_connection):
        """Verify that the sp500 table has columns."""
        result = db_connection.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'sp500'").fetchone()
        column_count = result[0]
        assert column_count > 0, "sp500 table has no columns"
    
    def test_us_exchange_has_columns(self, db_connection):
        """Verify that the us_exchange table has columns."""
        result = db_connection.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'us_exchange'").fetchone()
        column_count = result[0]
        assert column_count > 0, "us_exchange table has no columns"
    
    def test_nasdaq_specific_columns(self, db_connection):
        """Verify that nasdaq table contains expected columns."""
        columns = db_connection.execute("DESCRIBE nasdaq").fetchall()
        column_names = [col[0] for col in columns]
        expected_columns = ['Symbol']
        for col in expected_columns:
            assert col in column_names, f"Expected column '{col}' not found in nasdaq table"
    
    def test_sp500_specific_columns(self, db_connection):
        """Verify that sp500 table contains expected columns."""
        columns = db_connection.execute("DESCRIBE sp500").fetchall()
        column_names = [col[0] for col in columns]
        expected_columns = ['Date', 'Open', 'High', 'Close', 'Volume']
        for col in expected_columns:
            assert col in column_names, f"Expected column '{col}' not found in sp500 table"
    
    def test_us_exchange_specific_columns(self, db_connection):
        """Verify that us_exchange table contains expected columns."""
        columns = db_connection.execute("DESCRIBE us_exchange").fetchall()
        column_names = [col[0] for col in columns]
        expected_columns = ['Date', 'EUR', 'JPY']
        for col in expected_columns:
            assert col in column_names, f"Expected columns not found in us_exchange table"


class TestDataQuality:
    """Tests for data quality and validity."""
    
    def test_nasdaq_no_null_symbols(self, db_connection):
        """Verify that nasdaq table has no null symbols."""
        result = db_connection.execute("SELECT COUNT(*) FROM nasdaq WHERE Symbol IS NULL").fetchone()
        null_count = result[0]
        assert null_count == 0, f"Found {null_count} null symbols in nasdaq table"
    
    def test_sp500_no_null_dates(self, db_connection):
        """Verify that sp500 table has no null dates."""
        result = db_connection.execute("SELECT COUNT(*) FROM sp500 WHERE Date IS NULL").fetchone()
        null_count = result[0]
        assert null_count == 0, f"Found {null_count} null dates in sp500 table"
    
    def test_us_exchange_no_null_dates(self, db_connection):
        """Verify that us_exchange table has no null dates."""
        result = db_connection.execute("SELECT COUNT(*) FROM us_exchange WHERE Date IS NULL").fetchone()
        null_count = result[0]
        assert null_count == 0, f"Found {null_count} null dates in us_exchange table"
    
    def test_sp500_positive_volumes(self, db_connection):
        """Verify that sp500 volume values are reasonable (>= 0)."""
        result = db_connection.execute("SELECT COUNT(*) FROM sp500 WHERE Volume < 0").fetchone()
        negative_count = result[0]
        assert negative_count == 0, f"Found {negative_count} negative volume values in sp500"
