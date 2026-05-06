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
    
    def test_company_history_table_exists(self, db_connection):
        """Verify that the company_history table exists."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'company_history'"
        ).fetchone()
        assert result[0] == 1, "company_history table does not exist"
    
    def test_us_exchange_table_exists(self, db_connection):
        """Verify that the us_exchange table exists."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'us_exchange'"
        ).fetchone()
        assert result[0] == 1, "us_exchange table does not exist"
    
    def test_sp500_companies_table_exists(self, db_connection):
        """Verify that the sp500_companies table exists."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'sp500_companies'"
        ).fetchone()
        assert result[0] == 1, "sp500_companies table does not exist"
    
    def test_forbes_employers_table_exists(self, db_connection):
        """Verify that the forbes_employers table exists."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'forbes_employers'"
        ).fetchone()
        assert result[0] == 1, "forbes_employers table does not exist"


class TestDataIntegrity:
    """Tests for data content and integrity."""
    
    def test_nasdaq_row_count(self, db_connection):
        """Verify that the nasdaq table has the expected number of rows."""
        result = db_connection.execute("SELECT COUNT(*) FROM nasdaq").fetchone()
        row_count = result[0]
        assert row_count == 3426, f"Expected 3426 rows in nasdaq, got {row_count}"
    
    def test_company_history_row_count(self, db_connection):
        """Verify that the company_history table has the expected number of rows."""
        result = db_connection.execute("SELECT COUNT(*) FROM company_history").fetchone()
        row_count = result[0]
        assert row_count == 481133, f"Expected 481133 rows in company_history, got {row_count}"
    
    def test_us_exchange_row_count(self, db_connection):
        """Verify that the us_exchange table has the expected number of rows."""
        result = db_connection.execute("SELECT COUNT(*) FROM us_exchange").fetchone()
        row_count = result[0]
        assert row_count == 71, f"Expected 71 rows in us_exchange, got {row_count}"
    
    def test_sp500_companies_row_count(self, db_connection):
        """Verify that the sp500_companies table has the expected number of rows."""
        result = db_connection.execute("SELECT COUNT(*) FROM sp500_companies").fetchone()
        row_count = result[0]
        assert row_count == 98, f"Expected 98 rows in sp500_companies, got {row_count}"
    
    def test_forbes_employers_row_count(self, db_connection):
        """Verify that the forbes_employers table has the expected number of rows."""
        result = db_connection.execute("SELECT COUNT(*) FROM forbes_employers").fetchone()
        row_count = result[0]
        assert row_count == 700, f"Expected 700 rows in forbes_employers, got {row_count}"


class TestTableStructure:
    """Tests for table schema and structure."""
    
    def test_nasdaq_has_columns(self, db_connection):
        """Verify that the nasdaq table has columns."""
        result = db_connection.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'nasdaq'").fetchone()
        column_count = result[0]
        assert column_count > 0, "nasdaq table has no columns"
    
    def test_company_history_has_columns(self, db_connection):
        """Verify that the company_history table has columns."""
        result = db_connection.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'company_history'").fetchone()
        column_count = result[0]
        assert column_count > 0, "company_history table has no columns"
    
    def test_us_exchange_has_columns(self, db_connection):
        """Verify that the us_exchange table has columns."""
        result = db_connection.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'us_exchange'").fetchone()
        column_count = result[0]
        assert column_count > 0, "us_exchange table has no columns"
    
    def test_sp500_companies_has_columns(self, db_connection):
        """Verify that the sp500_companies table has columns."""
        result = db_connection.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'sp500_companies'").fetchone()
        column_count = result[0]
        assert column_count > 0, "sp500_companies table has no columns"
    
    def test_forbes_employers_has_columns(self, db_connection):
        """Verify that the forbes_employers table has columns."""
        result = db_connection.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'forbes_employers'").fetchone()
        column_count = result[0]
        assert column_count > 0, "forbes_employers table has no columns"
    
    def test_nasdaq_specific_columns(self, db_connection):
        """Verify that nasdaq table contains expected columns."""
        columns = db_connection.execute("DESCRIBE nasdaq").fetchall()
        column_names = [col[0] for col in columns]
        expected_columns = ['Symbol']
        for col in expected_columns:
            assert col in column_names, f"Expected column '{col}' not found in nasdaq table"
    
    def test_company_history_specific_columns(self, db_connection):
        """Verify that company_history table contains expected columns."""
        columns = db_connection.execute("DESCRIBE company_history").fetchall()
        column_names = [col[0] for col in columns]
        expected_columns = ['Date', 'Open', 'High', 'Close', 'Volume']
        for col in expected_columns:
            assert col in column_names, f"Expected column '{col}' not found in company_history table"
    
    def test_us_exchange_specific_columns(self, db_connection):
        """Verify that us_exchange table contains expected columns."""
        columns = db_connection.execute("DESCRIBE us_exchange").fetchall()
        column_names = [col[0] for col in columns]
        expected_columns = ['Date', 'EUR', 'JPY']
        for col in expected_columns:
            assert col in column_names, f"Expected columns not found in us_exchange table"
    
    def test_sp500_companies_specific_columns(self, db_connection):
        """Verify that sp500_companies table contains expected columns."""
        columns = db_connection.execute("DESCRIBE sp500_companies").fetchall()
        column_names = [col[0] for col in columns]
        expected_columns = ['Ticker', 'Name', 'Sector', 'MarketCap', 'Employees']
        for col in expected_columns:
            assert col in column_names, f"Expected column '{col}' not found in sp500_companies table"
    
    def test_forbes_employers_specific_columns(self, db_connection):
        """Verify that forbes_employers table contains expected columns."""
        columns = db_connection.execute("DESCRIBE forbes_employers").fetchall()
        column_names = [col[0] for col in columns]
        expected_columns = ['rank', 'company', 'employees', 'publish_year']
        for col in expected_columns:
            assert col in column_names, f"Expected column '{col}' not found in forbes_employers table"


class TestDataQuality:
    """Tests for data quality and validity."""
    
    def test_nasdaq_no_null_symbols(self, db_connection):
        """Verify that nasdaq table has no null symbols."""
        result = db_connection.execute("SELECT COUNT(*) FROM nasdaq WHERE Symbol IS NULL").fetchone()
        null_count = result[0]
        assert null_count == 0, f"Found {null_count} null symbols in nasdaq table"
    
    def test_company_history_no_null_dates(self, db_connection):
        """Verify that company_history table has no null dates."""
        result = db_connection.execute("SELECT COUNT(*) FROM company_history WHERE Date IS NULL").fetchone()
        null_count = result[0]
        assert null_count == 0, f"Found {null_count} null dates in company_history table"

    def test_us_exchange_no_null_dates(self, db_connection):
        """Verify that us_exchange table has no null dates."""
        result = db_connection.execute("SELECT COUNT(*) FROM us_exchange WHERE Date IS NULL").fetchone()
        null_count = result[0]
        assert null_count == 0, f"Found {null_count} null dates in us_exchange table"
    
    def test_company_history_positive_volumes(self, db_connection):
        """Verify that company_history volume values are reasonable (>= 0)."""
        result = db_connection.execute("SELECT COUNT(*) FROM company_history WHERE Volume < 0").fetchone()
        negative_count = result[0]
        assert negative_count == 0, f"Found {negative_count} negative volume values in company_history"
    
    def test_sp500_companies_no_null_tickers(self, db_connection):
        """Verify that sp500_companies table has no null tickers."""
        result = db_connection.execute("SELECT COUNT(*) FROM sp500_companies WHERE Ticker IS NULL").fetchone()
        null_count = result[0]
        assert null_count == 0, f"Found {null_count} null tickers in sp500_companies table"
    
    def test_forbes_employers_no_null_companies(self, db_connection):
        """Verify that forbes_employers table has no null company names."""
        result = db_connection.execute("SELECT COUNT(*) FROM forbes_employers WHERE company IS NULL").fetchone()
        null_count = result[0]
        assert null_count == 0, f"Found {null_count} null company names in forbes_employers table"

