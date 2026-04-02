#!/usr/bin/env python3
"""Check TrustedZone for missing values and data quality issues"""

import duckdb
import sys

# Connect to the database
conn = duckdb.connect('TrustedZone.duckdb')

print("="*80)
print("NULL VALUE ANALYSIS - TrustedZone Database")
print("="*80)

# NASDAQ
print("\n[NASDAQ TABLE]")
nasdaq_nulls = conn.execute("""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN Symbol IS NULL THEN 1 ELSE 0 END) as symbol_nulls,
        SUM(CASE WHEN Name IS NULL THEN 1 ELSE 0 END) as name_nulls,
        SUM(CASE WHEN LastSale IS NULL THEN 1 ELSE 0 END) as lastsale_nulls,
        SUM(CASE WHEN MarketCap IS NULL THEN 1 ELSE 0 END) as marketcap_nulls,
        SUM(CASE WHEN IPOyear IS NULL THEN 1 ELSE 0 END) as ipoyear_nulls
    FROM nasdaq
""").fetchall()
print(nasdaq_nulls[0])

# S&P 500
print("\n[S&P 500 TABLE]")
sp500_nulls = conn.execute("""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN Date IS NULL THEN 1 ELSE 0 END) as date_nulls,
        SUM(CASE WHEN Open IS NULL THEN 1 ELSE 0 END) as open_nulls,
        SUM(CASE WHEN High IS NULL THEN 1 ELSE 0 END) as high_nulls,
        SUM(CASE WHEN Low IS NULL THEN 1 ELSE 0 END) as low_nulls,
        SUM(CASE WHEN Close IS NULL THEN 1 ELSE 0 END) as close_nulls,
        SUM(CASE WHEN Volume IS NULL THEN 1 ELSE 0 END) as volume_nulls
    FROM sp500
""").fetchall()
print(sp500_nulls[0])

# US Exchange
print("\n[US_EXCHANGE TABLE]")
exchange_nulls = conn.execute("""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN Date IS NULL THEN 1 ELSE 0 END) as date_nulls,
        SUM(CASE WHEN EUR IS NULL THEN 1 ELSE 0 END) as eur_nulls,
        SUM(CASE WHEN JPY IS NULL THEN 1 ELSE 0 END) as jpy_nulls
    FROM us_exchange
""").fetchall()
print(exchange_nulls[0])

# Check for NEGATIVE values in price columns
print("\n" + "="*80)
print("NEGATIVE VALUE CHECK")
print("="*80)

print("\n[NASDAQ - Negative Prices]")
nasdaq_neg = conn.execute("""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN LastSale < 0 THEN 1 ELSE 0 END) as neg_lastsale,
        SUM(CASE WHEN MarketCap < 0 THEN 1 ELSE 0 END) as neg_marketcap
    FROM nasdaq
""").fetchall()
print(nasdaq_neg[0])

print("\n[S&P 500 - Negative Prices/Volumes]")
sp500_neg = conn.execute("""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN Open < 0 THEN 1 ELSE 0 END) as neg_open,
        SUM(CASE WHEN High < 0 THEN 1 ELSE 0 END) as neg_high,
        SUM(CASE WHEN Low < 0 THEN 1 ELSE 0 END) as neg_low,
        SUM(CASE WHEN Close < 0 THEN 1 ELSE 0 END) as neg_close,
        SUM(CASE WHEN Volume < 0 THEN 1 ELSE 0 END) as neg_volume
    FROM sp500
""").fetchall()
print(sp500_neg[0])

print("\n[US_EXCHANGE - Non-positive Rates]")
exchange_neg = conn.execute("""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN EUR <= 0 THEN 1 ELSE 0 END) as nonpos_eur,
        SUM(CASE WHEN JPY <= 0 THEN 1 ELSE 0 END) as nonpos_jpy
    FROM us_exchange
""").fetchall()
print(exchange_neg[0])

# Check for data inconsistencies
print("\n" + "="*80)
print("DATA INCONSISTENCY CHECK - S&P 500")
print("="*80)

print("\n[High < Low violations]")
high_low = conn.execute("""
    SELECT COUNT(*) as violations
    FROM sp500
    WHERE High < Low
""").fetchall()
print(f"violations: {high_low[0][0]}")

# Check a sample of problematic rows
print("\n" + "="*80)
print("SAMPLE OF DATA - First 5 rows of each table")
print("="*80)

print("\n[NASDAQ Sample]")
nasdaq_sample = conn.execute("SELECT * FROM nasdaq LIMIT 3").fetchdf()
print(nasdaq_sample.to_string())

print("\n[S&P 500 Sample]")
sp500_sample = conn.execute("SELECT * FROM sp500 LIMIT 3").fetchdf()
print(sp500_sample.to_string())

print("\n[US_EXCHANGE Sample]")
exchange_sample = conn.execute("SELECT * FROM us_exchange LIMIT 3").fetchdf()
print(exchange_sample.to_string())

conn.close()
