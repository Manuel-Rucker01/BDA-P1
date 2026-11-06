"""v3 data layer (WS1 of docs/IMPROVEMENT_PLAN_v3.md).

* ``universe``  -- survivorship-free, point-in-time US common-stock universe
  (Alpaca assets + corporate actions + brute-force symbol discovery),
  symbol-reuse detection and the PIT ``eligible`` rule.
* ``prices``    -- 2016 -> last close daily SIP bars, Parquet store partitioned
  by year, resumable bulk download, incremental ``update`` and ``load_prices``.
* ``quality``   -- data-quality checks and ``prices/quality_report.md``.

Data lives outside git under ``$BDA_DATA_V3`` (see ``_alpaca.data_dir``).
"""
