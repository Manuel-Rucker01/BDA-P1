"""Cloud snapshot refresh (trading_agent/snapshot_refresh.py): offline tests on copies of
the committed snapshots with fake fetchers, no network."""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.trading_agent import altall, earnings, optimized
from DataAnalysisPipeline2.trading_agent import snapshot_refresh as sr
from DataAnalysisPipeline2.v3 import round11 as r11

TODAY, END = "2026-11-07", "2026-11-06"


@pytest.fixture
def dirs(tmp_path):
    alt, fund = tmp_path / "alt", tmp_path / "fund"
    shutil.copytree(altall.SNAPSHOT_DIR, alt)
    shutil.copytree(optimized.SNAPSHOT_DIR, fund)
    return str(alt), str(fund)


def _files(d):
    return {f: open(os.path.join(d, f), "rb").read() for f in sorted(os.listdir(d))}


# --------------------------------------------------------------------------- #
# alt data
# --------------------------------------------------------------------------- #
def _stale_alt(alt, fund, as_of="2026-10-30"):
    """Rewrite the copied snapshot as if it had been built on ``as_of``."""
    src = altall.read_snapshot(alt)
    ev = src["events"]
    old = ev[pd.to_datetime(ev["filing_date"]) <= as_of]
    ciks = sr.universe_ciks(fund) | set(ev["issuer_cik"].dropna().astype("int64"))
    altall.build_snapshot(alt, events=old, si=src["si"], insider_as_of=as_of, ciks=ciks, today=as_of)
    return ev, old


def test_altdata_topup_reproduces_the_full_snapshot(dirs):
    """Stale snapshot + the missing days (what the EDGAR feed returns) == the snapshot
    built directly from the full event table."""
    alt, fund = dirs
    full_src = altall.read_snapshot(alt)
    full, old = _stale_alt(alt, fund)
    missing = full[~full["accession"].isin(set(old["accession"]))]
    calls = {}

    def fake_insider(src, end, ciks, budget_s, log=print):
        calls["start"], calls["end"] = src["insider_as_of"], pd.Timestamp(end)
        calls["has_universe"] = sr.universe_ciks(fund) <= ciks
        return {**src, "events": pd.concat([src["events"], missing], ignore_index=True),
                "insider_as_of": pd.Timestamp(full_src["insider_as_of"]), "source": "snapshot+edgar_daily"}

    info = sr.refresh_altdata(END, out=alt, fund_dir=fund, today=TODAY, refresh_insider=fake_insider,
                              refresh_si=lambda si, d, log=print: si)
    assert calls["start"] == pd.Timestamp("2026-10-30") and calls["end"] == pd.Timestamp(END)
    assert calls["has_universe"]
    assert info["insider_as_of"] == str(full_src["insider_as_of"].date()) and info["new_events"] == len(missing)
    got = altall.read_snapshot(alt)
    key = ["avail_date", "accession", "code", "shares", "price"]
    pd.testing.assert_frame_equal(got["events"].sort_values(key).reset_index(drop=True),
                                  full_src["events"].sort_values(key).reset_index(drop=True))
    assert "incremental" in got["manifest"]["source"] and got["manifest"]["refreshed_from"] == "2026-10-30"
    altall.check_fresh(got, TODAY, 14)


def _si_with(settlement):
    """Fake FINRA top-up adding one file settled on ``settlement``."""
    def f(si, d, log=print):
        extra = si.tail(1).assign(settlement_date=pd.Timestamp(settlement))
        return pd.concat([si, extra], ignore_index=True)
    return f


def test_altdata_feed_not_reaching_end_fails_and_keeps_files(dirs):
    alt, fund = dirs
    before = _files(alt)
    with pytest.raises(sr.RefreshFailed, match="Form 4 feed only reached"):
        sr.refresh_altdata("2026-11-20", out=alt, fund_dir=fund, today="2026-11-21",
                           refresh_insider=lambda src, end, ciks, b, log=print: src,      # nothing new
                           refresh_si=_si_with("2026-11-14"))
    assert _files(alt) == before


def test_altdata_missing_finra_file_fails(dirs):
    alt, fund = dirs
    before = _files(alt)
    # by 2026-11-30 the 2026-11-14 settlement is public; the snapshot ends 2026-10-15 and no download happens
    with pytest.raises(sr.RefreshFailed, match="FINRA"):
        sr.refresh_altdata("2026-11-27", out=alt, fund_dir=fund, today="2026-11-30",
                           refresh_insider=lambda *a, **k: pytest.fail("insider refresh must not run"),
                           refresh_si=lambda si, d, log=print: si)
    assert _files(alt) == before


# --------------------------------------------------------------------------- #
# earnings
# --------------------------------------------------------------------------- #
def test_kept_to_raw_round_trips_the_new_york_date(dirs):
    alt, _ = dirs
    kept = earnings.read_snapshot(alt)["kept"]
    back = r11.retained_events(sr.kept_to_raw(kept))
    a = kept.sort_values(["cik", "adate"]).reset_index(drop=True)
    b = back.sort_values(["cik", "adate"]).reset_index(drop=True)
    assert (a["cik"].to_numpy() == b["cik"].to_numpy()).all() and (a["adate"] == b["adate"]).all()


def _synthetic_raw(seed=0, n_cik=40):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(1, n_cik + 1):
        d = pd.Timestamp("2025-01-10") + pd.Timedelta(days=int(rng.integers(0, 40)))
        while d < pd.Timestamp("2026-12-31"):
            rows.append({"cik": 1000 + c, "acceptance": (d + pd.Timedelta(hours=int(rng.integers(6, 22))))
                         .tz_localize("America/New_York").tz_convert("UTC"), "src10q": 0})
            d += pd.Timedelta(days=int(rng.choice([20, 45, 91, 92, 95])))
    return pd.DataFrame(rows)


def test_earnings_topup_equals_full_rebuild(dirs, monkeypatch):
    alt, fund = dirs
    raw = _synthetic_raw()
    ny = pd.to_datetime(raw["acceptance"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    as_of, end = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-11-06")
    earnings.build_snapshot(alt, raw=raw[ny <= as_of], as_of=as_of, today=as_of)
    new = raw[(ny > as_of) & (ny <= end)]
    seen = {}

    def fake_fetch(start, stop, ciks, client=None, log=print):
        seen["start"], seen["ciks"] = pd.Timestamp(start), ciks
        f = pd.DataFrame({"cik": new["cik"].to_numpy(), "accn": [f"a{i}" for i in range(len(new))], "form": "8-K",
                          "filing_date": ny[new.index].dt.strftime("%Y-%m-%d").to_numpy(),
                          "acceptance": new["acceptance"].to_numpy(), "items": "2.02,9.01"})
        noise = f.head(3).assign(items="5.02", accn=["x1", "x2", "x3"])         # non-earnings 8-Ks are ignored
        return pd.concat([f, noise], ignore_index=True), stop, {"days": 1}

    info = sr.refresh_earnings(end, out=alt, fund_dir=fund, today="2026-11-07", fetch=fake_fetch)
    assert seen["start"] == as_of + pd.Timedelta(days=1) and 1001 in seen["ciks"]
    assert info["as_of"] == "2026-11-06" and info["new_202_events"] == len(new)
    got = earnings.read_snapshot(alt)["kept"].sort_values(["cik", "adate"]).reset_index(drop=True)
    ref = earnings.compact(raw[ny <= end]).sort_values(["cik", "adate"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, ref, check_dtype=False)
    # and the blackout inputs are identical
    pd.testing.assert_series_equal(r11.expected_dates(got, "2026-11-06"), r11.expected_dates(ref, "2026-11-06"))


def test_earnings_nothing_covered_fails(dirs):
    alt, fund = dirs
    before = _files(alt)
    with pytest.raises(sr.RefreshFailed):
        sr.refresh_earnings("2026-11-20", out=alt, fund_dir=fund,
                            fetch=lambda *a, **k: (pd.DataFrame(), None, {"days": 0}))
    assert _files(alt) == before


def test_earnings_up_to_date_is_a_no_op(dirs):
    alt, fund = dirs
    before = _files(alt)
    info = sr.refresh_earnings("2026-11-06", out=alt, fund_dir=fund,
                               fetch=lambda *a, **k: pytest.fail("no fetch when up to date"))
    assert info["up_to_date"] and _files(alt) == before


# --------------------------------------------------------------------------- #
# fundamentals
# --------------------------------------------------------------------------- #
def test_fundamentals_rebuild_is_staged_and_trimmed(dirs, monkeypatch):
    """A rebuild from the committed rows (as if companyfacts had not changed) gives
    back the committed snapshot; an older companyfacts is refused without writing."""
    _, fund = dirs
    committed = pd.read_parquet(os.path.join(fund, "fundamentals_asof.parquet"))
    monkeypatch.setattr(sr, "build_fund_asof", lambda zp, ciks, log=print: committed.copy())
    info = sr.refresh_fundamentals(out=fund, today="2026-11-07", zip_path="unused.zip")
    assert info["as_of"] == "2026-11-05"
    got = pd.read_parquet(os.path.join(fund, "fundamentals_asof.parquet"))
    pd.testing.assert_frame_equal(got, committed, check_categorical=False)
    man = json.load(open(os.path.join(fund, "manifest.json")))
    assert "companyfacts" in man["source"] and man["window_start"] == "2023-11-07"

    before = _files(fund)
    older = committed[committed["last_filed"] < "2026-10-01"]
    monkeypatch.setattr(sr, "build_fund_asof", lambda zp, ciks, log=print: older.copy())
    with pytest.raises(sr.RefreshFailed, match="older"):
        sr.refresh_fundamentals(out=fund, today="2026-11-07", zip_path="unused.zip")
    assert _files(fund) == before


def test_build_fund_asof_skips_missing_and_malformed(tmp_path, monkeypatch):
    import zipfile

    from DataAnalysisPipeline2.v3 import fundamentals as fu
    zp = tmp_path / "cf.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("CIK0000000001.json", json.dumps({"cik": 1, "facts": {}}))
        z.writestr("CIK0000000002.json", "not json")
    row = pd.DataFrame({"cik": [1], "asof_date": [pd.Timestamp("2026-01-02")], "last_filed": [pd.Timestamp("2026-01-01")]})
    monkeypatch.setattr(fu, "build_company_asof", lambda cf, c: row if c == 1 else pd.DataFrame())
    out = sr.build_fund_asof(str(zp), {1, 2, 3}, log=lambda *_: None)
    assert out["cik"].tolist() == [1]


# --------------------------------------------------------------------------- #
# verify / CLI
# --------------------------------------------------------------------------- #
def test_verify_uses_the_bot_rules(dirs):
    alt, fund = dirs
    ok = sr.verify("2026-11-07", alt_dir=alt, fund_dir=fund)
    assert ok["problems"] == [] and ok["altdata"]["insider_age_days"] == 1
    late = sr.verify("2026-12-01", alt_dir=alt, fund_dir=fund)
    assert any(p.startswith("altdata") for p in late["problems"])
    assert any(p.startswith("earnings") for p in late["problems"])


def test_run_isolates_part_failures(monkeypatch):
    monkeypatch.setattr(sr, "refresh_altdata", lambda *a, **k: (_ for _ in ()).throw(sr.RefreshFailed("boom")))
    monkeypatch.setattr(sr, "refresh_earnings", lambda *a, **k: {"as_of": "x"})
    monkeypatch.setattr(sr, "verify", lambda today: {"problems": []})
    res = sr.run(["altdata", "earnings"], end="2026-11-06", today="2026-11-07", log=lambda *_: None)
    assert res["failed"] == ["altdata"] and res["parts"]["earnings"]["ok"]
    assert "boom" in res["parts"]["altdata"]["error"]


def test_main_exits_2_without_user_agent(monkeypatch, capsys):
    from DataAnalysisPipeline2.v3 import edgar

    def missing():
        raise edgar.SECUserAgentMissing("SEC_USER_AGENT is not set.")
    monkeypatch.setattr(edgar, "sec_user_agent", missing)
    monkeypatch.setattr(sr, "run", lambda *a, **k: pytest.fail("must not run without a User-Agent"))
    assert sr.main(["--parts", "altdata"]) == 2
    assert "SEC_USER_AGENT" in capsys.readouterr().out
