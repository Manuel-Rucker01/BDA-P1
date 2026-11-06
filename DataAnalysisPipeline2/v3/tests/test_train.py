"""Walk-forward folds never train on test-period or embargoed dates."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import dataset as ds
from DataAnalysisPipeline2.v3 import train as tr

CAL = np.array(pd.bdate_range("2017-01-02", "2021-12-31").strftime("%Y-%m-%d"))
WEEKLY = ds.weekly_decision_dates(CAL, start="2017-01-01")
FEATS = ["f0", "f1", "f2"]


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in WEEKLY:
        x = rng.normal(size=(n, 3))
        y = x[:, 0] * 0.3 + rng.normal(size=n)
        rows.append(pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "Date": d,
                                  "f0": x[:, 0], "f1": x[:, 1], "f2": x[:, 2],
                                  "target_raw_30": pd.Series(y).rank(pct=True).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def test_train_window_respects_embargo():
    pos = {d: i for i, d in enumerate(CAL)}
    for y in (2019, 2020, 2021):
        start = f"{y}-01-01"
        k0 = int(np.searchsorted(CAL, start))
        dates = tr.train_window_dates(WEEKLY, CAL, start)
        assert max(dates) < start
        assert all(pos[d] + 30 + 10 < k0 for d in dates)           # label end + 10 sessions
        nxt = [d for d in WEEKLY if d not in dates and d < start]
        assert all(pos[d] >= k0 - tr.EMBARGO_SESSIONS for d in nxt)  # only embargoed dates left out


def test_fit_val_split_is_embargoed():
    dates = tr.train_window_dates(WEEKLY, CAL, "2020-01-01")
    fit, val = tr.split_fit_val(dates, CAL)
    pos = {d: i for i, d in enumerate(CAL)}
    assert max(fit) < min(val)
    assert pos[max(fit)] < pos[min(val)] - tr.EMBARGO_SESSIONS
    assert set(fit) | set(val) <= set(dates)


def test_walk_forward_never_sees_test_dates(monkeypatch):
    seen = []

    class Dummy:
        def predict(self, X):
            return np.asarray(X)[:, 0]

    def fake_fit(frame, feats, target, calendar, models=None, refit=True, log=print):
        seen.append(sorted(frame["Date"].unique()))
        return {"m": Dummy()}, {"best_iter": {}}

    monkeypatch.setattr(tr, "fit_ensemble", fake_fit)
    f = _frame()
    preds, folds = tr.walk_forward(f, FEATS, "target_raw_30", CAL, years=(2019, 2020, 2021), log=lambda *a: None)
    pos = {d: i for i, d in enumerate(CAL)}
    for y, dates in zip((2019, 2020, 2021), seen):
        k0 = int(np.searchsorted(CAL, f"{y}-01-01"))
        assert max(pos[d] for d in dates) < k0 - tr.EMBARGO_SESSIONS
        test = preds[preds["fold"] == y]
        assert test["Date"].str[:4].eq(str(y)).all() and not set(dates) & set(test["Date"])
    assert set(preds["Date"]) == {d for d in WEEKLY if "2019" <= d[:4] <= "2021"}


def test_real_ensemble_learns_signal():
    f = _frame(n=60)
    dates = tr.train_window_dates(f["Date"].unique(), CAL, "2021-01-01")
    trn = f[f["Date"].isin(set(dates))]
    ms, info = tr.fit_ensemble(trn, FEATS, "target_raw_30", CAL, log=lambda *a: None)
    assert set(ms) == set(tr.MODEL_NAMES)
    te = f[f["Date"] >= "2021-01-01"]
    p = tr.predict_ensemble(ms, te, FEATS)
    assert p.min() > 0 and p.max() <= 1
    ic = pd.DataFrame({"p": p, "x": te["f0"].to_numpy(), "d": te["Date"].to_numpy()}) \
        .groupby("d").apply(lambda g: g["p"].corr(g["x"], method="spearman"), include_groups=False)
    assert ic.mean() > 0.3
    assert info["fit_range"][1] < info["val_range"][0]
