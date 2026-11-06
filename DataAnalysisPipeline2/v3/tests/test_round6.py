"""Round-6 composite weights are PIT and shrunk; v2 fit_fn plugs into the walk-forward."""
import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import round6 as r6
from DataAnalysisPipeline2.v3 import train as tr

CAL = np.array(pd.bdate_range("2019-01-01", "2021-12-31").strftime("%Y-%m-%d"))
WEEKLY = [d for d in pd.Series(CAL).groupby(pd.to_datetime(pd.Series(CAL)).dt.to_period("W-SUN")).max()]


def test_composite_weights_use_only_completed_labels():
    ics = {5: pd.Series(0.10, index=WEEKLY), 63: pd.Series(-0.05, index=WEEKLY)}
    W = r6.composite_weights(ics, WEEKLY, CAL).set_index("Date")
    pos = {d: i for i, d in enumerate(CAL)}
    first = W.index[0]
    assert np.isclose(W.loc[first, "w5"], 0.5) and np.isclose(W.loc[first, "w63"], 0.5)   # no history -> equal
    # once >= 26 completed 5-day ICs exist: w5 = 0.5*1 + 0.5*0.5 = 0.75, w63 = 0.25 (negative IC -> 0 + shrink)
    late = W.index[-1]
    assert np.isclose(W.loc[late, "w5"], 0.75) and np.isclose(W.loc[late, "w63"], 0.25)
    # a huge IC on a date whose 63-session label is not complete by t must not move w(t)
    t = WEEKLY[60]
    spike = ics[63].copy()
    recent = [d for d in WEEKLY if pos[t] - 63 < pos[d] <= pos[t]]
    spike.loc[recent] = 5.0
    W2 = r6.composite_weights({5: ics[5], 63: spike}, [t], CAL).set_index("Date")
    W1 = r6.composite_weights(ics, [t], CAL).set_index("Date")
    pd.testing.assert_frame_equal(W1, W2)


def test_embargo_per_horizon():
    assert r6.embargo_for(5) == tr.EMBARGO_SESSIONS and r6.embargo_for(63) == 73
