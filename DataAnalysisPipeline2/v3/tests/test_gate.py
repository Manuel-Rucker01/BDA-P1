"""Pre-registered selection rules and gate bookkeeping."""
from DataAnalysisPipeline2.v3 import gate


def test_choose_variant_requires_gain_and_t():
    base = {"mean_ic": 0.02, "nw_t": 3.0}
    assert gate.choose_variant({"base": base, "h21": {"mean_ic": 0.024, "nw_t": 4}}) == "base"
    assert gate.choose_variant({"base": base, "h21": {"mean_ic": 0.03, "nw_t": 2.9}}) == "base"
    assert gate.choose_variant({"base": base, "h21": {"mean_ic": 0.03, "nw_t": 3.5},
                                "sn30": {"mean_ic": 0.04, "nw_t": 3.2}}) == "sn30"
    assert gate.choose_variant({"base": base, "common_kg": {"mean_ic": 0.2, "nw_t": 9}}) == "base"


def test_choose_construction_prefers_drawdown_ok():
    perf = {"lo30": {"sharpe": 1.2, "max_dd": -0.40}, "lo50": {"sharpe": 1.0, "max_dd": -0.30},
            "ls40": {"sharpe": 0.8, "max_dd": -0.10}, "ew_universe": {"sharpe": 5, "max_dd": 0}}
    assert gate.choose_construction(perf) == "lo50"
    assert gate.choose_construction({"lo30": {"sharpe": 1.2, "max_dd": -0.5},
                                     "lo50": {"sharpe": 1.0, "max_dd": -0.6}}) == "lo30"


def test_evaluate_gate_all_or_nothing():
    rows = gate.evaluate_gate({"mean_ic": 0.03, "nw_t": 2.5}, {str(y): 0.01 for y in range(2019, 2027)},
                              {"sharpe": 1.0, "max_dd": -0.2}, 0.8, 0.95, 0.13)
    assert all(r["pass"] for r in rows) and len(rows) == 7
    rows = gate.evaluate_gate({"mean_ic": 0.03, "nw_t": float("nan")}, {}, {"sharpe": 1.0, "max_dd": -0.2},
                              0.8, 0.95, 0.13)
    assert not rows[1]["pass"] and not rows[2]["pass"]
