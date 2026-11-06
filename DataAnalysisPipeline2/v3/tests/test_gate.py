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


def test_round5_preregistration():
    assert gate.N_TRIALS_R5 == 71
    assert len(gate.R5_CONFIGS) <= 8
    assert gate.R5_PRIMARY_AUM in gate.R5_AUMS
    assert {c["kind"] for c in gate.R5_CONFIGS.values()} == {"absolute", "active", "130_30"}


def test_choose_config_r5_and_gate_rows():
    perf = {"te4_top1000": {"ir_spy": 0.3, "max_dd": -0.40}, "te4_top500": {"ir_spy": 0.1, "max_dd": -0.30},
            "mv_abs_all": {"ir_spy": -0.2, "max_dd": -0.20}, "unregistered": {"ir_spy": 9, "max_dd": 0}}
    assert gate.choose_config_r5(perf) == "te4_top500"
    assert gate.choose_config_r5({"te4_top1000": {"ir_spy": 0.3, "max_dd": -0.4},
                                  "te4_top500": {"ir_spy": 0.1, "max_dd": -0.5}}) == "te4_top1000"
    rows = gate.evaluate_gate_r5({"mean_ic": 0.03, "nw_t": 2.5}, {str(y): 0.01 for y in range(2019, 2027)},
                                 {"sharpe": 1.0, "max_dd": -0.2}, 0.8, 0.95, 0.13, -0.1, (-0.8, 0.6))
    assert len(rows) == 8 and rows[-1]["id"] == "r5" and not rows[-1]["pass"]
    assert all(r["pass"] for r in rows[:-1])
    assert "N=71" in rows[5]["criterion"]
