# BDA Quant — convenience commands.  Run `make` or `make help` to list targets.
# Override defaults inline, e.g.  make run-ui PORT=8010  |  make account ACCOUNT=weekly

PY       ?= python
PORT     ?= 8000
ACCOUNT  ?= monthly
UNIVERSE ?= full
STRATEGY ?= top_k
TOPK     ?= 10
DASH     := DataAnalysisPipeline2.dashboard.app
AGENT    := DataAnalysisPipeline2.trading_agent.run
SCHED    := DataAnalysisPipeline2.trading_agent.scheduled_rebalance
MON      := DataAnalysisPipeline2.trading_agent.monitoring
RETRAIN  := DataAnalysisPipeline2.v3.retrain
RETRAIN_ARGS ?=
BT       := DataAnalysisPipeline2/scripts/backtests
HALT     := DataAnalysisPipeline2/trading_agent/agent_logs/accounts/$(ACCOUNT)/.agent_halted

.DEFAULT_GOAL := help
.PHONY: help install run-ui ui propose run-live account rebalance-check rebalance-start \
        rebalance monitor retrain retrain-altall backtest backtest-full test clean reset-halt

help: ## Show this help
	@echo "BDA Quant — available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Vars: ACCOUNT=$(ACCOUNT) PY=$(PY) PORT=$(PORT) UNIVERSE=$(UNIVERSE) STRATEGY=$(STRATEGY) TOPK=$(TOPK)"

install: ## Install Python dependencies (agent + dashboard)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r DataAnalysisPipeline2/trading_agent/requirements.txt
	$(PY) -m pip install -r DataAnalysisPipeline2/dashboard/requirements.txt

run-ui: ## Launch the web dashboard (http://127.0.0.1:$(PORT))
	DASHBOARD_PORT=$(PORT) $(PY) -m $(DASH)

ui: run-ui ## Alias for run-ui

account: ## Read-only snapshot of ACCOUNT (equity, positions, halt)
	ACTIVE_ACCOUNT=$(ACCOUNT) $(PY) DataAnalysisPipeline2/scripts/account_status.py

propose: ## Dry-run the model on ACCOUNT — prints the proposed book, places NO orders
	ACTIVE_ACCOUNT=$(ACCOUNT) $(PY) -m $(AGENT) --universe $(UNIVERSE) --strategy $(STRATEGY) --top-k $(TOPK) --dry-run

run-live: ## LIVE run now on ACCOUNT (paper) — submits orders, ignores the schedule
	@echo ">>> LIVE run on '$(ACCOUNT)': this SUBMITS orders to that paper account. Ctrl-C to abort."
	ACTIVE_ACCOUNT=$(ACCOUNT) $(PY) -m $(AGENT) --universe $(UNIVERSE) --strategy $(STRATEGY) --top-k $(TOPK) --live

rebalance-check: ## Scheduler preflight for ACCOUNT (read-only): would it trade now, and why?
	$(PY) -m $(SCHED) --account $(ACCOUNT)

rebalance-start: ## One-time START: rebalance ACCOUNT now even off-schedule (all safety checks still apply)
	@echo ">>> Starting '$(ACCOUNT)': this SUBMITS orders now (paper only)."
	$(PY) -m $(SCHED) --account $(ACCOUNT) --force-due --execute --notify

rebalance: ## What the scheduler runs: rebalance ACCOUNT only if due and safe
	$(PY) -m $(SCHED) --account $(ACCOUNT) --execute --notify

monitor: ## Refresh ops/monitoring.{csv,md} for ACCOUNT (realised IC, drift, alerts; read-only)
	ACTIVE_ACCOUNT=$(ACCOUNT) $(PY) -m $(MON) --account $(ACCOUNT)

retrain: ## Quarterly retrain via the v3 pipeline CLI (refresh -> train -> gate); pass RETRAIN_ARGS=...
	@if $(PY) -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('DataAnalysisPipeline2.v3') and u.find_spec('$(RETRAIN)') else 1)" 2>/dev/null; then \
		$(PY) -m $(RETRAIN) $(RETRAIN_ARGS); \
	else \
		echo "retrain: '$(RETRAIN)' is not available yet."; \
		echo "  The v3 training pipeline (WS3 of docs/IMPROVEMENT_PLAN_v3.md) provides it; until it lands,"; \
		echo "  retrain manually with the v2 notebooks/scripts and follow the 'Quarterly retrain due' issue checklist."; \
		exit 1; \
	fi

retrain-altall: ## Retrain the paper model (model_v3_altall) with the fixed recipe + validation; pass RETRAIN_ARGS=...
	$(PY) -m DataAnalysisPipeline2.v3.retrain_altall $(RETRAIN_ARGS)

backtest: ## Fast canonical OOS backtest (clean post-training window only)
	cd $(BT) && BACKTEST_POST_ONLY=1 $(PY) canonical_oos.py

backtest-full: ## Full canonical OOS backtest (both windows, slower)
	cd $(BT) && $(PY) canonical_oos.py

test: ## Run all unit tests (agent safety, scheduler, overlay, scoring, helpers)
	$(PY) -m pytest -q DataAnalysisPipeline2/trading_agent/ DataAnalysisPipeline2/features/ $(BT)/test_overlay.py $(BT)/test_scoring.py

reset-halt: ## Clear ACCOUNT's local drawdown halt file (manual reset after review)
	@if [ -f $(HALT) ]; then rm -v $(HALT); else echo "no halt file for '$(ACCOUNT)' — not halted"; fi

clean: ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@echo "cleaned pycache"
