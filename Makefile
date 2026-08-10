# BDA Quant — convenience commands.  Run `make` or `make help` to list targets.
# Override defaults inline, e.g.  make run-ui PORT=8010  |  make propose STRATEGY=high_confidence

PY       ?= python
PORT     ?= 8000
UNIVERSE ?= full
STRATEGY ?= top_k
TOPK     ?= 10
DASH     := DataAnalysisPipeline2.dashboard.app
AGENT    := DataAnalysisPipeline2.trading_agent.run
BT       := DataAnalysisPipeline2/scripts/backtests
HALT     := DataAnalysisPipeline2/trading_agent/agent_logs/.agent_halted

.DEFAULT_GOAL := help
.PHONY: help install run-ui ui propose run-live account backtest backtest-full test clean reset-halt

help: ## Show this help
	@echo "BDA Quant — available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Vars: PY=$(PY) PORT=$(PORT) UNIVERSE=$(UNIVERSE) STRATEGY=$(STRATEGY) TOPK=$(TOPK)"

install: ## Install Python dependencies (agent + dashboard)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r DataAnalysisPipeline2/trading_agent/requirements.txt
	$(PY) -m pip install -r DataAnalysisPipeline2/dashboard/requirements.txt

run-ui: ## Launch the web dashboard (http://127.0.0.1:$(PORT))
	DASHBOARD_PORT=$(PORT) $(PY) -m $(DASH)

ui: run-ui ## Alias for run-ui

propose: ## Dry-run the model — prints the proposed book, places NO orders
	$(PY) -m $(AGENT) --universe $(UNIVERSE) --strategy $(STRATEGY) --top-k $(TOPK) --dry-run

run-live: ## LIVE run — submits orders to the active (paper) Alpaca account
	@echo ">>> LIVE run: this SUBMITS orders to the active paper account. Ctrl-C to abort."
	$(PY) -m $(AGENT) --universe $(UNIVERSE) --strategy $(STRATEGY) --top-k $(TOPK) --live

account: ## Read-only snapshot of the active account (equity, positions, halt)
	$(PY) DataAnalysisPipeline2/scripts/account_status.py

backtest: ## Fast canonical OOS backtest (clean post-training window only)
	cd $(BT) && BACKTEST_POST_ONLY=1 $(PY) canonical_oos.py

backtest-full: ## Full canonical OOS backtest (both windows, slower)
	cd $(BT) && $(PY) canonical_oos.py

test: ## Run the unit tests (overlay, scoring, bot helpers)
	$(PY) -m pytest -q $(BT)/test_overlay.py $(BT)/test_scoring.py \
		DataAnalysisPipeline2/trading_agent/test_bot_helpers.py

reset-halt: ## Clear the drawdown circuit-breaker halt file (manual reset)
	@if [ -f $(HALT) ]; then rm -v $(HALT); else echo "no halt file — not halted"; fi

clean: ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@echo "cleaned pycache"
