# Paper-model retrain log (`model_v3_altall`, signal `sn30_altall_hd`)

Quarterly retrain of the paper opt-in model with the FIXED recipe
(`v3/final_altall.py`; engineering, not an evaluation trial -- N_TRIALS and the
gate are unchanged). Each run validates the candidate against the live model
before replacing it: (1) identical schema / feature list, (2) Spearman of the
two models' scores on the latest decision date >= 0.8 (top-100 overlap
reported), (3) live == offline parity of the bot's model input on that date,
(4) informational out-of-sample IC of the live model on the newly labelled weeks.

How to run (locally; needs `$BDA_DATA_V3`, Alpaca keys and `SEC_USER_AGENT` in
`DataAnalysisPipeline2/trading_agent/.env`; ~1-2 h with the data refresh):

    python -m DataAnalysisPipeline2.v3.retrain_altall                  # refresh -> frame -> train -> validate -> promote
    python -m DataAnalysisPipeline2.v3.retrain_altall --validate-only  # dry run on the live model
    DataAnalysisPipeline2/scripts/retrain_altall.sh                    # + pytest, commit, push (scheduler entry point)

Scheduler entry (not installed automatically), e.g. cron on the first Saturday
of Jan/Apr/Jul/Oct at 06:00 when no research round is running:

    0 6 1-7 1,4,7,10 6 cd /path/to/BDA-P1 && DataAnalysisPipeline2/scripts/retrain_altall.sh >> /tmp/retrain_altall.log 2>&1

Rollback: `python -m DataAnalysisPipeline2.v3.retrain_altall --rollback`
(restores `ExploitationZone/model_v3_altall.prev.*`), or `git revert` the commit.

## 2026-11-08 15:10 -- validated (not promoted: validate-only)

- mode: validate-only; live model `v3_sn30_altall_hd_20261107` (training to 2026-09-18); candidate trained to 2026-09-18
- steps: validate 11s
- (1) schema: pass (65 features)
- (2) agreement on 2026-11-04: spearman 1.0 (min 0.8), top-100 overlap 100 over 3147 names -> pass
- (3) live==offline on 2026-11-04: 3147 names, alt/fund max |d| 5.2e-06, cells > 1e-3 4.89e-06, max |d| 0.0763, pred spearman 1.0, top-100 overlap 100 -> pass
- (4) live-model OOS IC after 2026-09-18: no newly labelled weeks

