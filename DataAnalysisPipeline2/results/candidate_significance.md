# Item 6 — Is the Candidate's OOS Edge Real or Noise?

Statistical-significance audit of the candidate (`candv1`) top-K book, reusing the **exact** read-only inference + backtest engine of `verify_candidate_oos.py` (which uses `candidate_eval.augment_extra_factors` for the 8 candidate price-factors and `common` for selection/sizing). No model was retrained; no search was re-run. We reproduce the single already-chosen weekly top-K (k=10, inverse-vol) configuration and interrogate its realised **per-period (weekly) return series**.

## Method

- **Active return series**: candidate − EW-B&H, and candidate − production, per weekly period.
- **Stationary (block) bootstrap** (Politis–Romano): geometric block length, mean ~6 weeks (captures weekly autocorrelation), **5000 resamples**, seed 12345. A single shared index draw per iteration drives all strategies, so the (candidate − production) difference and the IR-vs-B&H use coherent paired resamples. 95% CIs are the 2.5/97.5 percentiles of the resampled statistic.
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014): the Probabilistic Sharpe Ratio of the observed per-period Sharpe against the **expected-maximum** Sharpe from N independent trials, `SR* = sqrt(Var[SR]) * ((1-γ)·Z[1-1/N] + γ·Z[1-1/(Ne)])`, using sample skew and (non-excess) kurtosis. DSR = P(true SR > SR*). The DSR>0.95 bar is the 'survives multiple testing' threshold.

### Trials count N (documented)

N counts the DISTINCT strategy / cadence / transform / model configurations whose Sharpe/IR this project inspected while iterating toward the candidate (tallied from `results/*.md` and the `verify_*.py` backtests on this branch):

| Source | Distinct configs |
| :--- | :---: |
| Track D rebalance-horizon: {Pure top-K, Overlay 80/20} x {weekly, biweekly, monthly} | 6 |
| Track E1 scoring: {baseline, sector_neutral b0, sector_neutral b0.5, beta_adj, vol_adj} x {weekly, monthly} | 10 |
| Overlay OOS: {top-K, Overlay 90/10, 80/20, 70/30} (distinct active books) | 4 |
| Candidate-vs-prod OOS: {production, candv1} x {weekly, monthly} | 4 |
| Other verify_* live backtests inspected (regime filters, exact horizons, hmm/kalman, subsets) — coarse | 6 |
| **Total (central estimate)** | **30** |

We use **N=30** as the central estimate and report sensitivity at N=20 and N=40. Larger N raises the SR* bar (more searching ⇒ a higher Sharpe is needed to be credible).

## Post-Training OOS (2026-03-20 to 2026-05-15)

Weekly rebalances: **8** ⇒ **8** return periods. **SMALL-SAMPLE WARNING: with this few periods every CI below is wide and the DSR is fragile — treat as directional, not conclusive.**

### Point estimates and bootstrap 95% CIs

| Strategy | Metric | Point | Bootstrap mean [95% CI] | CI excludes 0? |
| :--- | :--- | :---: | :---: | :---: |
| candv1 | Cum return % | +22.508 | +22.702 [+9.212, +37.406] | YES |
| candv1 | Ann. Sharpe | +5.280 | +5.366 [+3.251, +7.964] | YES |
| candv1 | IR vs EW-B&H | -0.750 | -0.594 [-3.768, +4.564] | no |
| production | Cum return % | +18.342 | +18.668 [+1.606, +38.002] | YES |
| production | Ann. Sharpe | +3.211 | +3.239 [+0.520, +5.878] | YES |
| production | IR vs EW-B&H | -3.079 | -3.923 [-14.112, +1.309] | no |
| candv1 − production | Cum return Δ (pp) | +4.166 | +4.035 [-3.418, +11.361] | no |

### Deflated Sharpe (multiple-testing adjusted)

| N trials | per-period SR | E[max] SR* | PSR(SR>0) | DSR (SR>SR*) | Survives >0.95? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 30 | 0.732 | 0.299 | 0.997 | 0.950 | YES |
| 20 | 0.732 | 0.274 | 0.997 | 0.959 | YES |
| 40 | 0.732 | 0.316 | 0.997 | 0.944 | no |

(SR variance across trials estimated on-window = 0.02084; sample skew=0.99, kurtosis=2.57.)

### Verdict

- Candidate IR-vs-B&H 95% CI INCLUDES 0 (-0.594 [-3.768, +4.564]). Cannot reject zero excess-over-B&H at 95%.
- (Candidate − Production) cum-return 95% CI INCLUDES 0 (+4.03 [-3.42, +11.36] pp). Candidate vs production difference is within sampling noise.
- Deflated Sharpe (N=30 trials): per-period SR=0.732, E[max] benchmark SR*=0.299, PSR(>0)=0.997, **DSR=0.950**. Survives multiple-testing haircut (>0.95).

> **Window verdict: SUGGESTIVE but not conclusive** — one significance test passes, the other does not.

## Pre-Training OOS (2023-07-01 to 2025-03-01)

Weekly rebalances: **86** ⇒ **86** return periods. Long window — this is where the statistical power lives (subject to the survivorship + residual static-embedding caveat).

### Point estimates and bootstrap 95% CIs

| Strategy | Metric | Point | Bootstrap mean [95% CI] | CI excludes 0? |
| :--- | :--- | :---: | :---: | :---: |
| candv1 | Cum return % | +39.266 | +43.062 [-13.138, +125.582] | no |
| candv1 | Ann. Sharpe | +1.023 | +0.990 [-0.337, +2.234] | no |
| candv1 | IR vs EW-B&H | +0.418 | +0.405 [-0.996, +1.786] | no |
| production | Cum return % | +19.606 | +23.611 [-28.083, +106.555] | no |
| production | Ann. Sharpe | +0.592 | +0.545 [-0.931, +1.933] | no |
| production | IR vs EW-B&H | -0.305 | -0.298 [-1.442, +0.962] | no |
| candv1 − production | Cum return Δ (pp) | +19.661 | +19.450 [-13.530, +58.724] | no |

### Deflated Sharpe (multiple-testing adjusted)

| N trials | per-period SR | E[max] SR* | PSR(SR>0) | DSR (SR>SR*) | Survives >0.95? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 30 | 0.142 | 0.062 | 0.917 | 0.782 | no |
| 20 | 0.142 | 0.057 | 0.917 | 0.796 | no |
| 40 | 0.142 | 0.066 | 0.917 | 0.771 | no |

(SR variance across trials estimated on-window = 0.00090; sample skew=0.90, kurtosis=4.33.)

### Verdict

- Candidate IR-vs-B&H 95% CI INCLUDES 0 (+0.405 [-0.996, +1.786]). Cannot reject zero excess-over-B&H at 95%.
- (Candidate − Production) cum-return 95% CI INCLUDES 0 (+19.45 [-13.53, +58.72] pp). Candidate vs production difference is within sampling noise.
- Deflated Sharpe (N=30 trials): per-period SR=0.142, E[max] benchmark SR*=0.062, PSR(>0)=0.917, **DSR=0.782**. Does NOT clear the 0.95 multiple-testing bar — Sharpe is plausibly luck given the trials run.

> **Window verdict: WITHIN NOISE** — we cannot claim a real edge on this window after honest error bars and the multiple-testing haircut.

## Bottom line — what we can and cannot claim

**We CAN claim:**

- (Nothing rises to a 95%/DSR-significant positive claim on these windows.)

**We CANNOT claim:**

- Even on the long Pre-Training window we CANNOT reject zero excess-over-B&H at 95% (IR CI straddles 0).
- The candidate's Sharpe does NOT clear the deflated-Sharpe (N=30) bar — after the multiple-testing haircut, the Sharpe alone is not credible.
- The candidate-vs-production cum-return gap's 95% CI INCLUDES 0 — we cannot claim the candidate is statistically distinguishable from production.
- The clean Post-Training window (~8 weekly periods) is too short for any conclusive significance call — its CIs are very wide; it is corroborating, not decisive.

## Caveats

- The Pre-Training window has current-membership survivorship bias and residual static-embedding look-ahead (mitigated, not removed, by candv1's PIT-safe embeddings). Its statistical power comes at the cost of these biases inflating the edge.
- The Post-Training window is contemporaneous/clean but has too few periods for power.
- The block bootstrap preserves short-horizon autocorrelation but assumes the OOS return distribution is otherwise representative; structural regime breaks are not modelled.
- Deflated-Sharpe N is a documented estimate, not an exhaustive census; the N-sensitivity rows bound its effect. Trials are treated as independent (a standard simplifying assumption that, if anything, makes the haircut conservative when trials are correlated).
- Costs follow the same 5 bps proportional-turnover model as the source backtest; long-only, no leverage.
