"""Regenerate the Walk-Forward Bake-Off section of report.tex from the
authoritative CSVs produced by kg_embeddings_classifier.py.

This replaces antigravity's hand-written tables, which contained:
  - unescaped underscores in model names
  - bare % characters that LaTeX interprets as comments
  - a 100x scaling bug on the "Combined - Embedding" decile-spread block

The replacement is keyed off the section header and runs through the
"Operational Quantitative Infrastructure" anchor, so the rest of the paper
is left untouched.

Run AFTER the bake-off finishes (so the CSVs reflect the latest run).
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
EZ = os.path.join(ROOT, "ExploitationZone")
REPORT = os.path.join(ROOT, "report.tex")

PER_FOLD = os.path.join(EZ, "per_fold_results.csv")
SIGN_TESTS = os.path.join(EZ, "sign_tests.csv")
PAIRWISE = os.path.join(EZ, "pairwise_comparisons.csv")
OOF_CORR = os.path.join(EZ, "oof_correlation_matrix.csv")

FEATURE_ORDER = ["tabular_only", "embedding_only", "tabular+embedding"]
FEATURE_LABEL = {
    "tabular_only": "tabular only",
    "embedding_only": "embedding only",
    "tabular+embedding": "tabular + embedding",
}
MODEL_ORDER = [
    "SoftVote", "SoftVote_Rank", "SoftVote_Diverse",
    "LightGBM", "LightGBM_Rank",
    "XGBoost", "XGBoost_Rank",
    "CatBoost", "CatBoost_Rank",
    "RandomForest", "Stack", "MLP", "Ridge",
]


def latex_escape(s: str) -> str:
    """Conservatively escape strings for LaTeX text-mode cells."""
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def signif_bold(value: str, mark: bool) -> str:
    return f"\\textbf{{{value}}}" if mark else value


# ── Load CSVs ────────────────────────────────────────────────────────────────

fold = pd.read_csv(PER_FOLD)
sign = pd.read_csv(SIGN_TESTS)
pair = pd.read_csv(PAIRWISE)
corr = pd.read_csv(OOF_CORR, index_col=0)

# ── Summary: mean + SE per (feature_set, model) ─────────────────────────────

records = []
for (feat, model), grp in fold.groupby(["feature_set", "model"]):
    K = len(grp)
    ic_mean = grp["ic"].mean()
    ic_se = grp["ic"].std(ddof=1) / np.sqrt(K) if K > 1 else float("nan")
    ds_mean = grp["decile_spread_pct"].mean()
    ds_se = grp["decile_spread_pct"].std(ddof=1) / np.sqrt(K) if K > 1 else float("nan")
    s_row = sign[(sign.feature_set == feat) & (sign.model == model)]
    n_pos_ic = int(s_row["n_pos_ic"].values[0]) if not s_row.empty else 0
    n_pos_ds = int(s_row["n_pos_decile_spread"].values[0]) if not s_row.empty else 0
    records.append(dict(
        feature_set=feat, model=model, K=K,
        ic_mean=ic_mean, ic_se=ic_se,
        ds_mean=ds_mean, ds_se=ds_se,
        n_pos_ic=n_pos_ic, n_pos_ds=n_pos_ds,
    ))
summary = pd.DataFrame(records)

# ── Table I: walk-forward bake-off ───────────────────────────────────────────

rows = []
for feat in FEATURE_ORDER:
    feat_rows = summary[summary.feature_set == feat].copy()
    feat_rows["mo"] = pd.Categorical(feat_rows["model"], categories=MODEL_ORDER, ordered=True)
    feat_rows = feat_rows.sort_values("mo")
    best_ic_model = feat_rows.sort_values("ic_mean", ascending=False).iloc[0]["model"]
    best_ds_model = feat_rows.sort_values("ds_mean", ascending=False).iloc[0]["model"]
    first = True
    for _, r in feat_rows.iterrows():
        ic_cell = f"{r['ic_mean']:+.4f} ({r['ic_se']:.4f})"
        ds_cell = f"{r['ds_mean']:+.2f}\\% ({r['ds_se']:.2f}\\%)"
        ic_cell = signif_bold(ic_cell, r["model"] == best_ic_model)
        ds_cell = signif_bold(ds_cell, r["model"] == best_ds_model)
        feat_cell = latex_escape(FEATURE_LABEL[feat]) if first else ""
        first = False
        rows.append(
            f"{feat_cell:<22} & {latex_escape(r['model']):<18} & "
            f"{ic_cell:<32} & {int(r['n_pos_ic'])}/{int(r['K'])} & "
            f"{ds_cell:<32} & {int(r['n_pos_ds'])}/{int(r['K'])} \\\\"
        )
    rows.append("\\midrule")
if rows and rows[-1] == "\\midrule":
    rows.pop()

table1 = (
    "\\begin{table*}[t]\n"
    "\\centering\n"
    "\\caption{Walk-Forward Bake-Off summary. Mean and standard error "
    "($\\text{SE}=\\sigma/\\sqrt{K}$) across $K=5$ expanding folds, with a "
    "30-day embargo between train and test. The IC sign and DS sign columns "
    "report how many of the $K$ folds had a positive metric (a one-sided "
    "binomial sign test against $\\text{H}_0=0.5$).}\n"
    "\\label{tab:walk_forward_full}\n"
    "\\small\n"
    "\\begin{tabular}{llcccc}\n"
    "\\toprule\n"
    "\\textbf{Feature Set} & \\textbf{Model} & "
    "\\textbf{Spearman IC mean (SE)} & \\textbf{IC sign} & "
    "\\textbf{Decile Spread mean (SE)} & \\textbf{DS sign} \\\\\n"
    "\\midrule\n"
    + "\n".join(rows) + "\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table*}\n"
)

# ── Table II: pairwise comparisons ───────────────────────────────────────────

# Build clean labels for comparison rows.
COMP_LABEL = {
    "embedding_only - tabular_only": "Embedding $-$ Tabular",
    "tabular+embedding - tabular_only": "Combined $-$ Tabular",
    "tabular+embedding - embedding_only": "Combined $-$ Embedding",
}

pair["mo"] = pd.Categorical(pair["model"], categories=MODEL_ORDER, ordered=True)
pair = pair.sort_values(["metric", "comparison", "mo"]).reset_index(drop=True)

# Significance: BOTH CI excludes zero AND sign-test p < 0.20 (with K=5 the
# minimum two-sided p is 0.0625 at 5/5; 0.375 at 4/5; 1.0 at 3/5).
pair["significant"] = (~pair["ci_includes_zero"].astype(bool)) & (pair["sign_p"] < 0.20)

pw_rows = []
prev_metric = None
prev_comp = None
for _, r in pair.iterrows():
    metric_lbl = "Spearman IC" if r["metric"] == "ic" else "Decile Spread"
    comp_lbl = COMP_LABEL.get(r["comparison"], r["comparison"])
    metric_cell = metric_lbl if r["metric"] != prev_metric else ""
    comp_cell = comp_lbl if (r["comparison"] != prev_comp or r["metric"] != prev_metric) else ""
    prev_metric = r["metric"]
    prev_comp = r["comparison"]

    if r["metric"] == "ic":
        diff = f"{r['mean_diff']:+.4f}"
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    else:
        diff = f"{r['mean_diff']:+.2f}\\%"
        ci = f"[{r['ci_lo']:+.2f}\\%, {r['ci_hi']:+.2f}\\%]"
    p = f"{r['sign_p']:.3f}" if r["sign_p"] >= 0.001 else "<0.001"

    if r["significant"]:
        diff = signif_bold(diff, True)
        ci = signif_bold(ci, True)
        p = signif_bold(p, True)

    pw_rows.append(
        f"{metric_cell:<14} & {comp_cell:<24} & {latex_escape(r['model']):<18} & "
        f"{diff:<22} & {ci:<32} & {p:<10} \\\\"
    )

table2 = (
    "\\begin{table*}[t]\n"
    "\\centering\n"
    "\\caption{Paired pairwise comparisons across the three feature "
    "configurations, holding the model constant. The mean delta is the "
    "average of $K=5$ paired fold-level differences; the 95\\% confidence "
    "interval is a 10{{,}}000-resample paired bootstrap on those five "
    "differences; the sign-test $p$ is a two-sided binomial test on the "
    "signs of the differences. Bold rows are those whose CI excludes zero "
    "\\emph{and} whose sign-test $p<0.20$.}\n"
    "\\label{tab:pairwise_stats}\n"
    "\\small\n"
    "\\begin{tabular}{llcccc}\n"
    "\\toprule\n"
    "\\textbf{Metric} & \\textbf{Comparison} & \\textbf{Model} & "
    "\\textbf{Mean delta} & \\textbf{95\\% bootstrap CI} & "
    "\\textbf{Sign-test } $p$ \\\\\n"
    "\\midrule\n"
    + "\n".join(pw_rows) + "\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table*}\n"
)

# ── Headline numbers for the prose ──────────────────────────────────────────

def lookup(feat, model, col):
    row = summary[(summary.feature_set == feat) & (summary.model == model)]
    if row.empty:
        return float("nan")
    return float(row[col].values[0])


best_ic_combined = summary[summary.feature_set == "tabular+embedding"].sort_values("ic_mean", ascending=False).iloc[0]
best_ds_combined = summary[summary.feature_set == "tabular+embedding"].sort_values("ds_mean", ascending=False).iloc[0]
best_ic_embed = summary[summary.feature_set == "embedding_only"].sort_values("ic_mean", ascending=False).iloc[0]
best_ic_tabular = summary[summary.feature_set == "tabular_only"].sort_values("ic_mean", ascending=False).iloc[0]

# Significant rows in IC: Combined - Tabular
sig_pairs = pair[(pair.metric == "ic") &
                 (pair.comparison == "tabular+embedding - tabular_only") &
                 pair.significant]

# Ridge row for the honest-reporting paragraph (Combined - Tabular)
ridge_row = pair[(pair.metric == "ic") &
                 (pair.comparison == "tabular+embedding - tabular_only") &
                 (pair.model == "Ridge")]
ridge_ic_diff = float(ridge_row["mean_diff"].values[0]) if not ridge_row.empty else float("nan")
ridge_ic_lo = float(ridge_row["ci_lo"].values[0]) if not ridge_row.empty else float("nan")
ridge_ic_hi = float(ridge_row["ci_hi"].values[0]) if not ridge_row.empty else float("nan")
ridge_p = float(ridge_row["sign_p"].values[0]) if not ridge_row.empty else float("nan")

# Diverse-ensemble result
div_row = summary[(summary.feature_set == "tabular+embedding") &
                  (summary.model == "SoftVote_Diverse")]
if not div_row.empty:
    div_ic_mean = float(div_row["ic_mean"].values[0])
    div_ic_se = float(div_row["ic_se"].values[0])
    div_ds_mean = float(div_row["ds_mean"].values[0])
    div_ds_se = float(div_row["ds_se"].values[0])
    div_sentence = (
        f"The diverse-ensemble selector (Part 4 P5) pruned the SoftVote pool "
        f"to members whose pairwise OOF Pearson correlation stays below 0.85, "
        f"yielding SoftVote\\_Diverse at $+{div_ic_mean:.4f}$ IC "
        f"(SE ${div_ic_se:.4f}$) and ${div_ds_mean:+.2f}\\%$ decile spread "
        f"(SE ${div_ds_se:.2f}\\%$). "
    )
else:
    div_sentence = ""

# Correlation summary
def cr(a, b):
    try:
        return corr.loc[a, b]
    except Exception:
        return float("nan")

corr_models = [m for m in ("Ridge", "RandomForest", "LightGBM", "XGBoost", "CatBoost", "MLP")
               if m in corr.index]
ridge_corrs = [abs(cr("Ridge", m)) for m in corr_models if m != "Ridge"]
ridge_lo = min(ridge_corrs) if ridge_corrs else float("nan")
ridge_hi = max(ridge_corrs) if ridge_corrs else float("nan")
gb_pairs = [("LightGBM", "XGBoost"), ("LightGBM", "CatBoost"), ("XGBoost", "CatBoost")]
gb_corrs = [cr(a, b) for (a, b) in gb_pairs if a in corr.index and b in corr.index]
gb_lo = min(gb_corrs) if gb_corrs else float("nan")
gb_hi = max(gb_corrs) if gb_corrs else float("nan")

# ── Prose paragraphs ────────────────────────────────────────────────────────

prose = (
    "\\subsection{Walk-Forward Bake-Off Results}\n"
    f"Table~\\ref{{tab:walk_forward_full}} reports the full $K=5$ "
    f"walk-forward bake-off (mean and SE across folds, plus the count of "
    f"folds in which each metric was positive). On the tabular-only "
    f"configuration the predictive signal is modest "
    f"(best model {latex_escape(best_ic_tabular['model'])} at "
    f"$+{best_ic_tabular['ic_mean']:.4f}$ IC, SE ${best_ic_tabular['ic_se']:.4f}$). "
    f"Structural KGE features alone carry substantially more signal "
    f"({latex_escape(best_ic_embed['model'])} reaches $+{best_ic_embed['ic_mean']:.4f}$ IC, "
    f"SE ${best_ic_embed['ic_se']:.4f}$). "
    f"Fusing tabular and embedding features delivers the best combination of "
    f"signal and stability: {latex_escape(best_ic_combined['model'])} "
    f"achieves $+{best_ic_combined['ic_mean']:.4f}$ IC "
    f"(SE ${best_ic_combined['ic_se']:.4f}$, "
    f"{int(best_ic_combined['n_pos_ic'])}/{int(best_ic_combined['K'])} folds "
    f"positive) and {latex_escape(best_ds_combined['model'])} maximises the "
    f"long/short decile spread at "
    f"${best_ds_combined['ds_mean']:+.2f}\\%$ "
    f"(SE ${best_ds_combined['ds_se']:.2f}\\%$, "
    f"{int(best_ds_combined['n_pos_ds'])}/{int(best_ds_combined['K'])} folds positive). "
    f"Cross-sectional Z standardisation (Part 4 P3) and the small MLP "
    f"regressor (Part 4 P4) participate in the bake-off so the paper can "
    f"argue from a fair non-linear baseline, not a missing one.\n\n"
    + table1 + "\n"
    "\\subsection{Rigorous Pairwise Statistical Inference}\n"
    f"Holding the model constant, we compare each feature configuration "
    f"against the others. For each model and each comparison we take the "
    f"$K=5$ paired fold-level differences, run a two-sided binomial sign "
    f"test against $\\text{{H}}_0=0.5$, and bootstrap a 95\\% CI on the "
    f"mean of those five differences from 10{{,}}000 paired resamples. "
    f"With $K=5$ the minimum attainable two-sided sign-test $p$ is "
    f"$0.0625$ at $5/5$ — the bootstrap CI is the complementary range "
    f"estimate. We flag a row as significant only when both the CI excludes "
    f"zero \\emph{{and}} the sign-test $p<0.20$ "
    f"(Table~\\ref{{tab:pairwise_stats}}).\n\n"
    f"On the IC, the Combined-minus-Tabular contrast clears that bar for "
    f"{len(sig_pairs)} of the "
    f"{len(pair[(pair.metric == 'ic') & (pair.comparison == 'tabular+embedding - tabular_only')])} "
    f"models tested, with paired-mean IC improvements ranging from "
    f"${sig_pairs['mean_diff'].min():+.4f}$ to "
    f"${sig_pairs['mean_diff'].max():+.4f}$ "
    f"and 95\\% bootstrap CIs entirely above zero. "
    f"The Ridge baseline (Part 4 P1) is materially weaker: its "
    f"Combined-minus-Tabular IC gain is ${ridge_ic_diff:+.4f}$ with a "
    f"bootstrap CI of $[{ridge_ic_lo:+.4f}, {ridge_ic_hi:+.4f}]$ and "
    f"sign-test $p={ridge_p:.3f}$ — that CI crosses zero, so we cannot "
    f"reject the null that adding KG embeddings does nothing for Ridge. "
    f"Non-linear ensembles are doing the heavy lifting; linearity alone "
    f"does not extract the cross-feature interaction the KG embeddings "
    f"encode. Rank-aware losses (LightGBM lambdarank, XGBoost "
    f"rank:pairwise, CatBoost YetiRankPairwise) underperformed their MSE "
    f"counterparts in every feature configuration and are reported in "
    f"Table~\\ref{{tab:walk_forward_full}} for transparency rather than "
    f"recommended for deployment.\n\n"
    + table2 + "\n"
    "\\subsection{OOF Diversity Diagnostic and Ensemble Construction}\n"
    f"To understand whether SoftVote is averaging genuinely independent "
    f"signal or just adding similar trees together, we compute the pairwise "
    f"Pearson correlation of out-of-fold predictions on the combined "
    f"feature set across the $K=5$ folds. The gradient-boosted trees "
    f"(CatBoost, LightGBM, XGBoost) correlate with each other in the "
    f"range $r\\in[{gb_lo:.3f}, {gb_hi:.3f}]$ — strong but not redundant. "
    f"Ridge correlates with the trees only at $r\\in[{ridge_lo:.3f}, "
    f"{ridge_hi:.3f}]$, which is exactly the diversity profile that lets "
    f"a simple rank-average ensemble extract additional variance "
    f"reduction. {div_sentence}"
    f"The deployed artefact persisted to \\texttt{{best\\_model.pkl}} is "
    f"the SoftVote ensemble over the gradient-boosted regressors on the "
    f"combined feature set, with the cross-sectional Z preprocessing flag "
    f"set so that the live trading bot and the backtests apply the same "
    f"per-date normalisation used at training time.\n\n"
)

# ── Splice into report.tex ───────────────────────────────────────────────────

with open(REPORT, "r") as f:
    src = f.read()

START_PAT = re.compile(r"\\subsection\{Walk-Forward Bake-Off Results\}")
END_PAT = re.compile(r"\\section\{Operational Quantitative Infrastructure")

m_start = START_PAT.search(src)
m_end = END_PAT.search(src)
if not m_start or not m_end:
    raise SystemExit("Could not locate splice anchors in report.tex")

new_src = src[:m_start.start()] + prose + src[m_end.start():]
with open(REPORT, "w") as f:
    f.write(new_src)

print("OK: regenerated tables + prose")
print(f"  best IC (combined): {best_ic_combined['model']}  IC={best_ic_combined['ic_mean']:+.4f} (SE {best_ic_combined['ic_se']:.4f})")
print(f"  best DS (combined): {best_ds_combined['model']}  DS={best_ds_combined['ds_mean']:+.2f}% (SE {best_ds_combined['ds_se']:.2f}%)")
print(f"  significant Combined-Tabular IC rows: {len(sig_pairs)}/{len(pair[(pair.metric=='ic')&(pair.comparison=='tabular+embedding - tabular_only')])}")
print(f"  SoftVote_Diverse present: {'yes' if not div_row.empty else 'no'}")
