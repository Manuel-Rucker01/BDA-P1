import pandas as pd
import numpy as np

# Load data
df_results = pd.read_csv('/Users/manuelruckerabella/Workspace/UNI/Q6/BDA/BDA-P1/ExploitationZone/per_fold_results.csv')
df_sign = pd.read_csv('/Users/manuelruckerabella/Workspace/UNI/Q6/BDA/BDA-P1/ExploitationZone/sign_tests.csv')
df_pair = pd.read_csv('/Users/manuelruckerabella/Workspace/UNI/Q6/BDA/BDA-P1/ExploitationZone/pairwise_comparisons.csv')

# Calculate mean and standard error for per-fold results
# SE = std / sqrt(K) where K = 5
def get_summary_stats():
    grouped = df_results.groupby(['feature_set', 'model'])
    records = []
    for (feat_set, model), grp in grouped:
        ic_mean = grp['ic'].mean()
        ic_se = grp['ic'].std() / np.sqrt(len(grp))
        ds_mean = grp['decile_spread_pct'].mean()
        ds_se = grp['decile_spread_pct'].std() / np.sqrt(len(grp))
        
        # Get sign test counts
        sign_row = df_sign[(df_sign['feature_set'] == feat_set) & (df_sign['model'] == model)]
        if not sign_row.empty:
            ic_pos = int(sign_row['n_pos_ic'].values[0])
            ds_pos = int(sign_row['n_pos_decile_spread'].values[0])
            K = int(sign_row['K'].values[0])
        else:
            ic_pos, ds_pos, K = 0, 0, 5
            
        records.append({
            'feature_set': feat_set,
            'model': model,
            'ic_mean': ic_mean,
            'ic_se': ic_se,
            'ic_sign': f"{ic_pos}/{K}",
            'ds_mean': ds_mean,
            'ds_se': ds_se,
            'ds_sign': f"{ds_pos}/{K}"
        })
    return pd.DataFrame(records)

df_sum = get_summary_stats()

feature_sets_ordered = ['tabular_only', 'embedding_only', 'tabular+embedding']
model_order = [
    'SoftVote', 'SoftVote_Rank', 'LightGBM', 'LightGBM_Rank', 
    'XGBoost', 'XGBoost_Rank', 'CatBoost', 'CatBoost_Rank', 
    'RandomForest', 'Stack', 'Ridge'
]

latex_rows = []
for fs in feature_sets_ordered:
    fs_label = fs.replace('_', ' ')
    first_fs = True
    
    # Sort models by order
    df_fs = df_sum[df_sum['feature_set'] == fs].copy()
    # Apply ordering
    df_fs['model_cat'] = pd.Categorical(df_fs['model'], categories=model_order, ordered=True)
    df_fs = df_fs.sort_values('model_cat')
    
    # Let's find the peak model for bolding (by IC mean)
    peak_model = df_fs.sort_values('ic_mean', ascending=False)['model'].values[0]
    peak_ds_model = df_fs.sort_values('ds_mean', ascending=False)['model'].values[0]
    
    for _, row in df_fs.iterrows():
        model = row['model']
        ic_str = f"{row['ic_mean']:+.4f} ({row['ic_se']:.4f})"
        ds_str = f"{row['ds_mean']:+.2f}\\% ({row['ds_se']:.2f}\\%)"
        
        if model == peak_model:
            ic_str = f"\\textbf{{{ic_str}}}"
        if model == peak_ds_model:
            ds_str = f"\\textbf{{{ds_str}}}"
            
        fs_col = fs_label if first_fs else ""
        first_fs = False
        
        latex_rows.append(f"{fs_col:<20} & {model:<15} & {ic_str:<30} & {row['ic_sign']:<5} & {ds_str:<30} & {row['ds_sign']:<5} \\\\")
    latex_rows.append("\\midrule")

# Remove last midrule
if latex_rows:
    latex_rows.pop()

# Sort pairwise table properly
df_pair_sorted = df_pair.copy()
df_pair_sorted['model_cat'] = pd.Categorical(df_pair_sorted['model'], categories=model_order, ordered=True)
df_pair_sorted = df_pair_sorted.sort_values(['metric', 'comparison', 'model_cat'])

latex_pair_rows = []
current_metric = ""
current_comp = ""

for _, row in df_pair_sorted.iterrows():
    metric = row['metric']
    comp = row['comparison']
    model = row['model']
    
    metric_label = "Spearman IC" if metric == "ic" else "Decile Spread"
    comp_label = comp.replace('tabular_only', 'Tabular').replace('embedding_only', 'Embedding').replace('tabular+embedding', 'Combined')
    
    metric_col = metric_label if metric != current_metric else ""
    comp_col = comp_label if comp != current_comp or metric != current_metric else ""
    
    current_metric = metric
    current_comp = comp
    
    diff_val = row['mean_diff']
    ci_lo = row['ci_lo']
    ci_hi = row['ci_hi']
    
    if metric == "ic":
        diff_str = f"{diff_val:+.4f}"
        ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
    else:
        diff_str = f"{diff_val:+.2f}\\%"
        ci_str = f"[{ci_lo:+.2f}\\%, {ci_hi:+.2f}\\%]"
        
    p_val = f"{row['sign_p']:.3f}" if row['sign_p'] >= 0.001 else "<0.001"
    
    # Bold if significant (i.e. CI does not contain zero)
    if not row['ci_includes_zero']:
        diff_str = f"\\textbf{{{diff_str}}}"
        ci_str = f"\\textbf{{{ci_str}}}"
        p_val = f"\\textbf{{{p_val}}}"
        
    latex_pair_rows.append(f"{metric_col:<15} & {comp_col:<25} & {model:<15} & {diff_str:<20} & {ci_str:<30} & {p_val:<10} \\\\")

with open('table_output.txt', 'w') as f:
    f.write("=================== TABLE 1: WALK-FORWARD BAKE-OFF ===================\n")
    f.write("\n".join(latex_rows))
    f.write("\n\n=================== TABLE 2: PAIRWISE COMPARISONS ===================\n")
    f.write("\n".join(latex_pair_rows))
    f.write("\n")

print("Wrote tables successfully to table_output.txt!")
