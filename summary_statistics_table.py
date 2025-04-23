
# summary_statistics_table.py
# This script generates descriptive statistics and p-values comparing expert vs novice/intermediate groups
# for OSATS, UWOMSA, and operative time.

import pandas as pd
import scipy.stats as stats

# Load dataset
df = pd.read_excel("Mycrovascular OSATs (Responses).xlsx")

# Preprocessing
df["Group_Binary"] = df["Group"].apply(lambda x: 1 if str(x).lower() == "expert" else 0)
numeric_cols = {
    "Operative Time (Minutes)": "Operative Time (Mins)",
    "Respect for Tissue": "RfT_AP",
    "Time and Motion": "TnM_AP",
    "Instrument Handling": "Instrument_AP",
    "Flow of Operation": "Flow_AP",
    "Knowledge of Procedure": "Knowledge_AP",
    "UWOMSA A (Dexterity)": "UWOMSA_A_AP",
    "UWOMSA B (Patency)": "UWOMSA_B_AP",
    "UWOMSA C (Leak/Flow)": "UWOMSA_C_AP"
}

# Clean and subset
df_clean = df.copy()
for col in numeric_cols.values():
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
df_clean = df_clean.dropna(subset=numeric_cols.values())

# Summary stats and group comparisons
results = []
for label, col in numeric_cols.items():
    if label == "Operative Time (Minutes)":
        total_mean = df_clean[col].mean()
        total_sd = df_clean[col].std()
        expert_mean = df_clean[df_clean["Group_Binary"] == 1][col].mean()
        expert_sd = df_clean[df_clean["Group_Binary"] == 1][col].std()
        novice_mean = df_clean[df_clean["Group_Binary"] == 0][col].mean()
        novice_sd = df_clean[df_clean["Group_Binary"] == 0][col].std()
        p = stats.ttest_ind(
            df_clean[df_clean["Group_Binary"] == 1][col],
            df_clean[df_clean["Group_Binary"] == 0][col],
            equal_var=False
        ).pvalue
        total_str = f"{total_mean:.2f} ± {total_sd:.2f}"
        expert_str = f"{expert_mean:.2f} ± {expert_sd:.2f}"
        novice_str = f"{novice_mean:.2f} ± {novice_sd:.2f}"
    else:
        total_med = df_clean[col].median()
        total_iqr = df_clean[col].quantile([0.25, 0.75])
        expert_vals = df_clean[df_clean["Group_Binary"] == 1][col]
        novice_vals = df_clean[df_clean["Group_Binary"] == 0][col]
        expert_med = expert_vals.median()
        expert_iqr = expert_vals.quantile([0.25, 0.75])
        novice_med = novice_vals.median()
        novice_iqr = novice_vals.quantile([0.25, 0.75])
        p = stats.mannwhitneyu(expert_vals, novice_vals, alternative="two-sided").pvalue
        total_str = f"{total_med:.2f} ({total_iqr.iloc[0]:.2f}–{total_iqr.iloc[1]:.2f})"
        expert_str = f"{expert_med:.2f} ({expert_iqr.iloc[0]:.2f}–{expert_iqr.iloc[1]:.2f})"
        novice_str = f"{novice_med:.2f} ({novice_iqr.iloc[0]:.2f}–{novice_iqr.iloc[1]:.2f})"
    results.append({
        "Metric": label,
        "Total": total_str,
        "Expert": expert_str,
        "Novice/Intermediate": novice_str,
        "P-value": round(p, 3)
    })

# Save to CSV
summary_df = pd.DataFrame(results)
summary_df.to_csv("summary_statistics_comparison.csv", index=False)
print(summary_df)
