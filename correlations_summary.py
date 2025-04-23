
# correlations_summary.py
# This script computes Pearson correlation coefficients and p-values for the relationship
# between performance metrics (OSATS + UWOMSA + operative time) and surgical expertise.

import pandas as pd
from scipy.stats import pearsonr

# Load dataset
df = pd.read_excel("Mycrovascular OSATs (Responses).xlsx")

# Convert and clean data
df_clean = df.copy()
df_clean["Group_Binary"] = df_clean["Group"].apply(lambda x: 1 if str(x).lower() == "expert" else 0)
numeric_cols = ["UWOMSA_A_AP", "UWOMSA_B_AP", "UWOMSA_C_AP",
                "RfT_AP", "TnM_AP", "Instrument_AP", "Flow_AP", "Knowledge_AP",
                "Operative Time (Mins)"]

for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
df_clean = df_clean.dropna(subset=numeric_cols)

# Compute Pearson correlations
results = []
for feature in numeric_cols:
    r, p = pearsonr(df_clean[feature], df_clean["Group_Binary"])
    results.append({"Feature": feature.replace("_AP", "").replace(" (Mins)", ""), "Pearson r": round(r, 3), "p-value": round(p, 3)})

# Create and save table
results_df = pd.DataFrame(results).sort_values(by="Pearson r", ascending=False)
results_df.to_csv("correlation_results.csv", index=False)
print(results_df)
