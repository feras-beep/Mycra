
# inter_rater_reliability.py
# This script calculates Pearson correlation and Cohen's kappa for inter-rater reliability
# across OSATS and UWOMSA scoring domains.

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

# Load the dataset
df = pd.read_excel("Mycrovascular OSATs (Responses).xlsx")

# Define paired variables
domains = {
    "UWOMSA A PREPARATION": ("UWOMSA_A_AP", "UWOMSA_A_SW"),
    "UWOMSA B SUTURING": ("UWOMSA_B_AP", "UWOMSA_B_SW"),
    "UWOMSA C FINAL PRODUCT": ("UWOMSA_C_AP", "UWOMSA_C_SW"),
    "INSTRUMENT HANDLING": ("Instrument_AP", "Instrument_SW"),
    "RESPECT FOR TISSUE": ("RfT_AP", "RfT_SW"),
    "TIME AND MOTION": ("TnM_AP", "TnM_SW"),
    "FLOW OF OPERATION": ("Flow_AP", "Flow_SW"),
    "KNOWLEDGE OF PROCEDURE": ("Knowledge_AP", "Knowledge_SW")
}

# Function to classify Pearson interpretation
def interpret_pearson(r):
    abs_r = abs(r)
    if abs_r >= 0.5:
        return "High Degree"
    elif abs_r >= 0.3:
        return "Moderate Degree"
    elif abs_r >= 0.1:
        return "Low Degree"
    else:
        return "No Correlation"

# Function to classify Cohen's kappa
def interpret_kappa(kappa):
    if kappa >= 0.81:
        return "Almost perfect"
    elif kappa >= 0.61:
        return "Substantial"
    elif kappa >= 0.41:
        return "Moderate"
    elif kappa >= 0.21:
        return "Fair"
    elif kappa >= 0.01:
        return "Slight"
    else:
        return "Poor"

# Calculate reliability metrics
results = []
for label, (ap_col, sw_col) in domains.items():
    df[ap_col] = pd.to_numeric(df[ap_col], errors="coerce")
    df[sw_col] = pd.to_numeric(df[sw_col], errors="coerce")
    valid = df[[ap_col, sw_col]].dropna()
    r, p = pearsonr(valid[ap_col], valid[sw_col])
    kappa = cohen_kappa_score(valid[ap_col], valid[sw_col])
    results.append({
        "Domain": label,
        "Pearson r": round(r, 3),
        "Pearson Interpretation": interpret_pearson(r),
        "P-value (r)": "<0.001" if p < 0.001 else round(p, 3),
        "Cohen's Kappa": round(kappa, 3),
        "Kappa Interpretation": interpret_kappa(kappa)
    })

# Save as CSV
reliability_df = pd.DataFrame(results)
reliability_df.to_csv("inter_rater_reliability.csv", index=False)
print(reliability_df)
