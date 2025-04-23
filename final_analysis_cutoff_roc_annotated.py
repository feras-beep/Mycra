import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load data (adjust path as needed)
df = pd.read_excel("Mycrovascular OSATs (Responses).xlsx")

# === Participant Preprocessing ===
df_clean = df.copy()
df_clean = df_clean[pd.to_numeric(df_clean["Operative Time (Mins)"], errors="coerce").notnull()]
df_clean["Group_Binary"] = df_clean["Group"].apply(lambda x: 1 if str(x).lower() == "expert" else 0)
df_clean["Operative Time (Mins)"] = df_clean["Operative Time (Mins)"].astype(float)

# === Define scoring domains ===
osats_cols = ["RfT_AP", "TnM_AP", "Instrument_AP", "Flow_AP", "Knowledge_AP"]
uwomsa_cols = ["UWOMSA_A_AP", "UWOMSA_B_AP", "UWOMSA_C_AP"]

# Clean numeric columns
for col in osats_cols + uwomsa_cols:
    df_clean = df_clean[pd.to_numeric(df_clean[col], errors="coerce").notnull()]
df_clean[osats_cols + uwomsa_cols] = df_clean[osats_cols + uwomsa_cols].astype(float)

# === Combined Scores ===
# Used for:
# - Model Performance (Combined Model)
# - ROC Curve for Combined Scores
df_clean["Combined_Score"] = df_clean[osats_cols + uwomsa_cols].mean(axis=1)
df_clean["UWOMSA_Mean"] = df_clean[uwomsa_cols].mean(axis=1)
df_clean["OSATS_Mean"] = df_clean[osats_cols].mean(axis=1)

# === ROC + Youden's J Cutoff Function ===
def get_roc_and_cutoff(y_true, y_score, inverse=False):
    fpr, tpr, thresholds = roc_curve(y_true, -y_score if inverse else y_score)
    youden_index = np.argmax(tpr - fpr)
    cutoff = thresholds[youden_index]
    auc = roc_auc_score(y_true, -y_score if inverse else y_score)
    accuracy = np.mean((y_score < cutoff if inverse else y_score >= cutoff) == y_true)
    return fpr, tpr, youden_index, cutoff, auc, accuracy

# === ROC Analyses ===
# Referenced in Results section: "Receiver Operating Characteristic (ROC) analysis..."
fpr_time, tpr_time, idx_time, cutoff_time, auc_time, acc_time = get_roc_and_cutoff(df_clean["Group_Binary"], df_clean["Operative Time (Mins)"], inverse=True)
fpr_comb, tpr_comb, idx_comb, cutoff_comb, auc_comb, acc_comb = get_roc_and_cutoff(df_clean["Group_Binary"], df_clean["Combined_Score"])
fpr_uwomsa, tpr_uwomsa, idx_uwomsa, cutoff_uwomsa, auc_uwomsa, acc_uwomsa = get_roc_and_cutoff(df_clean["Group_Binary"], df_clean["UWOMSA_Mean"])
fpr_osats, tpr_osats, idx_osats, cutoff_osats, auc_osats, acc_osats = get_roc_and_cutoff(df_clean["Group_Binary"], df_clean["OSATS_Mean"])

# === ROC Plot ===
# Used to generate the figure referenced under: ROC curves with AUC and optimal cutoffs
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_time, tpr_time, label=f"Operative Time (AUC={auc_time:.2f}, Cutoff={abs(cutoff_time):.2f}, Acc={acc_time:.2f})", color='blue')
ax.plot(fpr_comb, tpr_comb, label=f"Combined Score (AUC={auc_comb:.2f}, Cutoff={cutoff_comb:.2f}, Acc={acc_comb:.2f})", color='green')
ax.plot(fpr_uwomsa, tpr_uwomsa, label=f"UWOMSA Mean (AUC={auc_uwomsa:.2f}, Cutoff={cutoff_uwomsa:.2f}, Acc={acc_uwomsa:.2f})", color='orange')
ax.plot(fpr_osats, tpr_osats, label=f"OSATS Mean (AUC={auc_osats:.2f}, Cutoff={cutoff_osats:.2f}, Acc={acc_osats:.2f})", color='red')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Optimal points
ax.plot(fpr_time[idx_time], tpr_time[idx_time], 'o', color='blue')
ax.plot(fpr_comb[idx_comb], tpr_comb[idx_comb], 'o', color='green')
ax.plot(fpr_uwomsa[idx_uwomsa], tpr_uwomsa[idx_uwomsa], 'o', color='orange')
ax.plot(fpr_osats[idx_osats], tpr_osats[idx_osats], 'o', color='red')

ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curves with AUC, Cutoff, and Accuracy")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
