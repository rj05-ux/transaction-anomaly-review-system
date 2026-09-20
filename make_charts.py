import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

df = pd.read_csv("transactions_scored.csv", parse_dates=["timestamp"])

# Chart 1: Monthly flagged volume
monthly = df.set_index("timestamp").resample("ME")["flagged_for_review"].sum()
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(monthly.index.strftime("%b"), monthly.values, color="#1F4E79")
ax.set_title("Flagged Transactions by Month", fontsize=12, fontweight="bold")
ax.set_ylabel("Flagged count")
plt.tight_layout()
plt.savefig("chart_monthly.png", dpi=150)
plt.close()

# Chart 2: Flag rate by payment mode
by_mode = df.groupby("payment_mode")["flagged_for_review"].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(by_mode.index, by_mode.values, color="#2E75B6")
ax.set_title("Flag Rate (%) by Payment Mode", fontsize=12, fontweight="bold")
ax.set_xlabel("% of transactions flagged")
plt.tight_layout()
plt.savefig("chart_mode.png", dpi=150)
plt.close()

# Chart 3: Reason breakdown
reasons = {
    "High-amount\ndeviation": df["rule_flag_high_amount"].sum(),
    "Off-home-city": df["is_off_home_city"].sum(),
    "Odd hour\n(1-4 AM)": df["is_odd_hour"].sum(),
    "Isolation Forest\n(ML model)": df["iso_anomaly"].sum(),
}
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(reasons.keys(), reasons.values(), color="#548235")
ax.set_title("Anomaly Signals Triggered (counts, overlapping)", fontsize=12, fontweight="bold")
ax.set_ylabel("Transactions")
plt.tight_layout()
plt.savefig("chart_reasons.png", dpi=150)
plt.close()

print("Charts saved.")
