"""
Transaction Anomaly Detection Analysis
----------------------------------------
Business goal: flag transactions that deviate from a customer's own
normal behavior (amount, time, location) for manual review, reducing
exposure to fraudulent or erroneous transactions.

Approach: per-customer behavioral z-scores (amount) + rule-based flags
(odd hour, off-home-city) combined into a review-priority score, cross-
checked against an unsupervised Isolation Forest model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

df = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["is_off_home_city"] = (df["city"] != df["customer_home_city"]).astype(int)
df["is_odd_hour"] = df["hour"].between(1, 4).astype(int)

# ---- Per-customer behavioral baseline (amount z-score) ----
cust_stats = df.groupby("customer_id")["amount"].agg(["mean", "std"]).rename(
    columns={"mean": "cust_avg_amount", "std": "cust_std_amount"})
cust_stats["cust_std_amount"] = cust_stats["cust_std_amount"].replace(0, np.nan)
df = df.merge(cust_stats, on="customer_id", how="left")
df["amount_zscore"] = (df["amount"] - df["cust_avg_amount"]) / df["cust_std_amount"]
df["amount_zscore"] = df["amount_zscore"].fillna(0)

# ---- Rule-based review score (0-3+) ----
df["rule_flag_high_amount"] = (df["amount_zscore"] > 3).astype(int)
df["review_score"] = (
    df["rule_flag_high_amount"] * 2
    + df["is_off_home_city"]
    + df["is_odd_hour"]
)

# ---- Unsupervised cross-check: Isolation Forest ----
features = df[["amount", "amount_zscore", "is_off_home_city", "is_odd_hour", "hour"]].fillna(0)
iso = IsolationForest(contamination=0.02, random_state=42, n_estimators=200)
df["iso_anomaly"] = (iso.fit_predict(features) == -1).astype(int)

# ---- Final flag: rule-based score >= 2 OR isolation forest agrees ----
df["flagged_for_review"] = ((df["review_score"] >= 2) | (df["iso_anomaly"] == 1)).astype(int)

# ---- Summary stats ----
total = len(df)
flagged = df["flagged_for_review"].sum()
print(f"Total transactions analyzed: {total:,}")
print(f"Flagged for review: {flagged:,} ({flagged/total*100:.2f}%)")
print(f"  - High-amount deviations: {df['rule_flag_high_amount'].sum():,}")
print(f"  - Off-home-city transactions: {df['is_off_home_city'].sum():,}")
print(f"  - Odd-hour (1-4 AM) transactions: {df['is_odd_hour'].sum():,}")
print(f"  - Isolation Forest anomalies: {df['iso_anomaly'].sum():,}")

by_category = df[df["flagged_for_review"] == 1]["merchant_category"].value_counts()
print("\nFlagged transactions by merchant category (top 5):")
print(by_category.head(5))

by_mode = df[df["flagged_for_review"] == 1]["payment_mode"].value_counts()
print("\nFlagged transactions by payment mode:")
print(by_mode)

monthly = df.set_index("timestamp").resample("ME")["flagged_for_review"].sum()
print("\nFlagged volume by month:")
print(monthly)

df.to_csv("transactions_scored.csv", index=False)
print("\nSaved scored dataset -> transactions_scored.csv")
