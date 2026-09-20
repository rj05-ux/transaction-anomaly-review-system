"""
Synthetic UPI / Digital Payments Transaction Dataset Generator
----------------------------------------------------------------
Simulates realistic customer transaction behavior for a fintech
platform, with a small proportion of injected anomalous transactions
(behaviorally unusual, not necessarily "confirmed fraud" - framed as
transactions warranting review, per BRD scope).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_CUSTOMERS = 800
N_TRANSACTIONS = 12000
START_DATE = datetime(2026, 1, 1)
DAYS_SPAN = 180

merchant_categories = [
    "Groceries", "Food Delivery", "Utility Bills", "E-commerce",
    "Fuel", "Entertainment", "Travel", "Healthcare", "Education",
    "P2P Transfer", "Recharge/DTH", "Insurance Premium"
]
payment_modes = ["UPI", "Debit Card", "Credit Card", "Net Banking", "Wallet"]
cities = ["Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Ahmedabad"]

# Give each customer a "home" city and a typical spend profile
customers = pd.DataFrame({
    "customer_id": [f"CUST{str(i).zfill(4)}" for i in range(N_CUSTOMERS)],
    "home_city": np.random.choice(cities, N_CUSTOMERS),
    "avg_txn_amount": np.random.gamma(shape=2.0, scale=650, size=N_CUSTOMERS) + 150,
    "monthly_txn_count": np.random.poisson(lam=14, size=N_CUSTOMERS) + 2,
})

rows = []
txn_id = 100000

for _, cust in customers.iterrows():
    n_txns = int(cust["monthly_txn_count"] * (DAYS_SPAN / 30))
    for _ in range(n_txns):
        day_offset = np.random.randint(0, DAYS_SPAN)
        hour = int(np.clip(np.random.normal(14, 5), 0, 23))
        minute = np.random.randint(0, 60)
        ts = START_DATE + timedelta(days=day_offset, hours=hour, minutes=minute)

        amount = max(20, np.random.normal(cust["avg_txn_amount"], cust["avg_txn_amount"] * 0.4))
        category = np.random.choice(merchant_categories)
        mode = np.random.choice(payment_modes, p=[0.45, 0.2, 0.15, 0.1, 0.1])
        city = cust["home_city"] if np.random.rand() > 0.05 else np.random.choice(cities)

        rows.append({
            "transaction_id": f"TXN{txn_id}",
            "customer_id": cust["customer_id"],
            "timestamp": ts,
            "amount": round(amount, 2),
            "merchant_category": category,
            "payment_mode": mode,
            "city": city,
            "customer_home_city": cust["home_city"],
        })
        txn_id += 1

df = pd.DataFrame(rows)

# ---- Inject anomalous transactions (~1.8% of volume) ----
n_anomalies = int(len(df) * 0.018)
anomaly_idx = np.random.choice(df.index, n_anomalies, replace=False)

for idx in anomaly_idx:
    kind = np.random.choice(["high_value", "odd_hour", "foreign_city", "rapid_repeat"], p=[0.4, 0.25, 0.2, 0.15])
    if kind == "high_value":
        df.loc[idx, "amount"] = df.loc[idx, "amount"] * np.random.uniform(6, 15)
    elif kind == "odd_hour":
        ts = df.loc[idx, "timestamp"]
        df.loc[idx, "timestamp"] = ts.replace(hour=np.random.randint(1, 4))
    elif kind == "foreign_city":
        cust_home = df.loc[idx, "customer_home_city"]
        other_cities = [c for c in cities if c != cust_home]
        df.loc[idx, "city"] = np.random.choice(other_cities)
        df.loc[idx, "amount"] = df.loc[idx, "amount"] * np.random.uniform(2, 4)

df = df.sort_values("timestamp").reset_index(drop=True)
df.to_csv("transactions.csv", index=False)
print(f"Generated {len(df)} transactions across {N_CUSTOMERS} customers.")
print(f"Injected {n_anomalies} anomalous-pattern transactions for detection testing.")
