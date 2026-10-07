"""Generate a SYNTHETIC customer purchase dataset for testing the pipeline.

Replace dataset/customer_purchase.csv with a real dataset (e.g. from Kaggle)
for your final report. Real column names can be changed in config.py.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

age = rng.integers(18, 66, N)
gender = rng.choice(["Male", "Female"], N)
income = np.clip(rng.normal(55000, 20000, N), 15000, 150000).round(-2)
browsing = np.clip(rng.normal(15, 8, N), 1, 60).round(1)
prev = rng.choice(["Yes", "No"], N, p=[0.4, 0.6])
visits = np.clip(rng.poisson(6, N), 1, 30)

# Hidden rule: purchase probability rises with these signals, but outcomes stay
# random (like real customers), so the models cannot be perfect.
score = 1.5 * (
    -1.9
    + 0.025 * (age - 35)
    + 0.00003 * (income - 50000)
    + 0.05 * browsing
    + 0.9 * (prev == "Yes")
    + 0.07 * visits
)
purchased = (rng.random(N) < 1 / (1 + np.exp(-score))).astype(int)

df = pd.DataFrame({
    "Customer_ID": np.arange(1001, 1001 + N),
    "Age": age,
    "Gender": gender,
    "Income": income,
    "Browsing_Time": browsing,
    "Previous_Purchase": prev,
    "Num_Visits": visits,
    "Purchased": purchased,
})

# Add a few missing values so preprocessing has something to do
for col in ["Age", "Income", "Browsing_Time"]:
    df.loc[rng.choice(N, 15, replace=False), col] = np.nan

df.to_csv("dataset/customer_purchase.csv", index=False)
print(f"Saved dataset/customer_purchase.csv  shape={df.shape}")
print(df["Purchased"].value_counts().to_string())
