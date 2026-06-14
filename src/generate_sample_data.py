"""
generate_sample_data.py
=======================
Generates a realistic synthetic dataset that mirrors the IBM Synthetic
Financial Dataset schema. Use this for quick testing if you don't have
the real dataset yet.

Usage:
    python -m src.generate_sample_data --rows 100000 --out data/fraud_dataset.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


def generate(n_rows: int = 100_000) -> pd.DataFrame:
    """
    Generate synthetic fraud transaction dataset.
    
    Args:
        n_rows: Number of transactions to generate
        
    Returns:
        DataFrame with synthetic transactions
    """
    print(f"[INFO] Generating {n_rows:,} synthetic transactions...")

    tx_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    type_probs = [0.34, 0.08, 0.35, 0.13, 0.10]

    step   = rng.integers(1, 744, size=n_rows)
    types  = rng.choice(tx_types, size=n_rows, p=type_probs)
    amount = np.abs(rng.lognormal(mean=6.5, sigma=2.0, size=n_rows)).round(2)

    n_customers  = max(1, n_rows // 10)
    n_merchants  = max(1, n_rows // 20)
    name_orig = [f"C{str(rng.integers(1_000_000, 9_999_999)):s}" for _ in range(n_customers)]
    name_dest = [f"M{str(rng.integers(1_000_000, 9_999_999)):s}" for _ in range(n_merchants)]

    orig_ids = rng.choice(name_orig, size=n_rows)
    dest_ids = rng.choice(name_dest, size=n_rows)

    old_bal_orig = np.abs(rng.lognormal(mean=8.5, sigma=2.5, size=n_rows)).round(2)
    new_bal_orig = np.maximum(0, old_bal_orig - amount).round(2)
    old_bal_dest = np.abs(rng.lognormal(mean=7.0, sigma=2.5, size=n_rows)).round(2)
    new_bal_dest = (old_bal_dest + amount).round(2)

    # Fraud logic: higher rates for TRANSFER and CASH_OUT with large amounts
    base_fraud_prob = 0.013
    fraud_prob = np.where(
        (np.isin(types, ["TRANSFER", "CASH_OUT"])) & (amount > 200_000),
        0.65,
        np.where(
            np.isin(types, ["TRANSFER", "CASH_OUT"]),
            0.05,
            base_fraud_prob,
        ),
    )
    is_fraud = rng.binomial(1, fraud_prob).astype(int)

    # For fraud rows: wipe out balance (common fraud pattern)
    new_bal_orig = np.where(is_fraud == 1, 0.0, new_bal_orig)
    new_bal_dest = np.where(is_fraud == 1, old_bal_dest, new_bal_dest)  # dest unchanged

    df = pd.DataFrame({
        "step":             step,
        "type":             types,
        "amount":           amount,
        "nameOrig":         orig_ids,
        "oldbalanceOrg":    old_bal_orig,
        "newbalanceOrig":   new_bal_orig,
        "nameDest":         dest_ids,
        "oldbalanceDest":   old_bal_dest,
        "newbalanceDest":   new_bal_dest,
        "isFraud":          is_fraud,
        "isFlaggedFraud":   np.zeros(n_rows, dtype=int),
    })

    print(f"[INFO] Generated. Fraud rate: {df['isFraud'].mean():.4%}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--out",  type=str, default="data/fraud_dataset.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = generate(args.rows)
    df.to_csv(args.out, index=False)
    print(f"[INFO] Saved to {args.out}")
    print("[INFO] Now run: python train_model.py --data", args.out)
