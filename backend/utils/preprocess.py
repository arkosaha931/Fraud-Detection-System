import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

import joblib

# ==============================
# ENSURE MODEL DIR EXISTS
# ==============================

os.makedirs("model", exist_ok=True)


# ==============================
# LOAD DATASET
# ==============================

def load_data(path):

    df = pd.read_csv(path)

    print(f"\n✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


# ==============================
# FEATURE ENGINEERING
# ==============================

def add_behavioral_features(df):

    # ----------------------------
    # TEMPORAL FEATURES
    # ----------------------------

    # Hour of day derived from step
    # PaySim: 1 step = 1 hour
    df["transaction_hour"] = df["step"] % 24

    # Is transaction happening at night (10pm - 6am)
    # Fraud often happens at odd hours
    df["is_night_transaction"] = (
        (df["transaction_hour"] >= 22) |
        (df["transaction_hour"] <= 6)
    ).astype(int)

    # ----------------------------
    # BALANCE FEATURES
    # ----------------------------

    # How much of sender's balance is being sent
    # High ratio = draining account = suspicious
    df["amount_ratio"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    )

    # Balance delta for sender
    # Legitimate: newbalance = old - amount
    # Fraud: sometimes balance doesn't change (account manipulation)
    df["balance_diff_orig"] = (
        df["newbalanceOrig"] - (df["oldbalanceOrg"] - df["amount"])
    )

    # Balance delta for receiver
    df["balance_diff_dest"] = (
        df["newbalanceDest"] - (df["oldbalanceDest"] + df["amount"])
    )

    # If sender balance didn't drop correctly → suspicious
    df["orig_balance_mismatch"] = (
        df["balance_diff_orig"].abs() > 1
    ).astype(int)

    # If receiver balance didn't rise correctly → suspicious
    df["dest_balance_mismatch"] = (
        df["balance_diff_dest"].abs() > 1
    ).astype(int)

    # Was sender account empty before?
    df["zero_balance_orig"] = (
        df["oldbalanceOrg"] == 0
    ).astype(int)

    # Did sender fully drain the account?
    df["account_drained"] = (
        (df["newbalanceOrig"] == 0) &
        (df["oldbalanceOrg"] > 0)
    ).astype(int)

    # ----------------------------
    # TRANSACTION FREQUENCY
    # Simulate per-sender frequency
    # using step windows (realistic for PaySim)
    # ----------------------------

    # Count transactions per sender in same step window (±12 steps)
    # This simulates burst behavior (many txns in short time)
    df = df.sort_values("step")

    freq_map = (
        df.groupby("nameOrig")["step"]
        .transform("count")
    )

    df["transaction_frequency"] = freq_map

    # ----------------------------
    # DEVICE & LOCATION RISK
    # PaySim has no device/location columns
    # We simulate these realistically using
    # statistical patterns from fraud rows
    # so the model actually LEARNS them
    # ----------------------------

    # Fraud in PaySim is concentrated in:
    # TRANSFER and CASH_OUT types
    # We use this to simulate device_risk:
    # High-risk device = transaction type is TRANSFER or CASH_OUT
    # AND amount is above 75th percentile

    amount_75th = df["amount"].quantile(0.75)

    df["device_risk"] = (
        (df["type"].isin(["TRANSFER", "CASH_OUT"])) &
        (df["amount"] > amount_75th)
    ).astype(int)

    # location_risk:
    # Simulate foreign/risky location as:
    # transaction at night AND high amount
    # This creates real variance so model learns it

    amount_median = df["amount"].median()

    df["location_risk"] = (
        (df["is_night_transaction"] == 1) &
        (df["amount"] > amount_median)
    ).astype(int)

    # ----------------------------
    # AMOUNT BINS (log scale)
    # Neural nets work better with
    # log-transformed skewed features
    # ----------------------------

    df["log_amount"] = np.log1p(df["amount"])

    df["log_oldbalanceOrg"] = np.log1p(df["oldbalanceOrg"])

    df["log_newbalanceOrig"] = np.log1p(df["newbalanceOrig"])

    return df


# ==============================
# PREPROCESS DATA
# ==============================

def preprocess_data(df):

    # ADD ENGINEERED FEATURES
    df = add_behavioral_features(df)

    # REMOVE NAME COLUMNS
    df = df.drop(
        ["nameOrig", "nameDest"],
        axis=1
    )

    # ENCODE TRANSACTION TYPE
    encoder = LabelEncoder()

    df["type"] = encoder.fit_transform(df["type"])

    # SAVE ENCODER
    joblib.dump(encoder, "model/label_encoder.pkl")

    print("\n✅ Encoder saved.")
    print(f"   Known types: {list(encoder.classes_)}")

    # DROP INTERMEDIATE COLUMNS
    # (used for feature creation but not for model input)
    cols_to_drop = [
        "isFraud",
        "balance_diff_orig",
        "balance_diff_dest"
    ]

    # FEATURES
    X = df.drop(
        [c for c in cols_to_drop if c in df.columns],
        axis=1
    )

    # TARGET
    y = df["isFraud"]

    # SAVE FEATURE COLUMNS
    joblib.dump(
        X.columns.tolist(),
        "model/feature_columns.pkl"
    )

    print("\n✅ Feature columns saved.")
    print(f"   Total features: {len(X.columns)}")
    print(f"   Columns: {X.columns.tolist()}")

    # SCALE FEATURES
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # SAVE SCALER
    joblib.dump(scaler, "model/scaler.pkl")

    print("\n✅ Scaler saved.")

    return X_scaled, y, X.columns.tolist()
