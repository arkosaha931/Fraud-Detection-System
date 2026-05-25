import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

import joblib



os.makedirs("model", exist_ok=True)



PAYSIM_STEP_MAX = 744



def load_data(path):

    df = pd.read_csv(path)

    print(f"\n✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    return df




def add_behavioral_features(df):

  

    df["step"] = (df["step"] % PAYSIM_STEP_MAX) + 1

    df["transaction_hour"] = df["step"] % 24

    df["is_night_transaction"] = (
        (df["transaction_hour"] >= 22) |
        (df["transaction_hour"] <= 6)
    ).astype(int)


    df["amount_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    df["high_amount_ratio"] = (
        df["amount_ratio"] > 0.75
    ).astype(int)

    df["extreme_amount_ratio"] = (
        df["amount_ratio"] > 0.95
    ).astype(int)


    df["log_amount"]         = np.log1p(df["amount"])
    df["log_oldbalanceOrg"]  = np.log1p(df["oldbalanceOrg"])
    df["log_newbalanceOrig"] = np.log1p(df["newbalanceOrig"])

    

    expected_new_orig = df["oldbalanceOrg"] - df["amount"]
    expected_new_dest = df["oldbalanceDest"] + df["amount"]

    df["orig_balance_mismatch"] = (
        (df["newbalanceOrig"] - expected_new_orig).abs() > 1
    ).astype(int)

    df["dest_balance_mismatch"] = (
        (df["newbalanceDest"] - expected_new_dest).abs() > 1
    ).astype(int)

    

    df["zero_balance_orig"] = (
        df["oldbalanceOrg"] == 0
    ).astype(int)

    df["account_drained"] = (
        (df["newbalanceOrig"] == 0) &
        (df["oldbalanceOrg"] > 0)
    ).astype(int)

    df["zero_balance_dest"] = (
        df["oldbalanceDest"] == 0
    ).astype(int)

    df["dest_no_increase"] = (
        df["newbalanceDest"] <= df["oldbalanceDest"]
    ).astype(int)

   

    df["device_risk"] = (
        (df["type"].isin(["TRANSFER", "CASH_OUT"])) &
        (df["amount_ratio"] > 0.65)
    ).astype(int)

  
    df["location_risk"] = (
        (df["is_night_transaction"] == 1) &
        (df["amount_ratio"] > 0.45)
    ).astype(int)

    return df




def preprocess_data(df):

    # FEATURE ENGINEERING
    df = add_behavioral_features(df)

    # DROP NAME COLUMNS
    df = df.drop(["nameOrig", "nameDest"], axis=1)

    # ENCODE TRANSACTION TYPE
    encoder = LabelEncoder()
    df["type"] = encoder.fit_transform(df["type"])

    joblib.dump(encoder, "model/label_encoder.pkl")

    print(f"\n✅ Encoder saved.")
    print(f"   Known types: {list(encoder.classes_)}")

    # SEPARATE FEATURES AND TARGET
    X = df.drop(["isFraud"], axis=1)
    y = df["isFraud"]

    # SAVE FEATURE COLUMNS
    feature_cols = X.columns.tolist()

    joblib.dump(feature_cols, "model/feature_columns.pkl")

    print(f"\n✅ Feature columns saved.")
    print(f"   Total features : {len(feature_cols)}")
    print(f"   Columns        : {feature_cols}")

    # SCALE FEATURES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "model/scaler.pkl")

    print(f"\n✅ Scaler saved.")

    return X_scaled, y, feature_cols