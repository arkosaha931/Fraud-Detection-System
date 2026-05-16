import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

import joblib

# ==============================
# LOAD DATASET
# ==============================

def load_data(path):

    df = pd.read_csv(path)

    return df


# ==============================
# FEATURE ENGINEERING
# ==============================

def add_behavioral_features(df):

    # REALISTIC FEATURES

    df["transaction_hour"] = (
        df["step"] % 24
    )

    # DEFAULT VALUES
    # (Can be improved later)

    df["transaction_frequency"] = 1

    df["device_risk"] = 0

    df["location_risk"] = 0

    # AMOUNT RATIO

    df["amount_ratio"] = (

        df["amount"]

        /

        (df["oldbalanceOrg"] + 1)

    )

    return df


# ==============================
# PREPROCESS DATA
# ==============================

def preprocess_data(df):

    # ADD FEATURES

    df = add_behavioral_features(df)

    # REMOVE UNUSED COLUMNS

    df = df.drop(
        ["nameOrig", "nameDest"],
        axis=1
    )

    # ENCODE TRANSACTION TYPE

    encoder = LabelEncoder()

    df["type"] = encoder.fit_transform(
        df["type"]
    )

    # SAVE ENCODER

    joblib.dump(
        encoder,
        "model/label_encoder.pkl"
    )

    # FEATURES

    X = df.drop(
        "isFraud",
        axis=1
    )

    # TARGET

    y = df["isFraud"]

    # SAVE FEATURE COLUMNS

    joblib.dump(
        X.columns.tolist(),
        "model/feature_columns.pkl"
    )

    print("\nFEATURE COLUMNS:\n")

    print(X.columns.tolist())

    # SCALE FEATURES

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # SAVE SCALER

    joblib.dump(
        scaler,
        "model/scaler.pkl"
    )

    return X_scaled, y