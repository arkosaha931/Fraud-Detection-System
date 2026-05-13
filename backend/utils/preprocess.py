import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# LOAD DATASET
# ==============================

def load_data(path):

    df = pd.read_csv(path)

    return df

# ==============================
# PREPROCESS DATA
# ==============================

def preprocess_data(df):

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

    X = df.drop("isFraud", axis=1)

    # TARGET

    y = df["isFraud"]

    # SCALE FEATURES

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # SAVE SCALER

    joblib.dump(
        scaler,
        "model/scaler.pkl"
    )

    return X_scaled, y