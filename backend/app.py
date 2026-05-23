from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.models import load_model

# mongo_config.py is inside database/
from database.mongo_config import (
    transactions_collection,
    accounts_collection
)

import pandas as pd
import numpy as np
import joblib

from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ==============================
# PAYMENT TYPE MAPPING
# Frontend types → PaySim trained types
# ==============================

PAYMENT_TYPE_MAP = {
    "Google Pay"   : "TRANSFER",
    "Net Banking"  : "PAYMENT",
    "Cash Withdraw": "CASH_OUT",
    "Card"         : "PAYMENT",
    "TRANSFER"     : "TRANSFER",
    "PAYMENT"      : "PAYMENT",
    "CASH_OUT"     : "CASH_OUT",
    "CASH_IN"      : "CASH_IN",
    "DEBIT"        : "DEBIT"
}

# ==============================
# LOAD ALL MODEL ARTIFACTS
# ==============================

print("⏳ Loading model artifacts...")

model            = load_model("model/fraud_model.keras")
scaler           = joblib.load("model/scaler.pkl")
encoder          = joblib.load("model/label_encoder.pkl")
feature_columns  = joblib.load("model/feature_columns.pkl")
isolation_forest = joblib.load("model/isolation_forest.pkl")
iso_norm_params  = joblib.load("model/iso_norm_params.pkl")

# Best threshold saved from training (0.7541)
# We use it only for the NN score check,
# NOT for the final 3-zone decision
best_threshold   = joblib.load("model/best_threshold.pkl")

print("✅ All model artifacts loaded.")
print(f"   Features  : {len(feature_columns)}")
print(f"   Threshold : {best_threshold:.4f}")

# ==============================
# WEIGHTS (must match training)
# ==============================

NN_WEIGHT  = 0.70
ISO_WEIGHT = 0.30

# ==============================
# DECISION THRESHOLDS
# These are tuned for real simulator use.
# PaySim best_threshold (0.75) was calibrated
# on bulk dataset fraud patterns — too high
# for single live transactions.
# These zones give realistic safe/suspicious/fraud
# ==============================

SAFE_THRESHOLD       = 0.30   # below this → safe
SUSPICIOUS_THRESHOLD = 0.55   # below this → suspicious, above → fraud


# ==============================
# HELPER: COMPUTE REAL FEATURES
# ==============================

def compute_features(
    sender_account,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    encoded_type,
    step,
    transaction_hour
):
    """
    Compute ALL features exactly as done in preprocess.py
    so inference matches training perfectly.
    """

    # TEMPORAL
    is_night_transaction = int(
        transaction_hour >= 22 or transaction_hour <= 6
    )

    # BALANCE FEATURES
    amount_ratio = amount / (oldbalanceOrg + 1)

    # Did balances change correctly?
    # In fraud: attacker manipulates balances externally
    # so the difference won't be zero
    balance_diff_orig = newbalanceOrig - (oldbalanceOrg - amount)
    balance_diff_dest = newbalanceDest - (oldbalanceDest + amount)

    orig_balance_mismatch = int(abs(balance_diff_orig) > 1)
    dest_balance_mismatch = int(abs(balance_diff_dest) > 1)

    zero_balance_orig = int(oldbalanceOrg == 0)

    # Fully drained account = strong fraud signal
    account_drained = int(
        newbalanceOrig == 0 and oldbalanceOrg > 0
    )

    # TRANSACTION FREQUENCY (last 24 hours only)
    last_24h = datetime.now() - timedelta(hours=24)

    recent_count = transactions_collection.count_documents({
        "sender_account": sender_account,
        "timestamp"     : {"$gte": last_24h}
    })

    transaction_frequency = recent_count + 1

    # DEVICE RISK
    # Matches training: TRANSFER/CASH_OUT + high amount
    AMOUNT_75TH_PERCENTILE = 300000.0

    try:
        type_name = encoder.inverse_transform([encoded_type])[0]
    except Exception:
        type_name = "PAYMENT"

    device_risk = int(
        type_name in ["TRANSFER", "CASH_OUT"] and
        amount > AMOUNT_75TH_PERCENTILE
    )

    # LOCATION RISK
    # Matches training: night transaction + above-median amount
    AMOUNT_MEDIAN = 74871.0

    location_risk = int(
        is_night_transaction == 1 and
        amount > AMOUNT_MEDIAN
    )

    # LOG TRANSFORMS (matches training)
    log_amount         = np.log1p(amount)
    log_oldbalanceOrg  = np.log1p(oldbalanceOrg)
    log_newbalanceOrig = np.log1p(newbalanceOrig)

    features = {
        "step"                  : step,
        "type"                  : encoded_type,
        "amount"                : amount,
        "oldbalanceOrg"         : oldbalanceOrg,
        "newbalanceOrig"        : newbalanceOrig,
        "oldbalanceDest"        : oldbalanceDest,
        "newbalanceDest"        : newbalanceDest,
        "isFlaggedFraud"        : 0,
        "transaction_hour"      : transaction_hour,
        "is_night_transaction"  : is_night_transaction,
        "amount_ratio"          : amount_ratio,
        "orig_balance_mismatch" : orig_balance_mismatch,
        "dest_balance_mismatch" : dest_balance_mismatch,
        "zero_balance_orig"     : zero_balance_orig,
        "account_drained"       : account_drained,
        "transaction_frequency" : transaction_frequency,
        "device_risk"           : device_risk,
        "location_risk"         : location_risk,
        "log_amount"            : log_amount,
        "log_oldbalanceOrg"     : log_oldbalanceOrg,
        "log_newbalanceOrig"    : log_newbalanceOrig
    }

    return features, transaction_frequency


# ==============================
# API: PREDICT
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        # ==============================
        # PARSE INPUT
        # ==============================

        sender_account   = data.get("sender_account")
        receiver_account = data.get("receiver_account")
        pin              = data.get("pin")
        amount           = float(data.get("amount", 0))
        payment_type_raw = data.get("type", "PAYMENT")
        location         = data.get("location", "Unknown")
        device_info      = data.get("device_info", "Unknown Device")

        # ==============================
        # FETCH ACCOUNTS FROM MONGO
        # ==============================

        sender = accounts_collection.find_one({
            "account_number": sender_account
        })

        receiver = accounts_collection.find_one({
            "account_number": receiver_account
        })

        if not sender:
            return jsonify({
                "status" : "failed",
                "message": "Sender account not found"
            })

        if not receiver:
            return jsonify({
                "status" : "failed",
                "message": "Receiver account not found"
            })

        if sender["pin"] != pin:
            return jsonify({
                "status" : "failed",
                "message": "Invalid PIN"
            })

        # ==============================
        # BALANCES
        # ==============================

        oldbalanceOrg  = float(sender["balance"])
        oldbalanceDest = float(receiver["balance"])

        if amount > oldbalanceOrg:
            return jsonify({
                "status" : "failed",
                "message": "Insufficient Balance"
            })

        newbalanceOrig = oldbalanceOrg  - amount
        newbalanceDest = oldbalanceDest + amount

        # ==============================
        # ENCODE PAYMENT TYPE
        # ==============================

        payment_type_mapped = PAYMENT_TYPE_MAP.get(
            payment_type_raw, "PAYMENT"
        )

        try:
            encoded_type = int(
                encoder.transform([payment_type_mapped])[0]
            )
        except Exception:
            encoded_type = int(
                encoder.transform(["PAYMENT"])[0]
            )

        # ==============================
        # TEMPORAL
        # ==============================

        now              = datetime.now()
        step             = now.hour
        transaction_hour = now.hour

        # ==============================
        # COMPUTE ALL FEATURES
        # ==============================

        features_dict, transaction_frequency = compute_features(
            sender_account   = sender_account,
            amount           = amount,
            oldbalanceOrg    = oldbalanceOrg,
            newbalanceOrig   = newbalanceOrig,
            oldbalanceDest   = oldbalanceDest,
            newbalanceDest   = newbalanceDest,
            encoded_type     = encoded_type,
            step             = step,
            transaction_hour = transaction_hour
        )

        # ==============================
        # BUILD INPUT DATAFRAME
        # ==============================

        input_df = pd.DataFrame([features_dict])

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # ==============================
        # SCALE
        # ==============================

        input_scaled = scaler.transform(input_df)

        # ==============================
        # NEURAL NETWORK SCORE
        # ==============================

        nn_prob = float(
            model.predict(input_scaled, verbose=0)[0][0]
        )

        # ==============================
        # ISOLATION FOREST SCORE
        # ==============================

        iso_raw = float(
            isolation_forest.score_samples(input_scaled)[0]
        )

        iso_min = float(iso_norm_params["min"])
        iso_max = float(iso_norm_params["max"])

        iso_score = float(
            1.0 - ((iso_raw - iso_min) / (iso_max - iso_min + 1e-9))
        )
        iso_score = float(np.clip(iso_score, 0.0, 1.0))

        # ==============================
        # COMBINED ML SCORE
        # Pure ML — no manual rule boosting
        # ==============================

        fraud_probability = float(
            (NN_WEIGHT * nn_prob) +
            (ISO_WEIGHT * iso_score)
        )
        fraud_probability = float(np.clip(fraud_probability, 0.0, 1.0))
        fraud_probability = round(fraud_probability, 4)

        # ==============================
        # 3-ZONE DECISION
        # Uses tuned live thresholds —
        # NOT the dataset best_threshold
        # ==============================

        if fraud_probability < SAFE_THRESHOLD:
            status = "safe"

        elif fraud_probability < SUSPICIOUS_THRESHOLD:
            status = "suspicious"

        else:
            status = "fraud"

        # ==============================
        # UPDATE BALANCES IN MONGO
        # Only deduct money if safe
        # ==============================

        if status == "safe":

            accounts_collection.update_one(
                {"account_number": sender_account},
                {"$set": {"balance": newbalanceOrig}}
            )

            accounts_collection.update_one(
                {"account_number": receiver_account},
                {"$set": {"balance": newbalanceDest}}
            )

        # ==============================
        # SAVE FULL TRANSACTION TO MONGO
        # ==============================

        transaction_doc = {
            "sender_account"       : sender_account,
            "receiver_account"     : receiver_account,
            "amount"               : amount,
            "type"                 : payment_type_raw,
            "type_mapped"          : payment_type_mapped,
            "location"             : location,
            "device_info"          : device_info,
            "transaction_hour"     : transaction_hour,
            "is_night_transaction" : features_dict["is_night_transaction"],
            "transaction_frequency": transaction_frequency,
            "device_risk"          : features_dict["device_risk"],
            "location_risk"        : features_dict["location_risk"],
            "orig_balance_mismatch": features_dict["orig_balance_mismatch"],
            "dest_balance_mismatch": features_dict["dest_balance_mismatch"],
            "account_drained"      : features_dict["account_drained"],
            "amount_ratio"         : round(features_dict["amount_ratio"], 6),
            "nn_probability"       : round(nn_prob, 4),
            "iso_score"            : round(iso_score, 4),
            "fraud_probability"    : fraud_probability,
            "status"               : status,
            "timestamp"            : now
        }

        result = transactions_collection.insert_one(transaction_doc)

        # Confirm insert happened
        print(f"✅ Transaction saved → {result.inserted_id} | status={status} | fraud_prob={fraud_probability}")

        # ==============================
        # RESPONSE TO FRONTEND
        # ==============================

        return jsonify({
            "fraud_probability"    : fraud_probability,
            "nn_probability"       : round(nn_prob, 4),
            "iso_score"            : round(iso_score, 4),
            "status"               : status,
            "sender_balance"       : newbalanceOrig,
            "transaction_frequency": transaction_frequency
        })

    except Exception as e:

        import traceback
        print("❌ ERROR:", traceback.format_exc())

        return jsonify({
            "status" : "error",
            "message": str(e)
        })


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    app.run(debug=True)