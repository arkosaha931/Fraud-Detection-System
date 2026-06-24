from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.models import load_model

from database.mongo_config import (
    transactions_collection,
    accounts_collection
)

import pandas as pd
import numpy as np
import joblib

from datetime import datetime


app = Flask(__name__)
CORS(app)


PAYMENT_TYPE_MAP = {
    "Google Pay":    "TRANSFER",
    "PhonePe":       "TRANSFER",
    "Paytm":         "TRANSFER",
    "Net Banking":   "PAYMENT",
    "Cash Withdraw": "CASH_OUT",
    "Card":          "PAYMENT",
    "TRANSFER":      "TRANSFER",
    "PAYMENT":       "PAYMENT",
    "CASH_OUT":      "CASH_OUT",
    "CASH_IN":       "CASH_IN",
    "DEBIT":         "DEBIT"
}

PAYSIM_STEP_MAX = 744


print("\nLoading Fraud Detection Models...")

model = load_model(
    "model/fraud_model.keras",
    compile=False
)

scaler = joblib.load(
    "model/scaler.pkl"
)

encoder = joblib.load(
    "model/label_encoder.pkl"
)

feature_columns = joblib.load(
    "model/feature_columns.pkl"
)

isolation_forest = joblib.load(
    "model/isolation_forest.pkl"
)

iso_norm_params = joblib.load(
    "model/iso_norm_params.pkl"
)

best_threshold = joblib.load(
    "model/best_threshold.pkl"
)

print(f"✅ Models Loaded Successfully")
print(f"   Best threshold : {best_threshold:.4f}")

NN_WEIGHT      = 0.65
ISO_WEIGHT     = 0.15
FEATURE_WEIGHT = 0.20

SAFE_ZONE       = max(0.30, best_threshold * 0.65)
SUSPICIOUS_ZONE = min(0.85, best_threshold * 1.55)

print(f"   Safe zone       : < {SAFE_ZONE:.4f}")
print(f"   Suspicious zone : {SAFE_ZONE:.4f} – {SUSPICIOUS_ZONE:.4f}")
print(f"   Fraud zone      : > {SUSPICIOUS_ZONE:.4f}")




def compute_feature_score(features):

    score = 0.0

    if features["account_drained"] == 1:
        score += 0.35

    if features["extreme_amount_ratio"] == 1:
        score += 0.20
    elif features["high_amount_ratio"] == 1:
        score += 0.12

    if features["orig_balance_mismatch"] == 1:
        score += 0.15

    if features["dest_balance_mismatch"] == 1:
        score += 0.12

    if features["dest_no_increase"] == 1:
        score += 0.12

    if features["device_risk"] == 1:
        score += 0.12

    if features["location_risk"] == 1:
        score += 0.10

    return float(np.clip(score, 0.0, 1.0))




def analyze_risk_flags(features, amount, oldbalanceOrg):

    flags  = {}
    detail = {}
    r = features["amount_ratio"]

    if features["account_drained"] == 1:
        flags["account_drained"] = True
        detail["account_drained"] = "Full account drain"

    if features["extreme_amount_ratio"] == 1:
        flags["extreme_amount_ratio"] = True
        detail["extreme_amount_ratio"] = f"Extreme ratio {r:.2f} — >95% of balance"
    elif features["high_amount_ratio"] == 1:
        flags["high_amount_ratio"] = True
        detail["high_amount_ratio"] = f"High ratio {r:.2f} — >75% of balance"

    if features["orig_balance_mismatch"] == 1:
        flags["orig_balance_mismatch"] = True
        detail["orig_balance_mismatch"] = "Sender balance does not reconcile"

    if features["dest_balance_mismatch"] == 1:
        flags["dest_balance_mismatch"] = True
        detail["dest_balance_mismatch"] = "Receiver balance does not reconcile"

    if features["dest_no_increase"] == 1:
        flags["dest_no_increase"] = True
        detail["dest_no_increase"] = "Receiver balance did not increase after transfer"

    if features["device_risk"] == 1:
        flags["device_risk"] = True
        detail["device_risk"] = "High-value TRANSFER/CASH_OUT on risky channel"

    if features["location_risk"] == 1:
        flags["location_risk"] = True
        detail["location_risk"] = "Night-time transaction with high amount ratio"

    if features["is_night_transaction"] == 1:
        flags["is_night_transaction"] = True
        detail["is_night_transaction"] = "Transaction between 10 PM – 6 AM"

    if amount >= 50000 and r >= 0.40:
        flags["large_partial_drain"] = True
        detail["large_partial_drain"] = (
            f"Large amount Rs{amount:,.0f} with {r*100:.1f}% of balance"
        )

    if amount >= 100000:
        flags["very_large_amount"] = True
        detail["very_large_amount"] = f"Very large transaction: Rs{amount:,.0f}"

    return flags, detail




def compute_behavioral_floor(flags, features):

    floor = 0.0

    drain_signals = sum([
        flags.get("account_drained",      False),
        flags.get("extreme_amount_ratio", False),
        flags.get("high_amount_ratio",    False),
    ])

    integrity_signals = sum([
        flags.get("orig_balance_mismatch", False),
        flags.get("dest_balance_mismatch", False),
        flags.get("dest_no_increase",      False),
    ])

    behavioral_signals = sum([
        flags.get("device_risk",          False),
        flags.get("location_risk",        False),
        flags.get("is_night_transaction", False),
        flags.get("large_partial_drain",  False),
        flags.get("very_large_amount",    False),
    ])

    total = drain_signals + integrity_signals + behavioral_signals

    
    if total >= 3:
        floor = max(floor, 0.42)

    if total >= 4:
        floor = max(floor, 0.56)

   
    if drain_signals >= 1 and behavioral_signals >= 1:
        floor = max(floor, 0.52)

    
    if features["extreme_amount_ratio"] == 1 and features["account_drained"] == 0:
        floor = max(floor, 0.44)

    
    if flags.get("large_partial_drain") and flags.get("is_night_transaction"):
        floor = max(floor, 0.48)

    # Balance integrity failures
    if integrity_signals >= 2:
        floor = max(floor, 0.54)

    if integrity_signals >= 1 and drain_signals >= 1:
        floor = max(floor, 0.62)

    
    if features["account_drained"] == 1:
        floor = max(floor, SUSPICIOUS_ZONE + 0.05)

    return float(floor)



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

    is_night_transaction = int(
        transaction_hour >= 22 or transaction_hour <= 6
    )

    amount_ratio         = amount / (oldbalanceOrg + 1)
    high_amount_ratio    = int(amount_ratio > 0.75)
    extreme_amount_ratio = int(amount_ratio > 0.95)

    log_amount         = np.log1p(amount)
    log_oldbalanceOrg  = np.log1p(oldbalanceOrg)
    log_newbalanceOrig = np.log1p(newbalanceOrig)

    orig_balance_mismatch = int(
        abs(newbalanceOrig - (oldbalanceOrg - amount)) > 1
    )

    dest_balance_mismatch = int(
        abs(newbalanceDest - (oldbalanceDest + amount)) > 1
    )

    zero_balance_orig = int(oldbalanceOrg == 0)

    account_drained = int(
        newbalanceOrig <= 0 and oldbalanceOrg > 0
    )

    zero_balance_dest = int(oldbalanceDest == 0)

    dest_no_increase = int(newbalanceDest <= oldbalanceDest)

    try:
        type_name = encoder.inverse_transform([encoded_type])[0]
    except Exception:
        type_name = "PAYMENT"

    device_risk = int(
        type_name in ["TRANSFER", "CASH_OUT"] and
        amount_ratio > 0.65
    )

    location_risk = int(
        is_night_transaction == 1 and
        amount_ratio > 0.45
    )

    features = {
        "step":                   step,
        "type":                   encoded_type,
        "amount":                 amount,
        "oldbalanceOrg":          oldbalanceOrg,
        "newbalanceOrig":         newbalanceOrig,
        "oldbalanceDest":         oldbalanceDest,
        "newbalanceDest":         newbalanceDest,
        "isFlaggedFraud":         0,
        "transaction_hour":       transaction_hour,
        "is_night_transaction":   is_night_transaction,
        "amount_ratio":           amount_ratio,
        "high_amount_ratio":      high_amount_ratio,
        "extreme_amount_ratio":   extreme_amount_ratio,
        "log_amount":             log_amount,
        "log_oldbalanceOrg":      log_oldbalanceOrg,
        "log_newbalanceOrig":     log_newbalanceOrig,
        "orig_balance_mismatch":  orig_balance_mismatch,
        "dest_balance_mismatch":  dest_balance_mismatch,
        "zero_balance_orig":      zero_balance_orig,
        "account_drained":        account_drained,
        "zero_balance_dest":      zero_balance_dest,
        "dest_no_increase":       dest_no_increase,
        "device_risk":            device_risk,
        "location_risk":          location_risk,
    }

    return features




@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":  "running",
        "message": "Fraud Detection API Running"
    })



@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        sender_account   = data.get("sender_account")
        receiver_account = data.get("receiver_account")
        pin              = data.get("pin")
        amount           = float(data.get("amount", 0))
        payment_type_raw = data.get("type", "PAYMENT")

        # ---- FETCH ACCOUNTS ----

        sender = accounts_collection.find_one({
            "account_number": sender_account
        })

        receiver = accounts_collection.find_one({
            "account_number": receiver_account
        })

        if not sender:
            return jsonify({
                "status":  "failed",
                "message": "Sender account not found"
            })

        if not receiver:
            return jsonify({
                "status":  "failed",
                "message": "Receiver account not found"
            })

        # ---- PIN VALIDATION ----

        if str(sender["pin"]) != str(pin):
            return jsonify({
                "status":  "failed",
                "message": "Invalid PIN"
            })

        oldbalanceOrg  = float(sender["balance"])
        oldbalanceDest = float(receiver["balance"])

        # ---- AMOUNT VALIDATION ----

        if amount <= 0:
            return jsonify({
                "status":  "failed",
                "message": "Invalid Amount"
            })

        if amount > oldbalanceOrg:
            return jsonify({
                "status":  "failed",
                "message": "Insufficient Balance",
                "balance": oldbalanceOrg
            })

        # ---- UPDATED BALANCES ----

        newbalanceOrig = oldbalanceOrg  - amount
        newbalanceDest = oldbalanceDest + amount

        # ---- PAYMENT TYPE ENCODING ----

        payment_type_mapped = PAYMENT_TYPE_MAP.get(
            payment_type_raw, "PAYMENT"
        )

        encoded_type = int(
            encoder.transform([payment_type_mapped])[0]
        )

       

        now              = datetime.now()
        step             = int((now.timestamp() // 3600) % PAYSIM_STEP_MAX) + 1
        transaction_hour = now.hour

       

        features_dict = compute_features(
            sender_account,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
            encoded_type,
            step,
            transaction_hour
        )

        # ---- BUILD INPUT DATAFRAME ----

        input_df = pd.DataFrame([features_dict])

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        input_scaled = scaler.transform(input_df)

        # ---- NEURAL NETWORK SCORE ----

        nn_raw   = float(model.predict(input_scaled, verbose=0)[0][0])
        nn_score = float(np.clip(nn_raw, 0.0, 1.0))

        # ---- ISOLATION FOREST SCORE ----

        iso_raw   = float(isolation_forest.score_samples(input_scaled)[0])
        iso_min   = float(iso_norm_params["min"])
        iso_max   = float(iso_norm_params["max"])

        iso_score = float(
            np.clip(
                1.0 - ((iso_raw - iso_min) / (iso_max - iso_min + 1e-9)),
                0.0,
                1.0
            )
        )

        # ---- FEATURE SCORE ----

        feature_score = compute_feature_score(features_dict)

        # ---- RISK FLAGS ----

        risk_flags, risk_detail = analyze_risk_flags(
            features_dict, amount, oldbalanceOrg
        )

        # ---- BEHAVIORAL FLOOR ----

        behavioral_floor = compute_behavioral_floor(risk_flags, features_dict)

       

        raw_score = float(np.clip(
            (NN_WEIGHT      * nn_score)  +
            (ISO_WEIGHT     * iso_score) +
            (FEATURE_WEIGHT * feature_score),
            0.0,
            1.0
        ))

        fraud_probability = round(max(raw_score, behavioral_floor), 4)

        # ---- DECISION ENGINE ----

        if fraud_probability < SAFE_ZONE:
            status = "safe"
        elif fraud_probability < SUSPICIOUS_ZONE:
            status = "suspicious"
        else:
            status = "fraud"

        # ---- CONSOLE LOG ----

        active_flags = [k for k, v in risk_flags.items() if v]

        print("\n" + "="*60)
        print("TRANSACTION ANALYSIS")
        print("="*60)
        print(f"amount                : {amount}")
        print(f"sender old balance    : {oldbalanceOrg}")
        print(f"sender new balance    : {newbalanceOrig}")
        print(f"receiver old balance  : {oldbalanceDest}")
        print(f"receiver new balance  : {newbalanceDest}")
        print(f"amount_ratio          : {features_dict['amount_ratio']:.4f}")
        print(f"step (mapped)         : {step}")
        print(f"transaction_hour      : {transaction_hour}")
        print(f"--- Risk Flags ---")
        for flag in active_flags:
            print(f"  ✦ {flag:28s} : {risk_detail[flag]}")
        if not active_flags:
            print("  (none)")
        print(f"--- Scores ---")
        print(f"  nn_raw              : {nn_raw:.8f}")
        print(f"  nn_score            : {nn_score:.4f}")
        print(f"  iso_score           : {iso_score:.4f}")
        print(f"  feature_score       : {feature_score:.4f}")
        print(f"  raw_score           : {raw_score:.4f}")
        print(f"  behavioral_floor    : {behavioral_floor:.4f}")
        print(f"  fraud_probability   : {fraud_probability}")
        print(f"--- Decision ---")
        print(f"  best_threshold      : {best_threshold:.4f}")
        print(f"  safe_zone           : {SAFE_ZONE:.4f}")
        print(f"  suspicious_zone     : {SUSPICIOUS_ZONE:.4f}")
        print(f"  status              : {status}")
        print("="*60)

       

        if status == "fraud":

            print("❌ FRAUD TRANSACTION BLOCKED\n")

            transactions_collection.insert_one({
                "sender_account":    sender_account,
                "receiver_account":  receiver_account,
                "amount":            amount,
                "type":              payment_type_raw,
                "fraud_probability": fraud_probability,
                "risk_flags":        active_flags,
                "status":            "fraud",
                "timestamp":         datetime.now()
            })

            return jsonify({
                "status":            "fraud",
                "message":           "Fraudulent transaction blocked",
                "fraud_probability": fraud_probability,
                "nn_score":          round(nn_score, 6),
                "iso_score":         round(iso_score, 4),
                "feature_score":     round(feature_score, 4),
                "risk_flags":        active_flags,
                "balance":           round(oldbalanceOrg, 2)
            })

        
        accounts_collection.update_one(
            {"account_number": sender_account},
            {"$set": {"balance": round(newbalanceOrig, 2)}}
        )

        accounts_collection.update_one(
            {"account_number": receiver_account},
            {"$set": {"balance": round(newbalanceDest, 2)}}
        )

        
        transactions_collection.insert_one({
            "sender_account":    sender_account,
            "receiver_account":  receiver_account,
            "amount":            amount,
            "type":              payment_type_raw,
            "fraud_probability": fraud_probability,
            "risk_flags":        active_flags,
            "status":            status,
            "timestamp":         datetime.now()
        })

        

        return jsonify({
            "status":
                status,
            "message":
                "Transaction Successful"
                if status == "safe"
                else "Suspicious Transaction — OTP required",
            "fraud_probability":
                fraud_probability,
            "nn_score":
                round(nn_score, 6),
            "iso_score":
                round(iso_score, 4),
            "feature_score":
                round(feature_score, 4),
            "risk_flags":
                active_flags,
            "balance":
                round(newbalanceOrig, 2),
            "receiver_balance":
                round(newbalanceDest, 2)
        })

    except Exception as e:

        import traceback

        print("\nERROR OCCURRED:")
        print(traceback.format_exc())

        return jsonify({
            "status":  "error",
            "message": str(e)
        })




if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )