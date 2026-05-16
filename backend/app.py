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

# ==============================
# LOAD MODEL
# ==============================

model = load_model(
    "model/fraud_model.keras"
)

# ==============================
# LOAD SCALER
# ==============================

scaler = joblib.load(
    "model/scaler.pkl"
)

# ==============================
# LOAD ENCODER
# ==============================

encoder = joblib.load(
    "model/label_encoder.pkl"
)

# ==============================
# LOAD FEATURE COLUMNS
# ==============================

feature_columns = joblib.load(
    "model/feature_columns.pkl"
)

# ==============================
# API
# ==============================

@app.route("/predict", methods=["POST"])

def predict():

    try:

        data = request.json

        sender_account = data.get(
            "sender_account"
        )

        receiver_account = data.get(
            "receiver_account"
        )

        pin = data.get("pin")

        amount = float(
            data.get("amount", 0)
        )

        payment_type = data.get(
            "type",
            "PAYMENT"
        )

        location = data.get(
            "location",
            "Unknown"
        )

        device_info = data.get(
            "device_info",
            "Unknown Device"
        )

        # ==============================
        # FETCH ACCOUNTS
        # ==============================

        sender = accounts_collection.find_one({

            "account_number":
                sender_account
        })

        receiver = accounts_collection.find_one({

            "account_number":
                receiver_account
        })

        if not sender:

            return jsonify({

                "status": "failed",

                "message":
                    "Sender account not found"
            })

        if not receiver:

            return jsonify({

                "status": "failed",

                "message":
                    "Receiver account not found"
            })

        if sender["pin"] != pin:

            return jsonify({

                "status": "failed",

                "message":
                    "Invalid PIN"
            })

        # ==============================
        # BALANCES
        # ==============================

        oldbalanceOrg = sender["balance"]

        oldbalanceDest = receiver["balance"]

        if amount > oldbalanceOrg:

            return jsonify({

                "status": "failed",

                "message":
                    "Insufficient Balance"
            })

        newbalanceOrig = (
            oldbalanceOrg - amount
        )

        newbalanceDest = (
            oldbalanceDest + amount
        )

        # ==============================
        # ENCODE TYPE
        # ==============================

        try:

            encoded_type = encoder.transform(
                [payment_type]
            )[0]

        except:

            encoded_type = 0

        # ==============================
        # FEATURES
        # ==============================

        step = datetime.now().hour

        transaction_hour = (
            datetime.now().hour
        )

        recent_transactions = (

            transactions_collection.count_documents({

                "sender_account":
                    sender_account
            })

        )

        transaction_frequency = (
            recent_transactions + 1
        )

        amount_ratio = (

            amount

            /

            (oldbalanceOrg + 1)

        )

        # ==============================
        # DEVICE RISK
        # ==============================

        device_risk = 0

        risky_devices = [

            "linux",
            "unknown",
            "rooted"
        ]

        for risky in risky_devices:

            if risky in device_info.lower():

                device_risk = 1

        # ==============================
        # LOCATION RISK
        # ==============================

        location_risk = 0

        risky_locations = [

            "foreign",
            "unknown"
        ]

        for risky in risky_locations:

            if risky in location.lower():

                location_risk = 1

        isFlaggedFraud = 0

        # ==============================
        # DATAFRAME INPUT
        # ==============================

        input_data = pd.DataFrame([{

            "step":
                step,

            "type":
                encoded_type,

            "amount":
                amount,

            "oldbalanceOrg":
                oldbalanceOrg,

            "newbalanceOrig":
                newbalanceOrig,

            "oldbalanceDest":
                oldbalanceDest,

            "newbalanceDest":
                newbalanceDest,

            "isFlaggedFraud":
                isFlaggedFraud,

            "transaction_hour":
                transaction_hour,

            "transaction_frequency":
                transaction_frequency,

            "device_risk":
                device_risk,

            "location_risk":
                location_risk,

            "amount_ratio":
                amount_ratio
        }])

        # COLUMN ORDER FIX

        input_data = input_data[
            feature_columns
        ]

        # ==============================
        # SCALE
        # ==============================

        input_scaled = scaler.transform(
            input_data
        )

        # ==============================
        # PREDICT
        # ==============================

        prediction = model.predict(
            input_scaled,
            verbose=0
        )[0][0]

        fraud_probability = float(
            prediction
        )

        # ==============================
        # RULE BOOSTING
        # ==============================

        if amount > 10000:

            fraud_probability += 0.15

        if transaction_frequency > 5:

            fraud_probability += 0.10

        if device_risk == 1:

            fraud_probability += 0.10

        if location_risk == 1:

            fraud_probability += 0.10

        fraud_probability = min(
            fraud_probability,
            1.0
        )

        fraud_probability = round(
            fraud_probability,
            2
        )

        # ==============================
        # DECISION
        # ==============================

        if fraud_probability < 0.30:

            status = "safe"

        elif fraud_probability < 0.70:

            status = "suspicious"

        else:

            status = "fraud"

        # ==============================
        # UPDATE BALANCES
        # ==============================

        if status == "safe":

            accounts_collection.update_one(

                {
                    "account_number":
                        sender_account
                },

                {
                    "$set": {
                        "balance":
                            newbalanceOrig
                    }
                }
            )

            accounts_collection.update_one(

                {
                    "account_number":
                        receiver_account
                },

                {
                    "$set": {
                        "balance":
                            newbalanceDest
                    }
                }
            )

        # ==============================
        # SAVE TRANSACTION
        # ==============================

        transaction_data = {

            "sender_account":
                sender_account,

            "receiver_account":
                receiver_account,

            "amount":
                amount,

            "type":
                payment_type,

            "location":
                location,

            "device_info":
                device_info,

            "transaction_hour":
                transaction_hour,

            "transaction_frequency":
                transaction_frequency,

            "device_risk":
                device_risk,

            "location_risk":
                location_risk,

            "amount_ratio":
                amount_ratio,

            "fraud_probability":
                fraud_probability,

            "status":
                status,

            "timestamp":
                datetime.now()
        }

        transactions_collection.insert_one(
            transaction_data
        )

        # ==============================
        # RESPONSE
        # ==============================

        return jsonify({

            "fraud_probability":
                fraud_probability,

            "status":
                status,

            "sender_balance":
                newbalanceOrig,

            "transaction_frequency":
                transaction_frequency
        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)
        })


# ==============================
# RUN
# ==============================

if __name__ == "__main__":

    app.run(debug=True)