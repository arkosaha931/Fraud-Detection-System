from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.models import load_model

from database.mongo_config import (
    transactions_collection,
    accounts_collection
)

import numpy as np
import joblib

from datetime import datetime

app = Flask(__name__)

CORS(app)

# LOAD MODEL

model = load_model(
    "model/fraud_model.h5"
)

# LOAD SCALER

scaler = joblib.load(
    "model/scaler.pkl"
)

# LOAD ENCODER

encoder = joblib.load(
    "model/label_encoder.pkl"
)

# API

@app.route("/predict", methods=["POST"])

def predict():

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

    # FETCH ACCOUNTS

    sender = accounts_collection.find_one({

        "account_number":
            sender_account
    })

    receiver = accounts_collection.find_one({

        "account_number":
            receiver_account
    })

    # VALIDATION

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

    # PIN CHECK

    if sender["pin"] != pin:

        return jsonify({

            "status": "failed",

            "message":
                "Invalid PIN"
        })

    # BALANCES

    oldbalanceOrg = sender["balance"]

    oldbalanceDest = receiver["balance"]

   

    if amount > oldbalanceOrg:

        return jsonify({

            "status": "failed",

            "message":
                "Insufficient Balance"
        })


    newbalanceOrig = oldbalanceOrg - amount

    newbalanceDest = oldbalanceDest + amount

    # ENCODE PAYMENT TYPE

    try:

        encoded_type = encoder.transform(
            [payment_type]
        )[0]

    except:

        encoded_type = 0

    # ANN INPUT

    step = 1

    isFlaggedFraud = 0

    input_data = np.array([[

        step,

        encoded_type,

        amount,

        oldbalanceOrg,

        newbalanceOrig,

        oldbalanceDest,

        newbalanceDest,

        isFlaggedFraud
    ]])

    

    input_scaled = scaler.transform(
        input_data
    )

    # PREDICT

    prediction = model.predict(
        input_scaled
    )[0][0]

    fraud_probability = round(
        float(prediction),
        2
    )

    

    if fraud_probability < 0.4:

        status = "safe"

    elif fraud_probability < 0.8:

        status = "suspicious"

    else:

        status = "fraud"

   

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

    # STORE TRANSACTION

    transaction_data = {

        "sender_account":
            sender_account,

        "receiver_account":
            receiver_account,

        "amount":
            amount,

        "type":
            payment_type,

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

    return jsonify({

        "fraud_probability":
            fraud_probability,

        "status":
            status,

        "sender_balance":
            newbalanceOrig
    })

if __name__ == "__main__":

    app.run(debug=True)