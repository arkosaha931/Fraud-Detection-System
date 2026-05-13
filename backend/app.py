from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.models import load_model

from database.mongo_config import (
    transactions_collection
)

import numpy as np
import joblib

from datetime import datetime

app = Flask(__name__)

CORS(app)

# ==============================
# LOAD TRAINED MODEL
# ==============================

model = load_model(
    "model/fraud_model.h5"
)

# ==============================
# LOAD SCALER + ENCODER
# ==============================

scaler = joblib.load(
    "model/scaler.pkl"
)

encoder = joblib.load(
    "model/label_encoder.pkl"
)

# ==============================
# API ROUTE
# ==============================

@app.route("/predict", methods=["POST"])

def predict():

    data = request.json

    amount = float(
        data.get("amount", 0)
    )

    payment_type = data.get(
        "type",
        "PAYMENT"
    )

    # ==============================
    # ENCODE PAYMENT TYPE
    # ==============================

    try:

        encoded_type = encoder.transform(
            [payment_type]
        )[0]

    except:

        encoded_type = 0

    # ==============================
    # DUMMY VALUES
    # (TEMPORARY)
    # ==============================

    oldbalanceOrg = amount * 2
    newbalanceOrig = oldbalanceOrg - amount

    oldbalanceDest = 0
    newbalanceDest = amount

    # ==============================
    # PREPARE INPUT
    # ==============================

    input_data = np.array([[
        encoded_type,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    ]])

    # ==============================
    # SCALE INPUT
    # ==============================

    input_scaled = scaler.transform(
        input_data
    )

    # ==============================
    # AI PREDICTION
    # ==============================

    prediction = model.predict(
        input_scaled
    )[0][0]

    fraud_probability = round(
        float(prediction),
        2
    )

    # ==============================
    # DECISION ENGINE
    # ==============================

    if fraud_probability < 0.4:

        status = "safe"

    elif fraud_probability < 0.8:

        status = "suspicious"

    else:

        status = "fraud"

    # ==============================
    # STORE IN MONGODB
    # ==============================

    transaction_data = {

        "amount": amount,

        "type": payment_type,

        "fraud_probability":
            fraud_probability,

        "status": status,

        "timestamp":
            datetime.now()
    }

    transactions_collection.insert_one(
        transaction_data
    )

    # ==============================
    # RETURN RESPONSE
    # ==============================

    return jsonify({

        "fraud_probability":
            fraud_probability,

        "status":
            status
    })


# ==============================
# RUN FLASK
# ==============================

if __name__ == "__main__":

    app.run(debug=True)