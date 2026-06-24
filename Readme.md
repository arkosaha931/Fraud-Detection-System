FraudLens AI

FraudLens AI is an intelligent fraud detection system designed to protect users from fraudulent online transactions including UPI payments, bank transfers, card payments, and cash withdrawals.

The system combines **Deep Learning**, **Anomaly Detection**, and **Behavioral Risk Analysis** to identify suspicious financial transactions in real time and prevent potential financial losses.

---

🚀 Features

* Real-time transaction fraud detection
* Deep Neural Network based classification
* Isolation Forest anomaly detection
* Behavioral risk analysis engine
* Fraud probability scoring
* Automatic transaction blocking
* Suspicious transaction flagging
* MongoDB transaction logging
* Interactive payment simulation dashboard
* Support for:

  * Google Pay
  * PhonePe
  * Paytm
  * Net Banking
  * Card Payments
  * Cash Withdrawals



🧠 AI Architecture

FraudLens uses a hybrid fraud detection approach:

1. Neural Network Model

* Multi-layer Deep Neural Network
* Batch Normalization
* Dropout Regularization
* Binary Focal Loss
* Class imbalance handling

2. Isolation Forest

* Detects unusual transaction patterns
* Identifies anomalies not seen during training

3. Behavioral Risk Engine

Custom fraud indicators including:

* Account Drain Detection
* High Amount Ratio Detection
* Extreme Amount Ratio Detection
* Balance Integrity Checks
* Night Transaction Monitoring
* Device Risk Analysis
* Location Risk Analysis
* Destination Balance Verification



📊 Fraud Scoring Formula

Final Fraud Score is calculated using:

Fraud Score =
(0.65 × Neural Network Score)

* (0.15 × Isolation Forest Score)
* (0.20 × Behavioral Feature Score)

Based on the final score:

| Score Range | Decision   |
| ----------- | ---------- |
| Low Risk    | Safe       |
| Medium Risk | Suspicious |
| High Risk   | Fraud      |


🛠️ Technologies Used

Backend

* Python
* Flask
* Flask-CORS

Machine Learning

* TensorFlow / Keras
* Scikit-Learn
* Isolation Forest
* SMOTE
* NumPy
* Pandas

Database

* MongoDB Atlas
* PyMongo

Frontend

* HTML5
* CSS3
* JavaScript


 📈 Example Workflow

1. User enters transaction details.
2. Frontend sends data to Flask API.
3. FraudLens generates behavioral features.
4. Neural Network predicts fraud probability.
5. Isolation Forest detects anomalies.
6. Risk engine analyzes suspicious patterns.
7. Combined fraud score is generated.
8. Transaction is:

   * Approved
   * Marked Suspicious
   * Blocked as Fraud


🔒 Security Features

* PIN verification
* Balance validation
* Fraud probability thresholding
* Transaction history logging
* Real-time fraud blocking
* Behavioral anomaly detection


 🎯 Future Enhancements

* OTP verification module
* SMS & Email alerts
* User behavioral profiling
* Geo-location fraud detection
* Device fingerprinting
* Real-time streaming analytics
* Mobile application support





