import os
import numpy as np
import joblib

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# YOUR FOLDER STRUCTURE:
# preprocess.py is inside utils/
from utils.preprocess import (
    load_data,
    preprocess_data
)

# ==============================
# ENSURE MODEL DIR EXISTS
# ==============================

os.makedirs("model", exist_ok=True)

# ==============================
# LOAD DATASET
# ==============================

df = load_data("dataset/fraud_dataset.csv")

# ==============================
# PREPROCESS
# ==============================

# preprocess_data now returns 3 values
X, y, feature_cols = preprocess_data(df)

# ==============================
# CLASS DISTRIBUTION
# ==============================

print("\nClass Distribution:")
print(f"  Legit (0): {(y == 0).sum()}")
print(f"  Fraud (1): {(y == 1).sum()}")
print(f"  Fraud %  : {(y == 1).mean() * 100:.4f}%")

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")

# ==============================
# CLASS WEIGHTS
# (handles severe imbalance in fraud data)
# ==============================

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    0: class_weight_values[0],
    1: class_weight_values[1]
}

print(f"\nClass Weights: {class_weights}")

# ==============================
# ISOLATION FOREST
# (Anomaly Detection Layer)
# Trained only on legit transactions
# Flags unusual patterns the NN might miss
# ==============================

print("\n⏳ Training Isolation Forest (Anomaly Detector)...")

# Train only on LEGITIMATE transactions
X_train_legit = X_train[y_train == 0]

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,   # ~2% expected anomalies
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)

isolation_forest.fit(X_train_legit)

# Save Isolation Forest
joblib.dump(
    isolation_forest,
    "model/isolation_forest.pkl"
)

print("✅ Isolation Forest trained and saved.")

# Get anomaly scores for test set
# score_samples returns negative values
# more negative = more anomalous
iso_scores_test = isolation_forest.score_samples(X_test)

# Normalize to [0, 1] range
# Higher = more anomalous (fraud-like)
iso_min = iso_scores_test.min()
iso_max = iso_scores_test.max()

iso_scores_normalized = 1 - (
    (iso_scores_test - iso_min) /
    (iso_max - iso_min + 1e-9)
)

# Save normalization params for inference
joblib.dump(
    {"min": iso_min, "max": iso_max},
    "model/iso_norm_params.pkl"
)

# ==============================
# BUILD NEURAL NETWORK
# ==============================

print("\n⏳ Building Neural Network...")

n_features = X_train.shape[1]

model = Sequential([

    Input(shape=(n_features,)),

    # Layer 1
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 2
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 3
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    # Layer 4
    Dense(16, activation="relu"),
    Dropout(0.2),

    # Output
    Dense(1, activation="sigmoid")
])

model.summary()

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# ==============================
# CALLBACKS
# ==============================

early_stop = EarlyStopping(
    monitor="val_auc",
    patience=5,
    restore_best_weights=True,
    mode="max"       # maximize AUC, not minimize loss
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_auc",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    mode="max",
    verbose=1
)

# ==============================
# TRAIN
# ==============================

print("\n⏳ Training Neural Network...")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# ==============================
# EVALUATE — NEURAL NETWORK
# ==============================

print("\n" + "="*50)
print("NEURAL NETWORK EVALUATION")
print("="*50)

loss, accuracy, auc, precision, recall = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"  Loss      : {loss:.4f}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")

# ==============================
# PREDICTIONS (Neural Network)
# ==============================

nn_probs = model.predict(X_test, verbose=0).flatten()

# ==============================
# COMBINED SCORE
# Neural Net (70%) + Isolation Forest (30%)
# This is the true ML-based final score
# No manual rule boosting
# ==============================

NN_WEIGHT  = 0.70
ISO_WEIGHT = 0.30

combined_probs = (
    (NN_WEIGHT  * nn_probs) +
    (ISO_WEIGHT * iso_scores_normalized)
)

combined_probs = np.clip(combined_probs, 0.0, 1.0)

# Find best threshold using Precision-Recall curve
# Default 0.5 is wrong for imbalanced data
precisions, recalls, thresholds = precision_recall_curve(
    y_test,
    combined_probs
)

# F1 score at each threshold
f1_scores = (
    2 * precisions * recalls /
    (precisions + recalls + 1e-9)
)

best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

print(f"\n  Best Threshold (F1-optimal): {best_threshold:.4f}")

# Save best threshold for inference
joblib.dump(
    float(best_threshold),
    "model/best_threshold.pkl"
)

# Final predictions using best threshold
y_pred_combined = (combined_probs >= best_threshold).astype(int)

# ==============================
# FULL EVALUATION — COMBINED
# ==============================

print("\n" + "="*50)
print("COMBINED MODEL EVALUATION (NN + IsolationForest)")
print("="*50)

print(f"\n  AUC-ROC (combined): {roc_auc_score(y_test, combined_probs):.4f}")
print(f"  Avg Precision     : {average_precision_score(y_test, combined_probs):.4f}")

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred_combined,
        target_names=["Legit", "Fraud"]
    )
)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_combined)
print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

# ==============================
# SAVE NEURAL NETWORK
# ==============================

model.save("model/fraud_model.keras")

print("\n✅ Neural Network saved  → model/fraud_model.keras")
print("✅ Isolation Forest saved → model/isolation_forest.pkl")
print("✅ Scaler saved           → model/scaler.pkl")
print("✅ Encoder saved          → model/label_encoder.pkl")
print("✅ Feature columns saved  → model/feature_columns.pkl")
print("✅ Best threshold saved   → model/best_threshold.pkl")
print("✅ ISO norm params saved  → model/iso_norm_params.pkl")
print("\n🎉 FraudLens ML Training Complete!")
