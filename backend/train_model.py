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

from tensorflow.keras.losses import BinaryFocalCrossentropy

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score
)

from imblearn.over_sampling import SMOTE

from utils.preprocess import (
    load_data,
    preprocess_data
)



os.makedirs("model", exist_ok=True)



df = load_data("dataset/fraud_dataset.csv")


X, y, feature_cols = preprocess_data(df)

print("\nClass Distribution:")
print(f"Legit : {(y == 0).sum()}")
print(f"Fraud : {(y == 1).sum()}")



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)



print("\nApplying SMOTE Oversampling...")

smote = SMOTE(
    sampling_strategy=0.3,   # fraud : legit = 30:100
    random_state=42,
    k_neighbors=5
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(f"Legit : {(y_train_resampled == 0).sum()}")
print(f"Fraud : {(y_train_resampled == 1).sum()}")



print("\nTraining Isolation Forest...")

X_train_legit = X_train[y_train == 0]

isolation_forest = IsolationForest(
    n_estimators=300,
    contamination=0.001,   # tighter — expect <0.1% anomalies in legit
    random_state=42,
    n_jobs=-1
)

isolation_forest.fit(X_train_legit)

joblib.dump(
    isolation_forest,
    "model/isolation_forest.pkl"
)

print("✅ Isolation Forest Saved")


iso_scores_train = isolation_forest.score_samples(X_train)

iso_min = float(iso_scores_train.min())
iso_max = float(iso_scores_train.max())

print(f"\nISO norm range — min: {iso_min:.4f}  max: {iso_max:.4f}")

joblib.dump(
    {
        "min": iso_min,
        "max": iso_max
    },
    "model/iso_norm_params.pkl"
)

print("✅ ISO norm params saved")


iso_scores_test_raw = isolation_forest.score_samples(X_test)

iso_scores_test_normalized = np.clip(
    1.0 - (
        (iso_scores_test_raw - iso_min) /
        (iso_max - iso_min + 1e-9)
    ),
    0.0,
    1.0
)


fraud_count = int((y_train == 1).sum())
legit_count = int((y_train == 0).sum())
total       = fraud_count + legit_count

weight_for_0 = (1 / legit_count) * (total / 2.0)
weight_for_1 = (1 / fraud_count) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"\nClass weights — legit: {weight_for_0:.4f}  fraud: {weight_for_1:.4f}")


print("\nBuilding Neural Network...")

n_features = X_train.shape[1]

model = Sequential([

    Input(shape=(n_features,)),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.35),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dropout(0.15),

    Dense(1, activation="sigmoid")
])

model.summary()



model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss=BinaryFocalCrossentropy(
        gamma=3.0,
        apply_class_balancing=True
    ),

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


early_stop = EarlyStopping(
    monitor="val_auc",
    patience=8,
    restore_best_weights=True,
    mode="max"
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_auc",
    factor=0.4,
    patience=4,
    min_lr=1e-7,
    mode="max",
    verbose=1
)



print("\nTraining Neural Network...")

history = model.fit(
    X_train_resampled,
    y_train_resampled,

    validation_data=(X_test, y_test),

    epochs=80,
    batch_size=512,

    class_weight=class_weight,

    callbacks=[
        early_stop,
        reduce_lr
    ],

    verbose=1
)



print("\n" + "="*50)
print("NEURAL NETWORK EVALUATION")
print("="*50)

loss, accuracy, auc, precision, recall = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Loss      : {loss:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"AUC       : {auc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")



NN_WEIGHT      = 0.65
ISO_WEIGHT     = 0.15
FEATURE_WEIGHT = 0.20

nn_probs = model.predict(
    X_test,
    verbose=0
).flatten()


import pandas as pd

feature_cols_loaded = joblib.load("model/feature_columns.pkl")
scaler_loaded       = joblib.load("model/scaler.pkl")

X_test_orig = scaler_loaded.inverse_transform(X_test)
df_test     = pd.DataFrame(X_test_orig, columns=feature_cols_loaded)

def compute_feature_score_batch(df_batch):
    """
    Vectorised version of app.py compute_feature_score.
    transaction_frequency / high_frequency removed.
    Returns a numpy array of scores in [0, 1].
    """
    score = np.zeros(len(df_batch))

    # Full account drain — strongest signal
    score += 0.35 * (df_batch["account_drained"]      == 1).values

    # Amount vs balance ratio
    score += 0.20 * (df_batch["extreme_amount_ratio"]  == 1).values
    score += 0.12 * (
        (df_batch["high_amount_ratio"]    == 1) &
        (df_batch["extreme_amount_ratio"] == 0)
    ).values

    # Balance integrity
    score += 0.15 * (df_batch["orig_balance_mismatch"] == 1).values
    score += 0.12 * (df_batch["dest_balance_mismatch"] == 1).values

    # Receiver balance unchanged after transfer
    score += 0.12 * (df_batch["dest_no_increase"]      == 1).values

    # Behavioral / contextual
    score += 0.12 * (df_batch["device_risk"]           == 1).values
    score += 0.10 * (df_batch["location_risk"]         == 1).values

    return np.clip(score, 0.0, 1.0)

feature_scores_test = compute_feature_score_batch(df_test)



combined_probs = np.clip(
    (NN_WEIGHT      * nn_probs)                    +
    (ISO_WEIGHT     * iso_scores_test_normalized)  +
    (FEATURE_WEIGHT * feature_scores_test),
    0.0,
    1.0
)



precisions, recalls, thresholds = precision_recall_curve(
    y_test,
    combined_probs
)

f1_scores_arr = (
    2 * precisions * recalls /
    (precisions + recalls + 1e-9)
)

best_idx = np.argmax(f1_scores_arr)

best_threshold = float(thresholds[best_idx])

print(f"\n✅ Best Threshold : {best_threshold:.4f}")

joblib.dump(
    best_threshold,
    "model/best_threshold.pkl"
)



y_pred = (
    combined_probs >= best_threshold
).astype(int)

print("\n" + "="*50)
print("FINAL MODEL EVALUATION")
print("="*50)

print(f"AUC ROC : {roc_auc_score(y_test, combined_probs):.4f}")

print(
    f"Average Precision : "
    f"{average_precision_score(y_test, combined_probs):.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legit", "Fraud"]
    )
)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(f"TN : {cm[0][0]}")
print(f"FP : {cm[0][1]}")
print(f"FN : {cm[1][0]}")
print(f"TP : {cm[1][1]}")


model.save("model/fraud_model.keras")

print("\n✅ All models saved successfully.")