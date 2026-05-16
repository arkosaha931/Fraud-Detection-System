import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.utils.class_weight import (
    compute_class_weight
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from utils.preprocess import (
    load_data,
    preprocess_data
)

# ==============================
# LOAD DATASET
# ==============================

df = load_data(
    "dataset/fraud_dataset.csv"
)

# ==============================
# PREPROCESS
# ==============================

X, y = preprocess_data(df)

# ==============================
# CLASS DISTRIBUTION
# ==============================

print("\nClass Distribution:\n")

print(y.value_counts())

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

# ==============================
# CLASS WEIGHTS
# ==============================

class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(y_train),

    y=y_train
)

class_weights = {

    0: class_weights[0],

    1: class_weights[1]
}

print("\nClass Weights:\n")

print(class_weights)

# ==============================
# BUILD MODEL
# ==============================

model = Sequential([

    Input(shape=(X_train.shape[1],)),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])

# ==============================
# COMPILE
# ==============================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

# ==============================
# EARLY STOPPING
# ==============================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True
)

# ==============================
# TRAIN
# ==============================

history = model.fit(

    X_train,

    y_train,

    epochs=15,

    batch_size=128,

    validation_split=0.2,

    callbacks=[early_stop],

    class_weight=class_weights,

    verbose=1
)

# ==============================
# EVALUATE
# ==============================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nTest Accuracy: {accuracy:.4f}")

# ==============================
# PREDICTIONS
# ==============================

y_pred_probs = model.predict(X_test)

y_pred = (
    y_pred_probs > 0.5
).astype(int)

print("\nClassification Report:\n")

print(

    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:\n")

print(

    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==============================
# SAVE MODEL
# ==============================

model.save(
    "model/fraud_model.keras"
)

print(
    "\n✅ Fraud Detection Model Trained Successfully"
)