import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from utils.preprocess import (
    load_data,
    preprocess_data
)


df = load_data(
    "dataset/fraud_dataset.csv"
)


X, y = preprocess_data(df)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = Sequential([

    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


model.fit(

    X_train,
    y_train,

    epochs=5,

    batch_size=64,

    validation_split=0.2
)


model.save(
    "model/fraud_model.h5"
)

print("✅ Model Trained Successfully")