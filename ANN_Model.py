import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

X_SCALE = 1000.0
Y_SCALE = 1_000_000_000.0


def train_ann():
    df = pd.read_csv("randomized_cubic_function.csv")

    x = df["x"].values.astype(np.float32)
    y = df["y"].values.astype(np.float32)

    x_scaled = x / X_SCALE
    y_scaled = y / Y_SCALE

    model = Sequential([
        Input(shape=(1,)),
        Dense(64, activation="tanh"),
        Dense(64, activation="tanh"),
        Dense(32, activation="tanh"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    model.fit(
        x_scaled,
        y_scaled,
        epochs=1000,
        batch_size=32,
        verbose=0
    )

    return model


def predict(x, model):
    x_scaled = np.array([[x]], dtype=np.float32) / X_SCALE

    prediction_scaled = model.predict(
        x_scaled,
        verbose=0
    )[0][0]

    return prediction_scaled * Y_SCALE