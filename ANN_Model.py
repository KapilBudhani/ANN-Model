import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def train_ann():
    x = np.linspace(-10, 10, 1000)
    y = x ** 3
    
    model = Sequential([
        Dense(32, activation="tanh", input_shape=(1,)),
        Dense(32, activation="tanh"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.fit(
        x,
        y,
        epochs=500,
        batch_size=32,
        verbose=0
    )

    return model
