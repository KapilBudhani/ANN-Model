import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def train_ann():
    df = pd.read_csv("randomized_cubic_function.csv")
    x = df['x']
    y = df['y']
    
    model = Sequential([
            Dense(64, activation="tanh", input_shape=(1,)),
            Dense(32, activation="tanh"),
            Dense(1),
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