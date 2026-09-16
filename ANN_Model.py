import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def train_ann():
    df = pd.read_csv("randomized_cubic_function.csv")
    X = df[["x"]].values
    y = df[["y"]].values

    # 1. Scale inputs and targets to zero mean and unit variance
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 2. Build the model with SILU/Swish activation (prevents vanishing gradients)
    model = Sequential(
        [
            Dense(64, activation="swish", input_shape=(1,)),
            Dense(64, activation="swish"),
            Dense(32, activation="swish"),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    # 3. Fit on scaled data
    model.fit(X_scaled, y_scaled, epochs=1000, batch_size=32, verbose=0)

    # Return model alongside scalers needed for inference
    return model, scaler_x, scaler_y


def predict(model, scaler_x, scaler_y, raw_x):
    # Scale raw input -> predict -> inverse scale back to target range
    x_input = np.array([[raw_x]])
    x_scaled = scaler_x.transform(x_input)
    y_scaled_pred = model.predict(x_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_scaled_pred)
    return y_pred[0][0]
