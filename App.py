import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from ann_model import train_ann


@st.cache_resource
def load_model():
    return train_ann()


model = load_model()


st.title("ANN Simulation: y = x³")

st.write("Enter a value of x and let the ANN predict y.")


if "x_values" not in st.session_state:
    st.session_state.x_values = []

if "actual_values" not in st.session_state:
    st.session_state.actual_values = []

if "predicted_values" not in st.session_state:
    st.session_state.predicted_values = []


x = st.number_input(
    "Enter value of x",
    value=0.0
)


col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("Predict")

with col2:
    reset_button = st.button("Reset")


if reset_button:
    st.session_state.x_values = []
    st.session_state.actual_values = []
    st.session_state.predicted_values = []

    st.rerun()


if predict_button:

    actual = x ** 3

    prediction = model.predict(
        np.array([[x]], dtype=np.float32),
        verbose=0
    )[0][0]

    st.session_state.x_values.append(x)
    st.session_state.actual_values.append(actual)
    st.session_state.predicted_values.append(float(prediction))

    st.subheader("Prediction")

    st.write(f"**Input x:** {x}")
    st.write(f"**Actual value (x³):** {actual:.6f}")
    st.write(f"**ANN prediction:** {prediction:.6f}")


if len(st.session_state.x_values) > 0:

    fig, ax = plt.subplots()

    order = np.argsort(st.session_state.x_values)

    x_plot = np.array(st.session_state.x_values)[order]
    actual_plot = np.array(st.session_state.actual_values)[order]
    predicted_plot = np.array(st.session_state.predicted_values)[order]

    ax.plot(
        x_plot,
        actual_plot,
        marker="o",
        label="Actual y = x³"
    )

    ax.plot(
        x_plot,
        predicted_plot,
        marker="x",
        linestyle="--",
        label="ANN Prediction"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Actual vs ANN Predicted Values")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)