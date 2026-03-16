from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def build_lstm_model(
    window_size: int,
    n_features: int,
    horizon: int,
    lstm_units: int,
    learning_rate: float,
):
    """Build and compile an LSTM forecasting model."""

    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(lstm_units),
        Dense(horizon),
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"])
    

    return model