from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_lstm_model(
    window_size: int,
    n_features: int,
    horizon: int,
    lstm_units: int,
    learning_rate: float,
    dropout: float = 0.0,
    l2_strength: float = 0.0,
) -> Sequential:
    """Build and compile an LSTM forecasting model."""

    model = Sequential(
        [
        Input(shape=(window_size, n_features)),
        LSTM(
            lstm_units,
            dropout=dropout,
            kernel_regularizer=l2(l2_strength) if l2_strength > 0 else None
        ),
        Dense(horizon),
        ]
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )
    
    return model