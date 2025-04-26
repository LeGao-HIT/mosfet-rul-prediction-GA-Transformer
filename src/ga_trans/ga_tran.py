import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm  # if you want progress bars for loops
# from tensorflow.keras.utils import plot_model  # uncomment if you need to save model diagram


def load_series_from_excel(path, column_index=1):
    """
    Load a single column of numeric data from an Excel file and return as a 2D numpy array.
    """
    df = pd.read_excel(path)
    series = pd.to_numeric(df.iloc[:, column_index], errors='coerce')
    cleaned = series.dropna().values.reshape(-1, 1)
    if cleaned.size == 0:
        raise RuntimeError(f"No valid numeric data found in column {column_index} of {path}")
    return cleaned


def create_dataset(dataset, look_back):
    """
    Turn a 2D array (n_samples x 1) into sequences for time-series.
    Returns X of shape (n_samples-look_back, look_back) and y of shape (n_samples-look_back,).
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i : i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)


def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout_rate, activation):
    """
    A single Transformer encoder block (pre-Norm).
    """
    # multi-head self-attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout_rate
    )(x, x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Add()([x, inputs])

    # feed-forward
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation)(y)
    y = layers.Dropout(dropout_rate)(y)
    y = layers.Add()([y, x])
    return y


def build_ga_transformer(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    mlp_dropout,
    dropout_rate,
    activation="relu",
):
    """
    Build a GA-Transformer regression model.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout_rate, activation)

    x = layers.Flatten()(x)
    for units in mlp_units:
        x = layers.Dense(units, activation=activation)(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(1, activation="linear")(x)
    return keras.Model(inputs, outputs)


def train_and_evaluate():
    # Paths
    base = Path(__file__).parent
    input_file = base / "../../data/processed/A36_EWMA.xlsx"
    output_dir = base / "../../data/output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    look_back = 3
    head_size = 32
    num_heads = 16
    ff_dim = 16
    num_transformer_blocks = 1
    mlp_units = [16]
    dropout_rate = 0.2
    mlp_dropout = 0.2
    batch_size = 64
    learning_rate = 0.01
    max_epochs = 400
    val_split = 0.2
    patience = 20

    # 1. Load and split
    data = load_series_from_excel(input_file, column_index=1)
    train_size = int(len(data) * 0.7)
    train, test = data[:train_size], data[train_size:]
    print(f"train shape: {train.shape}, test shape: {test.shape}")

    # 2. Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # 3. Create sequences
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, y_test = create_dataset(test_scaled, look_back)
    x_train = X_train.reshape(*X_train.shape, 1)
    x_test = X_test.reshape(*X_test.shape, 1)

    # 4. Build & compile model
    model = build_ga_transformer(
        input_shape=x_train.shape[1:],
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        mlp_units=mlp_units,
        mlp_dropout=mlp_dropout,
        dropout_rate=dropout_rate,
        activation="relu",
    )
    model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adadelta(learning_rate=learning_rate),
        metrics=["mean_absolute_error"],
    )
    model.summary()

    # 5. Train
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, verbose=1, restore_best_weights=True)
    history = model.fit(
        x_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=2,
    )

    # 6. Plot & save loss curves
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot = output_dir / "loss_curve.png"
    plt.savefig(loss_plot)
    plt.show()
    print(f"Saved loss curve to {loss_plot}")

    # 7. Save loss history
    loss_df = pd.DataFrame({
        "Epoch": np.arange(1, len(history.history["loss"]) + 1),
        "Train Loss": history.history["loss"],
        "Val Loss": history.history["val_loss"],
    })
    loss_df.to_excel(output_dir / "training_history.xlsx", index=False)

    # 8. Predict & inverse-scale
    y_pred = model.predict(x_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    # 9. Metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_inv - y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.2f}%")

    # 10. Plot predictions
    plt.figure()
    plt.plot(y_test_inv, label="True")
    plt.plot(y_pred_inv, label="Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("GA-Transformer Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "predictions.png")
    plt.show()

    # 11. Save results
    results_df = pd.DataFrame({
        "Sample": np.arange(len(y_test_inv)),
        "True": y_test_inv.flatten(),
        "Predicted": y_pred_inv.flatten()
    })
    results_df.to_excel(output_dir / "predicted_results.xlsx", index=False)
    print(f"Saved predictions to {output_dir / 'predicted_results.xlsx'}")


if __name__ == "__main__":
    train_and_evaluate()
