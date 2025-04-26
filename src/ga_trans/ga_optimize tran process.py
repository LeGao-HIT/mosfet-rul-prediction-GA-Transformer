import os
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from deap import base, creator, tools, algorithms

# =============================================================================
# 1. Load & preprocess data
# =============================================================================

def load_series(path: Path, column_index: int = 1) -> np.ndarray:
    """
    Load a single numeric column from an Excel file, drop NaNs, and return as (n,1) array.
    """
    df = pd.read_excel(path)
    series = pd.to_numeric(df.iloc[:, column_index], errors='coerce')
    arr = series.dropna().values.reshape(-1, 1)
    if arr.size == 0:
        raise RuntimeError(f"No valid numeric data in column {column_index} of {path}")
    return arr

def split_and_scale(data: np.ndarray, train_frac: float = 0.7):
    """
    Scale full series to [0,1], then split into train/test.
    Returns scaler, train_scaled, test_scaled.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    split = int(len(data_scaled) * train_frac)
    return scaler, data_scaled[:split], data_scaled[split:]


# =============================================================================
# 2. Time-series dataset creation
# =============================================================================

def create_dataset(series: np.ndarray, look_back: int):
    """
    Transform a (n,1) series into X of shape (n-look_back, look_back, 1) and y of shape (n-look_back,).
    """
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back, 0])
        y.append(series[i+look_back, 0])
    X = np.array(X)
    y = np.array(y)
    return X.reshape(-1, look_back, 1), y


# =============================================================================
# 3. Model builder & evaluator
# =============================================================================

def build_transformer(input_shape, params):
    """
    Build a single-block Transformer + simple MLP head for regression.
    params: dict with keys head_size, num_heads, ff_dim, dropout_rate, activation.
    """
    inp = keras.Input(shape=input_shape)
    x = inp
    # One transformer block
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(
        key_dim=params["head_size"],
        num_heads=params["num_heads"],
        dropout=params["dropout_rate"]
    )(x_norm, x_norm)
    attn = layers.Dropout(params["dropout_rate"])(attn)
    x = layers.Add()([attn, x])

    y_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    y_conv = layers.Conv1D(
        filters=params["ff_dim"],
        kernel_size=1,
        activation=params["activation"]
    )(y_norm)
    y_drop = layers.Dropout(params["dropout_rate"])(y_conv)
    x = layers.Add()([y_drop, x])

    # MLP head
    flat = layers.Flatten()(x)
    mlp = layers.Dense(params["ff_dim"], activation=params["activation"])(flat)
    mlp = layers.Dropout(params["dropout_rate"])(mlp)
    out = layers.Dense(1, activation="linear")(mlp)

    return keras.Model(inp, out)


def evaluate_individual(individual, keys, grid_values, train_scaled, test_scaled):
    """
    DEAP fitness function: decode individual, train model, return negative MSE.
    """
    # Decode
    params = {k: grid_values[i][gene] for i, (k, gene) in enumerate(zip(keys, individual))}
    look_back = params["sliding_window"]

    # Data prep
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, y_test = create_dataset(test_scaled, look_back)

    # Build & compile
    model = build_transformer(X_train.shape[1:], params)
    optimizer = keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(loss="mse", optimizer=optimizer)

    # Train with early stopping
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=params["batch_size"],
        validation_split=0.2,
        callbacks=[es],
        verbose=0
    )

    # Evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()
    mse = mean_squared_error(y_test, y_pred)

    # Free memory
    keras.backend.clear_session()
    return (-mse,)


# =============================================================================
# 4. GA setup
# =============================================================================

def setup_ga(param_grid):
    """
    Create DEAP toolbox, fitness, and population based on param_grid.
    Returns toolbox, keys, grid_values.
    """
    keys = list(param_grid.keys())
    grid_values = [param_grid[k] for k in keys]
    n_params = len(keys)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Attribute: random index for each hyperparameter
    for i, key in enumerate(keys):
        toolbox.register(f"attr_{key}", random.randrange, len(grid_values[i]))
    # Individual & population
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        [toolbox.__getattribute__(f"attr_{k}") for k in keys],
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=[0]*n_params,
        up=[len(vals)-1 for vals in grid_values],
        indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox, keys, grid_values


# =============================================================================
# 5. Main execution
# =============================================================================

def main():
    random.seed(42)
    tf.random.set_seed(42)

    # Paths
    base = Path(__file__).parent
    data_file = base / "../../data/processed/A36_EWMA.xlsx"
    output_dir = base / "../../data/ga_op_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load & split
    data = load_series(data_file)
    scaler, train_scaled, test_scaled = split_and_scale(data, train_frac=0.7)

    # Hyperparameter grid
    param_grid = {
        "sliding_window": list(range(1, 16)),
        "head_size": [16, 32, 64, 128, 256],
        "num_heads": list(range(1, 17)),
        "ff_dim": [16, 32, 64, 128, 256],
        "batch_size": [32, 64, 128, 256],
        "learning_rate": [1e-2, 1e-3, 1e-4],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "activation": ['relu', 'tanh', 'selu', 'elu', 'sigmoid']
    }

    # GA setup
    toolbox, keys, grid_values = setup_ga(param_grid)
    toolbox.register(
        "evaluate",
        evaluate_individual,
        keys=keys,
        grid_values=grid_values,
        train_scaled=train_scaled,
        test_scaled=test_scaled
    )

    # Run GA
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: -ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.6, mutpb=0.3,
        ngen=50, stats=stats,
        halloffame=hof, verbose=True
    )

    # Best hyperparameters
    best = hof[0]
    best_params = {k: grid_values[i][best[i]] for i, k in enumerate(keys)}
    print("Best hyperparameters:", best_params)
    with open(output_dir / "best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)

    # Retrain on full train set and evaluate
    look_back = best_params["sliding_window"]
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, y_test = create_dataset(test_scaled, look_back)

    model = build_transformer(X_train.shape[1:], best_params)
    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"])
    )

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=best_params["batch_size"],
        validation_split=0.2,
        callbacks=[es],
        verbose=1
    )

    # Save model
    model.save(output_dir / "best_ga_transformer_model.h5")

    # Predict & inverse-scale
    y_pred = model.predict(X_test).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Save predictions
    pd.DataFrame({
        "y_true": y_test_inv,
        "y_pred": y_pred_inv
    }).to_excel(output_dir / "best_predictions.xlsx", index=False)

    mse_final = mean_squared_error(y_test_inv, y_pred_inv)
    print(f"Final Test MSE: {mse_final:.6f}, RMSE: {np.sqrt(mse_final):.6f}")

if __name__ == "__main__":
    main()
