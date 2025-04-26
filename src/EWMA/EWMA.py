import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ewma(data, alpha):
    """
    Compute the Exponentially Weighted Moving Average (EWMA) of a 1D array.

    Parameters:
        data (array-like): Input data sequence.
        alpha (float): Smoothing factor in (0, 1], higher alpha weighs recent values more.

    Returns:
        list: Smoothed data.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    smoothed = [data[0]]
    for x in data[1:]:
        smoothed.append(alpha * x + (1 - alpha) * smoothed[-1])
    return smoothed


def load_and_clean(path, column_index=1):
    """
    Load an Excel file and extract one numeric column, dropping invalid entries.

    Parameters:
        path (str or Path): Path to the .xlsx file.
        column_index (int): Zero-based index of the column to process.

    Returns:
        np.ndarray: Cleaned numeric data.
    """
    df = pd.read_excel(path)
    raw_series = df.iloc[:, column_index]
    # Replace any placeholder strings with NaN, then drop
    numeric = pd.to_numeric(raw_series.replace('--', np.nan), errors='coerce')
    cleaned = numeric.dropna().values
    if cleaned.size == 0:
        raise RuntimeError(f"No valid numeric data found in column {column_index}")
    return cleaned


def plot_results(original, smoothed, output_path):
    """
    Plot original vs. EWMA-smoothed data and save the figure.

    Parameters:
        original (array-like): Raw data.
        smoothed (array-like): EWMA data.
        output_path (str or Path): Where to save the .png file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(original, label='Original Data')
    plt.plot(smoothed, label='EWMA Smoothed', linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('EWMA Smoothing Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def main():
    # Configuration
    input_path = Path(__file__).parent / "../../data/raw/38.xlsx"
    output_excel = Path(__file__).parent / "../../data/processed/A38_EWMA.xlsx"
    output_plot  = Path(__file__).parent / "../../data/processed/A38_ewma_denoising.png"
    alpha = 0.2

    # Ensure directories exist
    os.makedirs(output_excel.parent, exist_ok=True)

    # Load and clean data
    data = load_and_clean(input_path, column_index=1)

    # Compute EWMA
    ewma_data = ewma(data, alpha)

    # Save results to Excel
    result_df = pd.DataFrame({
        'Original Data': data,
        'EWMA': ewma_data
    })
    result_df.to_excel(output_excel, index=False)
    print(f"Saved smoothed data to {output_excel}")

    # Plot and save figure
    plot_results(data, ewma_data, output_plot)
    print(f"Saved plot to {output_plot}")


if __name__ == "__main__":
    main()
