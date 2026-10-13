"""Evaluation metrics and optional diagnostic plot."""
import numpy as np
import matplotlib.pyplot as plt


def metrics(actual, predicted) -> dict[str, float]:
    """Return MAE, RMSE, and MAPE in percent (excluding zero targets)."""
    a, b = np.asarray(actual), np.asarray(predicted)
    nonzero = a != 0
    return {'mae': float(np.mean(np.abs(a - b))),
            'rmse': float(np.sqrt(np.mean((a - b) ** 2))),
            'mape': float(100 * np.mean(np.abs((a[nonzero] - b[nonzero]) / a[nonzero])))}


def plot_forecast(timestamps, actual, predicted, output: str) -> None:
    """Save the first week of actual and predicted hourly load."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(timestamps[:168], actual[:168], label='Actual')
    ax.plot(timestamps[:168], predicted[:168], label='Forecast')
    ax.set_ylabel('Load (MW)')
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
