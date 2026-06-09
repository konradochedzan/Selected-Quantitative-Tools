from pathlib import Path
from typing import Iterable, Union

import joblib
import pandas as pd

def save_model(model, file_path: Union[str, Path]) -> None:
    """
    Persist a fitted model (including its coefficients/parameters)
    to disk using Joblib’s fast, compressed pickle format.

    Parameters
    ----------
    model      : any scikit-learn / statsmodels / lightgbm model, or any object
                 that is pickle-serialisable.
    file_path  : str | Path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path.resolve()}")


def load_model(file_path: Union[str, Path]):
    """
    Restore a model previously saved with `save_model`.

    Returns
    -------
    object  – the model, ready for `.predict(...)`
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    return joblib.load(file_path)


def save_predictions(
    dates: Iterable, predictions: Iterable, csv_path: Union[str, Path]
) -> None:
    """
    Save predictions to a tidy two-column CSV:

        date,prediction
        2025-01-01,123.4
        2025-01-02,125.7
        …

    Parameters
    ----------
    dates        : 1-D iterable of datetimes / strings convertible by
                   `pandas.to_datetime`.
    predictions  : 1-D iterable of floats / ints (same length as `dates`).
    csv_path     : destination file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="raise"),
            "prediction": predictions,
        }
    )
    df.to_csv(csv_path, index=False)
    print(f" Predictions saved to {csv_path.resolve()}")


def load_predictions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a CSV created by `save_predictions` and return a DataFrame
    with the ‘date’ column parsed as pandas Timestamps.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")
    return pd.read_csv(csv_path, parse_dates=["date"])