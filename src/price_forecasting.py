"""Simple price forecasting pipeline.

This module loads historical price data from a CSV file, trains a regression
model and produces a short-term forecast.  It is intentionally lightweight so
that it can run without any external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class ForecastResult:
    """Container with model evaluation metrics and forecasts."""

    model: Pipeline
    mae: float
    forecast: pd.DataFrame


def load_price_data(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file with date and price columns.

    The CSV is expected to contain at least two columns named ``Date`` and
    ``Price``.  Dates are converted to pandas ``datetime64`` objects and sorted
    in chronological order.
    """

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", "Price": "price"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _build_model(degree: int = 3) -> Pipeline:
    """Create the regression pipeline used for forecasting."""

    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg", LinearRegression()),
        ]
    )


def _split_train_test(df: pd.DataFrame, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test segments preserving ordering."""

    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be between 1 and len(df) - 1")

    return df.iloc[:-test_size], df.iloc[-test_size:]


def _date_to_ordinal(dates: Iterable[pd.Timestamp]) -> np.ndarray:
    """Convert a sequence of pandas timestamps to ordinal integers."""

    return np.array([d.toordinal() for d in dates]).reshape(-1, 1)


def train_and_forecast(
    df: pd.DataFrame,
    *,
    horizon_days: int = 7,
    polynomial_degree: int = 3,
    test_size: int = 7,
) -> ForecastResult:
    """Train the regression model and forecast future prices.

    Parameters
    ----------
    df:
        Historical price observations with ``date`` and ``price`` columns.
    horizon_days:
        Number of days to forecast into the future.
    polynomial_degree:
        Degree of the polynomial features applied to the ordinal date feature.
    test_size:
        Number of trailing observations reserved for testing.
    """

    if horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")

    train_df, test_df = _split_train_test(df, test_size)

    model = _build_model(degree=polynomial_degree)
    X_train = _date_to_ordinal(train_df["date"])
    y_train = train_df["price"].to_numpy()
    model.fit(X_train, y_train)

    # Evaluate on the hold-out period.
    X_test = _date_to_ordinal(test_df["date"])
    y_test = test_df["price"].to_numpy()
    predictions = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, predictions))

    # Produce the horizon-day forecast.
    last_date = df["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)
    future_features = _date_to_ordinal(future_dates)
    future_predictions = model.predict(future_features)
    forecast = pd.DataFrame({"date": future_dates, "forecast_price": future_predictions})

    return ForecastResult(model=model, mae=mae, forecast=forecast)


def run_example(csv_path: Path, horizon_days: int = 7) -> ForecastResult:
    """Load the provided dataset and run the forecasting pipeline."""

    df = load_price_data(csv_path)
    result = train_and_forecast(df, horizon_days=horizon_days)
    return result


def main() -> None:
    """Entry point for running the example from the command line."""

    csv_path = Path(__file__).resolve().parent.parent / "data" / "price_history.csv"
    result = run_example(csv_path)

    print("Hold-out mean absolute error: {:.3f}".format(result.mae))
    print("\nForecasted prices:")
    print(result.forecast.to_string(index=False, formatters={"forecast_price": "{:.2f}".format}))


if __name__ == "__main__":
    main()
