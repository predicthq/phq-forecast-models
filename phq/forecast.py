import logging

from datetime import date

import pandas as pd
import numpy as np
from pandas import DataFrame


from .performance import get_forecast_metrics
from .models import PhqForecastModel
from .settings import EVAL_FORECAST_HORIZON, EVAL_TRAIN_RATIO


log = logging.getLogger(__name__)


def _test_train_split(
    X: DataFrame, y: DataFrame, start: date, end: date
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = X[X["date"] < start].copy()
    y_train = y[y["date"] < start].copy()

    X_test = X[(X["date"] >= start) & (X["date"] <= end)].copy()
    y_test = y[(y["date"] >= start) & (y["date"] <= end)].copy()

    return X_train, y_train, X_test, y_test


def _forecast_window(X: pd.DataFrame, y: pd.DataFrame, start: date):
    end = start + pd.Timedelta(days=EVAL_FORECAST_HORIZON - 1)
    X_train, y_train, X_test, y_test = _test_train_split(X, y, start, end)

    # Skip this testing period if there is no training/testing data
    if len(X_train) == 0 or len(X_test) == 0:
        return None

    # Fit model
    grid_search = PhqForecastModel.cross_validation(X_train, y_train["demand"])

    # Make predictions
    prediction = grid_search.best_estimator_.predict(X_test)
    prediction = np.maximum(0, prediction)  # Ensure that predictions are non-negative

    result = y_test
    result["forecast"] = prediction

    return result


def _get_forecast_start_dates(timeseries: pd.DataFrame):
    count = int(len(timeseries) * EVAL_TRAIN_RATIO)
    start_date = timeseries.loc[count, "date"]
    dataset_end_date = timeseries["date"].max()

    while start_date <= dataset_end_date:
        yield start_date
        start_date += pd.Timedelta(days=EVAL_FORECAST_HORIZON)


def _rolling_cross_validation(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    analysis_results = []

    for forecast_start_date in _get_forecast_start_dates(X):
        res_window = _forecast_window(X, y, forecast_start_date)
        analysis_results.append(res_window)

    return pd.concat(analysis_results)


def evaluate_forecast_model(X: pd.DataFrame, y: pd.DataFrame) -> dict:
    forecast_results = _rolling_cross_validation(X, y)

    result = {
        "mape": get_forecast_metrics(forecast_results),
        "forecast_results": forecast_results,
    }

    return result
