import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the mean absolute percentage error (MAPE).
    """
    nonzero = y_true != 0

    return mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero]) * 100
