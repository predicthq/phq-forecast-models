from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from .settings import (
    CATEGORICAL_FEATURES,
    XGBOOST__N_ESTIMATORS,
    XGBOOST__LEARNING_RATE,
    XGBOOST__MAX_DEPTH,
    N_SPLITS,
)

NON_FEATURE_COLUMNS = [
    "date",
    "demand",
    "is_imputed_demand",
]


def xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
):
    """
    Trains an XGBoost model using a pipeline that preprocesses numeric and categorical data,
    and performs grid search with time series cross-validation.

    Parameters:
        X: DataFrame containing training features.
        y: Series containing training target values.
        n_splits: Number of splits for TimeSeriesSplit.

    Returns:
        grid_search: Fitted GridSearchCV object.
        numeric_cols: List of numeric feature columns.
        categorical_cols: List of categorical feature columns.
    """
    # define the time series cross-validator
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # initialize the XGBRegressor
    model = XGBRegressor()

    # define transformers for numeric and categorical columns
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(drop=None, handle_unknown="ignore"))]
    )

    # determine which columns are numeric and which are categorical
    feature_columns = [x for x in X.columns if x not in NON_FEATURE_COLUMNS]
    categorical_cols = [col for col in feature_columns if col in CATEGORICAL_FEATURES]
    numeric_cols = [col for col in feature_columns if col not in categorical_cols]

    # create a column transformer to apply the transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # define the parameter grid for GridSearchCV
    param_grid = {
        "model__n_estimators": XGBOOST__N_ESTIMATORS,
        "model__learning_rate": XGBOOST__LEARNING_RATE,
        "model__max_depth": XGBOOST__MAX_DEPTH,
    }

    # create the pipeline
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # set up GridSearchCV using time series cross-validation
    grid_search = GridSearchCV(
        estimator=pipe, param_grid=param_grid, cv=tscv, scoring="neg_mean_squared_error"
    )

    # fit the model
    grid_search.fit(X, y)

    return grid_search


class PhqForecastModel:
    def __init__(self):
        self.model = None

    @staticmethod
    def cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS):
        grid_search = xgboost_model(X, y, n_splits=n_splits)
        return grid_search

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        # Skip this period if there is no training data
        if len(X) == 0:
            return None

        grid_search = self.cross_validation(X, y["demand"])
        self.model = grid_search.best_estimator_

    def predict(self, X: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model has not been trained")

        yp = self.model.predict(X)
        yp = np.maximum(0, yp)  # clip any negative predictions

        return yp
