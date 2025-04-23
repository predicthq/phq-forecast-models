import logging
from datetime import datetime

import pandas as pd
import numpy as np
from predicthq import Client

from .settings import TREND_FEATURES

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d"

DAYS_PER_WEEK = 7
MONTHS_PER_YEAR = 12
DAYS_PER_YEAR = 365.25  # accounts for leap years
TWO_PI = 2 * np.pi

RANK_BAND = 20


def _get_event_feature_value(feature_name: str, feature_values: dict) -> float:
    if "_rank_" in feature_name:
        rank_values = [
            int(rank)
            for rank, value in feature_values["rank_levels"].items()
            if value > 0
        ]
        return max(rank_values, default=0) * RANK_BAND

    if "_impact_severe_weather" in feature_name:
        return feature_values["stats"]["max"]

    return feature_values["stats"]["sum"]


def _get_event_features(
    phq: Client, analysis_id: str, start: datetime, end: datetime
) -> list[dict]:
    result = []

    for feature in phq.features.obtain_features(
        active__gte=start.strftime(DATE_FORMAT),
        active__lte=end.strftime(DATE_FORMAT),
        beam__analysis_id=analysis_id,
    ).iter_all():
        feature_data = feature.dict()

        date = feature_data.pop("date")
        daily_features = {"date": date}

        for feature_name, feature_values in feature_data.items():
            daily_features[feature_name] = _get_event_feature_value(
                feature_name, feature_values
            )

        result.append(daily_features)

    return result


def prepare_time_trend_features(X: pd.DataFrame):
    df = X.copy()

    df["day_of_week"] = df["date"].dt.dayofweek  # categorical
    df["sin_day_of_week"] = np.sin(TWO_PI * df["date"].dt.dayofweek / DAYS_PER_WEEK)
    df["cos_day_of_week"] = np.cos(TWO_PI * df["date"].dt.dayofweek / DAYS_PER_WEEK)

    df["month"] = df["date"].dt.month  # categorical
    df["sin_month_of_year"] = np.sin(TWO_PI * df["date"].dt.month / MONTHS_PER_YEAR)
    df["cos_month_of_year"] = np.cos(TWO_PI * df["date"].dt.month / MONTHS_PER_YEAR)
    df["sin_day_of_year"] = np.sin(TWO_PI * df["date"].dt.dayofyear / DAYS_PER_YEAR)
    df["cos_day_of_year"] = np.cos(TWO_PI * df["date"].dt.dayofyear / DAYS_PER_YEAR)

    df["year"] = df["date"].dt.year  # categorical

    # check for consecutive dates
    date_diff = df["date"].diff().dt.days
    if (date_diff[1:] != 1).any():
        log.warning("Missing dates detected")

    # add lag7 based on demand
    df["demand_lag7"] = df["demand"].shift(7)

    return df[["date"] + TREND_FEATURES]


def prepare_event_features(analysis_id: str, access_token: str):
    phq_client = Client(access_token=access_token)
    analysis = phq_client.beam.analysis.get(analysis_id=analysis_id)

    # extract the start and end dates from the analysis
    date_range = analysis.readiness_checks.date_range
    start_date = date_range.start
    end_date = date_range.end

    features = _get_event_features(phq_client, analysis_id, start_date, end_date)
    if len(features) > 0:
        features_df = pd.DataFrame(features)
        features_df["date"] = pd.to_datetime(features_df["date"])
    else:
        # Create a DataFrame with just the 'date' column from start_date to end_date
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        features_df = pd.DataFrame({"date": date_range})

    return features_df


def prepare_forecast_features(
    demand_df: pd.DataFrame, analysis_id: str, window_size: int, access_token: str
):
    phq_client = Client(access_token=access_token)

    # calculate the start and end date for the following week
    max_date = demand_df["date"].max()
    start_date = max_date + pd.Timedelta(days=1)
    end_date = max_date + pd.Timedelta(days=window_size)

    features = _get_event_features(phq_client, analysis_id, start_date, end_date)
    if len(features) > 0:
        features_df = pd.DataFrame(features)
        features_df["date"] = pd.to_datetime(features_df["date"])
    else:
        # Create a DataFrame with just the 'date' column from start_date to end_date
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        features_df = pd.DataFrame({"date": date_range})

    train_and_future_df = pd.concat([demand_df, features_df], ignore_index=True)
    train_and_future_df["date"] = pd.to_datetime(train_and_future_df["date"])

    test_future_df = prepare_time_trend_features(train_and_future_df)
    test_future_df = test_future_df[-len(features_df) :].copy()

    return train_and_future_df.merge(test_future_df, on="date", how="inner").drop(
        columns=["demand", "is_imputed_demand"]
    )
