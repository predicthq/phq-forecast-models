import pandas as pd
import logging

from . import metrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEMAND_PERCENT_THRESHOLD = 5


def get_forecast_metrics(results: pd.DataFrame) -> float:
    results = results.sort_values("date")

    # Check if there are small postive demand values
    results["demand_percent"] = results["demand"] / results["demand"].mean() * 100
    low_demand_days = results[
        (results["demand"] > 0)
        & (results["demand_percent"] < DEMAND_PERCENT_THRESHOLD)
    ]

    # Output warning message if there are days with small demand values
    if len(low_demand_days) > 0:
        log.warning(
            f"There are positive demand values that are less than {DEMAND_PERCENT_THRESHOLD}% of the mean demand. "
            "This might lead to extremely large MAPE. "
            "Please check whether these small demand values represent actual daily demand.",
        )

    # Skip where demand or forecast is null or days with missing demand values
    valid = (
        results["demand"].notnull()
        & results["forecast"].notnull()
        & (results["is_imputed_demand"]== 0)
    )
    results = results[valid]

    y_true = results["demand"]
    y_pred = results["forecast"]

    return metrics.mape(y_true, y_pred)
