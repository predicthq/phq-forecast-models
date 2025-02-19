import pandas as pd
import logging

from . import metrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEMAND_PERCENT_THRESHOLD = 5


def get_forecast_metrics(
    results: pd.DataFrame, ignore_low_demand: bool = True
) -> float:
    results = results.sort_values("date")

    if ignore_low_demand:
        results["demand_percent"] = results["demand"] / results["demand"].mean() * 100
        low_demand_days = results[
            (results["demand"] > 0)
            & (results["demand_percent"] < DEMAND_PERCENT_THRESHOLD)
        ]

        if len(low_demand_days) > 0:
            log.warning(
                f"There are demand values that are less than {DEMAND_PERCENT_THRESHOLD}% of the mean demand.",
                "Forecasting results for days with these small demand values and the days following them are skipped.",
            )
            date_upper = results[results["demand_percent"] < DEMAND_PERCENT_THRESHOLD][
                "date"
            ].values[0]
            results = results[results["date"] < date_upper]

    # skip where demand or forecast is null or days with missing demand values
    valid = (
        results["demand"].notnull()
        & results["forecast"].notnull()
        & results["is_imputed_demand"]
        == 0
    )
    results = results[valid]

    y_true = results["demand"]
    y_pred = results["forecast"]

    return metrics.mape(y_true, y_pred)
