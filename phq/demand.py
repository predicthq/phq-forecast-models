import pandas as pd

ROLL_PERIOD = 15


def process_demand_data(demand_df: pd.DataFrame) -> pd.DataFrame:
    demand_df = demand_df.copy()

    # sort by date
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    demand_df = demand_df.sort_values("date").set_index("date")

    # resample to fill in missing dates
    demand_df = demand_df.resample("D").asfreq()
    demand_df.reset_index(inplace=True)

    # mark missing values
    demand_df["is_imputed_demand"] = demand_df["demand"].isnull().astype(int)
    undefined_demand_filter = demand_df["demand"].isnull()

    # impute missing values
    demand_df["demand"] = demand_df["demand"].fillna(
        demand_df["demand"].rolling(ROLL_PERIOD, min_periods=1, center=True).mean()
    )

    # forward fill
    demand_df["demand"] = demand_df["demand"].ffill()
    demand_df.loc[undefined_demand_filter, "is_imputed_demand"] = 1

    return demand_df
