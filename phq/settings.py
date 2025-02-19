# selected trend features to be used in the model
TREND_FEATURES = [
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_month_of_year",
    "cos_month_of_year",
    "sin_day_of_year",
    "cos_day_of_year",
    "year",
    "demand_lag7",
]

# define the categorical features
CATEGORICAL_FEATURES = [
    "year",
]

# XGBoost hyperparameters
XGBOOST__N_ESTIMATORS = [50, 100]
XGBOOST__LEARNING_RATE = [0.01, 0.1, 0.2]
XGBOOST__MAX_DEPTH = [3, 5, 7]

EVAL_FORECAST_HORIZON = 7  # forecast period used during evaluation
EVAL_TRAIN_RATIO = 0.6  # the ratio of the dataset to be used for training
N_SPLITS = 5  # the number of splits for cross-validation
