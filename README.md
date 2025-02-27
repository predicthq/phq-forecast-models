# PredictHQ Forecast Notebook

PredictHQ’s Forecast models, built with XGBoost, bring event-driven demand forecasting to production scale. They seamlessly integrate PredictHQ Event Features into a high-performing model, making it easy to incorporate real-world event data for more accurate forecasts.

The [Forecast Notebook](https://github.com/predicthq/phq-forecast-models) demonstrates how to apply PredictHQ’s model in real forecasting scenarios, guiding users through feature engineering, model evaluation, and forecasting. It compares a baseline model (using only time trends) with a PredictHQ-enhanced model to quantify the value of event intelligence.

By using this notebook, users can quickly validate the impact of event-driven forecasting and best practices. It also serves as a stepping stone to the upcoming PredictHQ Forecasts API, a scalable solution for event-driven demand forecasting.

## Prerequisites
The notebook supports two execution modes, controlled by the `RUN_SETTING` variable. Choose the appropriate mode based on whether you’re using the provided sample data, or your own demand with PredictHQ data fetched via our APIs.

### Required Software:
- Python 3.10+
- Required dependencies installed (`requirements.txt`)

### PredictHQ API Access (optional):
- Test out the notebook using the provided sample data, or use your own data alongside [PredictHQ APIs](https://docs.predicthq.com/api/overview).
- [Follow our guide](https://docs.predicthq.com/api/overview/authenticating) for generating an API access token.

### Files:
- `data/sample_config.json`: Sample configuration file.
- `data/sample_demand.csv`: Sample demand data.
- `data/sample_event_features.csv`: Sample historical PredictHQ event features for model evaluation (snapshot captured in Feb 2025).
- `data/sample_forecasting_features.csv`: Sample forward-looking PredictHQ event features for forecasting (snapshot captured in Feb 2025).
- `phq/settings.py`: Model evaluation and forecast settings.

### Data Requirements:

- The provided sample data is based on a fictional restaurant located in London.
- Demand data - if using your own demand data, please follow the format of the sample data (`date`, `demand`).
- PredictHQ Features API data - in order to fetch PredictHQ Features relevant to your own business and location you will be guided through using Beam for Feature Importance and Features API to fetch the relevant features.

## Using Your Own Data

### PredictHQ API Token
Generate a [PredictHQ API Token](https://docs.predicthq.com/api/overview/authenticating) and store it in an env var so it can be used in the notebook:

```python
PHQ_ACCESS_TOKEN = os.environ.get("PHQ_ACCESS_TOKEN") or "XXXXXX"
```

### Configuration

Daily Demand CSV File
- The demand dataset must be formatted as a CSV file with the required columns (`date`, `demand`).

Configuration JSON File
- `lat`: Latitude of the location.
- `lon`: Longitude of the location.
- `industry`: The industry associated with the demand data. A list of supported industries can be found in [our API docs](https://docs.predicthq.com/api/beam/create-an-analysis).
- `name`: The Beam Analysis name (something that makes sense to you).

Example:

```json
{
  "name": "sample_beam_analysis",
  "lat": 51.50396,
  "lon": 0.00476,
  "industry": "restaurants"
}
```

These two files (demand data csv and config json) will be used for creating a Beam Analysis in order to work out which PredictHQ event features are relevant to your business and location. For additional requirements (e.g., minimum demand volume, industry settings), refer to our [Beam API docs](https://docs.predicthq.com/api/beam).

## Model Evaluation

The model evaluation process assesses the accuracy and effectiveness of demand forecast models when PredictHQ Event Features are included. It uses an expanding window forecasting approach, where the model is initially trained on 60% of the dataset and forecasts the period defined by `EVAL_FORECAST_HORIZON`. The initial training window is then iteratively expanded, followed by forecasting the next period.

To optimize the forecast model, grid search with time-series cross-validation is applied to explore different hyperparameter combinations for XGBoost. This includes:

- Number of trees (`XGBOOST__N_ESTIMATORS`)
- Learning rate (`XGBOOST__LEARNING_RATE`)
- Maximum depth of trees (`XGBOOST__MAX_DEPTH`)

`N_SPLITS`-fold time-series cross-validation evaluates each hyperparameter combination. The best-performing set identified through cross-validation is then used to train the final model. Model performance is measured using Mean Absolute Percentage Error (MAPE).

### Configurable Parameters

The default settings are optimized based on best practices and work well for most demand forecasting tasks. However, you can customize the model evaluation by modifying the following parameters in `settings.py` if required:

- Initial Training Window (`EVAL_TRAIN_RATIO`)
  - Default: 0.6 (60% of data for training)
  - Alternatives:
    - 0.5: when rapid shifts occur in demand patterns or limited historical data is available.
    - 0.7+: if historical data is extensive and patterns are stable.
- Forecast Window (`EVAL_FORECAST_HORIZON`)
  - Default: 7 (7 days)
  - Alternatives:
    - 3: Short-term forecasts (e.g., daily adjustments).
    - 14: Long-term forecasts (e.g., biweekly planning).
- Number of Cross-Validation Folds (`N_SPLITS`)
  - Default: 5
  - Alternatives:
    - 3 for faster training.
    - 10 for more robust (but slower) validation.

## Forecast Window

During model evaluation, an expanding window approach is used to simulate future forecasting scenarios. In deployment, the forecast window (`FORECAST_HORIZON`) can be customized to align with specific business needs. This parameter is configurable in the notebook, with a default value of 7 days, but it can be adjusted based on the forecasting objective:

- 3-day horizon: Suitable for short-term operational adjustments.
- 14-day horizon: Useful for longer-term planning, such as biweekly demand projections.

Note: If `EVAL_FORECAST_HORIZON` or  `FORECAST_HORIZON` is configured to be larger than 7 days, the `demand_lag7` feature in `settings.py` must be excluded.

## Resources
- [Forecast Notebook](https://github.com/predicthq/phq-forecast-models)
- [PredictHQ APIs](https://docs.predicthq.com/api/overview)
- [PredictHQ API Token](https://docs.predicthq.com/api/overview/authenticating)
- [Beam](https://docs.predicthq.com/api/beam)
- [PredictHQ Event Features](https://docs.predicthq.com/api/features)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)
