import logging
import time
from io import StringIO

import pandas as pd
from predicthq import Client
from predicthq.endpoints.v1.beam.schemas import CreateAnalysisResponse


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_beam_analysis(
    name: str,
    lat: float,
    lon: float,
    demand_df: pd.DataFrame,
    access_token: str,
    industry: str | None = None,
    sleep: int = 5,
) -> CreateAnalysisResponse:
    # create the PredictHQ client
    phq = Client(access_token=access_token)

    # create a new analysis
    analysis = phq.beam.analysis.create(
        name=name,
        location__geopoint={"lat": str(lat), "lon": str(lon)},
        demand_type__industry=industry,
    )

    analysis_id = analysis.analysis_id
    log.info(f"Created analysis '{name}' with id: {analysis_id}")

    # upload demand data
    phq.beam.analysis.upload_demand(
        analysis_id=analysis_id, json=StringIO(demand_df.to_json(orient="records"))
    )
    log.info(f"Uploaded demand data for analysis {analysis_id}")

    # wait for demand to be processed
    completed = False
    while not completed:
        log.info("Waiting for feature importance ...")
        time.sleep(sleep)

        analysis = phq.beam.analysis.get(analysis_id=analysis_id)
        completed = analysis.processing_completed.feature_importance

    # assign the analysis id to the analysis if it is not already set
    if not hasattr(analysis, "analysis_id") or analysis.analysis_id is None:
        analysis.analysis_id = analysis_id

    # get feature importance results
    feature_importance = phq.beam.analysis.get_feature_importance(
        analysis_id=analysis_id
    )
    feature_importance = [
        x for x in feature_importance.feature_importance if x.important is True
    ]

    if not feature_importance:
        log.warning(f"No feature importance found for analysis {analysis.analysis_id}")

    return analysis
