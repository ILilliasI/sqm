import logging
from pathlib import Path

import pandas as pd

from .constants import COHESION_METRICS, COMPLEXITY_METRICS
from .dataset import filter_dataset, prepare_dataset

logger = logging.getLogger(__name__)


def get_results(
    dataset: pd.DataFrame, columns: list[str], aggregate_func: str = "mean"
) -> pd.DataFrame:
    """
    Return aggregated data for specified columns.
    """
    assert aggregate_func in [
        "mean",
        "count",
        "sum",
    ], "`aggregate_func` must be one of 'mean', 'count' or 'sum'"

    if aggregate_func == "mean":
        return dataset.groupby("group")[columns].mean()
    elif aggregate_func == "count":
        return dataset.groupby("group")[columns].count()
    return dataset.groupby("group")[columns].sum()


def evaluate_metrics(data_folder_path: str, output_folder_path: str) -> None:
    dataset = prepare_dataset(data_folder_path=data_folder_path)
    logger.info(f"Collected dataset from {data_folder_path}")

    filtered_dataset = filter_dataset(dataset=dataset)
    logger.info(f"Filtered dataset from {data_folder_path}")
    cohesion_results = get_results(dataset=filtered_dataset, columns=COHESION_METRICS)
    complexity_results = get_results(
        dataset=filtered_dataset, columns=COMPLEXITY_METRICS
    )
    loc_results = get_results(
        dataset=filtered_dataset, columns=["loc"], aggregate_func="sum"
    )
    lc_results = get_results(
        dataset=filtered_dataset, columns=["loc"], aggregate_func="mean"
    )
    classes_results = get_results(
        dataset=filtered_dataset, columns=["group"], aggregate_func="count"
    )

    additional_results = loc_results.join(lc_results, rsuffix="_mean")
    additional_results = additional_results.join(classes_results)

    cohesion_results.to_csv(str(Path(output_folder_path) / "cohesion_res.csv"))
    complexity_results.to_csv(str(Path(output_folder_path) / "complexity_res.csv"))
    additional_results.to_csv(str(Path(output_folder_path) / "additional_res.csv"))
    logger.info(f"Saved evaliation results to {output_folder_path}")
