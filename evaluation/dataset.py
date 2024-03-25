import math
from pathlib import Path

import pandas as pd

from .constants import (
    ER_GROUP_EXCEPTIONS,
    ER_GROUP_NAME,
    ER_SUFFIXES,
    JAVA_FILE_EXTENSION,
    METRIC_NAMES,
    REST_GROUP_NAME,
    UTILS_GROUP_NAME,
    UTILS_SUFFIXES,
)


def separate_groups(
    df: pd.DataFrame, er_group_exceptions: list[str] | None = None
) -> pd.DataFrame:
    """
    Add group label (-Er/-Or, -Utils, Rest groups) and class name (without .java extension) to dataset classes.
    """
    groups_dataset = df.copy()
    groups_dataset["class_name"] = groups_dataset["java_file"].map(
        lambda java_file: java_file.split("/")[-1][: -len(JAVA_FILE_EXTENSION)]
    )
    groups_dataset["group"] = groups_dataset["class_name"].apply(
        lambda class_name: (
            ER_GROUP_NAME
            if any(class_name.lower().endswith(suffix) for suffix in ER_SUFFIXES)
            else REST_GROUP_NAME
        )
    )
    if er_group_exceptions:
        groups_dataset["group"] = groups_dataset.apply(
            lambda row: (
                REST_GROUP_NAME
                if any(
                    row["class_name"].lower().endswith(suffix.lower())
                    for suffix in er_group_exceptions
                )
                else row["group"]
            ),
            axis=1,
        )
    groups_dataset["group"] = groups_dataset.apply(
        lambda row: (
            UTILS_GROUP_NAME
            if any(
                row["class_name"].lower().endswith(suffix) for suffix in UTILS_SUFFIXES
            )
            else row["group"]
        ),
        axis=1,
    )
    return groups_dataset


def get_metrics_df(data_folder_path: str, metric_names: list[str]) -> pd.DataFrame:
    """
    Collect dataset for specified metrics.
    """
    columns = ["java_file", *metric_names]
    df = pd.DataFrame(columns=columns)
    for user_dir in Path(data_folder_path).iterdir():
        if user_dir.is_dir():
            for repo_dir in Path(user_dir).iterdir():
                if repo_dir.is_dir():
                    for file_path in Path(repo_dir).iterdir():
                        if file_path.name.lower() == "all.csv":
                            try:
                                new_df = pd.read_csv(file_path, low_memory=False)
                            except Exception:
                                continue
                            new_df = new_df[
                                (new_df[metric_names].notna().all(axis=1))
                                & (new_df[metric_names] != "-").all(axis=1)
                            ][columns]
                            if not new_df.empty:
                                df = pd.concat(
                                    [df if not df.empty else None, new_df], sort=False
                                )
    df[metric_names] = df[metric_names].apply(pd.to_numeric, errors="coerce")
    return df


def prepare_dataset(data_folder_path: str) -> pd.DataFrame:
    """
    Collect specified metrics from CAM dataset, filter dataset, divide dataset info three groups of interest (-Er/-Or, -Utils, Rest).
    """
    return separate_groups(
        df=get_metrics_df(data_folder_path=data_folder_path, metric_names=METRIC_NAMES),
        er_group_exceptions=ER_GROUP_EXCEPTIONS,
    )


def filter_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    return filter_rest_group(filter_dataset_outliers(dataset))


def filter_dataset_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter outliers by (LoC - blank lines) value.
    """
    loc_blanks_diff = df["loc"] - df["blanks"]

    min_limit = math.floor(loc_blanks_diff.quantile(0.01))
    max_limit = math.ceil(loc_blanks_diff.quantile(0.99))

    return df[(loc_blanks_diff > min_limit) & (loc_blanks_diff < max_limit)]


def filter_rest_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter classes from rest group with static methods and attributes.
    """
    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df["group"] != REST_GROUP_NAME) | ((filtered_df["smtds"] == 0))
    ]
    filtered_df = filtered_df[
        (filtered_df["group"] != REST_GROUP_NAME) | ((filtered_df["sattrs"] == 0))
    ]
    return filtered_df
