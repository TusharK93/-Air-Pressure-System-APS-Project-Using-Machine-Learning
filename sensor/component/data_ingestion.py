
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.exception import SensorException
from sensor.data_access.sensor_data import SensorData
from sensor.constant.training_pipeline import (
    SCHEMA_FILE_PATH,
    TARGET_COLUMN,
)
from sensor.utils.main_utils import read_yaml_file
from sensor.utils import mlflow_utils as mlu


class DataIngestion:
    def __init__(
        self,
        data_ingestion_config: config_entity.DataIngestionConfig,
    ):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e, sys) from e

    # ==========================================================
    # EXPORT FEATURE STORE
    # ==========================================================

    def export_data_into_feature_store(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            drop_columns = self._schema_config.get("drop_columns", [])

            existing_drop_columns = [
                col for col in drop_columns if col in dataframe.columns
            ]

            dataframe = dataframe.drop(
                columns=existing_drop_columns,
                errors="ignore",
            )

            feature_store_file_path = (
                self.data_ingestion_config.feature_store_file_path
            )

            os.makedirs(
                os.path.dirname(feature_store_file_path),
                exist_ok=True,
            )

            dataframe.to_csv(
                feature_store_file_path,
                index=False,
                header=True,
            )

            logging_message = (
                f"Feature Store Saved: {feature_store_file_path}"
            )
            print(logging_message)

            return dataframe

        except Exception as e:
            raise SensorException(e, sys) from e

    # ==========================================================
    # TRAIN TEST SPLIT
    # ==========================================================

    def split_data_as_train_test(
        self,
        dataframe: pd.DataFrame,
    ) -> artifact_entity.DataIngestionArtifact:
        try:
            if len(dataframe) < 2:
                raise ValueError(
                    f"Dataset contains only {len(dataframe)} rows."
                )

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.test_size,
                random_state=42,
            )

            train_file_path = (
                self.data_ingestion_config.train_file_path
            )
            test_file_path = (
                self.data_ingestion_config.test_file_path
            )

            os.makedirs(
                os.path.dirname(train_file_path),
                exist_ok=True,
            )
            os.makedirs(
                os.path.dirname(test_file_path),
                exist_ok=True,
            )

            train_set.to_csv(
                train_file_path,
                index=False,
                header=True,
            )

            test_set.to_csv(
                test_file_path,
                index=False,
                header=True,
            )

            # MLflow: record split sizes.
            mlu.log_metrics(
                {
                    "train_rows": len(train_set),
                    "test_rows": len(test_set),
                }
            )

            return artifact_entity.DataIngestionArtifact(
                feature_store_file_path=(
                    self.data_ingestion_config.feature_store_file_path
                ),
                train_file_path=train_file_path,
                test_file_path=test_file_path,
            )

        except Exception as e:
            raise SensorException(e, sys) from e

    # ==========================================================
    # MAIN INGESTION
    # ==========================================================

    def initiate_data_ingestion(
        self,
    ) -> artifact_entity.DataIngestionArtifact:
        try:
            print("=" * 60)
            print("DATA INGESTION STARTED")
            print("=" * 60)

            sensor_data = SensorData()

            # Import CSV into MongoDB.
            project_root = Path(__file__).resolve().parents[2]
            csv_path = project_root / "aps_failure_training_set1.csv"

            print(f"Reading CSV: {csv_path}")

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Training CSV was not found: {csv_path}"
                )

            inserted = sensor_data.save_csv_file(
                file_path=str(csv_path),
                collection_name=(
                    self.data_ingestion_config.collection_name
                ),
            )

            print(f"Inserted Records: {inserted}")

            dataframe = sensor_data.export_collection_as_dataframe(
                collection_name=(
                    self.data_ingestion_config.collection_name
                )
            )

            print(f"Data Shape After MongoDB: {dataframe.shape}")

            if dataframe.empty:
                raise ValueError(
                    "MongoDB returned an empty dataframe."
                )

            dataframe.replace("na", np.nan, inplace=True)

            # MLflow: dataset-level parameters and metrics.
            # Repeated CSV imports may create duplicate MongoDB records,
            # depending on the implementation of save_csv_file().
            mlu.log_params(
                {
                    "source_csv": csv_path.name,
                    "test_size": (
                        self.data_ingestion_config.test_size
                    ),
                    "mongo_collection": (
                        self.data_ingestion_config.collection_name
                    ),
                }
            )

            mlu.log_metrics(
                {
                    "mongo_records_inserted_this_run": int(inserted),
                    "rows_from_mongo": int(dataframe.shape[0]),
                    "columns_from_mongo": int(dataframe.shape[1]),
                    "missing_value_ratio": float(
                        dataframe.isna().mean().mean()
                    ),
                }
            )

            # Log target distribution when the target column is available.
            if TARGET_COLUMN in dataframe.columns:
                target_values = (
                    dataframe[TARGET_COLUMN]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                positives = int((target_values == "pos").sum())
                negatives = int((target_values == "neg").sum())

                mlu.log_metrics(
                    {
                        "class_pos_count": positives,
                        "class_neg_count": negatives,
                        "class_pos_rate": (
                            positives / max(positives + negatives, 1)
                        ),
                    }
                )

            # Export feature store and create train/test datasets.
            dataframe = self.export_data_into_feature_store(dataframe)

            artifact = self.split_data_as_train_test(dataframe)

            print("=" * 60)
            print("DATA INGESTION COMPLETED")
            print("=" * 60)

            return artifact

        except Exception as e:
            raise SensorException(e, sys) from e
