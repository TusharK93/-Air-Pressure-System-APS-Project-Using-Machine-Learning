import os
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pathlib import Path
from sensor.entity import config_entity
from sensor.entity import artifact_entity

from sensor.exception import SensorException

from sensor.data_access.sensor_data import SensorData

from sensor.constant.training_pipeline import SCHEMA_FILE_PATH

from sensor.utils.main_utils import read_yaml_file


class DataIngestion:

    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e, sys)

    # ==========================================================
    # EXPORT FEATURE STORE
    # ==========================================================

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:

            drop_columns = self._schema_config.get("drop_columns", [])

            existing_drop_columns = [
                col for col in drop_columns if col in dataframe.columns
            ]

            dataframe = dataframe.drop(
                columns=existing_drop_columns,
                errors="ignore"
            )

            feature_store_file_path = (
                self.data_ingestion_config.feature_store_file_path
            )

            os.makedirs(
                os.path.dirname(feature_store_file_path),
                exist_ok=True
            )

            dataframe.to_csv(
                feature_store_file_path,
                index=False,
                header=True
            )

            print(f"Feature Store Saved: {feature_store_file_path}")

            return dataframe

        except Exception as e:
            raise SensorException(e, sys)

    # ==========================================================
    # TRAIN TEST SPLIT
    # ==========================================================

    def split_data_as_train_test(
        self,
        dataframe: pd.DataFrame
    ) -> artifact_entity.DataIngestionArtifact:

        try:

            if len(dataframe) < 2:
                raise Exception(
                    f"Dataset contains only {len(dataframe)} rows."
                )

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.test_size,
                random_state=42
            )

            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path

            os.makedirs(
                os.path.dirname(train_file_path),
                exist_ok=True
            )

            train_set.to_csv(
                train_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                test_file_path,
                index=False,
                header=True
            )

            return artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=train_file_path,
                test_file_path=test_file_path
            )

        except Exception as e:
            raise SensorException(e, sys)

    # ==========================================================
    # MAIN INGESTION
    # ==========================================================

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:

        try:

            print("=" * 60)
            print("DATA INGESTION STARTED")
            print("=" * 60)

            sensor_data = SensorData()

            # --------------------------------------------------
            # IMPORT CSV INTO MONGODB EVERY TIME
            # --------------------------------------------------

            project_root = Path(__file__).resolve().parents[2]
            csv_path = project_root / "aps_failure_training_set1.csv"

            print(f"Reading CSV: {csv_path}")

            inserted = sensor_data.save_csv_file(
                file_path=csv_path,
                collection_name=self.data_ingestion_config.collection_name
            )

            print(f"Inserted Records: {inserted}")

            dataframe = sensor_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            print(f"Data Shape After MongoDB: {dataframe.shape}")

            if dataframe.empty:
                raise Exception("MongoDB returned empty dataframe.")

            dataframe.replace("na", np.nan, inplace=True)

            dataframe = self.export_data_into_feature_store(dataframe)

            artifact = self.split_data_as_train_test(dataframe)

            print("=" * 60)
            print("DATA INGESTION COMPLETED")
            print("=" * 60)

            return artifact

        except Exception as e:
            raise SensorException(e, sys)