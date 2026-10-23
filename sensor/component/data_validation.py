
import os
import sys
import pandas as pd

from scipy.stats import ks_2samp

from sensor.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from sensor.entity.config_entity import DataValidationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import mlflow_utils as mlu
from sensor.utils.main_utils import (
    read_yaml_file,
    write_yaml_file,
)
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise SensorException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e, sys) from e

    def validate_number_of_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> bool:
        try:
            schema_columns = self._schema_config["columns"]
            drop_columns = self._schema_config["drop_columns"]

            expected_columns = []

            for item in schema_columns:
                if isinstance(item, dict):
                    column_name = list(item.keys())[0]

                    if column_name not in drop_columns:
                        expected_columns.append(column_name)

            logging.info("Expected columns: %s", len(expected_columns))
            logging.info("Actual columns: %s", len(dataframe.columns))

            missing_columns = [
                column
                for column in expected_columns
                if column not in dataframe.columns
            ]

            if missing_columns:
                logging.error("Missing columns: %s", missing_columns)
                return False

            return True

        except Exception as e:
            raise SensorException(e, sys) from e

    def is_numerical_column_exist(
        self,
        dataframe: pd.DataFrame,
    ) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]

            missing_columns = [
                column
                for column in numerical_columns
                if column not in dataframe.columns
            ]

            if missing_columns:
                logging.error(
                    "Missing numerical columns: %s",
                    missing_columns,
                )
                return False

            return True

        except Exception as e:
            raise SensorException(e, sys) from e

    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05,
    ) -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()

                if d1.empty or d2.empty:
                    p_value = 0.0
                    is_drifted = True
                    status = False
                else:
                    test_result = ks_2samp(d1, d2)
                    p_value = float(test_result.pvalue)
                    is_drifted = p_value < threshold

                    if is_drifted:
                        status = False

                report[column] = {
                    "p_value": p_value,
                    "drift_status": is_drifted,
                }

            drift_report_file_path = (
                self.data_validation_config.drift_report_file_path
            )

            dir_path = os.path.dirname(drift_report_file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(
                file_path=drift_report_file_path,
                content=report,
            )

            # MLflow: log drift metrics and the report.
            drifted_columns = sum(
                1
                for item in report.values()
                if item["drift_status"]
            )

            mlu.log_metrics(
                {
                    "drift_columns_checked": len(report),
                    "drift_columns_detected": drifted_columns,
                    "drift_ratio": (
                        drifted_columns / max(len(report), 1)
                    ),
                    "drift_free": 1 if status else 0,
                }
            )

            mlu.log_artifact(
                drift_report_file_path,
                "data_validation",
            )

            logging.info(
                "Data drift check completed. Drifted columns: %s/%s",
                drifted_columns,
                len(report),
            )

            return status

        except Exception as e:
            raise SensorException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message = ""

            train_file_path = (
                self.data_ingestion_artifact.train_file_path
            )
            test_file_path = (
                self.data_ingestion_artifact.test_file_path
            )

            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)

            # Validate expected columns.
            if not self.validate_number_of_columns(train_dataframe):
                error_message += "Train dataframe column mismatch\n"

            if not self.validate_number_of_columns(test_dataframe):
                error_message += "Test dataframe column mismatch\n"

            # Validate numerical columns.
            if not self.is_numerical_column_exist(train_dataframe):
                error_message += "Train numerical columns missing\n"

            if not self.is_numerical_column_exist(test_dataframe):
                error_message += "Test numerical columns missing\n"

            if error_message:
                raise ValueError(error_message)

            # Detect drift between training and test datasets.
            status = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe,
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=train_file_path,
                valid_test_file_path=test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=(
                    self.data_validation_config.drift_report_file_path
                ),
            )

            logging.info(
                "Data Validation Artifact: %s",
                data_validation_artifact,
            )

            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
