
import gc
import os
import sys

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from sensor.entity.config_entity import DataTransformationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import mlflow_utils as mlu
from sensor.utils.main_utils import (
    save_numpy_array_data,
    save_object,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise SensorException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e, sys) from e

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            logging.info("Creating preprocessing object")

            simple_imputer = SimpleImputer(
                strategy="constant",
                fill_value=0,
            )

            robust_scaler = RobustScaler()

            preprocessor = Pipeline(
                steps=[
                    ("Imputer", simple_imputer),
                    ("RobustScaler", robust_scaler),
                ]
            )

            return preprocessor

        except Exception as e:
            raise SensorException(e, sys) from e

    @staticmethod
    def convert_target(target_series: pd.Series) -> np.ndarray:
        """Convert APS target labels into numeric binary labels."""

        target_series = target_series.replace(
            {
                "neg": 0,
                "pos": 1,
            }
        )

        target_series = pd.to_numeric(
            target_series,
            errors="coerce",
        )

        if target_series.isna().any():
            raise ValueError(
                "Target column contains missing or invalid labels."
            )

        target_values = target_series.to_numpy(dtype=np.int8)

        if not np.isin(target_values, [0, 1]).all():
            raise ValueError(
                "Target column must contain only binary labels 0 and 1."
            )

        return target_values

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        try:
            logging.info("DATA TRANSFORMATION STARTED")

            # 1. Read train and test data.
            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )

            test_df = self.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            logging.info("Train shape: %s", train_df.shape)
            logging.info("Test shape: %s", test_df.shape)

            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' is missing "
                    "from training data."
                )

            if TARGET_COLUMN not in test_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' is missing "
                    "from testing data."
                )

            # 2. Separate features and target labels.
            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN]
            )

            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN]
            )

            y_train = self.convert_target(train_df[TARGET_COLUMN])
            y_test = self.convert_target(test_df[TARGET_COLUMN])

            if list(input_feature_train_df.columns) != list(
                input_feature_test_df.columns
            ):
                raise ValueError(
                    "Training and testing feature columns do not match."
                )

            del train_df, test_df
            gc.collect()

            # 3. Create the preprocessing pipeline.
            preprocessor = self.get_data_transformer_object()

            # 4. Fit only on training data to prevent data leakage.
            preprocessor.fit(input_feature_train_df)

            # 5. Transform training features.
            transformed_input_train_feature = preprocessor.transform(
                input_feature_train_df
            )

            transformed_input_train_feature = np.asarray(
                transformed_input_train_feature,
                dtype=np.float32,
            )

            del input_feature_train_df
            gc.collect()

            # 6. Transform test features using the fitted preprocessor.
            transformed_input_test_feature = preprocessor.transform(
                input_feature_test_df
            )

            transformed_input_test_feature = np.asarray(
                transformed_input_test_feature,
                dtype=np.float32,
            )

            del input_feature_test_df
            gc.collect()

            logging.info("FEATURE TRANSFORMATION COMPLETED")

            # 7. Do not resample the test set.
            # No SMOTE or SMOTETomek is applied.

            # 8. Combine transformed features and target labels.
            train_arr = np.column_stack(
                (
                    transformed_input_train_feature,
                    y_train,
                )
            ).astype(np.float32, copy=False)

            test_arr = np.column_stack(
                (
                    transformed_input_test_feature,
                    y_test,
                )
            ).astype(np.float32, copy=False)

            del transformed_input_train_feature
            del transformed_input_test_feature
            del y_train, y_test
            gc.collect()

            logging.info("Transformed train shape: %s", train_arr.shape)
            logging.info("Transformed test shape: %s", test_arr.shape)

            # 9. Log preprocessing parameters and metrics to MLflow.
            mlu.log_params(
                {
                    "imputer": "SimpleImputer(constant, fill_value=0)",
                    "scaler": "RobustScaler",
                    "resampling": "none",
                }
            )

            mlu.log_metrics(
                {
                    "transformed_train_rows": train_arr.shape[0],
                    "transformed_test_rows": test_arr.shape[0],
                    "transformed_feature_count": train_arr.shape[1] - 1,
                    "transformed_train_pos_rate": float(
                        train_arr[:, -1].mean()
                    ),
                    "transformed_test_pos_rate": float(
                        test_arr[:, -1].mean()
                    ),
                }
            )

            # 10. Create output directories.
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.transformed_train_file_path
                ),
                exist_ok=True,
            )

            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.transformed_test_file_path
                ),
                exist_ok=True,
            )

            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.transformed_object_file_path
                ),
                exist_ok=True,
            )

            # 11. Save transformed training data.
            save_numpy_array_data(
                file_path=(
                    self.data_transformation_config.transformed_train_file_path
                ),
                array=train_arr,
            )

            del train_arr
            gc.collect()

            # 12. Save transformed test data.
            save_numpy_array_data(
                file_path=(
                    self.data_transformation_config.transformed_test_file_path
                ),
                array=test_arr,
            )

            del test_arr
            gc.collect()

            # 13. Save the fitted preprocessor.
            save_object(
                file_path=(
                    self.data_transformation_config.transformed_object_file_path
                ),
                obj=preprocessor,
            )

            logging.info("train.npy SAVED")
            logging.info("test.npy SAVED")
            logging.info("preprocessing.pkl SAVED")

            # 14. Return the transformation artifact.
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=(
                    self.data_transformation_config.transformed_object_file_path
                ),
                transformed_train_file_path=(
                    self.data_transformation_config.transformed_train_file_path
                ),
                transformed_test_file_path=(
                    self.data_transformation_config.transformed_test_file_path
                ),
            )

            logging.info("DATA TRANSFORMATION FINISHED SUCCESSFULLY")

            return data_transformation_artifact

        except Exception as e:
            logging.exception("DATA TRANSFORMATION FAILED")
            raise SensorException(e, sys) from e
