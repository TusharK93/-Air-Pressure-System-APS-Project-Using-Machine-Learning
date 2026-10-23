
import os
import sys
import pandas as pd

from sensor.exception import SensorException
from sensor.logger import logging

from sensor.entity.artifact_entity import (
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from sensor.entity.config_entity import ModelEvaluationConfig

from sensor.ml.metric.classification_metric import (
    get_classification_score,
)
from sensor.ml.model.estimator import (
    SensorModel,
    ModelResolver,
    TargetValueMapping,
)
from sensor.utils.main_utils import (
    save_object,
    load_object,
    write_yaml_file,
)
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.utils import mlflow_utils as mlu


class ModelEvaluation:

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig = None,
        data_validation_artifact: DataValidationArtifact = None,
        model_trainer_artifact: ModelTrainerArtifact = None,
        model_eval_config: ModelEvaluationConfig = None,
    ):
        try:
            # Accept either configuration parameter name.
            self.model_eval_config = (
                model_eval_config or model_evaluation_config
            )

            if self.model_eval_config is None:
                raise ValueError(
                    "ModelEvaluationConfig must be provided."
                )

            if data_validation_artifact is None:
                raise ValueError(
                    "DataValidationArtifact must be provided."
                )

            if model_trainer_artifact is None:
                raise ValueError(
                    "ModelTrainerArtifact must be provided."
                )

            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    def initiate_model_evaluation(
        self,
    ) -> ModelEvaluationArtifact:
        try:
            valid_train_file_path = (
                self.data_validation_artifact.valid_train_file_path
            )
            valid_test_file_path = (
                self.data_validation_artifact.valid_test_file_path
            )

            # Load and combine validated datasets.
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)
            df = pd.concat(
                [train_df, test_df],
                ignore_index=True,
            )

            y_true = df[TARGET_COLUMN].copy()
            y_true.replace(
                TargetValueMapping().to_dict(),
                inplace=True,
            )
            df.drop(
                TARGET_COLUMN,
                axis=1,
                inplace=True,
            )

            train_model_file_path = (
                self.model_trainer_artifact.trained_model_file_path
            )

            model_resolver = ModelResolver()
            is_model_accepted = True

            # First model: there is no deployed model to compare against.
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=None,
                    best_model_path=None,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=(
                        self.model_trainer_artifact.test_metric_artifact
                    ),
                    best_model_metric_artifact=None,
                )

                logging.info(
                    "No previous model exists. Accepting the first model."
                )

                mlu.log_metrics(
                    {
                        "eval_is_first_model": 1,
                        "eval_is_model_accepted": 1,
                        "eval_rows": len(df),
                    }
                )

                mlu.log_classification_metrics(
                    "eval_candidate",
                    self.model_trainer_artifact.test_metric_artifact,
                )

                model_eval_report = (
                    model_evaluation_artifact.__dict__
                )

                write_yaml_file(
                    self.model_eval_config.report_file_path,
                    model_eval_report,
                )

                mlu.log_artifact(
                    self.model_eval_config.report_file_path,
                    "model_evaluation",
                )

                logging.info(
                    "Model evaluation artifact: %s",
                    model_evaluation_artifact,
                )

                return model_evaluation_artifact

            # Load the existing model and the new candidate.
            latest_model_path = model_resolver.get_best_model_path()

            latest_model = load_object(
                file_path=latest_model_path
            )
            train_model = load_object(
                file_path=train_model_file_path
            )

            # Evaluate both models on the same evaluation data.
            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            trained_metric = get_classification_score(
                y_true,
                y_trained_pred,
            )
            latest_metric = get_classification_score(
                y_true,
                y_latest_pred,
            )

            improved_accuracy = (
                trained_metric.f1_score - latest_metric.f1_score
            )

            # Accept the candidate only if it exceeds the threshold.
            is_model_accepted = (
                improved_accuracy
                > self.model_eval_config.change_threshold
            )

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric,
            )

            # Record evaluation metrics and decision in MLflow.
            mlu.log_metrics(
                {
                    "eval_is_first_model": 0,
                    "eval_is_model_accepted": (
                        1 if is_model_accepted else 0
                    ),
                    "eval_improved_f1": improved_accuracy,
                    "eval_change_threshold": (
                        self.model_eval_config.change_threshold
                    ),
                    "eval_rows": len(df),
                }
            )

            mlu.log_classification_metrics(
                "eval_candidate",
                trained_metric,
            )
            mlu.log_classification_metrics(
                "eval_deployed",
                latest_metric,
            )

            mlu.set_tags(
                {
                    "compared_against_model": latest_model_path,
                    "model_decision": (
                        "accepted" if is_model_accepted else "rejected"
                    ),
                }
            )

            # Save the evaluation report.
            model_eval_report = model_evaluation_artifact.__dict__

            write_yaml_file(
                self.model_eval_config.report_file_path,
                model_eval_report,
            )

            mlu.log_artifact(
                self.model_eval_config.report_file_path,
                "model_evaluation",
            )

            logging.info(
                "Model evaluation artifact: %s",
                model_evaluation_artifact,
            )

            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
