
import gc
import os
import sys

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

from sensor.utils.main_utils import (
    load_numpy_array_data,
    save_object,
    load_object,
)
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import mlflow_utils as mlu

from sensor.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from sensor.entity.config_entity import ModelTrainerConfig

from sensor.ml.metric.classification_metric import (
    get_classification_score,
)
from sensor.ml.model.estimator import SensorModel


# APS Scania challenge cost: false alarm = 10, missed failure = 500.
APS_COST_FALSE_POSITIVE = 10
APS_COST_FALSE_NEGATIVE = 500


def log_extended_metrics(prefix, y_true, y_pred, model, x):
    """Log confusion matrix, APS cost, accuracy, and ROC-AUC to MLflow."""
    try:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        mlu.log_metrics(
            {
                f"{prefix}_tn": int(tn),
                f"{prefix}_fp": int(fp),
                f"{prefix}_fn": int(fn),
                f"{prefix}_tp": int(tp),
                f"{prefix}_aps_cost": int(
                    APS_COST_FALSE_POSITIVE * fp
                    + APS_COST_FALSE_NEGATIVE * fn
                ),
                f"{prefix}_accuracy": float(
                    (tn + tp) / max(tn + fp + fn + tp, 1)
                ),
            }
        )
    except Exception:
        logging.exception("Could not compute confusion-matrix metrics")

    try:
        if len(np.unique(y_true)) == 2:
            mlu.log_metric(
                f"{prefix}_roc_auc",
                roc_auc_score(
                    y_true,
                    model.predict_proba(x)[:, 1],
                ),
            )
    except Exception:
        logging.exception("Could not compute ROC-AUC")


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    def perform_hyper_paramter_tunig(self):
        """Hyperparameter tuning is not run in the production pipeline."""
        return None

    def train_model(self, x_train, y_train):
        try:
            # Determine class imbalance using training data only.
            negatives = int(np.count_nonzero(y_train == 0))
            positives = int(np.count_nonzero(y_train == 1))

            if negatives == 0 or positives == 0:
                raise ValueError(
                    "Training data must contain both classes: 0 and 1."
                )

            scale_pos_weight = negatives / positives

            logging.info(
                "Training class counts: negatives=%s, positives=%s",
                negatives,
                positives,
            )

            # Memory-conscious XGBoost configuration.
            xgb_clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                max_bin=64,
                n_jobs=1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbosity=1,
            )

            # MLflow: log training configuration.
            mlu.log_params(
                {
                    "model_type": "XGBClassifier",
                    "objective": "binary:logistic",
                    "n_estimators": 100,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "tree_method": "hist",
                    "max_bin": 64,
                    "scale_pos_weight": round(scale_pos_weight, 4),
                    "random_state": 42,
                }
            )

            mlu.log_metrics(
                {
                    "fit_train_negatives": negatives,
                    "fit_train_positives": positives,
                }
            )

            logging.info("Starting XGBoost model training")
            xgb_clf.fit(x_train, y_train)
            logging.info("XGBoost model training completed")

            return xgb_clf

        except Exception as e:
            raise SensorException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("MODEL TRAINING STARTED")

            # 1. Load transformed arrays.
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            if train_arr.ndim != 2 or test_arr.ndim != 2:
                raise ValueError(
                    "Transformed train/test arrays must be two-dimensional."
                )

            if (
                train_arr.shape[1] < 2
                or test_arr.shape[1] != train_arr.shape[1]
            ):
                raise ValueError(
                    "Invalid transformed arrays: features or target are missing."
                )

            logging.info("Train array shape: %s", train_arr.shape)
            logging.info("Test array shape: %s", test_arr.shape)

            # 2. Separate features and target labels.
            x_train = np.asarray(train_arr[:, :-1], dtype=np.float32)
            y_train = train_arr[:, -1].astype(np.int32, copy=False)

            x_test = np.asarray(test_arr[:, :-1], dtype=np.float32)
            y_test = test_arr[:, -1].astype(np.int32, copy=False)

            del train_arr, test_arr
            gc.collect()

            if not np.isin(y_train, [0, 1]).all():
                raise ValueError(
                    "Training target contains values other than 0 and 1."
                )

            if not np.isin(y_test, [0, 1]).all():
                raise ValueError(
                    "Testing target contains values other than 0 and 1."
                )

            # 3. Train model.
            model = self.train_model(
                x_train=x_train,
                y_train=y_train,
            )

            # 4. Calculate training metrics.
            y_train_pred = model.predict(x_train)

            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )

            logging.info(
                "Training metrics: %s",
                classification_train_metric,
            )

            mlu.log_classification_metrics(
                "train",
                classification_train_metric,
            )

            log_extended_metrics(
                "train",
                y_train,
                y_train_pred,
                model,
                x_train,
            )

            if (
                classification_train_metric.f1_score
                <= self.model_trainer_config.expected_accuracy
            ):
                raise ValueError(
                    "Trained model did not meet the configured "
                    "expected training F1 score."
                )

            del y_train_pred
            gc.collect()

            # 5. Calculate test metrics.
            y_test_pred = model.predict(x_test)

            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )

            logging.info(
                "Testing metrics: %s",
                classification_test_metric,
            )

            mlu.log_classification_metrics(
                "test",
                classification_test_metric,
            )

            log_extended_metrics(
                "test",
                y_test,
                y_test_pred,
                model,
                x_test,
            )

            # 6. Check the train-test F1 gap.
            diff = abs(
                classification_train_metric.f1_score
                - classification_test_metric.f1_score
            )

            logging.info("Train-test F1 difference: %s", diff)

            mlu.log_metrics(
                {
                    "train_test_f1_gap": diff,
                    "expected_train_f1_threshold": (
                        self.model_trainer_config.expected_accuracy
                    ),
                    "overfit_gap_threshold": (
                        self.model_trainer_config
                        .overfitting_underfitting_threshold
                    ),
                }
            )

            if (
                diff
                > self.model_trainer_config.overfitting_underfitting_threshold
            ):
                raise ValueError(
                    "Train-test F1 difference exceeds the configured "
                    "threshold. Model needs further experimentation."
                )

            del y_test_pred
            gc.collect()

            # 7. Load fitted preprocessor.
            preprocessor = load_object(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_object_file_path
                )
            )

            # 8. Save model and preprocessing wrapper.
            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir_path, exist_ok=True)

            sensor_model = SensorModel(
                preprocessor=preprocessor,
                model=model,
            )

            save_object(
                file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),
                obj=sensor_model,
            )

            logging.info(
                "Trained model saved at: %s",
                self.model_trainer_config.trained_model_file_path,
            )

            mlu.log_artifact(
                self.model_trainer_config.trained_model_file_path,
                "trained_model",
            )

            mlu.log_metric(
                "trained_model_size_mb",
                os.path.getsize(
                    self.model_trainer_config.trained_model_file_path
                ) / (1024 * 1024),
            )

            # 9. Return model trainer artifact.
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )

            logging.info(
                "Model trainer artifact: %s",
                model_trainer_artifact,
            )
            logging.info("MODEL TRAINING COMPLETED SUCCESSFULLY")

            return model_trainer_artifact

        except Exception as e:
            logging.exception("MODEL TRAINING FAILED")
            raise SensorException(e, sys) from e
