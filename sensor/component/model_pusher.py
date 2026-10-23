
import os
import sys
import shutil

from sensor.exception import SensorException
from sensor.entity.artifact_entity import (
    ModelPusherArtifact,
    ModelEvaluationArtifact,
)
from sensor.entity.config_entity import ModelPusherConfig
from sensor.utils import mlflow_utils as mlu


class ModelPusher:

    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_eval_artifact: ModelEvaluationArtifact,
    ):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            trained_model_path = (
                self.model_eval_artifact.trained_model_path
            )

            # Copy the accepted model to the main model location.
            model_file_path = (
                self.model_pusher_config.model_file_path
            )

            os.makedirs(
                os.path.dirname(model_file_path),
                exist_ok=True,
            )

            shutil.copy(
                src=trained_model_path,
                dst=model_file_path,
            )

            # Save a timestamped model version.
            saved_model_path = (
                self.model_pusher_config.saved_model_path
            )

            os.makedirs(
                os.path.dirname(saved_model_path),
                exist_ok=True,
            )

            shutil.copy(
                src=trained_model_path,
                dst=saved_model_path,
            )

            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path,
                model_file_path=model_file_path,
            )

            # MLflow: record the accepted model location.
            mlu.set_tags(
                {
                    "pushed_model_path": saved_model_path,
                    "pushed_model_version": os.path.basename(
                        os.path.dirname(saved_model_path)
                    ),
                }
            )

            mlu.log_artifact(
                saved_model_path,
                "accepted_model",
            )

            return model_pusher_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
