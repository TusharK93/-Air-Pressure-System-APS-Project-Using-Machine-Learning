
import gc
import sys
import traceback

from sensor.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
 
from sensor.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)

from sensor.exception import SensorException
from sensor.logger import logging


from sensor.component.data_ingestion import DataIngestion
from sensor.component.data_validation import DataValidation
from sensor.component.data_transformation import DataTransformation
from sensor.component.data_trainer import ModelTrainer
from sensor.component.model_evaluation import ModelEvaluation
from sensor.component.model_pusher import ModelPusher



class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

        self.data_ingestion_artifact = None
        self.data_validation_artifact = None
        self.data_transformation_artifact = None
        self.model_trainer_artifact = None
        self.model_evaluation_artifact = None
        self.model_pusher_artifact = None

        self.is_pipeline_running = False

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")

            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_ingestion = DataIngestion(
                data_ingestion_config=data_ingestion_config
            )

            artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed")
            return artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_validation = DataValidation(
                data_validation_config=data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )

            artifact = data_validation.initiate_data_validation()

            logging.info("Data validation completed")
            return artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_data_transformation(
        self,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_transformation = DataTransformation(
                data_transformation_config=data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )

            artifact = data_transformation.initiate_data_transformation()

            logging.info("Data transformation completed")
            return artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")

            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )

            artifact = model_trainer.initiate_model_trainer()

            logging.info("Model training completed")
            return artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_model_evaluation(
        self,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation")

            model_evaluation_config = ModelEvaluationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_evaluation = ModelEvaluation(
                model_evaluation_config=model_evaluation_config,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            artifact = model_evaluation.initiate_model_evaluation()

            logging.info("Model evaluation completed")
            return artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_model_pusher(
        self,
        model_eval_artifact: ModelEvaluationArtifact,
    ) -> ModelPusherArtifact:
        try:
            logging.info("Starting model pushing")

            model_pusher_config = ModelPusherConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_pusher = ModelPusher(
                model_pusher_config=model_pusher_config,
                model_eval_artifact=model_eval_artifact,
            )

            artifact = model_pusher.initiate_model_pusher()

            logging.info("Model pushing completed")
            return artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    def run_pipeline(self) -> ModelEvaluationArtifact:
        """Run all six stages of the APS model training pipeline."""

        if self.is_pipeline_running:
            raise RuntimeError("This training pipeline is already running.")

        self.is_pipeline_running = True

        try:
            logging.info("=" * 60)
            logging.info("APS TRAINING PIPELINE STARTED")
            logging.info("=" * 60)

            # 1. Data ingestion
            logging.info("[1/6] DATA INGESTION")
            self.data_ingestion_artifact = self.start_data_ingestion()

            if self.data_ingestion_artifact is None:
                raise RuntimeError("Data ingestion returned no artifact.")

            gc.collect()

            # 2. Data validation
            logging.info("[2/6] DATA VALIDATION")
            self.data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=self.data_ingestion_artifact
            )

            if self.data_validation_artifact is None:
                raise RuntimeError("Data validation returned no artifact.")

            # Release the pipeline's reference to the ingestion artifact.
            self.data_ingestion_artifact = None
            gc.collect()

            # 3. Data transformation
            logging.info("[3/6] DATA TRANSFORMATION")
            self.data_transformation_artifact = (
                self.start_data_transformation(
                    data_validation_artifact=self.data_validation_artifact
                )
            )

            if self.data_transformation_artifact is None:
                raise RuntimeError(
                    "Data transformation returned no artifact."
                )

            gc.collect()

            # 4. Model training
            logging.info("[4/6] MODEL TRAINING")
            self.model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=self.data_transformation_artifact
            )

            if self.model_trainer_artifact is None:
                raise RuntimeError("Model training returned no artifact.")

            # Release the pipeline's reference to the transformation artifact.
            self.data_transformation_artifact = None
            gc.collect()

            # 5. Model evaluation
            logging.info("[5/6] MODEL EVALUATION")
            self.model_evaluation_artifact = self.start_model_evaluation(
                data_validation_artifact=self.data_validation_artifact,
                model_trainer_artifact=self.model_trainer_artifact,
            )

            if self.model_evaluation_artifact is None:
                raise RuntimeError("Model evaluation returned no artifact.")

            # These artifacts are no longer needed by later pipeline stages.
            self.data_validation_artifact = None
            self.model_trainer_artifact = None
            gc.collect()

            # 6. Model acceptance and pushing
            logging.info("[6/6] MODEL ACCEPTANCE AND PUSHING")

            is_model_accepted = (
                self.model_evaluation_artifact.is_model_accepted
            )

            logging.info("Model accepted: %s", is_model_accepted)

            if is_model_accepted:
                self.model_pusher_artifact = self.start_model_pusher(
                    model_eval_artifact=self.model_evaluation_artifact
                )
                logging.info("Model pushing completed successfully.")
            else:
                logging.warning(
                    "Model rejected. The model will not be pushed."
                )

            # Keep a local reference to the result before cleanup.
            result = self.model_evaluation_artifact

            logging.info("=" * 60)
            logging.info("APS TRAINING PIPELINE COMPLETED")
            logging.info("=" * 60)

            return result

        except Exception as e:
            logging.error("APS TRAINING PIPELINE FAILED")
            logging.error(traceback.format_exc())
            raise SensorException(e, sys) from e

        finally:
            self.is_pipeline_running = False

            # Clear references held by this pipeline instance.
            self.data_ingestion_artifact = None
            self.data_validation_artifact = None
            self.data_transformation_artifact = None
            self.model_trainer_artifact = None
            self.model_evaluation_artifact = None
            self.model_pusher_artifact = None

            gc.collect()

            logging.info(
                "Training pipeline cleanup completed. Running: %s",
                self.is_pipeline_running,
            )
