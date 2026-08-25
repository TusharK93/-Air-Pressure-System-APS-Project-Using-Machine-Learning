import sys
import traceback

from sensor.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from sensor.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)

from sensor.component.data_ingestion import DataIngestion
from sensor.component.data_validation import DataValidation
from sensor.component.data_transformation import DataTransformation
from sensor.component.data_trainer import ModelTrainer
from sensor.component.model_evaluation import ModelEvaluation
from sensor.component.model_pusher import ModelPusher

from sensor.exception import SensorException


class TrainPipeline:

    def __init__(self):

        try:

            self.training_pipeline_config = (
                TrainingPipelineConfig()
            )

            self.is_pipeline_running = False

            # Initialize artifacts
            self.data_ingestion_artifact = None
            self.data_validation_artifact = None
            self.data_transformation_artifact = None
            self.model_trainer_artifact = None
            self.model_evaluation_artifact = None
            self.model_pusher_artifact = None

        except Exception as e:

            raise SensorException(
                e,
                sys
            )

    # =========================================================
    # DATA INGESTION
    # =========================================================

    def start_data_ingestion(
        self
    ) -> DataIngestionArtifact:

        try:

            print("=" * 60)
            print("DATA INGESTION STARTED")
            print("=" * 60)

            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=
                self.training_pipeline_config
            )

            data_ingestion = DataIngestion(
                data_ingestion_config=
                data_ingestion_config
            )

            data_ingestion_artifact = (
                data_ingestion
                .initiate_data_ingestion()
            )

            print(
                "DATA INGESTION ARTIFACT:",
                data_ingestion_artifact
            )

            print("=" * 60)
            print("DATA INGESTION COMPLETED")
            print("=" * 60)

            return data_ingestion_artifact

        except Exception as e:

            traceback.print_exc()

            raise SensorException(
                e,
                sys
            )

    # =========================================================
    # DATA VALIDATION
    # =========================================================

    def start_data_validation(
        self,
        data_ingestion_artifact:
        DataIngestionArtifact
    ) -> DataValidationArtifact:

        try:

            print("=" * 60)
            print("DATA VALIDATION STARTED")
            print("=" * 60)

            data_validation_config = (
                DataValidationConfig(
                    training_pipeline_config=
                    self.training_pipeline_config
                )
            )

            data_validation = DataValidation(
                data_ingestion_artifact=
                data_ingestion_artifact,

                data_validation_config=
                data_validation_config
            )

            data_validation_artifact = (
                data_validation
                .initiate_data_validation()
            )

            print(
                "DATA VALIDATION ARTIFACT:",
                data_validation_artifact
            )

            print("=" * 60)
            print("DATA VALIDATION COMPLETED")
            print("=" * 60)

            return data_validation_artifact

        except Exception as e:

            traceback.print_exc()

            raise SensorException(
                e,
                sys
            )

    # =========================================================
    # DATA TRANSFORMATION
    # =========================================================

    def start_data_transformation(
        self,
        data_validation_artifact:
        DataValidationArtifact
    ) -> DataTransformationArtifact:

        try:

            print("=" * 60)
            print("DATA TRANSFORMATION STARTED")
            print("=" * 60)

            data_transformation_config = (
                DataTransformationConfig(
                    training_pipeline_config=
                    self.training_pipeline_config
                )
            )

            data_transformation = DataTransformation(
                data_validation_artifact=
                data_validation_artifact,

                data_transformation_config=
                data_transformation_config
            )

            data_transformation_artifact = (
                data_transformation
                .initiate_data_transformation()
            )

            print(
                "DATA TRANSFORMATION ARTIFACT:",
                data_transformation_artifact
            )

            print("=" * 60)
            print("DATA TRANSFORMATION COMPLETED")
            print("=" * 60)

            return data_transformation_artifact

        except Exception as e:

            traceback.print_exc()

            raise SensorException(
                e,
                sys
            )

    # =========================================================
    # MODEL TRAINING
    # =========================================================

    def start_model_trainer(
        self,
        data_transformation_artifact:
        DataTransformationArtifact
    ) -> ModelTrainerArtifact:

        try:

            print("=" * 60)
            print("MODEL TRAINER STARTED")
            print("=" * 60)

            model_trainer_config = (
                ModelTrainerConfig(
                    training_pipeline_config=
                    self.training_pipeline_config
                )
            )

            model_trainer = ModelTrainer(
                model_trainer_config=
                model_trainer_config,

                data_transformation_artifact=
                data_transformation_artifact
            )

            model_trainer_artifact = (
                model_trainer
                .initiate_model_trainer()
            )

            print(
                "MODEL TRAINER ARTIFACT:",
                model_trainer_artifact
            )

            print("=" * 60)
            print("MODEL TRAINER COMPLETED")
            print("=" * 60)

            return model_trainer_artifact

        except Exception as e:

            traceback.print_exc()

            raise SensorException(
                e,
                sys
            )

    # =========================================================
    # MODEL EVALUATION
    # =========================================================

    def start_model_evaluation(
        self,
        data_validation_artifact:
        DataValidationArtifact,

        model_trainer_artifact:
        ModelTrainerArtifact

    ) -> ModelEvaluationArtifact:

        try:

            print("=" * 60)
            print("MODEL EVALUATION STARTED")
            print("=" * 60)

            model_eval_config = (
                ModelEvaluationConfig(
                    training_pipeline_config=
                    self.training_pipeline_config
                )
            )

            model_eval = ModelEvaluation(
                model_eval_config=
                model_eval_config,

                data_validation_artifact=
                data_validation_artifact,

                model_trainer_artifact=
                model_trainer_artifact
            )

            model_eval_artifact = (
                model_eval
                .initiate_model_evaluation()
            )

            print(
                "MODEL EVALUATION ARTIFACT:",
                model_eval_artifact
            )

            print(
                "MODEL ACCEPTED:",
                model_eval_artifact.is_model_accepted
            )

            print("=" * 60)
            print("MODEL EVALUATION COMPLETED")
            print("=" * 60)

            return model_eval_artifact

        except Exception as e:

            traceback.print_exc()

            raise SensorException(
                e,
                sys
            )

    # =========================================================
    # MODEL PUSHER
    # =========================================================

    def start_model_pusher(
        self,
        model_eval_artifact:
        ModelEvaluationArtifact

    ) -> ModelPusherArtifact:

        try:

            print("=" * 60)
            print("MODEL PUSHER STARTED")
            print("=" * 60)

            model_pusher_config = (
                ModelPusherConfig(
                    training_pipeline_config=
                    self.training_pipeline_config
                )
            )

            model_pusher = ModelPusher(
                model_pusher_config=
                model_pusher_config,

                model_eval_artifact=
                model_eval_artifact
            )

            model_pusher_artifact = (
                model_pusher
                .initiate_model_pusher()
            )

            print(
                "MODEL PUSHER ARTIFACT:",
                model_pusher_artifact
            )

            print("=" * 60)
            print("MODEL PUSHER COMPLETED")
            print("=" * 60)

            return model_pusher_artifact

        except Exception as e:

            traceback.print_exc()

            raise SensorException(
                e,
                sys
            )

    # =========================================================
    # RUN COMPLETE TRAINING PIPELINE
    # =========================================================

    def run_pipeline(self):

        try:

            self.is_pipeline_running = True

            print("\n")
            print("=" * 70)
            print("            TRAINING PIPELINE STARTED")
            print("=" * 70)

            # =================================================
            # 1. DATA INGESTION
            # =================================================

            self.data_ingestion_artifact = (
                self.start_data_ingestion()
            )

            if self.data_ingestion_artifact is None:

                raise RuntimeError(
                    "Data ingestion artifact is None"
                )

            # =================================================
            # 2. DATA VALIDATION
            # =================================================

            self.data_validation_artifact = (
                self.start_data_validation(
                    data_ingestion_artifact=
                    self.data_ingestion_artifact
                )
            )

            if self.data_validation_artifact is None:

                raise RuntimeError(
                    "Data validation artifact is None"
                )

            # =================================================
            # 3. DATA TRANSFORMATION
            # =================================================

            self.data_transformation_artifact = (
                self.start_data_transformation(
                    data_validation_artifact=
                    self.data_validation_artifact
                )
            )

            if self.data_transformation_artifact is None:

                raise RuntimeError(
                    "Data transformation artifact is None"
                )

            # =================================================
            # 4. MODEL TRAINING
            # =================================================

            self.model_trainer_artifact = (
                self.start_model_trainer(
                    data_transformation_artifact=
                    self.data_transformation_artifact
                )
            )

            if self.model_trainer_artifact is None:

                raise RuntimeError(
                    "Model trainer artifact is None"
                )

            # =================================================
            # 5. MODEL EVALUATION
            # =================================================

            self.model_evaluation_artifact = (
                self.start_model_evaluation(

                    data_validation_artifact=
                    self.data_validation_artifact,

                    model_trainer_artifact=
                    self.model_trainer_artifact
                )
            )

            if self.model_evaluation_artifact is None:

                raise RuntimeError(
                    "Model evaluation artifact is None"
                )

            # =================================================
            # 6. MODEL ACCEPTANCE
            # =================================================

            print("=" * 70)

            print(
                "MODEL ACCEPTED:",
                self.model_evaluation_artifact
                .is_model_accepted
            )

            print("=" * 70)

            # =================================================
            # 7. MODEL PUSHER
            # =================================================

            if (
                self.model_evaluation_artifact
                .is_model_accepted
            ):

                self.model_pusher_artifact = (
                    self.start_model_pusher(
                        model_eval_artifact=
                        self.model_evaluation_artifact
                    )
                )

                print("=" * 70)
                print("MODEL ACCEPTED")
                print("MODEL PUSHER COMPLETED")
                print("=" * 70)

            else:

                print("=" * 70)
                print("MODEL REJECTED")
                print(
                    "Model will NOT be pushed."
                )
                print("=" * 70)

            # =================================================
            # PIPELINE COMPLETED
            # =================================================

            print("=" * 70)
            print("TRAINING PIPELINE COMPLETED")
            print("=" * 70)

            return self.model_evaluation_artifact

        except Exception as e:

            print("\n")
            print("=" * 70)
            print("             TRAINING PIPELINE FAILED")
            print("=" * 70)

            traceback.print_exc()

            print("=" * 70)

            raise SensorException(
                e,
                sys
            )

        finally:

            self.is_pipeline_running = False

            print(
                "Pipeline running status:",
                self.is_pipeline_running
            )