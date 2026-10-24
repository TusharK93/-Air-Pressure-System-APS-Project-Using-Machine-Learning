
import os
import traceback
from contextlib import nullcontext

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from sensor.constant.training_pipeline import (
    SAVED_MODEL_DIR,
    SCHEMA_FILE_PATH,
)
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import read_yaml_file, load_object
from sensor.constant.application import APP_PORT
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.logger import logging


# =========================================================
# MLFLOW CONFIGURATION
# =========================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "APS-Failure-Prediction",
)

MLFLOW_ENABLED = False
MLFLOW_AUTOLOG_ENABLED = False


def configure_mlflow():
    """
    Configure remote MLflow tracking and scikit-learn autologging.
    """

    global MLFLOW_ENABLED, MLFLOW_AUTOLOG_ENABLED

    if not MLFLOW_TRACKING_URI:
        logging.warning(
            "MLFLOW_TRACKING_URI is missing. "
            "MLflow tracking is disabled."
        )
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Automatically record supported scikit-learn training details.
        mlflow.sklearn.autolog(
                  log_models=False,
                  log_input_examples=False,
                  log_model_signatures=False,
                  log_datasets=False,
             )

        MLFLOW_ENABLED = True
        MLFLOW_AUTOLOG_ENABLED = True

        logging.info("MLflow tracking configured.")
        logging.info(
            "MLflow experiment: %s",
            MLFLOW_EXPERIMENT_NAME,
        )

    except Exception:
        MLFLOW_ENABLED = False
        MLFLOW_AUTOLOG_ENABLED = False

        logging.exception(
            "MLflow configuration failed."
        )


configure_mlflow()


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="APS Prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# TRAINING WITH MLFLOW RUN + TRACE
# =========================================================

def run_training_with_trace(train_pipeline):
    """
    Create an MLflow run for the training pipeline.

    Scikit-learn autologging records supported estimator
    parameters, metrics and model artifacts during fit().
    """

    if not MLFLOW_ENABLED:
        logging.warning(
            "Running training without MLflow tracking."
        )
        return train_pipeline.run_pipeline()

    # If the tracking server cannot create a run, do not prevent
    # the existing training pipeline from running.
    try:
        active_run = mlflow.start_run(
            run_name="APS_Training_Pipeline"
        )
    except Exception:
        logging.exception(
            "Could not start MLflow run. "
            "Running training without MLflow."
        )
        return train_pipeline.run_pipeline()

    # Once the run is created, let training exceptions propagate.
    # This avoids accidentally running a failed training job twice.
    with active_run:
        mlflow.set_tag("project", "APS-Failure-Prediction")
        mlflow.set_tag("operation", "training")
        mlflow.set_tag("training_pipeline", "TrainPipeline")

        # Add an explicit trace when supported by the installed
        # MLflow version and tracking backend.
        with mlflow.start_span(
            name="APS_Training_Pipeline"
        ) as span:
            span.set_inputs({
                "pipeline": "TrainPipeline",
                "project": "APS Failure Prediction",
            })

            result = train_pipeline.run_pipeline()

            span.set_outputs({
                "status": "completed",
            })

            mlflow.set_tag("training_status", "completed")

            return result


# =========================================================
# PREDICTION TRACE
# =========================================================

def run_prediction_with_trace(model, dataframe):
    """
    Trace prediction without recording uploaded CSV values.
    """

    if not MLFLOW_ENABLED:
        return model.predict(dataframe)

    with mlflow.start_span(
        name="APS_Model_Prediction"
    ) as span:

        span.set_inputs({
            "row_count": int(len(dataframe)),
            "feature_count": int(len(dataframe.columns)),
            "features": [
                str(column) for column in dataframe.columns
            ],
            "model_type": type(model).__name__,
        })

        predictions = model.predict(dataframe)

        span.set_outputs({
            "prediction_count": int(len(predictions)),
            "status": "completed",
        })

        return predictions


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "APS Prediction API",
        "mlflow_enabled": MLFLOW_ENABLED,
        "mlflow_autolog_enabled": MLFLOW_AUTOLOG_ENABLED,
        "mlflow_experiment": MLFLOW_EXPERIMENT_NAME,
    }


# =========================================================
# HOME
# =========================================================

@app.get("/")
def index():
    return RedirectResponse(url="/docs")


# =========================================================
# DEBUG MODEL
# =========================================================

@app.get("/debug-model")
def debug_model():
    try:
        model_dir = SAVED_MODEL_DIR

        result = {
            "saved_model_dir": model_dir,
            "absolute_path": os.path.abspath(model_dir),
            "exists": os.path.exists(model_dir),
            "contents": [],
        }

        if os.path.exists(model_dir):
            result["contents"] = os.listdir(model_dir)

        model_resolver = ModelResolver(model_dir=model_dir)

        result["model_exists"] = model_resolver.is_model_exists()

        if result["model_exists"]:
            result["model_path"] = (
                model_resolver.get_best_model_path()
            )

        return result

    except Exception as e:
        logging.exception("Debug model endpoint failed.")

        return {
            "error": str(e),
            "saved_model_dir": SAVED_MODEL_DIR,
            "absolute_path": os.path.abspath(SAVED_MODEL_DIR),
        }


# =========================================================
# TRAIN
# =========================================================

@app.get("/train")
def train_route():
    try:
        logging.info("=" * 60)
        logging.info("TRAINING STARTED")
        logging.info("=" * 60)

        mongo_url = os.getenv("MONGO_DB_URL")

        if not mongo_url:
            raise RuntimeError(
                "MONGO_DB_URL environment variable "
                "is not set on Render."
            )

        train_pipeline = TrainPipeline()

        run_training_with_trace(train_pipeline)

        logging.info("TRAINING COMPLETED")

        model_resolver = ModelResolver(
            model_dir=SAVED_MODEL_DIR
        )

        if model_resolver.is_model_exists():
            model_path = model_resolver.get_best_model_path()

            return {
                "status": "success",
                "message": "Training completed successfully.",
                "model_path": model_path,
                "mlflow_enabled": MLFLOW_ENABLED,
                "experiment": MLFLOW_EXPERIMENT_NAME,
            }

        return Response(
            content="Training completed, but model was not found.",
            status_code=500,
        )

    except Exception as e:
        logging.exception("TRAINING FAILED")

        return Response(
            content=f"Training failed: {repr(e)}",
            status_code=500,
        )


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    temp_file_path = "prediction.csv"
    output_path = "prediction_output.csv"

    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a CSV file.",
            )

        model_resolver = ModelResolver(
            model_dir=SAVED_MODEL_DIR
        )

        if not model_resolver.is_model_exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Model not found on server. "
                    "Run training and verify the model directory."
                ),
            )

        # Read the uploaded CSV.
        uploaded_content = await file.read()

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_content)

        df = pd.read_csv(temp_file_path)

        # Normalize missing-value markers.
        df.replace("na", np.nan, inplace=True)

        # Load the project schema.
        schema = read_yaml_file(SCHEMA_FILE_PATH)
        drop_columns = schema.get("drop_columns", [])

        df.drop(
            columns=drop_columns,
            inplace=True,
            errors="ignore",
        )

        # Remove the target if the input contains it.
        if "class" in df.columns:
            df.drop(columns=["class"], inplace=True)

        # Match the training feature order.
        feature_columns = [
            col
            for col in schema["numerical_columns"]
            if col not in drop_columns
        ]

        # Add missing expected columns.
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[feature_columns]

        # Load the trained model.
        model_path = model_resolver.get_best_model_path()
        model = load_object(model_path)

        # Predict and trace the operation.
        prediction = run_prediction_with_trace(model, df)

        df["prediction"] = prediction

        # Convert encoded labels to original labels.
        df["prediction"] = df["prediction"].replace(
            TargetValueMapping().reverse_mapping()
        )

        df.to_csv(output_path, index=False)

        return FileResponse(
            output_path,
            filename="prediction_output.csv",
            media_type="text/csv",
        )

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("PREDICTION FAILED")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    finally:
        # Clean up the uploaded input file.
        # The output file is retained until the response is sent.
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except OSError:
            logging.warning("Could not remove temporary input CSV.")


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":
    from uvicorn import run

    run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", APP_PORT)),
    )
