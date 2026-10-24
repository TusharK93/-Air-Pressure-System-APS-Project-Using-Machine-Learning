
import os
import tempfile
from pathlib import Path
from contextlib import nullcontext

import mlflow
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
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
# MLFLOW / DAGSHUB CONFIGURATION
# =========================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/tukumawat73/aps-fault-detection.mlflow",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "APS-Failure-Prediction",
)

MLFLOW_ENABLED = False


def configure_mlflow():
    """Configure MLflow tracking without training a model."""

    global MLFLOW_ENABLED

    try:
        # Credentials are read from Render Environment variables.
        # Never hard-code your DagsHub token here.
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username

        if password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Set the experiment for traces and tracking operations.
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        MLFLOW_ENABLED = True

        logging.info("MLflow tracking configured.")
        logging.info("MLflow URI: %s", MLFLOW_TRACKING_URI)
        logging.info(
            "MLflow experiment: %s",
            MLFLOW_EXPERIMENT_NAME,
        )

    except Exception:
        MLFLOW_ENABLED = False
        logging.exception(
            "MLflow configuration failed. "
            "The API will continue without MLflow tracing."
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
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "APS Prediction API",
        "mlflow_enabled": MLFLOW_ENABLED,
        "mlflow_experiment": MLFLOW_EXPERIMENT_NAME,
    }


# =========================================================
# HOME
# =========================================================

@app.get("/")
def index():
    return RedirectResponse(url="/docs")


# =========================================================
# MLFLOW TRACE DIAGNOSTIC
# Does not load a dataset or train a model.
# =========================================================

@app.get("/debug-mlflow-trace")
@mlflow.trace
def debug_mlflow_trace():
    return {
        "status": "success",
        "message": "Render-to-MLflow trace test",
        "training_performed": False,
        "mlflow_enabled": MLFLOW_ENABLED,
        "experiment": MLFLOW_EXPERIMENT_NAME,
    }


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
# TRAINING TRACE
# Training runs only when /train is explicitly requested.
# =========================================================

def run_training_with_trace(train_pipeline):
    """Run training with an MLflow trace when available."""

    if not MLFLOW_ENABLED:
        logging.warning(
            "MLflow is disabled; running without tracing."
        )
        return train_pipeline.run_pipeline()

    try:
        # This span records pipeline execution without automatically
        # logging datasets, model artifacts, or uploaded data.
        with mlflow.start_span(
            name="APS_Training_Pipeline"
        ) as span:
            span.set_inputs({
                "pipeline": "TrainPipeline",
                "operation": "training",
            })

            result = train_pipeline.run_pipeline()

            span.set_outputs({
                "status": "completed",
            })

            return result

    except Exception:
        logging.exception(
            "Training failed. Check Render logs for the full traceback."
        )
        raise


@app.get("/train")
def train_route():
    try:
        logging.info("APS TRAINING REQUESTED")

        if not os.getenv("MONGO_DB_URL"):
            raise RuntimeError(
                "MONGO_DB_URL environment variable "
                "is not set on Render."
            )

        train_pipeline = TrainPipeline()

        run_training_with_trace(train_pipeline)

        model_resolver = ModelResolver(
            model_dir=SAVED_MODEL_DIR
        )

        if not model_resolver.is_model_exists():
            raise RuntimeError(
                "Training returned, but no saved model was found."
            )

        return {
            "status": "success",
            "message": "Training completed successfully.",
            "model_path": model_resolver.get_best_model_path(),
            "mlflow_enabled": MLFLOW_ENABLED,
            "experiment": MLFLOW_EXPERIMENT_NAME,
        }

    except Exception as e:
        logging.exception("TRAINING FAILED")

        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}",
        ) from e


# =========================================================
# PREDICTION TRACE
# Only metadata is recorded; CSV values are not passed to MLflow.
# =========================================================

def run_prediction_with_trace(model, dataframe):
    """Predict with a trace containing metadata, not row contents."""

    if not MLFLOW_ENABLED:
        return model.predict(dataframe)

    with mlflow.start_span(
        name="APS_Model_Prediction"
    ) as span:
        span.set_inputs({
            "row_count": int(len(dataframe)),
            "feature_count": int(len(dataframe.columns)),
            "features": [str(c) for c in dataframe.columns],
            "model_type": type(model).__name__,
        })

        predictions = model.predict(dataframe)

        span.set_outputs({
            "prediction_count": int(len(predictions)),
            "status": "completed",
        })

        return predictions


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    output_path = None

    try:
        if (
            not file.filename
            or not file.filename.lower().endswith(".csv")
        ):
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
                    "Verify the saved model directory."
                ),
            )

        # Read directly from the uploaded file stream to avoid
        # creating an additional copy of the uploaded bytes.
        df = pd.read_csv(file.file)

        # Normalize missing-value markers.
        df.replace("na", np.nan, inplace=True)

        # Load schema.
        schema = read_yaml_file(SCHEMA_FILE_PATH)
        drop_columns = schema.get("drop_columns", [])

        df.drop(
            columns=drop_columns,
            inplace=True,
            errors="ignore",
        )

        # Remove target if supplied in the CSV.
        if "class" in df.columns:
            df.drop(columns=["class"], inplace=True)

        # Match training feature order.
        feature_columns = [
            col
            for col in schema["numerical_columns"]
            if col not in drop_columns
        ]

        # Add missing expected features.
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[feature_columns]

        # Load the saved preprocessing/model wrapper.
        model_path = model_resolver.get_best_model_path()
        model = load_object(model_path)

        # Run prediction with metadata-only tracing.
        prediction = run_prediction_with_trace(model, df)

        # Convert numeric target predictions back to labels.
        df["prediction"] = prediction
        df["prediction"] = df["prediction"].replace(
            TargetValueMapping().reverse_mapping()
        )

        # Use a unique output file for concurrent requests.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="aps_prediction_",
            delete=False,
            newline="",
            encoding="utf-8",
        ) as output_file:
            output_path = output_file.name

        df.to_csv(output_path, index=False)

        return FileResponse(
            path=output_path,
            filename="prediction_output.csv",
            media_type="text/csv",
            background=BackgroundTask(
                lambda: Path(output_path).unlink(missing_ok=True)
            ),
        )

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("PREDICTION FAILED")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        ) from e

    finally:
        await file.close()

        # Delete the output if an error happened before FileResponse
        # took responsibility for sending and cleaning it up.
        # Successful responses clean up through BackgroundTask.
        # FileResponse is returned before this block finishes, so
        # avoid deleting a successful output here.


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", APP_PORT)),
    )
import os
import mlflow
from fastapi import HTTPException

@app.get("/debug-mlflow-trace")
@mlflow.trace
def debug_mlflow_trace():
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise HTTPException(
            status_code=500,
            detail="MLFLOW_TRACKING_URI is missing",
        )

    return {
        "status": "success",
        "message": "Live MLflow trace test",
        "training_performed": False,
    }
