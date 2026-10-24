
import os
import tempfile
from pathlib import Path

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
MLFLOW_CONFIG_ERROR = None
MLFLOW_EXPERIMENT_ID = None


def configure_mlflow():
    """Configure MLflow using Render environment variables."""

    global MLFLOW_ENABLED
    global MLFLOW_CONFIG_ERROR
    global MLFLOW_EXPERIMENT_ID

    try:
        # Do not hard-code credentials in this file.
        # Configure these in Render Environment settings.
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username

        if password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        experiment = mlflow.set_experiment(
            MLFLOW_EXPERIMENT_NAME
        )

        MLFLOW_EXPERIMENT_ID = experiment.experiment_id
        MLFLOW_ENABLED = True
        MLFLOW_CONFIG_ERROR = None

        logging.info("MLflow configuration succeeded.")
        logging.info("MLflow experiment: %s", MLFLOW_EXPERIMENT_NAME)
        logging.info("MLflow experiment ID: %s", MLFLOW_EXPERIMENT_ID)

    except Exception as exc:
        MLFLOW_ENABLED = False
        MLFLOW_CONFIG_ERROR = (
            f"{type(exc).__name__}: {exc}"
        )

        logging.exception("MLflow configuration failed.")


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
# TRACE FLUSH HELPER
# =========================================================

def flush_mlflow_traces():
    """
    Flush pending asynchronous trace writes when the installed
    MLflow version exposes the flush function.
    """

    flush_function = getattr(
        mlflow,
        "flush_trace_async_logging",
        None,
    )

    if callable(flush_function):
        flush_function()
        return True

    logging.warning(
        "This MLflow version does not expose "
        "flush_trace_async_logging()."
    )
    return False


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "APS Prediction API",
        "mlflow_enabled": MLFLOW_ENABLED,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "mlflow_experiment": MLFLOW_EXPERIMENT_NAME,
        "mlflow_experiment_id": MLFLOW_EXPERIMENT_ID,
        "mlflow_configuration_error": MLFLOW_CONFIG_ERROR,
    }


# =========================================================
# HOME
# =========================================================

@app.get("/")
def index():
    return RedirectResponse(url="/docs")


# =========================================================
# REAL LIVE TRACE PROBE
# No dataset loading or model training.
# =========================================================

@mlflow.trace
def live_trace_probe():
    return {
        "message": "APS Render live tracing test",
        "training_performed": False,
    }


@app.get("/debug-mlflow-trace")
def debug_mlflow_trace():
    """
    Execute a traced function, flush pending writes, and try to
    verify that traces can be queried from the configured backend.
    """

    if not MLFLOW_ENABLED:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "MLflow configuration failed.",
                "error": MLFLOW_CONFIG_ERROR,
            },
        )

    try:
        # Execute the decorated function: this creates a trace.
        result = live_trace_probe()

        # Flush asynchronous trace writes if supported.
        flush_available = flush_mlflow_traces()

        # Try to query recent traces. This is an additional
        # verification step, not the trace creation itself.
        trace_query_status = "not_checked"
        trace_count = None
        trace_query_error = None

        search_traces = getattr(
            mlflow,
            "search_traces",
            None,
        )

        if callable(search_traces):
            try:
                traces = search_traces(
                    experiment_ids=[MLFLOW_EXPERIMENT_ID],
                    max_results=10,
                )

                trace_count = len(traces)
                trace_query_status = "query_succeeded"

            except Exception as query_exc:
                trace_query_status = "query_failed"
                trace_query_error = (
                    f"{type(query_exc).__name__}: {query_exc}"
                )

                logging.exception(
                    "Could not query MLflow traces."
                )

        return {
            "status": "success",
            "message": "The traced function executed.",
            "training_performed": False,
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "experiment_id": MLFLOW_EXPERIMENT_ID,
            "trace_function_executed": True,
            "trace_flush_available": flush_available,
            "trace_query_status": trace_query_status,
            "recent_trace_count": trace_count,
            "trace_query_error": trace_query_error,
            "result": result,
            "note": (
                "A successful function execution does not alone "
                "prove that the trace is visible in the DagsHub UI. "
                "Check the experiment's Traces tab."
            ),
        }

    except Exception as exc:
        logging.exception("Live MLflow trace test failed.")

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Live MLflow trace test failed.",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc


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

        model_resolver = ModelResolver(
            model_dir=model_dir
        )

        result["model_exists"] = (
            model_resolver.is_model_exists()
        )

        if result["model_exists"]:
            result["model_path"] = (
                model_resolver.get_best_model_path()
            )

        return result

    except Exception as exc:
        logging.exception("Debug model endpoint failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Debug model failed: {exc}",
        ) from exc


# =========================================================
# TRAINING TRACE
# Training runs only when /train is explicitly requested.
# =========================================================

def run_training_with_trace(train_pipeline):
    if not MLFLOW_ENABLED:
        logging.warning(
            "MLflow is disabled; training will run without tracing."
        )
        return train_pipeline.run_pipeline()

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

    except Exception as exc:
        logging.exception("TRAINING FAILED")

        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {exc}",
        ) from exc


# =========================================================
# PREDICTION TRACE
# Records metadata only, not uploaded CSV row contents.
# =========================================================

def run_prediction_with_trace(model, dataframe):
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
# PREDICT
# =========================================================

@app.post("/predict")
async def predict_route(
    file: UploadFile = File(...),
):
    output_path = None
    response_owns_output = False

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

        df = pd.read_csv(file.file)
        df.replace("na", np.nan, inplace=True)

        schema = read_yaml_file(SCHEMA_FILE_PATH)
        drop_columns = schema.get("drop_columns", [])

        df.drop(
            columns=drop_columns,
            inplace=True,
            errors="ignore",
        )

        if "class" in df.columns:
            df.drop(
                columns=["class"],
                inplace=True,
            )

        feature_columns = [
            column
            for column in schema["numerical_columns"]
            if column not in drop_columns
        ]

        for column in feature_columns:
            if column not in df.columns:
                df[column] = np.nan

        df = df[feature_columns]

        model_path = model_resolver.get_best_model_path()
        model = load_object(model_path)

        prediction = run_prediction_with_trace(
            model,
            df,
        )

        df["prediction"] = prediction
        df["prediction"] = df["prediction"].replace(
            TargetValueMapping().reverse_mapping()
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="aps_prediction_",
            delete=False,
            newline="",
            encoding="utf-8",
        ) as output_file:
            output_path = output_file.name

        df.to_csv(
            output_path,
            index=False,
        )

        response = FileResponse(
            path=output_path,
            filename="prediction_output.csv",
            media_type="text/csv",
            background=BackgroundTask(
                lambda: Path(output_path).unlink(
                    missing_ok=True
                )
            ),
        )

        # FileResponse's background task now owns cleanup.
        response_owns_output = True
        return response

    except HTTPException:
        raise

    except Exception as exc:
        logging.exception("PREDICTION FAILED")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    finally:
        await file.close()

        # Clean up files created before a successful FileResponse
        # took ownership of the output file.
        if output_path and not response_owns_output:
            try:
                Path(output_path).unlink(missing_ok=True)
            except OSError:
                logging.exception(
                    "Could not remove temporary prediction file."
                )


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
