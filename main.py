
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
).strip()

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "aps-fault-detection",
).strip()

MLFLOW_ENABLED = False
MLFLOW_CONFIG_ERROR = None
MLFLOW_EXPERIMENT_ID = None


def configure_mlflow():
    """Configure the MLflow tracking backend and target experiment."""
    global MLFLOW_ENABLED
    global MLFLOW_CONFIG_ERROR
    global MLFLOW_EXPERIMENT_ID

    try:
        if not MLFLOW_TRACKING_URI:
            raise ValueError("MLFLOW_TRACKING_URI is empty.")

        if not os.getenv("MLFLOW_TRACKING_USERNAME"):
            raise ValueError(
                "MLFLOW_TRACKING_USERNAME is missing."
            )

        if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
            raise ValueError(
                "MLFLOW_TRACKING_PASSWORD is missing."
            )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        experiment = mlflow.set_experiment(
            MLFLOW_EXPERIMENT_NAME
        )

        MLFLOW_EXPERIMENT_ID = experiment.experiment_id
        MLFLOW_ENABLED = True
        MLFLOW_CONFIG_ERROR = None

        logging.info("MLflow configured successfully.")
        logging.info("MLflow URI: %s", MLFLOW_TRACKING_URI)
        logging.info("MLflow experiment: %s", MLFLOW_EXPERIMENT_NAME)
        logging.info("MLflow experiment ID: %s", MLFLOW_EXPERIMENT_ID)

    except Exception as exc:
        MLFLOW_ENABLED = False
        MLFLOW_CONFIG_ERROR = f"{type(exc).__name__}: {exc}"
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
# MLFLOW HELPERS
# =========================================================

def flush_mlflow_traces():
    """Flush queued trace writes if this MLflow version supports it."""
    flush_function = getattr(
        mlflow,
        "flush_trace_async_logging",
        None,
    )

    if callable(flush_function):
        flush_function()
        return True

    logging.warning(
        "MLflow trace flush is unavailable in this installed version."
    )
    return False


def require_mlflow():
    if not MLFLOW_ENABLED or not MLFLOW_EXPERIMENT_ID:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "MLflow is not configured.",
                "error": MLFLOW_CONFIG_ERROR,
            },
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
# LIVE TRACE TEST
# Does not train a model or read a dataset.
# =========================================================

@mlflow.trace(name="APS_Render_Live_Trace_Test")
def live_trace_probe():
    return {
        "message": "APS Render live tracing test",
        "training_performed": False,
    }


@app.get("/debug-mlflow-trace")
def debug_mlflow_trace():
    require_mlflow()

    try:
        # Ensure the intended experiment is selected for this request.
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.set_experiment(
            MLFLOW_EXPERIMENT_NAME
        )

        result = live_trace_probe()

        flush_available = flush_mlflow_traces()

        # Search this experiment, not an unrelated debug experiment.
        traces_df = mlflow.search_traces(
            experiment_ids=[experiment.experiment_id],
            max_results=20,
            order_by=["timestamp_ms DESC"],
        )

        trace_records = []

        if isinstance(traces_df, pd.DataFrame):
            for _, row in traces_df.iterrows():
                record = {}

                for column in (
                    "trace_id",
                    "client_request_id",
                    "state",
                    "status",
                    "request",
                    "response",
                    "execution_time_ms",
                    "timestamp_ms",
                    "trace_name",
                ):
                    if column in traces_df.columns:
                        value = row[column]

                        if value is not None and not (
                            isinstance(value, float)
                            and np.isnan(value)
                        ):
                            record[column] = (
                                value.item()
                                if isinstance(value, np.generic)
                                else value
                            )

                trace_records.append(record)

        return {
            "status": "success",
            "message": "Live trace probe executed.",
            "training_performed": False,
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "experiment_id": experiment.experiment_id,
            "trace_flush_available": flush_available,
            "trace_query_status": "query_succeeded",
            "recent_trace_count": len(trace_records),
            "newest_traces": trace_records,
            "probe_result": result,
            "note": (
                "Check newest_traces for the trace ID and request. "
                "A successful query alone does not prove that this "
                "specific trace was persisted."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logging.exception("Live MLflow trace diagnostic failed.")

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Could not verify live MLflow traces.",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "experiment_name": MLFLOW_EXPERIMENT_NAME,
                "experiment_id": MLFLOW_EXPERIMENT_ID,
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
            "contents": (
                os.listdir(model_dir)
                if os.path.exists(model_dir)
                else []
            ),
        }

        resolver = ModelResolver(model_dir=model_dir)
        result["model_exists"] = resolver.is_model_exists()

        if result["model_exists"]:
            result["model_path"] = resolver.get_best_model_path()

        return result

    except Exception as exc:
        logging.exception("Debug model endpoint failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Debug model failed: {exc}",
        ) from exc


# =========================================================
# TRAINING TRACE
# Training happens only when /train is called.
# =========================================================

def run_training_with_trace(train_pipeline):
    if not MLFLOW_ENABLED:
        logging.warning(
            "MLflow disabled; running training without MLflow tracing."
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

        span.set_outputs({"status": "completed"})
        return result


@app.get("/train")
def train_route():
    try:
        logging.info("APS TRAINING REQUESTED")

        if not os.getenv("MONGO_DB_URL"):
            raise RuntimeError(
                "MONGO_DB_URL environment variable is missing."
            )

        train_pipeline = TrainPipeline()
        run_training_with_trace(train_pipeline)

        resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)

        if not resolver.is_model_exists():
            raise RuntimeError(
                "Training finished but no saved model was found."
            )

        return {
            "status": "success",
            "message": "Training completed successfully.",
            "model_path": resolver.get_best_model_path(),
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
# Only metadata is recorded; CSV row contents are not logged.
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
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
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

        resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)

        if not resolver.is_model_exists():
            raise HTTPException(
                status_code=404,
                detail="Model not found on server.",
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
            df.drop(columns=["class"], inplace=True)

        feature_columns = [
            col
            for col in schema["numerical_columns"]
            if col not in drop_columns
        ]

        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[feature_columns]

        model = load_object(resolver.get_best_model_path())
        predictions = run_prediction_with_trace(model, df)

        df["prediction"] = predictions
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

        df.to_csv(output_path, index=False)

        response = FileResponse(
            path=output_path,
            filename="prediction_output.csv",
            media_type="text/csv",
            background=BackgroundTask(
                lambda: Path(output_path).unlink(missing_ok=True)
            ),
        )

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

        if output_path and not response_owns_output:
            try:
                Path(output_path).unlink(missing_ok=True)
            except OSError:
                logging.exception(
                    "Could not delete temporary prediction output."
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
