
import dataclasses
import json
import logging as std_logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from starlette.background import BackgroundTasks

from sensor.constant.application import APP_PORT
from sensor.constant.training_pipeline import (
    SAVED_MODEL_DIR,
    SCHEMA_FILE_PATH,
    TARGET_COLUMN,
)
from sensor.logger import logging
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils import mlflow_utils as mlu
from sensor.utils.main_utils import load_object, read_yaml_file


# =========================================================
# APPLICATION AND MLFLOW STARTUP
# =========================================================

_train_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Do not block Render's startup while connecting to DagsHub.
    threading.Thread(
        target=mlu.setup_mlflow,
        kwargs={"force": True},
        name="mlflow-setup",
        daemon=True,
    ).start()
    yield


app = FastAPI(
    title="APS Prediction API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HELPERS
# =========================================================

def _jsonable(value):
    """Convert values to JSON-safe values for API responses."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)[:500]


def _as_dict(value):
    if value is None:
        return None

    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)

    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return _jsonable(value)


def _log_prediction_runs_enabled():
    return os.getenv(
        "MLFLOW_LOG_PREDICTION_RUNS", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _delete_file(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logging.exception("Could not delete temporary prediction output.")


def _search_recent_traces(experiment_id, limit=10):
    """Search recent MLflow traces when supported by the installed version."""
    if mlflow is None:
        return []

    try:
        frame = mlflow.search_traces(
            experiment_ids=[experiment_id],
            max_results=limit,
        )
    except TypeError:
        frame = mlflow.search_traces(
            locations=[experiment_id],
            max_results=limit,
        )

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []

    wanted = (
        "trace_id",
        "request_id",
        "state",
        "status",
        "request_time",
        "timestamp_ms",
        "execution_duration",
        "execution_time_ms",
        "request",
        "response",
    )

    records = []
    for _, row in frame.iterrows():
        record = {}
        for column in wanted:
            if column not in frame.columns:
                continue

            value = row[column]
            if value is None:
                continue

            if isinstance(value, (float, np.floating)) and np.isnan(value):
                continue

            if isinstance(value, np.generic):
                value = value.item()

            record[column] = _jsonable(value)

        records.append(record)

    return records


# =========================================================
# BASIC ENDPOINTS
# =========================================================

@app.get("/")
def index():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Basic health check; it should work even if MLflow is unavailable."""
    return {
        "status": "ok",
        "service": "APS Prediction API",
        **mlu.status(),
    }


@app.get("/mlflow-status")
def mlflow_status():
    """Retry MLflow setup and return its current status."""
    mlu.setup_mlflow(force=True)
    return {
        **mlu.status(),
        "experiments_url": mlu.experiments_url(),
    }


@app.get("/debug-model")
def debug_model():
    """Check whether a trained model is available."""
    try:
        model_dir = SAVED_MODEL_DIR
        resolver = ModelResolver(model_dir=model_dir)

        result = {
            "saved_model_dir": model_dir,
            "absolute_path": os.path.abspath(model_dir),
            "exists": os.path.exists(model_dir),
            "contents": (
                os.listdir(model_dir)
                if os.path.exists(model_dir)
                else []
            ),
            "model_exists": resolver.is_model_exists(),
        }

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
# MLFLOW TRACE DIAGNOSTIC
# =========================================================

@app.get("/debug-mlflow-trace")
def debug_mlflow_trace():
    """Create a small trace without training or reading a dataset."""
    if not mlu.ensure_mlflow():
        raise HTTPException(
            status_code=503,
            detail={
                "message": "MLflow is not configured.",
                **mlu.status(),
            },
        )

    try:
        with mlu.span(
            "APS_Render_Live_Trace_Test",
            inputs={"probe": True},
        ) as span:
            span.set_outputs({
                "message": "APS Render live tracing test",
                "training_performed": False,
            })

        mlu.flush_traces()
        trace_id = mlu.last_trace_id()

        persisted = False
        verify_error = None

        if trace_id and mlflow is not None:
            try:
                persisted = mlflow.get_trace(trace_id) is not None
            except Exception as exc:
                verify_error = f"{type(exc).__name__}: {exc}"[:500]
        else:
            verify_error = "No trace ID was returned."

        try:
            status = mlu.status()
            experiment_id = status.get("experiment_id")
            recent = (
                _search_recent_traces(experiment_id)
                if experiment_id
                else []
            )
            search_error = None
        except Exception as exc:
            recent = []
            search_error = f"{type(exc).__name__}: {exc}"[:500]

        return {
            "status": "success" if persisted else "NOT_VERIFIED",
            "trace_id": trace_id,
            "trace_persisted_on_server": persisted,
            "verify_error": verify_error,
            "search_error": search_error,
            "recent_trace_count": len(recent),
            "recent_traces": recent,
            **mlu.status(),
        }

    except Exception as exc:
        logging.exception("MLflow trace diagnostic failed.")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "MLflow trace diagnostic failed.",
                "error_type": type(exc).__name__,
                "error": str(exc),
                **mlu.status(),
            },
        ) from exc


# =========================================================
# MLFLOW RUN / METRIC DIAGNOSTIC
# =========================================================

@app.get("/debug-mlflow-run")
def debug_mlflow_run():
    """Test whether MLflow runs, metrics and artifacts can be logged."""
    if not mlu.ensure_mlflow():
        raise HTTPException(
            status_code=503,
            detail={
                "message": "MLflow is not configured.",
                **mlu.status(),
            },
        )

    try:
        run_name = f"render_probe_{int(time.time())}"

        with mlu.pipeline_run(
            run_name,
            tags={"pipeline": "debug-probe"},
            reuse_active=False,
        ) as run:
            if run is None:
                raise RuntimeError(
                    "MLflow run could not be started; check Render logs."
                )

            mlu.log_param("source", "render")
            mlu.log_metric("probe_value", 1.0)
            mlu.log_metric("probe_value", 2.0, step=1)
            mlu.log_text("Render MLflow probe", "probe/probe.txt")
            run_id = run.info.run_id

        return {
            "status": "success",
            "run_id": run_id,
            "experiments_url": mlu.experiments_url(),
        }

    except Exception as exc:
        logging.exception("MLflow run diagnostic failed.")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "MLflow run diagnostic failed.",
                "error": str(exc),
                **mlu.status(),
            },
        ) from exc


# =========================================================
# TRAINING
# =========================================================

@app.get("/train")
def train_route():
    # Prevent two simultaneous training jobs on the same instance.
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A training run is already in progress.",
        )

    try:
        logging.info("APS TRAINING REQUESTED")

        if not os.getenv("MONGO_DB_URL"):
            raise RuntimeError(
                "MONGO_DB_URL environment variable is missing."
            )

        train_pipeline = TrainPipeline()
        result = train_pipeline.run_pipeline()

        resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not resolver.is_model_exists():
            raise RuntimeError(
                "Training finished but no saved model was found."
            )

        return {
            "status": "success",
            "message": "Training completed successfully.",
            "model_path": resolver.get_best_model_path(),
            "is_model_accepted": getattr(
                result, "is_model_accepted", None
            ),
            "improved_f1": getattr(
                result, "improved_accuracy", None
            ),
            "candidate_metrics": _as_dict(
                getattr(result, "train_model_metric_artifact", None)
            ),
            "deployed_metrics": _as_dict(
                getattr(result, "best_model_metric_artifact", None)
            ),
            "mlflow_enabled": mlu.is_enabled(),
            "mlflow_run_id": getattr(
                train_pipeline, "mlflow_run_id", None
            ),
            "mlflow_experiment": mlu.experiment_name(),
            "experiments_url": mlu.experiments_url(),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logging.exception("APS TRAINING FAILED")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {exc}",
        ) from exc

    finally:
        _train_lock.release()


# =========================================================
# PREDICTION METRICS
# Log aggregate data only; do not log uploaded CSV contents.
# =========================================================

def _log_prediction_run(summary, labels=None, predictions=None):
    if not _log_prediction_runs_enabled():
        return

    if not mlu.ensure_mlflow():
        logging.warning("Skipping prediction metrics: MLflow unavailable.")
        return

    try:
        with mlu.pipeline_run(
            f"predict_{int(time.time())}",
            tags={
                "pipeline": "aps-inference",
                "model_version": str(summary.get("model_version", "unknown")),
            },
            reuse_active=False,
        ) as run:
            if run is None:
                return

            # Do not log the uploaded filename: it may contain personal data.
            mlu.log_params({
                "model_version": str(summary.get("model_version", "unknown")),
                "source": "prediction_api",
            })

            metric_keys = (
                "rows",
                "feature_count",
                "missing_features_added",
                "input_missing_ratio",
                "predicted_positive",
                "predicted_negative",
                "predicted_positive_rate",
                "latency_read_sec",
                "latency_predict_sec",
                "latency_total_sec",
            )

            mlu.log_metrics({
                key: float(summary[key])
                for key in metric_keys
                if summary.get(key) is not None
            })

            # Only calculate labelled-batch metrics if ground-truth labels
            # were present and could be parsed successfully.
            if labels is not None and predictions is not None:
                from sklearn.metrics import (
                    confusion_matrix,
                    f1_score,
                    precision_score,
                    recall_score,
                )

                if len(labels) == len(predictions) and len(labels) > 0:
                    tn, fp, fn, tp = confusion_matrix(
                        labels,
                        predictions,
                        labels=[0, 1],
                    ).ravel()

                    mlu.log_metrics({
                        "live_f1_score": float(
                            f1_score(labels, predictions, zero_division=0)
                        ),
                        "live_precision": float(
                            precision_score(
                                labels, predictions, zero_division=0
                            )
                        ),
                        "live_recall": float(
                            recall_score(
                                labels, predictions, zero_division=0
                            )
                        ),
                        "live_tn": int(tn),
                        "live_fp": int(fp),
                        "live_fn": int(fn),
                        "live_tp": int(tp),
                        "live_aps_cost": int(10 * fp + 500 * fn),
                    })

    except Exception:
        # Prediction should still succeed if tracking is unavailable.
        logging.exception("Could not log prediction metrics to MLflow.")


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict_route(file: UploadFile = File(...)):
    output_path = None
    response_owns_output = False
    started = time.time()

    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a CSV file.",
            )

        resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not resolver.is_model_exists():
            raise HTTPException(
                status_code=404,
                detail="Model not found on server. Train or deploy a model first.",
            )

        # ---- Read and prepare CSV ------------------------------------
        read_started = time.time()
        df = pd.read_csv(file.file)
        df.replace("na", np.nan, inplace=True)

        schema = read_yaml_file(SCHEMA_FILE_PATH)
        drop_columns = schema.get("drop_columns", [])

        df.drop(
            columns=drop_columns,
            inplace=True,
            errors="ignore",
        )

        labels = None
        if TARGET_COLUMN in df.columns:
            try:
                mapped_labels = df[TARGET_COLUMN].replace(
                    TargetValueMapping().to_dict()
                )
                labels = pd.to_numeric(
                    mapped_labels, errors="raise"
                ).astype(int).to_numpy()
            except (TypeError, ValueError):
                labels = None

            df.drop(columns=[TARGET_COLUMN], inplace=True)

        feature_columns = [
            column
            for column in schema["numerical_columns"]
            if column not in drop_columns
        ]

        missing_features_added = 0
        for column in feature_columns:
            if column not in df.columns:
                df[column] = np.nan
                missing_features_added += 1

        df = df[feature_columns]
        read_sec = time.time() - read_started

        # ---- Load model ----------------------------------------------
        model_path = resolver.get_best_model_path()
        model = load_object(model_path)
        model_version = os.path.basename(os.path.dirname(model_path))

        # ---- Predict -------------------------------------------------
        predict_started = time.time()
        predictions = np.asarray(model.predict(df))
        predict_sec = time.time() - predict_started

        if len(predictions) != len(df):
            raise RuntimeError(
                "The model returned a different number of predictions "
                "than the number of input rows."
            )

        positives = int(np.sum(predictions == 1))
        rows = int(len(df))
        total_sec = time.time() - started

        summary = {
            "model_version": model_version,
            "rows": rows,
            "feature_count": int(len(df.columns)),
            "missing_features_added": missing_features_added,
            "input_missing_ratio": (
                float(df.isna().mean().mean())
                if rows and len(df.columns)
                else 0.0
            ),
            "predicted_positive": positives,
            "predicted_negative": rows - positives,
            "predicted_positive_rate": positives / rows if rows else 0.0,
            "latency_read_sec": round(read_sec, 4),
            "latency_predict_sec": round(predict_sec, 4),
            "latency_total_sec": round(total_sec, 4),
        }

        # Use readable target labels in the returned CSV.
        output_df = df.copy()
        output_df["prediction"] = predictions
        output_df["prediction"] = output_df["prediction"].replace(
            TargetValueMapping().reverse_mapping()
        )

        # ---- Write response file ------------------------------------
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="aps_prediction_",
            delete=False,
            newline="",
            encoding="utf-8",
        ) as output_file:
            output_path = output_file.name

        output_df.to_csv(output_path, index=False)

        # Schedule cleanup and metric logging after the response.
        tasks = BackgroundTasks()
        tasks.add_task(_delete_file, output_path)
        tasks.add_task(_log_prediction_run, summary, labels, predictions)

        response = FileResponse(
            path=output_path,
            filename="prediction_output.csv",
            media_type="text/csv",
            background=tasks,
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
        try:
            file.file.close()
        except Exception:
            pass

        if output_path and not response_owns_output:
            _delete_file(output_path)


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
