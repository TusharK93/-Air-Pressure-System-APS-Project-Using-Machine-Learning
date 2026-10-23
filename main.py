<<<<<<< HEAD

import os
import tempfile
from pathlib import Path

import mlflow
=======
import dataclasses
import json
import logging as _stdlib_logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

>>>>>>> baa53c2 (Initial commit)
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
<<<<<<< HEAD
from starlette.background import BackgroundTask
from starlette.responses import RedirectResponse

from sensor.constant.training_pipeline import (
    SAVED_MODEL_DIR,
    SCHEMA_FILE_PATH,
=======
from starlette.background import BackgroundTasks
from starlette.responses import RedirectResponse

from sensor.logger import logging  # configures stdout + file logging first
from sensor.utils import mlflow_utils as mlu
from sensor.constant.training_pipeline import (
    SAVED_MODEL_DIR,
    SCHEMA_FILE_PATH,
    TARGET_COLUMN,
>>>>>>> baa53c2 (Initial commit)
)
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import read_yaml_file, load_object
from sensor.constant.application import APP_PORT
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
<<<<<<< HEAD
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
=======

mlflow = mlu.mlflow  # may be None if the package failed to import


# =========================================================
# STARTUP: configure MLflow in the background
# A slow / unreachable DagsHub must never block Render's
# port binding, and a failed attempt is retried later by
# mlu.ensure_mlflow() instead of staying disabled forever.
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
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
>>>>>>> baa53c2 (Initial commit)
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD

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
=======
_train_lock = threading.Lock()


# =========================================================
# HELPERS
# =========================================================

def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)[:300]


def _log_prediction_runs_enabled() -> bool:
    return os.getenv("MLFLOW_LOG_PREDICTION_RUNS", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )


class _WarningCapture(_stdlib_logging.Handler):
    """Collects MLflow's own warnings so the debug endpoint can show them."""

    def __init__(self):
        super().__init__(level=_stdlib_logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(
            f"{record.levelname}: {record.getMessage()}"[:500]
        )


def _search_recent_traces(experiment_id, limit=10):
    """Return recent traces as JSON-safe dicts (works on MLflow 2.x and 3.x)."""
    try:
        frame = mlflow.search_traces(
            locations=[experiment_id], max_results=limit
        )
    except TypeError:
        frame = mlflow.search_traces(
            experiment_ids=[experiment_id], max_results=limit
        )

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []

    wanted = (
        "trace_id", "request_id", "state", "status", "request_time",
        "timestamp_ms", "execution_duration", "execution_time_ms",
        "request", "response",
    )

    rows = []
    for _, row in frame.iterrows():
        record = {}
        for column in wanted:
            if column in frame.columns:
                value = row[column]
                if isinstance(value, float) and np.isnan(value):
                    continue
                if isinstance(value, np.generic):
                    value = value.item()
                record[column] = _jsonable(value)
        rows.append(record)
    return rows


# =========================================================
# HOME / HEALTH
>>>>>>> baa53c2 (Initial commit)
# =========================================================

@app.get("/")
def index():
    return RedirectResponse(url="/docs")


<<<<<<< HEAD
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
=======
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "APS Prediction API",
        **mlu.status(),
    }


@app.get("/mlflow-status")
def mlflow_status():
    """Re-tries the MLflow connection and reports exactly what happened."""
    mlu.setup_mlflow(force=True)
    return {**mlu.status(), "experiments_url": mlu.experiments_url()}


# =========================================================
# DIAGNOSTIC 1: does the server accept TRACES?
# =========================================================

@app.get("/debug-mlflow-trace")
def debug_mlflow_trace():
    if not mlu.ensure_mlflow():
        raise HTTPException(
            status_code=503,
            detail={"message": "MLflow is not configured.", **mlu.status()},
        )

    capture = _WarningCapture()
    mlflow_logger = _stdlib_logging.getLogger("mlflow")
    mlflow_logger.addHandler(capture)

    try:
        with mlu.span(
            "APS_Render_Live_Trace_Test",
            inputs={"probe": True},
        ) as probe:
            probe.set_outputs(
                {"message": "APS Render live tracing test", "ts": time.time()}
            )

        mlu.flush_traces()
        trace_id = mlu.last_trace_id()

        persisted = False
        verify_error = None

        if trace_id:
            try:
                persisted = mlflow.get_trace(trace_id) is not None
            except Exception as exc:
                verify_error = f"{type(exc).__name__}: {exc}"[:500]
        else:
            verify_error = "MLflow did not report a trace id for the probe."

        search_error = None
        recent = []
        try:
            recent = _search_recent_traces(mlu.status()["experiment_id"])
        except Exception as exc:
            search_error = f"{type(exc).__name__}: {exc}"[:500]

        return {
            "status": "success" if persisted else "NOT_PERSISTED",
            "trace_id": trace_id,
            "trace_persisted_on_server": persisted,
            "verify_error": verify_error,
            "search_error": search_error,
            "recent_trace_count": len(recent),
            "recent_traces": recent,
            "mlflow_warnings": capture.messages,
            "how_to_read_this": (
                "trace_persisted_on_server=true means the server stored the "
                "trace. If false, read verify_error and mlflow_warnings: a "
                "404 / 'not found' / 'endpoint' message means the tracking "
                "server (DagsHub) does not accept traces; a 401/403 means "
                "the username/token is wrong. Runs and metrics can still "
                "work - test them with /debug-mlflow-run."
            ),
            **mlu.status(),
        }
    finally:
        mlflow_logger.removeHandler(capture)


# =========================================================
# DIAGNOSTIC 2: do RUNS / METRICS / ARTIFACTS work?
# =========================================================

@app.get("/debug-mlflow-run")
def debug_mlflow_run():
    if not mlu.ensure_mlflow():
        raise HTTPException(
            status_code=503,
            detail={"message": "MLflow is not configured.", **mlu.status()},
        )

    try:
        with mlu.pipeline_run(
            f"render_probe_{int(time.time())}",
            tags={"pipeline": "debug-probe"},
            reuse_active=False,
        ) as run:
            if run is None:
                raise RuntimeError("start_run failed; see service logs.")

            mlu.log_param("source", "render")
            mlu.log_metric("probe_value", 1.0)
            mlu.log_metric("probe_value", 2.0, step=1)
            mlu.log_text("render probe artifact", "probe/probe.txt")
            run_id = run.info.run_id

        return {
            "status": "success",
            "run_id": run_id,
            "experiments_url": mlu.experiments_url(),
            "note": (
                "Open the experiments URL: a run named render_probe_* with "
                "metric probe_value and an artifact probe/probe.txt should "
                "exist."
            ),
        }
    except Exception as exc:
        logging.exception("MLflow run probe failed")
        raise HTTPException(
            status_code=500,
            detail={"message": "Run probe failed.", "error": str(exc)},
>>>>>>> baa53c2 (Initial commit)
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
<<<<<<< HEAD
                os.listdir(model_dir)
                if os.path.exists(model_dir)
                else []
=======
                os.listdir(model_dir) if os.path.exists(model_dir) else []
>>>>>>> baa53c2 (Initial commit)
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
<<<<<<< HEAD
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
=======
# TRAIN
# One MLflow run + one trace per call; every stage logs its
# own metrics (see sensor/pipeline and sensor/component).
# =========================================================

@app.get("/train")
def train_route():
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A training run is already in progress.",
        )

>>>>>>> baa53c2 (Initial commit)
    try:
        logging.info("APS TRAINING REQUESTED")

        if not os.getenv("MONGO_DB_URL"):
<<<<<<< HEAD
            raise RuntimeError(
                "MONGO_DB_URL environment variable is missing."
            )

        train_pipeline = TrainPipeline()
        run_training_with_trace(train_pipeline)
=======
            raise RuntimeError("MONGO_DB_URL environment variable is missing.")

        train_pipeline = TrainPipeline()
        result = train_pipeline.run_pipeline()
>>>>>>> baa53c2 (Initial commit)

        resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)

        if not resolver.is_model_exists():
<<<<<<< HEAD
            raise RuntimeError(
                "Training finished but no saved model was found."
            )
=======
            raise RuntimeError("Training finished but no saved model was found.")

        def _as_dict(item):
            return dataclasses.asdict(item) if item is not None else None
>>>>>>> baa53c2 (Initial commit)

        return {
            "status": "success",
            "message": "Training completed successfully.",
            "model_path": resolver.get_best_model_path(),
<<<<<<< HEAD
            "mlflow_enabled": MLFLOW_ENABLED,
            "experiment": MLFLOW_EXPERIMENT_NAME,
        }

=======
            "is_model_accepted": result.is_model_accepted,
            "improved_f1": result.improved_accuracy,
            "candidate_metrics": _as_dict(result.train_model_metric_artifact),
            "deployed_metrics": _as_dict(result.best_model_metric_artifact),
            "mlflow_enabled": mlu.is_enabled(),
            "mlflow_run_id": train_pipeline.mlflow_run_id,
            "mlflow_experiment": mlu.experiment_name(),
            "experiments_url": mlu.experiments_url(),
        }

    except HTTPException:
        raise

>>>>>>> baa53c2 (Initial commit)
    except Exception as exc:
        logging.exception("TRAINING FAILED")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {exc}",
        ) from exc

<<<<<<< HEAD

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
=======
    finally:
        _train_lock.release()


# =========================================================
# PREDICTION METRICS (logged after the response is sent)
# Only aggregate numbers are stored - never CSV row contents.
# =========================================================

def _log_prediction_run(summary: dict, labels=None, predictions=None):
    if not _log_prediction_runs_enabled():
        return

    try:
        with mlu.pipeline_run(
            f"predict_{int(time.time())}",
            tags={
                "pipeline": "aps-inference",
                "model_version": summary.get("model_version"),
            },
            reuse_active=False,
        ) as run:
            if run is None:
                return

            mlu.log_params({
                "model_version": summary.get("model_version"),
                "upload_filename": summary.get("filename"),
            })
            mlu.log_metrics({
                "rows": summary["rows"],
                "feature_count": summary["feature_count"],
                "missing_features_added": summary["missing_features_added"],
                "input_missing_ratio": summary["input_missing_ratio"],
                "predicted_positive": summary["predicted_positive"],
                "predicted_negative": summary["predicted_negative"],
                "predicted_positive_rate": summary["predicted_positive_rate"],
                "latency_read_sec": summary["latency_read_sec"],
                "latency_predict_sec": summary["latency_predict_sec"],
                "latency_total_sec": summary["latency_total_sec"],
            })

            # If the uploaded CSV also contains the true "class" column,
            # we can log live accuracy metrics for this batch.
            if labels is not None and predictions is not None:
                from sklearn.metrics import (
                    confusion_matrix, f1_score, precision_score, recall_score,
                )

                tn, fp, fn, tp = confusion_matrix(
                    labels, predictions, labels=[0, 1]
                ).ravel()
                mlu.log_metrics({
                    "live_f1_score": f1_score(labels, predictions, zero_division=0),
                    "live_precision": precision_score(labels, predictions, zero_division=0),
                    "live_recall": recall_score(labels, predictions, zero_division=0),
                    "live_tn": int(tn),
                    "live_fp": int(fp),
                    "live_fn": int(fn),
                    "live_tp": int(tp),
                    "live_aps_cost": int(10 * fp + 500 * fn),
                })
    except Exception:
        logging.exception("Could not log prediction run to MLflow.")


def _delete_file(path: str):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logging.exception("Could not delete temporary prediction output.")


# =========================================================
# PREDICT
# Sync handler -> FastAPI runs it in a worker thread, so the
# (blocking) MLflow network calls never stall the event loop.
# =========================================================

@app.post("/predict")
def predict_route(file: UploadFile = File(...)):
    output_path = None
    response_owns_output = False
    started = time.time()

    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a CSV file.")
>>>>>>> baa53c2 (Initial commit)

        resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)

        if not resolver.is_model_exists():
<<<<<<< HEAD
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

=======
            raise HTTPException(status_code=404, detail="Model not found on server.")

        with mlu.span(
            "APS_Prediction_Request",
            inputs={"filename": file.filename},
        ) as root:

            # ---- 1. read + prepare the CSV ---------------------------
            with mlu.span("read_and_prepare_csv") as step:
                t0 = time.time()

                df = pd.read_csv(file.file)
                df.replace("na", np.nan, inplace=True)

                schema = read_yaml_file(SCHEMA_FILE_PATH)
                drop_columns = schema.get("drop_columns", [])

                df.drop(columns=drop_columns, inplace=True, errors="ignore")

                labels = None
                if TARGET_COLUMN in df.columns:
                    try:
                        labels = (
                            df[TARGET_COLUMN]
                            .replace(TargetValueMapping().to_dict())
                            .astype(int)
                            .to_numpy()
                        )
                    except Exception:
                        labels = None
                    df.drop(columns=[TARGET_COLUMN], inplace=True)

                feature_columns = [
                    col for col in schema["numerical_columns"]
                    if col not in drop_columns
                ]

                added = 0
                for col in feature_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                        added += 1

                df = df[feature_columns]
                read_sec = time.time() - t0

                step.set_outputs({
                    "rows": int(len(df)),
                    "feature_count": int(len(df.columns)),
                    "missing_features_added": added,
                })

            # ---- 2. load model ----------------------------------------
            with mlu.span("load_model"):
                model_path = resolver.get_best_model_path()
                model = load_object(model_path)
                model_version = os.path.basename(os.path.dirname(model_path))

            # ---- 3. predict -------------------------------------------
            with mlu.span(
                "model_predict",
                inputs={
                    "row_count": int(len(df)),
                    "feature_count": int(len(df.columns)),
                    "model_type": type(model).__name__,
                    "model_version": model_version,
                },
            ) as step:
                t1 = time.time()
                predictions = np.asarray(model.predict(df))
                predict_sec = time.time() - t1

                positives = int((predictions == 1).sum())
                step.set_outputs({
                    "prediction_count": int(len(predictions)),
                    "predicted_positive": positives,
                })

            total_sec = time.time() - started
            rows = int(len(df))

            summary = {
                "filename": file.filename,
                "model_version": model_version,
                "rows": rows,
                "feature_count": int(len(df.columns)),
                "missing_features_added": added,
                "input_missing_ratio": float(df.isna().mean().mean()) if rows else 0.0,
                "predicted_positive": positives,
                "predicted_negative": rows - positives,
                "predicted_positive_rate": (positives / rows) if rows else 0.0,
                "latency_read_sec": round(read_sec, 4),
                "latency_predict_sec": round(predict_sec, 4),
                "latency_total_sec": round(total_sec, 4),
            }

            root.set_outputs({
                k: summary[k]
                for k in (
                    "rows", "predicted_positive", "predicted_positive_rate",
                    "latency_total_sec", "model_version",
                )
            })

        # ---- 4. build the response file --------------------------------
>>>>>>> baa53c2 (Initial commit)
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

<<<<<<< HEAD
=======
        tasks = BackgroundTasks()
        tasks.add_task(_delete_file, output_path)
        tasks.add_task(_log_prediction_run, summary, labels, predictions)

>>>>>>> baa53c2 (Initial commit)
        response = FileResponse(
            path=output_path,
            filename="prediction_output.csv",
            media_type="text/csv",
<<<<<<< HEAD
            background=BackgroundTask(
                lambda: Path(output_path).unlink(missing_ok=True)
            ),
=======
            background=tasks,
>>>>>>> baa53c2 (Initial commit)
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
<<<<<<< HEAD
        await file.close()

        if output_path and not response_owns_output:
            try:
                Path(output_path).unlink(missing_ok=True)
            except OSError:
                logging.exception(
                    "Could not delete temporary prediction output."
                ) 
=======
        try:
            file.file.close()
        except Exception:
            pass

        if output_path and not response_owns_output:
            _delete_file(output_path)
>>>>>>> baa53c2 (Initial commit)


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
