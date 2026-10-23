"""
Central, fail-safe MLflow helpers for the APS project.

Design rules
------------
* Every helper is a harmless no-op when MLflow is disabled, mis-configured or
  unreachable, so tracking problems can NEVER break /train or /predict.
* Failures are never silent: they are printed to stdout (visible in Render's
  log panel) and stored in status() so /health and /mlflow-status can show them.
* setup_mlflow() can be called again later (with a cool-down), so one network
  blip at boot does not disable tracking for the whole life of the process.

Environment variables (set them in the Render dashboard)
---------------------------------------------------------
MLFLOW_TRACKING_URI       e.g. https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME  your DagsHub username
MLFLOW_TRACKING_PASSWORD  a DagsHub ACCESS TOKEN (not your account password)
MLFLOW_EXPERIMENT_NAME    optional, default "aps-fault-detection"
"""

import contextlib
import logging
import math
import os
import threading
import time

# ---------------------------------------------------------------------------
# These must be set BEFORE mlflow is imported.
#   * synchronous trace export  -> errors show up immediately, no lost queue
#   * few retries / short timeout -> a dead server cannot hang a request
# Anything you set yourself in Render's dashboard wins (setdefault).
# ---------------------------------------------------------------------------
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")

try:
    import mlflow
except Exception as _import_exc:  # pragma: no cover
    mlflow = None
    _IMPORT_ERROR = f"{type(_import_exc).__name__}: {_import_exc}"
else:
    _IMPORT_ERROR = None

log = logging.getLogger("aps.mlflow")

DEFAULT_TRACKING_URI = "https://dagshub.com/tukumawat73/aps-fault-detection.mlflow"
DEFAULT_EXPERIMENT = "aps-fault-detection"

_lock = threading.RLock()
_state = {
    "enabled": False,
    "error": None,
    "experiment_id": None,
    "last_attempt": 0.0,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def tracking_uri() -> str:
    return (os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI).strip()


def experiment_name() -> str:
    return (os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT).strip()


def is_enabled() -> bool:
    return bool(_state["enabled"])


def setup_mlflow(force: bool = False, cooldown: float = 30.0) -> bool:
    """Configure MLflow. Returns True when tracking is usable.

    Safe to call repeatedly: when already enabled it returns immediately, and
    after a failure it will not retry more often than once per `cooldown` s.
    """
    with _lock:
        if _state["enabled"] and not force:
            return True

        now = time.time()
        if (
            not force
            and _state["last_attempt"]
            and now - _state["last_attempt"] < cooldown
        ):
            return False
        _state["last_attempt"] = now

        try:
            if mlflow is None:
                raise RuntimeError(f"mlflow could not be imported: {_IMPORT_ERROR}")

            uri = tracking_uri()
            if not uri:
                raise ValueError("MLFLOW_TRACKING_URI is empty.")

            if uri.startswith("http"):
                if not (os.getenv("MLFLOW_TRACKING_USERNAME") or "").strip():
                    raise ValueError("MLFLOW_TRACKING_USERNAME is missing.")
                if not (os.getenv("MLFLOW_TRACKING_PASSWORD") or "").strip():
                    raise ValueError("MLFLOW_TRACKING_PASSWORD is missing.")

            mlflow.set_tracking_uri(uri)
            experiment = mlflow.set_experiment(experiment_name())

            _state["experiment_id"] = experiment.experiment_id
            _state["enabled"] = True
            _state["error"] = None
            print(
                f"[mlflow] ENABLED uri={uri} experiment={experiment_name()} "
                f"id={experiment.experiment_id} version={mlflow.__version__}",
                flush=True,
            )
            return True

        except Exception as exc:
            _state["enabled"] = False
            _state["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[mlflow] DISABLED -> {_state['error']}", flush=True)
            log.exception("MLflow configuration failed")
            return False


def ensure_mlflow() -> bool:
    """Cheap check used before every MLflow call; retries setup if needed."""
    if _state["enabled"]:
        return True
    return setup_mlflow()


def status() -> dict:
    """Safe-to-expose status (never includes secrets)."""
    return {
        "mlflow_enabled": bool(_state["enabled"]),
        "mlflow_error": _state["error"],
        "mlflow_version": getattr(mlflow, "__version__", None),
        "tracking_uri": tracking_uri(),
        "experiment_name": experiment_name(),
        "experiment_id": _state["experiment_id"],
        "username_set": bool((os.getenv("MLFLOW_TRACKING_USERNAME") or "").strip()),
        "password_set": bool((os.getenv("MLFLOW_TRACKING_PASSWORD") or "").strip()),
        "async_trace_logging": os.getenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"),
    }


def experiments_url() -> str:
    """Link to the DagsHub experiments page (or the raw tracking URI)."""
    uri = tracking_uri()
    if uri.endswith(".mlflow"):
        return uri[: -len(".mlflow")] + "/experiments"
    return uri


# ---------------------------------------------------------------------------
# Internal safe-call helper
# ---------------------------------------------------------------------------
def _safe(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        print(f"[mlflow] {label} failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def has_active_run() -> bool:
    if mlflow is None or not _state["enabled"]:
        return False
    try:
        return mlflow.active_run() is not None
    except Exception:
        return False


def active_run_id():
    if not has_active_run():
        return None
    return mlflow.active_run().info.run_id


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def pipeline_run(run_name: str, tags: dict = None, reuse_active: bool = True):
    """Start an MLflow run (or reuse the active one). Yields the run or None.

    Marks the run FAILED (and stores the error) if the body raises.
    """
    if not ensure_mlflow():
        yield None
        return

    if reuse_active and has_active_run():
        yield mlflow.active_run()
        return

    try:
        run = mlflow.start_run(run_name=run_name, tags=tags or {})
    except Exception as exc:
        print(f"[mlflow] start_run failed: {type(exc).__name__}: {exc}", flush=True)
        yield None
        return

    final_status = "FINISHED"
    try:
        yield run
    except BaseException as exc:
        final_status = "FAILED"
        _safe(
            "set_tag(error)",
            mlflow.set_tag,
            "error",
            f"{type(exc).__name__}: {exc}"[:5000],
        )
        raise
    finally:
        _safe("end_run", mlflow.end_run, final_status)


# ---------------------------------------------------------------------------
# Logging wrappers (all no-ops without an active run)
# ---------------------------------------------------------------------------
def _clean_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def log_param(key: str, value) -> None:
    if not has_active_run() or value is None:
        return
    _safe(f"log_param({key})", mlflow.log_param, key, str(value)[:500])


def log_params(params: dict) -> None:
    for key, value in (params or {}).items():
        log_param(key, value)


def log_metric(key: str, value, step: int = None) -> None:
    number = _clean_number(value)
    if number is None or not has_active_run():
        return
    if step is None:
        _safe(f"log_metric({key})", mlflow.log_metric, key, number)
    else:
        _safe(f"log_metric({key})", mlflow.log_metric, key, number, step=step)


def log_metrics(metrics: dict, step: int = None) -> None:
    for key, value in (metrics or {}).items():
        log_metric(key, value, step=step)


def set_tags(tags: dict) -> None:
    if not has_active_run():
        return
    clean = {k: str(v)[:5000] for k, v in (tags or {}).items() if v is not None}
    if clean:
        _safe("set_tags", mlflow.set_tags, clean)


def log_artifact(path: str, artifact_path: str = None) -> None:
    if not has_active_run() or not path or not os.path.isfile(path):
        return
    _safe(f"log_artifact({os.path.basename(path)})", mlflow.log_artifact, path, artifact_path)


def log_dict(data: dict, artifact_file: str) -> None:
    if not has_active_run():
        return
    _safe(f"log_dict({artifact_file})", mlflow.log_dict, data, artifact_file)


def log_text(text: str, artifact_file: str) -> None:
    if not has_active_run():
        return
    _safe(f"log_text({artifact_file})", mlflow.log_text, text, artifact_file)


def log_classification_metrics(prefix: str, metric_artifact, step: int = None) -> None:
    """Log f1 / precision / recall from a ClassificationMetricArtifact."""
    if metric_artifact is None:
        return
    log_metrics(
        {
            f"{prefix}_f1_score": getattr(metric_artifact, "f1_score", None),
            f"{prefix}_precision": getattr(metric_artifact, "precision_score", None),
            f"{prefix}_recall": getattr(metric_artifact, "recall_score", None),
        },
        step=step,
    )


# ---------------------------------------------------------------------------
# Tracing (spans)
# ---------------------------------------------------------------------------
class _NullSpan:
    """Stand-in span used when tracing is unavailable."""

    def set_inputs(self, *a, **k):
        pass

    def set_outputs(self, *a, **k):
        pass

    def set_attribute(self, *a, **k):
        pass

    def set_attributes(self, *a, **k):
        pass


@contextlib.contextmanager
def span(name: str, inputs: dict = None, span_type: str = None):
    """Create a trace span; degrades to a no-op if MLflow is unavailable."""
    if not ensure_mlflow():
        yield _NullSpan()
        return

    cm = None
    try:
        if span_type:
            cm = mlflow.start_span(name=name, span_type=span_type)
        else:
            cm = mlflow.start_span(name=name)
        sp = cm.__enter__()
    except Exception as exc:
        print(f"[mlflow] start_span({name}) failed: {type(exc).__name__}: {exc}", flush=True)
        yield _NullSpan()
        return

    if inputs:
        _safe("span.set_inputs", sp.set_inputs, inputs)

    try:
        yield sp
    except BaseException as exc:
        try:
            cm.__exit__(type(exc), exc, exc.__traceback__)
        except BaseException:
            pass
        raise
    else:
        _safe(f"span.exit({name})", cm.__exit__, None, None, None)


@contextlib.contextmanager
def stage(name: str):
    """Span + wall-clock duration metric for one pipeline stage."""
    started = time.time()
    ok = False
    try:
        with span(f"stage_{name}", inputs={"stage": name}) as sp:
            yield sp
        ok = True
    finally:
        log_metric(f"duration_{name}_sec", round(time.time() - started, 3))
        if not ok:
            set_tags({"failed_stage": name})


def flush_traces() -> bool:
    """Flush queued trace exports (only matters if async logging is on)."""
    func = getattr(mlflow, "flush_trace_async_logging", None)
    if callable(func):
        _safe("flush_trace_async_logging", func)
        return True
    return False


def last_trace_id():
    """ID of the most recent trace created in this thread (None if unknown)."""
    if mlflow is None:
        return None
    getter = getattr(mlflow, "get_last_active_trace_id", None)
    if callable(getter):
        return _safe("get_last_active_trace_id", getter)
    getter = getattr(mlflow, "get_last_active_trace", None)
    if callable(getter):
        trace = _safe("get_last_active_trace", getter)
        info = getattr(trace, "info", None)
        return getattr(info, "trace_id", None) or getattr(info, "request_id", None)
    return None
