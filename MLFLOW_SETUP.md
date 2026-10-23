# APS – MLflow tracking setup

## Render environment variables
| Name | Value |
|---|---|
| MLFLOW_TRACKING_URI | https://dagshub.com/tukumawat73/aps-fault-detection.mlflow |
| MLFLOW_TRACKING_USERNAME | your DagsHub username |
| MLFLOW_TRACKING_PASSWORD | a DagsHub **access token** (not your account password) |
| MLFLOW_EXPERIMENT_NAME | aps-fault-detection (optional) |
| MONGO_DB_URL | your Mongo connection string |
| MLFLOW_LOG_PREDICTION_RUNS | true / false (optional, default true) |
| MLFLOW_ENABLE_ASYNC_TRACE_LOGGING | optional, default false (sync). Set true only after traces are confirmed working |

## Check order after deploying
1. `/health`            -> mlflow_enabled must be true (else read mlflow_error)
2. `/debug-mlflow-run`  -> a run "render_probe_*" must appear in DagsHub (tests creds, metrics, artifacts)
3. `/debug-mlflow-trace`-> trace_persisted_on_server must be true (tests TRACE support)
   If 2 works and 3 says NOT_PERSISTED, DagsHub is not accepting traces: runs/metrics
   still work, only the Traces tab is affected.
4. `/train`             -> one run "train_<timestamp>" with all metrics
5. `POST /predict`      -> trace "APS_Prediction_Request" + a run "predict_<ts>"

## What gets logged
Training run (`train_*`): per-stage durations, rows/columns/class balance/missing ratio,
drift counts + drift_report.yaml, transformed shapes, XGBoost params, train/test
f1/precision/recall/accuracy/ROC-AUC, TN/FP/FN/TP, APS cost (10*FP + 500*FN),
train-test F1 gap, candidate-vs-deployed comparison, accept/reject decision,
model.pkl + evaluation report as artifacts.
Prediction runs (`predict_*`): rows, positive rate, missing ratio, latencies and,
if the uploaded CSV contains the `class` column, live f1/precision/recall/cost.

## Files in this bundle
main.py, requirements.txt, sensor/logger.py, sensor/utils/mlflow_utils.py (new),
sensor/pipeline/training_pipeline.py, sensor/component/{data_ingestion,
data_validation,data_transformation,data_trainer,model_evaluation,model_pusher}.py
