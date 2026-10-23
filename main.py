```python
import os
import traceback

import mlflow
import pandas as pd
import numpy as np

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.responses import (
    Response,
    FileResponse
)

from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from sensor.constant.training_pipeline import (
    SAVED_MODEL_DIR,
    SCHEMA_FILE_PATH
)

from sensor.pipeline.training_pipeline import TrainPipeline

from sensor.utils.main_utils import (
    read_yaml_file,
    load_object
)

from sensor.constant.application import APP_PORT

from sensor.ml.model.estimator import (
    ModelResolver,
    TargetValueMapping
)

from sensor.logger import logging


# =========================================================
# MLFLOW CONFIGURATION
# =========================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "APS-Failure-Prediction"
)

MLFLOW_ENABLED = False

if MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        MLFLOW_ENABLED = True

        logging.info("MLflow tracking configured successfully")
        logging.info(
            "MLflow experiment: %s",
            MLFLOW_EXPERIMENT_NAME
        )

    except Exception:
        logging.exception(
            "MLflow configuration failed; tracing disabled"
        )
else:
    logging.warning(
        "MLFLOW_TRACKING_URI is missing; tracing disabled"
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="APS Prediction API",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MLFLOW TRACE HELPERS
# =========================================================

def run_training_with_trace(train_pipeline):
    """
    Trace the complete APS training pipeline.
    """

    if not MLFLOW_ENABLED:
        return train_pipeline.run_pipeline()

    with mlflow.start_span(
        name="APS_Training_Pipeline"
    ) as span:

        span.set_inputs({
            "pipeline": "TrainPipeline",
            "project": "APS Failure Prediction"
        })

        result = train_pipeline.run_pipeline()

        span.set_outputs({
            "status": "completed"
        })

        return result


def run_prediction_with_trace(model, dataframe):
    """
    Trace model prediction without logging uploaded CSV values.
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
                str(column)
                for column in dataframe.columns
            ],
            "model_type": type(model).__name__
        })

        predictions = model.predict(dataframe)

        span.set_outputs({
            "prediction_count": int(len(predictions)),
            "status": "completed"
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
        "mlflow_enabled": MLFLOW_ENABLED
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
            "contents": []
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

    except Exception as e:

        traceback.print_exc()

        return {
            "error": str(e),
            "saved_model_dir": SAVED_MODEL_DIR,
            "absolute_path": os.path.abspath(SAVED_MODEL_DIR)
        }


# =========================================================
# TRAIN
# =========================================================

@app.get("/train")
def train_route():

    try:
        print("=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)

        mongo_url = os.getenv("MONGO_DB_URL")

        if not mongo_url:
            raise RuntimeError(
                "MONGO_DB_URL environment variable "
                "is not set on Render"
            )

        print("MongoDB environment variable found")

        train_pipeline = TrainPipeline()

        print("TrainPipeline initialized")

        # ---------------------------------------------
        # MLFLOW TRAINING TRACE
        # ---------------------------------------------

        run_training_with_trace(train_pipeline)

        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        model_resolver = ModelResolver(
            model_dir=SAVED_MODEL_DIR
        )

        if model_resolver.is_model_exists():

            model_path = (
                model_resolver.get_best_model_path()
            )

            print("MODEL FOUND:", model_path)

            return Response(
                content=(
                    "Training Successful. "
                    f"Model created at: {model_path}"
                ),
                status_code=200
            )

        return Response(
            content=(
                "Training completed, "
                "but model was not found."
            ),
            status_code=500
        )

    except Exception as e:

        print("=" * 60)
        print("TRAINING FAILED")
        print("=" * 60)

        traceback.print_exc()

        return Response(
            content=f"Training failed: {repr(e)}",
            status_code=500
        )


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict_route(
    file: UploadFile = File(...)
):

    try:

        # ---------------------------------------------
        # CHECK MODEL
        # ---------------------------------------------

        model_resolver = ModelResolver(
            model_dir=SAVED_MODEL_DIR
        )

        if not model_resolver.is_model_exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Model not found on server. "
                    "Make sure the trained model exists "
                    "in the configured model directory."
                )
            )

        print("Model found")

        # ---------------------------------------------
        # SAVE UPLOADED FILE
        # ---------------------------------------------

        temp_file_path = "prediction.csv"

        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # ---------------------------------------------
        # READ CSV
        # ---------------------------------------------

        df = pd.read_csv(temp_file_path)

        print("Original Prediction Shape:", df.shape)

        # ---------------------------------------------
        # REPLACE NA
        # ---------------------------------------------

        df.replace("na", np.nan, inplace=True)

        # ---------------------------------------------
        # LOAD SCHEMA
        # ---------------------------------------------

        schema = read_yaml_file(SCHEMA_FILE_PATH)

        drop_columns = schema.get("drop_columns", [])

        # ---------------------------------------------
        # DROP UNNECESSARY COLUMNS
        # ---------------------------------------------

        df.drop(
            columns=drop_columns,
            inplace=True,
            errors="ignore"
        )

        # ---------------------------------------------
        # REMOVE TARGET
        # ---------------------------------------------

        if "class" in df.columns:
            df.drop(
                columns=["class"],
                inplace=True
            )

        # ---------------------------------------------
        # TRAINING FEATURE ORDER
        # ---------------------------------------------

        feature_columns = schema["numerical_columns"]

        feature_columns = [
            col
            for col in feature_columns
            if col not in drop_columns
        ]

        # ---------------------------------------------
        # ADD MISSING COLUMNS
        # ---------------------------------------------

        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        # ---------------------------------------------
        # KEEP TRAINING COLUMNS
        # ---------------------------------------------

        df = df[feature_columns]

        print("Final Prediction Shape:", df.shape)

        # ---------------------------------------------
        # LOAD MODEL
        # ---------------------------------------------

        model_path = model_resolver.get_best_model_path()

        print("Loading model from:", model_path)

        model = load_object(model_path)

        print("Model loaded successfully")

        # ---------------------------------------------
        # MLFLOW PREDICTION TRACE
        # ---------------------------------------------

        prediction = run_prediction_with_trace(
            model,
            df
        )

        df["prediction"] = prediction

        # ---------------------------------------------
        # REVERSE TARGET MAPPING
        # ---------------------------------------------

        df["prediction"] = df["prediction"].replace(
            TargetValueMapping().reverse_mapping()
        )

        # ---------------------------------------------
        # OUTPUT
        # ---------------------------------------------

        output_path = "prediction_output.csv"

        df.to_csv(
            output_path,
            index=False
        )

        print("Prediction completed successfully")

        # ---------------------------------------------
        # RETURN FILE
        # ---------------------------------------------

        return FileResponse(
            output_path,
            filename="prediction_output.csv",
            media_type="text/csv"
        )

    except HTTPException:
        raise

    except Exception as e:

        print("=" * 60)
        print("PREDICTION FAILED")
        print("=" * 60)

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    from uvicorn import run

    try:
        run(
            "main:app",
            host="0.0.0.0",
            port=int(
                os.getenv("PORT", APP_PORT)
            )
        )

    except Exception as e:
        logging.exception(e)
        print(e)
```
