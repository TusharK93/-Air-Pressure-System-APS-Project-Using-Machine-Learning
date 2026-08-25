import os

from sensor.constant.training_pipeline import (
    SAVED_MODEL_DIR,
    MODEL_FILE_NAME
)


class TargetValueMapping:

    def __init__(self):

        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self):

        return self.__dict__

    def reverse_mapping(self):

        mapping_response = self.to_dict()

        return dict(
            zip(
                mapping_response.values(),
                mapping_response.keys()
            )
        )


class SensorModel:

    def __init__(self, preprocessor, model):

        try:
            self.preprocessor = preprocessor
            self.model = model

        except Exception as e:
            raise e

    def predict(self, x):

        try:

            x_transform = self.preprocessor.transform(x)

            y_hat = self.model.predict(x_transform)

            return y_hat

        except Exception as e:
            raise e


class ModelResolver:

    def __init__(self, model_dir=SAVED_MODEL_DIR):

        try:

            self.model_dir = model_dir

        except Exception as e:
            raise e

    def get_best_model_path(self) -> str:

        try:

            if not os.path.exists(self.model_dir):

                raise FileNotFoundError(
                    f"Model directory does not exist: "
                    f"{self.model_dir}"
                )

            # Only accept numeric timestamp directories
            timestamps = []

            for item in os.listdir(self.model_dir):

                item_path = os.path.join(
                    self.model_dir,
                    item
                )

                if (
                    os.path.isdir(item_path)
                    and item.isdigit()
                ):

                    timestamps.append(
                        int(item)
                    )

            if not timestamps:

                raise FileNotFoundError(
                    f"No timestamp model directories found "
                    f"inside: {self.model_dir}"
                )

            latest_timestamp = max(timestamps)

            latest_model_path = os.path.join(
                self.model_dir,
                str(latest_timestamp),
                MODEL_FILE_NAME
            )

            if not os.path.isfile(latest_model_path):

                raise FileNotFoundError(
                    f"Model file not found: "
                    f"{latest_model_path}"
                )

            return latest_model_path

        except Exception as e:

            raise e

    def is_model_exists(self) -> bool:

        try:

            if not os.path.exists(self.model_dir):

                return False

            model_path = self.get_best_model_path()

            return os.path.isfile(model_path)

        except Exception:

            return False