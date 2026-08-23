import sys
from typing import Optional
import json

import numpy as np
import pandas as pd

from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.constant.database import DATABASE_NAME
from sensor.exception import SensorException


class SensorData:
    """
    Export MongoDB collection as pandas DataFrame
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(
                database_name=DATABASE_NAME
            )

        except Exception as e:
            raise SensorException(e, sys)

    def save_csv_file(
        self,
        file_path: str,
        collection_name: str,
        database_name: Optional[str] = None
    ):
        """
        Read CSV file and insert its records into MongoDB.
        """

        try:
            # ======================================
            # READ CSV
            # ======================================

            data_frame = pd.read_csv(file_path)

            data_frame.reset_index(
                drop=True,
                inplace=True
            )

            # ======================================
            # CONVERT DATAFRAME TO RECORDS
            # ======================================

            records = list(
                json.loads(
                    data_frame.T.to_json()
                ).values()
            )

            # ======================================
            # GET MONGODB COLLECTION
            # ======================================

            if database_name is None:
                collection = (
                    self.mongo_client.database[
                        collection_name
                    ]
                )

            else:
                collection = (
                    self.mongo_client.client[
                        database_name
                    ][
                        collection_name
                    ]
                )

            # ======================================
            # INSERT RECORDS
            # ======================================

            if records:
                collection.insert_many(records)

            print(
                f"Successfully inserted {len(records)} "
                f"records into MongoDB"
            )

            return len(records)

        except Exception as e:
            raise SensorException(e, sys)

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export MongoDB collection as pandas DataFrame.
        """

        try:
            # ======================================
            # GET MONGODB COLLECTION
            # ======================================

            if database_name is None:
                collection = (
                    self.mongo_client.database[
                        collection_name
                    ]
                )

            else:
                collection = (
                    self.mongo_client.client[
                        database_name
                    ][
                        collection_name
                    ]
                )

            # ======================================
            # FETCH DATA
            # ======================================

            cursor = collection.find()

            df = pd.DataFrame(list(cursor))

            # ======================================
            # EMPTY CHECK
            # ======================================

            if df.shape[0] == 0:
                print(
                    "No data found in MongoDB collection"
                )
                return pd.DataFrame()

            # ======================================
            # DROP MONGODB _id
            # ======================================

            if "_id" in df.columns:
                df.drop(
                    columns=["_id"],
                    inplace=True
                )

            # ======================================
            # REPLACE "na" WITH NaN
            # ======================================

            df.replace(
                "na",
                np.nan,
                inplace=True
            )

            print(
                f"DataFrame Shape: {df.shape}"
            )

            return df

        except Exception as e:
            raise SensorException(e, sys)