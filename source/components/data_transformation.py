import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from source.exception import CustomException
from source.feature_config import TARGET_COLUMN, get_training_drop_columns
from source.logger import logging
from source.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprecessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

    def strip_and_lower_columns(self, df):
        logging.info("Stripping and lowercasing column names.")
        prepared_df = df.copy()
        prepared_df.columns = prepared_df.columns.str.strip().str.lower()
        return prepared_df

    def drop_unnecessary_columns(self, df):
        columns_to_drop = [column for column in get_training_drop_columns() if column in df.columns]
        logging.info("Dropping unnecessary columns: %s", columns_to_drop)
        return df.drop(columns=columns_to_drop, errors="ignore")

    def prepare_features(self, df):
        prepared_df = self.strip_and_lower_columns(df)
        return self.drop_unnecessary_columns(prepared_df)

    def get_data_transformer_object(self, df):
        logging.info("Handling data columns")
        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehotencoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        logging.info("Categorical columns: %s", categorical_columns)
        logging.info("Numerical columns: %s", numerical_columns)

        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical_pipeline", numerical_pipeline, numerical_columns),
                ("categorical_pipeline", categorical_pipeline, categorical_columns),
            ]
        )
        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info("Obtaining preprocessing object")
            logging.info("Splitting the data into features and target")

            input_feature_train_df = self.prepare_features(
                train_df.drop(columns=[TARGET_COLUMN], axis=1)
            )
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = self.prepare_features(
                test_df.drop(columns=[TARGET_COLUMN], axis=1)
            )
            target_feature_test_df = test_df[TARGET_COLUMN]

            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training data and testing data")

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.values.reshape(-1, 1)]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values.reshape(-1, 1)]

            save_object(
                file_path=self.data_transform_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Saved preprocessing object successfully")

            return (
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error("Error occurred in data transformation: %s", str(e))
            raise CustomException(e, sys)
