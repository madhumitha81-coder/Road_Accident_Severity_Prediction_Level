import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from source.exception import CustomException


def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    model_results = {}

    try:

        for model_name, model in models.items():            

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_f1 = f1_score(y_train, y_train_pred, average="weighted")
            test_f1 = f1_score(y_test, y_test_pred, average="weighted")

            model_results[model_name] = {
                "train_accuracy": train_accuracy,
                "val_accuracy": test_accuracy,
                "train_f1_weighted": train_f1,
                "val_f1_weighted": test_f1,
            }

        return model_results
    
    except Exception as e:
        raise CustomException(e, sys)

    
def best_model(result, metric_name="val_f1_weighted"):

    high = float("-inf")
    model_name = None

    for name, metrics in result.items():
        metric_value = metrics.get(metric_name, float("-inf"))
        if metric_value > high:
            high = metric_value
            model_name = name

    return model_name

def best_params(model, param, X_train, y_train):

    try:
        # Define cross-validation strategy
        cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2, random_state = 42)

        grid_cv = GridSearchCV(estimator = model, param_grid = param, cv = cv, scoring = "accuracy")  # scoring="accuracy" for classification
        res = grid_cv.fit(X_train, y_train)

        return res.best_params_

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
