import os
import json

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from source.logger import logging


from source.utils import save_object

@dataclass

class ModelTrainerConfig():

    trained_model_file_path = os.path.join("artifacts",'model.pkl')
    metrics_file_path = os.path.join("artifacts", "metrics.json")

class ModelTrainer:
    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        logging.info("Spliting the dataset into train and tests sets")

        X_train, y_train, X_test, y_test = (
            train_array[:, :-1], 
            train_array[:, -1], 
            test_array[:, :-1], 
            test_array[:, -1]
        )

        best_model_name = "RandomForestClassifier"
        best_params = {
            "criterion": "gini",
            "max_features": "log2",
            "n_estimators": 25,
            "random_state": 42,
        }
        best_model_instance = RandomForestClassifier(**best_params)

        logging.info(
            "Using notebook-selected RandomForest parameters for reproducible training: %s",
            best_params,
        )
        best_model_instance.fit(X_train, y_train)
        
        # Evaluate the tuned model
        y_train_pred = best_model_instance.predict(X_train)
        y_pred = best_model_instance.predict(X_test)
        metrics = {
            "best_model": best_model_name,
            "best_params": best_params,
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "val_accuracy": accuracy_score(y_test, y_pred),
            "train_f1_weighted": f1_score(y_train, y_train_pred, average="weighted"),
            "val_f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        }

        logging.info(f"Tuned model metrics: {metrics}") 

        # Save the best model
        save_object(
            file_path = self.model_trainer_config.trained_model_file_path,
            obj = best_model_instance
        )

        metrics_dir = os.path.dirname(self.model_trainer_config.metrics_file_path)
        os.makedirs(metrics_dir, exist_ok=True)
        with open(self.model_trainer_config.metrics_file_path, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

        return metrics
            
