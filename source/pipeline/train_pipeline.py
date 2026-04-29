import json

from source.components.data_ingestion import DataIngestion
from source.components.data_transformation import DataTransformation
from source.components.model_trainer import ModelTrainer


def run_training():
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_data_path,
        test_data_path,
    )

    trainer = ModelTrainer()
    return trainer.initiate_model_trainer(train_arr, test_arr)


if __name__ == "__main__":
    metrics = run_training()
    print(json.dumps(metrics, indent=2))
