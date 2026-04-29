import os
import sys
import pandas as pd

from source.exception import CustomException
from source.logger import logging
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from source.components.data_transformation import DataTransformation

from source.components.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)

@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    source_data_path: str = os.path.join('Datasets', 'processed', 'processed_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method")

        try:

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            source_data_path = os.path.join(project_root, self.ingestion_config.source_data_path)
            df = pd.read_csv(source_data_path)

            logging.info("Read the dataset successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["accident_severity"],
            )
            train_set.to_csv(self.ingestion_config.train_data_path, index = False)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(str(e), sys)
        

if __name__ == "__main__":
    
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data,test_data)

    trainer = ModelTrainer()
    print(trainer.initiate_model_trainer(train_arr, test_arr))
