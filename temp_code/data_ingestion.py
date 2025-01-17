import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    processed_train_path: str = os.path.join('artifacts', "processed_train.npy")
    processed_test_path: str = os.path.join('artifacts', "processed_test.npy")

class DataIngestion:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Load dataset
            df = pd.read_csv(self.data_dir)

            # Create directory for saving data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Data transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path, class_names_path = data_transformation.initiate_data_transformation(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

            # Save processed data
            np.save(self.ingestion_config.processed_train_path, train_arr)
            np.save(self.ingestion_config.processed_test_path, test_arr)

            return (
                self.ingestion_config.processed_train_path,
                self.ingestion_config.processed_test_path,
                preprocessor_path,
                class_names_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='artifacts')
    args = parser.parse_args()
    
    data_dir = os.path.join(args.train_data, 'data.csv')
    
    obj = DataIngestion(data_dir)
    processed_train_path, processed_test_path, preprocessor_path, class_names_path = obj.initiate_data_ingestion()
    print(f"Processed train data path: {processed_train_path}")
    print(f"Processed test data path: {processed_test_path}")
    print(f"Preprocessor object path: {preprocessor_path}")
    print(f"Class names path: {class_names_path}")