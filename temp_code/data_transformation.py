import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import save_object
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    label_encoder_obj_file_path = os.path.join('artifacts', "label_encoder.pkl")
    class_names_file_path = os.path.join('artifacts', "class_names.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.85
            )
            return tfidf_vectorizer
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Columns for text input and labels
            text_column = "text"
            target_column = "label"
            input_feature_train_df = train_df[text_column]
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df[text_column]
            target_feature_test_df = test_df[target_column]

            # Encode target labels
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            # Save label encoder object
            save_object(
                file_path=self.data_transformation_config.label_encoder_obj_file_path,
                obj=label_encoder
            )

            # Save class names
            class_names = label_encoder.classes_
            with open(self.data_transformation_config.class_names_file_path, 'wb') as f:
                pickle.dump(class_names, f)

            # Initialize and apply TF-IDF vectorizer
            preprocessing_obj = self.get_data_transformer_object()
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()

            # Combine input features and labels into a single array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save TF-IDF preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.class_names_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)