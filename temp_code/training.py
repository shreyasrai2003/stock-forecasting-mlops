import os
import sys
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from src.exception import CustomException
from src.utils import save_object, load_object
from dataclasses import dataclass
import pickle

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "best_model.h5")
    preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")
    train_data_file_path = os.path.join("artifacts", "processed_train.npy")
    test_data_file_path = os.path.join("artifacts", "processed_test.npy")
    class_names_file_path = os.path.join("artifacts", "class_names.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, preprocessor=None):
        try:
            # Load preprocessed data
            train_array = np.load(self.model_trainer_config.train_data_file_path)
            test_array = np.load(self.model_trainer_config.test_data_file_path)

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Load class names
            class_names = load_object(self.model_trainer_config.class_names_file_path)
            num_classes = len(class_names)

            # One-hot encode target variables
            y_train = to_categorical(y_train, num_classes=num_classes)
            y_test = to_categorical(y_test, num_classes=num_classes)

            # Reshape input for LSTM (samples, timesteps, features)
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

            # Build LSTM model
            model = Sequential([
                LSTM(64, activation='tanh', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dense(32, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])

            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Train model
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            start_time = time.time()
            model.fit(X_train, y_train, 
                      validation_data=(X_test, y_test), 
                      epochs=50, 
                      batch_size=32, 
                      callbacks=[early_stopping],
                      verbose=2)
            end_time = time.time()

            print(f"Training completed in {end_time - start_time:.2f} seconds.")

            # Save the trained model
            model.save(self.model_trainer_config.trained_model_file_path)
            print("Best model saved successfully.")

            return model

        except Exception as e:
            print(f"Error occurred: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer()