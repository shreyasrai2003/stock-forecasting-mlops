import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
from keras.models import load_model
import joblib  # Use joblib for saving/loading the scaler

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    df.set_index('Date', inplace=True)
    data = df.filter(['Close'])
    return data.values, df

# Create training and testing datasets
def create_train_test_data(dataset, scaler):
    training_data_len = int(np.ceil(len(dataset) * .95))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:int(training_data_len), :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, scaler

# Build and train the LSTM model
def build_and_train_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=3)
    return model

# Evaluate the model
def evaluate_model(model, x_test, y_test, scaler):
    predictions = scaler.inverse_transform(model.predict(x_test))
    return np.sqrt(np.mean(((predictions - y_test) ** 2)))

# Load the best model and scaler
def load_best_model(model_filename, scaler_filename):
    model = load_model(model_filename)  # Load the model using Keras
    scaler = joblib.load(scaler_filename)  # Load the scaler using joblib
    return model, scaler

# Main function
def main():
    file_path = 'dataset_daily.csv'
    best_model_filename = 'best_lstm_stock_model_daily.h5'  # Save as .h5 file
    best_scaler_filename = 'best_scaler_daily.joblib'  # Save as .joblib file

    # Load and preprocess data
    dataset, _ = load_and_preprocess_data(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train, y_train, x_test, y_test, scaler = create_train_test_data(dataset, scaler)

    # Train new model
    print("Training new model...")
    new_model = build_and_train_model(x_train, y_train)
    new_rmse = evaluate_model(new_model, x_test, y_test, scaler)
    print(f"New Model RMSE: {new_rmse}")

    # Load best model
    if os.path.exists(best_model_filename) and os.path.exists(best_scaler_filename):
        print("Loading best model...")
        best_model, best_scaler = load_best_model(best_model_filename, best_scaler_filename)
        best_rmse = evaluate_model(best_model, x_test, y_test, best_scaler)
        print(f"Best Model RMSE: {best_rmse}")

        # Compare models
        if new_rmse < best_rmse:
            print("New model is better. Saving new model as the best model.")
            best_model, best_scaler = new_model, scaler
    else:
        print("No existing best model. Saving new model as the best model.")
        best_model, best_scaler = new_model, scaler

    # Save the best model and scaler
    best_model.save(best_model_filename)  # Save the model as .h5 file
    joblib.dump(scaler, best_scaler_filename)  # Save the scaler as .joblib file

    print(f"Best model saved as {best_model_filename}")
    print(f"Best scaler saved as {best_scaler_filename}")

if __name__ == "__main__":
    main()