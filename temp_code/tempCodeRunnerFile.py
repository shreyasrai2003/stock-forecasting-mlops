from flask import Flask, render_template
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import joblib  # Use joblib for loading the scaler
from keras.models import load_model
app = Flask(__name__)

# Load the best model and scaler
def load_best_model():
    # Correct relative path to the model folder
    model_folder = os.path.join(os.path.dirname(__file__), 'model')
    model_filename = os.path.join(model_folder, 'best_lstm_stock_model_daily.h5')  # Load .h5 file
    scaler_filename = os.path.join(model_folder, 'best_scaler_daily.joblib')  # Load .joblib file

    model = load_model(model_filename)  # Load the model using Keras
    scaler = joblib.load(scaler_filename)  # Load the scaler using joblib
    return model, scaler

# Load the dataset and get the last 60 days of data
def load_last_60_days_data():
    # Correct relative path to the dataset file
    file_path = os.path.join(os.path.dirname(__file__), 'dataset_daily.csv')
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    df.set_index('Date', inplace=True)
    last_60_days = df['Close'].values[-60:]
    return last_60_days, df

# Predict the current day's stock price
def predict_current_day_price(model, scaler, last_60_days):
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    x_test = np.array([last_60_days_scaled])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predicted_price_scaled = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
    return predicted_price

# Get the last 5 actual stock prices
def get_last_5_actual_prices(df):
    last_5_prices = df['Close'].tail(5).values
    last_5_dates = df.index[-5:].strftime('%Y-%m-%d').tolist()
    return last_5_dates, last_5_prices

# Determine the KPI arrow color based on the trend
def get_kpi_color(last_5_prices, predicted_price):
    # Compare the predicted price with the last value
    if predicted_price > last_5_prices[-1]:
        return 'green'
    else:
        return 'red'


# Generate a graph for the last 5 actual prices and predicted price
def generate_graph(last_5_dates, last_5_prices, predicted_price):
    # Convert last_5_prices to a list
    last_5_prices = list(last_5_prices)
    
    # Add today's date for the predicted price
    today_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # Use yesterday's date for prediction
    last_5_dates.append(today_date)
    last_5_prices.append(predicted_price)

    plt.figure(figsize=(8, 4))
    plt.plot(last_5_dates, last_5_prices, marker='o', label='Actual Prices', color='blue')
    plt.plot(today_date, predicted_price, marker='o', label='Predicted Price', color='yellow')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Last 5 Actual Prices and Predicted Price')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the plot as a base64 string
    graph_data = base64.b64encode(buf.read()).decode('utf-8')
    return graph_data

# Calculate the average price of the month
def calculate_average_price(df):
    current_month = df.index[-1].strftime('%Y-%m')  # Get the current month
    monthly_data = df[df.index.strftime('%Y-%m') == current_month]
    return monthly_data['Close'].mean()

@app.route('/')
def home():
    # Load the best model and scaler
    model, scaler = load_best_model()

    # Load the last 60 days of data
    last_60_days, df = load_last_60_days_data()

    # Predict the current day's stock price
    predicted_price = predict_current_day_price(model, scaler, last_60_days)

    # Get the last 5 actual stock prices
    last_5_dates, last_5_prices = get_last_5_actual_prices(df)

    # Determine the KPI arrow color (including the predicted price in the trend analysis)
    kpi_color = get_kpi_color(last_5_prices, predicted_price)

    # Generate the graph
    graph_data = generate_graph(last_5_dates, last_5_prices, predicted_price)

    # Calculate the average price of the month
    average_price = calculate_average_price(df)

    # Render the template with the data
    return render_template(
        'index.html',
        predicted_price=f"{predicted_price:.2f}",
        kpi_color=kpi_color,
        last_5_dates=last_5_dates,
        last_5_prices=last_5_prices,
        graph_data=graph_data,
        average_price=f"{average_price:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)