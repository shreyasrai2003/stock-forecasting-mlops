import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from pytz import timezone
from azure.storage.blob import BlobServiceClient
from io import StringIO

# Azure and API configurations
AZURE_CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")
CONTAINER_NAME = "stock-forecasting-mlops-dataset"
BLOB_NAME = "data/dataset_daily.csv"  # Updated for daily data
TAIL_FILE_PATH = "data/tail_daily.csv"  # Updated for daily data

API_KEY = "NLLJW4Y93AVKHNNR"  # Replace with your Alpha Vantage API key
SYMBOL = "IBM"  # Replace with your desired stock symbol
URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={API_KEY}&outputsize=full"

# Fetch daily data from Alpha Vantage API
response = requests.get(URL)
data = response.json()

# Extract the daily time series data
time_series = data.get("Time Series (Daily)", {})

if time_series:
    # Convert the time series data into a DataFrame
    df = pd.DataFrame.from_dict(time_series, orient="index")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Convert the index to Eastern Time (ET)
    eastern = timezone("US/Eastern")
    df.index = df.index.tz_localize("UTC").tz_convert(eastern)

    # Get the last date in the existing dataset from Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    try:
        # Download the existing dataset from Azure Blob Storage
        blob_client = container_client.get_blob_client(BLOB_NAME)
        existing_blob = blob_client.download_blob().readall().decode("utf-8")
        existing_df = pd.read_csv(StringIO(existing_blob), index_col="Date", parse_dates=True)

        # Get the last date in the existing dataset
        last_date = existing_df.index.max()
    except Exception as e:
        print(f"Error processing existing blob: {e}")
        last_date = None

    # If no existing dataset, set the last date to a default (e.g., 1 year ago)
    if last_date is None:
        last_date = datetime.now(eastern) - timedelta(days=365)

    # Filter data for dates after the last date in the existing dataset and up to yesterday
    yesterday = datetime.now(eastern) - timedelta(days=1)
    new_data = df[(df.index > last_date) & (df.index <= yesterday)]

    if not new_data.empty:
        # Save the new data locally
        os.makedirs(os.path.dirname(TAIL_FILE_PATH), exist_ok=True)
        new_data.to_csv(TAIL_FILE_PATH, index_label="Date")
        print(f"New data saved locally to {TAIL_FILE_PATH}")

        # Combine the existing data with the new data
        combined_df = pd.concat([existing_df, new_data])
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]  # Remove duplicates

        # Upload the updated dataset back to Azure Blob Storage
        with StringIO() as buffer:
            combined_df.to_csv(buffer, index_label="Date")
            buffer.seek(0)
            blob_client.upload_blob(buffer.getvalue(), overwrite=True)

        print(f"Data successfully updated in Azure Blob Storage: {BLOB_NAME}")
    else:
        print("No new data to add.")
else:
    print("Error: 'Time Series (Daily)' not found in the API response.")
    print(data)