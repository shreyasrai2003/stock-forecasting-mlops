import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from pytz import timezone
from azure.storage.blob import BlobServiceClient
from io import StringIO

AZURE_CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")
CONTAINER_NAME = "stock-forecasting-mlops-dataset"
BLOB_NAME = "data/dataset.csv"
TAIL_FILE_PATH = "data/tail.csv"

API_KEY = "NLLJW4Y93AVKHNNR"
SYMBOL = "IBM"
INTERVAL = "5min"
URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={SYMBOL}&interval={INTERVAL}&outputsize=full&apikey={API_KEY}"

response = requests.get(URL)
data = response.json()

time_series = data.get("Time Series (5min)", {})

if time_series:
    df = pd.DataFrame.from_dict(time_series, orient="index")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    eastern = timezone("US/Eastern")
    df.index = df.index.tz_localize("US/Eastern").tz_convert(eastern)

    previous_date = (datetime.now(eastern) - timedelta(days=1)).strftime("%Y-%m-%d")
    df = df[df.index.strftime("%Y-%m-%d") == previous_date]
    df = df.apply(pd.to_numeric)

    if not df.empty:
        os.makedirs(os.path.dirname(TAIL_FILE_PATH), exist_ok=True)
        df.to_csv(TAIL_FILE_PATH, index_label="Timestamp")
        print(f"API data saved locally to {TAIL_FILE_PATH}")
    else:
        print("No intraday data available for the previous day.")
        exit()

    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    try:
        blob_client = container_client.get_blob_client(BLOB_NAME)
        existing_blob = blob_client.download_blob().readall().decode("utf-8")
        existing_df = pd.read_csv(StringIO(existing_blob), index_col="Timestamp", parse_dates=True)

        combined_df = pd.concat([existing_df, df])
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

        with StringIO() as buffer:
            combined_df.to_csv(buffer, index_label="Timestamp")
            buffer.seek(0)
            blob_client.upload_blob(buffer.getvalue(), overwrite=True)

        print(f"Data successfully updated in Azure Blob Storage: {BLOB_NAME}")

    except Exception as e:
        print(f"Error processing existing blob: {e}")

        with StringIO() as buffer:
            df.to_csv(buffer, index_label="Timestamp")
            buffer.seek(0)
            blob_client.upload_blob(buffer.getvalue(), overwrite=True)

        print(f"New data successfully uploaded to Azure Blob Storage: {BLOB_NAME}")
else:
    print("Error: 'Time Series (5min)' not found in the API response.")
    print(data)
