import os
from azure.storage.blob import BlobServiceClient

# Azure configurations
AZURE_CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")
CONTAINER_NAME = "stock-forecasting-mlops-dataset"
LOCAL_FILE_PATH = "data/dataset_daily.csv"  # Local file to upload
BLOB_NAME = "data/dataset_daily.csv"  # Blob name in Azure

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

try:
    # Check if the container exists, and create it if it doesn't
    if not container_client.exists():
        container_client.create_container()

    # Initialize BlobClient
    blob_client = container_client.get_blob_client(BLOB_NAME)

    # Upload the local file to Azure Blob Storage
    with open(LOCAL_FILE_PATH, "rb") as file_data:
        blob_client.upload_blob(file_data, overwrite=True)

    print(f"File successfully uploaded to Azure Blob Storage: {BLOB_NAME}")
except Exception as e:
    print(f"Error uploading file: {e}")