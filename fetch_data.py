import requests
import json
import subprocess
import pandas as pd
from datetime import datetime, timedelta
import os

API_KEY = '6967ee16a8ccf1c678f9b7660f7ec505'

# Base URLs for different types of data
BASE_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
BASE_AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
BASE_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Directory to store fetched data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def send_to_model(data):
    api_url = "http://localhost:5000/predict"  # Replace with your model's API endpoint
    response = requests.post(api_url, json=data)
    print("Prediction Response:", response.json())

def save_and_version_data(file_path):
    """Save data file, stage with DVC, and push to remote storage."""
    try:
        # Add the file to DVC
        subprocess.run(["dvc", "add", file_path], check=True)

        # Commit changes to DVC
        subprocess.run(["dvc", "commit", file_path], check=True)

        # Push changes to remote storage
        subprocess.run(["dvc", "push"], check=True)
        print(f"Data versioned and pushed: {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during DVC operations: {e}")

def update_data(file_path, new_data):
    """
    Update the existing JSON file with new data, keeping only the last 10 days.
    """
    # Load existing data if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # Append new data with a timestamp
    new_entry = {"timestamp": datetime.now().isoformat(), "data": new_data}
    combined_data = existing_data + [new_entry]

    # Filter data to keep only the last 10 days
    ten_days_ago = datetime.now() - timedelta(days=1000)
    filtered_data = [
        entry for entry in combined_data
        if datetime.fromisoformat(entry["timestamp"]) > ten_days_ago
    ]
    

    # Save the filtered data back to the file
    with open(file_path, "w") as file:
        json.dump(filtered_data, file, indent=4)
    print(f"Updated data saved to {file_path}")
    
    #save_and_version_data(file_path)

def fetch_weather_data(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    response = requests.get(BASE_WEATHER_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        file_path = os.path.join(DATA_DIR, "weather_data.json")
        update_data(file_path, data)
    else:
        print(f"Failed to fetch weather data: {response.status_code}")

def fetch_air_pollution_data(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    response = requests.get(BASE_AIR_POLLUTION_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        file_path = os.path.join(DATA_DIR, "air_pollution_data.json")
        update_data(file_path, data)
    else:
        print(f"Failed to fetch air pollution data: {response.status_code}")

def fetch_forecast_data(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    response = requests.get(BASE_FORECAST_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        file_path = os.path.join(DATA_DIR, "forecast_data.json")
        update_data(file_path, data)
    else:
        print(f"Failed to fetch forecast data: {response.status_code}")

# Example: Islamabad, Pakistan coordinates
lat, lon = 33.6844, 73.0479

fetch_weather_data(lat, lon)
fetch_air_pollution_data(lat, lon)
fetch_forecast_data(lat, lon)

#data = "data/air_pollution_data.json"
#send_to_model(data)

