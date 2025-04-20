import pandas as pd
import json
import os
from datetime import datetime

# Directory to store CSV files
CSV_DIR = "csv_data"
os.makedirs(CSV_DIR, exist_ok=True)

def process_weather_data(json_file, csv_file):
    """Process weather_data.json and save it as CSV."""
    with open(json_file, "r") as file:
        data = json.load(file)

    # Flatten relevant fields
    rows = []
    for entry in data:
        weather = entry["data"]
        row = {
            "timestamp": entry["timestamp"],
            "longitude": weather["coord"]["lon"],
            "latitude": weather["coord"]["lat"],
            "weather_main": weather["weather"][0]["main"],
            "weather_description": weather["weather"][0]["description"],
            "temperature": weather["main"]["temp"],
            "feels_like": weather["main"]["feels_like"],
            "temp_min": weather["main"]["temp_min"],
            "temp_max": weather["main"]["temp_max"],
            "pressure": weather["main"]["pressure"],
            "humidity": weather["main"]["humidity"],
            "visibility": weather.get("visibility", None),
            "wind_speed": weather["wind"]["speed"],
            "wind_deg": weather["wind"]["deg"],
            "wind_gust": weather["wind"].get("gust", None),
            "cloud_coverage": weather["clouds"]["all"],
            "country": weather["sys"]["country"],
            "sunrise": datetime.utcfromtimestamp(weather["sys"]["sunrise"]).isoformat(),
            "sunset": datetime.utcfromtimestamp(weather["sys"]["sunset"]).isoformat(),
            "city": weather["name"]
        }
        rows.append(row)

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"Weather data saved to {csv_file}")

def process_air_pollution_data(json_file, csv_file):
    """Process air_pollution_data.json and save it as CSV."""
    with open(json_file, "r") as file:
        data = json.load(file)

    # Flatten relevant fields
    rows = []
    for entry in data:
        pollution = entry["data"]["list"][0]  # Assuming only one entry in the "list"
        row = {
            "timestamp": entry["timestamp"],
            "longitude": entry["data"]["coord"]["lon"],
            "latitude": entry["data"]["coord"]["lat"],
            "air_quality_index": pollution["main"]["aqi"],
            "co": pollution["components"]["co"],
            "no": pollution["components"]["no"],
            "no2": pollution["components"]["no2"],
            "o3": pollution["components"]["o3"],
            "so2": pollution["components"]["so2"],
            "pm2_5": pollution["components"]["pm2_5"],
            "pm10": pollution["components"]["pm10"],
            "nh3": pollution["components"]["nh3"]
        }
        rows.append(row)

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"Air pollution data saved to {csv_file}")


def process_forecast_data(json_file, csv_file):
    """Process forecast_data.json and save it as CSV."""
    with open(json_file, "r") as file:
        data = json.load(file)

    # Ensure we are working with a list of entries
    if not isinstance(data, list):
        print("Invalid data format: Expected a list of forecast entries.")
        return

    # Flatten forecast list
    rows = []
    for entry in data:
        timestamp = entry["timestamp"]
        forecast_data = entry["data"]

        for forecast in forecast_data.get("list", []):
            row = {
                "timestamp": timestamp,
                "forecast_time": forecast["dt_txt"],
                "temperature": forecast["main"]["temp"],
                "feels_like": forecast["main"]["feels_like"],
                "temp_min": forecast["main"]["temp_min"],
                "temp_max": forecast["main"]["temp_max"],
                "pressure": forecast["main"]["pressure"],
                "humidity": forecast["main"]["humidity"],
                "weather_main": forecast["weather"][0]["main"],
                "weather_description": forecast["weather"][0]["description"],
                "cloud_coverage": forecast["clouds"]["all"],
                "wind_speed": forecast["wind"]["speed"],
                "wind_deg": forecast["wind"]["deg"],
                "visibility": forecast.get("visibility", None),
            }
            rows.append(row)

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"Forecast data saved to {csv_file}")

# Paths to JSON files and corresponding CSV files
weather_json = "data/weather_data.json"
weather_csv = os.path.join(CSV_DIR, "weather_data.csv")

air_pollution_json = "data/air_pollution_data.json"
air_pollution_csv = os.path.join(CSV_DIR, "air_pollution_data.csv")

forecast_json = "data/forecast_data.json"
forecast_csv = os.path.join(CSV_DIR, "forecast_data.csv")

# Process the data
process_weather_data(weather_json, weather_csv)
process_air_pollution_data(air_pollution_json, air_pollution_csv)
process_forecast_data(forecast_json, forecast_csv)