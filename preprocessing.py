import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("csv_data/air_quality_data.csv")

# Handle missing values
# Fill missing numerical values with the median of their respective columns
data.fillna(data.median(numeric_only=True), inplace=True)

# Remove rows with identical values in all columns except the timestamp
data = data[~data.drop(columns=['timestamp']).duplicated(keep='first')]

# Remove outliers using the Interquartile Range (IQR) method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal to relevant columns
columns_to_check = ["air_quality_index", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
for column in columns_to_check:
    data = remove_outliers(data, column)

# Save the cleaned data to a new CSV file
data.to_csv("development/training_data/preprocessed_data.csv", index=False)
print("Data preprocessing done.")