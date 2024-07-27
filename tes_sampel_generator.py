import numpy as np
import pandas as pd

# Creating a synthetic dataset
np.random.seed(42)
n_days = 100
data = {
    'temperature': np.random.uniform(20, 30, n_days),  # Temperature in Celsius
    'wind_speed': np.random.uniform(0, 10, n_days),    # Wind speed in m/s
    'humidity': np.random.uniform(40, 90, n_days),     # Humidity in percentage
    'rain': np.random.choice(['yes', 'no'], n_days)    # Whether it rained or not
}

# Convert to DataFrame
df = pd.DataFrame(data)
print(df.head())

# Save dataset to CSV
df.to_csv('rain_prediction_data.csv', index=False)
