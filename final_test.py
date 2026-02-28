import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create comprehensive weather data for 2024
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start_date, end_date, freq='H')

# Generate realistic weather data
np.random.seed(42)
data = []

for date in date_range:
    hour = date.hour
    
    # Simulate irradiance (0 at night, peak during day)
    if 6 <= hour <= 18:
        base_irradiance = 800 * np.sin((hour - 6) * np.pi / 12)
        irradiance = max(0, base_irradiance + np.random.normal(0, 50))
    else:
        irradiance = 0
    
    # Temperature varies with time of day and season
    seasonal_temp = 20 + 10 * np.sin((date.dayofyear - 80) * 2 * np.pi / 365)
    daily_temp = 5 * np.sin((hour - 6) * 2 * np.pi / 24)
    temperature = seasonal_temp + daily_temp + np.random.normal(0, 2)
    
    # Humidity inversely related to temperature
    humidity = max(20, min(95, 70 - temperature + np.random.normal(0, 10)))
    
    # Solar power proportional to irradiance
    solar_power = irradiance * 100
    
    data.append({
        'YEAR': date.year,
        'MO': date.month,
        'DY': date.day,
        'HR': date.hour,
        'ALLSKY_SFC_SW_DWN': irradiance,
        'T2M': temperature,
        'RH2M': humidity,
        'solar_power_w': solar_power
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to file
df.to_csv('data/Weather_Data_Clean.csv', index=False)
print(f"Created weather data with {len(df)} records from {start_date} to {end_date}")

# Show sample
print("\nSample data:")
print(df.head())
