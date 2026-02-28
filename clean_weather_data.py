import pandas as pd

# Clean the Weather Data.csv file
df = pd.read_csv('data/Weather Data.csv', skiprows=15)

# Get the actual data rows (skip header info)
data_rows = df.iloc[1:].copy()

# Reset index and clean column names
data_rows = data_rows.reset_index(drop=True)

# Extract the key columns we need
clean_df = pd.DataFrame({
    'YEAR': data_rows['YEAR'],
    'MO': data_rows['MO'], 
    'DY': data_rows['DY'],
    'HR': data_rows['HR'],
    'ALLSKY_SFC_SW_DWN': data_rows['ALLSKY_SFC_SW_DWN'],
    'T2M': data_rows['T2M'],
    'RH2M': data_rows['RH2M']
})

# Save cleaned version
clean_df.to_csv('data/Weather_Data_Clean.csv', index=False)
print("Cleaned weather data saved to data/Weather_Data_Clean.csv")
