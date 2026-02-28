import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def process_training_dataset():
    """
    Process the provided Training_Dataset.csv which already contains
    integrated weather and production data
    """
    print("Processing Training_Dataset.csv...")
    
    # Read the dataset
    df = pd.read_csv('data/Training_Dataset.csv')
    
    # Convert DateTime to datetime first
    df['timestamp'] = pd.to_datetime(df['DateTime'])
    
    # Extract key weather variables (using original column names)
    df['irradiance'] = df['All Sky Surface Shortwave Downward Irradiance \n(Wh/m2) ']
    df['temperature'] = df['Temperature at 2 Meters (C) ']
    df['humidity'] = df['Relative Humidity at 2 Meters (%) ']
    
    # Calculate total solar power from all inverters
    power_columns = [col for col in df.columns if 'Production Power' in col and '(W)' in col]
    df['solar_power_w'] = df[power_columns].sum(axis=1)
    
    # Create temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # Create lag features
    df['lag_24h'] = df['solar_power_w'].shift(96)  # 96 * 15min = 24h
    df['lag_48h'] = df['solar_power_w'].shift(192)  # 192 * 15min = 48h
    
    # Select relevant columns for modeling
    model_columns = [
        'timestamp', 'solar_power_w', 'irradiance', 'temperature', 'humidity',
        'hour', 'day_of_week', 'month', 'day_of_year', 'lag_24h', 'lag_48h'
    ]
    
    df_model = df[model_columns].copy()
    
    # Remove rows with missing values
    df_model = df_model.dropna()
    
    print(f"Dataset shape after processing: {df_model.shape}")
    print(f"Date range: {df_model['timestamp'].min()} to {df_model['timestamp'].max()}")
    print(f"Missing values: {df_model.isnull().sum().sum()}")
    
    return df_model

def create_training_test_split(df):
    """
    Split data according to project timeline:
    - Training: Jan 2024 – Nov 2024 (11 months)
    - Testing: Dec 1-7, 2024 (test week)
    """
    # Define periods
    train_start = '2024-01-01'
    train_end = '2024-11-30'
    test_start = '2024-12-01'
    test_end = '2024-12-07'
    
    # Split data
    train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)].copy()
    test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)].copy()
    
    print(f"Training data: {len(train_df)} records ({train_start} to {train_end})")
    print(f"Test data: {len(test_df)} records ({test_start} to {test_end})")
    
    return train_df, test_df

def create_daytime_filter(df):
    """
    Create daytime filter for sMAPE calculation (irradiance > 0)
    """
    df['is_daytime'] = df['irradiance'] > 0
    daytime_count = df['is_daytime'].sum()
    total_count = len(df)
    
    print(f"Daytime records: {daytime_count}/{total_count} ({daytime_count/total_count*100:.1f}%)")
    
    return df

def analyze_data_quality(df):
    """
    Analyze data quality and create summary statistics
    """
    print("\nData Quality Analysis:")
    print("=" * 50)
    
    # Basic statistics
    print("Solar Power Statistics:")
    print(df['solar_power_w'].describe())
    
    print(f"\nIrradiance Statistics:")
    print(df['irradiance'].describe())
    
    print(f"\nTemperature Statistics:")
    print(df['temperature'].describe())
    
    print(f"\nHumidity Statistics:")
    print(df['humidity'].describe())
    
    # Check for data gaps
    time_diff = df['timestamp'].diff().dropna()
    expected_interval = pd.Timedelta(minutes=15)
    gaps = time_diff[time_diff != expected_interval]
    
    if len(gaps) > 0:
        print(f"\nData gaps found: {len(gaps)} instances")
        print(f"Largest gap: {gaps.max()}")
    else:
        print("\nNo data gaps found - regular 15-minute intervals")

def main():
    """
    Main processing function
    """
    print("=" * 60)
    print("PROCESSING PROVIDED TRAINING DATASET")
    print("=" * 60)
    
    # Process the dataset
    df = process_training_dataset()
    
    # Create daytime filter
    df = create_daytime_filter(df)
    
    # Analyze data quality
    analyze_data_quality(df)
    
    # Split into training and test sets
    train_df, test_df = create_training_test_split(df)
    
    # Save processed datasets
    df.to_csv('data/processed_training_data.csv', index=False)
    train_df.to_csv('data/train_final.csv', index=False)
    test_df.to_csv('data/test_final.csv', index=False)
    
    print(f"\nFiles saved:")
    print(f"- processed_training_data.csv: {len(df)} records")
    print(f"- train_final.csv: {len(train_df)} records")
    print(f"- test_final.csv: {len(test_df)} records")
    
    return df, train_df, test_df

if __name__ == "__main__":
    df, train_df, test_df = main()
