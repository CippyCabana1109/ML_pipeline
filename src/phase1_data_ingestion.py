import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import glob

def merge_solar_production_data():
    """
    Merge all monthly solar production CSV files into a unified dataframe
    """
    print("Starting solar production data merge...")
    
    # Get all CSV files in the Solar_Production_Data directory
    data_dir = "Solar_Production_Data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # Sort files by date to ensure proper chronological order
    csv_files.sort()
    
    all_data = []
    
    for file_path in csv_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Extract target variable (Active Power)
        df['solar_power_w'] = df['Active Power Mean (W) - 49960']
        
        # Keep only essential columns
        df_clean = df[['Timestamp', 'solar_power_w']].copy()
        
        all_data.append(df_clean)
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp
    merged_df = merged_df.sort_values('Timestamp').reset_index(drop=True)
    
    # Remove duplicates (keep last occurrence)
    merged_df = merged_df.drop_duplicates(subset=['Timestamp'], keep='last')
    
    print(f"Merged {len(csv_files)} files into {len(merged_df)} records")
    print(f"Date range: {merged_df['Timestamp'].min()} to {merged_df['Timestamp'].max()}")
    
    return merged_df

def get_nasa_power_data():
    """
    Placeholder for NASA POWER data retrieval
    This will need to be implemented based on specific location requirements
    """
    print("NASA POWER data integration needed...")
    print("Please provide:")
    print("- Latitude/Longitude coordinates")
    print("- Preferred API method (direct download vs API calls)")
    
    # For now, return empty dataframe with expected structure
    return pd.DataFrame()

def create_training_test_split(df):
    """
    Split data according to project timeline (adjusted for available data):
    - Training: Jan 2024 – Nov 2024 (11 months)
    - Testing: Dec 1-7, 2024 (last available test week)
    """
    # Convert to datetime if not already
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Define periods (adjusted for available data)
    train_start = '2024-01-01'
    train_end = '2024-11-30'
    test_start = '2024-12-01'
    test_end = '2024-12-07'
    
    # Split data
    train_df = df[(df['Timestamp'] >= train_start) & (df['Timestamp'] <= train_end)].copy()
    test_df = df[(df['Timestamp'] >= test_start) & (df['Timestamp'] <= test_end)].copy()
    
    print(f"Training data: {len(train_df)} records ({train_start} to {train_end})")
    print(f"Test data: {len(test_df)} records ({test_start} to {test_end})")
    
    return train_df, test_df

def create_temporal_features(df):
    """
    Create temporal features for modeling
    """
    df = df.copy()
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['month'] = df['Timestamp'].dt.month
    df['day_of_year'] = df['Timestamp'].dt.dayofyear
    
    # Create lag features (will be filled after weather data integration)
    df['lag_24h'] = df['solar_power_w'].shift(96)  # 96 * 15min = 24h
    df['lag_48h'] = df['solar_power_w'].shift(192)  # 192 * 15min = 48h
    
    return df

def main():
    """
    Main execution function for Phase 1
    """
    print("=" * 50)
    print("PHASE 1: DATA INGESTION & PREPROCESSING")
    print("=" * 50)
    
    # Step 1: Merge solar production data
    solar_df = merge_solar_production_data()
    
    # Step 2: Get NASA weather data (placeholder)
    weather_df = get_nasa_power_data()
    
    # Step 3: Create temporal features
    solar_df = create_temporal_features(solar_df)
    
    # Step 4: Split data
    train_df, test_df = create_training_test_split(solar_df)
    
    # Save processed data
    solar_df.to_csv('processed_solar_data.csv', index=False)
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    print("\nPhase 1 completed!")
    print("Files saved:")
    print("- processed_solar_data.csv (all data)")
    print("- train_data.csv (Jan 2024 - Feb 2025)")
    print("- test_data.csv (March 1-7, 2025)")
    
    return solar_df, train_df, test_df

if __name__ == "__main__":
    solar_df, train_df, test_df = main()
