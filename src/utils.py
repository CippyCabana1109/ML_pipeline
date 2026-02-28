"""
Utility functions for solar PV forecasting project
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        sMAPE as percentage
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape

def calculate_metrics(y_true, y_pred, daytime_filter=None):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        daytime_filter: Boolean array for daytime filtering (optional)
        
    Returns:
        Dictionary of metrics
    """
    if daytime_filter is not None:
        y_true = y_true[daytime_filter]
        y_pred = y_pred[daytime_filter]
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'smape': calculate_smape(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics

def create_daytime_filter(irradiance_series):
    """
    Create daytime filter based on irradiance > 0
    
    Args:
        irradiance_series: Pandas Series of irradiance values
        
    Returns:
        Boolean array indicating daytime hours
    """
    return irradiance_series > 0

def format_metrics_table(model_results):
    """
    Format model results into a comparison table
    
    Args:
        model_results: Dictionary of model results
        
    Returns:
        Pandas DataFrame with formatted metrics
    """
    metrics_df = pd.DataFrame(model_results).T
    metrics_df.columns = ['MAE (W)', 'RMSE (W)', 'sMAPE (%)', 'R²']
    
    # Round to appropriate decimal places
    metrics_df['MAE (W)'] = metrics_df['MAE (W)'].round(2)
    metrics_df['RMSE (W)'] = metrics_df['RMSE (W)'].round(2)
    metrics_df['sMAPE (%)'] = metrics_df['sMAPE (%)'].round(2)
    metrics_df['R²'] = metrics_df['R²'].round(4)
    
    return metrics_df

def save_results(results_dict, filepath):
    """
    Save results dictionary to CSV file
    
    Args:
        results_dict: Dictionary containing results
        filepath: Path to save the CSV file
    """
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def load_data(file_path):
    """
    Load and prepare data for modeling
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Prepared pandas DataFrame
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df
