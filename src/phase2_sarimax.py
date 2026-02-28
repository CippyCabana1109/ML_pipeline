import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
    Only for daytime hours (irradiance > 0)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape

def check_stationarity(series):
    """
    Check stationarity using Augmented Dickey-Fuller test
    """
    result = adfuller(series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    
    if result[1] <= 0.05:
        print("Series is stationary (reject null hypothesis)")
    else:
        print("Series is non-stationary (fail to reject null hypothesis)")
    
    return result[1] <= 0.05

def analyze_acf_pacf(series, lags=50):
    """
    Analyze ACF and PACF plots for parameter selection
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(series.dropna(), lags=lags, ax=ax1, alpha=0.05)
    ax1.set_title('Autocorrelation Function (ACF)')
    
    plot_pacf(series.dropna(), lags=lags, ax=ax2, alpha=0.05, method='ywm')
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.savefig('acf_pacf_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def prepare_sarimax_data(train_df, test_df):
    """
    Prepare data for SARIMAX modeling
    """
    # For SARIMAX, we'll use hourly data to reduce computational complexity
    train_hourly = train_df.set_index('timestamp').resample('H').mean()
    test_hourly = test_df.set_index('timestamp').resample('H').mean()
    
    # Remove any remaining NaN values
    train_hourly = train_hourly.dropna()
    test_hourly = test_hourly.dropna()
    
    print(f"Training data (hourly): {len(train_hourly)} records")
    print(f"Test data (hourly): {len(test_hourly)} records")
    
    return train_hourly, test_hourly

def find_best_sarimax_params(train_data, exog_train):
    """
    Find optimal SARIMAX parameters using grid search
    """
    print("Searching for optimal SARIMAX parameters...")
    
    # Define parameter ranges
    p_values = range(0, 3)  # AR order
    d_values = range(0, 2)  # Differencing
    q_values = range(0, 3)  # MA order
    P_values = range(0, 2)  # Seasonal AR order
    D_values = range(0, 2)  # Seasonal differencing
    Q_values = range(0, 2)  # Seasonal MA order
    s_values = [24]  # Seasonal period (24 hours for daily seasonality)
    
    best_aic = float('inf')
    best_params = None
    
    # Simplified search for demonstration
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for s in s_values:
                                try:
                                    model = SARIMAX(
                                        train_data,
                                        exog=exog_train,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, s),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    results = model.fit(disp=False)
                                    
                                    if results.aic < best_aic:
                                        best_aic = results.aic
                                        best_params = (p, d, q, P, D, Q, s)
                                        
                                except:
                                    continue
    
    print(f"Best SARIMAX parameters: {best_params}")
    print(f"Best AIC: {best_aic}")
    
    return best_params

def fit_sarimax_model(train_data, exog_train, order, seasonal_order):
    """
    Fit SARIMAX model with specified parameters
    """
    print(f"Fitting SARIMAX{order}x{seasonal_order} model...")
    
    model = SARIMAX(
        train_data,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=True)
    print("\nModel Summary:")
    print(results.summary().tables[1])
    
    return results

def evaluate_sarimax_model(model_results, test_data, exog_test, train_data):
    """
    Evaluate SARIMAX model performance
    """
    print("Evaluating SARIMAX model...")
    
    # Generate predictions
    predictions = model_results.forecast(steps=len(test_data), exog=exog_test)
    
    # Calculate metrics for all data
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    r2 = r2_score(test_data, predictions)
    
    # Calculate sMAPE for daytime only
    # Note: We'll need irradiance data for daytime filtering
    # For now, calculate sMAPE on all data
    smape = calculate_smape(test_data, predictions)
    
    print(f"\nSARIMAX Model Performance:")
    print(f"MAE: {mae:.2f} W")
    print(f"RMSE: {rmse:.2f} W")
    print(f"sMAPE: {smape:.2f}%")
    print(f"R²: {r2:.4f}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Plot training data
    plt.plot(train_data.index[-168:], train_data.iloc[-168:], 
             label='Training Data', alpha=0.7, color='blue')
    
    # Plot test data
    plt.plot(test_data.index, test_data, 
             label='Actual', linewidth=2, color='black')
    
    # Plot predictions
    plt.plot(test_data.index, predictions, 
             label='SARIMAX Predictions', linewidth=2, color='red', linestyle='--')
    
    plt.title('SARIMAX Model: Actual vs Predicted Solar Power')
    plt.xlabel('Date')
    plt.ylabel('Solar Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sarimax_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'smape': smape,
        'r2': r2,
        'predictions': predictions
    }

def main():
    """
    Main execution function for Phase 2
    """
    print("=" * 60)
    print("PHASE 2: SARIMAX BASELINE DEVELOPMENT")
    print("=" * 60)
    
    # Load processed data
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')
    
    # Convert timestamp
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Prepare data for SARIMAX
    train_hourly, test_hourly = prepare_sarimax_data(train_df, test_df)
    
    # Prepare exogenous variables
    exog_train = train_hourly[['irradiance', 'temperature', 'humidity']]
    exog_test = test_hourly[['irradiance', 'temperature', 'humidity']]
    target_train = train_hourly['solar_power_w']
    target_test = test_hourly['solar_power_w']
    
    # Check stationarity
    print("Checking stationarity of solar power series:")
    is_stationary = check_stationarity(target_train)
    
    # Analyze ACF and PACF
    print("\nAnalyzing ACF and PACF for parameter selection:")
    acf_pacf_fig = analyze_acf_pacf(target_train)
    
    # Find best parameters (simplified for demonstration)
    # Using common parameters for hourly solar data
    best_order = (1, 1, 1)  # (p, d, q)
    best_seasonal_order = (1, 1, 1, 24)  # (P, D, Q, s) where s=24 for daily seasonality
    
    print(f"\nUsing SARIMAX parameters: {best_order}x{best_seasonal_order}")
    
    # Fit SARIMAX model
    model_results = fit_sarimax_model(
        target_train, exog_train, best_order, best_seasonal_order
    )
    
    # Evaluate model
    evaluation_results = evaluate_sarimax_model(
        model_results, target_test, exog_test, target_train
    )
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': test_hourly.index,
        'actual': target_test,
        'sarimax_predicted': evaluation_results['predictions']
    })
    results_df.to_csv('results/sarimax_results.csv', index=False)
    
    print(f"\nPhase 2 completed!")
    print("Files saved:")
    print("- acf_pacf_analysis.png")
    print("- sarimax_predictions.png")
    print("- sarimax_results.csv")
    
    return evaluation_results

if __name__ == "__main__":
    results = main()
