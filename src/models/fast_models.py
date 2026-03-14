import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import time
warnings.filterwarnings('ignore')

def run_simple_models():
    """Simple fast models without Unicode issues"""
    print("SIMPLE FAST SOLAR FORECASTING")
    print("=" * 50)
    
    # Load data
    try:
        train_df = pd.read_csv('data/train_final.csv')
        test_df = pd.read_csv('data/test_final.csv')
        
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        
        print(f"Data loaded: {len(train_df)} train, {len(test_df)} test")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Prepare features
    feature_cols = ['irradiance', 'temperature', 'humidity', 'hour', 'day_of_week']
    X_train = train_df[feature_cols]
    y_train = train_df['solar_power_w']
    X_test = test_df[feature_cols]
    y_test = test_df['solar_power_w']
    
    results = {}
    times = {}
    
    # 1. Fast XGBoost
    print("Running XGBoost...")
    start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.2,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    times['XGBoost'] = time.time() - start
    print(f"XGBoost done in {times['XGBoost']:.1f}s")
    
    # 2. Simple Prophet
    print("Running Prophet...")
    start = time.time()
    train_prophet = train_df[['timestamp', 'solar_power_w']].rename(
        columns={'timestamp': 'ds', 'solar_power_w': 'y'}
    )
    
    prophet_model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.1
    )
    
    prophet_model.fit(train_prophet)
    future = pd.concat([
        train_prophet[['ds']], 
        test_df[['timestamp']].rename(columns={'timestamp': 'ds'})
    ])
    prophet_forecast = prophet_model.predict(future)
    prophet_pred = prophet_forecast['yhat'][len(train_df):].values
    times['Prophet'] = time.time() - start
    print(f"Prophet done in {times['Prophet']:.1f}s")
    
    # 3. Hybrid (Prophet + XGBoost)
    print("Running Hybrid...")
    start = time.time()
    
    # Train on residuals
    prophet_train_pred = prophet_forecast['yhat'][:len(train_df)].values
    residual_train = y_train - prophet_train_pred
    
    residual_model = xgb.XGBRegressor(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.3,
        random_state=42
    )
    residual_model.fit(X_train, residual_train)
    residual_pred = residual_model.predict(X_test)
    hybrid_pred = prophet_pred + residual_pred
    times['Hybrid'] = time.time() - start
    print(f"Hybrid done in {times['Hybrid']:.1f}s")
    
    # Calculate metrics
    for name, pred in [('XGBoost', xgb_pred), ('Prophet', prophet_pred), ('Hybrid', hybrid_pred)]:
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        smape = np.mean(2 * np.abs(pred - y_test) / (np.abs(pred) + np.abs(y_test))) * 100
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'sMAPE': smape,
            'predictions': pred
        }
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': test_df['timestamp'],
        'actual': y_test,
        'xgboost_predicted': xgb_pred,
        'prophet_predicted': prophet_pred,
        'hybrid_predicted': hybrid_pred
    })
    
    results_df.to_csv('results/simple_model_results.csv', index=False)
    
    # Create summary
    summary = []
    for name, metrics in results.items():
        summary.append({
            'Model': name,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            'sMAPE': metrics['sMAPE'],
            'Time_s': times.get(name, 0)
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('MAE')
    summary_df.to_csv('results/simple_summary.csv', index=False)
    
    # Find best model
    best_model = summary_df.iloc[0]['Model']
    best_mae = summary_df.iloc[0]['MAE']
    
    print("\nRESULTS SUMMARY:")
    print("=" * 40)
    for _, row in summary_df.iterrows():
        star = "*" if row['Model'] == best_model else " "
        print(f"{star} {row['Model']:<12} | MAE: {row['MAE']:>7.1f} | Time: {row['Time_s']:>5.1f}s")
    
    print("=" * 40)
    print(f"BEST MODEL: {best_model} (MAE: {best_mae:.2f}W)")
    
    # Quick plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['timestamp'][:168], y_test[:168], 'k-', label='Actual', linewidth=2)
    plt.plot(test_df['timestamp'][:168], xgb_pred[:168], 'b--', label='XGBoost', alpha=0.7)
    plt.plot(test_df['timestamp'][:168], prophet_pred[:168], 'g--', label='Prophet', alpha=0.7)
    plt.plot(test_df['timestamp'][:168], hybrid_pred[:168], 'r--', label='Hybrid', alpha=0.7)
    
    plt.title('Solar Forecasting Comparison (First Week)')
    plt.xlabel('Date')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/simple_comparison.png', dpi=150)
    plt.show()
    
    print("\nFiles created:")
    print("- results/simple_model_results.csv")
    print("- results/simple_summary.csv") 
    print("- results/simple_comparison.png")
    
    return summary_df

if __name__ == "__main__":
    results = run_simple_models()
