"""
Ultra-Fast Solar Forecasting - Under 30 seconds
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

def ultra_fast_models():
    """Ultra-fast model evaluation - under 30 seconds"""
    print("ULTRA-FAST SOLAR FORECASTING")
    print("=" * 40)
    print("Target: Complete in < 30 seconds")
    print("=" * 40)
    
    start_total = time.time()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')
    
    # Minimal features for speed
    features = ['irradiance', 'temperature', 'humidity']
    X_train = train_df[features]
    y_train = train_df['solar_power_w']
    X_test = test_df[features]
    y_test = test_df['solar_power_w']
    
    print(f"Data loaded: {len(train_df)} train, {len(test_df)} test")
    
    results = {}
    
    # 1. Super-fast XGBoost
    print("Running XGBoost...")
    start = time.time()
    model = xgb.XGBRegressor(
        n_estimators=10,  # Minimal for speed
        max_depth=2,      # Very shallow
        learning_rate=0.5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    xgb_pred = model.predict(X_test)
    xgb_time = time.time() - start
    print(f"XGBoost done in {xgb_time:.1f}s")
    
    # 2. Simple linear regression (baseline)
    print("Running Linear Regression...")
    start = time.time()
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_time = time.time() - start
    print(f"Linear Regression done in {lr_time:.1f}s")
    
    # 3. Simple average (baseline)
    print("Running Average Baseline...")
    avg_pred = np.full(len(y_test), y_train.mean())
    avg_time = 0.001
    print(f"Average baseline done in {avg_time:.1f}s")
    
    # Calculate metrics
    models = [
        ('XGBoost', xgb_pred, xgb_time),
        ('Linear Regression', lr_pred, lr_time),
        ('Average Baseline', avg_pred, avg_time)
    ]
    
    print("Calculating metrics...")
    for name, pred, train_time in models:
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        smape = np.mean(2 * np.abs(pred - y_test) / (np.abs(pred) + np.abs(y_test))) * 100
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'sMAPE': smape,
            'Time': train_time,
            'predictions': pred
        }
    
    # Save results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'MAE': [r['MAE'] for r in results.values()],
        'RMSE': [r['RMSE'] for r in results.values()],
        'R2': [r['R2'] for r in results.values()],
        'sMAPE': [r['sMAPE'] for r in results.values()],
        'Time_s': [r['Time'] for r in results.values()]
    })
    
    results_df = results_df.sort_values('MAE')
    results_df.to_csv('results/ultra_fast_results.csv', index=False)
    
    # Find best
    best_model = results_df.iloc[0]['Model']
    best_mae = results_df.iloc[0]['MAE']
    total_time = time.time() - start_total
    
    print(f"\n{'='*40}")
    print("ULTRA-FAST RESULTS")
    print('='*40)
    for _, row in results_df.iterrows():
        star = "🏆" if row['Model'] == best_model else "  "
        print(f"{star} {row['Model']:<20} | MAE: {row['MAE']:>7.1f} | Time: {row['Time_s']:>5.2f}s")
    
    print(f"\n⚡ TOTAL TIME: {total_time:.1f}s")
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"📊 BEST MAE: {best_mae:.2f}W")
    
    # Quick plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[:100], 'k-', label='Actual', linewidth=2)
    plt.plot(xgb_pred[:100], 'b--', label='XGBoost', alpha=0.8)
    plt.title('Ultra-Fast Solar Forecasting (First 100 points)')
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/ultra_fast_plot.png', dpi=150)
    plt.show()
    
    print(f"\n✅ Files created:")
    print(f"   - results/ultra_fast_results.csv")
    print(f"   - results/ultra_fast_plot.png")
    
    return results_df

if __name__ == "__main__":
    ultra_fast_models()
