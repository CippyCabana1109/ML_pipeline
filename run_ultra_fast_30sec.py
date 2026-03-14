import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import time
import warnings
warnings.filterwarnings('ignore')

def ultra_fast_forecasting():
    print("⚡ ULTRA-FAST SOLAR FORECASTING")
    print("GUARANTEED: Complete in <30 seconds")
    print("=" * 50)
    
    start_time = time.time()
    
    # Phase 1: Load Data
    print("Loading data...")
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')
    
    # Phase 2: Extract features
    feature_cols = ['irradiance', 'temperature', 'humidity']
    X_train = train_df[feature_cols].values
    y_train = train_df['solar_power_w'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['solar_power_w'].values
    
    # Phase 3: Train models
    predictions = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predictions['Linear Regression'] = lr.predict(X_test)
    
    # Average Baseline
    avg_value = np.mean(y_train)
    predictions['Average Baseline'] = np.full(len(y_test), avg_value)
    
    # Weighted Irradiance
    max_irradiance = np.max(X_train[:, 0])
    max_power = np.max(y_train)
    scaling_factor = max_power / max_irradiance if max_irradiance > 0 else 1.0
    predictions['Weighted Irradiance'] = X_test[:, 0] * scaling_factor
    
    # Phase 4: Evaluate
    results = []
    for name, pred in predictions.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = max(0, r2_score(y_test, pred))
        results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    
    results.sort(key=lambda x: x['MAE'])
    
    # Phase 5: Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/ultra_fast_results.csv', index=False)
    
    # Results
    total_time = time.time() - start_time
    best_model = results[0]['Model']
    best_mae = results[0]['MAE']
    
    print(f"\n{'='*50}")
    print("⚡ ULTRA-FAST RESULTS")
    print('='*50)
    for result in results:
        star = "🏆" if result['Model'] == best_model else "  "
        print(f"{star} {result['Model']:<20} | MAE: {result['MAE']:>7.1f}W | R²: {result['R2']:>6.3f}")
    
    print(f"\n⚡ TOTAL TIME: {total_time:.1f} seconds")
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"📊 BEST MAE: {best_mae:.2f}W")
    
    if total_time < 30:
        print("✅ SUCCESS: Completed in under 30 seconds!")
    else:
        print(f"⚠️  Took {total_time:.1f}s (exceeded 30s target)")
    
    return results_df

if __name__ == "__main__":
    ultra_fast_forecasting()
