"""
30-Second Solar Forecasting - Complete in under 30 seconds
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import time

def run_30sec_forecasting():
    """Complete forecasting in under 30 seconds"""
    print("30-SECOND SOLAR FORECASTING CHALLENGE")
    print("=" * 50)
    print("Target: Complete analysis in < 30 seconds")
    print("=" * 50)
    
    start_time = time.time()
    
    # Step 1: Load data (5 seconds)
    print("Step 1/5: Loading data...")
    data_start = time.time()
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')
    data_time = time.time() - data_start
    print(f"   Data loaded in {data_time:.1f}s")
    
    # Step 2: Prepare features (2 seconds)
    print("Step 2/5: Preparing features...")
    prep_start = time.time()
    features = ['irradiance', 'temperature', 'humidity']
    X_train = train_df[features]
    y_train = train_df['solar_power_w']
    X_test = test_df[features]
    y_test = test_df['solar_power_w']
    prep_time = time.time() - prep_start
    print(f"   Features prepared in {prep_time:.1f}s")
    
    # Step 3: Train models (10 seconds)
    print("Step 3/5: Training models...")
    train_start = time.time()
    
    # Model 1: Linear Regression (fastest)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    # Model 2: Simple average (baseline)
    avg_pred = np.full(len(y_test), y_train.mean())
    
    # Model 3: Weighted irradiance (physics-based)
    weight = 0.8  # Simple scaling factor
    weighted_pred = X_test['irradiance'] * weight
    
    train_time = time.time() - train_start
    print(f"   Models trained in {train_time:.1f}s")
    
    # Step 4: Evaluate (3 seconds)
    print("Step 4/5: Evaluating models...")
    eval_start = time.time()
    
    models = [
        ('Linear Regression', lr_pred),
        ('Average Baseline', avg_pred),
        ('Weighted Irradiance', weighted_pred)
    ]
    
    results = []
    for name, pred in models:
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'predictions': pred[:100]  # Keep first 100 for plotting
        })
    
    eval_time = time.time() - eval_start
    print(f"   Evaluation completed in {eval_time:.1f}s")
    
    # Step 5: Save results (2 seconds)
    print("Step 5/5: Saving results...")
    save_start = time.time()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAE')
    results_df.to_csv('results/30sec_results.csv', index=False)
    
    # Create simple plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    
    # Plot actual
    plt.plot(y_test.values[:100], 'k-', label='Actual', linewidth=3, alpha=0.9)
    
    # Plot predictions
    colors = ['blue', 'red', 'green']
    for i, (_, row) in enumerate(results_df.iterrows()):
        plt.plot(row['predictions'], '--', color=colors[i], 
                label=row['Model'], alpha=0.7, linewidth=2)
    
    plt.title('30-Second Solar Forecasting Results', fontsize=14, fontweight='bold')
    plt.xlabel('Time (first 100 test points)')
    plt.ylabel('Solar Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/30sec_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    save_time = time.time() - save_start
    print(f"   Results saved in {save_time:.1f}s")
    
    # Final summary
    total_time = time.time() - start_time
    best_model = results_df.iloc[0]['Model']
    best_mae = results_df.iloc[0]['MAE']
    
    print(f"\n{'='*50}")
    print("30-SECOND CHALLENGE RESULTS")
    print('='*50)
    
    for _, row in results_df.iterrows():
        star = "🏆" if row['Model'] == best_model else "  "
        print(f"{star} {row['Model']:<20} | MAE: {row['MAE']:>8.1f}W | R²: {row['R2']:>6.3f}")
    
    print(f"\n⚡ TOTAL TIME: {total_time:.1f} seconds")
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"📊 BEST MAE: {best_mae:.2f}W")
    
    # Success check
    if total_time < 30:
        print("✅ SUCCESS: Completed in under 30 seconds!")
    else:
        print(f"⚠️  Took {total_time:.1f}s (exceeded 30s target)")
    
    print(f"\n📁 Files created:")
    print(f"   - results/30sec_results.csv")
    print(f"   - results/30sec_plot.png")
    
    return results_df

if __name__ == "__main__":
    run_30sec_forecasting()
