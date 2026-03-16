import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Ensure the src/ directory is on sys.path so we can import utils when this
# file is executed as a script via `python src/models/hybrid_model.py`.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from utils import calculate_metrics, create_daytime_filter, save_results

def prepare_prophet_data(df):
    """
    Prepare data for Prophet model
    """
    prophet_df = df[['timestamp', 'solar_power_w']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def train_xgboost_base_model(train_df, feature_columns):
    """
    Train XGBoost model as primary base model
    """
    print("Training XGBoost base model...")
    
    # Prepare training data
    train_clean = train_df.dropna(subset=feature_columns + ['solar_power_w'])
    X_train = train_clean[feature_columns]
    y_train = train_clean['solar_power_w']
    
    print(f"  XGBoost base training rows: {len(X_train)}")
    
    # Train XGBoost model
    xgboost_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    
    print("  Fitting XGBoost base model...")
    xgboost_model.fit(X_train, y_train)
    print("  XGBoost base fit complete!")
    
    # Calculate training performance
    y_pred_train = xgboost_model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    print(f"  XGBoost base training MAE: {train_mae:.2f} W")
    
    return xgboost_model

def train_prophet_on_residuals(train_df, xgboost_predictions, feature_columns):
    """
    Train Prophet model on XGBoost residuals (REVERSE STACKING)
    
    Key insight: XGBoost is strong, so its residuals are small and clean.
    Prophet can learn these patterns more effectively than learning large Prophet errors.
    """
    print("Training Prophet model on XGBoost residuals (REVERSE STACKING)...")
    
    # Calculate XGBoost residuals
    xgboost_residuals = train_df['solar_power_w'].values - xgboost_predictions
    
    # Create dataframe for Prophet with residuals as target
    prophet_train_data = train_df[['timestamp']].copy()
    prophet_train_data['ds'] = pd.to_datetime(prophet_train_data['timestamp'])
    prophet_train_data['y'] = xgboost_residuals
    
    print(f"  Prophet training on {len(prophet_train_data)} residual samples")
    print(f"    Mean residual: {prophet_train_data['y'].mean():.2f} W (vs Prophet's original 7435W)")
    print(f"    Std residual: {prophet_train_data['y'].std():.2f} W")
    
    # Create Prophet model configured for small residuals
    prophet_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.01,  # Lower: focus on global trend
        seasonality_prior_scale=1.0,   # Lower: smaller expected seasonality
        seasonality_mode='additive',
        interval_width=0.9,
        uncertainty_samples=0,  # deterministic
    )
    
    # Add light hourly seasonality for residual patterns
    prophet_model.add_seasonality(
        name='hourly_residual_pattern',
        period=1,
        fourier_order=2,  # Simpler than before
        prior_scale=1.0,
    )
    
    print("  Fitting Prophet on residuals...")
    prophet_model.fit(prophet_train_data[['ds', 'y']])
    print("  Prophet residual fit complete!")
    
    return prophet_model

def train_prophet_model(train_df):
    """
    DEPRECATED: Old Prophet training kept for compatibility
    """
    pass

def generate_predictions(xgboost_model, prophet_model, train_df, test_df, feature_columns):
    """
    Generate predictions using reverse stacking (IMPROVED HYBRID)
    
    Step 1: XGBoost makes primary prediction
    Step 2: Prophet corrects XGBoost residuals
    Step 3: Final = XGBoost + Prophet_residual_correction
    """
    print("Generating reverse-stacked hybrid predictions...")
    
    # STEP 1: Get XGBoost predictions
    print("\n  Step 1: XGBoost primary predictions")
    train_clean = train_df.dropna(subset=feature_columns + ['solar_power_w'])
    X_train = train_clean[feature_columns]
    y_train = train_clean['solar_power_w']
    
    train_xgb_preds = xgboost_model.predict(X_train)
    test_clean = test_df.dropna(subset=feature_columns + ['solar_power_w'])
    X_test = test_clean[feature_columns]
    y_test = test_clean['solar_power_w']
    
    test_xgb_preds = xgboost_model.predict(X_test)
    
    train_xgb_mae = mean_absolute_error(y_train, train_xgb_preds)
    test_xgb_mae = mean_absolute_error(y_test, test_xgb_preds)
    print(f"    XGBoost train MAE: {train_xgb_mae:.2f} W")
    print(f"    XGBoost test MAE: {test_xgb_mae:.2f} W")
    
    # STEP 2: Get Prophet predictions on XGBoost residuals
    print("\n  Step 2: Prophet residual predictions")
    
    # Prepare test data for Prophet (with same time range as XGBoost predictions)
    prophet_test_data = test_clean[['timestamp']].copy()
    prophet_test_data['ds'] = pd.to_datetime(prophet_test_data['timestamp'])
    
    # Predict residuals using Prophet
    prophet_forecast = prophet_model.predict(prophet_test_data[['ds']])
    prophet_residual_preds = prophet_forecast['yhat'].values
    
    print(f"    Prophet predicted XGBoost residuals (MAE vs actual residuals):")
    actual_residuals = y_test - test_xgb_preds
    residual_mae = mean_absolute_error(actual_residuals, prophet_residual_preds)
    print(f"      {residual_mae:.2f} W")
    
    # STEP 3: Combine predictions
    print("\n  Step 3: Combine into final hybrid prediction")
    hybrid_predictions = test_xgb_preds + prophet_residual_preds
    
    print(f"    Hybrid predictions ready for {len(hybrid_predictions)} test samples")
    
    return {
        'hybrid_predictions': hybrid_predictions,
        'xgboost_predictions': test_xgb_preds,
        'prophet_residual_predictions': prophet_residual_preds,
        'actual_values': y_test,
        'test_data': test_clean,
        'actual_residuals': actual_residuals,
    }

def prepare_residual_features(df, residuals):
    """
    DEPRECATED: No longer needed in reverse stacking approach
    """
    pass

def train_residual_xgboost(train_df, train_residuals, feature_columns):
    """
    DEPRECATED: Replaced by train_prophet_on_residuals
    """
    pass

def create_hybrid_predictions(prophet_results, residual_model, test_df, feature_columns):
    """
    DEPRECATED: Replaced by generate_predictions function
    """
    pass

def evaluate_hybrid_model(prediction_results):
    """
    Evaluate improved hybrid model performance (reverse stacking)
    
    Compares:
    - XGBoost alone
    - Hybrid (XGBoost + Prophet residual correction)
    """
    print("Evaluating improved hybrid model...")
    
    hybrid_predictions = prediction_results['hybrid_predictions']
    xgboost_predictions = prediction_results['xgboost_predictions']
    actual_values = prediction_results['actual_values']
    test_df_clean = prediction_results['test_data']
    
    # Create daytime filter
    daytime_filter = create_daytime_filter(test_df_clean['irradiance'])
    
    # Calculate metrics for XGBoost alone
    xgb_all_metrics = calculate_metrics(actual_values, xgboost_predictions)
    xgb_daytime_metrics = calculate_metrics(actual_values, xgboost_predictions, daytime_filter)
    
    # Calculate metrics for Hybrid
    hybrid_all_metrics = calculate_metrics(actual_values, hybrid_predictions)
    hybrid_daytime_metrics = calculate_metrics(actual_values, hybrid_predictions, daytime_filter)
    
    print(f"\n{'='*60}")
    print(f"XGBOOST ALONE (Baseline)")
    print(f"{'='*60}")
    print(f"All Data:      MAE={xgb_all_metrics['mae']:.2f}W  RMSE={xgb_all_metrics['rmse']:.2f}W  sMAPE={xgb_all_metrics['smape']:.2f}%  R²={xgb_all_metrics['r2']:.4f}")
    print(f"Daytime Only:  MAE={xgb_daytime_metrics['mae']:.2f}W  RMSE={xgb_daytime_metrics['rmse']:.2f}W  sMAPE={xgb_daytime_metrics['smape']:.2f}%  R²={xgb_daytime_metrics['r2']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"IMPROVED HYBRID (XGBoost + Prophet Residual Correction)")
    print(f"{'='*60}")
    print(f"All Data:      MAE={hybrid_all_metrics['mae']:.2f}W  RMSE={hybrid_all_metrics['rmse']:.2f}W  sMAPE={hybrid_all_metrics['smape']:.2f}%  R²={hybrid_all_metrics['r2']:.4f}")
    print(f"Daytime Only:  MAE={hybrid_daytime_metrics['mae']:.2f}W  RMSE={hybrid_daytime_metrics['rmse']:.2f}W  sMAPE={hybrid_daytime_metrics['smape']:.2f}%  R²={hybrid_daytime_metrics['r2']:.4f}")
    
    # Compare improvements
    print(f"\n{'='*60}")
    print(f"IMPROVEMENT: Hybrid vs XGBoost")
    print(f"{'='*60}")
    mae_improvement = ((xgb_all_metrics['mae'] - hybrid_all_metrics['mae']) / xgb_all_metrics['mae']) * 100
    rmse_improvement = ((xgb_all_metrics['rmse'] - hybrid_all_metrics['rmse']) / xgb_all_metrics['rmse']) * 100
    smape_improvement = ((xgb_all_metrics['smape'] - hybrid_all_metrics['smape']) / xgb_all_metrics['smape']) * 100
    r2_improvement = ((hybrid_all_metrics['r2'] - xgb_all_metrics['r2']) / abs(xgb_all_metrics['r2'])) * 100
    
    mae_status = '[BETTER]' if mae_improvement > 0 else '[WORSE]'
    rmse_status = '[BETTER]' if rmse_improvement > 0 else '[WORSE]'
    smape_status = '[BETTER]' if smape_improvement > 0 else '[WORSE]'
    r2_status = '[BETTER]' if r2_improvement > 0 else '[WORSE]'
    
    print(f"MAE improvement:   {mae_improvement:+.2f}% {mae_status}")
    print(f"RMSE improvement:  {rmse_improvement:+.2f}% {rmse_status}")
    print(f"sMAPE improvement: {smape_improvement:+.2f}% {smape_status}")
    print(f"R² improvement:    {r2_improvement:+.2f}% {r2_status}")
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Actual vs Predictions for all three models
    plt.subplot(4, 1, 1)
    plt.plot(test_df_clean['timestamp'], actual_values, label='Actual', alpha=0.8, color='black', linewidth=2)
    plt.plot(test_df_clean['timestamp'], xgboost_predictions, label=f'XGBoost (MAE={xgb_all_metrics["mae"]:.0f}W)', alpha=0.7, color='blue')
    plt.plot(test_df_clean['timestamp'], hybrid_predictions, label=f'Hybrid (MAE={hybrid_all_metrics["mae"]:.0f}W)', alpha=0.7, color='green', linestyle='--')
    plt.title('Improved Hybrid: XGBoost vs Hybrid (XGBoost + Prophet Residual Correction)', fontsize=12, fontweight='bold')
    plt.ylabel('Solar Power (W)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals comparison
    plt.subplot(4, 1, 2)
    xgb_residuals = actual_values - xgboost_predictions
    hybrid_residuals = actual_values - hybrid_predictions
    plt.plot(test_df_clean['timestamp'], xgb_residuals, label='XGBoost Residuals', alpha=0.6, color='blue')
    plt.plot(test_df_clean['timestamp'], hybrid_residuals, label='Hybrid Residuals', alpha=0.6, color='green')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('Prediction Errors Comparison')
    plt.ylabel('Residual (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    plt.subplot(4, 1, 3)
    plt.hist(np.abs(xgb_residuals), bins=50, alpha=0.6, label=f'XGBoost |Errors| (MAE={xgb_all_metrics["mae"]:.0f})', color='blue')
    plt.hist(np.abs(hybrid_residuals), bins=50, alpha=0.6, label=f'Hybrid |Errors| (MAE={hybrid_all_metrics["mae"]:.0f})', color='green')
    plt.xlabel('Absolute Error (W)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Prophet's residual correction contribution
    plt.subplot(4, 1, 4)
    prophet_correction = prediction_results['prophet_residual_predictions']
    plt.plot(test_df_clean['timestamp'], prophet_correction, label='Prophet Residual Correction', alpha=0.7, color='purple')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('Prophet Residual Correction Values')
    plt.xlabel('Date')
    plt.ylabel('Correction (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/improved_hybrid_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': test_df_clean['timestamp'],
        'actual': actual_values,
        'xgboost': xgboost_predictions,
        'prophet_residual_correction': prediction_results['prophet_residual_predictions'],
        'hybrid': hybrid_predictions,
        'xgboost_residual': xgb_residuals,
        'hybrid_residual': hybrid_residuals,
    })
    results_df.to_csv('results/improved_hybrid_results.csv', index=False)
    
    return {
        'xgb_metrics': xgb_all_metrics,
        'hybrid_metrics': hybrid_all_metrics,
        'xgb_daytime': xgb_daytime_metrics,
        'hybrid_daytime': hybrid_daytime_metrics,
        'results_df': results_df
    }

def main():
    """
    Main execution function for Phase 4 - IMPROVED HYBRID MODEL
    
    NEW APPROACH (Reverse Stacking):
    1. Train XGBoost as primary model (strong baseline)
    2. Train Prophet on XGBoost residuals (clean, small residuals)
    3. Combine: Final prediction = XGBoost + Prophet_residual_correction
    
    This works better because:
    - XGBoost's residuals are small (~750W MAE), giving Prophet clean data
    - Prophet can learn true residual patterns without noise
    - Both algorithms work to their strengths
    """
    print("=" * 60)
    print("PHASE 4: IMPROVED HYBRID MODEL (Reverse Stacking)")
    print("=" * 60)
    
    # Load processed data
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')
    
    # Convert timestamp
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Define features for XGBoost
    xgboost_features = [
        'irradiance', 'temperature', 'humidity',
        'hour', 'day_of_week', 'month',
        'lag_24h', 'lag_48h'
    ]
    
    # STEP 1: Train XGBoost as primary model
    print("\n" + "="*60)
    print("STEP 1: Train XGBoost Primary Model")
    print("="*60)
    xgboost_model = train_xgboost_base_model(train_df, xgboost_features)
    
    # STEP 2: Train Prophet on XGBoost residuals
    print("\n" + "="*60)
    print("STEP 2: Train Prophet on XGBoost Residuals")
    print("="*60)
    
    # Get XGBoost training predictions to calculate residuals
    train_clean = train_df.dropna(subset=xgboost_features + ['solar_power_w'])
    X_train = train_clean[xgboost_features]
    xgb_train_preds = xgboost_model.predict(X_train)
    
    # Train Prophet on these residuals
    prophet_model = train_prophet_on_residuals(train_clean, xgb_train_preds, xgboost_features)
    
    # STEP 3: Generate hybrid predictions
    print("\n" + "="*60)
    print("STEP 3: Generate Hybrid Predictions")
    print("="*60)
    prediction_results = generate_predictions(xgboost_model, prophet_model, train_df, test_df, xgboost_features)
    
    # STEP 4: Evaluate and compare
    print("\n" + "="*60)
    print("STEP 4: Evaluate Models")
    print("="*60)
    evaluation_results = evaluate_hybrid_model(prediction_results)
    
    print(f"\nPhase 4 completed!")
    print("Files saved:")
    print("- results/improved_hybrid_comparison.png")
    print("- results/improved_hybrid_results.csv")
    
    return evaluation_results, xgboost_model, prophet_model

if __name__ == "__main__":
    results, xgb_model, prophet_model = main()
