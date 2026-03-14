import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils import calculate_metrics, create_daytime_filter, save_results

def prepare_prophet_data(df):
    """
    Prepare data for Prophet model
    """
    prophet_df = df[['timestamp', 'solar_power_w']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def train_prophet_model(train_df):
    """
    Train Prophet model for trend and seasonality
    """
    print("Training Prophet model...")
    
    # Prepare data for Prophet
    prophet_train = prepare_prophet_data(train_df)
    
    # Create Prophet model with custom seasonality
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000
    )
    
    # Add custom seasonality for solar patterns
    prophet_model.add_seasonality(
        name='hourly_solar',
        period=1,
        fourier_order=8,
        prior_scale=10.0
    )
    
    # Fit the model
    prophet_model.fit(prophet_train)
    
    return prophet_model

def generate_prophet_predictions(prophet_model, train_df, test_df):
    """
    Generate Prophet predictions and calculate residuals
    """
    print("Generating Prophet predictions...")
    
    # Prepare future dataframe
    prophet_train = prepare_prophet_data(train_df)
    prophet_test = prepare_prophet_data(test_df)
    
    # Combine for prediction
    full_prophet_df = pd.concat([prophet_train, prophet_test], ignore_index=True)
    
    # Generate predictions
    prophet_forecast = prophet_model.predict(full_prophet_df)
    
    # Extract predictions for training and test periods
    train_size = len(prophet_train)
    train_predictions = prophet_forecast['yhat'][:train_size].values
    test_predictions = prophet_forecast['yhat'][train_size:].values
    
    # Calculate residuals for training data
    train_residuals = prophet_train['y'].values - train_predictions
    
    # Calculate residuals for test data
    test_residuals = prophet_test['y'].values - test_predictions
    
    print(f"Prophet training MAE: {mean_absolute_error(prophet_train['y'], train_predictions):.2f} W")
    print(f"Prophet test MAE: {mean_absolute_error(prophet_test['y'], test_predictions):.2f} W")
    
    return {
        'prophet_model': prophet_model,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'train_residuals': train_residuals,
        'test_residuals': test_residuals,
        'forecast': prophet_forecast
    }

def prepare_residual_features(df, residuals):
    """
    Prepare features for residual modeling
    """
    # Create residual dataframe
    residual_df = df.copy()
    residual_df['residual'] = residuals
    
    # Create additional features for residual modeling
    residual_df['irradiance_squared'] = residual_df['irradiance'] ** 2
    residual_df['temperature_squared'] = residual_df['temperature'] ** 2
    residual_df['irradiance_temp_interaction'] = residual_df['irradiance'] * residual_df['temperature']
    
    # Cyclical features
    residual_df['hour_sin'] = np.sin(2 * np.pi * residual_df['hour'] / 24)
    residual_df['hour_cos'] = np.cos(2 * np.pi * residual_df['hour'] / 24)
    
    return residual_df

def train_residual_xgboost(train_df, train_residuals, feature_columns):
    """
    Train XGBoost model on residuals
    """
    print("Training XGBoost model on residuals...")
    
    # Prepare residual training data
    residual_train = prepare_residual_features(train_df, train_residuals)
    
    # Remove rows with NaN values
    residual_train_clean = residual_train.dropna(subset=feature_columns + ['residual'])
    
    # Prepare features and target
    X_train = residual_train_clean[feature_columns]
    y_train = residual_train_clean['residual']
    
    # Train XGBoost for residuals
    residual_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    residual_model.fit(X_train, y_train)
    
    # Feature importance for residuals
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': residual_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top residual features:")
    print(feature_importance.head(5))
    
    return residual_model, feature_importance

def create_hybrid_predictions(prophet_results, residual_model, test_df, feature_columns):
    """
    Create hybrid predictions by combining Prophet and residual XGBoost
    """
    print("Creating hybrid predictions...")
    
    # Prepare test data for residual prediction
    residual_test = prepare_residual_features(test_df, prophet_results['test_residuals'])
    residual_test_clean = residual_test.dropna(subset=feature_columns)
    
    # Predict residuals
    X_test = residual_test_clean[feature_columns]
    predicted_residuals = residual_model.predict(X_test)
    
    # Combine Prophet predictions with residual corrections
    prophet_test_preds = prophet_results['test_predictions'][:len(predicted_residuals)]
    hybrid_predictions = prophet_test_preds + predicted_residuals
    
    # Get actual values
    actual_values = residual_test_clean['solar_power_w'].values
    
    return hybrid_predictions, actual_values, residual_test_clean

def evaluate_hybrid_model(hybrid_predictions, actual_values, test_df_clean):
    """
    Evaluate hybrid model performance
    """
    print("Evaluating hybrid model...")
    
    # Create daytime filter
    daytime_filter = create_daytime_filter(test_df_clean['irradiance'])
    
    # Calculate metrics
    all_metrics = calculate_metrics(actual_values, hybrid_predictions)
    daytime_metrics = calculate_metrics(actual_values, hybrid_predictions, daytime_filter)
    
    print(f"\nHybrid Model Performance (All Data):")
    print(f"MAE: {all_metrics['mae']:.2f} W")
    print(f"RMSE: {all_metrics['rmse']:.2f} W")
    print(f"sMAPE: {all_metrics['smape']:.2f}%")
    print(f"R²: {all_metrics['r2']:.4f}")
    
    print(f"\nHybrid Model Performance (Daytime Only):")
    print(f"MAE: {daytime_metrics['mae']:.2f} W")
    print(f"RMSE: {daytime_metrics['rmse']:.2f} W")
    print(f"sMAPE: {daytime_metrics['smape']:.2f}%")
    print(f"R²: {daytime_metrics['r2']:.4f}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Hybrid Predictions
    plt.subplot(3, 1, 1)
    plt.plot(test_df_clean['timestamp'], actual_values, label='Actual', alpha=0.7, color='black')
    plt.plot(test_df_clean['timestamp'], hybrid_predictions, label='Hybrid Predicted', alpha=0.8, color='purple')
    plt.title('Hybrid Model: Actual vs Predicted Solar Power')
    plt.ylabel('Solar Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prophet vs Hybrid comparison
    plt.subplot(3, 1, 2)
    prophet_test_preds = test_df_clean['solar_power_w'] - test_df_clean['residual']  # Approximate
    plt.plot(test_df_clean['timestamp'], actual_values, label='Actual', alpha=0.7, color='black')
    plt.plot(test_df_clean['timestamp'], prophet_test_preds, label='Prophet Only', alpha=0.6, color='blue')
    plt.plot(test_df_clean['timestamp'], hybrid_predictions, label='Hybrid', alpha=0.8, color='purple')
    plt.title('Prophet vs Hybrid Comparison')
    plt.ylabel('Solar Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    plt.subplot(3, 1, 3)
    residuals = actual_values - hybrid_predictions
    plt.plot(test_df_clean['timestamp'], residuals, alpha=0.7, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Hybrid Model Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residual (W)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/hybrid_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': test_df_clean['timestamp'],
        'actual': actual_values,
        'prophet_predicted': prophet_test_preds,
        'hybrid_predicted': hybrid_predictions,
        'residuals': residuals
    })
    results_df.to_csv('results/hybrid_results.csv', index=False)
    
    return {
        'all_metrics': all_metrics,
        'daytime_metrics': daytime_metrics,
        'predictions': hybrid_predictions,
        'results_df': results_df
    }

def main():
    """
    Main execution function for Phase 4
    """
    print("=" * 60)
    print("PHASE 4: PROPHET + XGBOOST HYBRID MODEL")
    print("=" * 60)
    
    # Load processed data
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')
    
    # Convert timestamp
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Train Prophet model
    prophet_model = train_prophet_model(train_df)
    
    # Generate Prophet predictions and residuals
    prophet_results = generate_prophet_predictions(prophet_model, train_df, test_df)
    
    # Define features for residual modeling
    residual_features = [
        'irradiance', 'temperature', 'humidity',
        'hour', 'day_of_week', 'month',
        'lag_24h', 'lag_48h'
    ]
    
    # Train XGBoost on residuals
    residual_model, residual_importance = train_residual_xgboost(
        train_df, prophet_results['train_residuals'], residual_features
    )
    
    # Create hybrid predictions
    hybrid_preds, actual_vals, test_clean = create_hybrid_predictions(
        prophet_results, residual_model, test_df, residual_features
    )
    
    # Evaluate hybrid model
    evaluation_results = evaluate_hybrid_model(hybrid_preds, actual_vals, test_clean)
    
    # Save residual feature importance
    residual_importance.to_csv('results/hybrid_residual_importance.csv', index=False)
    
    print(f"\nPhase 4 completed!")
    print("Files saved:")
    print("- results/hybrid_predictions.png")
    print("- results/hybrid_results.csv")
    print("- results/hybrid_residual_importance.csv")
    
    return evaluation_results, prophet_model, residual_model

if __name__ == "__main__":
    results, prophet_model, residual_model = main()
