import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Ensure the src/ directory is on sys.path so we can import utils when this
# file is executed as a script via `python src/models/xgboost_model.py`.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from utils import calculate_metrics, create_daytime_filter, save_results

def prepare_xgboost_features(train_df, test_df):
    """
    Prepare features for XGBoost model
    """
    print("Preparing XGBoost features...")
    print(f"  Input: {len(train_df)} train, {len(test_df)} test records")
    
    # Combine train and test for consistent feature engineering
    print("  Combining datasets...")
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Create additional features
    print("  Creating feature columns...")
    full_df['is_daytime'] = create_daytime_filter(full_df['irradiance'])
    
    # Weather interaction features
    full_df['irradiance_temp'] = full_df['irradiance'] * full_df['temperature']
    full_df['irradiance_humidity'] = full_df['irradiance'] * full_df['humidity']
    
    # Cyclical features for time
    full_df['hour_sin'] = np.sin(2 * np.pi * full_df['hour'] / 24)
    full_df['hour_cos'] = np.cos(2 * np.pi * full_df['hour'] / 24)
    full_df['month_sin'] = np.sin(2 * np.pi * full_df['month'] / 12)
    full_df['month_cos'] = np.cos(2 * np.pi * full_df['month'] / 12)
    
    # Split back to train and test
    train_size = len(train_df)
    train_features = full_df[:train_size].copy()
    test_features = full_df[train_size:].copy()
    
    # Define feature columns
    feature_columns = [
        'irradiance', 'temperature', 'humidity',
        'hour', 'day_of_week', 'month',
        'lag_24h', 'lag_48h',
        'irradiance_temp', 'irradiance_humidity',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    
    # Remove rows with NaN values (from lag features)
    print("  Removing NaN rows...")
    train_clean = train_features.dropna(subset=feature_columns + ['solar_power_w'])
    test_clean = test_features.dropna(subset=feature_columns + ['solar_power_w'])
    
    print(f"  Final: {len(train_clean)} train, {len(test_clean)} test records")
    
    return train_clean, test_clean, feature_columns

def optimize_xgboost_hyperparameters(X_train, y_train):
    """
    Optimize XGBoost hyperparameters using GridSearchCV
    """
    print("Optimizing XGBoost hyperparameters...")
    
    # Define parameter grid (simplified for demonstration)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 8],
        'learning_rate': [0.1, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Create XGBoost regressor
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_

def train_xgboost_model(train_df, feature_columns, optimize_params=True):
    """
    Train XGBoost model
    """
    print("Training XGBoost model...")
    
    # Prepare training data
    X_train = train_df[feature_columns]
    y_train = train_df['solar_power_w']
    
    if optimize_params:
        model = optimize_xgboost_hyperparameters(X_train, y_train)
    else:
        # Use a very fast configuration for full-pipeline runs
        print("Training fast XGBoost model (no grid search)...")
        print(f"  Training on {len(X_train)} samples with {len(feature_columns)} features...")
        model = xgb.XGBRegressor(
            n_estimators=30,  # Even fewer trees for speed
            max_depth=3,      # Shallower trees
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0,  # Suppress XGBoost output
        )
        print("  Fitting model...")
        model.fit(X_train, y_train)
        print("  Model fit complete!")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    return model, feature_importance

def evaluate_xgboost_model(model, test_df, feature_columns):
    """
    Evaluate XGBoost model performance
    """
    print("Evaluating XGBoost model...")
    
    # Prepare test data
    X_test = test_df[feature_columns]
    y_test = test_df['solar_power_w']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create daytime filter
    daytime_filter = create_daytime_filter(test_df['irradiance'])
    
    # Calculate metrics
    all_metrics = calculate_metrics(y_test, y_pred)
    daytime_metrics = calculate_metrics(y_test, y_pred, daytime_filter)
    
    print(f"\nXGBoost Model Performance (All Data):")
    print(f"MAE: {all_metrics['mae']:.2f} W")
    print(f"RMSE: {all_metrics['rmse']:.2f} W")
    print(f"sMAPE: {all_metrics['smape']:.2f}%")
    print(f"R²: {all_metrics['r2']:.4f}")
    
    print(f"\nXGBoost Model Performance (Daytime Only):")
    print(f"MAE: {daytime_metrics['mae']:.2f} W")
    print(f"RMSE: {daytime_metrics['rmse']:.2f} W")
    print(f"sMAPE: {daytime_metrics['smape']:.2f}%")
    print(f"R²: {daytime_metrics['r2']:.4f}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Plot actual vs predicted
    plt.subplot(2, 1, 1)
    plt.plot(test_df['timestamp'], y_test, label='Actual', alpha=0.7, color='black')
    plt.plot(test_df['timestamp'], y_pred, label='XGBoost Predicted', alpha=0.8, color='green')
    plt.title('XGBoost Model: Actual vs Predicted Solar Power')
    plt.ylabel('Solar Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 1, 2)
    residuals = y_test - y_pred
    plt.plot(test_df['timestamp'], residuals, alpha=0.7, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Prediction Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residual (W)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/xgboost_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show() to avoid blocking in pipeline
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': test_df['timestamp'],
        'actual': y_test,
        'xgboost_predicted': y_pred,
        'residuals': residuals
    })
    results_df.to_csv('results/xgboost_results.csv', index=False)
    
    return {
        'all_metrics': all_metrics,
        'daytime_metrics': daytime_metrics,
        'predictions': y_pred,
        'results_df': results_df
    }

def main():
    """
    Main execution function for Phase 3
    """
    print("=" * 60)
    print("PHASE 3: XGBOOST MODEL DEVELOPMENT")
    print("=" * 60)
    
    # Load processed data
    print("Loading data files...")
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')
    print(f"  Loaded: {len(train_df)} train, {len(test_df)} test")
    
    # Convert timestamp
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Prepare features
    train_clean, test_clean, feature_columns = prepare_xgboost_features(train_df, test_df)
    
    # Train model (use fast default params, skip GridSearchCV to avoid long runs)
    xgb_model, feature_importance = train_xgboost_model(
        train_clean,
        feature_columns,
        optimize_params=False,
    )
    
    # Evaluate model
    evaluation_results = evaluate_xgboost_model(xgb_model, test_clean, feature_columns)
    
    # Save feature importance
    feature_importance.to_csv('results/xgboost_feature_importance.csv', index=False)
    
    print(f"\nPhase 3 completed!")
    print("Files saved:")
    print("- results/xgboost_predictions.png")
    print("- results/xgboost_results.csv")
    print("- results/xgboost_feature_importance.csv")
    
    return evaluation_results, xgb_model

if __name__ == "__main__":
    results, model = main()
