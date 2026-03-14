import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Ensure the src/ directory is on sys.path
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

def train_prophet_model(train_df):
    """
    Train standalone Prophet model
    """
    print("Training standalone Prophet model...")

    # Prepare data for Prophet
    prophet_train = prepare_prophet_data(train_df)
    print(f"  Prophet training rows: {len(prophet_train)}")

    # Create Prophet model
    prophet_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=5.0,
        holidays_prior_scale=5.0,
        mcmc_samples=0,
        interval_width=0.9,
        uncertainty_samples=0,  # deterministic, faster
    )

    print("  Fitting Prophet model...")
    prophet_model.fit(prophet_train)
    print("  Prophet fit complete!")

    return prophet_model

def generate_prophet_predictions(model, test_df):
    """
    Generate predictions using trained Prophet model
    """
    print("Generating Prophet predictions...")

    # Prepare test data
    prophet_test = prepare_prophet_data(test_df)

    # Make predictions
    forecast = model.predict(prophet_test)

    # Extract predictions
    predictions = forecast['yhat'].values

    return predictions, forecast

def evaluate_prophet_model():
    """
    Complete Prophet model training and evaluation
    """
    print("=" * 60)
    print("PHASE: PROPHET MODEL DEVELOPMENT")
    print("=" * 60)

    # Load data
    print("Loading data files...")
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')

    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    print(f"  Loaded: {len(train_df)} train, {len(test_df)} test")

    # Train Prophet model
    prophet_model = train_prophet_model(train_df)

    # Generate predictions
    prophet_pred, forecast = generate_prophet_predictions(prophet_model, test_df)

    # Evaluate performance
    print("Evaluating Prophet model...")

    # All data evaluation
    all_metrics = calculate_metrics(test_df['solar_power_w'].values, prophet_pred)

    print("\nProphet Model Performance (All Data):")
    print(f"MAE: {all_metrics['mae']:.2f} W")
    print(f"RMSE: {all_metrics['rmse']:.2f} W")
    print(f"sMAPE: {all_metrics['smape']:.2f}%")
    print(f"R²: {all_metrics['r2']:.4f}")

    # Daytime only evaluation
    daytime_filter = create_daytime_filter(test_df['irradiance'])
    if daytime_filter.sum() > 0:
        daytime_metrics = calculate_metrics(test_df['solar_power_w'].values, prophet_pred, daytime_filter)

        print("\nProphet Model Performance (Daytime Only):")
        print(f"MAE: {daytime_metrics['mae']:.2f} W")
        print(f"RMSE: {daytime_metrics['rmse']:.2f} W")
        print(f"sMAPE: {daytime_metrics['smape']:.2f}%")
        print(f"R²: {daytime_metrics['r2']:.4f}")

    # Create results dataframe
    results_df = pd.DataFrame({
        'timestamp': test_df['timestamp'],
        'actual': test_df['solar_power_w'],
        'prophet_predicted': prophet_pred
    })

    # Save results
    results_df.to_csv('results/prophet_results.csv', index=False)
    print("Results saved to results/prophet_results.csv")

    # Create visualization
    plt.figure(figsize=(15, 8))

    # Plot actual vs predicted
    plt.plot(test_df['timestamp'], test_df['solar_power_w'],
             label='Actual', color='black', linewidth=2, alpha=0.8)
    plt.plot(test_df['timestamp'], prophet_pred,
             label='Prophet Predicted', color='green', linewidth=2, linestyle='--', alpha=0.8)

    plt.title('Prophet Model: Actual vs Predicted Solar Generation', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Solar Power (W)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('results/prophet_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualization saved to results/prophet_predictions.png")

    print("Phase completed!")

    return results_df

if __name__ == "__main__":
    evaluate_prophet_model()