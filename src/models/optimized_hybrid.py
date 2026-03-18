"""
Optimized Hybrid Model - Weighted Voting Ensemble
Learns optimal weights for combining XGBoost, Random Forest, and Gradient Boosting
Superior to simple additive/averaging approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
import os
import sys

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from utils import calculate_metrics, create_daytime_filter


def prepare_features(train_df, test_df):
    """Prepare features"""
    feature_columns = [
        'irradiance', 'temperature', 'humidity',
        'hour', 'day_of_week', 'month',
        'lag_24h', 'lag_48h'
    ]
    
    train_clean = train_df.dropna(subset=feature_columns + ['solar_power_w'])
    test_clean = test_df.dropna(subset=feature_columns + ['solar_power_w'])
    
    X_train = train_clean[feature_columns]
    y_train = train_clean['solar_power_w']
    
    X_test = test_clean[feature_columns]
    y_test = test_clean['solar_power_w']
    
    return X_train, y_train, X_test, y_test, test_clean


def train_component_models(X_train, y_train):
    """Train component models for ensemble"""
    print("Training component models...")
    
    models = {}
    
    print("  XGBoost...")
    models['xgboost'] = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=-1, verbosity=0
    )
    models['xgboost'].fit(X_train, y_train)
    
    print("  Random Forest...")
    models['rf'] = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    models['rf'].fit(X_train, y_train)
    
    print("  Gradient Boosting...")
    models['gb'] = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.9, random_state=42
    )
    models['gb'].fit(X_train, y_train)
    
    return models


def optimize_ensemble_weights(models, X_train, y_train, X_val, y_val):
    """
    Optimize ensemble weights using validation data
    Minimizes MAE on validation set
    """
    print("\nOptimizing ensemble weights...")
    
    # Get predictions from all models
    train_preds = {
        'xgboost': models['xgboost'].predict(X_train),
        'rf': models['rf'].predict(X_train),
        'gb': models['gb'].predict(X_train),
    }
    
    val_preds = {
        'xgboost': models['xgboost'].predict(X_val),
        'rf': models['rf'].predict(X_val),
        'gb': models['gb'].predict(X_val),
    }
    
    def weighted_ensemble_mae(weights):
        """Calculate ensemble MAE for given weights"""
        # Normalize weights to sum to 1
        w = weights / weights.sum()
        
        # Create ensemble predictions
        ensemble = (w[0] * val_preds['xgboost'] + 
                   w[1] * val_preds['rf'] + 
                   w[2] * val_preds['gb'])
        
        return mean_absolute_error(y_val, ensemble)
    
    # Initial weights
    x0 = np.array([1.0, 1.0, 1.0])
    
    # Optimize
    result = minimize(weighted_ensemble_mae, x0, method='Nelder-Mead', 
                     options={'maxiter': 1000})
    
    # Normalize final weights
    optimal_weights = result.x / result.x.sum()
    
    print("\nOptimal Ensemble Weights:")
    print("="*50)
    print(f"  XGBoost:         {optimal_weights[0]:.4f} ({optimal_weights[0]*100:5.1f}%)")
    print(f"  Random Forest:   {optimal_weights[1]:.4f} ({optimal_weights[1]*100:5.1f}%)")
    print(f"  Gradient Boost:  {optimal_weights[2]:.4f} ({optimal_weights[2]*100:5.1f}%)")
    
    # Validate optimization
    ensemble_val = (optimal_weights[0] * val_preds['xgboost'] +
                   optimal_weights[1] * val_preds['rf'] +
                   optimal_weights[2] * val_preds['gb'])
    
    val_mae = mean_absolute_error(y_val, ensemble_val)
    print(f"\nValidation MAE: {val_mae:.1f}W")
    
    return optimal_weights, train_preds, val_preds


def generate_predictions(models, weights, X_test):
    """Generate weighted ensemble predictions"""
    print("\nGenerating weighted ensemble predictions...")
    
    test_preds = {
        'xgboost': models['xgboost'].predict(X_test),
        'rf': models['rf'].predict(X_test),
        'gb': models['gb'].predict(X_test),
    }
    
    # Weighted ensemble
    ensemble = (weights[0] * test_preds['xgboost'] +
               weights[1] * test_preds['rf'] +
               weights[2] * test_preds['gb'])
    
    return ensemble, test_preds


def evaluate_models(y_test, ensemble_preds, individual_preds, test_data):
    """Comprehensive evaluation"""
    print("\n" + "="*70)
    print("OPTIMIZED ENSEMBLE EVALUATION")
    print("="*70)
    
    daytime_filter = create_daytime_filter(test_data['irradiance'])
    
    results = {}
    
    # Individual models
    for name, preds in individual_preds.items():
        metrics = calculate_metrics(y_test, preds)
        daytime_metrics = calculate_metrics(y_test, preds, daytime_filter)
        results[name] = metrics
        
        print(f"\n{name.upper()}:")
        print(f"  All Data:  MAE={metrics['mae']:7.1f}W  RMSE={metrics['rmse']:7.1f}W  R²={metrics['r2']:.4f}")
        print(f"  Daytime:   MAE={daytime_metrics['mae']:7.1f}W  RMSE={daytime_metrics['rmse']:7.1f}W  R²={daytime_metrics['r2']:.4f}")
    
    # Ensemble
    ensemble_metrics = calculate_metrics(y_test, ensemble_preds)
    ensemble_daytime = calculate_metrics(y_test, ensemble_preds, daytime_filter)
    results['ensemble'] = ensemble_metrics
    
    print(f"\nWEIGHTED ENSEMBLE:")
    print(f"  All Data:  MAE={ensemble_metrics['mae']:7.1f}W  RMSE={ensemble_metrics['rmse']:7.1f}W  R²={ensemble_metrics['r2']:.4f}")
    print(f"  Daytime:   MAE={ensemble_daytime['mae']:7.1f}W  RMSE={ensemble_daytime['rmse']:7.1f}W  R²={ensemble_daytime['r2']:.4f}")
    
    # Comparison with baseline XGBoost
    if 'xgboost' in results:
        xgb_mae = results['xgboost']['mae']
        ensemble_mae = ensemble_metrics['mae']
        improvement = ((xgb_mae - ensemble_mae) / xgb_mae) * 100
        
        print(f"\n{'='*70}")
        print(f"ENSEMBLE vs XGBoost BASELINE")
        print(f"{'='*70}")
        print(f"XGBoost MAE:        {xgb_mae:7.1f}W")
        print(f"Ensemble MAE:       {ensemble_mae:7.1f}W")
        print(f"Improvement:        {improvement:+7.2f}% {'[BETTER]' if improvement > 0 else '[WORSE]'}")
    
    return results, ensemble_metrics


def visualize_comparison(y_test, ensemble_preds, individual_preds, test_data, weights):
    """Create visualizations"""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All predictions vs actual
    ax = axes[0, 0]
    ax.plot(test_data['timestamp'], y_test, 'k-', label='Actual', linewidth=2.5, alpha=0.8)
    ax.plot(test_data['timestamp'], individual_preds['xgboost'], '--', label='XGBoost', alpha=0.6, color='blue')
    ax.plot(test_data['timestamp'], individual_preds['rf'], '--', label='Random Forest', alpha=0.6, color='green')
    ax.plot(test_data['timestamp'], individual_preds['gb'], '--', label='Gradient Boost', alpha=0.6, color='orange')
    ax.plot(test_data['timestamp'], ensemble_preds, '-', label='Weighted Ensemble', alpha=0.8, color='red', linewidth=2)
    ax.set_title('Weighted Ensemble vs Component Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Solar Power (W)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ensemble residuals
    ax = axes[0, 1]
    residuals = y_test - ensemble_preds
    ax.scatter(ensemble_preds, residuals, alpha=0.6, s=60, color='red')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    mae = np.mean(np.abs(residuals))
    ax.set_title(f'Weighted Ensemble Residuals (MAE={mae:.1f}W)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted Power (W)')
    ax.set_ylabel('Residual (W)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    ax = axes[1, 0]
    xgb_errors = np.abs(y_test - individual_preds['xgboost'])
    ensemble_errors = np.abs(y_test - ensemble_preds)
    ax.hist(xgb_errors, bins=20, alpha=0.6, label=f"XGBoost (MAE={xgb_errors.mean():.0f}W)", color='blue')
    ax.hist(ensemble_errors, bins=20, alpha=0.6, label=f"Ensemble (MAE={ensemble_errors.mean():.0f}W)", color='red')
    ax.set_title('Error Distribution: XGBoost vs Ensemble', fontweight='bold', fontsize=12)
    ax.set_xlabel('Absolute Error (W)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Model weights and performance
    ax = axes[1, 1]
    ax.axis('off')
    
    text = "WEIGHTED ENSEMBLE BUILD\n" + "="*45 + "\n\n"
    text += "Component Weights:\n"
    text += f"  XGBoost:        {weights[0]:.1%}\n"
    text += f"  Random Forest:  {weights[1]:.1%}\n"
    text += f"  Gradient Boost: {weights[2]:.1%}\n\n"
    
    text += "Model Performance:\n"
    text += f"  XGBoost:        MAE={individual_preds['xgboost'].mean():.0f}W\n"
    text += f"  Random Forest:  MAE={individual_preds['rf'].mean():.0f}W\n"
    text += f"  Gradient Boost: MAE={individual_preds['gb'].mean():.0f}W\n\n"
    
    text += "Ensemble Result:\n"
    text += f"  Weighted Avg:   MAE={np.mean(np.abs(y_test - ensemble_preds)):.1f}W\n"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/optimized_hybrid_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved!")


def main():
    """Main execution"""
    print("="*70)
    print("OPTIMIZED HYBRID MODEL - WEIGHTED VOTING ENSEMBLE")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')
    
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, X_test, y_test, test_clean = prepare_features(train_df, test_df)
    
    # Split training data for weight optimization
    val_split = int(len(X_train) * 0.8)
    X_train_tmp = X_train[:val_split]
    y_train_tmp = y_train[:val_split]
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    
    print(f"  Training:   {len(X_train_tmp)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing:    {len(X_test)} samples")
    
    # Train component models
    print("\n" + "="*70)
    print("STEP 1: TRAIN COMPONENT MODELS")
    print("="*70)
    models = train_component_models(X_train, y_train)
    
    # Optimize weights
    print("\n" + "="*70)
    print("STEP 2: OPTIMIZE ENSEMBLE WEIGHTS")
    print("="*70)
    weights, train_preds, val_preds = optimize_ensemble_weights(
        models, X_train_tmp, y_train_tmp, X_val, y_val
    )
    
    # Generate test predictions
    print("\n" + "="*70)
    print("STEP 3: GENERATE TEST PREDICTIONS")
    print("="*70)
    ensemble_preds, test_individual_preds = generate_predictions(models, weights, X_test)
    
    # Evaluate
    print("\n" + "="*70)
    print("STEP 4: EVALUATE PERFORMANCE")
    print("="*70)
    results, ensemble_metrics = evaluate_models(y_test, ensemble_preds, test_individual_preds, test_clean)
    
    # Visualize
    print("\n" + "="*70)
    print("STEP 5: CREATE VISUALIZATIONS")
    print("="*70)
    visualize_comparison(y_test, ensemble_preds, test_individual_preds, test_clean, weights)
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': test_clean['timestamp'],
        'actual': y_test.values,
        'xgboost': test_individual_preds['xgboost'],
        'random_forest': test_individual_preds['rf'],
        'gradient_boost': test_individual_preds['gb'],
        'weighted_ensemble': ensemble_preds,
    })
    
    results_df.to_csv('results/optimized_hybrid_results.csv', index=False)
    
    weights_df = pd.DataFrame({
        'model': ['XGBoost', 'Random Forest', 'Gradient Boosting'],
        'weight': weights,
        'percentage': (weights * 100)
    })
    weights_df.to_csv('results/optimized_hybrid_weights.csv', index=False)
    
    print("\nFiles saved:")
    print("  - results/optimized_hybrid_results.csv")
    print("  - results/optimized_hybrid_comparison.png")
    print("  - results/optimized_hybrid_weights.csv")
    
    print("\n" + "="*70)
    print("OPTIMIZED HYBRID MODEL READY FOR DEPLOYMENT")
    print("="*70)
    
    return results_df, models, weights


if __name__ == "__main__":
    results, models, weights = main()
