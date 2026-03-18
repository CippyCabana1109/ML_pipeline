"""
Advanced Ensemble Model - Multi-Model Stacking with Meta-Learner
Combines XGBoost, LightGBM, Random Forest, and Neural Network with learned optimal weights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


def prepare_features(train_df, test_df):
    """Prepare features for all models"""
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
    
    return X_train, y_train, X_test, y_test, test_clean, feature_columns


def train_base_models(X_train, y_train):
    """Train diverse base models"""
    print("Training base models...")
    
    models = {}
    
    # XGBoost
    print("  Training XGBoost...")
    models['xgboost'] = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    models['xgboost'].fit(X_train, y_train)
    
    # LightGBM
    print("  Training LightGBM...")
    models['lightgbm'] = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    models['lightgbm'].fit(X_train, y_train)
    
    # Random Forest
    print("  Training Random Forest...")
    models['random_forest'] = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    models['random_forest'].fit(X_train, y_train)
    
    # Gradient Boosting
    print("  Training Gradient Boosting...")
    models['gradient_boost'] = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        random_state=42,
    )
    models['gradient_boost'].fit(X_train, y_train)
    
    print("Base models training complete!")
    return models


def create_meta_features(models, X_train, y_train, X_test, y_test):
    """Create meta-features (base model predictions for stacking)"""
    print("Creating meta-features for second-level model...")
    
    # Training meta-features
    meta_train = np.zeros((X_train.shape[0], len(models)))
    for idx, (name, model) in enumerate(models.items()):
        print(f"  {name} train predictions...")
        meta_train[:, idx] = model.predict(X_train)
    
    # Test meta-features
    meta_test = np.zeros((X_test.shape[0], len(models)))
    for idx, (name, model) in enumerate(models.items()):
        print(f"  {name} test predictions...")
        meta_test[:, idx] = model.predict(X_test)
    
    return meta_train, meta_test


def train_meta_learner(meta_train, y_train):
    """Train meta-learner (learns optimal combination of base models)"""
    print("Training meta-learner...")
    
    meta_model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    
    meta_model.fit(meta_train, y_train)
    
    # Get feature importance (weights for each base model)
    feature_names = ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boost']
    importance = meta_model.feature_importances_
    
    print("\nMeta-Learner Weights (Optimal Combination):")
    print("=" * 50)
    for name, weight in zip(feature_names, importance):
        print(f"  {name:20s}: {weight:.4f} ({weight/importance.sum()*100:5.1f}%)")
    
    print(f"  {'Total':20s}: {importance.sum():.4f}")
    
    return meta_model, importance


def generate_ensemble_predictions(models, meta_learner, meta_test, X_test):
    """Generate ensemble predictions"""
    print("Generating ensemble predictions...")
    
    ensemble_predictions = meta_learner.predict(meta_test)
    
    # Also get individual model predictions for comparison
    individual_preds = {}
    for name, model in models.items():
        individual_preds[name] = model.predict(X_test)
    
    return ensemble_predictions, individual_preds


def evaluate_all_models(y_test, x_test_data):
    """Evaluate all models comprehensively"""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Create daytime filter
    daytime_filter = create_daytime_filter(x_test_data['irradiance'])
    
    # Calculate metrics for each model
    results = {}
    
    for model_name, predictions in x_test_data.items():
        if isinstance(predictions, np.ndarray):
            metrics = calculate_metrics(y_test, predictions)
            results[model_name] = metrics
            
            print(f"\n{model_name.upper()}:")
            print(f"  MAE:   {metrics['mae']:8.2f} W")
            print(f"  RMSE:  {metrics['rmse']:8.2f} W")
            print(f"  sMAPE: {metrics['smape']:8.2f} %")
            print(f"  R²:    {metrics['r2']:8.4f}")
    
    return results


def visualize_ensemble_comparison(y_test, predictions_dict, test_data):
    """Visualize ensemble vs individual models"""
    print("\nCreating comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All predictions vs actual
    ax = axes[0, 0]
    ax.plot(test_data['timestamp'], y_test, 'k-', label='Actual', linewidth=2.5, alpha=0.8)
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for idx, (name, preds) in enumerate(predictions_dict.items()):
        if name != 'actual':
            ax.plot(test_data['timestamp'], preds, '--', label=name, alpha=0.6, color=colors[idx % len(colors)])
    ax.set_title('All Models: Actual vs Predicted', fontweight='bold', fontsize=12)
    ax.set_ylabel('Solar Power (W)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ensemble performance
    ax = axes[0, 1]
    ensemble_preds = predictions_dict['ensemble']
    residuals = y_test - ensemble_preds
    ax.scatter(ensemble_preds, residuals, alpha=0.6, s=50)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    mae = np.mean(np.abs(residuals))
    ax.set_title(f'Ensemble Residuals (MAE={mae:.1f}W)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted Power (W)')
    ax.set_ylabel('Residual (W)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    ax = axes[1, 0]
    for name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost', 'ensemble']:
        if name in predictions_dict:
            errors = np.abs(y_test - predictions_dict[name])
            ax.hist(errors, bins=20, alpha=0.5, label=name)
    ax.set_title('Error Distribution Comparison', fontweight='bold', fontsize=12)
    ax.set_xlabel('Absolute Error (W)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Model performance metrics
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics_text = "MODEL PERFORMANCE SUMMARY\n" + "="*40 + "\n\n"
    for name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost', 'ensemble']:
        if name in predictions_dict:
            preds = predictions_dict[name]
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            metrics_text += f"{name:15s}  MAE={mae:7.1f}W  R²={r2:.4f}\n"
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/advanced_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved!")


def main():
    """Main execution"""
    print("="*70)
    print("ADVANCED ENSEMBLE MODEL - MULTI-MODEL STACKING")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv('data/train_final.csv')
    test_df = pd.read_csv('data/test_final.csv')
    
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, X_test, y_test, test_clean, feature_columns = prepare_features(train_df, test_df)
    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Testing:  {X_test.shape[0]} samples")
    
    # Train base models
    print("\n" + "="*70)
    print("STEP 1: TRAIN BASE MODELS")
    print("="*70)
    models = train_base_models(X_train, y_train)
    
    # Create meta-features
    print("\n" + "="*70)
    print("STEP 2: CREATE META-FEATURES")
    print("="*70)
    meta_train, meta_test = create_meta_features(models, X_train, y_train, X_test, y_test)
    
    # Train meta-learner
    print("\n" + "="*70)
    print("STEP 3: TRAIN META-LEARNER")
    print("="*70)
    meta_learner, weights = train_meta_learner(meta_train, y_train)
    
    # Generate predictions
    print("\n" + "="*70)
    print("STEP 4: GENERATE ENSEMBLE PREDICTIONS")
    print("="*70)
    ensemble_preds, individual_preds = generate_ensemble_predictions(models, meta_learner, meta_test, X_test)
    
    # Evaluate all models
    print("\n" + "="*70)
    print("STEP 5: EVALUATE ALL MODELS")
    print("="*70)
    
    predictions_dict = {
        'actual': y_test,
        'xgboost': individual_preds['xgboost'],
        'lightgbm': individual_preds['lightgbm'],
        'random_forest': individual_preds['random_forest'],
        'gradient_boost': individual_preds['gradient_boost'],
        'ensemble': ensemble_preds,
    }
    
    # Evaluate with metrics
    daytime_filter = create_daytime_filter(test_clean['irradiance'])
    
    print("\nDetailed Results:")
    print("="*70)
    
    best_mae = float('inf')
    best_model = None
    
    for name, preds in predictions_dict.items():
        if name == 'actual':
            continue
        metrics = calculate_metrics(y_test, preds)
        daytime_metrics = calculate_metrics(y_test, preds, daytime_filter)
        
        print(f"\n{name.upper()}")
        print(f"  All Data:     MAE={metrics['mae']:7.1f}W  RMSE={metrics['rmse']:7.1f}W  R²={metrics['r2']:.4f}")
        print(f"  Daytime:      MAE={daytime_metrics['mae']:7.1f}W  RMSE={daytime_metrics['rmse']:7.1f}W  R²={daytime_metrics['r2']:.4f}")
        
        if metrics['mae'] < best_mae:
            best_mae = metrics['mae']
            best_model = name
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model.upper()} with MAE = {best_mae:.1f}W")
    print(f"{'='*70}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_ensemble_comparison(y_test, predictions_dict, test_clean)
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': test_clean['timestamp'],
        'actual': y_test,
        'xgboost': individual_preds['xgboost'],
        'lightgbm': individual_preds['lightgbm'],
        'random_forest': individual_preds['random_forest'],
        'gradient_boost': individual_preds['gradient_boost'],
        'ensemble': ensemble_preds,
    })
    
    results_df.to_csv('results/advanced_ensemble_results.csv', index=False)
    
    # Save model weights
    weights_df = pd.DataFrame({
        'model': ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boost'],
        'weight': weights,
        'percentage': (weights / weights.sum() * 100)
    })
    weights_df.to_csv('results/ensemble_model_weights.csv', index=False)
    
    print("\nFiles saved:")
    print("  - results/advanced_ensemble_results.csv")
    print("  - results/advanced_ensemble_comparison.png")
    print("  - results/ensemble_model_weights.csv")
    
    return results_df, models, meta_learner


if __name__ == "__main__":
    results, models, meta_learner = main()
