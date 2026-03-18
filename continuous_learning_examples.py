"""
Example: Continuous Learning Integration with Solar Production Models
Shows how to integrate continuous learning into your existing pipeline
"""

import sys
import os
from pathlib import Path

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

from src.models.continuous_learning import (
    ContinuousLearningPipeline,
    ConceptDriftDetector,
    EvolutionaryHyperparameterOptimizer,
    ModelVersionManager
)


def example_1_basic_incremental_learning():
    """
    Example 1: Basic incremental learning
    Shows how to continuously learn from new data streams
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Incremental Learning")
    print("="*70)
    
    # Simulate loading a pre-trained model
    print("\n1. Initialize base model and continuous learning pipeline...")
    
    # For demonstration, we'll create dummy data
    # In practice, load your trained XGBoost model
    import xgboost as xgb
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    X_train = np.random.randn(n_samples, 10)
    y_train = np.random.randn(n_samples) * 100 + 500
    
    # Train base model
    base_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
    base_model.fit(X_train, y_train)
    
    # Initialize pipeline
    pipeline = ContinuousLearningPipeline(model_name='solar_xgboost')
    pipeline.initialize(base_model)
    
    print("\n2. Simulate new data arriving in batches...")
    
    # Simulate 3 batches of new data
    for batch_idx in range(1, 4):
        print(f"\n--- Batch {batch_idx} ---")
        
        # Generate new data
        n_new = 200
        X_new = np.random.randn(n_new, 10)
        y_new = np.random.randn(n_new) * 100 + 500
        
        # Create timestamps for time-weighted learning
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=10),
            periods=n_new,
            freq='1H'
        )
        
        # Update model with new data
        result = pipeline.update_with_new_data(X_new, y_new, dates)
        
        if result['retrained']:
            print(f"  → Model successfully retrained!")
        else:
            print(f"  → Model performing well, no retraining needed")
    
    # Show evolution summary
    pipeline.get_model_improvement_summary()


def example_2_hyperparameter_evolution():
    """
    Example 2: Evolutionary hyperparameter optimization
    Shows how to automatically improve hyperparameters over time
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Evolutionary Hyperparameter Optimization")
    print("="*70)
    
    print("\n1. Generate synthetic solar production data...")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Create more realistic features (irradiance, temperature, humidity, etc)
    X_train = np.random.randn(n_samples, 10)
    # Solar power: strong correlation with irradiance
    y_train = (X_train[:, 0] * 200 + np.random.randn(n_samples) * 50).clip(0, None)
    
    # Split into train and validation
    split_idx = int(0.8 * n_samples)
    X_train_actual = X_train[:split_idx]
    y_train_actual = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    print(f"  Training samples: {len(X_train_actual)}")
    print(f"  Validation samples: {len(X_val)}")
    
    print("\n2. Initialize continuous learning pipeline...")
    
    # Create and train base model
    import xgboost as xgb
    base_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    base_model.fit(X_train_actual, y_train_actual)
    
    pipeline = ContinuousLearningPipeline(model_name='solar_evolved')
    pipeline.initialize(base_model)
    
    # Evaluate base model
    y_pred_base = base_model.predict(X_val)
    from sklearn.metrics import mean_absolute_error, r2_score
    base_mae = mean_absolute_error(y_val, y_pred_base)
    base_r2 = r2_score(y_val, y_pred_base)
    
    print(f"\n3. Base Model Performance:")
    print(f"  MAE: {base_mae:.2f} W")
    print(f"  R² Score: {base_r2:.4f}")
    
    print("\n4. Optimizing hyperparameters using genetic algorithm...")
    print("  (This may take a minute...)")
    
    # Optimize hyperparameters
    evolved_model, metrics = pipeline.optimize_hyperparameters(
        X_train_actual, y_train_actual,
        X_val, y_val,
        iterations=30  # Reduced for demo, use 50+ in production
    )
    
    print(f"\n5. Evolved Model Performance:")
    print(f"  MAE: {metrics['mae']:.2f} W (improvement: {((base_mae - metrics['mae']) / base_mae * 100):.1f}%)")
    print(f"  R² Score: {metrics['r2_score']:.4f} (improvement: {((metrics['r2_score'] - base_r2) / abs(base_r2) * 100):.1f}%)")
    

def example_3_drift_detection_and_recovery():
    """
    Example 3: Concept drift detection and automatic recovery
    Shows how model detects and adapts to changing environment
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Concept Drift Detection")
    print("="*70)
    
    print("\n1. Generate baseline and shifted data...")
    
    np.random.seed(42)
    
    # Create baseline training data
    X_train = np.random.randn(1000, 10)
    y_train = X_train[:, 0] * 200 + np.random.randn(1000) * 50 + 500
    
    # Create test data with SHIFT (concept drift)
    # Simulate seasonal change or system degradation
    X_test_normal = np.random.randn(200, 10)
    y_test_normal = X_test_normal[:, 0] * 200 + np.random.randn(200) * 50 + 500
    
    # Simulate concept drift (e.g., equipment degradation)
    X_test_drift = np.random.randn(200, 10)
    y_test_drift = X_test_drift[:, 0] * 150 + np.random.randn(200) * 80 + 300  # Changed pattern
    
    print(f"  Baseline: mean={np.mean(y_train):.0f}W")
    print(f"  Normal test: mean={np.mean(y_test_normal):.0f}W")
    print(f"  Drifted test: mean={np.mean(y_test_drift):.0f}W (simulated degradation)")
    
    print("\n2. Train base model...")
    
    import xgboost as xgb
    base_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
    base_model.fit(X_train, y_train)
    
    pipeline = ContinuousLearningPipeline(model_name='solar_adaptive')
    pipeline.initialize(base_model)
    
    print("\n3. Test on normal data (no drift)...")
    y_pred_normal = base_model.predict(X_test_normal)
    result = pipeline.update_with_new_data(X_test_normal, y_test_normal)
    
    print("\n4. Test on drifted data (concept drift)...")
    y_pred_drift = base_model.predict(X_test_drift)
    result = pipeline.update_with_new_data(X_test_drift, y_test_drift)
    
    if result['retrained']:
        print("\n✓ Model detected drift and adapted automatically!")


def example_4_production_usage():
    """
    Example 4: Production-ready usage pattern
    Shows how to integrate into a real monitoring system
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Production Usage Pattern")
    print("="*70)
    
    print("""
    In production, you would use the pipeline like this:

    # At startup, load your trained model
    with open('models/trained_xgboost.pkl', 'rb') as f:
        trained_model = pickle.load(f)
    
    # Initialize continuous learning
    pipeline = ContinuousLearningPipeline(model_name='production_solar')
    pipeline.initialize(trained_model)
    
    # In your data collection loop:
    while True:
        # Collect new sensor readings
        new_data_batch = collect_sensor_data()  # Shape: (batch_size, n_features)
        timestamps = get_timestamps()           # For time-weighted learning
        actual_production = get_actual_production()  # Ground truth
        
        # Update model with new data
        result = pipeline.update_with_new_data(
            new_data_batch,
            actual_production,
            timestamps
        )
        
        if result['retrained']:
            print(f"Model improved! New metrics: {result['metrics']}")
        
        # Every month, run hyperparameter optimization
        if datetime.now().day == 1:
            evolved_model, metrics = pipeline.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, iterations=50
            )
        
        # Monitor model versions
        pipeline.get_model_improvement_summary()
        
        # Make predictions for next hour
        predictions = pipeline.learner.model.predict(next_hour_features)
        return predictions
    """)
    
    print("\nKey Benefits in Production:")
    print("  ✓ Auto-detects performance degradation")
    print("  ✓ Learns from new patterns automatically")
    print("  ✓ Tracks model evolution over time")
    print("  ✓ Maintains historical versions for comparison")
    print("  ✓ Time-weights recent data more heavily")
    print("  ✓ No manual retraining needed")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONTINUOUS LEARNING PIPELINE - EXAMPLES")
    print("="*70)
    print("""
    This script demonstrates 4 key capabilities:
    1. Incremental learning from new data streams
    2. Evolutionary hyperparameter optimization
    3. Automatic drift detection and adaptation
    4. Production-ready integration pattern
    """)
    
    # Run examples
    example_1_basic_incremental_learning()
    example_2_hyperparameter_evolution()
    example_3_drift_detection_and_recovery()
    example_4_production_usage()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("""
    Next steps:
    1. Integrate continuous_learning.py into your models
    2. Modify your pipeline scripts to use ContinuousLearningPipeline
    3. Set up monitoring to track model performance
    4. Schedule periodic hyperparameter optimization
    5. Review model versions monthly in results/model_versions/
    """)
