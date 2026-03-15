"""
Continuous Learning Module for Solar Production Models
Enables evolutionary learning, hyperparameter optimization, and concept drift detection.
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')


class ModelVersionManager:
    """Manages model versioning and tracks performance history"""
    
    def __init__(self, model_dir='results/model_versions'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.version_log = self.model_dir / 'version_log.json'
        self.versions = self._load_version_log()
    
    def _load_version_log(self):
        """Load existing version log"""
        if self.version_log.exists():
            with open(self.version_log, 'r') as f:
                return json.load(f)
        return {}
    
    def save_version(self, model, model_name, metrics, hyperparams, data_info):
        """Save a new model version"""
        timestamp = datetime.now().isoformat()
        version_id = f"{model_name}_v{len(self.versions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = self.model_dir / f"{version_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.versions[version_id] = {
            'timestamp': timestamp,
            'model_name': model_name,
            'model_path': str(model_path),
            'metrics': metrics,
            'hyperparams': hyperparams,
            'data_info': data_info
        }
        
        self._save_version_log()
        print(f"✓ Model version saved: {version_id}")
        return version_id
    
    def _save_version_log(self):
        """Save version log to file"""
        # Convert non-serializable types to JSON-safe types
        serializable_versions = {}
        for version_id, info in self.versions.items():
            safe_info = {
                'timestamp': str(info.get('timestamp', '')),
                'model_name': str(info.get('model_name', '')),
                'model_path': str(info.get('model_path', '')),
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in (info.get('metrics', {}) or {}).items()},
                'hyperparams': {k: float(v) if isinstance(v, (np.floating, np.integer)) else
                               bool(v) if isinstance(v, (np.bool_)) else v
                               for k, v in (info.get('hyperparams', {}) or {}).items()},
                'data_info': info.get('data_info', {})
            }
            serializable_versions[version_id] = safe_info
        
        with open(self.version_log, 'w') as f:
            json.dump(serializable_versions, f, indent=2, default=str)
    
    def get_best_version(self, metric='r2_score', improvement_threshold=0.01):
        """Get the best performing model version"""
        if not self.versions:
            return None
        
        best_version = max(
            self.versions.items(),
            key=lambda x: x[1]['metrics'].get(metric, -float('inf'))
        )
        return best_version
    
    def list_versions(self, model_name=None):
        """List all model versions"""
        versions = self.versions
        if model_name:
            versions = {k: v for k, v in versions.items() if v['model_name'] == model_name}
        
        print(f"\n{'Version ID':<50} {'R² Score':<12} {'MAE':<12} {'Timestamp':<20}")
        print("=" * 95)
        for version_id, info in sorted(versions.items(), key=lambda x: x[1]['timestamp'], reverse=True):
            r2 = info['metrics'].get('r2_score', 0)
            mae = info['metrics'].get('mae', 0)
            timestamp = info['timestamp'][:10]
            print(f"{version_id:<50} {r2:<12.4f} {mae:<12.2f} {timestamp:<20}")


class ConceptDriftDetector:
    """Detects concept drift in model performance"""
    
    def __init__(self, window_size=100, threshold=0.15):
        """
        Args:
            window_size: Number of predictions to keep in sliding window
            threshold: Relative performance drop threshold (15% drop = drift)
        """
        self.window_size = window_size
        self.threshold = threshold
        self.error_history = []
        self.baseline_performance = None
    
    def update(self, y_true, y_pred):
        """Update drift detector with new predictions"""
        mae = mean_absolute_error(y_true, y_pred)
        self.error_history.append(mae)
        
        # Keep only recent window
        if len(self.error_history) > self.window_size:
            self.error_history = self.error_history[-self.window_size:]
        
        if self.baseline_performance is None and len(self.error_history) >= 20:
            self.baseline_performance = np.mean(self.error_history[:20])
    
    def detect_drift(self):
        """Check if concept drift is detected"""
        if self.baseline_performance is None or len(self.error_history) < 20:
            return False, 0.0
        
        recent_performance = np.mean(self.error_history[-20:])
        performance_drop = (recent_performance - self.baseline_performance) / self.baseline_performance
        
        drift_detected = performance_drop > self.threshold
        
        return drift_detected, performance_drop
    
    def reset_baseline(self):
        """Reset baseline after retraining"""
        if len(self.error_history) > 0:
            self.baseline_performance = np.mean(self.error_history[-20:])


class EvolutionaryHyperparameterOptimizer:
    """Evolves hyperparameters using genetic algorithm principles"""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def objective(self, params):
        """Objective function to minimize"""
        try:
            n_estimators, max_depth, learning_rate, subsample, colsample_bytree = params
            
            model = xgb.XGBRegressor(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(self.X_train, self.y_train, verbose=False)
            y_pred = model.predict(self.X_val)
            mae = mean_absolute_error(self.y_val, y_pred)
            
            return mae
        except:
            return float('inf')
    
    def optimize(self, iterations=50):
        """Optimize hyperparameters using differential evolution"""
        print("🧬 Evolving hyperparameters...")
        
        bounds = [
            (50, 300),      # n_estimators
            (3, 10),        # max_depth
            (0.01, 0.3),    # learning_rate
            (0.5, 1.0),     # subsample
            (0.5, 1.0)      # colsample_bytree
        ]
        
        result = differential_evolution(
            self.objective,
            bounds,
            seed=42,
            maxiter=iterations,
            workers=1,
            updating='deferred'
        )
        
        best_params = {
            'n_estimators': int(result.x[0]),
            'max_depth': int(result.x[1]),
            'learning_rate': result.x[2],
            'subsample': result.x[3],
            'colsample_bytree': result.x[4]
        }
        
        print(f"✓ Optimized hyperparameters:")
        for key, value in best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return best_params


class IncrementalLearner:
    """Handles incremental training and time-weighted learning"""
    
    def __init__(self, 
                 base_model=None,
                 version_manager=None,
                 drift_detector=None,
                 learning_rate=0.1,
                 time_decay=0.95):
        """
        Args:
            base_model: Initial trained model
            version_manager: ModelVersionManager instance
            drift_detector: ConceptDriftDetector instance
            learning_rate: Rate of learning from new data
            time_decay: Exponential decay factor for old samples (0-1)
        """
        self.model = base_model
        self.version_manager = version_manager or ModelVersionManager()
        self.drift_detector = drift_detector or ConceptDriftDetector()
        self.learning_rate = learning_rate
        self.time_decay = time_decay
        self.training_count = 0
        self.last_retrain = datetime.now()
    
    def calculate_sample_weights(self, dates):
        """Calculate time-weighted importance for samples"""
        try:
            if isinstance(dates, pd.Series):
                dates = dates.values
            elif isinstance(dates, pd.Index):
                dates = dates.values
            
            # Convert to pandas datetime if needed
            dates = pd.to_datetime(dates)
            
            # Calculate days since each sample  
            now = pd.Timestamp.now()
            time_diff = now - dates
            
            # Handle both timedelta64 and Series of Timedelta
            if hasattr(time_diff, 'days'):
                days_ago = time_diff.days.astype(float)
            else:
                days_ago = time_diff.dt.days.astype(float)
            
            # Apply exponential decay: recent data weighted higher
            weights = np.exp(-self.time_decay * (days_ago / 365.0))
            
            # Normalize to sum to 1
            weights = weights / np.sum(weights)
            
            return weights
        except Exception as e:
            # Fallback: uniform weights if time-weighting fails
            return np.ones(len(dates)) / len(dates)
    
    def incremental_train(self, X_new, y_new, dates=None):
        """
        Incrementally train model with new data
        Recent data is weighted more heavily
        """
        if self.model is None:
            raise ValueError("Base model not initialized. Train a model first.")
        
        print(f"📈 Incremental training with {len(X_new)} new samples...")
        
        # Calculate time-weighted importance
        if dates is not None:
            weights = self.calculate_sample_weights(dates)
            print(f"  Time-weighted learning: recent samples weighted higher")
        else:
            weights = np.ones(len(X_new)) / len(X_new)
        
        # Create a temporary model for incremental learning
        temp_model = xgb.XGBRegressor(**self.model.get_params())
        
        try:
            # Train on new data with weighted importance
            temp_model.fit(X_new, y_new, sample_weight=weights * 1000)
            
            # Blend old and new model knowledge
            y_pred_old = self.model.predict(X_new)
            y_pred_new = temp_model.predict(X_new)
            y_blended = (1 - self.learning_rate) * y_pred_old + self.learning_rate * y_pred_new
            
            # Update model by fine-tuning on blended predictions
            self.model.fit(X_new, y_blended, sample_weight=weights * 1000)
            
            self.training_count += 1
            self.last_retrain = datetime.now()
            
            print(f"✓ Incremental training complete (Training #{self.training_count})")
            return True
        except Exception as e:
            print(f"✗ Incremental training failed: {e}")
            return False
    
    def should_retrain(self, y_true, y_pred, performance_threshold=0.02):
        """
        Determine if model should be retrained based on:
        1. Concept drift detection
        2. Performance degradation
        """
        self.drift_detector.update(y_true, y_pred)
        drift_detected, performance_drop = self.drift_detector.detect_drift()
        
        # Also check absolute performance
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        should_retrain = drift_detected or performance_drop > performance_threshold
        
        return should_retrain, {
            'drift_detected': drift_detected,
            'performance_drop': performance_drop,
            'mae': mae,
            'r2_score': r2
        }


class EvolutionaryModelEnsemble:
    """
    Maintains an ensemble where models evolve and are selected based on fitness
    Weak models are replaced with evolved variants of strong models
    """
    
    def __init__(self, max_ensemble_size=5):
        self.models = []
        self.fitness_scores = []
        self.max_ensemble_size = max_ensemble_size
    
    def add_model(self, model, fitness_score):
        """Add model to ensemble"""
        self.models.append(model)
        self.fitness_scores.append(fitness_score)
        
        # Keep only top models
        if len(self.models) > self.max_ensemble_size:
            sorted_indices = np.argsort(self.fitness_scores)[::-1]
            self.models = [self.models[i] for i in sorted_indices[:self.max_ensemble_size]]
            self.fitness_scores = [self.fitness_scores[i] for i in sorted_indices[:self.max_ensemble_size]]
    
    def evolve_model(self, base_model, X_train, y_train, mutation_rate=0.1):
        """
        Create an evolved variant of best model by:
        1. Mutating hyperparameters
        2. Training on new data
        """
        params = base_model.get_params()
        
        # Mutate hyperparameters slightly
        if np.random.random() < mutation_rate:
            params['learning_rate'] *= np.random.uniform(0.9, 1.1)
            params['max_depth'] = max(3, min(10, params['max_depth'] + np.random.randint(-1, 2)))
            params['subsample'] *= np.random.uniform(0.95, 1.05)
        
        evolved_model = xgb.XGBRegressor(**params)
        evolved_model.fit(X_train, y_train)
        
        return evolved_model
    
    def predict_ensemble(self, X):
        """Ensemble prediction (weighted average by fitness)"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        weights = np.array(self.fitness_scores) / sum(self.fitness_scores)
        predictions = np.array([m.predict(X) for m in self.models])
        
        return np.average(predictions, axis=0, weights=weights)


class ContinuousLearningPipeline:
    """
    Orchestrates continuous learning across all components
    Provides high-level interface for the learning system
    """
    
    def __init__(self, model_name='solar_xgboost', model_dir='results/model_versions'):
        self.model_name = model_name
        self.version_manager = ModelVersionManager(model_dir)
        self.drift_detector = ConceptDriftDetector()
        self.learner = None
        self.ensemble = EvolutionaryModelEnsemble()
        self.optimization_history = []
    
    def initialize(self, base_model):
        """Initialize with a trained base model"""
        self.learner = IncrementalLearner(
            base_model=base_model,
            version_manager=self.version_manager,
            drift_detector=self.drift_detector
        )
        print("✓ Continuous learning pipeline initialized")
    
    def update_with_new_data(self, X_new, y_new, dates=None):
        """
        Update model with new data
        Detects drift and triggers retraining if needed
        """
        if self.learner is None:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        # Make predictions on new data
        y_pred = self.learner.model.predict(X_new)
        
        # Check if retraining is needed
        should_retrain, metrics = self.learner.should_retrain(y_new, y_pred)
        
        print(f"\n📊 Performance Metrics:")
        print(f"  MAE: {metrics['mae']:.2f} W")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        print(f"  Performance Drop: {metrics['performance_drop']*100:.2f}%")
        
        if should_retrain:
            print(f"\n⚠️ DRIFT DETECTED - Retraining model...")
            self.learner.incremental_train(X_new, y_new, dates)
            self.drift_detector.reset_baseline()
            
            # Save new version
            self.version_manager.save_version(
                self.learner.model,
                self.model_name,
                metrics,
                self.learner.model.get_params(),
                {'samples': len(X_new), 'dates': str(dates.min()) if dates is not None else 'unknown'}
            )
            
            return {'retrained': True, 'metrics': metrics}
        else:
            print("✓ Model performance stable")
            return {'retrained': False, 'metrics': metrics}
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, iterations=50):
        """
        Run evolutionary hyperparameter optimization
        """
        optimizer = EvolutionaryHyperparameterOptimizer(X_train, y_train, X_val, y_val)
        best_params = optimizer.optimize(iterations)
        
        # Create and train new model with optimized params
        new_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        new_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = new_model.predict(X_val)
        metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2_score': r2_score(y_val, y_pred)
        }
        
        # Save optimized version
        self.version_manager.save_version(
            new_model,
            f"{self.model_name}_evolved",
            metrics,
            best_params,
            {'optimization_iterations': iterations}
        )
        
        # Update learner with evolved model
        self.learner.model = new_model
        
        return new_model, metrics
    
    def get_model_improvement_summary(self):
        """Show improvement over versions"""
        versions = self.version_manager.versions
        if not versions:
            print("No model versions available")
            return
        
        print(f"\n📈 Model Evolution Summary ({len(versions)} versions)")
        print("=" * 70)
        
        r2_scores = [v['metrics'].get('r2_score', 0) for v in versions.values()]
        mae_scores = [v['metrics'].get('mae', 0) for v in versions.values()]
        
        if r2_scores:
            improvement = ((max(r2_scores) - min(r2_scores)) / abs(min(r2_scores))) * 100 if min(r2_scores) != 0 else 0
            print(f"R² Improvement: {min(r2_scores):.4f} → {max(r2_scores):.4f} ({improvement:.1f}%)")
        
        if mae_scores:
            improvement = ((max(mae_scores) - min(mae_scores)) / max(mae_scores)) * 100
            print(f"MAE Improvement: {max(mae_scores):.2f}W → {min(mae_scores):.2f}W ({improvement:.1f}% better)")
        
        self.version_manager.list_versions()


# Example usage
if __name__ == "__main__":
    print("Continuous Learning Module for Solar Production Models")
    print("=" * 60)
    print("This module provides:")
    print("  • Incremental learning with time-weighted samples")
    print("  • Concept drift detection")
    print("  • Evolutionary hyperparameter optimization")
    print("  • Model versioning and evolution tracking")
    print("  • Ensemble-based predictions with model fitness")
