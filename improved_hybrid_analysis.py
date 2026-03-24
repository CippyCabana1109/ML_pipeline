"""
Improved Hybrid Model Analysis with Better Error Handling
MSc Dissertation - Solar Forecasting System

This script creates an improved hybrid model with better handling of percentage errors,
particularly for nighttime generation values near zero.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for academic plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def generate_realistic_solar_data():
    """Generate realistic solar data with proper day/night cycles"""
    print("Generating realistic solar data for improved hybrid analysis...")
    
    # Create time series
    start_date = datetime(2023, 6, 1)  # Summer month for better generation
    end_date = datetime(2023, 6, 30)   # One month for detailed analysis
    dates = pd.date_range(start_date, end_date, freq='H')
    
    n_hours = len(dates)
    np.random.seed(42)
    
    # Generate realistic solar generation with proper day/night
    hours = np.array([d.hour for d in dates])
    days = np.array([d.day for d in dates])
    
    # Solar generation only during daylight hours (6am-6pm)
    solar_hours = (hours >= 6) & (hours <= 18)
    
    # Base solar pattern
    hour_factor = np.zeros(n_hours)
    hour_factor[solar_hours] = np.sin((hours[solar_hours] - 6) * np.pi / 12)
    
    # Daily variation
    daily_variation = 0.8 + 0.2 * np.sin(days * 2 * np.pi / 30)
    
    # Base generation (peak 4kW system)
    base_generation = hour_factor * daily_variation * 4000
    
    # Add weather effects
    cloud_factor = 1 - 0.4 * np.random.beta(2, 3, n_hours)
    temp_factor = 0.9 + 0.2 * np.random.normal(0, 1, n_hours)
    
    # Actual generation
    actual_generation = base_generation * cloud_factor * temp_factor
    actual_generation += np.random.normal(0, 50, n_hours)  # Measurement noise
    actual_generation = np.maximum(0, actual_generation)
    
    # Ensure nighttime is near zero
    actual_generation[~solar_hours] = np.random.exponential(5, np.sum(~solar_hours))
    
    # Weather variables
    temperature = 20 + 10 * np.sin(days * 2 * np.pi / 30) + 5 * np.random.normal(0, 1, n_hours)
    humidity = 50 + 30 * np.random.beta(2, 2, n_hours)
    wind_speed = 3 + 7 * np.random.gamma(2, 2, n_hours)
    cloud_cover = np.random.beta(2, 3, n_hours) * 100
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'hour': hours,
        'day': days,
        'actual_generation': actual_generation,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover,
        'is_daylight': solar_hours
    })
    
    print(f"OK Generated {len(df)} hourly data points with realistic day/night cycles")
    return df

def create_improved_models(df):
    """Create improved individual models with better feature engineering"""
    print("Creating improved individual model predictions...")
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size].copy()
    test_df = df[train_size:].copy()
    
    # Enhanced feature engineering
    def create_enhanced_features(data):
        features = data.copy()
        
        # Time features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day'] / 30)
        features['day_cos'] = np.cos(2 * np.pi * features['day'] / 30)
        
        # Solar position features
        features['solar_angle'] = np.maximum(0, np.sin((features['hour'] - 6) * np.pi / 12))
        features['is_solar_peak'] = ((features['hour'] >= 11) & (features['hour'] <= 13)).astype(int)
        
        # Weather interactions
        features['temp_cloud'] = features['temperature'] * features['cloud_cover']
        features['humidity_cloud'] = features['humidity'] * features['cloud_cover']
        
        # Lag features (only for daylight hours)
        features['prev_hour_gen'] = features['actual_generation'].shift(1)
        features['prev_same_hour'] = features['actual_generation'].shift(24)
        
        # Rolling averages
        features['rolling_3h'] = features['actual_generation'].rolling(3, min_periods=1).mean()
        features['rolling_24h'] = features['actual_generation'].rolling(24, min_periods=1).mean()
        
        return features
    
    train_features = create_enhanced_features(train_df)
    test_features = create_enhanced_features(test_df)
    
    # Handle missing values
    for col in train_features.columns:
        if train_features[col].dtype in ['float64', 'int64']:
            train_features[col] = train_features[col].fillna(train_features[col].median())
            test_features[col] = test_features[col].fillna(train_features[col].median())
    
    # Feature columns
    feature_cols = ['hour', 'day', 'temperature', 'humidity', 'wind_speed', 'cloud_cover',
                   'is_daylight', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                   'solar_angle', 'is_solar_peak', 'temp_cloud', 'humidity_cloud',
                   'prev_hour_gen', 'prev_same_hour', 'rolling_3h', 'rolling_24h']
    
    X_train = train_features[feature_cols]
    X_test = test_features[feature_cols]
    y_train = train_features['actual_generation']
    y_test = test_features['actual_generation']
    
    # Model 1: Random Forest (Best for non-linear patterns)
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Model 2: Gradient Boosting (Best for sequential learning)
    gb_model = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=6,
        min_samples_split=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    # Model 3: Physics-based (Best for solar physics)
    def physics_based_predict(data):
        """Physics-based solar generation model"""
        predictions = []
        
        for _, row in data.iterrows():
            if not row['is_daylight']:
                # Nighttime generation
                pred = np.random.exponential(5)  # Small baseline
            else:
                # Daytime generation based on physics
                # Clear sky generation
                clear_sky = 4000 * row['solar_angle']
                
                # Weather attenuation
                cloud_attenuation = 1 - 0.007 * row['cloud_cover']  # 0.7% loss per % cloud
                temp_efficiency = 1 - 0.004 * abs(row['temperature'] - 25)  # Optimal at 25°C
                
                pred = clear_sky * cloud_attenuation * temp_efficiency
                
                # Add some variability
                pred += np.random.normal(0, 50)
            
            predictions.append(max(0, pred))
        
        return np.array(predictions)
    
    physics_pred = physics_based_predict(test_features)
    
    # Model 4: Time Series (Best for temporal patterns)
    def time_series_predict(data):
        """Time series model with seasonal patterns"""
        predictions = []
        
        # Calculate historical patterns
        hourly_avg = train_df.groupby('hour')['actual_generation'].mean()
        daily_pattern = train_df.groupby('day')['actual_generation'].mean()
        
        for i, (_, row) in enumerate(data.iterrows()):
            if i < 24:
                # Use historical averages for initial predictions
                pred = hourly_avg.get(row['hour'], 0) * 0.8
            else:
                # Combine recent trend with seasonal patterns
                recent_avg = data.iloc[i-24:i]['actual_generation'].mean()
                seasonal_factor = hourly_avg.get(row['hour'], 1) / hourly_avg.mean()
                
                pred = recent_avg * seasonal_factor
            
            # Weather adjustment
            weather_factor = 1 - 0.005 * row['cloud_cover']
            pred *= weather_factor
            
            predictions.append(max(0, pred))
        
        return np.array(predictions)
    
    ts_pred = time_series_predict(test_features)
    
    # Model 5: XGBoost-like (Best for complex interactions)
    def xgboost_like_predict(X_train, y_train, X_test):
        """Enhanced gradient boosting model"""
        model = GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=7,
            subsample=0.8,
            min_samples_split=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)
    
    xgb_pred = xgboost_like_predict(X_train, y_train, X_test)
    
    # Store results
    results = {
        'datetime': test_features['datetime'],
        'actual': y_test,
        'Random Forest': rf_pred,
        'Gradient Boosting': gb_pred,
        'Physics-based': physics_pred,
        'Time Series': ts_pred,
        'XGBoost-like': xgb_pred
    }
    
    results_df = pd.DataFrame(results)
    print("OK Improved individual model predictions created")
    
    return results_df, feature_cols, rf_model, gb_model

def create_advanced_hybrid_model(results_df):
    """Create advanced hybrid model with intelligent combination"""
    print("Creating advanced hybrid model...")
    
    hybrid_data = results_df.copy()
    
    # Calculate individual model errors
    models = ['Random Forest', 'Gradient Boosting', 'Physics-based', 'Time Series', 'XGBoost-like']
    
    for model in models:
        hybrid_data[f'{model}_error'] = hybrid_data['actual'] - hybrid_data[model]
        hybrid_data[f'{model}_abs_error'] = np.abs(hybrid_data[f'{model}_error'])
        hybrid_data[f'{model}_squared_error'] = hybrid_data[f'{model}_error'] ** 2
    
    # Basic ensembles
    hybrid_data['simple_average'] = hybrid_data[models].mean(axis=1)
    hybrid_data['median_ensemble'] = hybrid_data[models].median(axis=1)
    
    # Performance-weighted ensemble
    def calculate_performance_weights(window_size=48):
        """Calculate weights based on recent performance"""
        weights = []
        
        for i in range(len(hybrid_data)):
            if i < window_size:
                # Use equal weights initially
                weight = np.ones(len(models)) / len(models)
            else:
                # Calculate recent RMSE for each model
                recent_rmse = []
                for model in models:
                    recent_errors = hybrid_data[f'{model}_squared_error'].iloc[i-window_size:i]
                    rmse = np.sqrt(np.mean(recent_errors))
                    recent_rmse.append(1 / (rmse + 1e-6))  # Inverse RMSE
                
                # Normalize weights
                weight = np.array(recent_rmse) / np.sum(recent_rmse)
            
            weights.append(weight)
        
        return np.array(weights)
    
    perf_weights = calculate_performance_weights()
    hybrid_data['performance_weighted'] = 0
    
    for i, model in enumerate(models):
        hybrid_data['performance_weighted'] += hybrid_data[model] * perf_weights[:, i]
    
    # Conditional ensemble based on time and conditions
    def conditional_ensemble(row):
        """Intelligent model selection based on conditions"""
        hour = row.name % 24 if hasattr(row, 'name') else 12
        
        # Daytime vs Nighttime
        if 6 <= hour <= 18:  # Daytime
            # Prefer physics-based and ML models during day
            return (row['Physics-based'] * 0.3 + 
                   row['XGBoost-like'] * 0.3 + 
                   row['Random Forest'] * 0.2 + 
                   row['performance_weighted'] * 0.2)
        else:  # Nighttime
            # Prefer time series and simple models at night
            return (row['Time Series'] * 0.4 + 
                   row['median_ensemble'] * 0.3 + 
                   row['simple_average'] * 0.3)
    
    hybrid_data['conditional_ensemble'] = hybrid_data.apply(conditional_ensemble, axis=1)
    
    # Final hybrid prediction (optimized combination)
    hybrid_data['hybrid_prediction'] = (
        hybrid_data['performance_weighted'] * 0.35 +
        hybrid_data['conditional_ensemble'] * 0.35 +
        hybrid_data['XGBoost-like'] * 0.15 +
        hybrid_data['simple_average'] * 0.15
    )
    
    print("OK Advanced hybrid model created")
    return hybrid_data

def calculate_improved_percentage_errors(results_df):
    """Calculate improved percentage errors with better handling of zero values"""
    print("Calculating improved percentage errors...")
    
    models = ['Random Forest', 'Gradient Boosting', 'Physics-based', 'Time Series', 'XGBoost-like', 'hybrid_prediction']
    error_metrics = []
    
    for model in models:
        if model in results_df.columns:
            actual = results_df['actual']
            predicted = results_df[model]
            
            # Basic metrics
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            
            # Improved percentage error calculation
            # Only calculate for actual values > 10W to avoid division by very small numbers
            significant_actual = actual > 10
            
            if np.sum(significant_actual) > 0:
                pct_errors_significant = np.abs((actual[significant_actual] - predicted[significant_actual]) / 
                                               actual[significant_actual]) * 100
                mape_significant = np.mean(pct_errors_significant)
                
                # Symmetric MAPE
                smape = np.mean(2 * np.abs(actual[significant_actual] - predicted[significant_actual]) / 
                               (np.abs(actual[significant_actual]) + np.abs(predicted[significant_actual]) + 1e-6)) * 100
            else:
                mape_significant = np.inf
                smape = np.inf
            
            # Overall percentage error (including all values)
            pct_errors_all = np.abs((actual - predicted) / (actual + 1e-6)) * 100
            mape_all = np.mean(pct_errors_all)
            
            # Error distribution
            error_std = np.std(pct_errors_significant) if np.sum(significant_actual) > 0 else np.inf
            error_95th = np.percentile(pct_errors_significant, 95) if np.sum(significant_actual) > 0 else np.inf
            
            # Accuracy metrics (for significant values only)
            if np.sum(significant_actual) > 0:
                accuracy_5 = np.mean(pct_errors_significant <= 5) * 100
                accuracy_10 = np.mean(pct_errors_significant <= 10) * 100
                accuracy_20 = np.mean(pct_errors_significant <= 20) * 100
            else:
                accuracy_5 = accuracy_10 = accuracy_20 = 0
            
            error_metrics.append({
                'Model': model,
                'MAE (W)': mae,
                'RMSE (W)': rmse,
                'R²': r2,
                'MAPE_All (%)': mape_all,
                'MAPE_Significant (%)': mape_significant,
                'sMAPE (%)': smape,
                'Error Std (%)': error_std,
                '95th Percentile Error (%)': error_95th,
                'Accuracy ≤5% (%)': accuracy_5,
                'Accuracy ≤10% (%)': accuracy_10,
                'Accuracy ≤20% (%)': accuracy_20,
                'Significant Points': np.sum(significant_actual)
            })
    
    error_df = pd.DataFrame(error_metrics)
    print("OK Improved percentage errors calculated")
    return error_df

def create_enhanced_visualizations(results_df, error_df):
    """Create enhanced visualizations with better error handling"""
    print("Creating enhanced visualizations...")
    
    models = ['Random Forest', 'Gradient Boosting', 'Physics-based', 'Time Series', 'XGBoost-like', 'hybrid_prediction']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Individual model comparison plots
    for model, color in zip(models, colors):
        if model in results_df.columns:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{model} Model - Comprehensive Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Time series (sample for clarity)
            sample_size = min(168, len(results_df))  # One week
            sample_data = results_df.iloc[:sample_size]
            
            ax1.plot(sample_data.index, sample_data['actual'], 'k-', linewidth=2, label='Actual', alpha=0.8)
            ax1.plot(sample_data.index, sample_data[model], color=color, linewidth=2, label=f'{model}', alpha=0.7)
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Generation (W)')
            ax1.set_title('Time Series Comparison (1-week sample)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Scatter plot
            ax2.scatter(results_df['actual'], results_df[model], alpha=0.6, s=20, 
                       color=color, edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(results_df['actual'].min(), results_df[model].min())
            max_val = max(results_df['actual'].max(), results_df[model].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            r2 = r2_score(results_df['actual'], results_df[model])
            ax2.set_xlabel('Actual Generation (W)')
            ax2.set_ylabel('Predicted Generation (W)')
            ax2.set_title(f'Scatter Plot (R² = {r2:.4f})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Error distribution
            errors = results_df['actual'] - results_df[model]
            ax3.hist(errors, bins=50, alpha=0.7, color=color, edgecolor='black')
            ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            ax3.set_xlabel('Prediction Error (W)')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'Error Distribution (Mean = {np.mean(errors):.2f}W)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Percentage error by generation level
            significant_actual = results_df['actual'] > 10
            if np.sum(significant_actual) > 0:
                pct_errors = np.abs((results_df['actual'][significant_actual] - 
                                   results_df[model][significant_actual]) / 
                                  results_df['actual'][significant_actual]) * 100
                
                ax4.scatter(results_df['actual'][significant_actual], pct_errors, 
                           alpha=0.6, s=15, color=color, edgecolors='black', linewidth=0.5)
                ax4.axhline(10, color='red', linestyle='--', alpha=0.7, label='10% Error Threshold')
                ax4.set_xlabel('Actual Generation (W)')
                ax4.set_ylabel('Percentage Error (%)')
                ax4.set_title('Percentage Error vs Generation Level')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filename = f'Enhanced_{model.replace(" ", "_").replace("-", "_")}_Analysis.png'
            filepath = f'DISSERTATION_FIGURES/{filename}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"FILES Enhanced {model} analysis saved: {filepath}")
    
    # 2. Comprehensive comparison plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Comprehensive Hybrid Model Comparison - Improved Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: MAE Comparison
    ax1 = axes[0, 0]
    mae_values = [error_df[error_df['Model'] == model]['MAE (W)'].values[0] for model in models]
    bars = ax1.bar(models, mae_values, color=colors, alpha=0.7)
    ax1.set_ylabel('MAE (W)')
    ax1.set_title('Mean Absolute Error')
    ax1.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: RMSE Comparison
    ax2 = axes[0, 1]
    rmse_values = [error_df[error_df['Model'] == model]['RMSE (W)'].values[0] for model in models]
    bars = ax2.bar(models, rmse_values, color=colors, alpha=0.7)
    ax2.set_ylabel('RMSE (W)')
    ax2.set_title('Root Mean Square Error')
    ax2.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: R² Comparison
    ax3 = axes[0, 2]
    r2_values = [error_df[error_df['Model'] == model]['R²'].values[0] for model in models]
    bars = ax3.bar(models, r2_values, color=colors, alpha=0.7)
    ax3.set_ylabel('R²')
    ax3.set_title('R² Score')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim([0, 1])
    for bar, value in zip(bars, r2_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: MAPE (Significant values only)
    ax4 = axes[1, 0]
    mape_values = [error_df[error_df['Model'] == model]['MAPE_Significant (%)'].values[0] for model in models]
    bars = ax4.bar(models, mape_values, color=colors, alpha=0.7)
    ax4.set_ylabel('MAPE (%)')
    ax4.set_title('MAPE (Significant Values >10W)')
    ax4.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars, mape_values):
        if value != np.inf:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Accuracy ≤10%
    ax5 = axes[1, 1]
    acc_values = [error_df[error_df['Model'] == model]['Accuracy ≤10% (%)'].values[0] for model in models]
    bars = ax5.bar(models, acc_values, color=colors, alpha=0.7)
    ax5.set_ylabel('Accuracy ≤10% (%)')
    ax5.set_title('Prediction Accuracy (≤10% Error)')
    ax5.tick_params(axis='x', rotation=45)
    ax5.set_ylim([0, 100])
    for bar, value in zip(bars, acc_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Error Distribution (Box Plot)
    ax6 = axes[1, 2]
    error_data = []
    for model in models:
        if model in results_df.columns:
            significant_actual = results_df['actual'] > 10
            if np.sum(significant_actual) > 0:
                pct_errors = np.abs((results_df['actual'][significant_actual] - 
                                   results_df[model][significant_actual]) / 
                                  results_df['actual'][significant_actual]) * 100
                error_data.append(pct_errors[pct_errors < 100])  # Remove extreme outliers
    
    bp = ax6.boxplot(error_data, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_ylabel('Percentage Error (%)')
    ax6.set_title('Error Distribution (Truncated at 100%)')
    ax6.tick_params(axis='x', rotation=45)
    
    # Plot 7: Sample Time Series (All models)
    ax7 = axes[2, 0]
    sample_size = min(120, len(results_df))  # 5 days
    sample_data = results_df.iloc[:sample_size]
    
    ax7.plot(sample_data.index, sample_data['actual'], 'k-', linewidth=3, label='Actual', alpha=0.9)
    for model, color in zip(models[:4], colors[:4]):  # Show first 4 models for clarity
        if model in results_df.columns:
            ax7.plot(sample_data.index, sample_data[model], color=color, linewidth=1.5, 
                    label=model, alpha=0.7)
    ax7.set_xlabel('Time (hours)')
    ax7.set_ylabel('Generation (W)')
    ax7.set_title('Sample Time Series Comparison (5 days)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Hybrid vs Best Individual
    ax8 = axes[2, 1]
    if 'hybrid_prediction' in results_df.columns:
        # Find best individual model by MAE
        best_model = error_df[error_df['Model'] != 'hybrid_prediction'].sort_values('MAE (W)').iloc[0]['Model']
        
        ax8.scatter(results_df['actual'], results_df['hybrid_prediction'], 
                   alpha=0.6, s=15, color='red', label='Hybrid', edgecolors='black')
        ax8.scatter(results_df['actual'], results_df[best_model], 
                   alpha=0.4, s=10, color='blue', label=f'Best: {best_model}')
        
        min_val = results_df['actual'].min()
        max_val = results_df['actual'].max()
        ax8.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)
        
        ax8.set_xlabel('Actual Generation (W)')
        ax8.set_ylabel('Predicted Generation (W)')
        ax8.set_title(f'Hybrid vs Best Individual ({best_model})')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # Plot 9: Overall Performance Ranking
    ax9 = axes[2, 2]
    # Create comprehensive ranking
    error_df_copy = error_df[error_df['Model'].isin(models)].copy()
    
    # Normalize metrics for ranking
    rankings = []
    for _, row in error_df_copy.iterrows():
        score = 0
        # Lower is better for errors
        score += row['MAE (W)'] / error_df_copy['MAE (W)'].max()
        score += row['RMSE (W)'] / error_df_copy['RMSE (W)'].max()
        if row['MAPE_Significant (%)'] != np.inf:
            score += row['MAPE_Significant (%)'] / error_df_copy[error_df_copy['MAPE_Significant (%)'] != np.inf]['MAPE_Significant (%)'].max()
        # Higher is better for R² and accuracy
        score += 1 - (row['R²'] / error_df_copy['R²'].max())
        score += 1 - (row['Accuracy ≤10% (%)'] / error_df_copy['Accuracy ≤10% (%)'].max())
        rankings.append(score)
    
    error_df_copy['Combined_Score'] = rankings
    error_df_copy = error_df_copy.sort_values('Combined_Score')
    
    bars = ax9.barh(error_df_copy['Model'], error_df_copy['Combined_Score'], 
                   color=[colors[models.index(m)] if m in models else 'gray' for m in error_df_copy['Model']], 
                   alpha=0.7)
    ax9.set_xlabel('Combined Error Score (Lower is Better)')
    ax9.set_title('Overall Performance Ranking')
    ax9.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save comprehensive plot
    filepath = 'DISSERTATION_FIGURES/Enhanced_Hybrid_Model_Comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILES Enhanced comprehensive comparison saved: {filepath}")
    return filepath

def main():
    """Main function for improved hybrid model analysis"""
    print("=" * 80)
    print("IMPROVED HYBRID MODEL ANALYSIS")
    print("MSc Dissertation - Solar Forecasting System")
    print("=" * 80)
    
    # Generate realistic data
    df = generate_realistic_solar_data()
    
    # Create improved models
    results_df, feature_cols, rf_model, gb_model = create_improved_models(df)
    
    # Create advanced hybrid model
    hybrid_results = create_advanced_hybrid_model(results_df)
    
    # Calculate improved percentage errors
    error_df = calculate_improved_percentage_errors(hybrid_results)
    
    # Create enhanced visualizations
    comprehensive_plot = create_enhanced_visualizations(hybrid_results, error_df)
    
    # Save results
    hybrid_results.to_csv('DISSERTATION_FIGURES/Improved_Hybrid_Model_Results.csv', index=False)
    error_df.to_csv('DISSERTATION_FIGURES/Improved_Hybrid_Error_Analysis.csv', index=False)
    
    print("\n" + "=" * 80)
    print("IMPROVED HYBRID MODEL ANALYSIS COMPLETED")
    print("=" * 80)
    
    # Display summary
    hybrid_metrics = error_df[error_df['Model'] == 'hybrid_prediction'].iloc[0]
    best_individual = error_df[error_df['Model'] != 'hybrid_prediction'].sort_values('MAE (W)').iloc[0]
    
    print(f"\nHYBRID MODEL PERFORMANCE (Significant Values >10W):")
    print(f"• MAE: {hybrid_metrics['MAE (W)']:.2f} W")
    print(f"• RMSE: {hybrid_metrics['RMSE (W)']:.2f} W")
    print(f"• R²: {hybrid_metrics['R²']:.4f}")
    print(f"• MAPE: {hybrid_metrics['MAPE_Significant (%)']:.2f}%")
    print(f"• Accuracy ≤10%: {hybrid_metrics['Accuracy ≤10% (%)']:.1f}%")
    
    print(f"\nBEST INDIVIDUAL MODEL ({best_individual['Model']}):")
    print(f"• MAE: {best_individual['MAE (W)']:.2f} W")
    print(f"• RMSE: {best_individual['RMSE (W)']:.2f} W")
    print(f"• R²: {best_individual['R²']:.4f}")
    print(f"• MAPE: {best_individual['MAPE_Significant (%)']:.2f}%")
    print(f"• Accuracy ≤10%: {best_individual['Accuracy ≤10% (%)']:.1f}%")
    
    mae_improvement = ((best_individual['MAE (W)'] - hybrid_metrics['MAE (W)']) / 
                      best_individual['MAE (W)'] * 100)
    print(f"\nHYBRID IMPROVEMENT:")
    print(f"• MAE Improvement: {mae_improvement:.2f}% over best individual model")
    
    print(f"\nFILES CREATED:")
    print(f"• Enhanced individual model analysis plots (6 files)")
    print(f"• {comprehensive_plot}")
    print(f"• Improved_Hybrid_Model_Results.csv - Detailed predictions")
    print(f"• Improved_Hybrid_Error_Analysis.csv - Improved error metrics")

if __name__ == "__main__":
    main()
