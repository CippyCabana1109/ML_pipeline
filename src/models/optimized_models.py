import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import time
warnings.filterwarnings('ignore')

class FastSolarForecaster:
    """
    Optimized solar forecasting models with speed and efficiency focus
    """
    
    def __init__(self):
        self.results = {}
        self.training_time = {}
        
    def fast_sarimax(self, train_data, exog_train, test_data, exog_test):
        """
        Fast SARIMAX with pre-optimized parameters
        """
        start_time = time.time()
        print("Running Fast SARIMAX...")
        
        # Use pre-optimized parameters for speed
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 24)
        
        # Downsample to 3-hour intervals for speed
        train_down = train_data.resample('3h').mean()
        test_down = test_data.resample('3h').mean()
        exog_train_down = exog_train.resample('3h').mean()
        exog_test_down = exog_test.resample('3h').mean()
        
        # Fit model with optimized settings
        model = SARIMAX(
            train_down.dropna(),
            exog=exog_train_down.dropna(),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=True
        )
        
        results = model.fit(disp=False, maxiter=50)  # Limit iterations for speed
        
        # Forecast
        forecast = results.get_forecast(
            steps=len(test_down.dropna()), 
            exog=exog_test_down.dropna()
        )
        
        # Upsample back to hourly
        predictions = forecast.predicted_mean
        predictions_hourly = np.repeat(predictions, 3)[:len(test_data)]
        
        training_time = time.time() - start_time
        self.training_time['SARIMAX'] = training_time
        
        print(f"SARIMAX completed in {training_time:.1f}s")
        
        return predictions_hourly
    
    def fast_xgboost(self, X_train, y_train, X_test):
        """
        Fast XGBoost with optimized parameters
        """
        start_time = time.time()
        print("Running Fast XGBoost...")
        
        # Optimized XGBoost parameters for speed
        model = xgb.XGBRegressor(
            n_estimators=50,  # Reduced from 100
            max_depth=4,       # Reduced from 6
            learning_rate=0.2,   # Increased for faster convergence
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,          # Use all cores
            tree_method='hist'     # Faster histogram method
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        training_time = time.time() - start_time
        self.training_time['XGBoost'] = training_time
        
        print(f"✅ XGBoost completed in {training_time:.1f}s")
        
        return predictions
    
    def fast_prophet_hybrid(self, train_df, test_df):
        """
        Fast Prophet + XGBoost Hybrid
        """
        start_time = time.time()
        print("🚀 Running Fast Prophet+XGBoost Hybrid...")
        
        # Prepare Prophet data
        train_prophet = train_df[['timestamp', 'solar_power_w']].rename(
            columns={'timestamp': 'ds', 'solar_power_w': 'y'}
        )
        
        # Fast Prophet with reduced parameters
        prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,  # Disabled for speed
            yearly_seasonality=False,  # Disabled for speed
            changepoint_prior_scale=0.1,
            n_changepoints=10,         # Reduced from default
            mcmc_samples=0,            # Disabled for speed
            interval_width=0.8,
            uncertainty_samples=0           # Disabled for speed
        )
        
        prophet_model.fit(train_prophet)
        
        # Create future dataframe
        future = pd.concat([
            train_prophet[['ds']], 
            test_df[['timestamp']].rename(columns={'timestamp': 'ds'})
        ])
        
        # Fast prediction
        prophet_forecast = prophet_model.predict(future)
        
        # Extract Prophet predictions
        prophet_train_pred = prophet_forecast['yhat'][:len(train_df)].values
        prophet_test_pred = prophet_forecast['yhat'][len(train_df):len(train_df)+len(test_df)].values
        
        # Fast XGBoost on residuals
        feature_cols = ['irradiance', 'temperature', 'humidity', 'hour', 'day_of_week']
        X_train = train_df[feature_cols]
        y_train = train_df['solar_power_w'] - prophet_train_pred
        X_test = test_df[feature_cols]
        
        # Fast XGBoost for residuals
        residual_model = xgb.XGBRegressor(
            n_estimators=30,      # Even fewer for residuals
            max_depth=3,
            learning_rate=0.3,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        residual_model.fit(X_train, y_train)
        residual_pred = residual_model.predict(X_test)
        
        # Combine predictions
        hybrid_pred = prophet_test_pred + residual_pred
        
        training_time = time.time() - start_time
        self.training_time['Hybrid'] = training_time
        
        print(f"✅ Prophet+XGBoost Hybrid completed in {training_time:.1f}s")
        
        return hybrid_pred
    
    def evaluate_all_models(self, train_df, test_df):
        """
        Run all models and collect results
        """
        print("🏁 Starting Optimized Model Evaluation...")
        print("=" * 60)
        
        # Prepare data for all models
        feature_cols = ['irradiance', 'temperature', 'humidity', 'hour', 'day_of_week', 'month']
        
        X_train = train_df[feature_cols]
        y_train = train_df['solar_power_w']
        X_test = test_df[feature_cols]
        y_test = test_df['solar_power_w']
        
        # Prepare SARIMAX data
        train_sarimax = train_df.set_index('timestamp').resample('h').mean()
        test_sarimax = test_df.set_index('timestamp').resample('h').mean()
        exog_train = train_sarimax[['irradiance', 'temperature', 'humidity']]
        exog_test = test_sarimax[['irradiance', 'temperature', 'humidity']]
        target_train = train_sarimax['solar_power_w']
        target_test = test_sarimax['solar_power_w']
        
        # Run all models
        results = {}
        
        try:
            sarimax_pred = self.fast_sarimax(target_train, exog_train, target_test, exog_test)
            results['SARIMAX'] = sarimax_pred[:len(y_test)]
        except Exception as e:
            print(f"⚠️ SARIMAX failed: {e}")
            results['SARIMAX'] = np.zeros(len(y_test))
        
        try:
            xgb_pred = self.fast_xgboost(X_train, y_train, X_test)
            results['XGBoost'] = xgb_pred
        except Exception as e:
            print(f"⚠️ XGBoost failed: {e}")
            results['XGBoost'] = np.zeros(len(y_test))
        
        try:
            hybrid_pred = self.fast_prophet_hybrid(train_df, test_df)
            results['Prophet+XGBoost'] = hybrid_pred
        except Exception as e:
            print(f"⚠️ Hybrid failed: {e}")
            results['Prophet+XGBoost'] = np.zeros(len(y_test))
        
        # Calculate metrics for all models
        model_metrics = {}
        for model_name, predictions in results.items():
            if len(predictions) == len(y_test):
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)
                smape = self.calculate_smape(y_test, predictions)
                
                model_metrics[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2,
                    'sMAPE': smape,
                    'predictions': predictions
                }
        
        self.results = model_metrics
        return model_metrics
    
    def calculate_smape(self, y_true, y_pred):
        """Calculate sMAPE"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        denominator = np.where(denominator == 0, 1e-10, denominator)
        smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
        return smape
    
    def create_results_dataframe(self, test_df):
        """
        Create comprehensive results dataframe
        """
        if not self.results:
            print("❌ No results to save")
            return None
        
        # Create base dataframe with timestamps and actual values
        results_df = pd.DataFrame({
            'timestamp': test_df['timestamp'],
            'actual': test_df['solar_power_w']
        })
        
        # Add predictions from each model
        for model_name, metrics in self.results.items():
            results_df[f'{model_name.lower().replace("+", "_").replace(" ", "_")}_predicted'] = metrics['predictions']
        
        # Calculate error columns
        for model_name in self.results.keys():
            pred_col = f'{model_name.lower().replace("+", "_").replace(" ", "_")}_predicted'
            results_df[f'{model_name.lower().replace("+", "_").replace(" ", "_")}_error'] = \
                np.abs(results_df['actual'] - results_df[pred_col])
        
        return results_df
    
    def save_comprehensive_results(self, test_df, save_path='results/optimized_model_results.csv'):
        """
        Save all results to CSV
        """
        results_df = self.create_results_dataframe(test_df)
        if results_df is not None:
            results_df.to_csv(save_path, index=False)
            print(f"✅ Results saved to {save_path}")
            
            # Also save summary metrics
            summary_data = []
            for model_name, metrics in self.results.items():
                summary_data.append({
                    'Model': model_name,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R²': metrics['R²'],
                    'sMAPE': metrics['sMAPE'],
                    'Training_Time_s': self.training_time.get(model_name, 0)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = save_path.replace('.csv', '_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"✅ Summary saved to {summary_path}")
            
            return results_df, summary_df
        
        return None, None
    
    def print_speed_summary(self):
        """
        Print training speed summary
        """
        print("\n⚡ TRAINING SPEED SUMMARY:")
        print("=" * 40)
        total_time = sum(self.training_time.values())
        for model, time_taken in self.training_time.items():
            percentage = (time_taken / total_time) * 100
            print(f"{model}: {time_taken:.1f}s ({percentage:.1f}%)")
        print(f"Total Time: {total_time:.1f}s")
        print("=" * 40)

def main():
    """
    Main optimized evaluation function
    """
    print("OPTIMIZED SOLAR FORECASTING PIPELINE")
    print("=" * 60)
    
    # Load data
    try:
        train_df = pd.read_csv('data/train_final.csv')
        test_df = pd.read_csv('data/test_final.csv')
        
        # Convert timestamps
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        
        print(f"✅ Data loaded: {len(train_df)} training, {len(test_df)} test records")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Initialize fast forecaster
    forecaster = FastSolarForecaster()
    
    # Run all optimized models
    start_total = time.time()
    model_metrics = forecaster.evaluate_all_models(train_df, test_df)
    total_time = time.time() - start_total
    
    # Print speed summary
    forecaster.print_speed_summary()
    
    # Save comprehensive results
    results_df, summary_df = forecaster.save_comprehensive_results(test_df)
    
    if results_df is not None:
        # Find best model
        best_model = min(model_metrics.keys(), key=lambda x: model_metrics[x]['MAE'])
        best_mae = model_metrics[best_model]['MAE']
        
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"📊 BEST MAE: {best_mae:.2f} W")
        print(f"⏱️ TOTAL TIME: {total_time:.1f}s")
        
        # Create quick visualization
        plt.figure(figsize=(15, 8))
        plt.plot(test_df['timestamp'][:168], test_df['solar_power_w'][:168], 
                 label='Actual', color='black', linewidth=2)
        
        colors = ['blue', 'green', 'red']
        for i, (model_name, metrics) in enumerate(model_metrics.items()):
            plt.plot(test_df['timestamp'][:168], metrics['predictions'][:168], 
                    label=model_name, color=colors[i], alpha=0.7, linestyle='--')
        
        plt.title('Optimized Solar Forecasting Comparison (First Week)')
        plt.xlabel('Date')
        plt.ylabel('Solar Power (W)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/optimized_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to results/optimized_comparison.png")
    
    print("\n🎉 OPTIMIZED EVALUATION COMPLETE!")
    print("Files generated:")
    print("- results/optimized_model_results.csv")
    print("- results/optimized_model_results_summary.csv") 
    print("- results/optimized_comparison.png")
    
    return model_metrics

if __name__ == "__main__":
    results = main()
