import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import format_metrics_table, save_results

def load_all_results():
    """
    Load results from all models
    """
    print("Loading model results...")
    
    # Initialize results dictionary
    model_results = {}
    
    # Load SARIMAX results (if available)
    try:
        sarimax_df = pd.read_csv('results/sarimax_results.csv')
        model_results['SARIMAX'] = {
            'mae': np.mean(np.abs(sarimax_df['actual'] - sarimax_df['sarimax_predicted'])),
            'rmse': np.sqrt(np.mean((sarimax_df['actual'] - sarimax_df['sarimax_predicted'])**2)),
            'smape': calculate_smape(sarimax_df['actual'], sarimax_df['sarimax_predicted']),
            'r2': calculate_r2(sarimax_df['actual'], sarimax_df['sarimax_predicted']),
            'data': sarimax_df
        }
        print("✅ SARIMAX results loaded")
    except FileNotFoundError:
        print("⚠️  SARIMAX results not found")
    
    # Load XGBoost results (if available)
    try:
        xgb_df = pd.read_csv('results/xgboost_results.csv')
        model_results['XGBoost'] = {
            'mae': np.mean(np.abs(xgb_df['actual'] - xgb_df['xgboost_predicted'])),
            'rmse': np.sqrt(np.mean((xgb_df['actual'] - xgb_df['xgboost_predicted'])**2)),
            'smape': calculate_smape(xgb_df['actual'], xgb_df['xgboost_predicted']),
            'r2': calculate_r2(xgb_df['actual'], xgb_df['xgboost_predicted']),
            'data': xgb_df
        }
        print("✅ XGBoost results loaded")
    except FileNotFoundError:
        print("⚠️  XGBoost results not found")
    
    # Load Hybrid results (if available)
    try:
        hybrid_df = pd.read_csv('results/hybrid_results.csv')
        model_results['Prophet+XGBoost'] = {
            'mae': np.mean(np.abs(hybrid_df['actual'] - hybrid_df['hybrid_predicted'])),
            'rmse': np.sqrt(np.mean((hybrid_df['actual'] - hybrid_df['hybrid_predicted'])**2)),
            'smape': calculate_smape(hybrid_df['actual'], hybrid_df['hybrid_predicted']),
            'r2': calculate_r2(hybrid_df['actual'], hybrid_df['hybrid_predicted']),
            'data': hybrid_df
        }
        print("✅ Hybrid results loaded")
    except FileNotFoundError:
        print("⚠️  Hybrid results not found")
    
    return model_results

def calculate_smape(y_true, y_pred):
    """Calculate sMAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-10, denominator)
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape

def calculate_r2(y_true, y_pred):
    """Calculate R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def create_comparison_table(model_results):
    """
    Create comprehensive comparison table
    """
    print("Creating comparison table...")
    
    # Extract metrics for comparison
    comparison_data = {}
    for model_name, results in model_results.items():
        comparison_data[model_name] = {
            'MAE (W)': results['mae'],
            'RMSE (W)': results['rmse'],
            'sMAPE (%)': results['smape'],
            'R²': results['r2']
        }
    
    # Create formatted table
    comparison_df = pd.DataFrame(comparison_data).T
    
    # Round to appropriate decimal places
    comparison_df['MAE (W)'] = comparison_df['MAE (W)'].round(2)
    comparison_df['RMSE (W)'] = comparison_df['RMSE (W)'].round(2)
    comparison_df['sMAPE (%)'] = comparison_df['sMAPE (%)'].round(2)
    comparison_df['R²'] = comparison_df['R²'].round(4)
    
    # Add ranking
    comparison_df['MAE Rank'] = comparison_df['MAE (W)'].rank()
    comparison_df['RMSE Rank'] = comparison_df['RMSE (W)'].rank()
    comparison_df['sMAPE Rank'] = comparison_df['sMAPE (%)'].rank()
    comparison_df['R² Rank'] = comparison_df['R²'].rank(ascending=False)
    
    # Calculate overall rank (lower is better for errors, higher for R²)
    comparison_df['Overall Rank'] = (
        comparison_df['MAE Rank'] + 
        comparison_df['RMSE Rank'] + 
        comparison_df['sMAPE Rank'] + 
        comparison_df['R² Rank']
    ) / 4
    
    comparison_df = comparison_df.sort_values('Overall Rank')
    
    print("\nModel Performance Comparison:")
    print(comparison_df.round(4))
    
    return comparison_df

def create_visualization_plots(model_results, comparison_df):
    """
    Create comprehensive visualization plots
    """
    print("Creating visualization plots...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Solar PV Forecasting Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Metrics Comparison Bar Chart
    ax1 = axes[0, 0]
    metrics = ['MAE (W)', 'RMSE (W)', 'sMAPE (%)']
    models = list(model_results.keys())
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [model_results[model][metric.lower().replace(' (w)', '').replace(' (%)', '')] for model in models]
        ax1.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Error Metrics')
    ax1.set_title('Error Metrics Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² Comparison
    ax2 = axes[0, 1]
    r2_values = [model_results[model]['r2'] for model in models]
    bars = ax2.bar(models, r2_values, color=['skyblue', 'lightgreen', 'salmon'], alpha=0.8)
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Comparison')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # Plot 3: Actual vs Predicted Comparison (if we have data)
    ax3 = axes[1, 0]
    if len(model_results) > 0:
        # Use the first available model for demonstration
        first_model = list(model_results.keys())[0]
        data = model_results[first_model]['data']
        
        # Plot actual values
        ax3.plot(data['timestamp'][:100], data['actual'][:100], 
                label='Actual', color='black', linewidth=2, alpha=0.8)
        
        # Plot predictions for each model
        colors = ['blue', 'green', 'red']
        for i, (model_name, results) in enumerate(model_results.items()):
            pred_col = [col for col in results['data'].columns if 'predicted' in col][0]
            ax3.plot(results['data']['timestamp'][:100], 
                    results['data'][pred_col][:100],
                    label=model_name, color=colors[i], alpha=0.7, linestyle='--')
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Solar Power (W)')
        ax3.set_title('Actual vs Predicted (Sample Period)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Overall Ranking
    ax4 = axes[1, 1]
    ranks = comparison_df['Overall Rank']
    bars = ax4.barh(comparison_df.index, ranks, color='gold', alpha=0.8)
    ax4.set_xlabel('Overall Rank (lower is better)')
    ax4.set_title('Overall Model Ranking')
    ax4.grid(True, alpha=0.3)
    
    # Add rank labels
    for bar, rank in zip(bars, ranks):
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{rank:.2f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_error_analysis(model_results):
    """
    Create detailed error analysis
    """
    print("Creating error analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Error Analysis Dashboard', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red']
    
    for i, (model_name, results) in enumerate(model_results.items()):
        data = results['data']
        pred_col = [col for col in data.columns if 'predicted' in col][0]
        residuals = data['actual'] - data[pred_col]
        
        # Residual distribution
        ax1 = axes[0, 0]
        ax1.hist(residuals, bins=50, alpha=0.6, color=colors[i], label=model_name)
        ax1.set_xlabel('Residual (W)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Residual Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals over time
        ax2 = axes[0, 1]
        ax2.plot(data['timestamp'][:200], residuals[:200], 
                alpha=0.7, color=colors[i], label=model_name)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Residual (W)')
        ax2.set_title('Residuals Over Time (Sample)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Actual vs Predicted scatter
        ax3 = axes[1, 0]
        ax3.scatter(data['actual'], data[pred_col], 
                   alpha=0.6, color=colors[i], label=model_name, s=10)
        
        # Perfect prediction line
        max_val = max(data['actual'].max(), data[pred_col].max())
        ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        ax3.set_xlabel('Actual (W)')
        ax3.set_ylabel('Predicted (W)')
        ax3.set_title('Actual vs Predicted')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Error by hour of day
        ax4 = axes[1, 1]
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            hourly_error = data.groupby('hour').apply(
                lambda x: np.mean(np.abs(x['actual'] - x[pred_col]))
            )
            ax4.plot(hourly_error.index, hourly_error.values, 
                    marker='o', color=colors[i], label=model_name)
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('MAE (W)')
            ax4.set_title('Error by Hour of Day')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_operational_assessment(comparison_df, model_results):
    """
    Generate operational suitability assessment
    """
    print("Generating operational assessment...")
    
    # Find best model
    best_model = comparison_df.index[0]
    best_metrics = comparison_df.loc[best_model]
    
    # Calculate improvements
    if len(comparison_df) > 1:
        second_best = comparison_df.index[1]
        improvement_mae = ((comparison_df.loc[second_best, 'MAE (W)'] - 
                          comparison_df.loc[best_model, 'MAE (W)']) / 
                          comparison_df.loc[second_best, 'MAE (W)']) * 100
        improvement_rmse = ((comparison_df.loc[second_best, 'RMSE (W)'] - 
                           comparison_df.loc[best_model, 'RMSE (W)']) / 
                           comparison_df.loc[second_best, 'RMSE (W)']) * 100
    else:
        improvement_mae = 0
        improvement_rmse = 0
    
    assessment = f"""
# OPERATIONAL ASSESSMENT REPORT

## Best Performing Model: {best_model}

### Performance Metrics:
- **MAE**: {best_metrics['MAE (W)']:.2f} W
- **RMSE**: {best_metrics['RMSE (W)']:.2f} W  
- **sMAPE**: {best_metrics['sMAPE (%)']:.2f}%
- **R²**: {best_metrics['R²']:.4f}

### Performance Improvements:
- **MAE Improvement**: {improvement_mae:.2f}% over second-best model
- **RMSE Improvement**: {improvement_rmse:.2f}% over second-best model

### Operational Suitability:

#### ✅ Strengths:
- High accuracy with low error metrics
- Strong R² indicating good explanatory power
- Consistent performance across different conditions

#### ⚠️ Considerations:
- Model complexity vs. accuracy trade-off
- Computational requirements for real-time forecasting
- Data requirements and preprocessing needs

#### 🎯 Recommendations for Next-Day Bidding:

1. **Primary Model**: Use {best_model} for next-day bidding forecasts
2. **Confidence Interval**: Implement prediction intervals for risk management
3. **Update Frequency**: Consider daily model retraining with latest data
4. **Backup Model**: Maintain {comparison_df.index[1] if len(comparison_df) > 1 else 'alternative'} as backup
5. **Monitoring**: Implement continuous performance monitoring and alerting

### Expected Business Impact:
- Improved bidding accuracy leading to better market positioning
- Reduced forecast risk and penalty exposure
- Enhanced operational efficiency through automated forecasting
"""
    
    # Save assessment
    with open('results/operational_assessment.md', 'w') as f:
        f.write(assessment)
    
    print("✅ Operational assessment saved to results/operational_assessment.md")
    
    return assessment

def main():
    """
    Main evaluation function
    """
    print("=" * 60)
    print("PHASE 5: MODEL EVALUATION AND COMPARISON")
    print("=" * 60)
    
    # Load all model results
    model_results = load_all_results()
    
    if len(model_results) == 0:
        print("❌ No model results found. Please run previous phases first.")
        return
    
    # Create comparison table
    comparison_df = create_comparison_table(model_results)
    
    # Save comparison table
    comparison_df.to_csv('results/model_comparison.csv')
    print("✅ Comparison table saved to results/model_comparison.csv")
    
    # Create visualizations
    create_visualization_plots(model_results, comparison_df)
    print("✅ Comparison plots saved to results/model_comparison.png")
    
    # Create error analysis
    create_error_analysis(model_results)
    print("✅ Error analysis saved to results/error_analysis.png")
    
    # Generate operational assessment
    assessment = generate_operational_assessment(comparison_df, model_results)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("Files generated:")
    print("- results/model_comparison.csv")
    print("- results/model_comparison.png")
    print("- results/error_analysis.png")
    print("- results/operational_assessment.md")
    
    print(f"\n🏆 Best performing model: {comparison_df.index[0]}")
    print(f"📊 Overall rank: {comparison_df['Overall Rank'].iloc[0]:.2f}")
    
    return comparison_df, model_results

if __name__ == "__main__":
    comparison_df, model_results = main()
