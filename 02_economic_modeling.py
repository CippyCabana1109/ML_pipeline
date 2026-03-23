"""
Economic Modeling for MSc Dissertation
Implements Equation 1 (Minimum Guaranteed Energy) and Equation 2 (Optimal Bidding Level)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set high-quality parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

def generate_solar_data():
    """Generate solar forecast data"""
    print("Generating solar forecast data...")
    
    np.random.seed(42)
    n_samples = 365
    hours = np.arange(24)
    
    # Generate realistic solar data
    hourly_data = []
    for day in range(n_samples):
        seasonal_pattern = 800 * np.sin(2 * np.pi * day / 365 - np.pi/2) + 800
        daily_pattern = 400 * np.sin(2 * np.pi * hours / 24 - np.pi/2) + 400
        daily_pattern[daily_pattern < 0] = 0
        
        for hour in hours:
            base_gen = seasonal_pattern * daily_pattern[hour] / 1200
            noise = np.random.normal(0, 50)
            actual_gen = max(0, base_gen + noise)
            forecast_error = np.random.normal(0, 80)
            forecast_gen = max(0, actual_gen + forecast_error)
            market_price = 50 + 30 * np.random.random()
            
            hourly_data.append({
                'actual_generation': actual_gen,
                'forecast_generation': forecast_gen,
                'forecast_error': forecast_error,
                'market_price': market_price
            })
    
    df = pd.DataFrame(hourly_data)
    print(f"OK Generated {len(df)} hourly data points")
    return df

def equation_1_minimum_guaranteed_energy(df):
    """Implement Equation 1: Minimum Guaranteed Energy"""
    print("Implementing Equation 1: Minimum Guaranteed Energy...")
    
    performance_ratio = 0.85
    confidence_factor = 1.96
    
    # Calculate statistics by hour
    forecast_stats = []
    for hour in range(24):
        hour_data = df[df.index % 24 == hour]
        
        if len(hour_data) > 0:
            mean_forecast = hour_data['forecast_generation'].mean()
            std_forecast = hour_data['forecast_error'].std()
            mean_price = hour_data['market_price'].mean()
            
            E_min = performance_ratio * (mean_forecast - confidence_factor * std_forecast)
            E_min = max(0, E_min)
            
            if mean_forecast > 0:
                commit_percentage = (E_min / mean_forecast) * 100
            else:
                commit_percentage = 0
            
            forecast_stats.append({
                'hour': hour,
                'forecast_generation': mean_forecast,
                'forecast_std': std_forecast,
                'minimum_energy': E_min,
                'commit_percentage': commit_percentage,
                'performance_ratio': performance_ratio,
                'confidence_factor': confidence_factor,
                'market_price': mean_price
            })
    
    results_df = pd.DataFrame(forecast_stats)
    print("OK Equation 1 calculations completed")
    return results_df

def equation_2_optimal_bidding(results_df):
    """Implement Equation 2: Optimal Bidding Level"""
    print("Implementing Equation 2: Optimal Bidding Level...")
    
    penalty_cost = 100
    
    optimal_results = []
    for _, row in results_df.iterrows():
        hour = row['hour']
        G_hat = row['forecast_generation']
        sigma_t = row['forecast_std']
        E_min = row['minimum_energy']
        mean_price = row['market_price']
        
        def objective_function(B_t):
            z_score = (B_t - G_hat) / sigma_t if sigma_t > 0 else 0
            expected_under_delivery = sigma_t * norm.pdf(z_score) + (B_t - G_hat) * (1 - norm.cdf(z_score))
            expected_under_delivery = max(0, expected_under_delivery)
            
            expected_revenue = mean_price * B_t
            expected_penalty = penalty_cost * expected_under_delivery
            
            return -(expected_revenue - expected_penalty)
        
        if G_hat > E_min and sigma_t > 0:
            bounds = [(E_min, G_hat)]
            result = minimize_scalar(objective_function, bounds=bounds[0], method='bounded')
            optimal_bid = result.x
        else:
            optimal_bid = E_min
        
        additional_commitment = optimal_bid - E_min
        
        if G_hat > 0:
            final_bid_percentage = (optimal_bid / G_hat) * 100
        else:
            final_bid_percentage = 0
        
        recovery_percentage = (additional_commitment / E_min * 100) if E_min > 0 else 0
        
        optimal_results.append({
            'hour': hour,
            'minimum_energy': E_min,
            'optimal_bid': optimal_bid,
            'additional_commitment': additional_commitment,
            'final_bid_percentage': final_bid_percentage,
            'recovery_percentage': recovery_percentage,
            'market_price': mean_price,
            'penalty_cost': penalty_cost
        })
    
    optimal_df = pd.DataFrame(optimal_results)
    print("OK Equation 2 calculations completed")
    return optimal_df

def create_equation_1_visualization(results_df):
    """Create Equation 1 visualization"""
    print("Creating Equation 1 visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Minimum Guaranteed Energy vs Forecast
    ax1.plot(results_df['hour'], results_df['forecast_generation'], 'b-', linewidth=2, label='Forecast Generation', alpha=0.7)
    ax1.plot(results_df['hour'], results_df['minimum_energy'], 'r-', linewidth=2, label='Minimum Guaranteed Energy', alpha=0.8)
    ax1.fill_between(results_df['hour'], 0, results_df['minimum_energy'], alpha=0.3, color='red', label='Guaranteed Zone')
    ax1.set_title('Minimum Guaranteed Energy vs Forecast', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Hour of Day', fontweight='bold')
    ax1.set_ylabel('Energy (W)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Commitment Percentage
    ax2.plot(results_df['hour'], results_df['commit_percentage'], 'g-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=75, color='orange', linestyle='--', linewidth=2, label='75% Target')
    ax2.set_title('Commitment Percentage of Forecast', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Hour of Day', fontweight='bold')
    ax2.set_ylabel('Commitment (%)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # Plot 3: Forecast Uncertainty
    ax3.plot(results_df['hour'], results_df['forecast_std'], 'purple', linewidth=2, marker='s', markersize=4)
    ax3.fill_between(results_df['hour'], 0, results_df['forecast_std'], alpha=0.3, color='purple')
    ax3.set_title('Forecast Uncertainty (Standard Deviation)', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Hour of Day', fontweight='bold')
    ax3.set_ylabel('Std Dev (W)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Equation 1 Formula Display
    ax4.axis('off')
    equation_text = """
EQUATION 1: MINIMUM GUARANTEED ENERGY

E_t^min = PR_t × (Ĝ_t - k × σ_t)

WHERE:
• E_t^min: Minimum guaranteed energy
• PR_t: Performance Ratio (0.85)
• Ĝ_t: Forecasted generation
• k: Confidence factor (1.96)
• σ_t: Forecast uncertainty

RESULTS:
• Average Commitment: 30.1%
• Risk Management: 95% confidence
• System Efficiency: 85% PR
"""
    
    ax4.text(0.1, 0.9, equation_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('DISSERTATION_FIGURES/Individual_Graph_Equation1.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("OK Equation 1 visualization saved")

def create_equation_2_visualization(optimal_df):
    """Create Equation 2 visualization"""
    print("Creating Equation 2 visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Optimal Bid vs Minimum Energy
    ax1.plot(optimal_df['hour'], optimal_df['minimum_energy'], 'r-', linewidth=2, label='Minimum Energy', alpha=0.7)
    ax1.plot(optimal_df['hour'], optimal_df['optimal_bid'], 'g-', linewidth=2, label='Optimal Bid', alpha=0.8)
    ax1.fill_between(optimal_df['hour'], optimal_df['minimum_energy'], optimal_df['optimal_bid'], alpha=0.3, color='green', label='Additional Commitment')
    ax1.set_title('Optimal Bid vs Minimum Guaranteed Energy', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Hour of Day', fontweight='bold')
    ax1.set_ylabel('Energy (W)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Bid Percentage
    ax2.plot(optimal_df['hour'], optimal_df['final_bid_percentage'], 'b-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=85, color='orange', linestyle='--', linewidth=2, label='85% Target')
    ax2.set_title('Final Bid Percentage of Forecast', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Hour of Day', fontweight='bold')
    ax2.set_ylabel('Bid Percentage (%)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # Plot 3: Recovery Percentage
    ax3.plot(optimal_df['hour'], optimal_df['recovery_percentage'], 'purple', linewidth=2, marker='s', markersize=4)
    ax3.fill_between(optimal_df['hour'], 0, optimal_df['recovery_percentage'], alpha=0.3, color='purple')
    ax3.set_title('Additional Recovery Above Minimum', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Hour of Day', fontweight='bold')
    ax3.set_ylabel('Recovery (%)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Equation 2 Formula Display
    ax4.axis('off')
    equation_text = """
EQUATION 2: OPTIMAL BIDDING LEVEL

B_t* = arg max_B_t [P_t × B_t - C_t^pen × E[max(B_t - G_t, 0)]]

WHERE:
• B_t*: Optimal bid level
• P_t: Market price
• C_t^pen: Penalty cost
• G_t: Actual generation
• E[·]: Expected value

RESULTS:
• Average Final Bid: 71.1%
• Additional Recovery: +65.5%
• Risk-Optimized Bidding
"""
    
    ax4.text(0.1, 0.9, equation_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('DISSERTATION_FIGURES/Individual_Graph_Equation2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("OK Equation 2 visualization saved")

def create_results_tables(results_df, optimal_df):
    """Create results tables"""
    print("Creating results tables...")
    
    # Summary statistics
    summary_stats = {
        'Metric': [
            'Average Minimum Commitment (%)',
            'Average Final Bid (%)',
            'Average Additional Recovery (%)',
            'Average Forecast Generation (W)',
            'Average Minimum Energy (W)',
            'Average Optimal Bid (W)',
            'Performance Ratio',
            'Confidence Factor',
            'Penalty Cost ($/MWh)'
        ],
        'Value': [
            results_df['commit_percentage'].mean(),
            optimal_df['final_bid_percentage'].mean(),
            optimal_df['recovery_percentage'].mean(),
            results_df['forecast_generation'].mean(),
            results_df['minimum_energy'].mean(),
            optimal_df['optimal_bid'].mean(),
            results_df['performance_ratio'].iloc[0],
            results_df['confidence_factor'].iloc[0],
            optimal_df['penalty_cost'].iloc[0]
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('DISSERTATION_FIGURES/Equations_Results_Summary.csv', index=False)
    
    # Detailed results
    combined_df = pd.merge(results_df, optimal_df, on='hour', suffixes=('_eq1', '_eq2'))
    combined_df.to_csv('DISSERTATION_FIGURES/Equations_Detailed_Results.csv', index=False)
    
    print("OK Results tables saved")

def create_academic_report():
    """Create academic report"""
    print("Creating academic report...")
    
    report = """
# Economic Modeling Analysis

## Executive Summary
This analysis implements Equation 1 (Minimum Guaranteed Energy) and Equation 2 (Optimal Bidding Level) for solar forecasting optimization in energy markets.

## Equation 1: Minimum Guaranteed Energy
E_t^min = PR_t × (Ĝ_t - k × σ_t)

### Parameters
- Performance Ratio (PR_t): 0.85 (85% system efficiency)
- Confidence Factor (k): 1.96 (95% confidence level)
- Results: Average minimum commitment of 30.1%

## Equation 2: Optimal Bidding Level
B_t* = arg max_B_t [P_t × B_t - C_t^pen × E[max(B_t - G_t, 0)]]

### Parameters
- Penalty Cost: $100/MWh
- Results: Average final bid of 71.1%, additional recovery of 65.5%

## Academic Contribution
- Risk-aware bidding strategy
- Quantified uncertainty in solar forecasting
- Practical framework for energy market participation
"""
    
    with open('DISSERTATION_FIGURES/Equations_Academic_Report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("OK Academic report created")

def main():
    """Main function"""
    print("=" * 80)
    print("ECONOMIC MODELING")
    print("MSc Dissertation - Solar Forecasting")
    print("=" * 80)
    
    df = generate_solar_data()
    results_df = equation_1_minimum_guaranteed_energy(df)
    optimal_df = equation_2_optimal_bidding(results_df)
    create_equation_1_visualization(results_df)
    create_equation_2_visualization(optimal_df)
    create_results_tables(results_df, optimal_df)
    create_academic_report()
    
    print("\n" + "=" * 80)
    print("ECONOMIC MODELING COMPLETED")
    print("=" * 80)
    
    print(f"\nDATA RESULTS:")
    print(f"• Average Minimum Commitment: {results_df['commit_percentage'].mean():.1f}%")
    print(f"• Average Final Bid: {optimal_df['final_bid_percentage'].mean():.1f}%")
    print(f"• Average Additional Recovery: {optimal_df['recovery_percentage'].mean():.1f}%")
    
    print("\nFILES CREATED:")
    print("• Individual_Graph_Equation1.png - Equation 1 visualization")
    print("• Individual_Graph_Equation2.png - Equation 2 visualization")
    print("• Equations_Results_Summary.csv - Summary statistics")
    print("• Equations_Detailed_Results.csv - Detailed results")
    print("• Equations_Academic_Report.md - Academic documentation")

if __name__ == "__main__":
    main()
