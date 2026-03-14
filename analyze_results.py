"""
Analyze Results - Professional Model Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results():
    """Load available results"""
    print("Loading results...")
    
    # Try simple results first
    if Path('results/simple_summary.csv').exists():
        df = pd.read_csv('results/simple_summary.csv')
        print("Found simple results")
        return df
    
    # Try comprehensive results
    if Path('results/tables/final_model_comparison.csv').exists():
        df = pd.read_csv('results/tables/final_model_comparison.csv')
        print("Found comprehensive results")
        return df
    
    print("No results found")
    return None

def analyze_and_plot(results_df):
    """Analyze and visualize results"""
    if results_df is None:
        return
    
    print("Analyzing results...")
    
    # Find best model
    best_idx = results_df['MAE'].idxmin()
    best_model = results_df.loc[best_idx, 'Model']
    best_mae = results_df.loc[best_idx, 'MAE']
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance comparison
    models = results_df['Model']
    mae_values = results_df['MAE']
    
    bars = ax1.bar(models, mae_values, color=['gold', 'silver', 'brown'])
    ax1.set_ylabel('MAE (W)')
    ax1.set_title('Model Performance Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mae in zip(bars, mae_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{mae:.1f}', ha='center', va='bottom')
    
    # Best model highlight
    ax2.axis('off')
    ax2.text(0.5, 0.7, f'🏆 BEST MODEL\n\n{best_model}\n\nMAE: {best_mae:.2f}W',
             fontsize=16, fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/figures/model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"📊 BEST MAE: {best_mae:.2f}W")
    
    return best_model, best_mae

def main():
    """Main analysis function"""
    print("RESULTS ANALYSIS")
    print("=" * 40)
    
    results_df = load_results()
    best_model, best_mae = analyze_and_plot(results_df)
    
    if best_model:
        print(f"\n✅ RECOMMENDATION: Use {best_model} for production")
        print(f"📈 EXPECTED ACCURACY: MAE {best_mae:.2f}W")

if __name__ == "__main__":
    main()
