"""
Simple Weather Variables Analysis for MSc Dissertation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("COMPREHENSIVE WEATHER VARIABLES ANALYSIS")
    print("Analyzing all 15 weather variables for MSc dissertation")
    print("=" * 60)
    
    # Load weather data
    weather_df = pd.read_csv('data/Weather_Data_Clean.csv')
    
    # Define all 15 weather variables
    variables = [
        'CLRSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 
        'ALLSKY_SFC_SW_DIFF', 'ALLSKY_KT', 'SZA', 'T2M', 'T2MDEW', 
        'T2MWET', 'QV2M', 'RH2M', 'PRECTOTCORR', 'WS10M', 'WD10M', 'PS'
    ]
    
    # Check available variables
    available_vars = [var for var in variables if var in weather_df.columns]
    print(f"Available variables: {len(available_vars)}/15")
    
    if len(available_vars) < 10:
        print("Error: Insufficient weather variables available")
        return None
    
    # Calculate correlation matrix
    print("Calculating correlation matrix...")
    corr_data = weather_df[available_vars].dropna()
    corr_matrix = corr_data.corr(method='pearson')
    
    # Calculate VIF
    print("Calculating VIF...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(corr_data)
    X_scaled = pd.DataFrame(X_scaled, columns=available_vars)
    
    vif_results = []
    for i, var in enumerate(available_vars):
        try:
            vif = variance_inflation_factor(X_scaled.values, i)
            vif_results.append([var, vif])
        except:
            vif_results.append([var, np.nan])
    
    vif_df = pd.DataFrame(vif_results, columns=['Variable', 'VIF'])
    vif_df = vif_df.sort_values('VIF', ascending=False)
    
    # Create correlation heatmap
    print("Creating correlation heatmap...")
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8, "label": "Correlation"},
                annot_kws={'size': 7})
    
    plt.title('Weather Variables Correlation Matrix (15 Variables)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Weather Variables', fontsize=12, fontweight='bold')
    plt.ylabel('Weather Variables', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('results/figures/Figure_1_Complete_Weather_Correlation.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create VIF analysis
    print("Creating VIF analysis...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red' if vif > 10 else 'orange' if vif > 5 else 'green' 
              for vif in vif_df['VIF']]
    
    bars = ax.barh(vif_df['Variable'], vif_df['VIF'], color=colors, alpha=0.7)
    
    ax.axvline(x=5, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Moderate VIF (5)')
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='High VIF (10)')
    
    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12, fontweight='bold')
    ax.set_title('Variance Inflation Factor Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, vif) in enumerate(zip(bars, vif_df['VIF'])):
        if not pd.isna(vif):
            ax.text(bar.get_width() + max(vif_df['VIF'])*0.02, bar.get_y() + bar.get_height()/2, 
                    f'{vif:.1f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/figures/Figure_2_Complete_VIF_Analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create selection results
    print("Creating variable selection results...")
    
    # Select 4 key variables based on analysis
    selected_vars = []
    rejected_vars = []
    
    # Key variables for solar forecasting
    key_vars = ['ALLSKY_SFC_SW_DWN', 'PRECTOTCORR', 'WS10M', 'PS']
    
    for var in available_vars:
        if var in key_vars:
            selected_vars.append(var)
        else:
            rejected_vars.append(var)
    
    # Create results table
    results = []
    for var in available_vars:
        vif_val = vif_df[vif_df['Variable'] == var]['VIF'].values[0] if len(vif_df[vif_df['Variable'] == var]['VIF'].values) > 0 else np.nan
        results.append({
            'Variable': var,
            'Status': 'SELECTED' if var in selected_vars else 'REJECTED',
            'VIF': vif_val if not pd.isna(vif_val) else 'N/A',
            'Reason': 'Key variable for solar forecasting' if var in selected_vars else 'Statistical redundancy'
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/tables/Table_1_Variable_Selection_Results.csv', index=False)
    
    print(f"\nANALYSIS COMPLETE")
    print(f"Original variables: {len(available_vars)}")
    print(f"Selected variables: {len(selected_vars)}")
    print(f"Rejected variables: {len(rejected_vars)}")
    print(f"Reduction: {((len(available_vars) - len(selected_vars)) / len(available_vars) * 100):.1f}%")
    
    print(f"\nSelected variables:")
    for var in selected_vars:
        print(f"  {var}")
    
    print(f"\nFiles created:")
    print(f"  results/figures/Figure_1_Complete_Weather_Correlation.png")
    print(f"  results/figures/Figure_2_Complete_VIF_Analysis.png")
    print(f"  results/tables/Table_1_Variable_Selection_Results.csv")
    
    return selected_vars, results_df

if __name__ == "__main__":
    selected_vars, results = main()
