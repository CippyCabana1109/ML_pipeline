"""
Hybrid Top Table - JPEG Format
MSc Dissertation - Solar Forecasting System

Creates a JPEG table with Hybrid model at the TOP of the list (first row)
as requested by the user.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for academic plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def create_hybrid_top_table():
    """Create table with Hybrid model at TOP"""
    print("Creating table with Hybrid model at TOP...")
    
    # Data with Hybrid at TOP (first row) - ORIGINAL MODELS ONLY
    table_data = [
        ['Realistic Hybrid', '578.45', '764.03', '2.51', '0.9990', 'Excellent'],
        ['XGBoost', '771.27', '1018.70', '3.34', '0.9990', 'Excellent'],
        ['Prophet', '7435.28', '9525.91', '32.64', '0.9082', 'Fair'],
        ['SARIMAX', '27774.28', '31491.35', '77.82', '-0.0033', 'Poor']
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Column headers
    col_labels = ['Model', 'MAE (W)', 'RMSE (W)', 'sMAPE (%)', 'R²', 'Performance']
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=col_labels,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color coding - Hybrid at TOP gets special treatment (4 models now)
    for i in range(len(table_data) + 1):  # +1 for header
        for j in range(len(col_labels)):
            if i == 0:  # Header row
                table[(0, j)].set_facecolor('#2E4057')  # Dark blue header
                table[(0, j)].set_text_props(weight='bold', color='white')
            elif i == 1:  # Hybrid model (TOP ROW) - Gold winner
                table[(1, j)].set_facecolor('#FFD700')  # Gold background
                table[(1, j)].set_text_props(weight='bold', color='black')
            elif i == 4:  # Worst performer (SARIMAX)
                table[(4, j)].set_facecolor('#FFCCCB')  # Light red
            else:  # Other models
                table[(i, j)].set_facecolor('#F8F9FA')  # Light gray
    
    # Add title
    plt.title('Solar Forecasting Model Comparison',
             fontsize=16, fontweight='bold', pad=20)
    
    # Add winner annotation for Hybrid
    plt.figtext(0.5, 0.02, '🏆 Realistic Hybrid Model: Best Performance Across All Metrics',
                ha='center', fontsize=12, fontweight='bold', color='#B8860B',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Save as JPEG
    output_path = 'DISSERTATION_FIGURES/Hybrid_Top_Table.jpeg'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='jpeg')
    plt.close()
    
    print(f"FILES Hybrid top table saved as JPEG: {output_path}")
    return output_path

def create_simple_hybrid_top_table():
    """Create a simpler version focused on Hybrid at top"""
    print("Creating simple Hybrid top table...")
    
    # Simplified data - ORIGINAL MODELS ONLY
    simple_data = [
        ['Realistic Hybrid', '578.45', '764.03', '2.51', '0.999'],
        ['XGBoost', '771.27', '1018.70', '3.34', '0.999'],
        ['Prophet', '7435.28', '9525.91', '32.64', '0.908'],
        ['SARIMAX', '27774.28', '31491.35', '77.82', '-0.003']
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Simple headers
    simple_headers = ['Model', 'MAE', 'RMSE', 'sMAPE', 'R²']
    
    # Create table
    table = ax.table(cellText=simple_data,
                    colLabels=simple_headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.35, 0.13, 0.13, 0.13, 0.13])
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    # Color coding - Hybrid at TOP
    for i in range(len(simple_data) + 1):
        for j in range(len(simple_headers)):
            if i == 0:  # Header
                table[(0, j)].set_facecolor('#1F4963')
                table[(0, j)].set_text_props(weight='bold', color='white')
            elif i == 1:  # Hybrid at TOP
                table[(1, j)].set_facecolor('#4CAF50')  # Green for winner
                table[(1, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    # Title
    plt.title('Table 1: Model Comparison',
             fontsize=14, fontweight='bold', pad=15)
    
    # Save as JPEG
    simple_output = 'DISSERTATION_FIGURES/Simple_Hybrid_Top.jpeg'
    plt.savefig(simple_output, dpi=300, bbox_inches='tight', facecolor='white', format='jpeg')
    plt.close()
    
    print(f"FILES Simple Hybrid top table saved: {simple_output}")
    return simple_output

def main():
    """Main function"""
    print("=" * 60)
    print("CREATING HYBRID TOP TABLE - JPEG FORMAT")
    print("=" * 60)
    
    # Create tables with Hybrid at TOP
    hybrid_top = create_hybrid_top_table()
    simple_hybrid = create_simple_hybrid_top_table()
    
    print("\n" + "=" * 60)
    print("HYBRID TOP TABLES COMPLETED")
    print("=" * 60)
    
    print(f"\n📊 TABLES CREATED:")
    print(f"• {hybrid_top} - Full table with Hybrid at TOP")
    print(f"• {simple_hybrid} - Simple table with Hybrid at TOP")
    
    print(f"\n🏆 HYBRID MODEL POSITION:")
    print(f"• Hybrid model is at TOP of list (first row)")
    print(f"• Highlighted as winner with special background")
    print(f"• Best performance clearly shown")
    
    print(f"\n📁 FORMAT:")
    print(f"• JPEG format as requested")
    print(f"• High resolution (300 DPI)")
    print(f"• Ready for dissertation use")

if __name__ == "__main__":
    main()
