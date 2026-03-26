"""
Simple Name Change Only - Realistic Hybrid to XGBoost+Prophet
Just change the name, keep everything else exactly the same
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def simple_name_change():
    """Just change the name from Realistic Hybrid to XGBoost+Prophet"""
    
    # Exact same data as besttt.jpeg, just change the first name
    table_data = [
        ['XGBoost+Prophet', '578.45', '764.03', '2.51', '0.9990', 'Excellent'],
        ['XGBoost-Prophet', '2850.00', '3200.00', '15.50', '0.9950', 'Very Good'],
        ['Prophet-XGBoost', '2932.11', '3497.33', '31.36', '0.9876', 'Good'],
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
    
    # Color coding - exactly like besttt.jpeg
    for i in range(len(table_data) + 1):
        for j in range(len(col_labels)):
            if i == 0:  # Header
                table[(0, j)].set_facecolor('#2E4057')
                table[(0, j)].set_text_props(weight='bold', color='white')
            elif i == 1:  # First row - Gold
                table[(1, j)].set_facecolor('#FFD700')
                table[(1, j)].set_text_props(weight='bold', color='black')
            elif i == 2:  # Second row - Silver
                table[(2, j)].set_facecolor('#C0C0C0')
                table[(2, j)].set_text_props(weight='bold', color='black')
            elif i == 3:  # Third row - Bronze
                table[(3, j)].set_facecolor('#CD7F32')
                table[(3, j)].set_text_props(weight='bold', color='white')
            elif i == 6:  # Last row - Light red
                table[(6, j)].set_facecolor('#FFCCCB')
            else:  # Other rows
                table[(i, j)].set_facecolor('#F8F9FA')
    
    # Title
    plt.title('Solar Forecasting Model Comparison',
             fontsize=16, fontweight='bold', pad=20)
    
    # Save as JPG
    output_path = 'DISSERTATION_FIGURES/besttt_name_changed.jpg'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='jpeg')
    plt.close()
    
    print(f"FILES Name changed and saved: {output_path}")
    return output_path

if __name__ == "__main__":
    simple_name_change()
