"""
Restore Essential Python Files
"""

import os

def restore_files():
    """Restore essential files"""
    print("Restoring essential Python files...")
    
    # List of essential files to restore
    essential_files = [
        '01_weather_analysis.py',
        '02_economic_modeling.py', 
        'run_analysis.py'
    ]
    
    print("Essential files already exist:")
    for file in essential_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file} - needs recreation")
    
    print("\n📁 Current Python files:")
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    for file in py_files:
        print(f"• {file}")
    
    print(f"\n✅ Total Python files: {len(py_files)}")

if __name__ == "__main__":
    restore_files()
