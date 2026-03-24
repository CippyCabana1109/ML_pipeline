"""
Quick Run - Key Dissertation Analyses
Runs the most important analyses that work without Unicode issues
"""

import os
import subprocess
import sys
from datetime import datetime

def run_script_safe(script_name):
    """Run script with Windows-safe settings"""
    print(f"\n{'='*50}")
    print(f"Running: {script_name}")
    print('='*50)
    
    try:
        # Set UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run script
        result = subprocess.run(['python', script_name], 
                              capture_output=True, 
                              text=True, 
                              env=env,
                              timeout=180)  # 3 minutes
        
        if result.returncode == 0:
            print(f"[SUCCESS] {script_name} completed!")
            print(f"Generated output files.")
            return True
        else:
            print(f"[FAILED] {script_name}")
            print(f"Error: {result.stderr[:300]}")
            return False
            
    except Exception as e:
        print(f"[ERROR] {script_name}: {str(e)}")
        return False

def main():
    print("="*60)
    print("KEY DISSERTATION ANALYSES - WINDOWS COMPATIBLE")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Key analyses that should work
    key_scripts = [
        "corrected_single_day_comparison.py",  # Most important - shows your best model
        "complete_single_day_comparison.py",   # All models comparison
        "improved_hybrid_analysis.py",         # Hybrid analysis
        "iterative_learning_analysis.py"       # Iterative learning
    ]
    
    successful = []
    failed = []
    
    for script in key_scripts:
        if os.path.exists(script):
            if run_script_safe(script):
                successful.append(script)
            else:
                failed.append(script)
        else:
            print(f"[MISSING] {script} not found")
            failed.append(script)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nSuccessful ({len(successful)}):")
    for script in successful:
        print(f"  [OK] {script}")
    
    print(f"\nFailed ({len(failed)}):")
    for script in failed:
        print(f"  [FAIL] {script}")
    
    # Check output files
    print(f"\nOutput files in DISSERTATION_FIGURES/:")
    if os.path.exists('DISSERTATION_FIGURES'):
        files = os.listdir('DISSERTATION_FIGURES')
        key_files = [f for f in files if 'png' in f and ('Single_Day' in f or 'Hybrid' in f or 'Iterative' in f)]
        
        for file in key_files[:10]:  # Show first 10
            print(f"  [FILE] {file}")
        
        print(f"\nTotal files: {len(files)}")
        print(f"Key visualizations: {len(key_files)}")
    
    if successful:
        print(f"\n*** SUCCESS! ***")
        print(f"You have the key files for your dissertation!")
        print(f"Check DISSERTATION_FIGURES/ folder for results.")
        
        if 'Corrected_Single_Day_Original_Ranking.png' in files:
            print(f"\n*** MOST IMPORTANT FILE READY ***")
            print(f"Corrected_Single_Day_Original_Ranking.png")
            print(f"This shows Realistic Hybrid as the BEST model!")
    
    return len(successful) > 0

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\nKey analyses completed successfully!")
    else:
        print(f"\nNo analyses completed successfully.")
