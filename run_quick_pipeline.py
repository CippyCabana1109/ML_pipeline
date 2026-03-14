"""
Quick Solar Forecasting Pipeline
Optimized for speed with essential analysis
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

def run_phase(phase_name, script_path):
    """Execute a pipeline phase"""
    print(f"\n{'='*50}")
    print(f"RUNNING {phase_name}")
    print('='*50)
    
    try:
        start_time = time.time()
        result = subprocess.run(['python', script_path], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        end_time = time.time()
        
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        
        phase_time = end_time - start_time
        print(f"✅ {phase_name} completed in {phase_time:.1f}s")
        return True, phase_time
    except subprocess.CalledProcessError as e:
        print(f"❌ {phase_name} failed:")
        print(e.stderr)
        return False, 0

def main():
    """Run quick pipeline"""
    print("QUICK SOLAR FORECASTING PIPELINE")
    print("=" * 50)
    print("Optimized for speed with essential analysis")
    print("=" * 50)
    
    # Check data directory
    if not Path('data').exists():
        print("❌ Data directory not found")
        return
    
    # Define quick pipeline phases
    phases = [
        ("Data Processing", "src/data_processing/data_processor.py"),
        ("Fast Models", "src/models/fast_models.py")
    ]
    
    # Execute pipeline
    completed = []
    total_time = 0
    
    for phase_name, script_path in phases:
        if Path(script_path).exists():
            success, phase_time = run_phase(phase_name, script_path)
            if success:
                completed.append(phase_name)
                total_time += phase_time
            else:
                print(f"⚠️ Phase failed: {phase_name}")
                break
        else:
            print(f"⚠️ Script not found: {script_path}")
    
    # Summary
    print(f"\n{'='*50}")
    print("QUICK PIPELINE SUMMARY")
    print('='*50)
    print(f"Completed: {len(completed)}/{len(phases)} phases")
    print(f"Total time: {total_time:.1f}s")
    
    for phase in completed:
        print(f"✅ {phase}")
    
    if len(completed) == len(phases):
        print("\n🚀 QUICK ANALYSIS COMPLETE!")
        print("📁 Results:")
        print("   - results/simple_model_results.csv")
        print("   - results/simple_summary.csv")
        print("   - results/simple_comparison.png")
        print("\n🏆 To find best model:")
        print("   1. Open results/simple_summary.csv")
        print("   2. Look for lowest MAE value")
    else:
        print(f"\n⚠️ Pipeline incomplete")

if __name__ == "__main__":
    main()
