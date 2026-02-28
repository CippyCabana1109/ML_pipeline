"""
Complete pipeline for solar PV forecasting model comparison
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src directory to path
sys.path.append('src')

def run_phase(phase_name, script_path):
    """
    Run a single phase of the pipeline
    
    Args:
        phase_name: Name of the phase
        script_path: Path to the Python script
    """
    print(f"\n{'='*60}")
    print(f"RUNNING {phase_name}")
    print('='*60)
    
    try:
        result = subprocess.run(['python', script_path], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        print(f"✅ {phase_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {phase_name} failed with error:")
        print(e.stderr)
        return False

def main():
    """
    Run the complete pipeline
    """
    print("🚀 Starting Solar PV Forecasting Pipeline")
    print("This will run all phases sequentially:")
    print("1. Data Ingestion & Preprocessing")
    print("2. SARIMAX Baseline Model")
    print("3. XGBoost Model")
    print("4. Prophet+XGBoost Hybrid")
    print("5. Model Evaluation & Comparison")
    
    # Check if data directory exists and has required files
    data_dir = Path('data')
    if not data_dir.exists():
        print("❌ Data directory not found. Please ensure data files are in place.")
        return
    
    # Define pipeline phases
    phases = [
        ("Phase 1: Data Ingestion", "src/process_training_dataset.py"),
        ("Phase 2: SARIMAX Model", "src/phase2_sarimax.py"),
        ("Phase 3: XGBoost Model", "src/phase3_xgboost.py"),
        ("Phase 4: Prophet+XGBoost Hybrid", "src/phase4_hybrid.py"),
        ("Phase 5: Model Evaluation", "src/phase5_evaluation.py")
    ]
    
    # Run each phase
    completed_phases = []
    for phase_name, script_path in phases:
        if os.path.exists(script_path):
            success = run_phase(phase_name, script_path)
            if success:
                completed_phases.append(phase_name)
            else:
                print(f"⚠️  {phase_name} failed. Stopping pipeline.")
                break
        else:
            print(f"⚠️  {script_path} not found. Skipping {phase_name}.")
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print('='*60)
    print(f"Completed phases: {len(completed_phases)}/{len(phases)}")
    for phase in completed_phases:
        print(f"✅ {phase}")
    
    if len(completed_phases) == len(phases):
        print("\n🎉 All phases completed successfully!")
        print("Check the 'results/' directory for outputs.")
    else:
        print(f"\n⚠️  Pipeline completed with {len(phases) - len(completed_phases)} phases failed.")

if __name__ == "__main__":
    main()
