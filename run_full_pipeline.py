"""
Professional Solar Forecasting Pipeline - Full Analysis
Complete pipeline with comprehensive evaluation and business analysis
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
    print(f"\n{'='*60}")
    print(f"RUNNING {phase_name}")
    print('='*60)
    
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
    """Run complete professional pipeline"""
    print("PROFESSIONAL SOLAR FORECASTING PIPELINE")
    print("=" * 60)
    print("Complete analysis with business impact assessment")
    print("=" * 60)
    
    # Check data directory
    if not Path('data').exists():
        print("❌ Data directory not found")
        return
    
    # Define professional pipeline phases
    phases = [
        ("Data Processing", "src/data_processing/data_processor.py"),
        ("SARIMAX Model", "src/models/sarimax_model.py"),
        ("XGBoost Model", "src/models/xgboost_model.py"),
        ("Hybrid Model", "src/models/hybrid_model.py"),
        ("Comprehensive Evaluation", "src/evaluation/comprehensive_evaluation.py")
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
                print(f"⚠️ Critical phase failed: {phase_name}")
                break
        else:
            print(f"⚠️ Script not found: {script_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print('='*60)
    print(f"Completed: {len(completed)}/{len(phases)} phases")
    print(f"Total time: {total_time:.1f}s")
    
    for phase in completed:
        print(f"✅ {phase}")
    
    if len(completed) == len(phases):
        print("\n🎉 FULL ANALYSIS COMPLETE!")
        print("📁 Results available in results/ directory")
        print("📊 Check results/tables/ for performance metrics")
        print("📈 Check results/figures/ for visualizations")
        print("📝 Check results/reports/ for business analysis")
    else:
        print(f"\n⚠️ Pipeline incomplete: {len(phases)-len(completed)} phases failed")

if __name__ == "__main__":
    main()
