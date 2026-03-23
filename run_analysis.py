"""
Simple MSc Dissertation Analysis Runner
"""

import subprocess
import sys

def run_script(script_name, description):
    """Run a script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run(['python', script_name], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False

def main():
    """Main runner"""
    print("=" * 80)
    print("MSc DISSERTATION ANALYSIS RUNNER")
    print("=" * 80)
    
    scripts = [
        ("01_weather_analysis.py", "Weather Variable Analysis"),
        ("02_economic_modeling.py", "Economic Modeling (Equations 1 & 2)")
    ]
    
    completed = []
    for script, description in scripts:
        success = run_script(script, description)
        if success:
            completed.append(description)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"Completed: {len(completed)}/{len(scripts)} analyses")
    
    for desc in completed:
        print(f"✓ {desc}")
    
    if len(completed) == len(scripts):
        print("\n🎓 ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print("📁 Check DISSERTATION_FIGURES/ for results")
    else:
        print("\n⚠️ Some analyses failed")

if __name__ == "__main__":
    main()
