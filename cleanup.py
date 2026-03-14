"""
Clean up unnecessary files
"""

import os

def cleanup_files():
    """Remove unnecessary files"""
    files_to_remove = [
        'PROJECT_SUMMARY.md',
        'plan.md',
        '.windsurfrules',
        'setup_world_standard.py',
        'create_professional_structure.py',
        'acf_pacf_analysis.png',
        'sarimax_predictions.png',
        'run_complete_pipeline.py',
        'run_fast_pipeline.py',
        'compare_results.py'
    ]
    
    removed = []
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            removed.append(file_path)
            print(f"Removed: {file_path}")
    
    return removed

if __name__ == "__main__":
    print("Cleaning up unnecessary files...")
    removed = cleanup_files()
    print(f"Removed {len(removed)} files")
