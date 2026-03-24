@echo off
echo ============================================================
echo MSc DISSERTATION - COMPLETE SOLAR FORECASTING ANALYSIS
echo ============================================================
echo.
echo This will run ALL analyses in the correct order:
echo 1. Economic Modeling
echo 2. Hybrid Model Analysis  
echo 3. Improved Hybrid Analysis
echo 4. Iterative Learning Analysis
echo 5. Complete Single Day Comparison
echo 6. Corrected Single Day Analysis
echo.
echo Press any key to start or Ctrl+C to cancel...
pause > nul

echo.
echo Starting complete analysis...
echo.

python run_complete_dissertation_analysis.py

echo.
echo ============================================================
echo Analysis complete! Check DISSERTATION_FIGURES/ folder
echo ============================================================
echo.
pause
