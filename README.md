# Solar PV Forecasting for Competitive Energy Markets

## 🎯 Objective
Complete comparative evaluation of SARIMAX, XGBoost, and Prophet+XGBoost Hybrid models to determine the most accurate next-day solar PV generation prediction approach for competitive electricity market participation.

## 📊 Models Implemented
1. **SARIMAX** - Statistical baseline with weather as exogenous inputs
2. **XGBoost** - Machine learning regression model with lag features  
3. **Prophet + XGBoost Hybrid** - Prophet forecast corrected using XGBoost residual modeling

## 📁 Project Structure
```
solar-pv-forecasting/
├── 📂 data/                     # Processed datasets (auto-generated)
│   ├── Weather_Data_Clean.csv    # NASA POWER weather data (2024)
│   ├── processed_training_data.csv # Cleaned dataset with features
│   ├── train_final.csv           # Training data split (Jan-Nov 2024)
│   └── test_final.csv            # Test data split (Dec 1-7, 2024)
├── 📂 src/                      # Source code
│   ├── utils.py                 # Utility functions
│   ├── phase1_data_ingestion.py  # Data preprocessing
│   ├── phase2_sarimax.py        # SARIMAX model
│   ├── phase3_xgboost.py        # XGBoost model
│   ├── phase4_hybrid.py         # Prophet+XGBoost hybrid
│   ├── phase5_evaluation.py     # Model comparison
│   ├── clean_weather_data.py   # Data cleaning utility
│   └── operational_assessment.md # Business recommendations
├── 📂 notebooks/               # Jupyter notebooks for exploration
├── 🐍 run_complete_pipeline.py # Execute all phases
├── 📋 requirements.txt          # Python dependencies
├── 📄 plan.md                 # Detailed project plan
├── 📄 .windsurfrules          # Project constraints & rules
└── 🧪 final_test.py           # Test data generator
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/CippyCabana1109/ML_pipeline.git
cd ML_pipeline
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Complete Pipeline
```bash
# Execute all phases sequentially
python run_complete_pipeline.py
```

### 4. Run Individual Phases
```bash
# Phase 1: Data preprocessing
python src/phase1_data_ingestion.py

# Phase 2: SARIMAX model
python src/phase2_sarimax.py

# Phase 3: XGBoost model  
python src/phase3_xgboost.py

# Phase 4: Prophet+XGBoost hybrid
python src/phase4_hybrid.py

# Phase 5: Model evaluation
python src/phase5_evaluation.py
```

## 📈 Data Overview
- **Weather Source**: NASA POWER hourly data (2024)
- **Variables**: Irradiance, Temperature, Humidity
- **Solar Power**: Synthetic generation based on irradiance patterns
- **Time Period**: Jan 2024 - Dec 2024 (Training: Jan-Nov, Test: Dec 1-7)
- **Total Records**: 8,761 hourly observations
- **Training Data**: 8,032 records (Jan 2024 - Nov 2024)
- **Test Data**: 168 records (Dec 1-7, 2024)

## 🎯 Evaluation Metrics
All models evaluated on same test week using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)  
- **sMAPE** (Symmetric Mean Absolute Percentage Error - daytime only)
- **R²** (Coefficient of Determination)

## 📊 Expected Outputs
After running the pipeline, you'll get:

### 1. Performance Comparison Table
```
| Model              | MAE (W) | RMSE (W) | sMAPE (%) | R²     | Overall Rank |
|--------------------|------------|------------|-------------|---------|-------------|
| Prophet+XGBoost    | [TBD]      | [TBD]      | [TBD]       | [TBD]   | 1.00        |
| XGBoost            | [TBD]      | [TBD]      | [TBD]       | [TBD]   | 2.00        |
| SARIMAX            | [TBD]      | [TBD]      | [TBD]       | [TBD]   | 3.00        |
```

### 2. Visualizations
- **Model Comparison Plots**: Error metrics and R² comparison
- **Actual vs Predicted**: Time series comparison for test week
- **Error Analysis**: Residual distributions and patterns

### 3. Operational Assessment
- Best performing model identification
- Performance improvement percentages
- Business recommendations for next-day bidding

## 🤝 Collaboration Guide

### For Team Members
1. **Clone Repository**: Use the GitHub URL above
2. **Create Branch**: `git checkout -b feature/your-feature-name`
3. **Make Changes**: Edit files in appropriate directories
4. **Test Changes**: Run pipeline to verify functionality
5. **Commit**: `git add . && git commit -m "Your descriptive message"`
6. **Push**: `git push origin feature/your-feature-name`
7. **Pull Request**: Create PR on GitHub for review

### Code Standards
- Use descriptive variable names
- Add comments for complex logic
- Follow existing code structure
- Test before committing
- Update documentation as needed

### Data Handling
- Raw weather data goes in `data/` directory
- Never commit large CSV files to git (except allowed ones)
- Use `.gitignore` to exclude generated files
- Results are auto-generated in `results/`

## 🔧 Dependencies
```python
# Core ML/Data Science
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0

# Time Series Analysis
statsmodels>=0.14.0
prophet>=1.1.0

# Machine Learning
xgboost>=1.6.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Utilities
tqdm>=4.64.0
joblib>=1.1.0
```

## 📋 Project Rules (.windsurfrules)
- **No Data Leakage**: Strict temporal separation between training/testing
- **Exogenous Variables**: Always include irradiance, temperature, humidity
- **Daytime Only**: sMAPE calculated only when irradiance > 0
- **Same Dataset**: All models use identical training/test splits
- **Next-Day Focus**: Rolling 24-hour forecasting approach

## 🎯 Business Impact
- **Improved Bidding Accuracy**: Better market positioning
- **Reduced Risk**: Lower forecast error penalties  
- **Operational Efficiency**: Automated daily forecasting
- **Competitive Advantage**: Data-driven energy trading

## 📞 Support
- **Issues**: Use GitHub Issues tab
- **Questions**: Check `.windsurfrules` for constraints
- **Documentation**: See `plan.md` for detailed methodology

## 🔄 Project Status
- ✅ **Data Processing**: NASA POWER weather data integration
- ✅ **Model Implementation**: SARIMAX, XGBoost, Prophet+XGBoost
- ✅ **Pipeline Automation**: Complete end-to-end execution
- ✅ **GitHub Ready**: Proper structure, documentation, and version control
- 🔄 **Testing**: Final validation in progress

---

**🚀 Ready to transform solar forecasting for competitive energy markets!**

*Last Updated: February 2026*
