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
│   ├── Training_Dataset.csv       # Raw integrated weather + production data
│   ├── processed_training_data.csv # Cleaned dataset with features
│   ├── train_final.csv           # Training data split
│   └── test_final.csv            # Test data split
├── 📂 src/                      # Source code
│   ├── utils.py                 # Utility functions
│   ├── phase1_data_ingestion.py  # Data preprocessing
│   ├── phase2_sarimax.py        # SARIMAX model
│   ├── phase3_xgboost.py        # XGBoost model
│   ├── phase4_hybrid.py         # Prophet+XGBoost hybrid
│   └── phase5_evaluation.py     # Model comparison
├── 📂 results/                  # Output files (auto-generated)
│   ├── model_comparison.csv       # Performance metrics table
│   ├── model_comparison.png       # Visualization plots
│   ├── error_analysis.png        # Error distribution analysis
│   └── operational_assessment.md # Business recommendations
├── 📂 notebooks/               # Jupyter notebooks for exploration
├── 🐍 run_complete_pipeline.py # Execute all phases
├── 📋 requirements.txt          # Python dependencies
├── 📄 plan.md                 # Detailed project plan
└── 📄 .windsurfrules          # Project constraints & rules
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/CippyCabana1109/ML_pipeline.git
cd ML_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation
```bash
python src/phase1_data_ingestion.py
```

### Model Training and Evaluation
```bash
# SARIMAX Baseline Model
python src/phase2_sarimax.py

# XGBoost Model
python src/phase3_xgboost.py

# Prophet+XGBoost Hybrid
python src/phase4_evaluation.py
```

### Complete Pipeline
```bash
# Run all phases sequentially
python run_complete_pipeline.py
```

## Results

### Model Performance Summary
| Model | MAE (W) | RMSE (W) | sMAPE (%) | R² |
|-------|---------|----------|-----------|----|
| SARIMAX | [TBD] | [TBD] | [TBD] | [TBD] |
| XGBoost | [TBD] | [TBD] | [TBD] | [TBD] |
| Prophet+XGBoost | [TBD] | [TBD] | [TBD] | [TBD] |

### Key Findings
[Results will be populated after model completion]

## Operational Suitability
The final model will be evaluated for:
- **Prediction Accuracy**: Next-day forecasting performance
- **Computational Efficiency**: Training and inference speed
- **Robustness**: Performance across different weather conditions
- **Implementation Complexity**: Ease of deployment in production

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- NASA POWER for weather data
- Statsmodels for SARIMAX implementation
- XGBoost for gradient boosting framework
- Facebook Prophet for time series decomposition

## Contact
Cyprian Cabana - [GitHub](https://github.com/CippyCabana1109)

---

**Note**: This project demonstrates the complete methodology for solar PV forecasting. The framework can be easily adapted to different time periods, locations, or renewable energy sources.
