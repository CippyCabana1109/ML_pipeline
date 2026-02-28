# Solar PV Forecasting for Competitive Energy Markets

## Overview
This project implements and compares three advanced forecasting models for next-day solar PV generation prediction, enabling participation in competitive electricity markets through accurate bidding strategies.

## Objective
Complete comparative evaluation of SARIMAX, XGBoost, and Prophet+XGBoost Hybrid models to determine the most accurate next-day solar PV generation prediction approach.

## Models Implemented
1. **SARIMAX** - Statistical baseline with weather as exogenous inputs
2. **XGBoost** - Machine learning regression model with lag features
3. **Prophet + XGBoost Hybrid** - Prophet forecast corrected using XGBoost residual modeling

## Data
- **PV Production**: 26 inverters, 5-minute intervals (Jan 2024 - Dec 2024)
- **Weather Variables**: NASA POWER data (irradiance, temperature, humidity)
- **Total Dataset**: 110,478 records
- **Training Period**: Jan 2024 - Nov 2024 (94,177 records)
- **Test Period**: Dec 1-7, 2024 (1,729 records)

## Project Structure
```
├── data/
│   ├── Training_Dataset.csv          # Raw integrated weather + production data
│   ├── processed_training_data.csv   # Cleaned dataset with features
│   ├── train_final.csv               # Training data split
│   └── test_final.csv                # Test data split
├── notebooks/
│   └── exploratory_analysis.ipynb    # Data exploration and visualization
├── src/
│   ├── phase1_data_ingestion.py      # Data preprocessing and merging
│   ├── phase2_sarimax.py            # SARIMAX model implementation
│   ├── phase3_xgboost.py             # XGBoost model implementation
│   ├── phase4_evaluation.py          # Model comparison and evaluation
│   └── utils.py                      # Utility functions
├── results/
│   ├── model_comparison.png          # Performance comparison plots
│   ├── predictions_test_week.png     # Actual vs Predicted visualization
│   ├── metrics_table.csv            # Final performance metrics
│   └── error_analysis.png            # Error distribution analysis
├── requirements.txt                  # Python dependencies
├── plan.md                          # Detailed project plan
├── .windsurfrules                   # Project rules and constraints
└── README.md                        # This file
```

## Evaluation Metrics
All models evaluated on the same test week using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **sMAPE** (Symmetric Mean Absolute Percentage Error - daytime only)
- **R²** (Coefficient of Determination)

## Key Features
- **No Data Leakage**: Strict temporal separation between training and testing
- **Daytime Filtering**: sMAPE calculated only for hours with irradiance > 0
- **Rolling Forecast**: Next-day prediction methodology
- **Weather Integration**: Exogenous variables (irradiance, temperature, humidity)
- **Temporal Features**: Hour, day of week, month, lag features (24h, 48h)

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone repository
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
