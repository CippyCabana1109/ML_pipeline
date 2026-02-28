# Solar Forecasting Model Comparison Plan

## Project Overview
Compare SARIMAX, XGBoost, and Prophet+XGBoost Hybrid models for next-day solar bidding using production data (Jan 2024 – Feb 2025) and NASA POWER weather data.

---

## Phase 1: Data Ingestion & Preprocessing ✅ COMPLETED

### 1.1 Solar Production Data Merging ✅
- [x] Used provided Training_Dataset.csv with integrated weather and production data
- [x] Extracted total solar power from 26 inverters
- [x] Validated timestamp consistency (5-minute intervals)
- [x] Created unified dataset (Dec 2023 – Jan 2025)

### 1.2 Weather Data Integration ✅
- [x] Extracted required exogenous variables from Training_Dataset.csv:
  - Surface shortwave downward radiation (irradiance)
  - Temperature at 2 meters
  - Relative humidity at 2 meters
- [x] Data already synchronized on timestamp

### 1.3 Data Quality & Feature Engineering ✅
- [x] Handled missing values (removed rows with NaN)
- [x] Created temporal features: hour, day_of_week, month, day_of_year
- [x] Generated lag features for XGBoost (t-24h, t-48h)
- [x] Split data: Training (Jan 2024 – Nov 2024), Testing (Dec 1-7, 2024)
- [x] Created daytime filter (irradiance > 0) for sMAPE calculation

### Phase 1 Results:
- **Total Records**: 110,478 (5-minute intervals)
- **Training Data**: 94,177 records (Jan 2024 - Nov 2024)
- **Test Data**: 1,729 records (Dec 1-7, 2024)
- **Daytime Data**: 53.6% of records have irradiance > 0
- **Data Range**: Dec 31, 2023 to Jan 24, 2025

---

## Phase 2: Baseline Development (SARIMAX)

### 2.1 Data Preparation
- [ ] Prepare training data with exogenous variables (irradiance, temperature, humidity)
- [ ] Check stationarity and apply differencing if needed
- [ ] Perform ACF/PACF analysis for parameter selection

### 2.2 Model Implementation
- [ ] Implement SARIMAX(p,d,q)(P,D,Q)s model
- [ ] Use weather variables as exogenous inputs
- [ ] Optimize hyperparameters using grid search or auto_arima
- [ ] Train on training period (Jan 2024 – Feb 2025)

### 2.3 Validation & Testing
- [ ] Generate predictions for test week (March 1-7, 2025)
- [ ] Apply daytime filter for evaluation
- [ ] Calculate baseline metrics: MAE, RMSE, sMAPE, R²

---

## Phase 3: Advanced Modeling

### 3.1 XGBoost Implementation
- [ ] Prepare feature matrix:
  - Target: Active Power
  - Features: irradiance, temperature, humidity, hour, day_of_week, lag_24h, lag_48h
- [ ] Split training data for validation
- [ ] Optimize XGBoost hyperparameters (n_estimators, max_depth, learning_rate)
- [ ] Train final model on full training period
- [ ] Generate test week predictions

### 3.2 Prophet+XGBoost Hybrid Implementation
- [ ] **Step 1 - Prophet**:
  - Train Prophet model on training data
  - Generate trend predictions for training and test periods
  - Calculate residuals: Actual - Prophet_Pred

- [ ] **Step 2 - XGBoost on Residuals**:
  - Train XGBoost on residuals using same features as standalone XGBoost
  - Predict residuals for test week
  - Combine: Final_Prediction = Prophet_Pred + Residual_Prediction

### 3.3 Model Validation
- [ ] Generate predictions for test week for both advanced models
- [ ] Apply daytime filter for evaluation
- [ ] Calculate metrics: MAE, RMSE, sMAPE, R²

---

## Phase 4: Evaluation & Interpretation

### 4.1 Comparative Analysis
- [ ] Create comprehensive metrics table:
  | Model | MAE | RMSE | sMAPE | R² |
  |-------|-----|------|-------|----|
  | SARIMAX | ? | ? | ? | ? |
  | XGBoost | ? | ? | ? | ? |
  | Prophet+XGBoost | ? | ? | ? | ? |

- [ ] Generate "Actual vs. Predicted" plots for March test week:
  - All three models on one plot for comparison
  - Individual model plots with error metrics
  - Daytime-only view to highlight solar production patterns

### 4.2 Bidding Suitability Assessment
- [ ] Analyze prediction accuracy during peak production hours
- [ ] Evaluate model reliability for next-day bidding decisions
- [ ] Assess computational efficiency and implementation complexity
- [ ] Provide recommendations for operational deployment

### 4.3 Documentation & Deliverables
- [ ] Compile all code into reproducible scripts
- [ ] Create comprehensive evaluation report
- [ ] Document model assumptions and limitations
- [ ] Prepare presentation of findings

---

## Success Criteria
- All models trained without data leakage
- sMAPE calculated only on daytime hours (irradiance > 0)
- Complete metrics table with all four evaluation measures
- Clear visualization of test week predictions
- Actionable recommendations for bidding strategy

## Timeline Estimate
- Phase 1: 2-3 days (data ingestion and preprocessing)
- Phase 2: 1-2 days (SARIMAX implementation)
- Phase 3: 2-3 days (XGBoost and Hybrid models)
- Phase 4: 1-2 days (evaluation and reporting)
- **Total: 6-10 days**
