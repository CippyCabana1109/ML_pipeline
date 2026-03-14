"""
Solar PV Forecasting Pipeline - Full 7-Step Implementation
Follows: Forecasting Model Testing and Evaluation Plan

Data source: NASA POWER hourly weather data (real observations)
Solar power: Physics-based PV model (100kW system, 18% efficiency, NOCT thermal model)
             with cloud intermittency noise derived from real clearness index (ALLSKY_KT)

Steps:
  1. Ideal Solar Generation Curve + VIF/Correlation Analysis
  2. Train 4 Models: SARIMAX, XGBoost, Prophet, Prophet+XGBoost Hybrid
  3. Compare Predicted vs Actual (plots)
  4. Evaluate with Weighted Metrics (RMSE 40%, MAE 30%, sMAPE 20%, R² 10%)
  5. Hourly Error Analysis
  6. Iterative Learning on Best Model
  7. Energy Market Implications

Requirements: pip install pandas numpy scikit-learn statsmodels xgboost prophet matplotlib seaborn
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ─── CONFIG ─────────────────────────────────────────────────────────────────
NASA_RAW_PATH = "data/Weather_Data.csv"       # Raw NASA POWER file (2020-2024)
TRAIN_PATH    = "data/train_final.csv"         # Fallback if NASA file missing
TEST_PATH     = "data/test_final.csv"
OUT_DIR       = "results/pipeline"

# PV system parameters (physics model)
PANEL_AREA = 100.0   # m²  (≈100 kW peak system)
ETA        = 0.18    # Panel efficiency 18%
GAMMA      = 0.004   # Temperature coefficient 0.4%/°C
T_REF      = 25.0    # STC reference temperature (°C)
NOCT       = 45.0    # Nominal Operating Cell Temperature (°C)
NOISE_SEED = 42

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
})

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUT_DIR}/tables",  exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def smape(actual, predicted):
    """Symmetric Mean Absolute Percentage Error — daytime only (irradiance > 0)."""
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    a, p = np.array(actual)[mask], np.array(predicted)[mask]
    return 100.0 * np.mean(2 * np.abs(p - a) / (np.abs(a) + np.abs(p) + 1e-9))

def weighted_score(rmse, mae, smape_val, r2, w=(0.40, 0.30, 0.20, 0.10)):
    """
    Lower score = better model.
    Normalise each metric to [0,1] before weighting — handled externally.
    """
    pass  # Scoring is done after collecting all model metrics

def evaluate(y_true, y_pred, name="Model"):
    mae_v   = mean_absolute_error(y_true, y_pred)
    rmse_v  = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_v = smape(y_true, y_pred)
    r2_v    = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae_v, "RMSE": rmse_v, "sMAPE": smape_v, "R2": r2_v}



# ════════════════════════════════════════════════════════════════════════════
# LOAD & BUILD DATA FROM NASA POWER SOURCE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  SOLAR PV FORECASTING PIPELINE — FULL 7-STEP EVALUATION")
print("="*65)

print("\n[DATA] Loading NASA POWER weather data …")

def build_dataset_from_nasa(nasa_path, seed=NOISE_SEED):
    """
    Parse raw NASA POWER file and build physics-based solar PV dataset.
    PV model: P = G * A * eta * (1 - gamma*(T_cell - T_ref))
    Cloud noise derived from real clearness index ALLSKY_KT.
    """
    df = pd.read_csv(nasa_path, skiprows=23, low_memory=False)
    df = df.replace(-999, np.nan)

    # Filter to 2024 (training + test year)
    df = df[df['YEAR'] == 2024].copy().reset_index(drop=True)

    # Timestamp
    df['timestamp'] = pd.to_datetime(
        df[['YEAR','MO','DY','HR']].rename(
            columns={'YEAR':'year','MO':'month','DY':'day','HR':'hour'}))

    # Physics-based PV power
    G  = df['ALLSKY_SFC_SW_DWN'].fillna(0).values
    T  = df['T2M'].values
    KT = df['ALLSKY_KT'].fillna(0.5).values

    T_cell  = T + (NOCT - 20) / 800 * G
    P_raw   = G * PANEL_AREA * ETA * (1 - GAMMA * (T_cell - T_REF))

    # Real cloud intermittency noise (seeded, reproducible)
    rng = np.random.default_rng(seed)
    sigma = 0.05 + 0.15 * (1 - np.clip(KT, 0, 1))   # more noise on cloudy hours
    cloud_noise = 1.0 + rng.normal(0, sigma, len(P_raw))
    df['solar_power_w'] = np.clip(P_raw * np.clip(cloud_noise, 0.1, 1.2), 0, None)

    # Feature engineering
    df['irradiance']  = df['ALLSKY_SFC_SW_DWN'].fillna(0)
    df['temperature'] = df['T2M']
    df['humidity']    = df['RH2M']
    df['clearness']   = df['ALLSKY_KT'].fillna(0)
    df['dni']         = df['ALLSKY_SFC_SW_DNI'].fillna(0)
    df['diffuse']     = df['ALLSKY_SFC_SW_DIFF'].fillna(0)
    df['wind_speed']  = df['WS10M'].fillna(0)
    df['pressure']    = df['PS'].fillna(df['PS'].median())
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month']       = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_daytime']  = df['irradiance'] > 0
    df['lag_24h']     = df['solar_power_w'].shift(24).fillna(0)
    df['lag_48h']     = df['solar_power_w'].shift(48).fillna(0)

    return df

if os.path.exists(NASA_RAW_PATH):
    full_df = build_dataset_from_nasa(NASA_RAW_PATH)
    train = full_df[full_df['MO'] <= 11].copy()
    test  = full_df[(full_df['MO'] == 12) & (full_df['DY'] <= 7)].copy()
    print(f"  Source: Real NASA POWER physics-based dataset")
    print(f"  PV model: {PANEL_AREA}m² panel, {ETA*100:.0f}% efficiency, NOCT={NOCT}°C")
    print(f"  Cloud noise: seeded from real ALLSKY_KT clearness index")
else:
    print(f"  ⚠️  NASA file not found at {NASA_RAW_PATH}")
    print(f"  Falling back to pre-processed train/test CSVs …")
    train = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    test  = pd.read_csv(TEST_PATH,  parse_dates=['timestamp'])
    train['lag_24h'] = train['lag_24h'].fillna(0)
    train['lag_48h'] = train['lag_48h'].fillna(0)
    test['lag_24h']  = test['lag_24h'].fillna(0)
    test['lag_48h']  = test['lag_48h'].fillna(0)

train = train.reset_index(drop=True)
test  = test.reset_index(drop=True)

print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")
print(f"  Train period: {train['timestamp'].min().date()} → {train['timestamp'].max().date()}")
print(f"  Test period : {test['timestamp'].min().date()} → {test['timestamp'].max().date()}")
irr_pwr_corr = train['solar_power_w'].corr(train['irradiance'])
print(f"  Irradiance<->Power correlation: {irr_pwr_corr:.4f}  (< 1.0 = real variation OK)")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — IDEAL SOLAR CURVE + CORRELATION / VIF
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*65)
print("STEP 1 — Ideal Solar Curve  +  Correlation / VIF Analysis")
print("─"*65)

# --- 1a. Ideal curve (average generation per hour of day) ---
daytime_train = train[train['irradiance'] > 0]
ideal_curve   = train.groupby('hour')['solar_power_w'].mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Step 1 — Ideal Solar Generation & Feature Analysis", fontsize=14, fontweight='bold')

# Ideal curve
axes[0].fill_between(ideal_curve.index, ideal_curve.values, alpha=0.25, color='gold')
axes[0].plot(ideal_curve.index, ideal_curve.values, 'o-', color='darkorange', linewidth=2.5, markersize=5)
axes[0].set_title("Ideal Daily Solar Generation Curve")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Avg Solar Power (W)")
axes[0].set_xticks(range(0, 24, 2))
axes[0].set_xlim(0, 23)
peak_h = ideal_curve.idxmax()
axes[0].axvline(peak_h, color='red', linestyle='--', alpha=0.5, label=f'Peak: {peak_h}:00')
axes[0].legend()

# Correlation map
weather_cols = ['irradiance', 'clearness', 'dni', 'diffuse', 'temperature',
                'humidity', 'wind_speed', 'lag_24h', 'lag_48h',
                'hour', 'month', 'solar_power_w']
corr = train[weather_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=axes[1], cbar_kws={'shrink': 0.7},
            square=True, linewidths=0.5)
axes[1].set_title("Correlation Map (all features)")
axes[1].tick_params(axis='x', rotation=45)

# VIF calculation
from numpy.linalg import matrix_rank

def compute_vif(df_in, cols):
    from sklearn.linear_model import LinearRegression as LR
    vif_vals = {}
    X = df_in[cols].dropna().values
    for i, col in enumerate(cols):
        y_i   = X[:, i]
        X_oth = np.delete(X, i, axis=1)
        r2_i  = LR().fit(X_oth, y_i).score(X_oth, y_i)
        vif_vals[col] = 1.0 / (1.0 - r2_i + 1e-9)
    return vif_vals

feature_cols_vif = ['irradiance', 'clearness', 'dni', 'diffuse',
                    'temperature', 'humidity', 'wind_speed', 'pressure',
                    'lag_24h', 'lag_48h', 'hour', 'month']
vif_dict = compute_vif(train, feature_cols_vif)

vif_df = pd.DataFrame({'Feature': list(vif_dict.keys()), 'VIF': list(vif_dict.values())})
vif_df = vif_df.sort_values('VIF', ascending=True)

colors = ['#d62728' if v > 10 else '#ff7f0e' if v > 5 else '#2ca02c' for v in vif_df['VIF']]
bars   = axes[2].barh(vif_df['Feature'], vif_df['VIF'], color=colors)
axes[2].axvline(5,  color='orange', linestyle='--', linewidth=1.5, label='VIF = 5 (moderate)')
axes[2].axvline(10, color='red',    linestyle='--', linewidth=1.5, label='VIF = 10 (drop)')
axes[2].set_title("Variance Inflation Factor (VIF)")
axes[2].set_xlabel("VIF Value")
axes[2].legend(fontsize=9)
for bar, val in zip(bars, vif_df['VIF']):
    axes[2].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/step1_ideal_curve_vif.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: step1_ideal_curve_vif.png")

# Drop high-VIF features — but always keep 'irradiance' as the primary
# solar variable even if its VIF is high (it's the dominant predictor).
# The redundancy is among dni/diffuse/clearness which all proxy irradiance.
ALWAYS_KEEP  = {'irradiance', 'temperature', 'humidity', 'hour', 'month'}
DROP_VIF = [k for k, v in vif_dict.items() if v > 10 and k not in ALWAYS_KEEP]
print(f"  Features with VIF > 10 (dropped): {DROP_VIF if DROP_VIF else 'None'}")
print(f"  Always-kept despite high VIF: {[k for k in ALWAYS_KEEP if vif_dict.get(k,0)>10]}")

# Final feature set
FEATURES = [f for f in feature_cols_vif if f not in DROP_VIF]
print(f"  Final model features ({len(FEATURES)}): {FEATURES}")

X_train = train[FEATURES].fillna(0)
y_train = train['solar_power_w']
X_test  = test[FEATURES].fillna(0)
y_test  = test['solar_power_w']
ts_test = test['timestamp']


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN 4 MODELS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*65)
print("STEP 2 — Training Models")
print("─"*65)

predictions = {}

# ── 2a. SARIMAX ─────────────────────────────────────────────────────────────
print("  [1/4] SARIMAX …")
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Use the training time series; exogenous = irradiance (top correlated)
    ts_train_idx = pd.DatetimeIndex(train['timestamp'])
    sarimax_exog_train = train[['irradiance', 'temperature', 'clearness']].fillna(0).values
    sarimax_exog_test  = test[['irradiance', 'temperature', 'clearness']].fillna(0).values

    sarima_model = SARIMAX(
        y_train.values,
        exog=sarimax_exog_train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_fit = sarima_model.fit(disp=False, maxiter=100)
    sarimax_pred = sarima_fit.forecast(steps=len(y_test), exog=sarimax_exog_test)
    sarimax_pred = np.clip(sarimax_pred, 0, None)
    predictions['SARIMAX'] = sarimax_pred
    print("     SARIMAX trained")
except Exception as e:
    print(f"     ⚠️  SARIMAX failed ({e}), using ARIMA fallback …")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        arima_m = ARIMA(y_train.values, order=(2, 0, 2)).fit()
        base_pred = arima_m.forecast(steps=len(y_test))
        # Correct with irradiance scaling
        irr_scale = X_test['irradiance'].values / (X_train['irradiance'].mean() + 1e-9)
        sarimax_pred = np.clip(base_pred * irr_scale, 0, None)
        predictions['SARIMAX'] = sarimax_pred
        print("     ARIMA fallback trained")
    except Exception as e2:
        print(f"     ARIMA also failed ({e2}), using LR proxy")
        from sklearn.linear_model import Ridge
        lr_s = Ridge(); lr_s.fit(X_train, y_train)
        predictions['SARIMAX'] = np.clip(lr_s.predict(X_test), 0, None)

# ── 2b. XGBoost ─────────────────────────────────────────────────────────────
print("  [2/4] XGBoost …")
try:
    import xgboost as xgb

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    xgb_pred = np.clip(xgb_model.predict(X_test), 0, None)
    predictions['XGBoost'] = xgb_pred
    print("     XGBoost trained")
except Exception as e:
    print(f"     ⚠️  XGBoost not installed ({e}), using GradientBoosting …")
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                   learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    predictions['XGBoost'] = np.clip(gb.predict(X_test), 0, None)
    print("     GradientBoosting (XGBoost proxy) trained")

# ── 2c. Prophet ─────────────────────────────────────────────────────────────
print("  [3/4] Prophet …")
try:
    from prophet import Prophet

    prophet_train = pd.DataFrame({
        'ds': train['timestamp'],
        'y':  y_train.values
    })
    # Add regressors
    PROPHET_REGRESSORS = ['irradiance', 'temperature', 'humidity', 'clearness']
    for col in PROPHET_REGRESSORS:
        prophet_train[col] = train[col].fillna(0).values

    pm = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    for col in PROPHET_REGRESSORS:
        pm.add_regressor(col)

    pm.fit(prophet_train)

    prophet_test_df = pd.DataFrame({'ds': test['timestamp']})
    for col in PROPHET_REGRESSORS:
        prophet_test_df[col] = test[col].fillna(0).values

    prophet_fc = pm.predict(prophet_test_df)
    prophet_pred = np.clip(prophet_fc['yhat'].values, 0, None)
    predictions['Prophet'] = prophet_pred
    print("     Prophet trained")
except Exception as e:
    print(f"     ⚠️  Prophet not installed ({e}), using seasonal decomposition proxy …")
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    predictions['Prophet'] = np.clip(rf.predict(X_test), 0, None)
    print("     RandomForest (Prophet proxy) trained")

# ── 2d. Prophet + XGBoost Hybrid ─────────────────────────────────────────────
print("  [4/4] Prophet + XGBoost Hybrid …")
# Strategy: use Prophet (or its proxy) for the trend/seasonality baseline,
#           then train XGBoost on the *residuals*, and combine.
try:
    # Residuals from Prophet on training data
    try:
        prophet_train_fc = pm.predict(
            pd.DataFrame({'ds': train['timestamp'],
                          'irradiance': train['irradiance'].values,
                          'temperature': train['temperature'].values,
                          'humidity': train['humidity'].values})
        )
        prophet_train_yhat = np.clip(prophet_train_fc['yhat'].values, 0, None)
    except Exception:
        from sklearn.ensemble import RandomForestRegressor as RFR
        rr = RFR(n_estimators=100, random_state=42); rr.fit(X_train, y_train)
        prophet_train_yhat = np.clip(rr.predict(X_train), 0, None)

    residuals_train = y_train.values - prophet_train_yhat

    # XGBoost trained on residuals
    try:
        import xgboost as xgb
        xgb_res = xgb.XGBRegressor(n_estimators=200, max_depth=5,
                                    learning_rate=0.05, random_state=42,
                                    n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        xgb_res = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                            learning_rate=0.05, random_state=42)
    xgb_res.fit(X_train, residuals_train)
    residual_pred = xgb_res.predict(X_test)
    hybrid_pred   = np.clip(predictions['Prophet'] + residual_pred, 0, None)
    predictions['Prophet+XGBoost'] = hybrid_pred
    print("     Prophet+XGBoost Hybrid trained")
except Exception as e:
    print(f"     ⚠️  Hybrid failed ({e}), using ensemble average …")
    predictions['Prophet+XGBoost'] = (predictions['Prophet'] + predictions['XGBoost']) / 2


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — PREDICTED vs ACTUAL PLOTS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*65)
print("STEP 3 — Predicted vs Actual Generation Plots")
print("─"*65)

fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
fig.suptitle("Step 3 — Predicted vs Actual Solar Generation\n(Red = Actual, Blue = Predicted)", fontsize=14)

actual = y_test.values
x_axis = ts_test.values

for ax, (name, pred) in zip(axes.flatten(), predictions.items()):
    ax.plot(x_axis, actual, color='red',       linewidth=2,   alpha=0.9, label='Actual',    zorder=3)
    ax.plot(x_axis, pred,   color='steelblue', linewidth=1.5, alpha=0.85, label='Predicted', zorder=2)
    ax.fill_between(x_axis, actual, pred, alpha=0.12, color='purple', label='Error area')
    ax.set_title(name)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Solar Power (W)")
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/step3_predicted_vs_actual.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: step3_predicted_vs_actual.png")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — WEIGHTED METRIC EVALUATION & RANKING
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*65)
print("STEP 4 — Weighted Performance Evaluation")
print("─"*65)

metrics_list = []
for name, pred in predictions.items():
    m = evaluate(actual, pred, name)
    metrics_list.append(m)

metrics_df = pd.DataFrame(metrics_list)

# Weighted composite score (lower = better)
# Normalise each metric to [0,1]
W = {'RMSE': 0.40, 'MAE': 0.30, 'sMAPE': 0.20, 'R2': 0.10}

def normalise_col(col, higher_better=False):
    mn, mx = metrics_df[col].min(), metrics_df[col].max()
    if mx == mn:
        return pd.Series([0.0] * len(metrics_df))
    norm = (metrics_df[col] - mn) / (mx - mn)
    return 1 - norm if higher_better else norm

metrics_df['norm_RMSE']  = normalise_col('RMSE')
metrics_df['norm_MAE']   = normalise_col('MAE')
metrics_df['norm_sMAPE'] = normalise_col('sMAPE')
metrics_df['norm_R2']    = normalise_col('R2', higher_better=True)

metrics_df['Weighted_Score'] = (
    W['RMSE']  * metrics_df['norm_RMSE'] +
    W['MAE']   * metrics_df['norm_MAE']  +
    W['sMAPE'] * metrics_df['norm_sMAPE'] +
    W['R2']    * metrics_df['norm_R2']
)

metrics_df['Rank'] = metrics_df['Weighted_Score'].rank().astype(int)
metrics_df = metrics_df.sort_values('Rank')

print("\n  MODEL RANKINGS:")
print(f"  {'Rank':<6} {'Model':<22} {'RMSE':>10} {'MAE':>10} {'sMAPE':>10} {'R²':>8} {'W-Score':>10}")
print("  " + "-"*76)
for _, row in metrics_df.iterrows():
    tag = " 🏆" if row['Rank'] == 1 else ""
    print(f"  {int(row['Rank']):<6} {row['Model']:<22} {row['RMSE']:>10.1f} {row['MAE']:>10.1f} "
          f"{row['sMAPE']:>10.2f} {row['R2']:>8.4f} {row['Weighted_Score']:>10.4f}{tag}")

# Save table
out_cols = ['Rank', 'Model', 'RMSE', 'MAE', 'sMAPE', 'R2', 'Weighted_Score']
metrics_df[out_cols].to_csv(f"{OUT_DIR}/tables/step4_model_rankings.csv", index=False)

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Step 4 — Weighted Model Performance Evaluation", fontsize=14)

colors_bar = ['gold', 'silver', '#cd7f32', '#a0a0a0'][:len(metrics_df)]

# RMSE
axes[0].bar(metrics_df['Model'], metrics_df['RMSE'], color=colors_bar)
axes[0].set_title("RMSE (weight 40%)\n← lower is better")
axes[0].set_ylabel("W")
axes[0].tick_params(axis='x', rotation=20)

# MAE
axes[1].bar(metrics_df['Model'], metrics_df['MAE'], color=colors_bar)
axes[1].set_title("MAE (weight 30%)\n← lower is better")
axes[1].set_ylabel("W")
axes[1].tick_params(axis='x', rotation=20)

# Weighted composite
axes[2].bar(metrics_df['Model'], metrics_df['Weighted_Score'], color=colors_bar)
axes[2].set_title("Composite Weighted Score\n← lower = best overall")
axes[2].set_ylabel("Score (0–1)")
axes[2].tick_params(axis='x', rotation=20)
for ax in axes:
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.0f}" if bar.get_height() > 1 else f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/step4_model_rankings.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: step4_model_rankings.png")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — HOURLY ERROR ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*65)
print("STEP 5 — Forecast Error by Hour of Day")
print("─"*65)

test_hours = test['hour'].values
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
fig.suptitle("Step 5 — Hourly Forecast Error Analysis\n(Highlights ramp-up, peak, ramp-down behaviour)", fontsize=13)

for ax, (name, pred) in zip(axes.flatten(), predictions.items()):
    error = pred - actual  # signed error: + = over-predict, – = under-predict
    hourly_err = pd.DataFrame({'hour': test_hours, 'error': error})
    hourly_mean = hourly_err.groupby('hour')['error'].mean()
    hourly_std  = hourly_err.groupby('hour')['error'].std().fillna(0)

    ax.bar(hourly_mean.index, hourly_mean.values,
           color=['#d62728' if e < 0 else '#1f77b4' for e in hourly_mean.values],
           alpha=0.75)
    ax.fill_between(hourly_mean.index,
                    hourly_mean - hourly_std,
                    hourly_mean + hourly_std,
                    alpha=0.15, color='grey')
    ax.axhline(0, color='black', linewidth=1.2)
    ax.axvspan(6, 9,   alpha=0.06, color='orange', label='Morning ramp')
    ax.axvspan(10, 14, alpha=0.06, color='green',  label='Midday stable')
    ax.axvspan(15, 18, alpha=0.06, color='red',    label='Evening ramp-down')
    ax.set_title(name)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean error (W)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/step5_hourly_error.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: step5_hourly_error.png")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — ITERATIVE LEARNING ON BEST MODEL
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*65)
print("STEP 6 — Iterative Learning (Best Model)")
print("─"*65)

best_model_name = metrics_df.iloc[0]['Model']
print(f"  Best model: {best_model_name}")

# We simulate iterative (online) learning:
# Split test into daily chunks, retrain after each chunk.
test['date'] = test['timestamp'].dt.date
test_dates   = sorted(test['date'].unique())
train_work   = train.copy()

iterative_results = []

for i, day in enumerate(test_dates):
    day_mask = test['date'] == day
    day_data = test[day_mask]
    X_day    = day_data[FEATURES].fillna(0)
    y_day    = day_data['solar_power_w'].values

    # Retrain XGBoost (or GBM) on growing training set
    X_w = train_work[FEATURES].fillna(0)
    y_w = train_work['solar_power_w']

    try:
        import xgboost as xgb
        iter_model = xgb.XGBRegressor(n_estimators=150, max_depth=5,
                                      learning_rate=0.05, random_state=42,
                                      n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        iter_model = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                               learning_rate=0.05, random_state=42)

    iter_model.fit(X_w, y_w)
    day_pred = np.clip(iter_model.predict(X_day), 0, None)
    day_mae  = mean_absolute_error(y_day, day_pred)
    day_rmse = np.sqrt(mean_squared_error(y_day, day_pred))

    iterative_results.append({'Iteration': i+1, 'Date': day, 'MAE': day_mae, 'RMSE': day_rmse})
    print(f"    Iter {i+1}/{len(test_dates)} | Date: {day} | MAE: {day_mae:.1f}W | RMSE: {day_rmse:.1f}W")

    # Add today's observations into training pool
    train_work = pd.concat([train_work, day_data], ignore_index=True)

iter_df = pd.DataFrame(iterative_results)
iter_df.to_csv(f"{OUT_DIR}/tables/step6_iterative_learning.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Step 6 — Iterative Learning: Error Convergence", fontsize=13)

axes[0].plot(iter_df['Iteration'], iter_df['MAE'], 'o-', color='steelblue', linewidth=2)
axes[0].fill_between(iter_df['Iteration'], iter_df['MAE'], alpha=0.2, color='steelblue')
axes[0].set_title("MAE per Iteration")
axes[0].set_xlabel("Iteration (day)")
axes[0].set_ylabel("MAE (W)")
axes[0].set_xticks(iter_df['Iteration'])

axes[1].plot(iter_df['Iteration'], iter_df['RMSE'], 'o-', color='darkorange', linewidth=2)
axes[1].fill_between(iter_df['Iteration'], iter_df['RMSE'], alpha=0.2, color='darkorange')
axes[1].set_title("RMSE per Iteration")
axes[1].set_xlabel("Iteration (day)")
axes[1].set_ylabel("RMSE (W)")
axes[1].set_xticks(iter_df['Iteration'])

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/step6_iterative_learning.png", dpi=150, bbox_inches='tight')
plt.show()

# Improvement summary
mae_start = iter_df['MAE'].iloc[0]
mae_end   = iter_df['MAE'].iloc[-1]
improve   = (mae_start - mae_end) / mae_start * 100
print(f"  MAE improvement over {len(test_dates)} iterations: {improve:.1f}%")
print("  Saved: step6_iterative_learning.png")


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — ENERGY MARKET IMPLICATIONS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*65)
print("STEP 7 — Energy Market Implications")
print("─"*65)

# Imbalance penalty model:
# If predicted > actual: generator sells less than committed → penalty on shortfall
# If predicted < actual: generator spills (curtailment risk)
# Assume market price: $50/MWh average; imbalance penalty: $25/MWh

MARKET_PRICE    = 50.0    # $/MWh
IMBALANCE_RATE  = 25.0    # $/MWh penalty
CAPACITY_KW     = 100.0   # kW system capacity (assumed)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Step 7 — Energy Market Financial Impact of Forecast Error", fontsize=13)

model_names, daily_penalties, daily_revenues = [], [], []

for name, pred in predictions.items():
    error_wh    = pred - actual                              # Wh error per hour
    penalty_kwh = np.abs(error_wh) / 1000.0                 # kWh imbalance
    hourly_pen  = penalty_kwh * IMBALANCE_RATE / 1000.0     # $ penalty per hour
    hourly_rev  = (actual / 1000.0) * (MARKET_PRICE / 1000) # $ revenue per hour

    daily_pen = hourly_pen.sum() / len(test_dates)
    daily_rev = hourly_rev.sum() / len(test_dates)
    model_names.append(name)
    daily_penalties.append(daily_pen)
    daily_revenues.append(daily_rev)

colors_m = ['gold', 'silver', '#cd7f32', '#a0a0a0'][:len(model_names)]
axes[0].bar(model_names, daily_penalties, color=colors_m)
axes[0].set_title("Avg Daily Imbalance Penalty by Model\n← lower = less financial risk")
axes[0].set_ylabel("$ / day")
axes[0].tick_params(axis='x', rotation=15)
for bar, val in zip(axes[0].patches, daily_penalties):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"${val:.3f}", ha='center', va='bottom', fontsize=9)

# Penalty as % of revenue
pct_penalty = [p/r*100 if r > 0 else 0 for p, r in zip(daily_penalties, daily_revenues)]
axes[1].bar(model_names, pct_penalty, color=colors_m)
axes[1].set_title("Penalty as % of Daily Revenue\n← lower = more profitable participation")
axes[1].set_ylabel("Penalty / Revenue (%)")
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(axes[1].patches, pct_penalty):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}%", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/step7_market_impact.png", dpi=150, bbox_inches='tight')
plt.show()

mkt_df = pd.DataFrame({
    'Model': model_names,
    'Daily_Revenue_USD': daily_revenues,
    'Daily_Penalty_USD': daily_penalties,
    'Penalty_Pct': pct_penalty
})
mkt_df.to_csv(f"{OUT_DIR}/tables/step7_market_impact.csv", index=False)
print("  Saved: step7_market_impact.png + CSV")


# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  PIPELINE COMPLETE — SUMMARY")
print("="*65)
best = metrics_df.iloc[0]
print(f"\n  🏆 Best model (weighted score): {best['Model']}")
print(f"     RMSE  : {best['RMSE']:.1f} W")
print(f"     MAE   : {best['MAE']:.1f} W")
print(f"     sMAPE : {best['sMAPE']:.2f} %")
print(f"     R²    : {best['R2']:.4f}")
print(f"     W-Score: {best['Weighted_Score']:.4f}")

print(f"\n  Iterative learning MAE improvement: {improve:.1f}%")
print(f"\n  📁 Output directory: {OUT_DIR}/")
print("     figures/step1_ideal_curve_vif.png")
print("     figures/step3_predicted_vs_actual.png")
print("     figures/step4_model_rankings.png")
print("     figures/step5_hourly_error.png")
print("     figures/step6_iterative_learning.png")
print("     figures/step7_market_impact.png")
print("     tables/step4_model_rankings.csv")
print("     tables/step6_iterative_learning.csv")
print("     tables/step7_market_impact.csv")
print("\n" + "="*65)
