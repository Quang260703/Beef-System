import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import optuna
from sklearn.base import clone  # <-- Use 'clone', not 'cloneq'

def evaluate_forecast(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    print(f"\n[{model_name}] Test Performance:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    print(f"Accuracy (1 - MAE/mean * 100): {acc:.2f}%")

# 1. Data Loading and Initial Feature Engineering
df = pd.read_csv("Cow_Calf.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

target_col = 'Gross_Revenue'

max_lag = 6  # Maximum lag period in months
for lag in range(max_lag):
    df[f'Gross_Revenue_lag{lag}'] = df['Gross_Revenue'].shift(lag+1)


df[f'{target_col}_Rolling_Means'] = df[target_col].shift(1).rolling(3).mean() 
df[f'{target_col}_Rolling_STD'] = df[target_col].shift(1).rolling(3).std()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 3. Correlation Heatmap for the 4 Original Predictors
# original_predictors = ["Exchange_Rate_JPY_USD", "Net_Gas_Price", "CPI", "Corn_Price"]
# plt.figure(figsize=(6, 5))
# corr = df[original_predictors].corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap for Original Predictors")
# plt.tight_layout()
# plt.show()

# 4. Define Features and Target
target = "Gross_Revenue"
features = [col for col in df.columns if col != target]
X = df[features]
y = df[target]

# 5. Time Series Train-Test Split (80% Train, 20% Test)
test_size = int(len(df) * 0.4)  # 20% test
X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]

# 6. Hyperparameter Optimization (with Proper Scaling in Each CV Fold)
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 1000, step=20),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    # Scale within each fold
    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        scaler_cv = StandardScaler()
        X_tr_scaled = scaler_cv.fit_transform(X_tr)
        X_val_scaled = scaler_cv.transform(X_val)
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr_scaled, y_tr, eval_set=[(X_val_scaled, y_val)])
        preds = model.predict(X_val_scaled)
        mae_score = mean_absolute_error(y_val, preds)
        scores.append(mae_score)
    
    return np.mean(scores)

import optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# 7. Retrain Final Model with Global Scaling
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train)])

# 8. Probabilistic Forecasting (Quantile Regression)
quantiles = [0.025, 0.5, 0.975]
quantile_models = {}
for q in quantiles:
    print(f"Training quantile model for alpha = {q}")
    model = clone(lgb.LGBMRegressor(**best_params))
    model.set_params(objective='quantile', alpha=q)
    model.fit(X_train_scaled, y_train)
    quantile_models[q] = model

predictions = {q: quantile_models[q].predict(X_test_scaled) for q in quantiles}

plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual', marker='o', color='black')
plt.plot(y_test.index, predictions[0.5], label='Predicted', marker='x', color='red')
plt.fill_between(y_test.index, predictions[0.025], predictions[0.975],
                 alpha=0.3, label='95% CI')
plt.title("Test Set Forecast with Confidence Interval")
plt.xlabel("Date")
plt.ylabel("Gross Revenue")
plt.legend()
plt.tight_layout()
plt.show()

# 9. Residual Correction for Training and Test
train_preds = final_model.predict(X_train_scaled)
train_residuals = y_train - train_preds
residual_model = lgb.LGBMRegressor()
residual_model.fit(X_train_scaled, train_residuals)

y_train_pred_adjusted = train_preds + residual_model.predict(X_train_scaled)
y_test_pred_adjusted = final_model.predict(X_test_scaled) + residual_model.predict(X_test_scaled)

evaluate_forecast("Final Model (Train)", y_train, y_train_pred_adjusted)
evaluate_forecast("Final Model (Test)", y_test, y_test_pred_adjusted)

# 10. Full Time-Series Visualization
plt.figure(figsize=(14, 7))
plt.plot(y.index, y, label='Actual Gross Revenue', color='black', linewidth=2)
plt.plot(y_train.index, y_train, label='Training Actual', color='green', linewidth=2)
plt.plot(y_test.index, y_test, label='Test Actual', color='blue', linewidth=2)
plt.plot(y_test.index, y_test_pred_adjusted, label='Test Prediction (Adjusted)', color='red', linestyle='--', linewidth=2)
plt.axvline(x=y_test.index[0], color='gray', linestyle='--', label='Train/Test Split')
plt.title("Gross Revenue Forecast: Full Time-Series (80% Train, 20% Test)")
plt.xlabel("Date")
plt.ylabel("Gross Revenue")
plt.legend()
plt.tight_layout()
plt.show()

print(df)
