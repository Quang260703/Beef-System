#underfitting issue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    print(f"\n[{model_name}] Test Performance:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

def main():
    # Load and prepare data
    data = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    target_col = 'Gross_Revenue'
    exog_cols = ['Net_Gas_Price', 'Corn_Price', 'CPI', 'Exchange_Rate_JPY_USD']
    
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test  = data.iloc[split_idx:]
    
    y_train = train[target_col]
    y_test  = test[target_col]
    X_train = train[exog_cols]
    X_test  = test[exog_cols]
    dates_train = train['Date']
    dates_test  = test['Date']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Define the objective function for Optuna
    def objective(trial):
        n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 150])
        learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        
        base_est = DecisionTreeRegressor(random_state=42, max_depth=max_depth)
        adb = AdaBoostRegressor(estimator=base_est,
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                random_state=42)
        
        errors = []
        for train_index, val_index in tscv.split(X_train_scaled):
            X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
            y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
            adb.fit(X_train_cv, y_train_cv)
            y_pred_cv = adb.predict(X_val_cv)
            error = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
            errors.append(error)
        return np.mean(errors)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print("\nBest Parameters:", study.best_params)
    
    best_adb = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(random_state=42, max_depth=study.best_params["max_depth"]),
        n_estimators=study.best_params["n_estimators"],
        learning_rate=study.best_params["learning_rate"],
        random_state=42
    )
    
    # Retrain on entire training set and evaluate on test set
    best_adb.fit(X_train_scaled, y_train)
    train_pred = best_adb.predict(X_train_scaled)
    test_pred  = best_adb.predict(X_test_scaled)
    
    evaluate_forecast("AdaBoost (Train)", y_train, train_pred)
    evaluate_forecast("AdaBoost", y_test, test_pred)
    
    # Plot results
    full_dates  = pd.concat([dates_train, dates_test])
    full_actual = pd.concat([y_train, y_test])
    full_pred   = np.concatenate([train_pred, test_pred])
    
    plt.figure(figsize=(12,6))
    plt.plot(full_dates, full_actual, label='Actual (All)', color='blue', marker='o')
    plt.plot(dates_train, train_pred, label='Predicted (Train)', color='green', marker='x')
    plt.plot(dates_test, test_pred, label='Predicted (Test)', color='red', marker='x')
    plt.axvline(x=dates_test.iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("AdaBoost: Actual vs. Predicted (Train + Test)")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
