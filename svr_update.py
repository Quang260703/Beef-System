import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

def evaluate_forecast(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    
    print(f"\n[{model_name}] Test Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"Accuracy (1 - MAE/mean * 100): {acc:.2f}%")

def main():
    # 1) LOAD AND PREPARE DATA
    df = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    target_col = 'Gross_Revenue'

    # Lag features with dynamic window sizing
    max_lag = 6  # Maximum lag period in months
    for lag in range(max_lag):
        df[f'Gross_Revenue_lag{lag}'] = df['Gross_Revenue'].shift(lag+1)

    # Rolling statistics of the previous 3 months
    df[f'{target_col}_Rolling_Means'] = df[target_col].shift(1).rolling(window=3).mean()
    df[f'{target_col}_Rolling_STD'] = df[target_col].shift(1).rolling(window=3).std()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    exog_cols = [col for col in df.columns if col != target_col and col != "Date"]

    split_idx = int(len(df) * 0.1)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]
    
    X_train_full = train_df[exog_cols]
    y_train_full = train_df[target_col]
    X_test_full  = test_df[exog_cols]
    y_test_full  = test_df[target_col]
    
    # 2) HYPERPARAM TUNING WITH OPTUNA ON THE TRAINING SET ONLY
    def objective(trial):
        C = trial.suggest_float('C', 1e-1, 1e1, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-3, 1, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear','rbf','poly'])

        params = {'C': C, 'epsilon': epsilon, 'kernel': kernel}

        if kernel == 'rbf':
            gamma = trial.suggest_float('gamma', 1e-3, 1e1, log=True)
            params['gamma'] = gamma
        elif kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)
            gamma = trial.suggest_float('gamma', 1e-3, 1e1, log=True)
            params['degree'] = degree
            params['gamma'] = gamma

        tscv = TimeSeriesSplit(n_splits=5)
        rmses = []
        for train_idx, val_idx in tscv.split(X_train_full):
            X_tr = X_train_full.iloc[train_idx]
            y_tr = y_train_full.iloc[train_idx]
            X_val = X_train_full.iloc[val_idx]
            y_val = y_train_full.iloc[val_idx]

            # scale features
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            model = SVR(**params)
            model.fit(X_tr_scaled, y_tr)
            pred_val = model.predict(X_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_val, pred_val))
            rmses.append(rmse)

        return np.mean(rmses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print("Best hyperparameters from Optuna:", best_params)

    # 3) ROLLING (WALK-FORWARD) FORECAST ON THE TEST SET
    rolling_preds = []
    
    for i in range(len(test_df)):
        # up to train + i rows
        idx = split_idx + i
        
        # define current training portion
        current_df = df.iloc[:idx]
        
        X_current = current_df[exog_cols]
        y_current = current_df[target_col]

        # scale features
        scaler = StandardScaler()
        X_current_scaled = scaler.fit_transform(X_current)

        # train SVR with best hyperparams
        model = SVR(**best_params)
        model.fit(X_current_scaled, y_current)

        # predict the next point (idx)
        X_next = df[exog_cols].iloc[idx:idx+1]
        X_next_scaled = scaler.transform(X_next)
        pred = model.predict(X_next_scaled)[0]
        rolling_preds.append(pred)

    # align rolling_preds with test index
    rolling_preds_series = pd.Series(rolling_preds, index=test_df.index)

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(train_df[exog_cols])

    model_full = SVR(**best_params)
    model_full.fit(X_train_full_scaled, y_train_full)

    y_train_pred_full = model_full.predict(X_train_full_scaled)
    train_preds_series = pd.Series(y_train_pred_full, index=train_df.index)

    # Evaluate training error
    evaluate_forecast("SVR Training", y_train_full, train_preds_series)

    # 4) EVALUATE AND PLOT
    evaluate_forecast("SVR Rolling", y_test_full, rolling_preds_series)

    # Plot
    plt.figure(figsize=(14, 6))

    plt.plot(train_df['Date'], y_train_full, label='Train Actual', color='blue', marker='o')
    plt.plot(test_df['Date'], y_test_full, label='Test Actual', color='black', marker='o')
    plt.plot(test_df['Date'], rolling_preds_series, label='Test Prediction (Rolling)', color='red', marker='x')
    
    plt.axvline(x=test_df['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("SVR Rolling-Fit (with Optuna Hyperparams): Actual vs. Predicted")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
