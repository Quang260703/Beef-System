import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pyswarm import pso

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

def mrmr(X, y, k):
    mi = mutual_info_regression(X, y)
    mi = pd.Series(mi, index=X.columns)
    selected = [mi.idxmax()]
    
    for _ in range(k - 1):
        remaining = list(set(X.columns) - set(selected))
        score = {}
        for feat in remaining:
            redundancy = np.nanmean([X[feat].corr(X[s]) for s in selected])
            score[feat] = mi[feat] - redundancy
        selected.append(max(score, key=score.get))
    
    return selected

def svr_pso_objective(params, X, y):
    C, epsilon, gamma = params
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)

    return np.mean(rmses)

def main():
    # 1) LOAD AND PREPARE DATA
    df = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    target_col = 'Gross_Revenue'

    max_lag = 6
    for lag in range(max_lag):
        df[f'Gross_Revenue_lag{lag}'] = df['Gross_Revenue'].shift(lag+1)

    df[f'{target_col}_Rolling_Means'] = df[target_col].shift(1).rolling(window=3).mean()
    df[f'{target_col}_Rolling_STD'] = df[target_col].shift(1).rolling(window=3).std()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    exog_cols = [col for col in df.columns if col != target_col and col != 'Date']

    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train_full = train_df[exog_cols]
    y_train_full = train_df[target_col]
    X_test_full = test_df[exog_cols]
    y_test_full = test_df[target_col]

    # MRMR feature selection
    k = 5
    selected_features = mrmr(X_train_full, y_train_full, k)
    print(f"Selected features using MRMR: {selected_features}")

    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]

    # PSO bounds for C, epsilon, gamma
    lb = [0.1, 0.001, 0.0001]
    ub = [100, 1, 10]

    print("Starting PSO hyperparameter tuning...")
    best_params, best_rmse = pso(svr_pso_objective, lb, ub, args=(X_train, y_train_full), swarmsize=30, maxiter=50)
    C, epsilon, gamma = best_params
    print(f"Best PSO parameters:\nC={C:.4f}, epsilon={epsilon:.4f}, gamma={gamma:.4f}")
    print(f"Best CV RMSE: {best_rmse:.4f}")

    # Rolling Forecast with best params
    rolling_preds = []
    for i in range(len(test_df)):
        idx = split_idx + i
        current_df = df.iloc[:idx]

        X_current = current_df[selected_features]
        y_current = current_df[target_col]

        scaler = MinMaxScaler()
        X_current_scaled = scaler.fit_transform(X_current)

        model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')

        model.fit(X_current_scaled, y_current)

        X_next = df[selected_features].iloc[idx:idx+1]
        X_next_scaled = scaler.transform(X_next)
        pred = model.predict(X_next_scaled)[0]
        rolling_preds.append(pred)

    rolling_preds_series = pd.Series(rolling_preds, index=test_df.index)

    # Train final model on full training set
    scaler_full = MinMaxScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)

    final_model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')

    final_model.fit(X_train_scaled, y_train_full)

    y_train_pred = final_model.predict(X_train_scaled)
    train_preds_series = pd.Series(y_train_pred, index=train_df.index)

    # Evaluate train and test performance
    evaluate_forecast("SVR Training", y_train_full, train_preds_series)
    evaluate_forecast("SVR Rolling", y_test_full, rolling_preds_series)

    # Plot results
    plt.figure(figsize=(14, 6))
    plt.plot(train_df['Date'], y_train_full, label='Train Actual', color='blue', marker='o')
    plt.plot(test_df['Date'], y_test_full, label='Test Actual', color='black', marker='o')
    plt.plot(test_df['Date'], rolling_preds_series, label='Test Prediction (Rolling)', color='green', marker='x')
    plt.axvline(x=test_df['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("SVR Rolling-Fit (Grid Search + TimeSeriesSplit CV): Actual vs. Predicted")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()