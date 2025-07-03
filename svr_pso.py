import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pyswarm import pso

def evaluate_forecast(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mse = mean_squared_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mpe = np.mean((actual - predicted) / actual) * 100
    rmspe = np.sqrt(np.mean(((actual - predicted) / actual) ** 2)) * 100
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    print(f"\n[{model_name}] Test Performance:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%, MPE: {mpe:.2f}%, RMSPE: {rmspe:.2f}%")
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    print(f"Accuracy (1 - MAE/mean * 100): {acc:.2f}%")

def mrmr(X, y, k, plot=True, mi_threshold=0.01):
    mi = mutual_info_regression(X, y, random_state=42)
    mi = pd.Series(mi, index=X.columns)
    
    mi_filtered = mi[mi > mi_threshold]
    if len(mi_filtered) == 0:
        raise ValueError("No features have MI above threshold. Lower the threshold.")
    
    selected = [mi_filtered.idxmax()]
    all_scores = {feat: [mi[feat]] for feat in X.columns}
    
    for _ in range(k - 1):
        remaining = list(set(X.columns) - set(selected))
        score = {}
        
        for feat in remaining:
            redundancy = np.nanmean([X[feat].corr(X[s]) for s in selected])
            score[feat] = mi[feat] - redundancy
        
        best = max(score, key=score.get)
        selected.append(best)

        for feat in X.columns:
            if feat in score:
                all_scores[feat].append(score[feat])
            else:
                all_scores[feat].append(all_scores[feat][-1])
    
    if plot:
        plt.figure(figsize=(12, 6))
        for feat, scores in all_scores.items():
            plt.plot(scores, label=feat, alpha=0.7)
        plt.xlabel("Selection Step")
        plt.ylabel("mRMR Score (MI - Correlation)")
        plt.title("mRMR Score Evolution of All Features")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.show()

    return selected, all_scores

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
    selected_features, all_scores = mrmr(X_train_full, y_train_full, k)
    print(f"Selected features using MRMR: {selected_features}")
    # Create a list of tuples (feature, last_score), replacing None with 0.0
    feat_scores = [(feat, scores[-1] if scores[-1] is not None else 0.0) 
                for feat, scores in all_scores.items()]

    # Sort descending by last_score
    feat_scores_sorted = sorted(feat_scores, key=lambda x: x[1], reverse=True)

    # Print sorted results
    for feat, val in feat_scores_sorted:
        print(f"{feat}: {val:.4f}")

    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]

    # Standardize before PCA (different from MinMaxScaler used for SVR)
    scaler_std = StandardScaler()
    X_train_std = scaler_std.fit_transform(X_train)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)

    explained_var = pca.explained_variance_ratio_ * 100 

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_full, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label=target_col)
    plt.xlabel(f'Principal Component 1 ({explained_var[0]:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({explained_var[1]:.1f}% variance)')
    plt.title(f'PCA of Training Features\n(PC1 + PC2 explain {explained_var.sum():.1f}% of variance)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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