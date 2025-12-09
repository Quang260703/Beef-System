import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

# Export forecast residuals, predictions, and error metrics to CSV files.
def export_forecast_results(
    model_name: str,
    dates: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
    filename_prefix: str = "svr_forecast_static"
):
    results_df = pd.DataFrame({
        'Date': dates.values,
        'Actual': actual.values,
        'Predicted': predicted.values,
        'Residual': actual.values - predicted.values,
    })

    results_csv = f"{filename_prefix}_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"[{model_name}] Residuals and predictions exported to: {results_csv}")

    # Compute error metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mse = mean_squared_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mpe = np.mean((actual - predicted) / actual) * 100
    rmspe = np.sqrt(np.mean(((actual - predicted) / actual) ** 2)) * 100
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R²', 'MSE', 'MAPE', 'MPE', 'RMSPE'],
        'Value': [mae, rmse, r2, mse, mape, mpe, rmspe]
    })

    metrics_csv = f"{filename_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[{model_name}] Evaluation metrics exported to: {metrics_csv}")

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

def main():
    # 1) LOAD AND PREPARE DATA
    df = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    target_col = 'Gross_Revenue'

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    exog_cols = [col for col in df.columns if col != target_col and col != "Date"]

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]
    
    X_train_full = train_df[exog_cols]
    y_train_full = train_df[target_col]
    X_test_full  = test_df[exog_cols]
    y_test_full  = test_df[target_col]

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

    X_train_full = X_train_full[selected_features]

    # Standardize before PCA (different from MinMaxScaler used for SVR)
    scaler_std = StandardScaler()
    X_train_std = scaler_std.fit_transform(X_train_full)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)

    # Create a DataFrame of PCA loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X_train_full.columns,
        columns=['PCA1', 'PCA2']
    )
    print("PCA Component Loadings:")
    print(loadings)
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

    plt.figure(figsize=(10, 5))
    loadings[['PCA1', 'PCA2']].plot(kind='bar', figsize=(12, 6))
    plt.title('Feature Contributions to PCA1 and PCA2')
    plt.ylabel('Loading Value')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
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
            scaler = MinMaxScaler()
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

    # Scale training features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(train_df[selected_features])
    y_train = train_df[target_col]

    # Train SVR with best hyperparameters on full training set
    model = SVR(**best_params)
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    X_test_scaled = scaler.transform(test_df[selected_features])
    static_preds = model.predict(X_test_scaled)

    # Convert to Series for easy evaluation
    static_preds_series = pd.Series(static_preds, index=test_df.index)

    # Training predictions
    y_train_pred = model.predict(X_train_scaled)
    train_preds_series = pd.Series(y_train_pred, index=train_df.index)

    # Evaluate
    evaluate_forecast("SVR Training", y_train, train_preds_series)
    evaluate_forecast("SVR Static", test_df[target_col], static_preds_series)

    plt.figure(figsize=(14, 6))

    plt.plot(train_df['Date'], y_train, label='Train Actual', color='blue', marker='o')
    plt.plot(test_df['Date'], test_df[target_col], label='Test Actual', color='black', marker='o')
    plt.plot(test_df['Date'], static_preds_series, label='Test Prediction (Static)', color='red', marker='x')

    plt.axvline(x=test_df['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("SVR Static Prediction (with Optuna Hyperparams)")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    export_forecast_results(
        model_name="SVR Static",
        dates=test_df['Date'],
        actual=y_test_full,
        predicted=static_preds_series,
        filename_prefix="svr_forecast"
    )


if __name__ == "__main__":
    main()
