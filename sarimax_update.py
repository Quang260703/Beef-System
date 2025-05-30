import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    print(f"Accuracy (1 - MAE/mean * 100): {acc:.2f}%")

def main():
    data = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    target_col = 'Gross_Revenue'

    # Lag features with dynamic window sizing
    max_lag = 6  # Maximum lag period in months
    for lag in range(max_lag):
        data[f'Gross_Revenue_lag{lag}'] = data['Gross_Revenue'].shift(lag+1)

    # Rolling statistics of the previous 3 months
    data[f'{target_col}_Rolling_Means'] = data[target_col].shift(1).rolling(window=3).mean()
    data[f'{target_col}_Rolling_STD'] = data[target_col].shift(1).rolling(window=3).std()
    
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    exog_cols = [col for col in data.columns if col != target_col and col != "Date"]

    
    split_idx = int(len(data) * 0.6)
    train = data.iloc[:split_idx]
    test  = data.iloc[split_idx:]
    
    endog_train = train[target_col]
    endog_test  = test[target_col]
    exog_train = train[exog_cols]
    exog_test  = test[exog_cols]
    dates_train = train['Date']
    dates_test  = test['Date']

    # We'll search over a small grid of p,d,q,P,D,Q around (0..2).
    
    p_values = [0,1,2]
    d_values = [0,1]
    q_values = [0,1,2]
    P_values = [0,1]
    D_values = [0,1]
    Q_values = [0,1]
    m = 12  # monthly seasonality

    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    best_model = None

    # Loop over possible orders
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            try:
                                model = SARIMAX(
                                    endog_train,
                                    exog=exog_train,
                                    order=(p,d,q),
                                    seasonal_order=(P,D,Q,m),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                results = model.fit(disp=False)
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p,d,q)
                                    best_seasonal_order = (P,D,Q,m)
                                    best_model = results
                            except:
                                # some combos won't converge; just skip
                                pass

    print(f"Best AIC: {best_aic:.2f} for order={best_order} seasonal={best_seasonal_order}")
    
    # If no model found, we bail
    if not best_model:
        print("No model converged. Try expanding or altering the grid.")
        return

    # Use the best_model for final predictions
    train_pred = best_model.predict(
        start=endog_train.index[0],
        end=endog_train.index[-1],
        exog=exog_train
    )
    test_pred = best_model.predict(
        start=endog_test.index[0],
        end=endog_test.index[-1],
        exog=exog_test
    )

    # Evaluate on test
    evaluate_forecast(f"SARIMAX Training", endog_train, train_pred)
    evaluate_forecast(f"SARIMAX{best_order}x{best_seasonal_order}", endog_test, test_pred)
    
    # Plot
    plt.figure(figsize=(12,6))
    
    full_dates = pd.concat([dates_train, dates_test])
    full_actual = pd.concat([endog_train, endog_test])
    
    plt.plot(full_dates, full_actual, label='Actual (All)', color='blue', marker='o')
    plt.plot(dates_test, test_pred, label='Test Prediction', color='red', marker='x')
    plt.axvline(x=dates_test.iloc[0], color='gray', linestyle='--', label='Train-Test Split')
    plt.title("SARIMAX Best-Order Model: Actual vs. Predicted")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
