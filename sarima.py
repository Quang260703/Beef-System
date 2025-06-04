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
    print(f"\n[{model_name}] Test Performance:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    print(f"Accuracy (1 - MAE/mean * 100): {acc:.2f}%")
    
def main():
    # read CSV
    data = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # define columns
    target_col = 'Gross_Revenue'
    
    # 80/20 split
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test  = data.iloc[split_idx:]
    
    endog_train = train[target_col]
    endog_test  = test[target_col]
    
    dates_train = train['Date']
    dates_test  = test['Date']

    # list to store rolling predictions
    rolling_preds = []

    # you can adjust these orders
    order = (1,1,2)
    seasonal_order = (0,1,1,12)

    # walk-forward over each row in test
    for i in range(len(test)):
        # up to train + i
        idx = split_idx + i

        current_endog = data[target_col].iloc[:idx]

        # define SARIMAX model
        model = SARIMAX(
            current_endog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # fit
        results = model.fit(disp=False)

        # forecast next point at 'idx'
        pred = results.predict(start=idx, end=idx)

        # store the single-step forecast
        rolling_preds.append(pred.iloc[0])

    # align predictions with test
    rolling_preds_series = pd.Series(rolling_preds, index=test.index)

    # evaluate on test
    evaluate_forecast("SARIMAX Rolling", endog_test, rolling_preds_series)

    # plot
    plt.figure(figsize=(12,6))

    full_dates = pd.concat([dates_train, dates_test])
    full_actual = pd.concat([endog_train, endog_test])
    
    # entire actual data in blue
    plt.plot(full_dates, full_actual, label='Actual (All)', color='blue', marker='o')

    # rolling forecast in red
    plt.plot(dates_test, rolling_preds_series, label='Test Prediction (Rolling)', color='red', marker='x')

    # mark train/test boundary
    plt.axvline(x=dates_test.iloc[0], color='gray', linestyle='--', label='Train-Test Split')

    plt.title("SARIMAX Rolling-Fit: Actual vs. Predicted")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
