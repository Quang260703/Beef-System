#overfitting issue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
#warnings.filterwarnings('ignore', category=FutureWarning)

def evaluate_forecast(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    
    print(f"\n[{model_name}] test performance:")
    print(f"mae: {mae:.2f}")
    print(f"rmse: {rmse:.2f}")
    print(f"r²: {r2:.2f}")
    print(f"Acc: {acc:.2f}")
    

def main():
    # load and sort data
    data = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    print("data info:")
    print(data.head())
    
    # define target column
    target_col = 'Gross_Revenue'
    
    # do an 40/60 train-test split
    split_idx = int(len(data) * 0.1)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    y_train = train[target_col].values
    y_test = test[target_col].values
    dates_train = train['Date']
    dates_test = test['Date']
    
    # naive forecast: each new point = the last actual
    # let's do a rolling naive for test
    test_pred = np.zeros_like(y_test)
    
    # the first test prediction is the last train value
    test_pred[0] = y_train[-1]
    # for subsequent test points, we use the previous test actual
    for i in range(1, len(y_test)):
        test_pred[i] = y_test[i-1]
    
    # evaluate
    train_pred = np.zeros_like(y_train)
    train_pred[0] = y_train[0]
    for i in range(1, len(y_train)):
        train_pred[i] = y_train[i - 1]

    evaluate_forecast("naive forecast (train)", y_train, train_pred)
    evaluate_forecast("naive forecast", y_test, test_pred)
    
    # plot actual vs predicted
    full_dates = pd.concat([dates_train, dates_test])
    full_actual = np.concatenate([y_train, y_test])
    # for train portion, we just keep the actual values
    full_pred = np.concatenate([y_train, test_pred])
    
    plt.figure(figsize=(12, 6))
    plt.plot(full_dates, full_actual, label='actual (all)', color='blue', marker='o')
    
    # mark train-test split
    plt.axvline(x=dates_test.iloc[0], color='gray', linestyle='--', label='train-test split')
    
    # show test predictions
    plt.plot(dates_train, train_pred, label='Predicted (Train)', color='green', marker='x')
    plt.plot(dates_test, test_pred, label='predicted (naive)', color='red', marker='x')
    
    plt.title("naive forecast: actual vs predicted (train + test)")
    plt.xlabel("date")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
