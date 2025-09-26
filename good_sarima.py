import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def evaluate_forecast(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mpe = np.mean((actual - predicted) / actual) * 100
    rmspe = np.sqrt(np.mean(np.square((actual - predicted) / actual))) * 100
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    
    print(f"\n[{model_name}] test performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAPE: {mape:.2f}")
    print(f"MPE: {mpe:.2f}")
    print(f"RMSPE: {rmspe:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"Accuracy: {acc:.2f}")



def analyze_data(df, col):
    print(f"Descriptive statistics for {col}:")
    print(df[col].describe())  # mean, std, min, max, quartiles

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.figure()

    # ASPV: shows variation in monthly data over all years
    monthly_avg = df.groupby(df.index.month)[col].mean()
    total_avg = df[col].mean()
    aspv = monthly_avg / total_avg
    plt.subplot(1, 2, 1)
    plt.bar(months, aspv)
    plt.title("Average Seasonal Price Variation (ASPV)")
    plt.xlabel("Month")
    plt.ylabel("Relative Price")
    print(aspv)
    hspi = monthly_avg.max()
    lspi = monthly_avg.min()
    overall_aspv = ((hspi - lspi) / ((hspi + lspi) / 2)) * 100
    print("Single ASPV for the whole dataset:", overall_aspv)

    # CV: shows variability of data
    monthly_std = df.groupby(df.index.month)[col].std()
    monthly_mean = df.groupby(df.index.month)[col].mean()
    cv = monthly_std / monthly_mean
    plt.subplot(1, 2, 2)
    plt.bar(months, cv)
    plt.title("Monthly Coefficient of Variation (CV)")
    plt.xlabel("Month")
    plt.ylabel("STD / Mean")
    print(cv)
    print("Single CV for the whole dataset:", df[col].std() / df[col].mean())

    # IPR: shows % increase in data between min and max values within a year
    yearly_max = df.groupby(df.index.year)[col].max()
    yearly_min = df.groupby(df.index.year)[col].min()
    ipr = (yearly_max / yearly_min).mean()

    # CDVI: shows instability of data
    total_cv = df[col].std() / df[col].mean()
    monthly_means = df.groupby(df.index.month)[col].mean()
    cdvi = (monthly_means.std() / monthly_means.mean()) / total_cv

    print("Intra Year Price Rise (IPR):", ipr)
    print("Cuddy Della Valle Index (CVDI):", cdvi)

    # ADF: check for stationarity (p > 0.05 indicates series is non-stationary)
    print("ADF p-value (after differencing):", adfuller(df[col])[1])

    # ACF and PACF plots
    plt.figure()
    plt.subplot(1,2,1)
    plot_acf(df[col], lags=40, ax=plt.gca())
    plt.title("ACF plot")

    plt.subplot(1,2,2)
    plot_pacf(df[col], lags=40, ax=plt.gca(), method='ols')  # method='ols' because original data is nonstationary
    plt.title("PACF plot")
    plt.tight_layout()
    plt.show()

    # Find optimal model order using auto_arima
    global stepwise_model
    stepwise_model = auto_arima(df['Real_Price'], m=12, seasonal=True, trace=True)
    print(stepwise_model.order, stepwise_model.seasonal_order) # (2, 1, 0)


def analyze_residuals(residuals):

    # plot residuals over time
    plt.figure()
    plt.plot(residuals)
    plt.title('Residuals over time')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.grid(True)
    plt.show()

    # histogram and density plot of residuals
    plt.figure()
    plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g')
    residuals = pd.Series(residuals)
    residuals.plot(kind='kde')
    plt.title('Histogram and density of residuals')
    plt.show()

    # ACF plot of residuals (check autocorrelation)
    plot_acf(residuals, lags=40)
    plt.title('ACF of residuals')
    plt.show()

    # PACF plot of residuals
    plot_pacf(residuals, lags=40)
    plt.title('PACF of residuals')
    plt.show()

    # ljung-box test on residuals
    ljung_box_results = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print("Ljung-Box test results:")
    print(ljung_box_results)



def main():
    # 1. Data Collection and Preparation: collect data, process/clean data, normalize data, analyze data

    df = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df = df.asfreq('MS') # tells Pandas that the data is monthly
    target_col = 'Gross_Revenue'
    df[target_col].interpolate(inplace=True) # interpolates any NaN data
    

    # adjust for inflation
    target_col = 'Gross_Revenue'
    df['Real_Price'] = df[target_col] / df['CPI'] * 100

    # plot inflation adjusted data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Real_Price'], marker='o', linestyle='-')
    plt.title('Inflation Adjusted Gross Revenue Over Time')
    plt.xlabel('Date')
    plt.ylabel('Inflation Adjusted Gross Revenue')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # normalize using MinMaxScaler
    scaler = MinMaxScaler()
    df['normalized'] = scaler.fit_transform(df[['Real_Price']])
    target_col = 'normalized'

    # manually difference data to account for seasonality
    df['seasonal_diff'] = df[target_col] - df[target_col].shift(12)
    df.dropna(subset=['seasonal_diff'], inplace=True)
    df['final_diff'] = df['seasonal_diff'] - df['seasonal_diff'].shift(1)
    df.dropna(subset=['final_diff'], inplace=True)

    # ADF: check for stationarity (p > 0.05 indicates series is non-stationary)
    print("ADF p-value (after differencing):", adfuller(df['final_diff'])[1])
    # ASPV and CV plots, IPR and CDVI indices, ADF p-value, ACF and PACF plots
    analyze_data(df, 'Real_Price')
    

    # 2. Model Identification (Box-Jenkins Step 1)
    
    # do an 80/20 train-test split
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    y_train = train['final_diff'].values
    y_test = test[target_col].values
    dates_train = train.index
    dates_test = test.index
    
    
    # 3. Parameter Estimation (Box-Jenkins Step 2)

    model = SARIMAX(y_train, order=stepwise_model.order, seasonal_order=stepwise_model.seasonal_order).fit()
    print(model.summary())


    # 4. Diagnostic Checking (Box-Jenkins Step 3)

    # save residuals to a csv for model comparison
    '''
    residuals_df = pd.DataFrame({'ARIMA Residuals': model.resid}, index=train.index)
    residuals_df.to_csv('arima_model_residuals.csv')
    '''
    analyze_residuals(model.resid)


    # 5-6. Model Evaluation and Forecasting (Box-Jenkins Step 4)

    # forecast differenced values
    forecast_stats = model.get_forecast(steps=len(y_test))
    diff_forecast = forecast_stats.predicted_mean
    confidence_intervals = forecast_stats.conf_int()

    # invert differencing
    last_seasonal_diff = train['seasonal_diff'].iloc[-1]
    seasonal_forecast = last_seasonal_diff + np.cumsum(diff_forecast)
    seasonal_base_values = df[target_col].iloc[split_idx - 12 : split_idx - 12 + len(test)].values
    original_test = test[target_col].values
    last_seasonal_value = train['seasonal_diff'].iloc[-1]
    seasonal_forecast = [last_seasonal_value + diff_forecast[0]]
    seasonal_forecast = last_seasonal_value + np.cumsum(diff_forecast)
    seasonal_forecast = np.array(seasonal_forecast)
    seasonal_base_values = df[target_col].iloc[split_idx - 12: split_idx - 12 + len(test)].values

    # undo seasonal differencing
    lower_diff = confidence_intervals[:, 0]
    upper_diff = confidence_intervals[:, 1]
    lower_seasonal = train['seasonal_diff'].iloc[-1] + np.cumsum(lower_diff)
    upper_seasonal = train['seasonal_diff'].iloc[-1] + np.cumsum(upper_diff)

    if len(seasonal_base_values) < len(lower_seasonal):
        pad_len = len(lower_seasonal) - len(seasonal_base_values)
        seasonal_base_values = np.pad(seasonal_base_values, (0, pad_len), mode='edge')

    lower_bound = lower_seasonal + seasonal_base_values
    upper_bound = upper_seasonal + seasonal_base_values

    if len(seasonal_base_values) < len(seasonal_forecast):
        pad_len = len(seasonal_forecast) - len(seasonal_base_values)
        seasonal_base_values = np.pad(seasonal_base_values, (0, pad_len), mode='edge')

    # rescale data back to original
    original_pred = scaler.inverse_transform((seasonal_forecast + seasonal_base_values).reshape(-1, 1)).flatten()
    original_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    lower_bound = scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()

    # save predicted-actual to csv
    residuals = original_test - original_pred
    residuals_df = pd.DataFrame({'ARIMA Residuals': residuals}, index=dates_test)
    residuals_df.to_csv('arima_residuals.csv')

    full_dates = np.concatenate([dates_train, dates_test])
    # full_actual = np.concatenate([train['Gross_Revenue'].values, test['Gross_Revenue'].values])

    # Plot results
    plt.figure()
    plt.plot(full_dates, df['Real_Price'], label='Actual', color='blue', marker='o')
    plt.axvline(x=dates_test[0], color='gray', linestyle='--', label='Train-Test Split')

    plt.plot(dates_test, original_pred, label='Predicted', color='red', marker='x')
    #plt.fill_between(dates_test, lower_bound, upper_bound, color='red', alpha=0.3, label='Confidence Interval')

    plt.title("ARIMA forecast: Actual vs Predicted (Train + Test)")
    plt.xlabel("Date")
    plt.ylabel("Inflation Adjusted Gross Revenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    evaluate_forecast("ARIMA with Manual Seasonal Differencing", original_test, original_pred)

    forecast_comparison = pd.DataFrame({
    "Date": dates_test,
    "Actual": original_test,
    "Forecast_SARIMA": original_pred,
    "Lower_CI": lower_bound,
    "Upper_CI": upper_bound
    })
    forecast_comparison.to_csv("/Users/mihir/Code/Beef_Value_Chain/Cow_Calf/Plot/updated_sarima_forecast_comparison.csv", index=False)




if __name__ == "__main__":
    main()
