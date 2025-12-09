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
    model = SARIMAX(
    y_train,
    order=stepwise_model.order,
    seasonal_order=stepwise_model.seasonal_order,
    ).fit()

    print(model.summary())

    # =======================================================
    # 2. Residual Diagnostics (Box-Jenkins Step 3)
    # =======================================================
    analyze_residuals(model.resid)

    # =======================================================
    # 3. WALK-FORWARD VALIDATION (Box-Jenkins Step 4)
    # =======================================================

    predictions = []
    lower_bounds = []
    upper_bounds = []

    history = train.copy()     # rolling window grows each step

    for t in range(len(test)):

        # Fit SARIMA on current history
        model = SARIMAX(
            history['seasonal_diff'],
            order=stepwise_model.order,
            seasonal_order=stepwise_model.seasonal_order,
        ).fit()

        # Forecast ONE step
        forecast_stats = model.get_forecast(steps=1)
        diff_forecast = forecast_stats.predicted_mean.iloc[0]

        ci = forecast_stats.conf_int()
        lower_diff = ci.iloc[0, 0]
        upper_diff = ci.iloc[0, 1]

        # ----------------------------
        # Undo seasonal differencing
        # ----------------------------
        last_seasonal = history['seasonal_diff'].iloc[-1]

        seasonal_forecast = last_seasonal + diff_forecast
        lower_seas = last_seasonal + lower_diff
        upper_seas = last_seasonal + upper_diff

        # seasonal base value y(t-12)
        seasonal_base = df[target_col].iloc[split_idx - 12 + t]

        y_pred_scaled = seasonal_forecast + seasonal_base
        y_low_scaled = lower_seas + seasonal_base
        y_up_scaled  = upper_seas + seasonal_base

        # ----------------------------
        # Invert scaling
        # ----------------------------
        y_pred = scaler.inverse_transform([[y_pred_scaled]])[0][0]
        y_low  = scaler.inverse_transform([[y_low_scaled]])[0][0]
        y_up   = scaler.inverse_transform([[y_up_scaled]])[0][0]

        predictions.append(y_pred)
        lower_bounds.append(y_low)
        upper_bounds.append(y_up)

        # ----------------------------
        # Expand rolling history using TRUE test value
        # ----------------------------
        actual_val = test.iloc[t][target_col]
        past_val = df[target_col].iloc[split_idx - 12 + t]

        new_seasonal_diff = actual_val - past_val

        history = pd.concat([
            history,
            pd.DataFrame({"seasonal_diff": [new_seasonal_diff]})
        ])


    # =======================================================
    # 4. Evaluation + Plotting
    # =======================================================

    # Convert test data back
    original_test = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # Save residuals
    residuals = original_test - np.array(predictions)
    residuals_df = pd.DataFrame({'SARIMA Residuals': residuals}, index=dates_test)
    residuals_df.to_csv('sarima_residuals.csv')

    # Full plot
    full_dates = np.concatenate([dates_train, dates_test])

    plt.figure(figsize=(12,6))
    plt.plot(full_dates, df['Real_Price'], label='Actual', color='blue', marker='o')
    plt.axvline(x=dates_test[0], color='gray', linestyle='--', label='Train-Test Split')

    plt.plot(dates_test, predictions, label='Walk-Forward Predicted', color='red', marker='x')
    # plt.fill_between(dates_test, lower_bounds, upper_bounds, alpha=0.3, color='red')

    plt.title("Walk-Forward SARIMA Forecast: Actual vs Predicted (Train + Test)")
    plt.xlabel("Date")
    plt.ylabel("Inflation Adjusted Gross Revenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    evaluate_forecast(
        "Walk-Forward SARIMA with Manual Seasonal Differencing",
        original_test,
        predictions
    )

if __name__ == "__main__":
    main()
