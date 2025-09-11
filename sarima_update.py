import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.dates as mdates

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# Perform Augmented Dickey-Fuller test for stationarity
def test_stationarity(series, title="ADF Test"):
    print(f"\n{title}")
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for k, v in result[4].items():
        print(f'\t{k}: {v:.4f}')
    return result[1] < 0.05  # True if stationary

# Analyze ACF and PACF plots
def analyze_acf_pacf(series, lags=24, title=""):
    print(f"\nACF/PACF Analysis: {title}")
    
    # Calculate ACF and PACF correctly
    acf_vals = acf(series, nlags=lags, fft=False)
    pacf_vals = pacf(series, nlags=lags, method='ywm')  # Correct PACF calculation
    
    # Plot ACF
    sm.graphics.tsa.plot_acf(series, lags=lags, fft=False, title=f'ACF - {title}')
    plt.show()
    
    # Plot PACF
    sm.graphics.tsa.plot_pacf(series, lags=lags, method='ywm', title=f'PACF - {title}')
    plt.show()
    
    # Interpretation guidance
    print("\nInterpretation Guidance:")
    seasonal_lags = [12, 24]  # Monthly data seasonality
    seasonal_acf = any(abs(acf_vals[lag]) > 0.3 for lag in seasonal_lags if lag < len(acf_vals))
    
    if seasonal_acf:
        print("- Strong seasonal pattern detected (spikes at lags 12/24)")
    else:
        print("- No strong seasonal pattern detected")
    
    # Check for AR signatures in PACF
    if abs(pacf_vals[1]) > 0.5 and all(abs(v) < 0.3 for v in pacf_vals[2:5]):
        print("- AR(1) signature: sharp PACF cutoff after lag 1")
    elif abs(pacf_vals[1]) > 0.4 and abs(pacf_vals[2]) > 0.3:
        print("- AR(2) signature: significant PACF at lags 1-2")
    
    # Check for MA signatures in ACF
    if abs(acf_vals[1]) > 0.5 and all(abs(v) < 0.3 for v in acf_vals[2:5]):
        print("- MA(1) signature: sharp ACF cutoff after lag 1")
    elif abs(acf_vals[1]) > 0.4 and abs(acf_vals[2]) > 0.3:
        print("- MA(2) signature: significant ACF at lags 1-2")
    
    if not (seasonal_acf or 
            any(abs(pacf_vals[i]) > 0.3 for i in range(1,5)) or 
            any(abs(acf_vals[i]) > 0.3 for i in range(1,5))):
        print("- White noise pattern: no significant autocorrelation")

# Decompose time series into trend, seasonal, and residual components
def decompose_series(series, period=12):
    return seasonal_decompose(series.interpolate().dropna(), 
                             model='additive', period=period)

# Plot the decomposition results
def plot_decomposition(decomposition):
    fig, axes = plt.subplots(4, 1, figsize=(12,10))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.show()

# Evaluate forecast performance metrics
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
    return mae, rmse

# Find optimal differencing order for stationarity
def find_optimal_d(series, max_d=2):
    d = 0
    current_series = series.copy()
    
    if test_stationarity(current_series, "Original Series"):
        print("Series is stationary (d=0)")
        return d
    
    for d in range(1, max_d + 1):
        current_series = current_series.diff().dropna()
        if test_stationarity(current_series, f"After {d}-order differencing"):
            print(f"Series became stationary after {d}-order differencing")
            return d
    
    print(f"Warning: Series still non-stationary after {max_d} differences")
    return max_d

# Find optimal seasonal differencing order for stationarity
def find_optimal_D(series, s=12, max_D=2):
    D = 0
    current_series = series.copy()
    
    if test_stationarity(current_series, "Original Series (Seasonal)"):
        print("Seasonal component is stationary (D=0)")
        return D
    
    for D in range(1, max_D + 1):
        current_series = current_series.diff(s).dropna()
        if test_stationarity(current_series, f"After {D}-order seasonal differencing (s={s})"):
            print(f"Seasonal component became stationary after {D}-order seasonal differencing")
            return D
    
    print(f"Warning: Seasonal component still non-stationary after {max_D} differences")
    return max_D

def main():
    # Load and prepare data
    print("Loading data...")
    data = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    summary = data['Gross_Revenue'].describe()
    print(summary)

    # Inflation adjustment
    if 'CPI' not in data.columns:
        print("\nWarning: CPI data not found - using nominal values")
        target_col = 'Gross_Revenue'
        data['Real_Price'] = data[target_col]
    else:
        print("\nAdjusting for inflation using CPI data")
        target_col = 'Gross_Revenue'
        data['Real_Price'] = data[target_col] / data['CPI'] * 100
    
    # EDA: Plot original vs real prices
    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], data[target_col], label='Nominal')
    plt.plot(data['Date'], data['Real_Price'], label='Real (Inflation-Adjusted)')
    plt.title('Nominal vs Real Prices')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Step 1: ACF/PACF Analysis on Original Series
    analyze_acf_pacf(data['Real_Price'], lags=36, title="Original Series")
    
    # Time series decomposition
    print("\nTime Series Decomposition:")
    decomposition = decompose_series(data.set_index('Date')['Real_Price'], period=12)
    plot_decomposition(decomposition)
    
    # 80/20 split
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    # Determine differencing orders
    print("\nDetermining Differencing Orders:")
    d = find_optimal_d(train['Real_Price'])
    D = find_optimal_D(train['Real_Price'], s=12)
    
    # Apply differencing
    stationary_series = train['Real_Price'].copy()
    if d > 0:
        stationary_series = stationary_series.diff(d).dropna()
    # Step 2: ACF/PACF Analysis on Stationary Series
    analyze_acf_pacf(stationary_series, lags=36, 
                    title=f"After {d}-order diff")
    
    if D > 0:
        stationary_series = stationary_series.diff(12 * D).dropna()
    analyze_acf_pacf(stationary_series, lags=36, 
                    title=f"After {D}-order seasonal diff and {d}-order diff")
    
    # Grid search for optimal parameters
    p_values = [0,1,2]
    d_values = [0,1,2]
    q_values = [0,1,2]
    P_values = [0,1]
    D_values = [0,1]
    Q_values = [0,1]
    m = 12  # monthly seasonality

    best_aic = float('inf')
    best_bic = float('inf')
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
                                    train['Real_Price'],
                                    order=(p,d,q),
                                    seasonal_order=(P,D,Q,m),
                                )
                                results = model.fit(disp=False)
                                print(f"AIC: {results.aic:.3f}, BIC: {results.bic:.3f}, order: {(p, d, q)}, seasonal_order: {(P, D, Q, m)}")
                                if results.aic < best_aic or (results.aic == best_aic and results.bic < best_bic):
                                    best_aic = results.aic
                                    best_bic = results.bic
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
    
    # Fit best model on training data
    best_model = SARIMAX(
        train['Real_Price'],
        order=best_order,
        seasonal_order=best_seasonal_order
    )
    results = best_model.fit(disp=False)
    print(results.summary())

    # Forecast entire test period
    forecast = results.get_forecast(steps=len(test))
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Extract forecast results
    test_actual = test['Real_Price']
    preds = pred_mean
    conf_lower = conf_int.iloc[:, 0]
    conf_upper = conf_int.iloc[:, 1]

    # Evaluate
    mae, rmse = evaluate_forecast(
        f"SARIMA({best_order})({best_seasonal_order})[12] - One-Time Fit",
        test_actual,
        preds
    )

    # Plot
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Entire time series
    ax1.plot(data['Date'], data['Real_Price'], label='Historical Data', color='blue', alpha=0.7)
    ax1.axvspan(data['Date'].iloc[0], data['Date'].iloc[split_idx-1], 
                color='green', alpha=0.1, label='Training Period')
    # ax1.plot(test['Date'], test_actual, label='Actual Test Values', 
    #         color='blue', linestyle='-', marker='o', markersize=5)
    ax1.plot(test['Date'], preds, label='Forecast', 
            color='red', linestyle='--', marker='x', markersize=6)
    ax1.fill_between(test['Date'], conf_lower, conf_upper, 
                    color='gray', alpha=0.3, label='95% Confidence Interval')
    ax1.axvline(x=test['Date'].iloc[0], color='gray', linestyle='--', label='Train-Test Split')

    ax1.set_title(f"SARIMA Forecast\nOrder: ({best_order}) Seasonal: ({best_seasonal_order})")
    ax1.set_ylabel("Real Price")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()