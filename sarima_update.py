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
    print(f'p-value: {result[1]:.4f}')
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
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot ACF
    sm.graphics.tsa.plot_acf(series, lags=lags, ax=ax1, title=f'ACF - {title}')
    
    # Plot PACF
    sm.graphics.tsa.plot_pacf(series, lags=lags, ax=ax2, method='ywm', title=f'PACF - {title}')
    
    plt.tight_layout()
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
    return mae, rmse, r2

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
    
    # Inflation adjustment
    if 'CPI' not in data.columns:
        print("\nWarning: CPI data not found - using nominal values")
        target_col = 'Gross_Revenue'
        data['Real_Price'] = data[target_col]
    else:
        print("\nAdjusting for inflation using CPI data")
        target_col = 'Gross_Revenue'
        data['Real_Price'] = data[target_col] / data['CPI'] * 100
    
    # Apply log transformation to stabilize variance
    print("Applying log transformation to stabilize variance...")
    data['Log_Real_Price'] = np.log(data['Real_Price'] + 1e-9)
    
    # EDA: Plot original vs real prices
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(data['Date'], data[target_col], label='Nominal')
    plt.plot(data['Date'], data['Real_Price'], label='Real (Inflation-Adjusted)')
    plt.title('Nominal vs Real Prices')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(data['Date'], data['Log_Real_Price'], label='Log Real Price', color='purple')
    plt.title('Log-Transformed Real Prices')
    plt.xlabel('Date')
    plt.ylabel('Log Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('price_analysis.png', dpi=300)
    plt.show()
    
    # Step 1: ACF/PACF Analysis on Original Series
    analyze_acf_pacf(data['Log_Real_Price'], lags=36, title="Original Series (Log Transformed)")
    
    # Time series decomposition
    print("\nTime Series Decomposition:")
    decomposition = decompose_series(data.set_index('Date')['Log_Real_Price'], period=12)
    plot_decomposition(decomposition)
    
    # 80/20 split
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    # Determine differencing orders
    print("\nDetermining Differencing Orders:")
    d = find_optimal_d(train['Log_Real_Price'])
    D = find_optimal_D(train['Log_Real_Price'], s=12)
    
    # Apply differencing
    stationary_series = train['Log_Real_Price'].copy()
    if d > 0:
        stationary_series = stationary_series.diff(d).dropna()
    if D > 0:
        stationary_series = stationary_series.diff(12 * D).dropna()
    
    # Step 2: ACF/PACF Analysis on Stationary Series
    analyze_acf_pacf(stationary_series, lags=36, 
                    title=f"After {d}-order diff and {D}-order seasonal diff")
    
    # Grid search for optimal parameters
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
                                    train['Log_Real_Price'],
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
    
    # Fit best model on training data
    print("\nFitting Best Model on Training Data...")
    best_model = SARIMAX(
        train['Log_Real_Price'],
        order=(p, d, q),
        seasonal_order=(P, D, Q, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    best_results = best_model.fit(disp=False)
    print(best_results.summary())

    # Prepare data structures
    history = train['Log_Real_Price'].copy()
    test_dates = test['Date']
    walk_forward_preds = []
    conf_lower = []
    conf_upper = []
    
    # Transformation functions
    def log_to_original(x):
        return np.exp(x) - 1e-9
    
    # Progress bar setup
    progress_bar = tqdm(total=len(test), desc="Walk-Forward Forecasting")
    
    # Walk-forward validation
    for i in range(len(test)):
        # Create SARIMA model
        model = SARIMAX(
            history,
            order=(p, d, q),
            seasonal_order=(P, D, Q, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit model
        results = model.fit(disp=False)
        
        # Forecast next step with confidence intervals
        forecast = results.get_forecast(steps=1)
        pred_mean = forecast.predicted_mean.iloc[0]
        pred_ci = forecast.conf_int().iloc[0]
        
        # Store predictions and confidence intervals
        walk_forward_preds.append(pred_mean)
        conf_lower.append(pred_ci[0])
        conf_upper.append(pred_ci[1])
        
        # Update history with actual test observation
        actual_value = test['Log_Real_Price'].iloc[i]
        history = pd.concat([history, pd.Series([actual_value])])
        
        # Update progress bar
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Create series for results
    walk_forward_series = pd.Series(walk_forward_preds, index=test.index)
    conf_lower_series = pd.Series(conf_lower, index=test.index)
    conf_upper_series = pd.Series(conf_upper, index=test.index)
    
    # Transform back to original scale
    test_original = log_to_original(test['Log_Real_Price'])
    preds_original = log_to_original(walk_forward_series)
    conf_lower_original = log_to_original(conf_lower_series)
    conf_upper_original = log_to_original(conf_upper_series)
    
    # Evaluate performance on original scale
    mae, rmse, r2 = evaluate_forecast(
        f"SARIMAX({p},{d},{q})({P},{D},{Q})[12] Walk-Forward", 
        test_original, 
        preds_original
    )
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot entire time series
    ax1.plot(data['Date'], data['Real_Price'], 
             label='Historical Data', color='blue', alpha=0.7)
    
    # Highlight training period
    ax1.axvspan(data['Date'].iloc[0], data['Date'].iloc[split_idx-1], 
                color='green', alpha=0.1, label='Training Period')
    
    # Plot test actuals
    ax1.plot(test_dates, test_original, 
             label='Actual Test Values', color='blue', linestyle='-', marker='o', markersize=5)
    
    # Plot walk-forward predictions
    ax1.plot(test_dates, preds_original, 
             label='Walk-Forward Forecast', color='red', linestyle='--', marker='x', markersize=6)
    
    # Plot confidence intervals
    ax1.fill_between(test_dates, conf_lower_original, conf_upper_original,
                     color='gray', alpha=0.3, label='95% Confidence Interval')
    
    # Train/test split line
    ax1.axvline(x=test_dates.iloc[0], color='gray', 
                linestyle='--', label='Train-Test Split')
    
    # Formatting
    ax1.set_title(f"Walk-Forward SARIMA Forecast\nOrder: ({p},{d},{q}) Seasonal: ({P},{D},{Q},12)")
    ax1.set_ylabel("Real Price")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Format x-axis dates
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Plot forecast errors
    forecast_errors = test_original - preds_original
    
    ax2.plot(test_dates, forecast_errors, 'ro-', label='Forecast Error')
    ax2.axhline(y=0, color='k', linestyle='-')
    ax2.fill_between(test_dates, -mae, mae, color='gray', alpha=0.2, label=f'±MAE ({mae:.2f})')
    
    # Formatting
    ax2.set_title(f"Forecast Errors (MAE: {mae:.2f}, RMSE: {rmse:.2f})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(True)
    
    # Format x-axis dates
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()