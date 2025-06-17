import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def test_stationarity(series, title="ADF Test"):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    print(f"\n{title}")
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for k, v in result[4].items():
        print(f'\t{k}: {v:.4f}')
    return result[1] < 0.05  # True if stationary

def decompose_series(series, period=12):
    """Decompose time series into trend, seasonal, and residual components"""
    decomposition = seasonal_decompose(
        series.interpolate().dropna(),
        model='additive',
        period=period
    )
    return decomposition

def plot_decomposition(decomposition):
    """Plot decomposition components"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,10))
    decomposition.observed.plot(ax=ax1, title='Observed')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
    decomposition.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    plt.show()

def evaluate_forecast(model_name, actual, predicted):
    """Calculate and print evaluation metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    print(f"\n[{model_name}] Test Performance:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    print(f"Accuracy (1 - MAE/mean * 100): {acc:.2f}%")
    return mae, rmse, r2

def find_optimal_d(series, max_d=2):
    """Determine optimal differencing order"""
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

def find_optimal_D(series, s=12, max_D=2):
    """Determine optimal seasonal differencing order"""
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

def sarima_grid_search(series, p_range, d, q_range, P_range, D, Q_range, s=12):
    """Perform grid search for optimal SARIMA parameters"""
    best_aic = float("inf")
    best_bic = float("inf")
    best_params = None
    
    param_combinations = list(itertools.product(p_range, q_range, P_range, Q_range))
    total_combinations = len(param_combinations)
    print(f"\nPerforming grid search over {total_combinations} parameter combinations...")
    
    for i, (p, q, P, Q) in enumerate(param_combinations):
        try:
            model = SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)
            
            if results.aic < best_aic:
                best_aic = results.aic
                best_bic = results.bic
                best_params = (p, d, q, P, D, Q)
                print(f"New best AIC: {best_aic:.1f} | Order: ({p},{d},{q}) Seasonal: ({P},{D},{Q},{s})")
                
        except Exception as e:
            continue
            
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{total_combinations} combinations")
    
    print(f"\nOptimal Parameters: Order={best_params[:3]}, Seasonal={best_params[3:]}")
    print(f"AIC: {best_aic:.1f}, BIC: {best_bic:.1f}")
    return best_params

def main():
    # Load and prepare data
    data = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Inflation adjustment (placeholder - add actual CPI data)
    if 'CPI' not in data.columns:
        print("\nWarning: CPI data not found - using nominal values")
        target_col = 'Gross_Revenue'
        data['Real_Price'] = data[target_col]
    else:
        print("\nAdjusting for inflation using CPI data")
        target_col = 'Gross_Revenue'
        data['Real_Price'] = data[target_col] / data['CPI'] * 100
    
    # EDA: Plot original vs real prices
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data[target_col], label='Nominal')
    plt.plot(data['Date'], data['Real_Price'], label='Real (Inflation-Adjusted)')
    plt.title('Nominal vs Real Prices')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
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
    
    # Plot ACF/PACF after differencing
    stationary_series = train['Real_Price'].copy()
    if d > 0:
        stationary_series = stationary_series.diff(d).dropna()
    if D > 0:
        stationary_series = stationary_series.diff(12 * D).dropna()
    
    print("\nACF/PACF Analysis of Stationary Series:")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(stationary_series, lags=24, ax=ax1)
    plot_pacf(stationary_series, lags=24, ax=ax2, method='ywm')
    plt.suptitle(f'ACF/PACF after {d}-order diff and {D}-order seasonal diff')
    plt.tight_layout()
    plt.show()
    
    # Grid search for optimal parameters
    print("\nModel Identification via Grid Search:")
    best_params = sarima_grid_search(
        train['Real_Price'],
        p_range=range(0, 3),
        d=d,
        q_range=range(0, 3),
        P_range=range(0, 2),
        D=D,
        Q_range=range(0, 2),
        s=12
    )
    p, d, q, P, D, Q = best_params
    
    # Fit best model
    print("\nFitting Best Model...")
    best_model = SARIMAX(
        train['Real_Price'],
        order=(p, d, q),
        seasonal_order=(P, D, Q, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    best_results = best_model.fit(disp=False)
    print(best_results.summary())
    
    # Model diagnostics
    print("\nModel Diagnostics:")
    best_results.plot_diagnostics(figsize=(12, 8))
    plt.suptitle('Model Residual Diagnostics', y=0.95)
    plt.tight_layout()
    plt.show()
    
    # Ljung-Box test for residual autocorrelation
    lb_test = acorr_ljungbox(best_results.resid, lags=[10], return_df=True)
    print("\nLjung-Box Test for Residual Autocorrelation:")
    print(f"Q-statistic: {lb_test['lb_stat'].values[0]:.4f}")
    print(f"p-value: {lb_test['lb_pvalue'].values[0]:.4f}")
    if lb_test['lb_pvalue'].values[0] > 0.05:
        print("Residuals are independent (good fit)")
    else:
        print("Residuals show autocorrelation (model may be inadequate)")
    
    # Rolling forecast with best model
    print("\nGenerating Rolling Forecasts...")
    rolling_preds = []
    all_dates = data['Date']
    real_prices = data['Real_Price']
    
    for i in range(len(test)):
        current_data = real_prices.iloc[:split_idx + i]
        
        model = SARIMAX(
            current_data,
            order=(p, d, q),
            seasonal_order=(P, D, Q, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        pred = results.forecast(steps=1).iloc[0]
        rolling_preds.append(pred)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Forecasted {i+1}/{len(test)} periods")
    
    # Create prediction series
    rolling_preds_series = pd.Series(rolling_preds, index=test.index)
    
    # Evaluate forecasts
    mae, rmse, r2 = evaluate_forecast(
        f"SARIMAX({p},{d},{q})({P},{D},{Q})[12]", 
        test['Real_Price'], 
        rolling_preds_series
    )
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    # Full actual data
    plt.plot(data['Date'], data['Real_Price'], 
             label='Actual (All)', color='blue', alpha=0.7)
    
    # Training period highlight
    plt.axvspan(data['Date'].iloc[0], data['Date'].iloc[split_idx-1], 
                color='green', alpha=0.1, label='Training Period')
    
    # Test period actuals
    plt.plot(test['Date'], test['Real_Price'], 
             label='Actual (Test)', color='blue', linestyle='-', marker='o')
    
    # Forecasts
    plt.plot(test['Date'], rolling_preds_series, 
             label='Forecast (Test)', color='red', linestyle='--', marker='x')
    
    # Confidence intervals
    last_train = data.iloc[split_idx-1]
    forecast_dates = pd.date_range(
        start=last_train['Date'] + pd.DateOffset(months=1),
        periods=len(test),
        freq='MS'
    )
    
    # Train/test split line
    plt.axvline(x=test['Date'].iloc[0], color='gray', 
                linestyle='--', label='Train-Test Split')
    
    plt.title(f"SARIMAX Forecast: Actual vs Predicted\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Real Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()