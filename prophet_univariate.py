import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

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
    
# Load dataset
data = pd.read_csv("Cow_Calf.csv")
data.reset_index(drop=True, inplace=True)

target_col = 'Gross_Revenue'

# Split data into training and testing
split_idx = int(len(data) * 0.8)
train = data.iloc[:split_idx]
test  = data.iloc[split_idx:]

dates_train = train['Date']
dates_test  = test['Date']
endog_train = train[target_col]
endog_test  = test[target_col]
full_dates = pd.concat([dates_train, dates_test])
full_actual = pd.concat([endog_train, endog_test])

# Prepare data for Prophet
train_prophet = train.rename(columns={"Date": "ds", "Gross_Revenue": "y"})
test_prophet = test.rename(columns={"Date": "ds"})

# Initialize and train Prophet model
model_prophet = Prophet()
model_prophet.fit(train_prophet)

# Forecast future values
forecast_prophet = model_prophet.predict(test_prophet)

print(forecast_prophet.head())

# Plot forecast
evaluate_forecast("Prophet Test", endog_test, forecast_prophet['yhat'].values)

# Highlight training/test split
plt.figure(figsize=(12,6))

# entire actual data in blue
plt.plot(full_dates, full_actual, label='Actual (All)', color='blue', marker='o')

# rolling forecast in red
plt.plot(dates_test, forecast_prophet['yhat'], label='Test Prediction', color='red', marker='x')

# mark train/test boundary
plt.axvline(x=dates_test.iloc[0], color='gray', linestyle='--', label='Train-Test Split')

plt.title("Prophet: Actual vs. Predicted")
plt.xlabel("Date")
plt.ylabel(target_col)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()