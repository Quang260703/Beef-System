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
data.dropna(inplace=True)
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

# Initialize list to store walk-forward predictions
rolling_preds = []
current_train = train.copy()

# Starting walk-forward validation
for i in range(len(test)):
    # Prepare training data
    train_df = current_train.rename(columns={"Date": "ds", "Gross_Revenue": "y"})
    
    # Prepare next point to forecast
    next_point = test.iloc[i:i+1].copy()
    next_point.rename(columns={"Date": "ds"}, inplace=True)
    
    # Create and train model
    model = Prophet()
    model.fit(train_df)
    
    # Make forecast
    forecast = model.predict(next_point)
    pred = forecast['yhat'].values[0]
    rolling_preds.append(pred)
    
    # Update training set with actual observation
    current_train = pd.concat([current_train, test.iloc[i:i+1]], axis=0)

# Convert predictions to numpy array
rolling_preds = np.array(rolling_preds)

# Evaluate walk-forward performance
evaluate_forecast("Prophet Walk-Forward", endog_test, rolling_preds)

# Highlight training/test split
plt.figure(figsize=(12,6))

# entire actual data in blue
plt.plot(full_dates, full_actual, label='Actual (All)', color='blue', marker='o')

# rolling forecast in red
plt.plot(dates_test, rolling_preds, label='Test Prediction', color='red', marker='x')

# mark train/test boundary
plt.axvline(x=dates_test.iloc[0], color='gray', linestyle='--', label='Train-Test Split')

plt.title("Prophet: Actual vs. Predicted")
plt.xlabel("Date")
plt.ylabel(target_col)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()