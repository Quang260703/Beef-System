import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import optuna

data = pd.read_csv('Cow_Calf.csv', parse_dates=['Date'])
data.sort_values('Date', inplace=True)
data.reset_index(drop=True, inplace=True)

# Define the target variable
target_col = 'Gross_Revenue'
values = data[target_col].values.reshape(-1, 1)

# Split data: 80% train, 10% validation, 10% test
n = len(data)
train_end = int(n * 0.4)
val_end = int(n * 0.6)

train = values[:train_end]
val   = values[train_end:val_end]
test  = values[val_end:]

# Scale data using MinMaxScaler (fit on train only)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
val_scaled   = scaler.transform(val)
test_scaled  = scaler.transform(test)


# Helper Function: Create Dataset with a Given Look-back

def create_dataset(dataset, look_back):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, 0])
        y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(y)

# Objective Function for Optuna Hyperparameter Tuning

def objective(trial):
    # Hyperparameters to tune
    look_back = trial.suggest_int("look_back", 3, 10)
    units = trial.suggest_int("units", 20, 100)
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 10, 30)
    
    # Create dataset using the current look_back value
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_val, y_val     = create_dataset(val_scaled, look_back)
    
    # Reshape to [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val   = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(look_back, 1)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Train the model (using the validation set directly)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              verbose=0, validation_data=(X_val, y_val))
    
    # Make predictions on validation set and compute RMSE
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

# Run Optuna Study

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("\nBest trial:")
trial = study.best_trial
print(f"  Best RMSE: {trial.value:.4f}")
for key, value in trial.params.items():
    print(f"  {key}: {value}")

# Rebuild the Model Using Best Hyperparameters

best_params = trial.params
look_back = best_params['look_back'] 

# For final training, combine training and validation sets
train_val_scaled = np.concatenate((train_scaled, val_scaled), axis=0)
X_train_val, y_train_val = create_dataset(train_val_scaled, look_back)
X_test, y_test = create_dataset(test_scaled, look_back)

# Reshape datasets to [samples, timesteps, features]
X_train_val = X_train_val.reshape((X_train_val.shape[0], X_train_val.shape[1], 1))
X_test      = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(best_params['units'], activation='relu', input_shape=(look_back, 1)))
if best_params['dropout'] > 0:
    model.add(Dropout(best_params['dropout']))
model.add(Dense(1))
optimizer = Adam(learning_rate=best_params['lr'])
model.compile(optimizer=optimizer, loss='mean_squared_error')

history = model.fit(X_train_val, y_train_val, epochs=best_params['epochs'], 
                    batch_size=best_params['batch_size'], verbose=1, validation_split=0.1)

# Forecasting on Test Data
test_pred = model.predict(X_test)
test_pred_inv = scaler.inverse_transform(test_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Compute evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
mae = mean_absolute_error(y_test_inv, test_pred_inv)
mape = np.mean(np.abs((y_test_inv - test_pred_inv) / np.where(y_test_inv==0, 1, y_test_inv))) * 100
r2 = r2_score(y_test_inv, test_pred_inv)

print("\nFinal Evaluation Metrics on Test Set:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  MAPE: {mape:.2f}%")
print(f"  R²: {r2:.2f}")

# Plotting the Results
plt.figure(figsize=(12, 6))

# Adjust date indices for plotting considering the look_back offset
train_dates = data['Date'].values[look_back:train_end]
val_dates   = data['Date'].values[train_end+look_back:val_end]
test_dates  = data['Date'].values[val_end+look_back:]

# For plotting, combine all scaled data and inverse transform
full_scaled = np.concatenate((train_scaled, val_scaled, test_scaled), axis=0)
full_actual_inv = scaler.inverse_transform(full_scaled)

plt.plot(data['Date'], full_actual_inv, label='Actual', marker='o', linestyle='-')
plt.plot(test_dates, test_pred_inv, label='Test Forecast', marker='x', linestyle='--')

# Mark split boundaries
plt.axvline(x=data['Date'].iloc[train_end], color='gray', linestyle='--', label='Train/Validation Split')
plt.axvline(x=data['Date'].iloc[val_end], color='black', linestyle='--', label='Validation/Test Split')

plt.title("LSTM Forecast for Gross Revenue (80/10/10 Split, Optimized with Optuna)")
plt.xlabel("Date")
plt.ylabel(target_col)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
