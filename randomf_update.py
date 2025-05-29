import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import optuna
import sys
import sklearn 
print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Optuna version:", optuna.__version__)

def evaluate_forecast(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    acc = 100.0 * (1 - mae / actual.mean()) if actual.mean() != 0 else float('nan')
    print(f"\n[{model_name}] Test Performance:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    print(f"Accuracy (1 - MAE/mean * 100): {acc:.2f}%")
    
# Data Loading and Validation
try:
    df = pd.read_csv('Cow_Calf.csv', parse_dates=["Date"])
except FileNotFoundError:
    raise FileNotFoundError("Required data file r'C:\\Users\\KaniyamattamLab\\OneDrive - Texas A&M AgriLife\\Desktop\\Cow_Calf.csv' not found. Please check file path.")


required_columns = {"Gross_Revenue", "Exchange_Rate_JPY_USD", 
                   "Net_Gas_Price", "CPI", "Corn_Price", "Date"}
missing_cols = required_columns - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Data Preparation
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").set_index("Date")

    # Lag features with dynamic window sizing
    max_lag = 6  # Maximum lag period in months
    for lag in range(max_lag):
        df[f'Gross_Revenue_lag{lag}'] = df['Gross_Revenue'].shift(lag+1)
    
    # Rolling statistics of the previous 3 months
    feature = "Gross_Revenue"
    df[f'{feature}_Rolling_Means'] = df[feature].shift(1).rolling(3).mean() 
    df[f'{feature}_Rolling_STD'] = df[feature].shift(1).rolling(3).std()
    
    # Final cleaning
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=False)
    print(f"Removed {initial_rows - len(df)} rows with missing values.")
    
    if len(df) < 24:
        raise ValueError("Insufficient data after preprocessing (minimum 24 months required).")
    
    return df

try:
    df = prepare_dataframe(df)
    pd.set_option('display.max_columns', None)
    print(df)
except Exception as e:
    print(f"Data preparation failed: {str(e)}")
    sys.exit(1)


# Time Series Split
def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.6):
    test_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:test_idx]
    test_df = df.iloc[test_idx:]
    
    # Create validation set from end of training data
    val_idx = int(len(train_df) * 0.8)
    val_df = train_df.iloc[val_idx:]
    train_df = train_df.iloc[:val_idx]
    
    print(f"Split sizes:\n- Train: {len(train_df)}\n- Val: {len(val_df)}\n- Test: {len(test_df)}")
    return train_df, val_df, test_df

try:
    target = "Gross_Revenue"
    features = [col for col in df.columns if col != target and col != "Date"]
    
    train_df, val_df, test_df = temporal_train_test_split(df)
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
except Exception as e:
    print(f"Data splitting failed: {str(e)}")
    sys.exit(1)

# Model Training Pipeline
class ForecastingPipeline:
   
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.residual_model = None
        self.feature_importances_ = None
        
    def optimize_hyperparameters(self, X, y, n_trials=50):
        
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train_fold)
                X_val_scaled = self.scaler.transform(X_val_fold)
                
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train_fold)
                
                preds = model.predict(X_val_scaled)
                mae = mean_absolute_error(y_val_fold, preds)
                scores.append(mae)
                
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def train(self, X_train, y_train, X_val, y_val):
        # Hyperparameter optimization
        best_params = self.optimize_hyperparameters(X_train, y_train)
        
        # Final model training
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)
        
        # Residual modeling
        train_preds = self.model.predict(X_train_scaled)
        residuals = y_train - train_preds
        self.residual_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.residual_model.fit(X_train_scaled, residuals)
        
        # Feature importance analysis
        self.feature_importances_ = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Validation performance
        val_preds = self.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        print(f"Validation MAE: {val_mae:.2f}")
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        base_preds = self.model.predict(X_scaled)
        residual_preds = self.residual_model.predict(X_scaled)
        return base_preds + residual_preds

# Execute Pipeline
try:
    pipeline = ForecastingPipeline()
    pipeline.train(X_train, y_train, X_val, y_val)
    
    # Generate predictions
    test_preds = pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f"Final Test MAE: {test_mae:.2f}")
    evaluate_forecast(f"Random Forest Training", y_train, pipeline.predict(X_train))
    evaluate_forecast(f"Random Forest", y_test, test_preds)
    
    
except Exception as e:
    print(f"Model training failed: {str(e)}")
    sys.exit(1)

# Results Visualization
def plot_forecast_results(y_train, y_val, y_test, val_preds, test_preds):
    plt.figure(figsize=(16, 8))
    
    # Plot historical data
    plt.plot(y_train.index, y_train, label='Training Data', lw=2)
    plt.plot(y_val.index, y_val, label='Validation Data', lw=2)
    plt.plot(y_test.index, y_test, label='Test Data', lw=2)
    
    # Plot predictions
    plt.plot(y_val.index, val_preds, '--', label='Validation Predictions', lw=2)
    plt.plot(y_test.index, test_preds, '--', label='Test Predictions', lw=2)
    
    # Add event markers
    plt.axvline(x=y_val.index[0], color='gray', linestyle=':', label='Validation Start')
    plt.axvline(x=y_test.index[0], color='black', linestyle=':', label='Test Start')
    
    plt.title("Gross Revenue Forecasting Performance", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Gross Revenue", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_forecast_results(y_train, y_val, y_test, 
                     pipeline.predict(X_val), 
                     pipeline.predict(X_test))