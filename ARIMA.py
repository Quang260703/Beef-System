import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Paths / column names
# -----------------------------
CSV_PATH = "Cow_Calf.csv"
DATE_COL = "Date"
NOMINAL_COL = "Calf_Price"
CPI_COL = "CPI"
CPI_BASE = 100.0

# -----------------------------
# Settings
# -----------------------------
TRAIN_FRAC = 0.80
D_FIXED = 1           # align with your Section 3.2: non-seasonal differencing d = 1
MAX_PQ = 6            # search range for p and q (keep modest)
PLOT = True

# -----------------------------
# Helpers
# -----------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # avoid division by zero in percentage metrics
    eps = 1e-12
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    mpe  = np.mean(((y_true - y_pred) / (y_true + eps))) * 100
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + eps)))) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    acc = 100.0 * (1 - mae / (np.mean(y_true) + eps))

    return {
        "MAE": mae, "MSE": mse, "RMSE": rmse,
        "MAPE": mape, "MPE": mpe, "RMSPE": rmspe,
        "R2": r2, "Acc(1-MAE/mean)%": acc
    }


def load_real_price():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    df = df.asfreq("MS")  # monthly start

    # numeric coercion
    df[NOMINAL_COL] = pd.to_numeric(df[NOMINAL_COL], errors="coerce")
    df[CPI_COL] = pd.to_numeric(df[CPI_COL], errors="coerce")

    # interpolate missing values safely (if any)
    df[[NOMINAL_COL, CPI_COL]] = df[[NOMINAL_COL, CPI_COL]].interpolate(limit_direction="both")
    df = df.dropna(subset=[NOMINAL_COL, CPI_COL])

    df["Real_Price"] = df[NOMINAL_COL] / df[CPI_COL] * CPI_BASE
    y = df["Real_Price"].dropna()

    return y


def walk_forward_arima(y, order, split_idx):
    """Expanding-window 1-step walk-forward forecasts on y (levels)."""
    preds, lowers, uppers = [], [], []
    test_index = y.index[split_idx:]

    for i, t in enumerate(test_index):
        y_train = y.iloc[: split_idx + i]  # expanding window up to t-1

        model = ARIMA(y_train, order=order).fit()
        fc = model.get_forecast(steps=1)

        mean = float(fc.predicted_mean.iloc[0])
        ci = fc.conf_int(alpha=0.05).iloc[0]  # 95% CI

        preds.append(mean)
        lowers.append(float(ci[0]))
        uppers.append(float(ci[1]))

    return pd.Series(preds, index=test_index), pd.Series(lowers, index=test_index), pd.Series(uppers, index=test_index)


def main():
    plt.rcParams["font.family"] = "Times New Roman"

    # 1) Load series
    y = load_real_price()

    # 2) Train/test split (chronological)
    split_idx = int(len(y) * TRAIN_FRAC)
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # 3) Identify ARIMA(p,d,q) using train only (NON-seasonal)
    arima_search = auto_arima(
        y_train,
        seasonal=False,
        d=D_FIXED,
        start_p=0, start_q=0,
        max_p=MAX_PQ, max_q=MAX_PQ,
        stepwise=True,
        information_criterion="aic",   # use AIC for selection; you can switch to "bic"
        suppress_warnings=True,
        error_action="ignore",
        trace=True
    )

    best_order = arima_search.order  # (p,d,q)
    print("\nBest ARIMA order selected on TRAIN:", best_order)

    # 4) Fit final ARIMA on training set for AIC/BIC table row
    arima_fit = ARIMA(y_train, order=best_order).fit()
    print("\nARIMA(train) AIC:", arima_fit.aic)
    print("ARIMA(train) BIC:", arima_fit.bic)

    # Save “Section 3.3 table row” for ARIMA
    table_row = pd.DataFrame([{
        "Model": "ARIMA",
        "Best order": str(best_order),
        "Seasonal order(s)": "-",
        "AIC": arima_fit.aic,
        "BIC": arima_fit.bic
    }])
    table_row.to_csv("Table_ARIMA_Order_AIC_BIC.csv", index=False)
    print("\nSaved: Table_ARIMA_Order_AIC_BIC.csv")

    # 5) Walk-forward forecasting on test set
    preds, lower, upper = walk_forward_arima(y, best_order, split_idx)

    metrics = compute_metrics(y_test.values, preds.values)
    print("\nWalk-forward test metrics (ARIMA):")
    for k, v in metrics.items():
        print(f"{k}: {v:.6g}")

    # Save forecasts
    out_df = pd.DataFrame({
        "Actual_Real_Price": y_test,
        "Forecast": preds,
        "Lower_95": lower,
        "Upper_95": upper
    })
    out_df.to_csv("ARIMA_WalkForward_Forecasts.csv")
    print("\nSaved: ARIMA_WalkForward_Forecasts.csv")

    # 6) Plot (optional)
    if PLOT:
        plt.figure(figsize=(14, 6))
        plt.plot(y_train.index, y_train.values, label="Train (Real Price)")
        plt.plot(y_test.index, y_test.values, label="Test (Actual)")
        plt.plot(preds.index, preds.values, label="Walk-forward Forecast")
        plt.fill_between(preds.index, lower.values, upper.values, alpha=0.2, label="95% CI")
        plt.axvline(y_test.index[0], linestyle="--", linewidth=2, label="Train/Test Split")
        plt.title("ARIMA Walk-forward Forecast (Real Calf Price)")
        plt.xlabel("Date")
        plt.ylabel("Real Price (CPI-adjusted)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Figure_ARIMA_WalkForward.png", dpi=600, bbox_inches="tight")
        plt.show()
        print("Saved: Figure_ARIMA_WalkForward.png")


if __name__ == "__main__":
    main()
