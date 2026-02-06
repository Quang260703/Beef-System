import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# File + column names
# -----------------------------
CSV_PATH = "Cow_Calf.csv"
DATE_COL = "Date"
NOMINAL_COL = "Calf_Price"
CPI_COL = "CPI"
CPI_BASE = 100.0

# -----------------------------
# SARIMA settings (from Section 3.2)
# -----------------------------
S = 12
D_FIXED = 1
d_FIXED = 1

# Search ranges for p,q,P,Q (keep modest)
P_RANGE = range(0, 3)   # p = 0..2
Q_RANGE = range(0, 3)   # q = 0..2
SP_RANGE = range(0, 2)  # P = 0..1
SQ_RANGE = range(0, 2)  # Q = 0..1

TRAIN_FRAC = 0.80

# Walk-forward settings
REFIT_EVERY = 1   # 1 = refit every step (true walk-forward). Use 6/12 to speed up.
PLOT = True

# -----------------------------
# Metrics (same family you used before)
# -----------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # avoid divide-by-zero issues
    eps = 1e-12
    denom = np.where(np.abs(y_true) < eps, np.nan, y_true)

    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    mpe = np.nanmean(((y_true - y_pred) / denom)) * 100
    rmspe = np.sqrt(np.nanmean(((y_true - y_pred) / denom) ** 2)) * 100

    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    acc = 100.0 * (1 - mae / np.nanmean(y_true)) if np.nanmean(y_true) != 0 else np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "MPE(%)": mpe,
        "RMSPE(%)": rmspe,
        "R2": r2,
        "Accuracy(%)": acc,
    }

# -----------------------------
# Load + prepare series
# -----------------------------
def load_real_price_series():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if NOMINAL_COL not in df.columns:
        raise KeyError(f"Missing '{NOMINAL_COL}'. Available columns: {list(df.columns)}")
    if CPI_COL not in df.columns:
        raise KeyError(f"Missing '{CPI_COL}'. Available columns: {list(df.columns)}")

    df["Real_Price"] = df[NOMINAL_COL] / df[CPI_COL] * CPI_BASE

    ts = df.set_index(DATE_COL)["Real_Price"].asfreq("MS")
    ts = ts.interpolate()  # fills any missing months smoothly
    ts.name = "Real_Price"
    return ts

# -----------------------------
# Grid search SARIMA on training set
# -----------------------------
def fit_sarima(y, order, seasonal_order):
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",                 # no deterministic trend term (we difference)
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)

def grid_search_sarima(y_train):
    best = {"aic": np.inf, "bic": np.inf, "order": None, "seasonal": None, "res": None}

    for p, q, P, Q in product(P_RANGE, Q_RANGE, SP_RANGE, SQ_RANGE):
        order = (p, d_FIXED, q)
        seasonal = (P, D_FIXED, Q, S)
        try:
            res = fit_sarima(y_train, order, seasonal)
            aic, bic = res.aic, res.bic

            # primary: lowest AIC; tie-breaker: lower BIC
            if (aic < best["aic"]) or (np.isclose(aic, best["aic"]) and bic < best["bic"]):
                best.update({"aic": aic, "bic": bic, "order": order, "seasonal": seasonal, "res": res})
            print(f"OK  order={order} seasonal={seasonal}  AIC={aic:.2f}  BIC={bic:.2f}")
        except Exception as e:
            print(f"SKIP order={order} seasonal={seasonal}  ({type(e).__name__})")
            continue

    if best["res"] is None:
        raise RuntimeError("No SARIMA model converged. Try expanding ranges or checking data.")
    return best

# -----------------------------
# Walk-forward (expanding window) one-step forecasts
# -----------------------------
def walk_forward_sarima(y, split_idx, order, seasonal_order):
    y_train0 = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    preds = []
    lowers = []
    uppers = []

    res_cached = None
    last_fit_end = split_idx

    for i in range(len(y_test)):
        end = split_idx + i
        y_expanding = y.iloc[:end]

        # refit schedule
        if (res_cached is None) or ((end - last_fit_end) >= REFIT_EVERY):
            res_cached = fit_sarima(y_expanding, order, seasonal_order)
            last_fit_end = end

        fc = res_cached.get_forecast(steps=1)
        pred = fc.predicted_mean.iloc[0]
        ci = fc.conf_int().iloc[0]

        preds.append(pred)
        lowers.append(ci[0])
        uppers.append(ci[1])

    pred_series = pd.Series(preds, index=y_test.index, name="Forecast")
    lo_series = pd.Series(lowers, index=y_test.index, name="Lo95")
    hi_series = pd.Series(uppers, index=y_test.index, name="Hi95")
    return y_test, pred_series, lo_series, hi_series

# -----------------------------
# Main
# -----------------------------
def main():
    plt.rcParams["font.family"] = "Times New Roman"

    y = load_real_price_series()

    split_idx = int(len(y) * TRAIN_FRAC)
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"Total n={len(y)}, train n={len(y_train)}, test n={len(y_test)}")

    # 1) Identify best SARIMA on training (AIC/BIC)
    best = grid_search_sarima(y_train)
    best_order = best["order"]
    best_seasonal = best["seasonal"]

    print("\nBEST SARIMA (training fit):")
    print("Order:", best_order)
    print("Seasonal order:", best_seasonal)
    print(f"AIC={best['aic']:.2f}  BIC={best['bic']:.2f}")

    # Save a one-row table for your dissertation table
    out_row = pd.DataFrame([{
        "Model": "SARIMA",
        "Best order": str(best_order),
        "Seasonal order(s)": f"{best_seasonal[:3]}_{best_seasonal[3]}",
        "AIC": best["aic"],
        "BIC": best["bic"]
    }])
    out_row.to_csv("Table_SARIMA_order_AIC_BIC.csv", index=False)
    print("Saved: Table_SARIMA_order_AIC_BIC.csv")

    # 2) Walk-forward evaluation on test
    y_test, y_pred, lo, hi = walk_forward_sarima(y, split_idx, best_order, best_seasonal)

    metrics = compute_metrics(y_test.values, y_pred.values)
    print("\nWalk-forward test performance (one-step ahead):")
    for k, v in metrics.items():
        print(f"{k}: {v:.6g}")

    # 3) Plot
    if PLOT:
        fig = plt.figure(figsize=(14, 6))
        plt.plot(y_train.index, y_train.values, label="Train")
        plt.plot(y_test.index, y_test.values, label="Test")
        plt.plot(y_pred.index, y_pred.values, label="Walk-forward forecast")

        plt.fill_between(y_test.index, lo.values, hi.values, alpha=0.2, label="95% CI")
        plt.axvline(y_test.index[0], linestyle="--", linewidth=1.5, label="Train/Test split")

        plt.title(f"SARIMA Walk-forward Forecast  order={best_order}  seasonal={best_seasonal}", fontsize=16)
        plt.xlabel("Year", fontsize=13)
        plt.ylabel("Real calf price", fontsize=13)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend()
        plt.tight_layout()
        plt.savefig("SARIMA_walkforward_forecast.png", dpi=600, bbox_inches="tight")
        plt.show()
        print("Saved: SARIMA_walkforward_forecast.png")

    # Save predictions
    out_pred = pd.DataFrame({
        "Actual": y_test,
        "Forecast": y_pred,
        "Lo95": lo,
        "Hi95": hi
    })
    out_pred.to_csv("SARIMA_walkforward_predictions.csv")
    print("Saved: SARIMA_walkforward_predictions.csv")

if __name__ == "__main__":
    main()

