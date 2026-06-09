from pathlib import Path
from itertools import combinations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
from scipy.stats import norm

def diebold_mariano(
    errors1,
    errors2,
    h: int = 1,
    power: int | float = 2,
    alternative: str = "two-sided",
    small_sample: bool = True,
):
    # ------------------------------------------------------------------ checks
    e1 = np.asarray(errors1, dtype=float).ravel()
    e2 = np.asarray(errors2, dtype=float).ravel()
    if e1.shape != e2.shape:
        raise ValueError("errors1 and errors2 must have the same length")
    T = e1.size
    if h < 1 or h > T:
        raise ValueError("h must be in [1, len(errors)]")

    # -------------------------------------------------------- loss differential
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d_mean = d.mean()

    # ---------------------- HAC variance: Newey–West truncation lag = h-1
    gamma = np.zeros(h)
    for k in range(h):
        if k == 0:
            # Variance (lag 0)
            gamma[k] = np.mean((d - d_mean) ** 2)
        else:
            # Autocovariances (lag k)
            gamma[k] = np.mean((d[:-k] - d_mean) * (d[k:] - d_mean))
    
    var_d = gamma[0] + 2.0 * gamma[1:].sum()
    
    # ADDITIONAL FIX: Check for near-zero variance
    if var_d <= 1e-15:
        # If variance is essentially zero, the losses are identical
        return 0.0, 1.0
    
    dm_stat = d_mean / np.sqrt(var_d / T)

    # -------------------------------------------- small-sample HLN correction
    if small_sample and h > 1:
        # Harvey–Leybourne–Newbold (1997) scaling factor
        K = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** 0.5
        dm_stat *= K

    # -------------------------------------------------------- p-value mapping
    if alternative == "two-sided":
        p_value = 2.0 * norm.sf(abs(dm_stat))
    elif alternative == "less":        # model 1 better
        p_value = norm.cdf(dm_stat)
    elif alternative == "greater":     # model 2 better
        p_value = norm.sf(dm_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return dm_stat, p_value

# ------------------------------------------------------------------------
# 1. LOAD ACTUAL RETURNS
# ------------------------------------------------------------------------
data = (
    pd.read_csv("data_non_std.csv", parse_dates=["date"])
      .rename(columns={"date": "Date"})
      .set_index("Date")
)
y_true = data["returns"]                       # full series

# ------------------------------------------------------------------------
# 2. GATHER PREDICTION FILES
# ------------------------------------------------------------------------
pred_dir = Path("results_autoencoder")  # directory with prediction CSVs
csv_paths = sorted(pred_dir.glob("*_predictions.csv"))   # one per model

if not csv_paths:
    raise FileNotFoundError("No prediction CSVs found in ./results/")

# ------------------------------------------------------------------------
# 3. METRIC FUNCTIONS
# ------------------------------------------------------------------------
def compute_metrics(y, y_hat):
    resid = y - y_hat

    mae  = mean_absolute_error(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    r2   = r2_score(y, y_hat)
    dw   = durbin_watson(resid)

    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    lb_stat, lb_p             = acorr_ljungbox(resid, lags=[10], return_df=False)

    return dict(
        MAE          = mae,
        RMSE         = rmse,
        R2           = r2,
        DurbinWatson = dw,
        JarqueBera   = jb_stat,
        JB_p         = jb_p,
        Skew         = skew,
        Kurtosis     = kurt,
        LjungBox10_p = lb_p[0],
    )

# ------------------------------------------------------------------------
# 4. LOOP THROUGH MODELS
# ------------------------------------------------------------------------
metrics = {}
errors  = {}          # store residuals for DM test

for csv_path in csv_paths:
    model_name = csv_path.stem.replace("_predictions", "")
    df_pred = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")

    # keep only dates present in both series  (OOS horizon)
    common = y_true.index.intersection(df_pred.index)
    if common.empty:
        raise ValueError(f"No overlapping dates for {model_name}")

    y = y_true.loc[common]
    y_hat = df_pred.loc[common, "prediction"]

    metrics[model_name] = compute_metrics(y, y_hat)
    errors[model_name]  = (y - y_hat)           # numpy array

# ------------------------------------------------------------------------
# 5. BUILD METRIC TABLE
# ------------------------------------------------------------------------
metrics_df = (
    pd.DataFrame(metrics)
      .T
      .sort_values("RMSE")
      .round(6)
)
print("\n==== Point-forecast quality ====\n")
print(metrics_df)

# Optionally save:
metrics_df.to_csv("model_comparison_metrics.csv")



# ------------------------------------------------------------------------
# 6. DIEBOLD–MARIANO PAIR-WISE TESTS
# ------------------------------------------------------------------------
def dm_matrix(
    err_dict: dict[str, pd.Series],
    horizon: int       = 1,
    power: int | float = 2,
    alternative: str   = "two-sided",
    small_sample: bool = False,
) -> pd.DataFrame:
    """
    Return symmetric matrix of Diebold–Mariano p-values for model pairs.
    Aligns residuals on shared index (usually dates).
    """
    models = list(err_dict)
    n = len(models)
    dm_p = pd.DataFrame(np.ones((n, n)), index=models, columns=models)

    for m1, m2 in combinations(models, 2):
        # Step 1: align on shared dates and drop missing values
        e1, e2 = err_dict[m1].align(err_dict[m2], join="inner")

        #e1 = e1.dropna()
        #e2 = e2.dropna()
        d = np.abs(e1.values) ** power - np.abs(e2.values) ** power
        # Step 2: check length
        if len(e1) < horizon + 2:
            print(f"Skipping {m1} vs {m2} (too short: {len(e1)} observations)")
            dm_p.loc[m1, m2] = dm_p.loc[m2, m1] = np.nan
            continue

        # Step 3: run the DM test
        stat, p = diebold_mariano(
            e1.values,
            e2.values,
            h=horizon,
            power=power,
            alternative=alternative,
            small_sample=small_sample,
        )

        # Step 4: store symmetric p-value
        dm_p.loc[m1, m2] = dm_p.loc[m2, m1] = p

    return dm_p



dm_pvals = dm_matrix(errors, horizon=1, power=2)

print("\n==== Diebold–Mariano p-values (squared-error loss) ====\n")
print(dm_pvals)

# Optionally save the table
dm_pvals.to_csv("model_dm_pvalues.csv")

# ADDITIONAL DEBUGGING: Print some statistics
print("\n==== Debugging Information ====")
for model_name, err in errors.items():
    print(f"{model_name}: mean={err.mean():.6f}, std={err.std():.6f}, n={len(err)}")