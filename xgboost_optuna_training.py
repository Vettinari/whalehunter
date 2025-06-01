import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib  # For saving scaler if needed, though not explicitly used for model saving here
import random
import os
import multiprocessing

# Lock all random seeds for reproducibility
RANDOM_SEED = 42

# Hardware Optimizations
N_CORES = multiprocessing.cpu_count()  # Use all available cores
print(f"Detected {N_CORES} CPU cores") # Simpler message


# Check for GPU availability (e.g., CUDA for NVIDIA, Metal for Apple Silicon)
def check_gpu_support():
    """Check if GPU training is available (tries CUDA first, then generic error)."""
    try:
        # Try to create a simple XGBoost model with GPU
        # For XGBoost >= 2.0, use device="cuda:0" or device="mps"
        # Assuming CUDA is the primary target for now based on logs
        test_model = xgb.XGBRegressor(tree_method="hist", device="cuda:0", n_estimators=1)
        X_test = np.random.random((10, 5))
        y_test = np.random.random(10)
        test_model.fit(X_test, y_test)
        print("✅ GPU support (CUDA or similar) detected and working!")
        return True
    except Exception as e:
        print(f"❌ GPU support not available or error during test: {e}")
        print("Using optimized CPU training instead.")
        return False


GPU_AVAILABLE = check_gpu_support()


def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set optuna sampler seed
    optuna.logging.set_verbosity(
        optuna.logging.INFO
    )  # Change to INFO to see more progress

    print(f"Random seeds locked to {seed} for reproducibility")


# Set seeds at the start
set_random_seeds()

# import xgboost.callback # Reverting from callback

TARGET_COLUMN = "user_revenue_usd_30d"


def load_data(file_path):
    """Loads data from a parquet file."""
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows.")
    return df


def create_revenue_bins(y, n_bins=10):
    """Create stratification bins for heavy-tailed revenue data."""
    # Handle zero values separately
    zero_mask = y == 0
    non_zero_y = y[~zero_mask]

    if len(non_zero_y) == 0:
        # All values are zero
        return np.zeros(len(y), dtype=int)

    # Create bins for non-zero values using quantiles
    bins = np.zeros(len(y), dtype=int)

    if len(non_zero_y) > 0:
        # Use quantile-based binning for non-zero values
        try:
            non_zero_bins = (
                pd.qcut(
                    non_zero_y,
                    q=min(n_bins - 1, len(non_zero_y.unique())),
                    labels=False,
                    duplicates="drop",
                )
                + 1
            )
            bins[~zero_mask] = non_zero_bins
        except ValueError:
            # Fallback to simple binning if qcut fails
            non_zero_bins = (
                pd.cut(
                    non_zero_y,
                    bins=min(n_bins - 1, len(non_zero_y.unique())),
                    labels=False,
                    duplicates="drop",
                )
                + 1
            )
            bins[~zero_mask] = non_zero_bins

    # Zero values get bin 0
    bins[zero_mask] = 0

    print(f"Created {len(np.unique(bins))} revenue bins:")
    for bin_id in sorted(np.unique(bins)):
        bin_mask = bins == bin_id
        bin_values = y[bin_mask]
        print(
            f"  Bin {bin_id}: {bin_mask.sum()} samples, "
            f"revenue range [{bin_values.min():.6f}, {bin_values.max():.6f}]"
        )

    return bins


def split_data(df, target_column):
    """Splits data into training and testing sets using stratified sampling."""
    print(f"Splitting data with stratified sampling. Target column: {target_column}")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Create stratification bins
    revenue_bins = create_revenue_bins(y, n_bins=10)

    # Use stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(X, revenue_bins))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Print distribution comparison
    train_bins = revenue_bins[train_idx]
    test_bins = revenue_bins[test_idx]

    print("Revenue distribution comparison:")
    print("Bin | Train % | Test %")
    print("----|---------|-------")
    for bin_id in sorted(np.unique(revenue_bins)):
        train_pct = (train_bins == bin_id).mean() * 100
        test_pct = (test_bins == bin_id).mean() * 100
        print(f"{bin_id:3d} | {train_pct:6.2f}% | {test_pct:5.2f}%")

    return X_train, X_test, y_train, y_test


def optuna_callback(study, trial):
    """Callback to show progress during Optuna optimization."""
    print(f"Trial {trial.number} completed:")
    print(f"  Value (RMSE): {trial.value:.6f}")
    print(f"  Best value so far: {study.best_value:.6f}")

    # Show additional metrics if available
    if hasattr(trial, "user_attrs"):
        if "mae" in trial.user_attrs:
            print(f"  MAE: {trial.user_attrs['mae']:.6f}")
        if "objective_used" in trial.user_attrs:
            print(f"  Objective: {trial.user_attrs['objective_used']}")
        if "best_n_estimators" in trial.user_attrs:
            print(f"  Best n_estimators: {trial.user_attrs['best_n_estimators']}")

    print(f"  Key parameters:")
    key_params = ["objective", "booster", "max_depth", "eta", "lambda", "alpha"]
    for param in key_params:
        if param in trial.params:
            print(f"    {param}: {trial.params[param]}")
    print("-" * 50)


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for XGBoost regressor optimized for M2 Mac."""
    print(f"\nStarting trial {trial.number}...")

    # Check if data has zero values to determine valid objectives
    has_zeros = (y_train == 0).any() or (y_val == 0).any()

    # Suggest objective function - exclude gamma if there are zeros
    if has_zeros:
        objective_choices = [
            "reg:squarederror",  # Standard MSE
            "reg:squaredlogerror",  # Good for heavy-tailed positive data
            "reg:pseudohubererror",  # Robust to outliers
            "reg:tweedie",  # Good for zero-inflated data
        ]
    else:
        objective_choices = [
            "reg:squarederror",  # Standard MSE
            "reg:squaredlogerror",  # Good for heavy-tailed positive data
            "reg:pseudohubererror",  # Robust to outliers
            "reg:gamma",  # Good for positive skewed data (only if no zeros)
            "reg:tweedie",  # Good for zero-inflated data
        ]

    objective_func = trial.suggest_categorical("objective", objective_choices)

    # SOLUTION 1: Add regularization to prevent overfitting to dominant features
    # SOLUTION 2: Increase model complexity for smoother predictions
    # SOLUTION 3: Add feature sampling to reduce dominance

    # M2 Mac optimized parameters with anti-clustering improvements
    params = {
        "objective": objective_func,
        "booster": trial.suggest_categorical(
            "booster", ["gbtree"]
        ),  # Focus on gbtree for speed
        # SOLUTION 1: Stronger regularization to prevent hard splits
        "lambda": trial.suggest_float(
            "lambda", 1e-3, 50.0, log=True
        ),  # Increased L2 regularization
        "alpha": trial.suggest_float(
            "alpha", 1e-3, 50.0, log=True
        ),  # Increased L1 regularization
        # SOLUTION 2: Feature sampling to reduce dominant feature impact
        "subsample": trial.suggest_float(
            "subsample", 0.6, 0.9
        ),  # More aggressive subsampling
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 0.8
        ),  # Stronger feature sampling
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.5, 0.9
        ),  # Per-level feature sampling
        "colsample_bynode": trial.suggest_float(
            "colsample_bynode", 0.5, 0.9
        ),  # Per-node feature sampling
        # SOLUTION 3: Model complexity for smoother predictions
        "max_depth": trial.suggest_int(
            "max_depth", 3, 8
        ),  # Shallower trees for smoother predictions
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 5, 50
        ),  # Higher minimum samples per leaf
        "max_delta_step": trial.suggest_float(
            "max_delta_step", 0, 10
        ),  # Limit step size for stability
        # SOLUTION 4: Learning rate and tree count for ensemble smoothing
        "eta": trial.suggest_float(
            "eta", 0.01, 0.2, log=True
        ),  # Lower learning rate for smoother learning
        "gamma": trial.suggest_float(
            "gamma", 0, 20
        ),  # Minimum loss reduction for splits
        "random_state": RANDOM_SEED,
        "n_jobs": N_CORES,  # Use all available cores
        "early_stopping_rounds": 50,  # Increased for better convergence
        # Hardware optimizations (GPU or CPU)
        "tree_method": "hist", # Standard 'hist' for both CPU and GPU with XGBoost >= 2.0
        "max_bin": 256,  # Optimized for memory bandwidth
        # SOLUTION 5: Interaction constraints to prevent over-reliance on single features
        "interaction_constraints": "[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]]",  # Force feature interactions
    }

    # GPU-specific optimizations
    if GPU_AVAILABLE:
        params["device"] = "cuda:0" # Use device="cuda:0" for XGBoost >= 2.0
        # params["predictor"] = "gpu_predictor" # This is often default or not needed with device setting
    else:
        # CPU optimizations
        params["device"] = "cpu" # Explicitly set to CPU if GPU not available
        params["max_bin"] = 512  # Higher bins for CPU as we have more memory bandwidth

    # Add Tweedie-specific parameter if using Tweedie
    if objective_func == "reg:tweedie":
        params["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.1, 1.9
        )

    # Set eval_metric after all parameters are defined
    if objective_func == "reg:squaredlogerror":
        eval_metric = "rmse"  # Use RMSE instead of RMSLE to avoid NaN issues
    elif objective_func in ["reg:pseudohubererror"]:
        eval_metric = "rmse"
    elif objective_func == "reg:gamma":
        eval_metric = "gamma-nloglik"
    elif objective_func == "reg:tweedie":
        # Tweedie metric needs the variance power parameter
        tweedie_power = params["tweedie_variance_power"]
        eval_metric = f"tweedie-nloglik@{tweedie_power}"
    else:
        eval_metric = "rmse"

    # Add eval_metric to params
    params["eval_metric"] = eval_metric

    # SOLUTION 6: More trees with lower learning rate for ensemble smoothing
    model = xgb.XGBRegressor(
        n_estimators=2000, **params
    )  # Increased back to 2000 for smoothing

    # Handle potential issues for different objectives
    if objective_func == "reg:squaredlogerror":
        # Ensure all values are positive for log-based objectives
        # Add a larger epsilon to avoid log(0) issues
        min_val = min(y_train.min(), y_val.min())
        if min_val <= 0:
            epsilon = abs(min_val) + 1e-6
        else:
            epsilon = 1e-6
        y_train_adj = y_train + epsilon
        y_val_adj = y_val + epsilon
    elif objective_func == "reg:gamma":
        # Gamma requires strictly positive values - add small epsilon
        min_val = min(y_train.min(), y_val.min())
        if min_val <= 0:
            epsilon = abs(min_val) + 1e-6
        else:
            epsilon = 1e-6
        y_train_adj = y_train + epsilon
        y_val_adj = y_val + epsilon
    else:
        y_train_adj = y_train
        y_val_adj = y_val

    model.fit(
        X_train,
        y_train_adj,
        eval_set=[(X_val, y_val_adj)],
        verbose=False,  # Reduced verbosity for speed
    )

    preds = model.predict(X_val)

    # Calculate RMSE for comparison (always use RMSE for Optuna optimization)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    # Store additional metrics for analysis
    mae = mean_absolute_error(y_val, preds)
    trial.set_user_attr("mae", mae)
    trial.set_user_attr("objective_used", objective_func)
    trial.set_user_attr("has_zeros", has_zeros)

    best_iteration_val = getattr(model, "best_iteration", None)
    if (
        best_iteration_val is not None and best_iteration_val > 0
    ):  # best_iteration can be 0 if first iter is best
        trial.set_user_attr(
            "best_n_estimators", best_iteration_val + 1
        )  # Store 1-indexed count
    else:  # If no early stopping or best_iteration is 0
        if hasattr(model, "n_estimators_"):  # This is set after fit
            trial.set_user_attr("best_n_estimators", model.n_estimators_)
        else:  # Fallback if somehow n_estimators_ is not available
            trial.set_user_attr("best_n_estimators", 2000)  # The initial n_estimators

    print(
        f"Trial {trial.number} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, Objective: {objective_func}"
    )
    return rmse


def analyze_target_distribution(y, dataset_name):
    """Analyze target distribution and provide insights."""
    print(f"\n=== Target Distribution Analysis for {dataset_name} ===")

    # Basic statistics
    print(f"Basic statistics:")
    print(f"  Count: {len(y):,}")
    print(f"  Mean: {y.mean():.6f}")
    print(f"  Median: {y.median():.6f}")
    print(f"  Std: {y.std():.6f}")
    print(f"  Min: {y.min():.6f}")
    print(f"  Max: {y.max():.6f}")

    # Distribution characteristics
    zero_pct = (y == 0).mean() * 100
    low_pct = (y < 0.1).mean() * 100
    high_pct = (y > 5.0).mean() * 100

    print(f"\nDistribution characteristics:")
    print(f"  Zero values: {zero_pct:.1f}%")
    print(f"  Values < 0.1: {low_pct:.1f}%")
    print(f"  Values > 5.0: {high_pct:.1f}%")
    print(f"  Skewness: {y.skew():.3f}")
    print(f"  Kurtosis: {y.kurtosis():.3f}")

    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(y, p):.6f}")

    # Recommendations
    print(f"\nRecommended approaches:")
    if zero_pct > 30:
        print(
            "  - High zero inflation detected: Consider Tweedie or Zero-Inflated models"
        )
    if y.skew() > 1.5:
        print(
            "  - High positive skew: Consider log transformation or Gamma distribution"
        )
    if y.kurtosis() > 3:
        print(
            "  - Heavy tails detected: Consider robust loss functions (Huber, Quantile)"
        )

    return {
        "zero_pct": zero_pct,
        "skew": y.skew(),
        "kurtosis": y.kurtosis(),
        "high_tail_pct": high_pct,
    }


def inverse_transform_revenue(y_scaled):
    """Convert log-scaled revenue back to true dollar values."""
    return np.expm1(y_scaled)  # Inverse of np.log1p


def calculate_business_metrics(y_true_dollars, y_pred_dollars):
    """Calculate business-relevant metrics using true dollar values."""
    # Remove any negative predictions (shouldn't happen but safety check)
    y_pred_dollars = np.maximum(y_pred_dollars, 0)

    # Basic metrics in dollars
    mae_dollars = mean_absolute_error(y_true_dollars, y_pred_dollars)
    rmse_dollars = np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))

    # Median Absolute Error (more robust to outliers)
    median_ae_dollars = np.median(np.abs(y_true_dollars - y_pred_dollars))

    # Mean Absolute Percentage Error (MAPE) - handle division by zero
    non_zero_mask = y_true_dollars > 0
    if non_zero_mask.sum() > 0:
        mape = (
            np.mean(
                np.abs(
                    (y_true_dollars[non_zero_mask] - y_pred_dollars[non_zero_mask])
                    / y_true_dollars[non_zero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.inf

    # Symmetric MAPE (handles zero values better)
    smape = (
        np.mean(
            2
            * np.abs(y_true_dollars - y_pred_dollars)
            / (np.abs(y_true_dollars) + np.abs(y_pred_dollars) + 1e-8)
        )
        * 100
    )

    # Revenue bucket analysis
    revenue_buckets = {
        "Zero Revenue": (y_true_dollars == 0).sum(),
        "Low Revenue ($0-$1)": ((y_true_dollars > 0) & (y_true_dollars <= 1)).sum(),
        "Medium Revenue ($1-$10)": (
            (y_true_dollars > 1) & (y_true_dollars <= 10)
        ).sum(),
        "High Revenue ($10-$100)": (
            (y_true_dollars > 10) & (y_true_dollars <= 100)
        ).sum(),
        "Very High Revenue ($100+)": (y_true_dollars > 100).sum(),
    }

    # Prediction accuracy by bucket
    bucket_accuracy = {}
    for bucket_name, count in revenue_buckets.items():
        if count > 0:
            if bucket_name == "Zero Revenue":
                mask = y_true_dollars == 0
            elif bucket_name == "Low Revenue ($0-$1)":
                mask = (y_true_dollars > 0) & (y_true_dollars <= 1)
            elif bucket_name == "Medium Revenue ($1-$10)":
                mask = (y_true_dollars > 1) & (y_true_dollars <= 10)
            elif bucket_name == "High Revenue ($10-$100)":
                mask = (y_true_dollars > 10) & (y_true_dollars <= 100)
            else:  # Very High Revenue
                mask = y_true_dollars > 100

            bucket_mae = mean_absolute_error(y_true_dollars[mask], y_pred_dollars[mask])
            bucket_accuracy[bucket_name] = {
                "count": count,
                "mae_dollars": bucket_mae,
                "mean_true": y_true_dollars[mask].mean(),
                "mean_pred": y_pred_dollars[mask].mean(),
            }

    return {
        "mae_dollars": mae_dollars,
        "rmse_dollars": rmse_dollars,
        "median_ae_dollars": median_ae_dollars,
        "mape": mape,
        "smape": smape,
        "revenue_buckets": revenue_buckets,
        "bucket_accuracy": bucket_accuracy,
    }


def create_business_plots(y_true_dollars, y_pred_dollars, dataset_name):
    """Create business-focused plots with true dollar values."""

    # 1. Actual vs Predicted in True Dollars
    fig_dollars = go.Figure()
    fig_dollars.add_trace(
        go.Scatter(
            x=y_true_dollars,
            y=y_pred_dollars,
            mode="markers",
            name="Actual vs. Predicted ($)",
            marker=dict(color="blue", opacity=0.6, size=4),
            text=[
                f"True: ${true:.2f}<br>Pred: ${pred:.2f}"
                for true, pred in zip(y_true_dollars, y_pred_dollars)
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig_dollars.add_trace(
        go.Scatter(
            x=[y_true_dollars.min(), y_true_dollars.max()],
            y=[y_true_dollars.min(), y_true_dollars.max()],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash"),
        )
    )
    fig_dollars.update_layout(
        title=f"Revenue Prediction: True Dollar Values - {dataset_name}",
        xaxis_title="Actual Revenue (USD)",
        yaxis_title="Predicted Revenue (USD)",
        xaxis_type="log",
        yaxis_type="log",
        width=800,
        height=600,
    )
    fig_dollars.write_html(f"revenue_dollars_{dataset_name.replace(' ', '_')}.html")

    # 2. Revenue Distribution Comparison
    fig_dist = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Actual Revenue Distribution",
            "Predicted Revenue Distribution",
            "Revenue Buckets Comparison",
            "Prediction Error by Revenue Level",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Actual revenue histogram
    fig_dist.add_trace(
        go.Histogram(x=y_true_dollars, nbinsx=50, name="Actual", opacity=0.7),
        row=1,
        col=1,
    )

    # Predicted revenue histogram
    fig_dist.add_trace(
        go.Histogram(x=y_pred_dollars, nbinsx=50, name="Predicted", opacity=0.7),
        row=1,
        col=2,
    )

    # Revenue buckets comparison
    business_metrics = calculate_business_metrics(y_true_dollars, y_pred_dollars)
    bucket_names = list(business_metrics["revenue_buckets"].keys())
    bucket_counts = list(business_metrics["revenue_buckets"].values())

    fig_dist.add_trace(
        go.Bar(x=bucket_names, y=bucket_counts, name="Revenue Buckets"), row=2, col=1
    )

    # Prediction error by revenue level
    revenue_ranges = np.logspace(-2, 2, 20)  # From $0.01 to $100
    error_by_range = []
    range_labels = []

    for i in range(len(revenue_ranges) - 1):
        mask = (y_true_dollars >= revenue_ranges[i]) & (
            y_true_dollars < revenue_ranges[i + 1]
        )
        if mask.sum() > 5:  # Only include ranges with sufficient data
            error = np.mean(np.abs(y_true_dollars[mask] - y_pred_dollars[mask]))
            error_by_range.append(error)
            range_labels.append(f"${revenue_ranges[i]:.2f}-${revenue_ranges[i+1]:.2f}")

    if error_by_range:
        fig_dist.add_trace(
            go.Bar(x=range_labels, y=error_by_range, name="MAE by Range"), row=2, col=2
        )

    fig_dist.update_layout(
        title=f"Revenue Analysis Dashboard - {dataset_name}",
        height=800,
        showlegend=True,
    )
    fig_dist.update_xaxes(title_text="Revenue ($)", row=1, col=1)
    fig_dist.update_xaxes(title_text="Revenue ($)", row=1, col=2)
    fig_dist.update_xaxes(title_text="Revenue Bucket", row=2, col=1)
    fig_dist.update_xaxes(title_text="Revenue Range", row=2, col=2)
    fig_dist.update_yaxes(title_text="Count", row=1, col=1)
    fig_dist.update_yaxes(title_text="Count", row=1, col=2)
    fig_dist.update_yaxes(title_text="Count", row=2, col=1)
    fig_dist.update_yaxes(title_text="MAE ($)", row=2, col=2)

    fig_dist.write_html(f"revenue_analysis_{dataset_name.replace(' ', '_')}.html")

    # 3. Business Performance Summary
    fig_summary = go.Figure()

    # Create a summary table
    summary_data = []
    for bucket_name, bucket_info in business_metrics["bucket_accuracy"].items():
        summary_data.append(
            [
                bucket_name,
                bucket_info["count"],
                f"${bucket_info['mean_true']:.2f}",
                f"${bucket_info['mean_pred']:.2f}",
                f"${bucket_info['mae_dollars']:.2f}",
            ]
        )

    fig_summary.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Revenue Bucket",
                    "Count",
                    "Avg Actual ($)",
                    "Avg Predicted ($)",
                    "MAE ($)",
                ],
                fill_color="lightblue",
                align="left",
            ),
            cells=dict(
                values=(
                    list(zip(*summary_data)) if summary_data else [[], [], [], [], []]
                ),
                fill_color="white",
                align="left",
            ),
        )
    )

    fig_summary.update_layout(
        title=f"Business Performance Summary - {dataset_name}", height=400
    )
    fig_summary.write_html(f"business_summary_{dataset_name.replace(' ', '_')}.html")

    return business_metrics


def create_ensemble_model(X_train, y_train, X_val, y_val, best_params):
    """Create an ensemble model to reduce clustering artifacts."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import VotingRegressor

    print("🔄 Creating ensemble model to reduce prediction clustering...")

    # XGBoost with best params (but more regularized)
    xgb_params = best_params.copy()
    xgb_params.update(
        {
            "lambda": max(xgb_params.get("lambda", 1), 10),  # Stronger L2
            "alpha": max(xgb_params.get("alpha", 1), 5),  # Stronger L1
            "max_depth": min(xgb_params.get("max_depth", 6), 5),  # Shallower
            "colsample_bytree": 0.7,  # More feature sampling
        }
    )

    if "objective" in xgb_params:
        del xgb_params["objective"]

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        random_state=RANDOM_SEED,
        **xgb_params,
    )

    # Random Forest for smooth predictions
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.7,
        random_state=RANDOM_SEED,
        n_jobs=N_CORES,
    )

    # Ridge regression for linear baseline
    ridge_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)

    # Create ensemble
    ensemble = VotingRegressor(
        [("xgb", xgb_model), ("rf", rf_model), ("ridge", ridge_model)],
        weights=[0.6, 0.3, 0.1],
    )  # XGB gets most weight but not all

    print("Training ensemble components...")
    ensemble.fit(X_train, y_train)

    return ensemble


def add_feature_engineering(X):
    """Add engineered features to reduce dominant feature impact."""
    X_eng = X.copy()

    # SOLUTION 7: Feature engineering to create smoother relationships
    if "sum_revenue_0h_48h" in X_eng.columns:
        # Log transform the dominant feature to reduce its impact
        X_eng["log_sum_revenue_0h_48h"] = np.log1p(X_eng["sum_revenue_0h_48h"])

        # Create binned version for smoother transitions
        X_eng["sum_revenue_0h_48h_binned"] = pd.qcut(
            X_eng["sum_revenue_0h_48h"], q=20, labels=False, duplicates="drop"
        )

        # Interaction features
        if "sum_revenue_0h_24h" in X_eng.columns:
            X_eng["revenue_ratio_24h_48h"] = X_eng["sum_revenue_24h_48h"] / (
                X_eng["sum_revenue_0h_24h"] + 1e-6
            )

    return X_eng


def train_and_evaluate(dataset_name, dataset_path):
    """Trains an XGBoost model using Optuna and evaluates it."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")

    df = load_data(dataset_path)

    if TARGET_COLUMN not in df.columns:
        print(f"ERROR: Target column '{TARGET_COLUMN}' not found in {dataset_name}.")
        print(f"Available columns: {df.columns.tolist()}")
        return None, None, None

    # Analyze target distribution
    target_stats = analyze_target_distribution(df[TARGET_COLUMN], dataset_name)

    # SOLUTION 7: Apply feature engineering to reduce clustering
    print(f"\n🔧 Applying feature engineering to reduce prediction clustering...")
    X_original = df.drop(columns=[TARGET_COLUMN])
    X_engineered = add_feature_engineering(X_original)
    y = df[TARGET_COLUMN]

    print(f"Original features: {X_original.shape[1]}")
    print(f"Engineered features: {X_engineered.shape[1]}")

    X_train, X_test, y_train, y_test = split_data(
        pd.concat([X_engineered, y], axis=1), TARGET_COLUMN
    )

    print(
        f"\nStarting Optuna hyperparameter optimization with anti-clustering improvements..."
    )
    print(
        f"Running 30 trials with 300 second timeout for faster iteration on M2..."
    )  # Reduced for speed
    # Create study with seeded sampler for reproducibility
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED, n_startup_trials=5
    )  # Reduced startup trials
    study = optuna.create_study(
        direction="minimize",
        study_name=f"xgb_optimization_{dataset_name}",
        sampler=sampler,
    )

    # Create stratified validation split for Optuna
    print("Creating validation split for hyperparameter optimization...")
    train_revenue_bins = create_revenue_bins(y_train, n_bins=10)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_SEED
    )
    train_opt_idx, val_opt_idx = next(sss_val.split(X_train, train_revenue_bins))

    X_train_opt, X_val_opt = X_train.iloc[train_opt_idx], X_train.iloc[val_opt_idx]
    y_train_opt, y_val_opt = y_train.iloc[train_opt_idx], y_train.iloc[val_opt_idx]

    study.optimize(
        lambda trial: objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt),
        n_trials=30,  # Reduced from 50 for faster iteration
        timeout=300,  # Reduced from 600 for faster iteration
        callbacks=[optuna_callback],
    )

    print(f"\n{'='*50}")
    print(f"Optimization completed for {dataset_name}!")
    print(f"{'='*50}")
    print(f"Best trial:")
    best_params_from_optuna = study.best_params
    best_n_estimators = study.best_trial.user_attrs.get(
        "best_n_estimators", 2000
    )  # Updated default
    best_objective = study.best_trial.user_attrs.get(
        "objective_used", "reg:squarederror"
    )

    print(f"  Value (RMSE): {study.best_value:.6f}")
    print(f"  Best objective function: {best_objective}")
    if "mae" in study.best_trial.user_attrs:
        print(f"  Best MAE: {study.best_trial.user_attrs['mae']:.6f}")
    print(f"  Params (excluding n_estimators determined by early stopping): ")
    for key, value in best_params_from_optuna.items():
        print(f"    {key}: {value}")
    print(f"  Best n_estimators (from early stopping): {best_n_estimators}")

    # SOLUTION 8: Train both single model and ensemble for comparison
    print(f"\n🎯 Training final models...")

    # Train regular XGBoost model
    print("1. Training regularized XGBoost model...")
    final_model_params = best_params_from_optuna.copy()

    # Remove objective from params since we'll set it explicitly
    if "objective" in final_model_params:
        del final_model_params["objective"]

    # M2 Mac optimized final model
    final_model_params.update(
        {
            "n_jobs": N_CORES,
            "tree_method": "hist", # Standard 'hist' for both CPU and GPU with XGBoost >= 2.0
            "max_bin": 256,  # Optimized for memory bandwidth (can be higher for CPU)
        }
    )

    if GPU_AVAILABLE:
        final_model_params["device"] = "cuda:0" # Use device="cuda:0" for XGBoost >= 2.0
    else:
        # CPU optimizations
        final_model_params["device"] = "cpu" # Explicitly set to CPU if GPU not available
        final_model_params["max_bin"] = 512  # Higher bins for CPU as we have more memory bandwidth

    final_model = xgb.XGBRegressor(
        objective=best_objective,  # Use the best objective found
        random_state=RANDOM_SEED,
        n_estimators=best_n_estimators,
        early_stopping_rounds=50,  # Increased for better convergence
        **final_model_params,
    )

    print("Fitting final XGBoost model on full training set...")

    # Handle different objectives for final training
    if best_objective == "reg:squaredlogerror":
        min_val = min(y_train.min(), y_test.min())
        if min_val <= 0:
            epsilon = abs(min_val) + 1e-6
        else:
            epsilon = 1e-6
        y_train_final = y_train + epsilon
        y_test_final = y_test + epsilon
    elif best_objective == "reg:gamma":
        min_val = min(y_train.min(), y_test.min())
        if min_val <= 0:
            epsilon = abs(min_val) + 1e-6
        else:
            epsilon = 1e-6
        y_train_final = y_train + epsilon
        y_test_final = y_test + epsilon
    else:
        y_train_final = y_train
        y_test_final = y_test

    final_model.fit(
        X_train,
        y_train_final,
        eval_set=[(X_test, y_test_final)],
        verbose=True,  # Keep verbose for final model to see progress
    )

    # Train ensemble model
    print("\n2. Training ensemble model to reduce clustering...")
    ensemble_model = create_ensemble_model(
        X_train, y_train_final, X_test, y_test_final, best_params_from_optuna
    )

    # Evaluate both models
    print("\n📊 Evaluating models...")

    # XGBoost predictions
    y_pred_xgb = final_model.predict(X_test)

    # Ensemble predictions
    y_pred_ensemble = ensemble_model.predict(X_test)

    # Standard metrics on log-scaled data for XGBoost
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # Standard metrics on log-scaled data for Ensemble
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    rmse_ensemble = np.sqrt(mse_ensemble)
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
    r2_ensemble = r2_score(y_test, y_pred_ensemble)

    print(f"XGBoost Model Metrics for {dataset_name} (Log-Scaled):")
    print(f"  MSE: {mse_xgb:.4f}")
    print(f"  RMSE: {rmse_xgb:.4f}")
    print(f"  MAE: {mae_xgb:.4f}")
    print(f"  R-squared: {r2_xgb:.4f}")

    print(f"\nEnsemble Model Metrics for {dataset_name} (Log-Scaled):")
    print(f"  MSE: {mse_ensemble:.4f}")
    print(f"  RMSE: {rmse_ensemble:.4f}")
    print(f"  MAE: {mae_ensemble:.4f}")
    print(f"  R-squared: {r2_ensemble:.4f}")

    # Choose best model based on RMSE
    if rmse_ensemble < rmse_xgb:
        print(
            f"\n🏆 Ensemble model performs better! Using ensemble for final analysis."
        )
        best_model = ensemble_model
        y_pred = y_pred_ensemble
        mse, rmse, mae, r2 = mse_ensemble, rmse_ensemble, mae_ensemble, r2_ensemble
        model_type = "Ensemble"
    else:
        print(f"\n🏆 XGBoost model performs better! Using XGBoost for final analysis.")
        best_model = final_model
        y_pred = y_pred_xgb
        mse, rmse, mae, r2 = mse_xgb, rmse_xgb, mae_xgb, r2_xgb
        model_type = "XGBoost"

    # Convert back to true dollar values for business metrics
    print(f"\n🔄 Converting predictions back to true dollar values...")
    y_test_dollars = inverse_transform_revenue(y_test)
    y_pred_dollars = inverse_transform_revenue(y_pred)

    # Calculate business metrics
    business_metrics = calculate_business_metrics(y_test_dollars, y_pred_dollars)

    print(
        f"\n💰 Business Metrics for {dataset_name} ({model_type} Model - True Dollar Values):"
    )
    print(f"  MAE: ${business_metrics['mae_dollars']:.2f}")
    print(f"  RMSE: ${business_metrics['rmse_dollars']:.2f}")
    print(f"  Median AE: ${business_metrics['median_ae_dollars']:.2f}")
    print(f"  MAPE: {business_metrics['mape']:.1f}%")
    print(f"  SMAPE: {business_metrics['smape']:.1f}%")

    print(f"\n📊 Revenue Distribution:")
    for bucket, count in business_metrics["revenue_buckets"].items():
        pct = (count / len(y_test_dollars)) * 100
        print(f"  {bucket}: {count} users ({pct:.1f}%)")

    print(f"\n🎯 Prediction Accuracy by Revenue Bucket:")
    for bucket, metrics in business_metrics["bucket_accuracy"].items():
        print(f"  {bucket}:")
        print(f"    Count: {metrics['count']}")
        print(f"    Avg Actual: ${metrics['mean_true']:.2f}")
        print(f"    Avg Predicted: ${metrics['mean_pred']:.2f}")
        print(f"    MAE: ${metrics['mae_dollars']:.2f}")

    # Create business-focused plots
    print(f"\n📈 Creating business analysis plots...")
    create_business_plots(
        y_test_dollars, y_pred_dollars, f"{dataset_name}_{model_type.lower()}"
    )

    # Create clustering analysis plot
    print(f"\n🔍 Creating clustering analysis plot...")
    create_clustering_analysis_plot(y_test, y_pred_xgb, y_pred_ensemble, dataset_name)

    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Model_Type": model_type}

    # Add business metrics to the metrics dictionary
    metrics.update(
        {
            "MAE_Dollars": business_metrics["mae_dollars"],
            "RMSE_Dollars": business_metrics["rmse_dollars"],
            "Median_AE_Dollars": business_metrics["median_ae_dollars"],
            "MAPE": business_metrics["mape"],
            "SMAPE": business_metrics["smape"],
        }
    )

    # Feature importance (only for XGBoost)
    if hasattr(final_model, "feature_importances_"):
        feature_importances = pd.Series(
            final_model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        print("\nXGBoost Feature Importances:")
        print(feature_importances.head(10))

        potential_omittable_features = feature_importances[
            feature_importances < 0.001
        ].index.tolist()
        print(
            f"\nPotentially omittable features (importance < 0.001) for {dataset_name}: {potential_omittable_features}"
        )
    else:
        feature_importances = pd.Series()

    # Original plots (log-scaled)
    fig_metrics = go.Figure(
        data=[
            go.Bar(
                name="Test Metrics", x=list(metrics.keys()), y=list(metrics.values())
            )
        ]
    )
    fig_metrics.update_layout(
        title_text=f"Model Performance Metrics - {dataset_name} ({model_type})"
    )
    fig_metrics.write_html(
        f"metrics_{dataset_name.replace(' ', '_')}_{model_type.lower()}.html"
    )

    if not feature_importances.empty:
        fig_feature_importance = px.bar(
            feature_importances.head(20),  # Show top 20 features
            x=feature_importances.head(20).values,
            y=feature_importances.head(20).index,
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            title=f"Feature Importance - {dataset_name} (XGBoost)",
        )
        fig_feature_importance.update_layout(yaxis={"categoryorder": "total ascending"})
        fig_feature_importance.write_html(
            f"feature_importance_{dataset_name.replace(' ', '_')}.html"
        )

    # Original scatter plot (log-scaled)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            name=f"Actual vs. Predicted ({model_type})",
            marker=dict(color="blue", opacity=0.5),
        )
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode="lines",
            name="Ideal Fit",
            line=dict(color="red", dash="dash"),
        )
    )
    fig_scatter.update_layout(
        title=f"Actual vs. Predicted Revenue (Log-Scaled) - {dataset_name} ({model_type})",
        xaxis_title="Actual Revenue (Log-Scaled)",
        yaxis_title="Predicted Revenue (Log-Scaled)",
    )
    fig_scatter.write_html(
        f"actual_vs_predicted_{dataset_name.replace(' ', '_')}_{model_type.lower()}.html"
    )

    return best_model, metrics, feature_importances


def create_clustering_analysis_plot(y_test, y_pred_xgb, y_pred_ensemble, dataset_name):
    """Create a plot comparing XGBoost vs Ensemble predictions to show clustering reduction."""

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "XGBoost Predictions (May Show Clustering)",
            "Ensemble Predictions (Smoother)",
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )

    # XGBoost scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred_xgb,
            mode="markers",
            name="XGBoost",
            marker=dict(color="red", opacity=0.6, size=3),
        ),
        row=1,
        col=1,
    )

    # Ensemble scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred_ensemble,
            mode="markers",
            name="Ensemble",
            marker=dict(color="blue", opacity=0.6, size=3),
        ),
        row=1,
        col=2,
    )

    # Add ideal fit lines
    min_val, max_val = y_test.min(), y_test.max()

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Clustering Analysis: XGBoost vs Ensemble - {dataset_name}",
        height=500,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Actual Revenue (Log-Scaled)", row=1, col=1)
    fig.update_xaxes(title_text="Actual Revenue (Log-Scaled)", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Revenue (Log-Scaled)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Revenue (Log-Scaled)", row=1, col=2)

    fig.write_html(f"clustering_analysis_{dataset_name.replace(' ', '_')}.html")


if __name__ == "__main__":
    datasets_to_process = {
        "scaled_revenue_30d_balanced_small": "data/scaled_revenue_30d_balanced_small.parquet",
        "scaled_revenue_30d_balanced": "data/scaled_revenue_30d_balanced.parquet",
        "scaled_revenue_30d": "data/scaled_revenue_30d.parquet",
    }

    results = {}

    for name, path in datasets_to_process.items():
        model, metrics, features = train_and_evaluate(name, path)
        if model and metrics:
            results[name] = {
                "model": model,
                "metrics": metrics,
                "feature_importances": features,
            }
        else:
            print(f"Skipping comparison for {name} due to errors during processing.")

    print("\n\n--- Comparison of Results ---")
    if len(results) < 2:
        print(
            "Cannot compare results as one or both datasets failed to process properly."
        )
    elif not all("metrics" in res for res in results.values()):
        print("Cannot compare results as metrics are missing for one or both datasets.")
    else:
        metrics_comparison_data = []
        for name, res in results.items():
            if "metrics" in res:  # Ensure metrics exist
                for metric_name, val in res["metrics"].items():
                    metrics_comparison_data.append(
                        {"Dataset": name, "Metric": metric_name, "Value": val}
                    )
            else:
                print(
                    f"Warning: Metrics not found for {name}, skipping from comparison plot."
                )

        if metrics_comparison_data:
            df_comparison = pd.DataFrame(metrics_comparison_data)

            fig_comparison = px.bar(
                df_comparison,
                x="Metric",
                y="Value",
                color="Dataset",
                barmode="group",
                title="Comparison of Model Metrics Across Datasets",
            )
            fig_comparison.write_html("metrics_comparison.html")

            print("\nMetrics Comparison Table:")
            # Check if df_comparison is not empty before pivoting
            if not df_comparison.empty:
                # Filter out rows where 'Value' might be NaN if a dataset failed catastrophically before metric calculation but still got an entry
                df_comparison_pivot = df_comparison.dropna(subset=["Value"]).pivot(
                    index="Dataset", columns="Metric", values="Value"
                )
                print(df_comparison_pivot)
            else:
                print("No data available for metrics comparison table.")
        else:
            print("No metrics data collected to generate comparison plot or table.")

    print("\nScript finished.")
    print("Generated HTML files with plots can be found in the current directory.")
