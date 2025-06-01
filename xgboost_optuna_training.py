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

# M2 Mac optimizations
N_CORES = multiprocessing.cpu_count()  # Use all available cores
print(f"Detected {N_CORES} CPU cores on M2 Mac")


# Check for GPU availability (Metal Performance Shaders on M2)
def check_gpu_support():
    """Check if GPU training is available on M2 Mac."""
    try:
        # Try to create a simple XGBoost model with GPU
        test_model = xgb.XGBRegressor(tree_method="gpu_hist", gpu_id=0, n_estimators=1)
        # Create dummy data to test
        X_test = np.random.random((10, 5))
        y_test = np.random.random(10)
        test_model.fit(X_test, y_test)
        print("✅ GPU support (Metal) detected and working!")
        return True
    except Exception as e:
        print(f"❌ GPU support not available: {e}")
        print("Using optimized CPU training instead")
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

    # M2 Mac optimized parameters
    params = {
        "objective": objective_func,
        "booster": trial.suggest_categorical(
            "booster", ["gbtree"]
        ),  # Focus on gbtree for speed
        "lambda": trial.suggest_float(
            "lambda", 1e-8, 10.0, log=True
        ),  # Increased range
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),  # Increased range
        "subsample": trial.suggest_float(
            "subsample", 0.7, 1.0
        ),  # Narrowed range for speed
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),  # Narrowed range for speed
        "max_depth": trial.suggest_int("max_depth", 4, 10),  # Reduced range for speed
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 15
        ),  # Reduced range
        "eta": trial.suggest_float(
            "eta", 0.05, 0.3, log=True
        ),  # Slightly higher minimum for faster convergence
        "random_state": RANDOM_SEED,
        "n_jobs": N_CORES,  # Use all available cores
        "early_stopping_rounds": 30,  # Reduced for faster trials
        # M2 Mac optimizations
        "tree_method": "gpu_hist" if GPU_AVAILABLE else "hist",  # Use GPU if available
        "max_bin": 256,  # Optimized for M2 memory bandwidth
    }

    # GPU-specific optimizations
    if GPU_AVAILABLE:
        params["gpu_id"] = 0
        params["predictor"] = "gpu_predictor"
    else:
        # CPU optimizations for M2
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

    # Reduced n_estimators for faster trials, early stopping will handle convergence
    model = xgb.XGBRegressor(n_estimators=1000, **params)  # Reduced from 2000

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
            trial.set_user_attr("best_n_estimators", 1000)  # The initial n_estimators

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

    X_train, X_test, y_train, y_test = split_data(df, TARGET_COLUMN)

    print(f"\nStarting Optuna hyperparameter optimization...")
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
        "best_n_estimators", 1000
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

    print(f"\nTraining final model with best parameters...")
    final_model_params = best_params_from_optuna.copy()

    # Remove objective from params since we'll set it explicitly
    if "objective" in final_model_params:
        del final_model_params["objective"]

    # M2 Mac optimized final model
    final_model_params.update(
        {
            "n_jobs": N_CORES,
            "tree_method": "gpu_hist" if GPU_AVAILABLE else "hist",
            "max_bin": 256 if GPU_AVAILABLE else 512,
        }
    )

    if GPU_AVAILABLE:
        final_model_params["gpu_id"] = 0
        final_model_params["predictor"] = "gpu_predictor"

    final_model = xgb.XGBRegressor(
        objective=best_objective,  # Use the best objective found
        random_state=RANDOM_SEED,
        n_estimators=best_n_estimators,
        early_stopping_rounds=30,  # Reduced for speed
        **final_model_params,
    )

    print("Fitting final model on full training set...")

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

    actual_n_estimators_final = getattr(final_model, "best_iteration", None)
    if actual_n_estimators_final is not None and actual_n_estimators_final > 0:
        print(
            f"  Final model actual n_estimators after early stopping: {actual_n_estimators_final + 1}"
        )
    elif hasattr(final_model, "n_estimators_"):
        print(
            f"  Final model actual n_estimators (no early stop or stop at max): {final_model.n_estimators_}"
        )
    else:
        print(
            f"  Final model actual n_estimators: {best_n_estimators} (could not read from model after fit)"
        )

    print("Evaluating model on the test set...")
    y_pred = final_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test Set Metrics for {dataset_name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R-squared: {r2:.4f}")

    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    feature_importances = pd.Series(
        final_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    potential_omittable_features = feature_importances[
        feature_importances < 0.001
    ].index.tolist()
    print(
        f"\nPotentially omittable features (importance < 0.001) for {dataset_name}: {potential_omittable_features}"
    )

    fig_metrics = go.Figure(
        data=[
            go.Bar(
                name="Test Metrics", x=list(metrics.keys()), y=list(metrics.values())
            )
        ]
    )
    fig_metrics.update_layout(title_text=f"Model Performance Metrics - {dataset_name}")
    fig_metrics.write_html(f"metrics_{dataset_name.replace(' ', '_')}.html")

    fig_feature_importance = px.bar(
        feature_importances,
        x=feature_importances.values,
        y=feature_importances.index,
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        title=f"Feature Importance - {dataset_name}",
    )
    fig_feature_importance.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_feature_importance.write_html(
        f"feature_importance_{dataset_name.replace(' ', '_')}.html"
    )

    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            name="Actual vs. Predicted",
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
        title=f"Actual vs. Predicted Revenue - {dataset_name}",
        xaxis_title="Actual Revenue (USD 30d)",
        yaxis_title="Predicted Revenue (USD 30d)",
    )
    fig_scatter.write_html(f"actual_vs_predicted_{dataset_name.replace(' ', '_')}.html")

    return final_model, metrics, feature_importances


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
