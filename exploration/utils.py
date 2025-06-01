import pandas as pd
import numpy as np
from category_encoders import TargetEncoder


def gender_map(val: str) -> str:
    val_lower = str(val).strip().lower()
    if pd.isna(val) or val_lower == "":
        return "unknown"
    if val_lower in ["male", "female"]:
        return val_lower
    else:
        return "male"  # or "unknown" if you want to catch all other cases


def cast_to_string(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    for col in _df.columns:
        if _df[col].dtype not in [
            "int64",
            "int32",
            "int16",
            "int8",
            "float64",
            "float32",
            "float16",
            "bool",
            "datetime64[ns]",
            "timedelta64[ns]",
            "uint64",
            "uint32",
            "uint16",
            "uint8",
        ]:
            _df[col] = _df[col].astype(str)
            # print(f"Casting '{col}' to string >> {_df[col].dtype}")
            # print(f"{col}:string")
    return _df


import pandas as pd


def top_categories(
    df: pd.DataFrame, col: str, n: int = 50, include_na: bool = False
) -> pd.DataFrame:
    counts = df[col].value_counts(dropna=not include_na).head(n)
    percentages = (counts / len(df) * 100).round(2)
    result = pd.DataFrame({"count": counts, "percentage": percentages})
    return result


def process_email(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Process the email column to extract domain and clean up the data.
    """
    df[col] = df[col].str.lower().str.strip()
    return df


def marital_status_map(val: str) -> str:
    _map = {
        "Divorcée ?": "divorced",
        "célibataire": "single",
        "сингл": "single",
        "Pojedynczy": "single",
        "soltero": "single",
        "Célibataire.": "single",
        "Marié": "married",
        "szingli": "single",
        "싱글": "single",
        "تک تک": "married",
        "Divorciado": "divorced",
        "casado": "married",
    }
    val_lower = str(val).replace(".", "").replace("?", "").strip().lower()
    if pd.isna(val) or val_lower == "":
        return "unknown"
    if val_lower in [
        "single",
        "divorced",
        "relationship",
        "widowed",
        "unknown",
        "married",
    ]:
        return val_lower
    elif val_lower in _map:
        return _map[val_lower]
    else:
        return "unknown"  # or "unknown" if you want to catch all other cases


def clip_column_by_percentile(
    df: pd.DataFrame,
    cols: str | list[str],
    lower_pct: float = 0.0,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        series = df[col].dropna()
        lower_val = series.quantile(lower_pct)
        upper_val = series.quantile(upper_pct)
        df[col] = df[col].clip(lower=lower_val, upper=upper_val)
    return df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    FunctionTransformer,
)


def compare_scalers(
    df: pd.DataFrame,
    column: str,
    scaler_list=None,
    save_path=None,
    dpi=300,
    figsize=(12, 8),
):
    """
    Apply multiple scalers to a column and plot KDE distributions for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Name of the column to scale and plot.
    scaler_list : list of str, optional
        List of scalers to apply. One or more of:
        - 'original': the raw data
        - 'standard' : StandardScaler
        - 'minmax'   : MinMaxScaler
        - 'robust'   : RobustScaler
        - 'power_yeo': PowerTransformer (Yeo-Johnson)
        - 'power_box': PowerTransformer (Box-Cox, requires positive values)
        - 'log'      : log(x + 1) transform
    save_path : str, optional
        Path to save the plot as PNG. If None, generates a default filename.
    dpi : int, optional
        Resolution for saved image (default: 300).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 8)).
    """
    # Default list if none provided
    if scaler_list is None:
        scaler_list = ["original", "standard", "minmax", "robust", "power_yeo", "log"]

    # Validate column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Extract and reshape data
    data = df[column].fillna(0).values

    if len(data) == 0:
        raise ValueError(f"No valid data found in column '{column}'")

    data = data.reshape(-1, 1)
    original = data.flatten()

    if np.any(original <= 0) and "power_box" in scaler_list:
        print(
            "Warning: Box-Cox transform requires all values to be positive. Skipping 'power_box' scaler."
        )
        scaler_list.remove("power_box")

    # Define transformers
    transformer_map = {
        "original": None,
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "power_yeo": PowerTransformer(method="yeo-johnson"),
        "power_box": PowerTransformer(method="box-cox"),
        "log": FunctionTransformer(np.log1p, validate=False),
    }

    # Create plot with better styling
    plt.figure(figsize=figsize)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    colors = plt.cm.Set3(np.linspace(0, 1, len(scaler_list)))

    # Plot original distribution if requested
    if "original" in scaler_list:
        pd.Series(original).plot.kde(label="Original", linewidth=2, color=colors[0])

    # Apply and plot each scaler
    color_idx = 1 if "original" in scaler_list else 0
    for scaler_name in scaler_list:
        if scaler_name == "original":
            continue
        if scaler_name not in transformer_map:
            print(f"Warning: Unsupported scaler '{scaler_name}'. Skipping.")
            continue

        try:
            transformer = transformer_map[scaler_name]

            transformed = transformer.fit_transform(data).flatten()
            pd.Series(transformed).plot.kde(
                label=scaler_name.replace("_", "-").title(),
                linewidth=2,
                color=colors[color_idx % len(colors)],
            )
            color_idx += 1

        except Exception as e:
            print(f"Warning: Failed to apply '{scaler_name}' scaler: {e}")
            continue

    # Improve plot styling
    plt.title(
        f'Distribution Comparison: "{column}" with Different Scalers',
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Scaled Values", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    if save_path is None:
        save_path = f'scaler_comparison_{column.replace(" ", "_")}.png'

    plt.savefig(
        save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Plot saved as: {save_path}")

    # Show the plot
    plt.show()


def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes by downcasting to the smallest possible types
    based on actual data ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to optimize.
    verbose : bool, optional
        Whether to print optimization details (default: True).

    Returns
    -------
    pd.DataFrame
        DataFrame with optimized dtypes.
    """
    # Create a copy to avoid modifying the original
    optimized_df = df.copy()

    # Track memory usage before optimization
    memory_before = optimized_df.memory_usage(deep=True).sum()

    # Define integer type ranges
    int_types = {
        "int8": (-128, 127),
        "int16": (-32768, 32767),
        "int32": (-2147483648, 2147483647),
        "int64": (-9223372036854775808, 9223372036854775807),
    }

    # Define unsigned integer type ranges
    uint_types = {
        "uint8": (0, 255),
        "uint16": (0, 65535),
        "uint32": (0, 4294967295),
        "uint64": (0, 18446744073709551615),
    }

    # Define float type ranges (approximate)
    float_types = {
        "float16": (-65504, 65504),
        "float32": (-3.4e38, 3.4e38),
        "float64": (-1.8e308, 1.8e308),
    }

    optimizations = []

    for column in optimized_df.columns:
        original_dtype = str(optimized_df[column].dtype)
        col_data = optimized_df[column]

        # Skip if all values are null
        if col_data.isna().all():
            continue

        # Get non-null values for analysis
        non_null_data = col_data.dropna()

        if len(non_null_data) == 0:
            continue

        # Handle numeric types
        if pd.api.types.is_numeric_dtype(col_data):
            col_min = non_null_data.min()
            col_max = non_null_data.max()

            # Check if all values are integers (even if stored as float)
            if pd.api.types.is_float_dtype(col_data):
                # Check if all non-null values are actually integers
                is_integer = non_null_data.apply(lambda x: float(x).is_integer()).all()

                if is_integer:
                    # Convert to integer and find best integer type
                    int_values = non_null_data.astype("int64")
                    int_min = int_values.min()
                    int_max = int_values.max()

                    # Find best unsigned integer type if all values are non-negative
                    if int_min >= 0:
                        for uint_type, (type_min, type_max) in uint_types.items():
                            if int_min >= type_min and int_max <= type_max:
                                optimized_df[column] = col_data.astype(uint_type)
                                optimizations.append(
                                    (column, original_dtype, uint_type)
                                )
                                break
                    else:
                        # Find best signed integer type
                        for int_type, (type_min, type_max) in int_types.items():
                            if int_min >= type_min and int_max <= type_max:
                                optimized_df[column] = col_data.astype(int_type)
                                optimizations.append((column, original_dtype, int_type))
                                break
                else:
                    # Keep as float but optimize float type
                    for float_type, (type_min, type_max) in float_types.items():
                        if col_min >= type_min and col_max <= type_max:
                            # Check if downcasting preserves precision
                            test_series = col_data.astype(float_type)
                            if (
                                test_series.equals(col_data)
                                or (test_series - col_data).abs().max() < 1e-6
                            ):
                                optimized_df[column] = test_series
                                optimizations.append(
                                    (column, original_dtype, float_type)
                                )
                                break

            elif pd.api.types.is_integer_dtype(col_data):
                # Already integer, find best integer type
                if col_min >= 0:
                    # Try unsigned types first
                    for uint_type, (type_min, type_max) in uint_types.items():
                        if col_min >= type_min and col_max <= type_max:
                            optimized_df[column] = col_data.astype(uint_type)
                            optimizations.append((column, original_dtype, uint_type))
                            break
                else:
                    # Use signed types
                    for int_type, (type_min, type_max) in int_types.items():
                        if col_min >= type_min and col_max <= type_max:
                            optimized_df[column] = col_data.astype(int_type)
                            optimizations.append((column, original_dtype, int_type))
                            break

        # # Handle categorical/object types
        # elif pd.api.types.is_object_dtype(col_data):
        #     # Check if it's a good candidate for categorical
        #     unique_count = non_null_data.nunique()
        #     total_count = len(non_null_data)

        #     # Convert to categorical if less than 50% unique values and more than 2 unique values
        #     if unique_count < total_count * 0.5 and unique_count > 1:
        #         optimized_df[column] = col_data.astype('category')
        #         optimizations.append((column, original_dtype, 'category'))

        # Handle boolean-like data
        elif pd.api.types.is_bool_dtype(col_data):
            # Already optimal for boolean data
            continue

    # Calculate memory usage after optimization
    memory_after = optimized_df.memory_usage(deep=True).sum()
    memory_reduction = memory_before - memory_after
    reduction_percentage = (memory_reduction / memory_before) * 100

    if verbose:
        print(f"Memory optimization completed:")
        print(f"  Original memory usage: {memory_before / 1024**2:.2f} MB")
        print(f"  Optimized memory usage: {memory_after / 1024**2:.2f} MB")
        print(
            f"  Memory reduction: {memory_reduction / 1024**2:.2f} MB ({reduction_percentage:.1f}%)"
        )
        print(f"  Columns optimized: {len(optimizations)}")

        if optimizations and len(optimizations) <= 20:  # Don't print too many
            print("\nOptimization details:")
            for col, old_type, new_type in optimizations:
                print(f"  {col}: {old_type} → {new_type}")
        elif len(optimizations) > 20:
            print(f"\nFirst 20 optimization details:")
            for col, old_type, new_type in optimizations:
                print(f"  {col}: {old_type} → {new_type}")

    return optimized_df


def analyze_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze DataFrame dtypes and provide optimization suggestions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        Analysis results with optimization suggestions.
    """
    analysis = []

    for column in df.columns:
        col_data = df[column]
        current_dtype = str(col_data.dtype)
        memory_usage = col_data.memory_usage(deep=True)

        # Get basic statistics
        non_null_count = col_data.count()
        null_count = col_data.isna().sum()

        suggestion = current_dtype  # Default: no change

        if pd.api.types.is_numeric_dtype(col_data) and non_null_count > 0:
            col_min = col_data.min()
            col_max = col_data.max()

            if pd.api.types.is_float_dtype(col_data):
                # Check if it's actually integer data
                non_null_data = col_data.dropna()
                if non_null_data.apply(lambda x: float(x).is_integer()).all():
                    if col_min >= 0:
                        if col_max <= 255:
                            suggestion = "uint8"
                        elif col_max <= 65535:
                            suggestion = "uint16"
                        elif col_max <= 4294967295:
                            suggestion = "uint32"
                    else:
                        if col_min >= -128 and col_max <= 127:
                            suggestion = "int8"
                        elif col_min >= -32768 and col_max <= 32767:
                            suggestion = "int16"
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            suggestion = "int32"
                else:
                    # Keep as float but suggest smaller float type if possible
                    if abs(col_min) <= 65504 and abs(col_max) <= 65504:
                        suggestion = "float32"

            elif pd.api.types.is_integer_dtype(col_data):
                if col_min >= 0:
                    if col_max <= 255:
                        suggestion = "uint8"
                    elif col_max <= 65535:
                        suggestion = "uint16"
                    elif col_max <= 4294967295:
                        suggestion = "uint32"
                else:
                    if col_min >= -128 and col_max <= 127:
                        suggestion = "int8"
                    elif col_min >= -32768 and col_max <= 32767:
                        suggestion = "int16"
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        suggestion = "int32"

        elif pd.api.types.is_object_dtype(col_data):
            unique_count = col_data.nunique()
            if unique_count < len(col_data) * 0.5 and unique_count > 1:
                suggestion = "category"

        analysis.append(
            {
                "column": column,
                "current_dtype": current_dtype,
                "suggested_dtype": suggestion,
                "memory_usage_bytes": memory_usage,
                "non_null_count": non_null_count,
                "null_count": null_count,
                "unique_count": col_data.nunique() if non_null_count > 0 else 0,
            }
        )

    return pd.DataFrame(analysis)


def scale_with_max_value(df: pd.DataFrame, col: str):
    df[col] = df[col] / df[col].max()
    return df


def day_name_to_num(day_name: str) -> int:
    """
    Convert an uppercase day name (e.g. 'MONDAY') to a number 0–6.
    Returns NaN for unrecognized values.
    """
    mapping = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    return mapping.get(str(day_name).lower())


def cast_ints_and_floats(df):
    for col in df.columns:
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.integer):
            df[col] = df[col].astype("int64")
        elif np.issubdtype(dtype, np.floating):
            df[col] = df[col].astype("float64")
    return df


def exclude_user_hourly_cum_perc_revenue_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        name for name in df.columns if "cum_perc_revenue_usd_hourly_" in name
    ]
    return df.drop(columns=cols_to_drop)


def scale_and_log_numerical_df(df):
    log_numerical_df = np.log1p(df.copy())
    log_numerical_df = log_numerical_df.replace([np.inf, -np.inf], np.nan)
    print(f"NA rows number: {log_numerical_df.isna().sum().sum()}")
    return log_numerical_df.dropna(axis=0, how="any")


def encode_categorical_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    target_column: str | None = None,
    encoding_type: str = "frequency",
) -> pd.DataFrame:
    """
    Encodes specified categorical columns in a DataFrame using either frequency
    or target encoding.

    For target encoding, if train_df and test_df are provided, the encoder is
    fit on train_df and applied to both. Otherwise, it's fit and transformed
    on the input df.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list[str]): A list of column names to encode.
        target_column (str | None, optional): The name of the target variable column.
            Required if encoding_type is 'target'. Defaults to None.
        encoding_type (str, optional): The type of encoding to perform.
            Options are "frequency" or "target". Defaults to "frequency".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None] | pd.DataFrame:
        If train_df and test_df are provided, returns a tuple of (encoded_train_df, encoded_test_df).
        Otherwise, returns the encoded DataFrame.

    Raises:
        ValueError: If encoding_type is 'target' and target_column is not provided.
        ValueError: If an unsupported encoding_type is provided.
        ValueError: If columns to encode are not found in the DataFrame.
    """
    _df = df.copy()
    columns = columns or _df.columns

    if encoding_type == "frequency":
        for col in columns:
            freq_map = _df[col].value_counts(normalize=True)
            _df[f"{col}_freq_encoded"] = _df[col].map(freq_map).fillna(0)
        _df = _df.drop(columns=columns)
        return _df

    elif encoding_type == "target":
        if target_column is None:
            raise ValueError("target_column must be provided for target encoding.")
        if target_column not in _df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in the DataFrame."
            )

        encoder = TargetEncoder(
            cols=columns, handle_missing="value", handle_unknown="value"
        )

        _df[columns] = encoder.fit_transform(_df[columns], _df[target_column])
        # Rename columns to indicate encoding
        rename_map = {
            col: f"{col}_target_encoded" for col in columns if col in _df.columns
        }
        _df = _df.rename(columns=rename_map)
        return _df

    else:
        raise ValueError(
            f"Unsupported encoding_type: {encoding_type}. "
            "Choose 'frequency' or 'target'."
        )


def convert_to_absolute_values(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "diff_n_user_orders_24h_48h_vs_0h_24h",
        "avg_user_order_revenue_usd_24h_48h",
        "perc_diff_avg_user_order_revenue_usd_24h_48h_vs_0h_24h",
        "perc_diff_n_user_orders_24h_48h_vs_0h_24h",
    ]
    for col in cols:
        df[col] = df[col].abs()

    return df
