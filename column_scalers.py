# Heuristics for scaler selection:
# 1. RobustScaler: If abs(skewness) > 1.5 OR kurtosis > 3.0 (Fisher's definition, where normal is ~0).
#    This handles significant skewness or outliers.
# 2. MinMaxScaler: Otherwise. Suitable for less skewed data, already somewhat normalized features,
#    or when a specific range (e.g., 0-1) is desired.

column_to_scaler_map = {
    # Based on analysis of data/revenue_30d.parquet after dropping categoricals
    # Column Name: (Skew, Kurtosis) -> Scaler
    "avg_user_order_revenue_usd_0h_24h": "RobustScaler",  # (13.16, 889.84)
    "avg_user_order_revenue_usd_0h_48h": "RobustScaler",  # (12.02, 756.41)
    "avg_user_order_revenue_usd_24h_48h": "RobustScaler",  # (7.34, 145.71)
    "cum_perc_revenue_usd_hourly_01": "RobustScaler",  # (353.15, 130846.94)
    "cum_perc_revenue_usd_hourly_02": "RobustScaler",  # (351.95, 130255.50)
    "cum_perc_revenue_usd_hourly_03": "RobustScaler",  # (351.44, 130001.02)
    "cum_perc_revenue_usd_hourly_04": "RobustScaler",  # (351.12, 129846.24)
    "cum_perc_revenue_usd_hourly_05": "RobustScaler",  # (350.91, 129740.94)
    "cum_perc_revenue_usd_hourly_06": "RobustScaler",  # (350.74, 129659.42)
    "cum_perc_revenue_usd_hourly_07": "RobustScaler",  # (350.62, 129599.23)
    "cum_perc_revenue_usd_hourly_08": "RobustScaler",  # (350.52, 129551.47)
    "cum_perc_revenue_usd_hourly_09": "RobustScaler",  # (350.43, 129505.79)
    "cum_perc_revenue_usd_hourly_10": "RobustScaler",  # (350.36, 129474.03)
    "cum_perc_revenue_usd_hourly_11": "RobustScaler",  # (350.31, 129448.53)
    "cum_perc_revenue_usd_hourly_12": "RobustScaler",  # (350.27, 129426.46)
    "cum_perc_revenue_usd_hourly_13": "RobustScaler",  # (350.23, 129407.33)
    "cum_perc_revenue_usd_hourly_14": "RobustScaler",  # (350.20, 129396.32)
    "cum_perc_revenue_usd_hourly_15": "RobustScaler",  # (350.18, 129382.80)
    "cum_perc_revenue_usd_hourly_16": "RobustScaler",  # (350.15, 129371.15)
    "cum_perc_revenue_usd_hourly_17": "RobustScaler",  # (350.13, 129362.36)
    "cum_perc_revenue_usd_hourly_18": "RobustScaler",  # (350.12, 129353.82)
    "cum_perc_revenue_usd_hourly_19": "RobustScaler",  # (350.10, 129344.40)
    "cum_perc_revenue_usd_hourly_20": "RobustScaler",  # (350.09, 129338.60)
    "cum_perc_revenue_usd_hourly_21": "RobustScaler",  # (350.08, 129335.37)
    "cum_perc_revenue_usd_hourly_22": "RobustScaler",  # (350.07, 129333.51)
    "cum_perc_revenue_usd_hourly_23": "RobustScaler",  # (434.01, 206237.87)
    "cum_perc_revenue_usd_hourly_24": "RobustScaler",  # (434.04, 206259.65)
    "cum_perc_revenue_usd_hourly_25": "RobustScaler",  # (434.09, 206293.84)
    "cum_perc_revenue_usd_hourly_26": "RobustScaler",  # (434.15, 206330.57)
    "cum_perc_revenue_usd_hourly_27": "RobustScaler",  # (434.25, 206392.94)
    "cum_perc_revenue_usd_hourly_28": "RobustScaler",  # (434.29, 206419.46)
    "cum_perc_revenue_usd_hourly_29": "RobustScaler",  # (434.33, 206445.89)
    "cum_perc_revenue_usd_hourly_30": "RobustScaler",  # (434.38, 206475.78)
    "cum_perc_revenue_usd_hourly_31": "RobustScaler",  # (460.70, 226180.92)
    "cum_perc_revenue_usd_hourly_32": "MinMaxScaler",  # (-0.40, -1.79)
    "cum_perc_revenue_usd_hourly_33": "MinMaxScaler",  # (-0.41, -1.78)
    "cum_perc_revenue_usd_hourly_34": "MinMaxScaler",  # (-0.43, -1.76)
    "cum_perc_revenue_usd_hourly_35": "MinMaxScaler",  # (-0.45, -1.75)
    "cum_perc_revenue_usd_hourly_36": "MinMaxScaler",  # (-0.46, -1.74)
    "cum_perc_revenue_usd_hourly_37": "MinMaxScaler",  # (-0.48, -1.72)
    "cum_perc_revenue_usd_hourly_38": "MinMaxScaler",  # (-0.50, -1.71)
    "cum_perc_revenue_usd_hourly_39": "MinMaxScaler",  # (-0.51, -1.70)
    "cum_perc_revenue_usd_hourly_40": "MinMaxScaler",  # (-0.53, -1.68)
    "cum_perc_revenue_usd_hourly_41": "MinMaxScaler",  # (-0.55, -1.66)
    "cum_perc_revenue_usd_hourly_42": "MinMaxScaler",  # (-0.57, -1.64)
    "cum_perc_revenue_usd_hourly_43": "MinMaxScaler",  # (-0.59, -1.62)
    "cum_perc_revenue_usd_hourly_44": "MinMaxScaler",  # (-0.61, -1.60)
    "cum_perc_revenue_usd_hourly_45": "MinMaxScaler",  # (-0.63, -1.57)
    "cum_perc_revenue_usd_hourly_46": "MinMaxScaler",  # (-0.66, -1.54)
    "cum_perc_revenue_usd_hourly_47": "MinMaxScaler",  # (-0.69, -1.50)
    "cum_perc_revenue_usd_hourly_48": "MinMaxScaler",  # (-0.73, -1.46)
    "diff_avg_user_order_revenue_usd_24h_48h_vs_0h_24h": "RobustScaler",  # (-7.71, 548.23)
    "diff_n_user_orders_24h_48h_vs_0h_24h": "RobustScaler",  # (-2.18, 27.65)
    "label_id": "RobustScaler",  # (5.21, 35.20) - float16, likely categorical ID if not target
    "n_user_orders_0h_24h": "RobustScaler",  # (4.30, 42.73)
    "n_user_orders_0h_48h": "RobustScaler",  # (4.98, 54.63)
    "n_user_orders_24h_48h": "RobustScaler",  # (6.95, 102.06)
    "payout": "RobustScaler",  # (27.68, 3837.57)
    "perc_diff_avg_user_order_revenue_usd_24h_48h_vs_0h_24h": "RobustScaler",  # (4.19, 64.62)
    "perc_diff_n_user_orders_24h_48h_vs_0h_24h": "RobustScaler",  # (1.55, 16.89)
    "soi_to_max_order_created_at_diff_days": "MinMaxScaler",  # (0.15, -1.22)
    "sum_revenue_0h_24h": "RobustScaler",  # (8.04, 230.46)
    "sum_revenue_0h_48h": "RobustScaler",  # (7.45, 139.53)
    "sum_revenue_24h_48h": "RobustScaler",  # (10.23, 271.12)
    "user_age": "MinMaxScaler",  # (-0.30, 0.02) - Appears pre-scaled
    "user_birth_day_of_month": "MinMaxScaler",  # (0.16, -1.19) - Appears pre-scaled
    "user_birth_month": "MinMaxScaler",  # (0.04, -1.18) - Appears pre-scaled
    "user_days_to_closest_birthday": "MinMaxScaler",  # (-0.01, -1.19) - Appears pre-scaled (-1 to 1)
    "user_first_order_revenue_usd": "RobustScaler",  # (20.93, 2217.70)
    "user_hours_to_first_order": "RobustScaler",  # (2.24, 4.24)
    "user_label_id": "RobustScaler",  # (5.21, 35.20) - uint8, likely categorical ID if not target
    "user_looking_for_age_max": "MinMaxScaler",  # (0.43, 0.87) - Appears pre-scaled
    "user_looking_for_age_min": "MinMaxScaler",  # (0.51, -0.25) - Appears pre-scaled
    "user_profile_description_length": "RobustScaler",  # (1.81, 2.53)
    "user_registration_confirmation_minutes_diff": "RobustScaler",  # (83.75, 8971.07)
    "user_registration_day_of_month": "MinMaxScaler",  # (-0.01, -1.17) - Appears pre-scaled
    "user_registration_day_of_week": "MinMaxScaler",  # (-0.05, -1.27) - Appears pre-scaled
    "user_registration_isoweek": "MinMaxScaler",  # (-0.05, -1.37) - Appears pre-scaled
    "user_registration_month": "MinMaxScaler",  # (-0.06, -1.40) - Appears pre-scaled
    "user_registration_part_of_day": "MinMaxScaler",  # (-0.07, -1.07) - Appears pre-scaled
    "user_registration_year": "MinMaxScaler",  # (-0.02, -0.79) - Appears pre-scaled, low nunique
    "user_revenue_usd_30d": "RobustScaler",  # (11.62, 235.99) - Target/Feature
    "user_revenue_usd_60d": "RobustScaler",  # (14.76, 399.94) - Target/Feature
    "user_revenue_usd_90d": "RobustScaler",  # (16.17, 470.64) - Target/Feature
}

if __name__ == "__main__":
    # Example of how to use this map
    # from sklearn.preprocessing import RobustScaler, MinMaxScaler
    # import pandas as pd

    # # Load your dataframe (ensure it only has the columns in the map)
    # # df = pd.read_parquet("data/revenue_30d.parquet")
    # # # Assuming 'categoricals' list is defined elsewhere and used to drop columns
    # # from exploration.column_config import categoricals # You would need this
    # # df = df.drop(columns=categoricals, errors='ignore')

    # print(f"Scaler map contains {len(column_to_scaler_map)} columns.")

    # # Example: Print columns and their assigned scalers
    # for col, scaler_name in column_to_scaler_map.items():
    #     print(f"Column: {col}, Assigned Scaler: {scaler_name}")

    # # You would then iterate through this map and apply the scalers
    # # For example:
    # # for column, scaler_type_str in column_to_scaler_map.items():
    # #     if column in df.columns:
    # #         data_to_scale = df[[column]]
    # #         if scaler_type_str == "RobustScaler":
    # #             scaler = RobustScaler()
    # #         elif scaler_type_str == "MinMaxScaler":
    # #             scaler = MinMaxScaler()
    # #         else:
    # #             print(f"Unknown scaler type {scaler_type_str} for column {column}")
    # #             continue
    # #         df[column] = scaler.fit_transform(data_to_scale)
    # #     else:
    # #         print(f"Column {column} from map not found in DataFrame.")
    print("column_to_scaler_map dictionary is defined.")
