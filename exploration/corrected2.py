# from exploration.utils import cast_to_string, gender_map, marital_status_map, optimize_dtypes, scale_with_max_value ,day_name_to_num
# from exploration.column_config import cum_perc_revenue_map, drop_columns, max_scalers

# df = raw_df.copy()
# df = df.rename(columns=cum_perc_revenue_map)
# cols = sorted([col for col in df.columns if col not in drop_columns])
# df = df[cols]

# df['payout'] = df['payout'].astype(float).round(2)
# df = cast_to_string(df)
# df['user_gender'] = df["user_gender"].apply(gender_map)
# df['user_marital_status'] = df["user_marital_status"].apply(marital_status_map)
# df['user_age'] = df['user_age'].clip(lower=16, upper=100)
# df['user_registration_day_of_week'] = df['user_registration_day_of_week'].apply(day_name_to_num)
# df['user_registration_year'] = df['user_registration_year'] - df['user_registration_year'].min()
# df = df.fillna(0)

# for scaler in max_scalers:
#     df = scale_with_max_value(df, scaler)

# df = optimize_dtypes(df)
# df.to_parquet('data/starting_data.parquet')