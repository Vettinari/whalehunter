import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

warnings.filterwarnings("ignore")


class ParquetTypeCorrector:
    """
    A class to read parquet files, analyze data types, and apply appropriate corrections.
    """

    def __init__(self, input_file: str, output_file: str = None):
        self.input_file = input_file
        self.output_file = output_file or input_file.replace(
            ".parquet", "_corrected.parquet"
        )
        self.df = None
        self.original_schema = None
        self.corrected_schema = None

    def load_data(self) -> pd.DataFrame:
        """Load the parquet file with error handling for type issues."""
        print("=" * 60)
        print("LOADING RAW DATA")
        print("=" * 60)

        try:
            # First, try to read the schema to understand the structure
            parquet_file = pq.ParquetFile(self.input_file)
            self.original_schema = parquet_file.schema_arrow

            print(f"Original schema has {len(self.original_schema)} columns")
            print(f"Number of rows: {parquet_file.metadata.num_rows:,}")

            # Try reading with pandas first
            try:
                print("Attempting to load with pandas...")
                self.df = pd.read_parquet(self.input_file)
                print("✓ Successfully loaded with pandas")
            except Exception as e:
                print(f"✗ Pandas failed: {e}")
                print("Attempting to load with pyarrow and type conversion...")

                # Read with pyarrow and handle problematic types
                table = pq.read_table(self.input_file)

                # Convert problematic date columns to strings first
                new_fields = []
                for field in table.schema:
                    field_name = field.name
                    field_type = str(field.type)

                    # Handle problematic date types
                    if any(
                        problem in field_type.lower()
                        for problem in ["dbdate", "timestamp64"]
                    ):
                        print(
                            f"Converting problematic type {field_name}: {field_type} -> string"
                        )
                        new_fields.append(pa.field(field_name, pa.string()))
                    else:
                        new_fields.append(field)

                # Apply schema conversion if needed
                if any(
                    pa.field(f.name, pa.string()) in new_fields for f in table.schema
                ):
                    new_schema = pa.schema(new_fields)
                    table = table.cast(new_schema)

                self.df = table.to_pandas()
                print("✓ Successfully loaded with pyarrow conversion")

            print(f"Data shape: {self.df.shape}")
            print(
                f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
            )

            return self.df

        except Exception as e:
            print(f"✗ Failed to load data: {e}")
            raise

    def analyze_columns(self) -> Dict[str, Any]:
        """Analyze column types and suggest corrections."""
        print("\n" + "=" * 60)
        print("ANALYZING COLUMN TYPES")
        print("=" * 60)

        analysis = {
            "numeric_columns": [],
            "text_columns": [],
            "time_columns": [],
            "boolean_columns": [],
            "categorical_columns": [],
            "problematic_columns": [],
            "corrections_needed": {},
        }

        for col in self.df.columns:
            dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()
            null_count = self.df[col].isnull().sum()
            sample_values = self.df[col].dropna().head(3).tolist()

            print(
                f"{col:40} | {str(dtype):15} | Nulls: {null_count:8} | Unique: {unique_count:8}"
            )

            # Analyze and categorize columns
            if self._is_numeric_column(col, dtype, sample_values):
                analysis["numeric_columns"].append(col)
                # Check if it should be integer instead of float
                if dtype == "float64" and self._could_be_integer(col):
                    analysis["corrections_needed"][col] = "int64"

            elif self._is_time_column(col, dtype, sample_values):
                analysis["time_columns"].append(col)
                analysis["corrections_needed"][col] = "datetime64[ns]"

            elif self._is_boolean_column(col, dtype, sample_values, unique_count):
                analysis["boolean_columns"].append(col)
                analysis["corrections_needed"][col] = "bool"

            elif self._is_categorical_column(col, dtype, unique_count, len(self.df)):
                analysis["categorical_columns"].append(col)
                analysis["corrections_needed"][col] = "category"

            else:
                analysis["text_columns"].append(col)
                # Check if string column needs optimization
                if dtype == "object" and self._could_be_string(col):
                    analysis["corrections_needed"][col] = "string"

        self._print_analysis_summary(analysis)
        return analysis

    def _is_numeric_column(self, col: str, dtype: Any, sample_values: List) -> bool:
        """Check if column should be numeric."""
        if pd.api.types.is_numeric_dtype(dtype):
            return True

        # Check for numeric keywords in column name
        numeric_keywords = [
            "revenue",
            "amount",
            "value",
            "price",
            "count",
            "sum",
            "total",
            "hour",
            "day",
        ]
        return any(keyword in col.lower() for keyword in numeric_keywords)

    def _is_time_column(self, col: str, dtype: Any, sample_values: List) -> bool:
        """Check if column should be datetime."""
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return True

        # Check for time keywords in column name
        time_keywords = ["date", "time", "timestamp", "doi", "created", "updated"]
        col_indicates_time = any(keyword in col.lower() for keyword in time_keywords)

        # Check sample values for date patterns
        if col_indicates_time and sample_values:
            try:
                pd.to_datetime(sample_values[0])
                return True
            except:
                pass

        return False

    def _is_boolean_column(
        self, col: str, dtype: Any, sample_values: List, unique_count: int
    ) -> bool:
        """Check if column should be boolean."""
        if pd.api.types.is_bool_dtype(dtype):
            return True

        # Check if binary numeric (0/1 only)
        if unique_count <= 2 and sample_values:
            unique_vals = set(sample_values)
            if unique_vals.issubset(
                {0, 1, True, False, "True", "False", "true", "false"}
            ):
                return True

        return False

    def _is_categorical_column(
        self, col: str, dtype: Any, unique_count: int, total_rows: int
    ) -> bool:
        """Check if column should be categorical."""
        # If less than 50 unique values and less than 10% of total rows, consider categorical
        if unique_count < 50 and unique_count < (total_rows * 0.1):
            return True
        return False

    def _could_be_integer(self, col: str) -> bool:
        """Check if float column could be integer."""
        if self.df[col].dtype == "float64":
            # Check if all non-null values are whole numbers
            non_null = self.df[col].dropna()
            if len(non_null) > 0:
                return (non_null % 1 == 0).all()
        return False

    def _could_be_string(self, col: str) -> bool:
        """Check if object column should be string type."""
        return True  # Most object columns should be string for clarity

    def _print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print summary of analysis."""
        print(f"\n{'='*60}")
        print("COLUMN ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Numeric columns: {len(analysis['numeric_columns'])}")
        print(f"Text columns: {len(analysis['text_columns'])}")
        print(f"Time columns: {len(analysis['time_columns'])}")
        print(f"Boolean columns: {len(analysis['boolean_columns'])}")
        print(f"Categorical columns: {len(analysis['categorical_columns'])}")
        print(f"Corrections needed: {len(analysis['corrections_needed'])}")

        if analysis["corrections_needed"]:
            print("\nSuggested type corrections:")
            for col, new_type in analysis["corrections_needed"].items():
                current_type = self.df[col].dtype
                print(f"  {col}: {current_type} -> {new_type}")

    def apply_corrections(self, analysis: Dict[str, Any]) -> pd.DataFrame:
        """Apply type corrections to the dataframe."""
        print(f"\n{'='*60}")
        print("APPLYING TYPE CORRECTIONS")
        print("=" * 60)

        df_corrected = self.df.copy()
        successful_corrections = 0
        failed_corrections = []

        for col, target_type in analysis["corrections_needed"].items():
            try:
                print(f"Converting {col} to {target_type}...")

                if target_type == "datetime64[ns]":
                    df_corrected[col] = pd.to_datetime(
                        df_corrected[col], errors="coerce"
                    )

                elif target_type == "int64":
                    # Handle NaN values for integer conversion
                    df_corrected[col] = df_corrected[col].fillna(0).astype("int64")

                elif target_type == "bool":
                    # Convert to boolean, handling various formats
                    df_corrected[col] = df_corrected[col].map(
                        {
                            1: True,
                            0: False,
                            "1": True,
                            "0": False,
                            "True": True,
                            "False": False,
                            "true": True,
                            "false": False,
                            True: True,
                            False: False,
                        }
                    )

                elif target_type == "category":
                    df_corrected[col] = df_corrected[col].astype("category")

                elif target_type == "string":
                    df_corrected[col] = df_corrected[col].astype("string")

                successful_corrections += 1
                print(f"  ✓ Successfully converted {col}")

            except Exception as e:
                failed_corrections.append((col, str(e)))
                print(f"  ✗ Failed to convert {col}: {e}")

        print(f"\nType correction summary:")
        print(f"  Successful: {successful_corrections}")
        print(f"  Failed: {len(failed_corrections)}")

        if failed_corrections:
            print("  Failed conversions:")
            for col, error in failed_corrections:
                print(f"    {col}: {error}")

        return df_corrected

    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage by downcasting numeric types."""
        print(f"\n{'='*60}")
        print("OPTIMIZING MEMORY USAGE")
        print("=" * 60)

        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Original memory usage: {original_memory:.1f} MB")

        df_optimized = df.copy()

        # Optimize integer columns
        int_cols = df_optimized.select_dtypes(include=["int64"]).columns
        for col in int_cols:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()

            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype("uint8")
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype("uint16")
                elif col_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype("uint32")
            else:  # Signed integers
                if col_min > -128 and col_max < 127:
                    df_optimized[col] = df_optimized[col].astype("int8")
                elif col_min > -32768 and col_max < 32767:
                    df_optimized[col] = df_optimized[col].astype("int16")
                elif col_min > -2147483648 and col_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype("int32")

        # Optimize float columns
        float_cols = df_optimized.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

        optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = (
            (original_memory - optimized_memory) / original_memory
        ) * 100

        print(f"Optimized memory usage: {optimized_memory:.1f} MB")
        print(f"Memory reduction: {memory_reduction:.1f}%")

        return df_optimized

    def save_corrected_data(self, df: pd.DataFrame):
        """Save the corrected dataframe as parquet."""
        print(f"\n{'='*60}")
        print("SAVING CORRECTED DATA")
        print("=" * 60)

        try:
            # Save as parquet
            df.to_parquet(self.output_file, compression="snappy", index=False)

            # Verify the saved file
            test_df = pd.read_parquet(self.output_file)

            print(f"✓ Successfully saved to: {self.output_file}")
            print(f"  Rows: {len(test_df):,}")
            print(f"  Columns: {len(test_df.columns):,}")

            # Show final schema
            print(f"\nFinal data types:")
            for col in test_df.columns[:20]:  # Show first 20 columns
                print(f"  {col:40} | {str(test_df[col].dtype):15}")

            if len(test_df.columns) > 20:
                print(f"  ... and {len(test_df.columns) - 20} more columns")

            return True

        except Exception as e:
            print(f"✗ Failed to save: {e}")
            return False

    def process(self) -> bool:
        """Main processing pipeline."""
        try:
            # Load data
            self.load_data()

            # Analyze column types
            analysis = self.analyze_columns()

            # Apply corrections
            df_corrected = self.apply_corrections(analysis)

            # Optimize memory
            df_optimized = self.optimize_memory(df_corrected)

            # Save corrected data
            success = self.save_corrected_data(df_optimized)

            if success:
                print(f"\n{'='*60}")
                print("PROCESS COMPLETED SUCCESSFULLY")
                print("=" * 60)
                print(f"Input file: {self.input_file}")
                print(f"Output file: {self.output_file}")
                print(f"Type corrections applied and data optimized!")

            return success

        except Exception as e:
            print(f"\n{'='*60}")
            print("PROCESS FAILED")
            print("=" * 60)
            print(f"Error: {e}")
            return False


def main():
    """Main function to run the parquet type correction."""
    print("WhaleHunter - Parquet Type Corrector")
    print("=" * 60)

    input_file = "raw_data.parquet"
    output_file = "corrected_data.parquet"

    # Check if input file exists
    import os

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please ensure the raw_data.parquet file is in the current directory.")
        return False

    # Create corrector instance
    corrector = ParquetTypeCorrector(input_file, output_file)

    # Process the data
    success = corrector.process()

    if success:
        print(f"\n🎉 Successfully created {output_file} with corrected data types!")
        print("\nYou can now use this file in your analysis scripts:")
        print(f"  df = pd.read_parquet('{output_file}')")
    else:
        print(f"\n❌ Failed to process the file. Check the error messages above.")

    return success


if __name__ == "__main__":
    main()
