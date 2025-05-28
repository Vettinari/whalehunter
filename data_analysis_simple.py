import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

def load_data_simple():
    """Load data with a simpler approach to bypass type issues"""
    try:
        print("Attempting to load with pandas read_parquet...")
        # Try with pandas directly, letting it handle type inference
        df = pd.read_parquet('raw_data.parquet')
        print("Success! Data loaded with pandas")
        return df
    except Exception as e1:
        print(f"Pandas approach failed: {e1}")
        
        try:
            print("Trying pyarrow with type coercion...")
            # Read with pyarrow and convert all problematic columns to strings
            parquet_file = pq.ParquetFile('raw_data.parquet')
            
            # Read in batches to handle memory
            print("Reading parquet metadata...")
            schema = parquet_file.schema_arrow
            
            # Convert all date-like fields to string in the schema
            import pyarrow as pa
            new_fields = []
            for field in schema:
                if 'date' in str(field.type).lower() or 'dbdate' in str(field.type).lower():
                    new_fields.append(pa.field(field.name, pa.string()))
                    print(f"Converting {field.name} to string")
                else:
                    new_fields.append(field)
            
            new_schema = pa.schema(new_fields)
            
            # Read the table
            table = pq.read_table('raw_data.parquet')
            
            # Convert to pandas DataFrame
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            print("Success with arrow types!")
            return df
            
        except Exception as e2:
            print(f"PyArrow approach failed: {e2}")
            
            # Final attempt: read with dask if available
            try:
                import dask.dataframe as dd
                print("Trying with dask...")
                ddf = dd.read_parquet('raw_data.parquet')
                df = ddf.compute()
                print("Success with dask!")
                return df
            except Exception as e3:
                print(f"Dask approach failed: {e3}")
                print("All approaches failed. The file may need preprocessing.")
                return None

def quick_analysis(df):
    """Perform quick analysis of the loaded data"""
    if df is None:
        print("No data to analyze")
        return
    
    print(f"\n{'='*60}")
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print(f"\n{'='*60}")
    print("COLUMN TYPES")
    print("="*60)
    print(df.dtypes.value_counts())
    
    print(f"\n{'='*60}")
    print("SAMPLE COLUMNS")
    print("="*60)
    for i, col in enumerate(df.columns[:20]):  # First 20 columns
        print(f"{i+1:2d}. {col}")
    if len(df.columns) > 20:
        print(f"... and {len(df.columns) - 20} more columns")
    
    # Look for key columns
    print(f"\n{'='*60}")
    print("KEY COLUMNS IDENTIFIED")
    print("="*60)
    
    # Revenue columns
    revenue_cols = [col for col in df.columns if 'revenue' in col.lower() and 'usd' in col.lower()]
    print(f"Revenue columns: {revenue_cols}")
    
    # User identification
    user_cols = [col for col in df.columns if 'user_id' in col.lower()]
    print(f"User ID columns: {user_cols}")
    
    # Time columns
    time_cols = [col for col in df.columns if any(t in col.lower() for t in ['timestamp', 'date', 'time'])]
    print(f"Time columns: {time_cols}")
    
    # Hourly revenue columns
    hourly_cols = [col for col in df.columns if 'hourly' in col.lower()]
    print(f"Hourly columns: {len(hourly_cols)} found")
    if hourly_cols:
        print(f"Sample hourly columns: {hourly_cols[:5]}")
    
    # Basic stats for revenue
    if revenue_cols:
        main_revenue = revenue_cols[0]  # Use first revenue column
        print(f"\n{'='*60}")
        print(f"REVENUE ANALYSIS - {main_revenue}")
        print("="*60)
        
        print(f"Total users: {len(df):,}")
        converters = df[df[main_revenue] > 0]
        print(f"Paying users: {len(converters):,} ({len(converters)/len(df)*100:.1f}%)")
        print(f"Non-paying users: {len(df) - len(converters):,} ({(len(df) - len(converters))/len(df)*100:.1f}%)")
        
        if len(converters) > 0:
            print(f"\nRevenue statistics:")
            print(f"  Total revenue: ${converters[main_revenue].sum():,.2f}")
            print(f"  Mean revenue per converter: ${converters[main_revenue].mean():.2f}")
            print(f"  Median revenue per converter: ${converters[main_revenue].median():.2f}")
            
            # Percentiles
            percentiles = [50, 75, 90, 95, 99]
            print(f"\nRevenue distribution:")
            for p in percentiles:
                val = np.percentile(converters[main_revenue], p)
                print(f"  {p}th percentile: ${val:.2f}")
    
    return df

if __name__ == "__main__":
    print("WhaleHunter - Simplified Data Analysis")
    print("="*60)
    
    df = load_data_simple()
    df = quick_analysis(df) 