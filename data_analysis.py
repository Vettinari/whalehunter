import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pyarrow.parquet as pq
import pyarrow as pa
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load the parquet file and perform initial exploration"""
    print("Loading data with pyarrow and handling type conversions...")
    
    try:
        # Read the parquet file with pyarrow
        table = pq.read_table('raw_data.parquet')
        print("Parquet schema loaded successfully")
        print(f"Number of columns: {len(table.schema)}")
        print(f"Number of rows: {len(table)}")
        
        # Convert problematic types before pandas conversion
        schema = table.schema
        new_schema = []
        
        for field in schema:
            if 'dbdate' in str(field.type).lower():
                # Convert dbdate to string first, then we'll handle it in pandas
                new_field = pa.field(field.name, pa.string())
                new_schema.append(new_field)
                print(f"Converting {field.name} from {field.type} to string")
            else:
                new_schema.append(field)
        
        # Cast the table to the new schema if needed
        if any('dbdate' in str(field.type).lower() for field in schema):
            new_schema = pa.schema(new_schema)
            table = table.cast(new_schema)
        
        # Convert to pandas
        df = table.to_pandas()
        
        print(f"Data shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\n" + "="*50)
        print("COLUMN INFORMATION")
        print("="*50)
        
        # Display column info
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumn names and types:")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_vals = df[col].nunique()
            print(f"  {col:30} | {str(dtype):15} | Nulls: {null_count:8} ({null_pct:5.1f}%) | Unique: {unique_vals:8}")
        
        print("\n" + "="*50)
        print("DATA SAMPLE")
        print("="*50)
        print(df.head())
        
        print("\n" + "="*50)
        print("BASIC STATISTICS")
        print("="*50)
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
        
        if numeric_cols:
            print("\nNumeric columns statistics (first 10):")
            print(df[numeric_cols[:10]].describe())
        
        # Look for potential revenue/value columns
        value_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['revenue', 'value', 'amount', 'price', 'spend', 'payment', 'usd'])]
        
        if value_cols:
            print(f"\nPotential value/revenue columns: {value_cols}")
            for col in value_cols[:5]:  # Limit to first 5 for readability
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"\n{col} distribution:")
                    print(f"  Min: {df[col].min()}")
                    print(f"  Max: {df[col].max()}")
                    print(f"  Mean: {df[col].mean():.2f}")
                    print(f"  Median: {df[col].median():.2f}")
                    print(f"  Zero values: {(df[col] == 0).sum()} ({(df[col] == 0).mean()*100:.1f}%)")
        
        # Look for time columns
        time_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['time', 'date', 'hour', 'day', 'doi'])]
        
        if time_cols:
            print(f"\nPotential time columns: {time_cols}")
            for col in time_cols[:3]:  # Limit for readability
                print(f"\n{col} sample values:")
                print(df[col].value_counts().head())
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_conversion_patterns(df):
    """Analyze conversion patterns and revenue distribution"""
    print("\n" + "="*50)
    print("CONVERSION ANALYSIS")
    print("="*50)
    
    # Try to identify revenue column
    revenue_cols = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['revenue', 'value', 'amount', 'spend', 'usd'])]
    
    if not revenue_cols:
        print("No obvious revenue column found. Please specify manually.")
        return
    
    # Use the first revenue column found
    revenue_col = revenue_cols[0]
    print(f"Using '{revenue_col}' as revenue column")
    
    # Convert potential converters
    converters = df[df[revenue_col] > 0]
    non_converters = df[df[revenue_col] == 0]
    
    conversion_rate = len(converters) / len(df) * 100
    
    print(f"Total users: {len(df):,}")
    print(f"Converters: {len(converters):,} ({conversion_rate:.1f}%)")
    print(f"Non-converters: {len(non_converters):,} ({100-conversion_rate:.1f}%)")
    
    if len(converters) > 0:
        print(f"\nRevenue statistics for converters:")
        print(f"  Total revenue: ${converters[revenue_col].sum():,.2f}")
        print(f"  Average revenue per converter: ${converters[revenue_col].mean():.2f}")
        print(f"  Median revenue per converter: ${converters[revenue_col].median():.2f}")
        
        # Revenue distribution
        print(f"\nRevenue distribution percentiles:")
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(converters[revenue_col], p)
            print(f"  {p}th percentile: ${val:.2f}")
        
        # Single order analysis
        if 'order_count' in df.columns or any('order' in col.lower() for col in df.columns):
            order_cols = [col for col in df.columns if 'order' in col.lower()]
            if order_cols:
                order_col = order_cols[0]
                single_order_users = converters[converters[order_col] == 1]
                print(f"\nSingle order users: {len(single_order_users)} ({len(single_order_users)/len(converters)*100:.1f}% of converters)")
    
    return revenue_col

def analyze_behavioral_features(df):
    """Analyze behavioral features for the first 72 hours"""
    print("\n" + "="*50)
    print("BEHAVIORAL FEATURES ANALYSIS")
    print("="*50)
    
    # Look for hourly features
    hourly_cols = [col for col in df.columns if 'hour' in col.lower()]
    
    if hourly_cols:
        print(f"Found {len(hourly_cols)} hourly columns")
        print("Sample hourly columns:", hourly_cols[:5])
        
        # Analyze hours covered
        hour_numbers = []
        for col in hourly_cols:
            # Try to extract hour number from column name
            import re
            hour_match = re.search(r'(\d+)', col)
            if hour_match:
                hour_numbers.append(int(hour_match.group(1)))
        
        if hour_numbers:
            print(f"Hour range covered: {min(hour_numbers)} to {max(hour_numbers)} hours")
            print(f"Total hours tracked: {len(set(hour_numbers))}")
    
    # Look for cumulative features
    cumulative_cols = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['cumulative', 'cum', 'total'])]
    
    if cumulative_cols:
        print(f"\nFound {len(cumulative_cols)} cumulative columns")
        print("Sample cumulative columns:", cumulative_cols[:5])
    
    # Look for behavioral indicators
    behavior_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['click', 'view', 'session', 'page', 'event', 'action'])]
    
    if behavior_cols:
        print(f"\nFound {len(behavior_cols)} behavioral columns")
        print("Sample behavioral columns:", behavior_cols[:5])
    
    return hourly_cols, cumulative_cols, behavior_cols

if __name__ == "__main__":
    print("WhaleHunter Data Analysis")
    print("="*50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    if df is not None:
        # Analyze conversion patterns
        revenue_col = analyze_conversion_patterns(df)
        
        # Analyze behavioral features
        hourly_cols, cumulative_cols, behavior_cols = analyze_behavioral_features(df)
        
        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE")
        print("="*50)
        print("Data loaded successfully. Ready for detailed analysis.") 