import pandas as pd
from datetime import datetime

def analyze_sources():
    try:
        df = pd.read_parquet("artifacts/2025-12-11/2025-12-11-sources.parquet")
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Check columns
    # Assuming columns are like 'source', 'published', 'title', etc.
    # The user didn't show the schema, so I'll print columns if I fail, but usually 'published' is the date.
    
    # Based on common RSS parsing (which likely produced this), there's usually a date/published field.
    # Let's inspect the first few rows to be sure of the date column name if we were interactive, 
    # but here I'll try to guess 'published' or 'date'.
    
    date_col = None
    for col in ['published', 'date', 'pubDate', 'published_parsed']:
        if col in df.columns:
            date_col = col
            break
    
    if not date_col:
        print(f"Could not find a date column. Available columns: {df.columns}")
        return

    # Convert to datetime
    # The data might be strings or already datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    
    # Reference date: 2025-12-11 (UTC)
    ref_date = pd.Timestamp("2025-12-11", tz="UTC")
    
    # Calculate age in days (floor)
    df['age_days'] = (ref_date - df[date_col]).dt.days
    
    # Group by Source and Age
    # Adjust age buckets: 0 (Today), 1 (Yesterday), ...
    
    # We want to see the distribution.
    # Let's create a pivot table or just group and print.
    
    # Clean up source names if needed (assuming 'source' column exists)
    if 'source' not in df.columns:
        print(f"Could not find 'source' column. Available columns: {df.columns}")
        return

    # Filter for reasonable range for display (e.g., -1 to 7+ days)
    # Negative days means future dated (timezone issues?) or very fresh
    
    pivot = pd.crosstab(df['source'], df['age_days'])
    
    print("Article Age Distribution (Days relative to 2025-12-11)")
    print("0 = Today, 1 = Yesterday, etc.\n")
    print(pivot)
    
    print("\n\nDetailed Stats:")
    print(df.groupby('source')['age_days'].describe()[['count', 'min', 'max', 'mean']])

if __name__ == "__main__":
    analyze_sources()
