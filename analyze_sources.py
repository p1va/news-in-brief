import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from dateutil import parser


def analyze_sources(parquet_file: str, show_samples: bool = False):
    """
    Analyze RSS sources from a parquet file.

    Args:
        parquet_file: Path to the parquet file to analyze
        show_samples: If True, show sample titles from each source
    """
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Error reading parquet file '{parquet_file}': {e}")
        return

    print(f"=== Analysis of {parquet_file} ===\n")
    print(f"Total articles: {len(df)}")
    print(f"Available columns: {list(df.columns)}\n")

    # Find date column
    date_col = None
    for col in ["published", "date", "pubDate", "published_parsed"]:
        if col in df.columns:
            date_col = col
            break

    if not date_col:
        print(f"Could not find a date column. Available columns: {df.columns}")
        return

    # Check for source column
    if "source" not in df.columns:
        print(f"Could not find 'source' column. Available columns: {df.columns}")
        return

    # Convert to datetime using dateutil.parser (more robust than pandas bulk parser)
    df[date_col] = df[date_col].apply(
        lambda x: parser.parse(x).astimezone(pytz.UTC) if pd.notna(x) else None
    )

    # Reference date: today (UTC)
    ref_date = pd.Timestamp.now(tz=pytz.UTC).normalize()
    ref_date_str = ref_date.strftime("%Y-%m-%d")

    # Calculate age in days
    df["age_days"] = (ref_date - df[date_col]).dt.days

    # Article Age Distribution
    print(f"Article Age Distribution (Days relative to {ref_date_str})")
    print("0 = Today, 1 = Yesterday, etc.\n")

    pivot = pd.crosstab(df["source"], df["age_days"])
    print(pivot)

    # Detailed Stats
    print("\n\nDetailed Stats by Source:")
    stats = df.groupby("source")["age_days"].describe()[["count", "min", "max", "mean"]]
    print(stats)

    # Articles by source
    print("\n\nArticle Count by Source:")
    print(df["source"].value_counts())

    # Show sample titles if requested
    if show_samples:
        print("\n\n=== Sample Titles by Source ===")
        for source in df["source"].unique():
            source_df = df[df["source"] == source].head(3)
            print(f"\n{source}:")
            for idx, row in source_df.iterrows():
                age = row["age_days"]
                title = row.get("title", "N/A")
                print(f"  [{age}d ago] {title}")

    # Check for issues
    print("\n\n=== Data Quality Checks ===")
    null_dates = df[df[date_col].isna()]
    if len(null_dates) > 0:
        print(f"⚠️  {len(null_dates)} articles with unparseable dates")

    if "title" in df.columns:
        null_titles = df[df["title"].isna()]
        if len(null_titles) > 0:
            print(f"⚠️  {len(null_titles)} articles with missing titles")

        # Check for HTML in titles
        html_titles = df[df["title"].str.contains("<", na=False)]
        if len(html_titles) > 0:
            print(f"⚠️  {len(html_titles)} articles with HTML tags in titles")
            print("Sample HTML titles:")
            for title in html_titles["title"].head(3):
                print(f"  {title[:100]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RSS source data from parquet files"
    )
    parser.add_argument("parquet_file", help="Path to parquet file to analyze")
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Show sample titles from each source",
    )
    args = parser.parse_args()

    if not Path(args.parquet_file).exists():
        print(f"Error: File '{args.parquet_file}' not found.")
        sys.exit(1)

    analyze_sources(args.parquet_file, show_samples=args.samples)


if __name__ == "__main__":
    main()
