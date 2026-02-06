#!/usr/bin/env python3
"""
Generate embeddings for news articles using OpenRouter or Voyage AI.

Usage:
    uv run python scripts/generate_embeddings.py <parquet_path> [--provider <openai|voyage>]
"""

import sys
import argparse
from pathlib import Path

# Allow running from scripts/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.embeddings import generate_embeddings_for_parquet

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for news articles.")
    parser.add_argument("parquet_path", type=Path, help="Path to input parquet file")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "voyage"], help="Embedding provider (default: openai)")
    
    args = parser.parse_args()

    if not args.parquet_path.exists():
        print(f"Error: File not found: {args.parquet_path}")
        sys.exit(1)

    try:
        output_path = generate_embeddings_for_parquet(args.parquet_path, provider_name=args.provider)
        print(f"Successfully generated embeddings: {output_path}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
