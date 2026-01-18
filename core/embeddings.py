import os
import time
import requests
import pandas as pd
from pathlib import Path
from typing import List

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL = "openai/text-embedding-3-large"
BATCH_SIZE = 50

def get_api_key() -> str:
    """Get OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return key

def create_embedding_text(row: pd.Series) -> str:
    """
    Create the text to embed from title and description.
    """
    title = str(row["title"]).strip()
    description = str(row["description"]).strip()

    if description and description != "nan":
        return f"{title}. {description}"
    return title

def fetch_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    """
    Fetch embeddings from OpenRouter API.
    """
    response = requests.post(
        OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise ValueError(f"API error {response.status_code}: {response.text}")

    data = response.json()
    # Sort by index to ensure correct order
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]

def generate_embeddings_for_parquet(parquet_path: Path) -> Path:
    """
    Generate embeddings for all articles in a parquet file if not already present.
    Returns path to the file with embeddings.
    """
    output_path = parquet_path.with_stem(parquet_path.stem + "-with-embeddings")
    
    if output_path.exists():
        print(f"Embeddings already exist at {output_path}")
        return output_path

    print(f"Loading {parquet_path} for embedding generation...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} articles")

    # Create embedding text
    df["embedding_text"] = df.apply(create_embedding_text, axis=1)

    api_key = get_api_key()
    all_embeddings = []
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Generating embeddings in {total_batches} batches...")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_texts = df["embedding_text"].iloc[start_idx:end_idx].tolist()

        print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch_texts)} articles)...", end=" ", flush=True)

        try:
            embeddings = fetch_embeddings(batch_texts, api_key)
            all_embeddings.extend(embeddings)
            print(f"done")
        except Exception as e:
            print(f"ERROR: {e}")
            raise

        if batch_idx < total_batches - 1:
            time.sleep(0.5)

    df["embedding"] = all_embeddings
    df.to_parquet(output_path)
    print(f"Saved embeddings to {output_path}")
    
    return output_path
