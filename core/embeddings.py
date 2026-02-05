import os
import time
import requests
import pandas as pd
from pathlib import Path
from typing import List, Protocol
import voyageai

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-large"

# Voyage AI configuration
VOYAGE_EMBEDDING_MODEL = "voyage-4-large"

BATCH_SIZE = 50

class EmbeddingProvider(Protocol):
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        ...

class OpenRouterEmbeddingProvider:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_EMBEDDING_MODEL,
                "input": texts,
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise ValueError(f"API error {response.status_code}: {response.text}")

        data = response.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

class VoyageEmbeddingProvider:
    def __init__(self):
        # voyageai.Client() automatically uses VOYAGE_API_KEY env var
        if not os.environ.get("VOYAGE_API_KEY"):
             raise ValueError("VOYAGE_API_KEY environment variable not set")
        self.client = voyageai.Client()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # model parameter is optional if we wanted to use default, but we'll specify it
        result = self.client.embed(texts, model=VOYAGE_EMBEDDING_MODEL, input_type="document")
        return result.embeddings

def get_embedding_provider(provider_name: str = "openai") -> EmbeddingProvider:
    if provider_name.lower() == "voyage":
        return VoyageEmbeddingProvider()
    elif provider_name.lower() == "openai":
        return OpenRouterEmbeddingProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

def create_embedding_text(row: pd.Series) -> str:
    """
    Create the text to embed from title and description.
    """
    title = str(row["title"]).strip()
    description = str(row["description"]).strip()

    if description and description != "nan":
        return f"{title}. {description}"
    return title

def generate_embeddings_for_parquet(parquet_path: Path, provider_name: str = "openai") -> Path:
    """
    Generate embeddings for all articles in a parquet file if not already present.
    Returns path to the file with embeddings.
    """
    # Unique suffix based on provider to avoid overwriting or mixing
    suffix = "-with-embeddings"
    if provider_name == "voyage":
        suffix = "-with-embeddings-voyage"
        
    output_path = parquet_path.with_stem(parquet_path.stem + suffix)
    
    if output_path.exists():
        print(f"Embeddings already exist at {output_path}")
        return output_path

    print(f"Loading {parquet_path} for embedding generation using {provider_name}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} articles")

    # Create embedding text
    df["embedding_text"] = df.apply(create_embedding_text, axis=1)

    provider = get_embedding_provider(provider_name)
    all_embeddings = []
    
    # Adjust batch size for Voyage if needed, but 50 is generally safe
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Generating embeddings in {total_batches} batches...")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_texts = df["embedding_text"].iloc[start_idx:end_idx].tolist()

        print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch_texts)} articles)...", end=" ", flush=True)

        try:
            embeddings = provider.get_embeddings(batch_texts)
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
