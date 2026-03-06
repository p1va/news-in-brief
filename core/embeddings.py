import os
import time
from pathlib import Path
from typing import List, Protocol

import pandas as pd
import requests
import voyageai

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-large"

# Voyage AI configuration
VOYAGE_EMBEDDING_MODEL = "voyage-4-large"

BATCH_SIZE = 50


class EmbeddingProvider(Protocol):
    def get_embeddings(self, texts: List[str]) -> List[List[float]]: ...


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
        result = self.client.embed(
            texts, model=VOYAGE_EMBEDDING_MODEL, input_type="document"
        )
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


def generate_embeddings_for_parquet(
    parquet_path: Path,
    provider_name: str = "openai",
    force_refresh: bool = False,
) -> Path:
    """
    Generate embeddings for articles in a parquet file.

    Default (incremental): reuse existing embeddings for articles that already
    have them (matched by link/URL), only call the API for new articles.

    force_refresh: delete existing embeddings and re-embed everything.

    Returns path to the file with embeddings.
    """
    # Unique suffix based on provider to avoid overwriting or mixing
    suffix = "-with-embeddings"
    if provider_name == "voyage":
        suffix = "-with-embeddings-voyage"

    output_path = parquet_path.with_stem(parquet_path.stem + suffix)

    if force_refresh and output_path.exists():
        print(f"Force refresh: deleting existing embeddings at {output_path}")
        output_path.unlink()

    print(f"Loading {parquet_path} for embedding generation using {provider_name}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} articles")

    # Try to reuse existing embeddings (incremental mode)
    new_df = df
    reused_df = pd.DataFrame()

    if not force_refresh and output_path.exists():
        existing_df = pd.read_parquet(output_path)
        if "embedding" in existing_df.columns and "link" in existing_df.columns:
            # Build a set of links that have embeddings
            existing_links = set(
                existing_df.loc[existing_df["link"].astype(str).str.len() > 0, "link"]
            )

            # Split current articles into already-embedded and new
            has_link = df["link"].astype(str).str.len() > 0
            already_embedded_mask = has_link & df["link"].isin(existing_links)

            if already_embedded_mask.any():
                # Reuse embeddings from existing file for matched articles
                matched_links = df.loc[already_embedded_mask, "link"]
                reused_df = existing_df[existing_df["link"].isin(matched_links)].copy()
                # Drop articles from existing that aren't in current sources
                # (they may have aged out)
                reused_df = reused_df[reused_df["link"].isin(df["link"])]
                new_df = df[~already_embedded_mask].reset_index(drop=True)

                print(
                    f"Reusing {len(reused_df)} existing embeddings, "
                    f"{len(new_df)} new articles to embed"
                )

    if len(new_df) == 0:
        # All articles already have embeddings, just save and return
        reused_df.to_parquet(output_path)
        print(
            f"No new articles to embed. Saved {len(reused_df)} articles to {output_path}"
        )
        return output_path

    # Create embedding text for new articles
    new_df = new_df.copy()
    new_df["embedding_text"] = new_df.apply(create_embedding_text, axis=1)

    provider = get_embedding_provider(provider_name)
    all_embeddings = []

    total_batches = (len(new_df) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Generating embeddings in {total_batches} batches...")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(new_df))
        batch_texts = new_df["embedding_text"].iloc[start_idx:end_idx].tolist()

        print(
            f"  Batch {batch_idx + 1}/{total_batches} ({len(batch_texts)} articles)...",
            end=" ",
            flush=True,
        )

        try:
            embeddings = provider.get_embeddings(batch_texts)
            all_embeddings.extend(embeddings)
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            raise

        if batch_idx < total_batches - 1:
            time.sleep(0.5)

    new_df["embedding"] = all_embeddings

    # Combine reused and newly embedded articles
    if len(reused_df) > 0:
        # Ensure both DataFrames have the same columns
        # new_df has embedding_text column, add it to reused if missing
        if "embedding_text" not in reused_df.columns:
            reused_df = reused_df.copy()
            reused_df["embedding_text"] = reused_df.apply(create_embedding_text, axis=1)
        combined_df = pd.concat([reused_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_parquet(output_path)
    print(f"Saved {len(combined_df)} articles with embeddings to {output_path}")

    return output_path
