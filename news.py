import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import feedparser
import pandas as pd

from config import FeedConfig


@dataclass
class NewsArticle:
    source: str
    date: str
    title: str


class NewsRepository:
    """
    Handles the retrieval and storage of news articles.
    Acts as a facade over the RSS fetching and Parquet storage.
    """

    def get_news(
        self,
        feeds: List[FeedConfig],
        storage_path: str,
        force_refresh: bool = False,
        max_age_days: int = 7,
    ) -> Dict[str, List[NewsArticle]]:
        """
        Main entry point. Tries to load from disk first.
        If missing or forced, fetches from RSS, saves to disk, then returns.
        """
        if not force_refresh and os.path.exists(storage_path):
            print(f"Found existing data at {storage_path}. Loading...")
            return self._load_from_parquet(storage_path)

        print("Fetching fresh news from RSS feeds...")
        articles = self._fetch_from_feeds(feeds, max_age_days=max_age_days)
        self._save_to_parquet(articles, storage_path)

        # Return in the grouped format expected by the application
        return self._group_by_source(articles)

    def _fetch_from_feeds(
        self, feeds: List[FeedConfig], limit: int = 50, max_age_days: int = 7
    ) -> List[NewsArticle]:
        articles = []
        print(f"Fetching news from {len(feeds)} sources (limit: {limit}, max_age: {max_age_days}d)...")

        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=max_age_days)

        for feed_config in feeds:
            display_name = f"{feed_config.country} - {feed_config.name}"
            try:
                # Parse the feed
                feed = feedparser.parse(feed_config.url)
                if feed.status != 200:
                    raise ValueError(f"Status code {feed.status}")

                # 1. Parse dates and filter
                valid_entries = []
                for entry in feed.entries:
                    try:
                        # Use pandas for robust parsing (handles most RSS formats)
                        published_dt = pd.to_datetime(entry.published, utc=True)
                        if published_dt >= cutoff_date:
                            valid_entries.append((published_dt, entry))
                    except Exception:
                        # If parsing fails or no date, skip to ensure quality
                        continue

                # 2. Sort by date descending (newest first)
                valid_entries.sort(key=lambda x: x[0], reverse=True)

                # 3. Take top N
                selected_entries = valid_entries[:limit]
                print(
                    f"  ✓ {display_name}: {len(selected_entries)} articles (filtered from {len(feed.entries)})"
                )

                for _, entry in selected_entries:
                    articles.append(
                        NewsArticle(
                            source=display_name,
                            date=str(entry.published),
                            title=str(entry.title),
                        )
                    )

                # Be nice to the servers
                time.sleep(1)

            except Exception as e:
                print(f"  ✗ {display_name}: {e}")

        return articles
    def _save_to_parquet(self, articles: List[NewsArticle], path: str) -> None:
        if not articles:
            print("No data to save.")
            return

        # Convert dataclasses to dicts for DataFrame
        data = [asdict(a) for a in articles]
        df = pd.DataFrame(data)
        df.to_parquet(path)
        print(f"Saved {len(df)} articles to {path}")

    def _load_from_parquet(self, path: str) -> Dict[str, List[NewsArticle]]:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return {}

        return self._group_by_source_df(df)

    def _group_by_source(
        self, articles: List[NewsArticle]
    ) -> Dict[str, List[NewsArticle]]:
        """Groups a flat list of articles by source."""
        grouped = {}
        for article in articles:
            if article.source not in grouped:
                grouped[article.source] = []
            grouped[article.source].append(article)
        return grouped

    def _group_by_source_df(self, df: pd.DataFrame) -> Dict[str, List[NewsArticle]]:
        """Converts DataFrame back to grouped dataclasses."""
        grouped = {}
        if not df.empty:
            for source, group in df.groupby("source"):
                entries = [
                    NewsArticle(
                        date=str(row["date"]),
                        title=str(row["title"]),
                        source=str(source),
                    )
                    for _, row in group.iterrows()
                ]
                grouped[str(source)] = entries
        return grouped
