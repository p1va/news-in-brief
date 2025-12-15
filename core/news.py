import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import feedparser
import pandas as pd
from dateutil import parser as dateutil_parser

from core.config import FeedConfig


@dataclass
class NewsArticle:
    source: str
    date: str
    title: str


def validate_article_entry(entry, source_name: str) -> tuple[bool, str, datetime | None]:
    """
    Validate an RSS feed entry.

    Returns:
        (is_valid, error_message, parsed_datetime)
    """
    # Check for title
    if not hasattr(entry, "title") or not entry.title:
        return False, "Missing title", None

    # Check for published date
    if not hasattr(entry, "published") or not entry.published:
        return False, "Missing published date", None

    # Try to parse the date using dateutil.parser (same as analysis script)
    try:
        parsed_dt = dateutil_parser.parse(entry.published)
        # Ensure timezone-aware
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
        else:
            parsed_dt = parsed_dt.astimezone(timezone.utc)
        return True, "", parsed_dt
    except Exception as e:
        return False, f"Invalid date format: {e}", None


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

        # Track validation statistics
        total_validation_failures = 0
        validation_failures_by_reason = {}

        for feed_config in feeds:
            display_name = f"{feed_config.country} - {feed_config.name}"
            try:
                # Parse the feed
                feed = feedparser.parse(feed_config.url)
                if feed.status != 200:
                    raise ValueError(f"Status code {feed.status}")

                # 1. Validate and filter entries
                valid_entries = []
                validation_failures = 0

                for entry in feed.entries:
                    is_valid, error_msg, parsed_dt = validate_article_entry(entry, display_name)

                    if not is_valid:
                        validation_failures += 1
                        total_validation_failures += 1
                        validation_failures_by_reason[error_msg] = (
                            validation_failures_by_reason.get(error_msg, 0) + 1
                        )
                        continue

                    # Check if article is within age limit
                    if parsed_dt >= cutoff_date:
                        valid_entries.append((parsed_dt, entry))

                # 2. Sort by date descending (newest first)
                valid_entries.sort(key=lambda x: x[0], reverse=True)

                # 3. Take top N
                selected_entries = valid_entries[:limit]

                # Report statistics
                status = f"  ✓ {display_name}: {len(selected_entries)} articles (filtered from {len(feed.entries)})"
                if validation_failures > 0:
                    status += f" [{validation_failures} validation failures]"
                print(status)

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

        # Print validation summary
        if total_validation_failures > 0:
            print(f"\n⚠️  Validation Summary: {total_validation_failures} entries failed validation")
            print("Failure breakdown:")
            for reason, count in sorted(validation_failures_by_reason.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count}")

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
