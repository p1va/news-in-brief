import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from io import BytesIO
from typing import Dict, List, Optional

import feedparser
import pandas as pd
import requests
from dateutil import parser as dateutil_parser

from core.config import CleaningConfig, FeedConfig

# Browser-like headers required by some CDNs (e.g., fanpage.it blocks requests without Accept-Encoding)
_FEED_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0",
    "Accept-Encoding": "gzip, deflate",
}


@dataclass
class NewsArticle:
    source: str
    date: str
    title: str
    country: str = ""
    description: str = ""
    author: str = ""
    link: str = ""


def _clean_html(text: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    if not text:
        return ""
    # Unescape HTML entities (e.g., &amp; -> &)
    cleaned = unescape(text)
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Normalize whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _clean_author(author_str: str) -> str:
    """Clean author field with additional processing for common RSS formats."""
    if not author_str:
        return ""
    # First apply basic HTML cleaning
    cleaned = _clean_html(author_str)
    # Handle email format: "email@domain.com (Name)" -> "Name"
    email_match = re.match(r"^[^\s]+@[^\s]+\s+\((.+)\)$", cleaned)
    if email_match:
        cleaned = email_match.group(1)
    # Remove 'di ' prefix (Italian for "by")
    cleaned = re.sub(r"^di\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _is_google_news_url(url: str) -> bool:
    """Check if URL is a Google News RSS feed."""
    return "news.google.com" in url


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace/punctuation)."""
    return re.sub(r"[\s\-–—|]+", " ", text.lower()).strip()


def clean_article(
    title: str,
    description: str,
    feed_config: FeedConfig,
    cleaning_config: Optional[CleaningConfig] = None,
) -> tuple[str, str]:
    """
    Apply cleaning rules to article title and description.

    Returns:
        (cleaned_title, cleaned_description)
    """
    if cleaning_config is None:
        cleaning_config = CleaningConfig()

    cleaned_title = title
    cleaned_desc = description

    # 1. Auto-clean Google News artifacts
    if cleaning_config.google_news_auto_clean and _is_google_news_url(feed_config.url):
        # Google News appends " - Source Name" to titles
        suffix_pattern = rf"\s*-\s*{re.escape(feed_config.name)}$"
        cleaned_title = re.sub(suffix_pattern, "", cleaned_title)
        # Description often has same pattern but without hyphen
        desc_suffix_pattern = rf"\s+{re.escape(feed_config.name)}$"
        cleaned_desc = re.sub(desc_suffix_pattern, "", cleaned_desc)

    # 2. Apply per-feed cleaning patterns
    if feed_config.cleaning:
        fc = feed_config.cleaning
        if fc.title_prefix:
            cleaned_title = re.sub(f"^{fc.title_prefix}", "", cleaned_title)
        if fc.title_suffix:
            cleaned_title = re.sub(f"{fc.title_suffix}$", "", cleaned_title)
        if fc.description_prefix:
            cleaned_desc = re.sub(f"^{fc.description_prefix}", "", cleaned_desc)
        if fc.description_suffix:
            cleaned_desc = re.sub(f"{fc.description_suffix}$", "", cleaned_desc)

    # 3. Strip whitespace after pattern removal
    cleaned_title = cleaned_title.strip()
    cleaned_desc = cleaned_desc.strip()

    # 4. Check for empty/useless description values
    if cleaned_desc in cleaning_config.empty_values:
        cleaned_desc = ""

    # 5. Check minimum description length
    if len(cleaned_desc) < cleaning_config.min_description_length:
        cleaned_desc = ""

    # 6. Dedupe: if description ≈ title, drop description
    if cleaning_config.dedupe_title_description and cleaned_desc:
        norm_title = _normalize_for_comparison(cleaned_title)
        norm_desc = _normalize_for_comparison(cleaned_desc)
        if norm_title == norm_desc:
            cleaned_desc = ""

    return cleaned_title, cleaned_desc


def validate_article_entry(
    entry, source_name: str
) -> tuple[bool, str, datetime | None]:
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
        cleaning_config: Optional[CleaningConfig] = None,
    ) -> Dict[str, List[NewsArticle]]:
        """
        Main entry point. Always fetches fresh RSS and merges with existing data.

        Default (incremental): fetch fresh RSS, merge with existing parquet,
        deduplicate by link (URL), apply max_age filter, save and return.

        force_refresh: fetch fresh RSS, ignore existing parquet (clean slate).
        """
        print("Fetching fresh news from RSS feeds...")
        fresh_articles = self._fetch_from_feeds(
            feeds, max_age_days=max_age_days, cleaning_config=cleaning_config
        )

        if not force_refresh and os.path.exists(storage_path):
            print(f"Merging with existing data at {storage_path}...")
            existing_articles = self._load_articles_from_parquet(storage_path)
            articles = self._merge_articles(
                existing=existing_articles,
                fresh=fresh_articles,
                max_age_days=max_age_days,
            )
        else:
            if force_refresh:
                print("Force refresh: ignoring existing data.")
            articles = fresh_articles

        self._save_to_parquet(articles, storage_path)

        # Return in the grouped format expected by the application
        return self._group_by_source(articles)

    def _fetch_from_feeds(
        self,
        feeds: List[FeedConfig],
        limit: int = 150,
        max_age_days: int = 7,
        cleaning_config: Optional[CleaningConfig] = None,
    ) -> List[NewsArticle]:
        articles = []
        print(
            f"Fetching news from {len(feeds)} sources (limit: {limit}, max_age: {max_age_days}d)..."
        )

        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=max_age_days)

        # Track validation statistics
        total_validation_failures = 0
        validation_failures_by_reason = {}

        for feed_config in feeds:
            display_name = f"{feed_config.country} - {feed_config.name}"  # For logging
            try:
                resp = requests.get(
                    feed_config.url, timeout=10.0, headers=_FEED_REQUEST_HEADERS
                )

                if resp.status_code != 200:
                    raise ValueError(f"Status code {resp.status_code}")

                # Put it to memory stream object universal feedparser
                content = BytesIO(resp.content)

                # Parse content
                feed = feedparser.parse(content)

                # 1. Validate and filter entries
                valid_entries = []
                validation_failures = 0

                for entry in feed.entries:
                    is_valid, error_msg, parsed_dt = validate_article_entry(
                        entry, display_name
                    )

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
                    # Extract and clean description (try both 'description' and 'summary')
                    raw_description = (
                        getattr(entry, "description", "")
                        or getattr(entry, "summary", "")
                        or ""
                    )
                    html_cleaned_description = _clean_html(raw_description)
                    html_cleaned_title = _clean_html(str(entry.title))

                    # Apply cleaning rules (Google News, per-feed patterns, etc.)
                    cleaned_title, cleaned_description = clean_article(
                        html_cleaned_title,
                        html_cleaned_description,
                        feed_config,
                        cleaning_config,
                    )

                    # Extract and clean author
                    raw_author = getattr(entry, "author", "") or ""
                    cleaned_author = _clean_author(raw_author)

                    # Extract link
                    article_link = getattr(entry, "link", "") or ""

                    articles.append(
                        NewsArticle(
                            source=feed_config.name,
                            date=str(entry.published),
                            title=cleaned_title,
                            country=feed_config.country,
                            description=cleaned_description,
                            author=cleaned_author,
                            link=article_link,
                        )
                    )

                # Be nice to the servers
                time.sleep(1)

            except Exception as e:
                print(f"  ✗ {display_name}: {e}")

        # Print validation summary
        if total_validation_failures > 0:
            print(
                f"\n⚠️  Validation Summary: {total_validation_failures} entries failed validation"
            )
            print("Failure breakdown:")
            for reason, count in sorted(
                validation_failures_by_reason.items(), key=lambda x: -x[1]
            ):
                print(f"  - {reason}: {count}")

        # Deduplicate articles by (source, title) - keeps first occurrence
        seen = set()
        unique_articles = []
        duplicates_removed = 0
        for article in articles:
            key = (article.source, article.title)
            if key not in seen:
                seen.add(key)
                unique_articles.append(article)
            else:
                duplicates_removed += 1

        if duplicates_removed > 0:
            print(
                f"\n🧹 Removed {duplicates_removed} duplicate articles (same source + title)"
            )

        return unique_articles

    def load_from_parquet(self, storage_path: str) -> Dict[str, List[NewsArticle]]:
        """Load articles from parquet without fetching. Returns grouped by source."""
        return self._load_from_parquet(storage_path)

    def _load_articles_from_parquet(self, path: str) -> List[NewsArticle]:
        """Load articles from parquet as a flat list of NewsArticle."""
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return []

        articles = []
        has_description = "description" in df.columns
        has_author = "author" in df.columns
        has_country = "country" in df.columns
        has_link = "link" in df.columns

        for _, row in df.iterrows():
            articles.append(
                NewsArticle(
                    source=str(row["source"]),
                    date=str(row["date"]),
                    title=str(row["title"]),
                    country=str(row["country"]) if has_country else "",
                    description=str(row["description"]) if has_description else "",
                    author=str(row["author"]) if has_author else "",
                    link=str(row["link"]) if has_link else "",
                )
            )
        return articles

    def _merge_articles(
        self,
        existing: List[NewsArticle],
        fresh: List[NewsArticle],
        max_age_days: int = 7,
    ) -> List[NewsArticle]:
        """
        Merge existing and fresh articles, deduplicating by link (URL).
        Fresh articles take precedence over existing ones with the same link.
        Articles without links fall back to (source, title) dedup.
        Drops articles older than max_age_days.
        """
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=max_age_days)

        # Build lookup from fresh articles (they take precedence)
        seen_links: dict[str, NewsArticle] = {}
        seen_source_title: dict[tuple[str, str], NewsArticle] = {}
        merged: List[NewsArticle] = []

        # Add fresh articles first (they take precedence)
        for article in fresh:
            if article.link:
                seen_links[article.link] = article
            else:
                seen_source_title[(article.source, article.title)] = article
            merged.append(article)

        # Add existing articles that aren't duplicates
        added_from_existing = 0
        for article in existing:
            # Skip if we already have this article from fresh fetch
            if article.link and article.link in seen_links:
                continue
            if not article.link and (article.source, article.title) in seen_source_title:
                continue

            # Apply age filter on existing articles
            try:
                parsed_dt = dateutil_parser.parse(article.date)
                if parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                else:
                    parsed_dt = parsed_dt.astimezone(timezone.utc)
                if parsed_dt < cutoff_date:
                    continue
            except Exception:
                pass  # Keep articles with unparseable dates

            # Track dedup keys
            if article.link:
                seen_links[article.link] = article
            else:
                seen_source_title[(article.source, article.title)] = article

            merged.append(article)
            added_from_existing += 1

        print(
            f"Merged: {len(fresh)} fresh + {added_from_existing} existing "
            f"= {len(merged)} total articles"
        )
        return merged

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
            # Check for new columns (backward compatibility with older parquet files)
            has_description = "description" in df.columns
            has_author = "author" in df.columns
            has_country = "country" in df.columns
            has_link = "link" in df.columns

            for source, group in df.groupby("source"):
                entries = [
                    NewsArticle(
                        date=str(row["date"]),
                        title=str(row["title"]),
                        source=str(source),
                        country=str(row["country"]) if has_country else "",
                        description=str(row["description"]) if has_description else "",
                        author=str(row["author"]) if has_author else "",
                        link=str(row["link"]) if has_link else "",
                    )
                    for _, row in group.iterrows()
                ]
                grouped[str(source)] = entries
        return grouped
