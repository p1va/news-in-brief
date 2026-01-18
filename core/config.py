import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Config file containing RSS sources
FEEDS_CONFIG_FILE = "feeds.toml"


@dataclass
class FeedCleaningConfig:
    """Per-feed cleaning overrides."""

    title_prefix: Optional[str] = None  # Regex pattern to strip from title start
    title_suffix: Optional[str] = None  # Regex pattern to strip from title end
    description_prefix: Optional[str] = None  # Regex pattern to strip from desc start
    description_suffix: Optional[str] = None  # Regex pattern to strip from desc end


@dataclass
class CleaningConfig:
    """Global cleaning configuration."""

    # Values to treat as empty/missing
    empty_values: List[str] = field(
        default_factory=lambda: ["undefined", "Continua a leggere", "Read more", ""]
    )
    # Minimum description length (shorter = treat as empty)
    min_description_length: int = 15
    # If description is (nearly) identical to title, drop it
    dedupe_title_description: bool = True
    # Auto-clean Google News feed artifacts (strip " - {feed.name}" suffix)
    google_news_auto_clean: bool = True
    # Junk topic prototypes to filter out (semantic similarity matching)
    # Articles semantically similar to these phrases get flagged as junk
    junk_topics: List[str] = field(default_factory=list)
    # Similarity threshold for junk detection (0.0-1.0, higher = stricter)
    junk_threshold: float = 0.35
    # HAC clustering distance threshold (0.0-1.0)
    # Lower = tighter clusters (articles must be very similar)
    # Higher = looser clusters (more articles grouped together)
    # 0.25 = ~75% similarity required, 0.40 = ~60% similarity required
    cluster_threshold: float = 0.40


@dataclass
class FeedConfig:
    name: str
    url: str
    country: str  # Optional: helpful for grouping if needed later
    cleaning: Optional[FeedCleaningConfig] = None


@dataclass
class ShowMetadata:
    name: str
    description: str
    author: str
    email: str
    language: str
    category: str
    host_name: str
    cover_image: str


@dataclass
class LLMConfig:
    model: str
    model_friendly_name: str


@dataclass
class TTSConfig:
    voice_id: str
    voice_name: str
    model: str


@dataclass
class RSSConfig:
    base_url: str


@dataclass
class ShowConfig:
    metadata: ShowMetadata
    llm: LLMConfig
    tts: TTSConfig
    rss: RSSConfig
    feeds: List[FeedConfig]
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)


def load_feeds(config_file: str = FEEDS_CONFIG_FILE) -> List[FeedConfig]:
    """Loads feed configurations from a TOML file."""
    try:
        with open(config_file, "rb") as f:
            data = tomllib.load(f)
            feeds = []
            for feed_data in data.get("feeds", []):
                feeds.append(
                    FeedConfig(
                        name=feed_data["name"],
                        url=feed_data["url"],
                        country=feed_data["country"],
                    )
                )
            return feeds
    except FileNotFoundError:
        print(f"Warning: Config file '{config_file}' not found.")
        return []
    except Exception as e:
        print(f"Error loading config file: {e}")
        return []


def load_show_config(show_dir: Path) -> ShowConfig:
    """Loads show configuration from show.toml in the show directory."""
    config_file = show_dir / "show.toml"

    try:
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

            # Parse metadata
            metadata = ShowMetadata(**data["metadata"])

            # Parse LLM config
            llm = LLMConfig(**data["llm"])

            # Parse TTS config
            tts = TTSConfig(**data["tts"])

            # Parse RSS config
            rss = RSSConfig(**data["rss"])

            # Parse global cleaning config (with defaults)
            cleaning_data = data.get("cleaning", {})
            cleaning = CleaningConfig(
                empty_values=cleaning_data.get(
                    "empty_values",
                    ["undefined", "Continua a leggere", "Read more", ""],
                ),
                min_description_length=cleaning_data.get("min_description_length", 15),
                dedupe_title_description=cleaning_data.get(
                    "dedupe_title_description", True
                ),
                google_news_auto_clean=cleaning_data.get("google_news_auto_clean", True),
                junk_topics=cleaning_data.get("junk_topics", []),
                junk_threshold=cleaning_data.get("junk_threshold", 0.35),
                cluster_threshold=cleaning_data.get("cluster_threshold", 0.40),
            )

            # Parse feeds (with optional per-feed cleaning)
            feeds = []
            for feed_data in data.get("feeds", []):
                # Parse per-feed cleaning config if present
                feed_cleaning = None
                if "cleaning" in feed_data:
                    fc = feed_data["cleaning"]
                    feed_cleaning = FeedCleaningConfig(
                        title_prefix=fc.get("title_prefix"),
                        title_suffix=fc.get("title_suffix"),
                        description_prefix=fc.get("description_prefix"),
                        description_suffix=fc.get("description_suffix"),
                    )

                feeds.append(
                    FeedConfig(
                        name=feed_data["name"],
                        url=feed_data["url"],
                        country=feed_data["country"],
                        cleaning=feed_cleaning,
                    )
                )

            return ShowConfig(
                metadata=metadata,
                llm=llm,
                tts=tts,
                rss=rss,
                feeds=feeds,
                cleaning=cleaning,
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"Show config file not found: {config_file}")
    except Exception as e:
        raise Exception(f"Error loading show config: {e}")

