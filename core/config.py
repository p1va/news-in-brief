import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Config file containing RSS sources
FEEDS_CONFIG_FILE = "feeds.toml"


@dataclass
class FeedConfig:
    name: str
    url: str
    country: str  # Optional: helpful for grouping if needed later


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

            # Parse feeds
            feeds = []
            for feed_data in data.get("feeds", []):
                feeds.append(
                    FeedConfig(
                        name=feed_data["name"],
                        url=feed_data["url"],
                        country=feed_data["country"],
                    )
                )

            return ShowConfig(
                metadata=metadata,
                llm=llm,
                tts=tts,
                rss=rss,
                feeds=feeds,
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"Show config file not found: {config_file}")
    except Exception as e:
        raise Exception(f"Error loading show config: {e}")

