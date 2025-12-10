import tomllib
from dataclasses import dataclass
from typing import List

# Config file containing RSS sources
FEEDS_CONFIG_FILE = "feeds.toml"


@dataclass
class FeedConfig:
    name: str
    url: str
    country: str  # Optional: helpful for grouping if needed later


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

