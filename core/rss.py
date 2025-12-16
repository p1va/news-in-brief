import os
from datetime import datetime
from pathlib import Path
from xml.dom import minidom

from feedgen.feed import FeedGenerator
import pytz
from mutagen.mp3 import MP3

from core.config import load_show_config


def generate_rss_feed(show_dir: Path):
    """
    Generate RSS feed for a specific show.

    Args:
        show_dir: Path to the show directory (e.g., Path("asia-in-brief"))
    """
    # Load show configuration
    config = load_show_config(show_dir)

    artifacts_dir = show_dir / "artifacts"
    base_url = config.rss.base_url
    feed_file = show_dir / "rss.xml"

    fg = FeedGenerator()
    fg.load_extension('podcast')

    fg.title(config.metadata.name)
    fg.description(config.metadata.description)
    fg.link(href=base_url, rel='alternate')
    fg.language(config.metadata.language)
    fg.author({'name': config.metadata.author, 'email': config.metadata.email})

    # Add podcast cover image
    cover_image_url = f"{base_url}/{config.metadata.cover_image}"
    fg.podcast.itunes_image(cover_image_url)
    fg.image(url=cover_image_url, title=config.metadata.name, link=base_url)

    # Iterate over artifact directories
    episodes = []
    if not artifacts_dir.exists():
        print(f"Artifacts directory {artifacts_dir} not found.")
        return

    for date_dir in artifacts_dir.iterdir():
        if date_dir.is_dir():
            try:
                date_obj = datetime.strptime(date_dir.name, "%Y-%m-%d")
                episodes.append((date_obj, date_dir))
            except ValueError:
                continue  # Skip non-date directories

    # Sort episodes by date, newest first
    episodes.sort(key=lambda x: x[0], reverse=True)

    for date_obj, date_dir in episodes:
        date_str = date_dir.name
        audio_file = date_dir / f"{date_str}-audio.mp3"
        script_file = date_dir / f"{date_str}-script.md"

        if not audio_file.exists():
            continue

        fe = fg.add_entry()
        # Unique ID for the episode
        fe.id(f"{base_url}/artifacts/{date_str}")
        fe.title(f"{config.metadata.name} - {date_str}")

        # Link to the audio file (Must be an absolute URL for RSS enclosures)
        audio_url = f"{base_url}/artifacts/{date_str}/{audio_file.name}"
        fe.link(href=audio_url)

        # Enclosure
        file_size = os.path.getsize(audio_file)
        fe.enclosure(audio_url, str(file_size), 'audio/mpeg')

        # PubDate
        # Assume published at 8 AM UTC on that day
        pub_date = date_obj.replace(hour=8, tzinfo=pytz.UTC)
        fe.pubDate(pub_date)

        # Description from script
        description = f"Episode for {date_str}"
        if script_file.exists():
            try:
                with open(script_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    description = content
            except Exception as e:
                print(f"Error reading script {script_file}: {e}")

        fe.description(description)

        # iTunes tags
        try:
            audio = MP3(audio_file)
            duration = int(audio.info.length)
            fe.podcast.itunes_duration(duration)
        except Exception as e:
            print(f"Error reading audio duration for {audio_file}: {e}")

    # Generate RSS XML and pretty-print it for better git diffs
    rss_string = fg.rss_str(pretty=True)

    # Use minidom to properly format the XML with nice indentation
    dom = minidom.parseString(rss_string)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="utf-8")

    # Remove extra blank lines that minidom adds
    lines = pretty_xml.decode("utf-8").split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    formatted_xml = "\n".join(non_empty_lines) + "\n"

    # Write the formatted XML to file
    with open(feed_file, "w", encoding="utf-8") as f:
        f.write(formatted_xml)

    print(f"RSS feed generated at {feed_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.rss <show-directory>")
        print("Example: python -m core.rss asia-in-brief")
        sys.exit(1)

    show_dir = Path(sys.argv[1])
    if not show_dir.exists():
        print(f"Error: Show directory '{show_dir}' not found.")
        sys.exit(1)

    generate_rss_feed(show_dir)
