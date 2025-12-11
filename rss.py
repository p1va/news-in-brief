import os
from datetime import datetime
from pathlib import Path
from feedgen.feed import FeedGenerator
import pytz
from mutagen.mp3 import MP3

ARTIFACTS_DIR = Path("artifacts")
# GitHub Pages URL for the repository
BASE_URL = "https://p1va.github.io/news-in-brief"
FEED_FILE = Path("rss.xml")

def generate_rss_feed():
    fg = FeedGenerator()
    fg.load_extension('podcast')
    
    fg.title('Asia in Brief')
    fg.description('Daily news summary from Asia.')
    fg.link(href=BASE_URL, rel='alternate')
    fg.language('en')
    
    # Iterate over artifact directories
    episodes = []
    if not ARTIFACTS_DIR.exists():
        print(f"Artifacts directory {ARTIFACTS_DIR} not found.")
        return

    for date_dir in ARTIFACTS_DIR.iterdir():
        if date_dir.is_dir():
            try:
                date_obj = datetime.strptime(date_dir.name, "%Y-%m-%d")
                episodes.append((date_obj, date_dir))
            except ValueError:
                continue # Skip non-date directories

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
        fe.id(f"{BASE_URL}/artifacts/{date_str}")
        fe.title(f"Asia in Brief - {date_str}")
        
        # Link to the audio file (Must be an absolute URL for RSS enclosures)
        audio_url = f"{BASE_URL}/artifacts/{date_str}/{audio_file.name}"
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
        
    fg.rss_str(pretty=True)
    fg.rss_file(str(FEED_FILE))
    print(f"RSS feed generated at {FEED_FILE}")

if __name__ == "__main__":
    generate_rss_feed()
