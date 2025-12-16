import datetime
import time
from pathlib import Path

import feedparser
from jinja2 import Environment, FileSystemLoader


def generate_html(show_dir: Path):
    """
    Generate static index.html for a show using the rss.xml content.
    """
    rss_file = show_dir / "rss.xml"
    if not rss_file.exists():
        print(f"Warning: RSS file not found at {rss_file}. Skipping HTML generation.")
        return

    # Parse RSS
    # feedparser can read from file path or URL
    feed = feedparser.parse(str(rss_file))
    
    if feed.bozo:
        print(f"Warning: Error parsing RSS feed: {feed.bozo_exception}")

    # Prepare data
    episodes = []
    for entry in feed.entries:
        audio_url = None
        # Find enclosure
        if hasattr(entry, 'enclosures'):
             for enclosure in entry.enclosures:
                 if enclosure.get('type', '').startswith('audio/'):
                     audio_url = enclosure.get('href')
                     break
        # Fallback to links if enclosures not populated directly
        if not audio_url and hasattr(entry, 'links'):
            for link in entry.links:
                if link.rel == 'enclosure':
                    audio_url = link.href
                    break
        
        # Format Date
        pub_date = entry.get('published', '')
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                # struct_time to datetime
                dt = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed))
                pub_date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass # Fallback to original string

        episodes.append({
            'title': entry.get('title', 'Untitled'),
            'published': pub_date,
            'description': entry.get('description', ''),
            'audio_url': audio_url
        })
    
    # Sort episodes by published date, newest first
    episodes.sort(key=lambda x: x['published'], reverse=True)

    # Setup Jinja2
    # Assuming 'templates' is in the root of the repo.
    # We need to find the repo root relative to this file.
    # this file is in core/site.py. Repo root is ../../
    repo_root = Path(__file__).parent.parent
    templates_dir = repo_root / "templates"
    
    if not templates_dir.exists():
         print(f"Error: Templates directory not found at {templates_dir}")
         return

    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    try:
        template = env.get_template("show_index.html.j2")
    except Exception as e:
        print(f"Error loading template: {e}")
        return
    
    # Check for image
    image_url = None
    if hasattr(feed.feed, 'image') and hasattr(feed.feed.image, 'href'):
        image_url = feed.feed.image.href
    elif hasattr(feed.feed, 'itunes_image'):
        image_url = feed.feed.itunes_image

    try:
        html_content = template.render(
            show_title=feed.feed.get('title', 'Podcast'),
            show_description=feed.feed.get('description', ''),
            show_image=image_url,
            rss_url="rss.xml",
            episodes=episodes
        )
        
        output_file = show_dir / "index.html"
        with open(output_file, "w") as f:
            f.write(html_content)
        print(f"Generated HTML at {output_file}")
    except Exception as e:
        print(f"Error rendering HTML: {e}")
