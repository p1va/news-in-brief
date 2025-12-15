import argparse
import datetime
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from core.config import load_show_config
from core.llm import OpenRouterLLM
from core.news import NewsRepository
from core.render import render_prompt_template
from core.rss import generate_rss_feed
from core.tts import TextToSpeech

load_dotenv()


def discover_shows() -> list[str]:
    """Discover all show directories in the repository."""
    shows = []
    for item in Path(".").iterdir():
        if item.is_dir() and (item / "show.toml").exists():
            shows.append(item.name)
    return shows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Podcast Daily Briefing Generator"
    )
    parser.add_argument(
        "--show",
        type=str,
        help="Show name (directory with show.toml). Use --all to generate for all shows.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate for all discovered shows"
    )
    parser.add_argument(
        "--fetch-only", action="store_true", help="Fetch news only, skip generation"
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force refresh from RSS feeds"
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Skip audio generation"
    )
    parser.add_argument(
        "--deep-dive", action="store_true", help="Include deep dive section"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=7,
        help="Max age of articles in days (default: 7)",
    )
    parser.add_argument(
        "--update-rss-only",
        action="store_true",
        help="Only regenerate RSS feed without creating new episode",
    )
    return parser.parse_args()


def generate_episode(show_name: str, args):
    """Generate a single episode for the specified show."""
    show_dir = Path(show_name)

    if not show_dir.exists():
        print(f"Error: Show directory '{show_name}' not found.")
        sys.exit(1)

    # Load show configuration
    try:
        config = load_show_config(show_dir)
    except Exception as e:
        print(f"Error loading show config: {e}")
        sys.exit(1)

    # If only updating RSS, skip episode generation
    if args.update_rss_only:
        print(f"\n--- {config.metadata.name}: Updating RSS Feed ---")
        try:
            generate_rss_feed(show_dir)
            print(f"RSS feed updated at {show_dir / 'rss.xml'}")
        except Exception as e:
            print(f"Error updating RSS feed: {e}")
            sys.exit(1)
        return

    # Set up paths
    issue_date = datetime.date.today().strftime("%Y-%m-%d")
    artifacts_folder = show_dir / "artifacts"
    issue_folder_path = artifacts_folder / issue_date
    prompts_folder = show_dir / "prompts"

    sources_path = issue_folder_path / f"{issue_date}-sources.parquet"
    script_path = issue_folder_path / f"{issue_date}-script.md"
    audio_path = issue_folder_path / f"{issue_date}-audio.mp3"

    # Ensure issue folder exists
    if not issue_folder_path.exists():
        print(f"Creating directory: {issue_folder_path}")
        issue_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {config.metadata.name}: Daily Briefing Generator [{issue_date}] ---")

    # 1. Acquire Data
    repo = NewsRepository()
    news_feed = repo.get_news(
        config.feeds,
        str(sources_path),
        force_refresh=args.force_refresh,
        max_age_days=args.max_age,
    )

    if not news_feed:
        print("No news data available. Exiting.")
        sys.exit(1)

    if args.fetch_only:
        print("Fetch complete. Exiting (--fetch-only).")
        return

    # 2. Generate Script
    if not script_path.exists():
        print("Generating script...")

        # Load previous episode's script for context
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        prev_script_path = artifacts_folder / yesterday_str / f"{yesterday_str}-script.md"
        previous_script = "No previous episode found."
        if prev_script_path.exists():
            with open(prev_script_path, "r") as f:
                previous_script = f.read()
            print(f"Loaded context from {yesterday_str}")

        try:
            # Get API key
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                print("Error: OPENROUTER_API_KEY not found in environment.")
                sys.exit(1)

            llm = OpenRouterLLM(
                model=config.llm.model,
                system_prompt=render_prompt_template(
                    "system_prompt.j2",
                    {
                        "current_date": issue_date,
                        "model_name": config.llm.model_friendly_name,
                        "host_name": config.metadata.host_name,
                        "today": datetime.date.today().strftime("%A %d"),
                        "include_deep_dive": args.deep_dive,
                        "yesterday_date": yesterday_str,
                    },
                    template_dir=str(prompts_folder),
                ),
                api_key=openrouter_api_key,
            )

            script_content = llm(
                render_prompt_template(
                    "user_message.j2",
                    {
                        "news_feed": news_feed,
                        "previous_script": previous_script,
                        "previous_date": yesterday_str,
                    },
                    template_dir=str(prompts_folder),
                )
            )

            with open(script_path, "w") as f:
                f.write(script_content)

            print(f"Script saved to {script_path}")

        except Exception as e:
            print(f"Error generating script: {e}")
            sys.exit(1)

    with open(script_path, "r") as f:
        script_content = f.read()

    # 3. Generate Audio
    if args.skip_audio:
        print("Skipping audio generation (--skip-audio).")
    elif audio_path.exists():
        print(f"Found existing audio at {audio_path}. Skipping TTS.")
    else:
        print("Generating audio...")

        try:
            tts = TextToSpeech()
            tts(script_content, str(audio_path))
        except Exception as e:
            print(f"Failed to generate audio: {e}")

    # 4. Update RSS Feed
    print("\nUpdating RSS feed...")
    try:
        generate_rss_feed(show_dir)
    except Exception as e:
        print(f"Warning: Failed to update RSS feed: {e}")

    print("\n--- Process Complete ---")
    print(f"Sources: {sources_path}")
    print(f"Script:  {script_path}")
    if not args.skip_audio:
        print(f"Audio:   {audio_path}")
    print(f"RSS:     {show_dir / 'rss.xml'}")


def main():
    args = parse_args()

    # Validate arguments
    if not args.all and not args.show:
        print("Error: Must specify either --show <name> or --all")
        print("\nAvailable shows:")
        shows = discover_shows()
        if shows:
            for show in shows:
                print(f"  - {show}")
        else:
            print("  No shows found. Create a directory with show.toml to get started.")
        sys.exit(1)

    # Generate episodes
    if args.all:
        shows = discover_shows()
        if not shows:
            print("No shows found.")
            sys.exit(1)
        print(f"Generating episodes for {len(shows)} show(s)...")
        for show in shows:
            generate_episode(show, args)
    else:
        generate_episode(args.show, args)


if __name__ == "__main__":
    main()
