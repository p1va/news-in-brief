import argparse
import datetime
import os
import sys

from dotenv import load_dotenv

from config import load_feeds
from llm import OpenRouterLLM
from news import NewsRepository
from render import render_prompt_template
from tts import TextToSpeech

load_dotenv()

# --- Configuration ---
ARTIFACTS_FOLDER = "artifacts"
ISSUE_DATE = datetime.date.today().strftime("%Y-%m-%d")
ISSUE_FOLDER_PATH = os.path.join(ARTIFACTS_FOLDER, ISSUE_DATE)

# Files
SOURCES_FILENAME = f"{ISSUE_DATE}-sources.parquet"
SOURCES_PATH = os.path.join(ISSUE_FOLDER_PATH, SOURCES_FILENAME)

SCRIPT_FILENAME = f"{ISSUE_DATE}-script.md"
SCRIPT_PATH = os.path.join(ISSUE_FOLDER_PATH, SCRIPT_FILENAME)

AUDIO_FILENAME = f"{ISSUE_DATE}-audio.mp3"
AUDIO_PATH = os.path.join(ISSUE_FOLDER_PATH, AUDIO_FILENAME)

# Models
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-v3.2-speciale"
MODEL_FRIENDLY_NAME = "DeepSeek-V3.2-Special"
HOST_NAME = "Amelia"


def ensure_issue_folder():
    if not os.path.exists(ISSUE_FOLDER_PATH):
        print(f"Creating directory: {ISSUE_FOLDER_PATH}")
        os.makedirs(ISSUE_FOLDER_PATH, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Asia in Brief: Daily Briefing Generator"
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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"--- Asia in Brief: Daily Briefing Generator [{ISSUE_DATE}] ---")
    ensure_issue_folder()

    # 1. Acquire Data
    feeds = load_feeds()
    repo = NewsRepository()
    news_feed = repo.get_news(
        feeds,
        SOURCES_PATH,
        force_refresh=args.force_refresh,
        max_age_days=args.max_age,
    )

    if not news_feed:
        print("No news data available. Exiting.")
        sys.exit(1)

    if args.fetch_only:
        print("Fetch complete. Exiting (--fetch-only).")
        return

    # Generate Script
    if os.path.exists(SCRIPT_PATH) is False:
        print("Generating script...")

        # Load previous episode's script for context
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        prev_script_path = os.path.join(
            ARTIFACTS_FOLDER, yesterday_str, f"{yesterday_str}-script.md"
        )
        previous_script = "No previous episode found."
        if os.path.exists(prev_script_path):
            with open(prev_script_path, "r") as f:
                previous_script = f.read()
            print(f"Loaded context from {yesterday_str}")

        try:
            llm = OpenRouterLLM(
                model=MODEL_NAME,
                system_prompt=render_prompt_template(
                    "system_prompt.j2",
                    {
                        "current_date": ISSUE_DATE,
                        "model_name": MODEL_FRIENDLY_NAME,
                        "host_name": HOST_NAME,
                        "today": datetime.date.today().strftime("%A %d"),
                        "include_deep_dive": args.deep_dive,
                        "yesterday_date": yesterday_str,
                    },
                ),
                api_key=OPENROUTER_API_KEY,
            )

            script_content = llm(
                render_prompt_template(
                    "user_message.j2",
                    {
                        "news_feed": news_feed,
                        "previous_script": previous_script,
                        "previous_date": yesterday_str,
                    },
                )
            )

            with open(SCRIPT_PATH, "w") as f:
                f.write(script_content)

            print(f"Script saved to {SCRIPT_PATH}")

        except Exception as e:
            print(f"Error generating script: {e}")
            sys.exit(1)

    with open(SCRIPT_PATH, "r") as f:
        script_content = f.read()

    # Generate Audio
    if args.skip_audio:
        print("Skipping audio generation (--skip-audio).")
    elif os.path.exists(AUDIO_PATH):
        print(f"Found existing audio at {AUDIO_PATH}. Skipping TTS.")
    else:
        print("Generating audio...")

        try:
            tts = TextToSpeech()
            tts(script_content, AUDIO_PATH)
        except Exception as e:
            print(f"Failed to generate audio: {e}")

    print("\n--- Process Complete ---")
    print(f"Sources: {SOURCES_PATH}")
    print(f"Script:  {SCRIPT_PATH}")
    if not args.skip_audio:
        print(f"Audio:   {AUDIO_PATH}")


if __name__ == "__main__":
    main()
