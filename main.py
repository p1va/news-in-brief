import datetime
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from core.config import load_show_config
from core.llm import OpenRouterLLM
from core.news import NewsRepository
from core.render import render_prompt_template
from core.rss import generate_rss_feed
from core.site import generate_html
from core.tts import TextToSpeech

load_dotenv()

app = typer.Typer(
    help="News In Brief",
    add_completion=False,
    no_args_is_help=True,
)

show_app = typer.Typer(help="Manage and generate shows")
app.add_typer(show_app, name="show")


def discover_shows() -> list[str]:
    """Discover all show directories in the repository, sorted by most recent modification."""
    shows_with_mtime = []
    for item in Path(".").iterdir():
        if item.is_dir() and (item / "show.toml").exists():
            try:
                # Get modification time of the show.toml file as a proxy for show freshness
                mtime = (item / "show.toml").stat().st_mtime
                shows_with_mtime.append((item.name, mtime))
            except FileNotFoundError:
                # This should not happen given the exists() check, but good for robustness
                continue

    # Sort by modification time, newest first
    shows_with_mtime.sort(key=lambda x: x[1], reverse=True)
    return [show_name for show_name, _ in shows_with_mtime]


def process_episode(
    show_name: str,
    fetch_only: bool = False,
    force_refresh: bool = False,
    skip_audio: bool = False,
    deep_dive: bool = False,
    use_speech_tags: bool = True,
    max_age: int = 7,
):
    """Generate a single episode for the specified show."""
    show_dir = Path(show_name)

    if not show_dir.exists():
        typer.secho(
            f"Error: Show directory '{show_name}' not found.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # Load show configuration
    try:
        config = load_show_config(show_dir)
    except Exception as e:
        typer.secho(f"Error loading show config: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Set up paths
    issue_date = datetime.date.today().strftime("%Y-%m-%d")
    artifacts_folder = show_dir / "artifacts"
    issue_folder_path = artifacts_folder / issue_date
    prompts_folder = show_dir / "prompts"

    sources_path = issue_folder_path / f"{issue_date}-sources.parquet"
    script_path = issue_folder_path / f"{issue_date}-script.md"
    audio_path = issue_folder_path / f"{issue_date}-audio.mp3"
    system_prompt_path = issue_folder_path / f"{issue_date}-system-prompt.md"
    user_message_path = issue_folder_path / f"{issue_date}-user-message.md"

    # Ensure issue folder exists
    if not issue_folder_path.exists():
        typer.echo(f"Creating directory: {issue_folder_path}")
        issue_folder_path.mkdir(parents=True, exist_ok=True)

    typer.secho(
        f"\n--- {config.metadata.name}: Daily Briefing Generator [{issue_date}] ---",
        fg=typer.colors.BLUE,
        bold=True,
    )

    # 1. Acquire Data
    repo = NewsRepository()
    news_feed = repo.get_news(
        config.feeds,
        str(sources_path),
        force_refresh=force_refresh,
        max_age_days=max_age,
    )

    if not news_feed:
        typer.secho("No news data available. Exiting.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    if fetch_only:
        typer.secho("Fetch complete. Exiting (--fetch-only).", fg=typer.colors.GREEN)
        return

    # 2. Generate Script
    if not script_path.exists():
        typer.echo("Generating script...")

        # Load previous episode's script for context
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        prev_script_path = (
            artifacts_folder / yesterday_str / f"{yesterday_str}-script.md"
        )
        previous_script = "No previous episode found."
        if prev_script_path.exists():
            with open(prev_script_path, "r") as f:
                previous_script = f.read()
            typer.echo(f"Loaded context from {yesterday_str}")

        try:
            # Get API key
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                typer.secho(
                    "Error: OPENROUTER_API_KEY not found in environment.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

            # Render prompts
            system_prompt = render_prompt_template(
                "system_prompt.j2",
                {
                    "current_date": issue_date,
                    "model_name": config.llm.model_friendly_name,
                    "host_name": config.metadata.host_name,
                    "today": datetime.date.today().strftime("%A %d"),
                    "include_deep_dive": deep_dive,
                    "use_speech_tags": use_speech_tags,
                    "yesterday_date": yesterday_str,
                },
                template_dir=str(prompts_folder),
            )

            user_message = render_prompt_template(
                "user_message.j2",
                {
                    "news_feed": news_feed,
                    "previous_script": previous_script,
                    "previous_date": yesterday_str,
                },
                template_dir=str(prompts_folder),
            )

            # Save rendered prompts for inspection
            with open(system_prompt_path, "w") as f:
                f.write(system_prompt)
            with open(user_message_path, "w") as f:
                f.write(user_message)
            typer.echo(f"Saved prompts to {issue_folder_path}")

            llm = OpenRouterLLM(
                model=config.llm.model,
                system_prompt=system_prompt,
                api_key=openrouter_api_key,
            )

            script_content = llm(user_message)

            with open(script_path, "w") as f:
                f.write(script_content)

            typer.echo(f"Script saved to {script_path}")

        except Exception as e:
            typer.secho(f"Error generating script: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    with open(script_path, "r") as f:
        script_content = f.read()

    # 3. Generate Audio
    if skip_audio:
        typer.echo("Skipping audio generation (--skip-audio).")
    elif audio_path.exists():
        typer.echo(f"Found existing audio at {audio_path}. Skipping TTS.")
    else:
        typer.echo("Generating audio...")

        try:
            tts = TextToSpeech()
            tts(script_content, str(audio_path))
        except Exception as e:
            typer.secho(f"Failed to generate audio: {e}", fg=typer.colors.RED)

    # 4. Update RSS Feed
    typer.echo("\nUpdating RSS feed...")
    try:
        generate_rss_feed(show_dir)
        generate_html(show_dir)
    except Exception as e:
        typer.secho(f"Warning: Failed to update RSS/HTML: {e}", fg=typer.colors.YELLOW)

    typer.secho("\n--- Process Complete ---", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Sources: {sources_path}")
    typer.echo(f"Script:  {script_path}")
    if not skip_audio:
        typer.echo(f"Audio:   {audio_path}")
    typer.echo(f"RSS:     {show_dir / 'rss.xml'}")


@show_app.command(name="generate")
def generate(
    show_name: Annotated[
        Optional[str],
        typer.Argument(
            help="Show name (directory with show.toml). Required unless --all is set.",
        ),
    ] = None,
    all_shows: Annotated[
        bool, typer.Option("--all", "-a", help="Generate for all discovered shows")
    ] = False,
    fetch_only: Annotated[
        bool, typer.Option("--fetch-only", help="Fetch news only, skip generation")
    ] = False,
    force_refresh: Annotated[
        bool, typer.Option("--force-refresh", help="Force refresh from RSS feeds")
    ] = False,
    skip_audio: Annotated[
        bool, typer.Option("--skip-audio", help="Skip audio generation")
    ] = False,
    deep_dive: Annotated[
        bool, typer.Option("--deep-dive", help="Include deep dive section")
    ] = False,
    use_speech_tags: Annotated[
        bool,
        typer.Option(
            "--use-speech-tags/--no-use-speech-tags",
            help="Enable speech tags for TTS (default: enabled)",
        ),
    ] = True,
    max_age: Annotated[
        int, typer.Option("--max-age", help="Max age of articles in days")
    ] = 7,
):
    """
    Generate daily briefing episode(s).
    """
    if not show_name and not all_shows:
        typer.secho(
            "Error: Must specify either a show name or --all", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    shows_to_process = []
    if all_shows:
        shows_to_process = discover_shows()
        if not shows_to_process:
            typer.secho("No shows found.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        typer.echo(f"Generating episodes for {len(shows_to_process)} show(s)...")
    else:
        # show_name is not None here because of the check above
        shows_to_process = [show_name]  # type: ignore

    for s in shows_to_process:
        process_episode(
            s,
            fetch_only=fetch_only,
            force_refresh=force_refresh,
            skip_audio=skip_audio,
            deep_dive=deep_dive,
            use_speech_tags=use_speech_tags,
            max_age=max_age,
        )


@show_app.command(name="update-rss")
def update_rss(
    show_name: Annotated[
        Optional[str],
        typer.Argument(
            help="Show name (directory with show.toml). Required unless --all is set.",
        ),
    ] = None,
    all_shows: Annotated[
        bool, typer.Option("--all", "-a", help="Update RSS for all discovered shows")
    ] = False,
):
    """
    Only regenerate RSS feed without creating new episode.
    """
    if not show_name and not all_shows:
        typer.secho(
            "Error: Must specify either a show name or --all", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    shows_to_process = []
    if all_shows:
        shows_to_process = discover_shows()
    else:
        shows_to_process = [show_name]  # type: ignore

    for s in shows_to_process:
        show_dir = Path(s)
        if not show_dir.exists():
            typer.secho(f"Error: Show directory '{s}' not found.", fg=typer.colors.RED)
            continue

        try:
            # We need to load config just to print the name, or just use dir name
            typer.echo(f"Updating RSS feed for {s}...")
            generate_rss_feed(show_dir)
            generate_html(show_dir)
            typer.secho(
                f"RSS feed updated at {show_dir / 'rss.xml'}", fg=typer.colors.GREEN
            )
        except Exception as e:
            typer.secho(f"Error updating RSS feed: {e}", fg=typer.colors.RED)


@show_app.command(name="list")
def list_shows():
    """
    List all discovered shows in the current directory.
    """
    shows = discover_shows()
    if shows:
        typer.echo("Available shows:")
        for show in shows:
            typer.echo(f"  - {show}")
    else:
        typer.echo("No shows found. Create a directory with show.toml to get started.")


if __name__ == "__main__":
    app()
