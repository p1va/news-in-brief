import datetime
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from core.analyzer import NewsAnalyzer, generate_stories_markdown, save_stories_markdown
from core.config import load_show_config
from core.embeddings import generate_embeddings_for_parquet
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


# Pipeline steps in order
PIPELINE_STEPS = ["fetch", "embed", "analyze", "prompt", "script", "audio"]


def get_step_index(step: str) -> int:
    """Get the index of a step, or -1 if invalid."""
    try:
        return PIPELINE_STEPS.index(step)
    except ValueError:
        return -1


def should_run_step(current_step: str, until_step: Optional[str]) -> bool:
    """Check if we should run the current step given the --until value."""
    if until_step is None:
        return True
    current_idx = get_step_index(current_step)
    until_idx = get_step_index(until_step)
    return current_idx <= until_idx


def discover_shows() -> list[str]:
    """Discover all show directories in the repository, sorted by most recent modification."""
    shows_with_mtime = []
    for item in Path(".").iterdir():
        if item.is_dir() and (item / "show.toml").exists():
            try:
                mtime = (item / "show.toml").stat().st_mtime
                shows_with_mtime.append((item.name, mtime))
            except FileNotFoundError:
                continue

    shows_with_mtime.sort(key=lambda x: x[1], reverse=True)
    return [show_name for show_name, _ in shows_with_mtime]


def process_episode(
    show_name: str,
    until_step: Optional[str] = None,
    force_refresh: bool = False,
    deep_dive: bool = False,
    use_speech_tags: bool = True,
    max_age: int = 2,
):
    """
    Generate a single episode for the specified show.

    Pipeline steps: fetch → embed → analyze → prompt → script → audio
    Use --until to stop at a specific step.
    """
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
    embeddings_path = (
        issue_folder_path / f"{issue_date}-sources-with-embeddings.parquet"
    )
    stories_path = issue_folder_path / f"{issue_date}-stories.md"
    system_prompt_path = issue_folder_path / f"{issue_date}-system-prompt.md"
    user_message_path = issue_folder_path / f"{issue_date}-user-message.md"
    script_path = issue_folder_path / f"{issue_date}-script.md"
    audio_path = issue_folder_path / f"{issue_date}-audio.mp3"

    # Ensure issue folder exists
    if not issue_folder_path.exists():
        typer.echo(f"Creating directory: {issue_folder_path}")
        issue_folder_path.mkdir(parents=True, exist_ok=True)

    typer.secho(
        f"\n--- {config.metadata.name}: Daily Briefing Generator [{issue_date}] ---",
        fg=typer.colors.BLUE,
        bold=True,
    )
    if until_step:
        typer.echo(f"Running pipeline until: {until_step}")

    # =========================================================================
    # STEP 1: FETCH - RSS feeds → parquet
    # =========================================================================
    if should_run_step("fetch", until_step):
        typer.echo("\n[1/6] Fetching RSS feeds...")
        repo = NewsRepository()
        news_feed = repo.get_news(
            config.feeds,
            str(sources_path),
            force_refresh=force_refresh,
            max_age_days=max_age,
            cleaning_config=config.cleaning,
        )

        if not news_feed:
            typer.secho("No news data available. Exiting.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

        typer.secho(
            f"Fetched {sum(len(v) for v in news_feed.values())} articles",
            fg=typer.colors.GREEN,
        )

        if not should_run_step("embed", until_step):
            typer.secho("\n--- Stopped at: fetch ---", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"Output: {sources_path}")
            return

    # =========================================================================
    # STEP 2: EMBED - Generate embeddings
    # =========================================================================
    if should_run_step("embed", until_step):
        typer.echo("\n[2/6] Generating embeddings...")

        if not sources_path.exists():
            typer.secho(
                "Error: Sources parquet not found. Run 'fetch' step first.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        embeddings_path = generate_embeddings_for_parquet(sources_path)
        typer.secho(f"Embeddings saved to {embeddings_path}", fg=typer.colors.GREEN)

        if not should_run_step("analyze", until_step):
            typer.secho("\n--- Stopped at: embed ---", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"Output: {embeddings_path}")
            return

    # =========================================================================
    # STEP 3: ANALYZE - Cluster + junk detection → stories.md
    # =========================================================================
    stories_markdown = ""
    if should_run_step("analyze", until_step):
        typer.echo("\n[3/6] Analyzing stories (clustering + junk filtering)...")

        if not embeddings_path.exists():
            typer.secho(
                "Error: Embeddings parquet not found. Run 'embed' step first.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        df = pd.read_parquet(embeddings_path)
        embeddings = np.array(df["embedding"].tolist())

        analyzer = NewsAnalyzer(threshold=config.cleaning.cluster_threshold)
        analysis = analyzer.analyze(df, embeddings)

        typer.echo(
            f"Found {len(analysis.top_stories)} top stories, "
            f"{len(analysis.niche_stories)} niche"
        )

        stories_markdown = generate_stories_markdown(analysis, issue_date)
        save_stories_markdown(analysis, stories_path, issue_date)
        typer.secho(f"Stories saved to {stories_path}", fg=typer.colors.GREEN)

        if not should_run_step("prompt", until_step):
            typer.secho(
                "\n--- Stopped at: analyze ---", fg=typer.colors.GREEN, bold=True
            )
            typer.echo(f"Output: {stories_path}")
            return

    # =========================================================================
    # STEP 4: PROMPT - Render prompts → markdown files
    # =========================================================================
    if should_run_step("prompt", until_step):
        typer.echo("\n[4/6] Rendering prompts...")

        # Load stories markdown if we didn't just generate it
        if not stories_markdown and stories_path.exists():
            stories_markdown = stories_path.read_text()

        # Load news feed for prompt
        if not sources_path.exists():
            typer.secho("Error: Sources parquet not found.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        repo = NewsRepository()
        news_feed = repo.get_news(
            config.feeds,
            str(sources_path),
            force_refresh=False,
            max_age_days=max_age,
            cleaning_config=config.cleaning,
        )

        # Load previous episode's script for context
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        prev_script_path = (
            artifacts_folder / yesterday_str / f"{yesterday_str}-script.md"
        )
        previous_script = "No previous episode found."
        if prev_script_path.exists():
            previous_script = prev_script_path.read_text()
            typer.echo(f"Loaded context from {yesterday_str}")

        system_prompt = render_prompt_template(
            "system_prompt.j2",
            {
                "current_date": issue_date,
                "model_name": config.llm.model_friendly_name,
                # "host_name": config.metadata.host_name,
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
                "stories_context": stories_markdown,
            },
            template_dir=str(prompts_folder),
        )

        system_prompt_path.write_text(system_prompt)
        user_message_path.write_text(user_message)
        typer.secho(f"Prompts saved to {issue_folder_path}", fg=typer.colors.GREEN)

        if not should_run_step("script", until_step):
            typer.secho(
                "\n--- Stopped at: prompt ---", fg=typer.colors.GREEN, bold=True
            )
            typer.echo(f"System prompt: {system_prompt_path}")
            typer.echo(f"User message:  {user_message_path}")
            return

    # =========================================================================
    # STEP 5: SCRIPT - LLM call → script.md
    # =========================================================================
    if should_run_step("script", until_step):
        typer.echo("\n[5/6] Generating script via LLM...")

        if script_path.exists():
            typer.echo(f"Script already exists at {script_path}. Skipping LLM call.")
        else:
            if not system_prompt_path.exists() or not user_message_path.exists():
                typer.secho(
                    "Error: Prompts not found. Run 'prompt' step first.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

            system_prompt = system_prompt_path.read_text()
            user_message = user_message_path.read_text()

            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                typer.secho(
                    "Error: OPENROUTER_API_KEY not found in environment.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

            llm = OpenRouterLLM(
                model=config.llm.model,
                system_prompt=system_prompt,
                api_key=openrouter_api_key,
            )

            script_content = llm(user_message)
            script_path.write_text(script_content)
            typer.secho(f"Script saved to {script_path}", fg=typer.colors.GREEN)

        if not should_run_step("audio", until_step):
            typer.secho(
                "\n--- Stopped at: script ---", fg=typer.colors.GREEN, bold=True
            )
            typer.echo(f"Output: {script_path}")
            return

    # =========================================================================
    # STEP 6: AUDIO - TTS → audio.mp3
    # =========================================================================
    if should_run_step("audio", until_step):
        typer.echo("\n[6/6] Generating audio via TTS...")

        if audio_path.exists():
            typer.echo(f"Audio already exists at {audio_path}. Skipping TTS.")
        else:
            if not script_path.exists():
                typer.secho(
                    "Error: Script not found. Run 'script' step first.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

            script_content = script_path.read_text()

            try:
                tts = TextToSpeech()
                tts(script_content, str(audio_path))
                typer.secho(f"Audio saved to {audio_path}", fg=typer.colors.GREEN)
            except Exception as e:
                typer.secho(f"Failed to generate audio: {e}", fg=typer.colors.RED)

    # =========================================================================
    # FINALIZE - Update RSS feed
    # =========================================================================
    typer.echo("\nUpdating RSS feed...")
    try:
        generate_rss_feed(show_dir)
        generate_html(show_dir)
    except Exception as e:
        typer.secho(f"Warning: Failed to update RSS/HTML: {e}", fg=typer.colors.YELLOW)

    typer.secho("\n--- Pipeline Complete ---", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Sources:  {sources_path}")
    typer.echo(f"Stories:  {stories_path}")
    typer.echo(f"Prompts:  {system_prompt_path}")
    typer.echo(f"Script:   {script_path}")
    typer.echo(f"Audio:    {audio_path}")
    typer.echo(f"RSS:      {show_dir / 'rss.xml'}")


def validate_until_step(value: Optional[str]) -> Optional[str]:
    """Validate the --until step value."""
    if value is None:
        return None
    if value not in PIPELINE_STEPS:
        valid_steps = ", ".join(PIPELINE_STEPS)
        raise typer.BadParameter(
            f"Invalid step '{value}'. Valid steps are: {valid_steps}"
        )
    return value


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
    until: Annotated[
        Optional[str],
        typer.Option(
            "--until",
            "-u",
            help="Stop after this step. Steps: fetch, embed, analyze, prompt, script, audio",
            callback=validate_until_step,
        ),
    ] = None,
    force_refresh: Annotated[
        bool, typer.Option("--force-refresh", "-f", help="Force refresh from RSS feeds")
    ] = False,
    deep_dive: Annotated[
        bool, typer.Option("--deep-dive", help="Include deep dive section in prompt")
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
    ] = 2,
):
    """
    Generate daily briefing episode(s).

    Pipeline steps (in order):
      1. fetch   - Download RSS feeds → parquet
      2. embed   - Generate embeddings for articles
      3. analyze - Cluster stories + filter junk → stories.md
      4. prompt  - Render LLM prompts → markdown files
      5. script  - Call LLM → script.md
      6. audio   - Generate TTS → audio.mp3

    Examples:
      python main.py show generate italy-today --until fetch
      python main.py show generate italy-today --until analyze
      python main.py show generate italy-today  # full pipeline
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
        shows_to_process = [show_name]  # type: ignore

    for s in shows_to_process:
        process_episode(
            s,
            until_step=until,
            force_refresh=force_refresh,
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
