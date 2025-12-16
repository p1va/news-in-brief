<div align="center">

# {News} In Brief

Generate AI-narrated daily briefing from RSS feeds.

</div>

### Listen

- **[Italy Today](https://p1va.github.io/news-in-brief/italy-today/)**: Daily briefing on Italian news
- **[Asia In Brief](https://p1va.github.io/news-in-brief/asia-in-brief/)**: Daily briefing on everything Asia

## Installation

```sh
uv sync
```

## Usage

Before running ensure `OPENROUTER_API_KEY` and `ELEVENLABS_API_KEY` env vars are set.

### Generate Episode for a Specific Show

```sh
uv run python main.py --show asia-in-brief
```

### Generate Episodes for All Shows

```sh
uv run python main.py --all
```

### Options

- `--fetch-only` - Fetch news only, skip script generation
- `--force-refresh` - Force refresh from RSS feeds (ignore cache)
- `--skip-audio` - Generate script but skip TTS
- `--deep-dive` - Include deep dive section in the episode
- `--max-age N` - Max age of articles in days (default: 7)
- `--update-rss-only` - Only regenerate RSS feed without creating new episode

### Regenerate RSS Feed

The RSS feed is automatically updated after each episode generation. To manually regenerate the RSS feed (useful after editing episodes or fixing metadata):

```sh
# Regenerate RSS for a specific show
uv run python main.py --show asia-in-brief --update-rss-only

# Regenerate RSS for all shows
uv run python main.py --all --update-rss-only

# Alternative: Direct invocation
uv run python -m core.rss asia-in-brief
```

## Creating a New Show

1. Create a new directory for your show:

```sh
mkdir my-new-show
mkdir my-new-show/prompts
```

2. Create `my-new-show/show.toml`:

```toml
[metadata]
name = "My New Show"
description = "Description of your show"
author = "Your Name"
email = "your@email.com"
language = "en"
category = "News"
host_name = "Host Name"
cover_image = "cover.jpg"

[llm]
model = "deepseek/deepseek-v3.2-speciale"
model_friendly_name = "DeepSeek-V3.2-Special"

[tts]
voice_id = "ZF6FPAbjXT4488VcRRnw"
voice_name = "Voice Name"
model = "eleven_v3"

[rss]
base_url = "https://your-username.github.io/repo-name/my-new-show"

[[feeds]]
name = "Source Name"
url = "https://example.com/rss"
country = "Country"

# Add more feeds...
```

3. Copy and customize prompt templates:

```sh
cp -r asia-in-brief/prompts/* my-new-show/prompts/
```

4. Generate your first episode:

```sh
uv run python main.py --show my-new-show
```

## GitHub Pages Setup

The repository is configured to serve as a GitHub Pages site, making RSS feeds and audio files publicly accessible.

1. Each show directory maps to a URL path: `/{show-name}/rss.xml`
2. Audio files are served from: `/{show-name}/artifacts/YYYY-MM-DD/YYYY-MM-DD-audio.mp3`
3. The `.nojekyll` file disables Jekyll processing for faster deployment