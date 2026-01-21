"""
Sources View - Generate interactive HTML view of news sources grouped by topic clusters.

Uses HAC (Hierarchical Agglomerative Clustering) to group similar articles,
then renders an HTML page with:
- Top stories (clusters) that can be expanded to see all articles
- Unclustered sources at the bottom
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from jinja2 import Template
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize


@dataclass
class Article:
    title: str
    source: str
    link: str
    description: str
    author: str
    date: str
    domain: str = ""
    time_display: str = ""
    parsed_date: datetime | None = None


@dataclass
class Cluster:
    id: int
    articles: list[Article]
    medoid_title: str
    medoid_description: str
    source_count: int
    source_diversity: float
    importance_score: float
    source_domains: list[str] = field(default_factory=list)


def extract_domain(url: str) -> str:
    """Extract domain from URL for favicon lookup."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def parse_datetime(date_str: str) -> datetime | None:
    """Parse date string and return datetime object."""
    if not date_str:
        return None

    # Common RSS date formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 822: "Tue, 20 Jan 2026 19:52:53 +0100"
        "%Y-%m-%dT%H:%M:%S%z",        # ISO 8601
        "%Y-%m-%dT%H:%M:%S.%f%z",     # ISO 8601 with microseconds
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return None


def format_time_display(dt: datetime | None, reference_date: str) -> str:
    """Format datetime for display. Show date if different from reference."""
    if not dt:
        return ""

    # Extract reference date (YYYY-MM-DD format from filename)
    ref_date_str = reference_date  # e.g., "2026-01-20"

    # Format article date
    article_date_str = dt.strftime("%Y-%m-%d")

    if article_date_str == ref_date_str:
        # Same day: just show time
        return dt.strftime("%H:%M")
    else:
        # Different day: show date and time
        return dt.strftime("%d %b %H:%M")  # e.g., "19 Jan 14:30"


def load_embeddings(parquet_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load parquet and extract embeddings as numpy array."""
    df = pd.read_parquet(parquet_path)
    embeddings = np.array(df["embedding"].tolist())
    return df, embeddings


def get_cluster_medoid(
    embeddings: np.ndarray, indices: np.ndarray, df: pd.DataFrame
) -> tuple[str, str, int]:
    """
    Find the 'Medoid' - the article closest to the center of the cluster.
    Returns: (title, description, index_in_df)
    """
    if len(embeddings) == 0:
        return "", "", -1

    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    dists = cosine_distances(embeddings, centroid).flatten()
    min_idx = np.argmin(dists)

    original_idx = indices[min_idx]
    return (
        df.iloc[original_idx]["title"],
        df.iloc[original_idx]["description"],
        original_idx,
    )


def run_hac_clustering(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    reference_date: str,
    threshold: float = 0.25,
    linkage: str = "average",
) -> tuple[list[Cluster], list[Article]]:
    """
    Run HAC clustering and return clusters and unclustered articles.

    Args:
        threshold: Distance threshold (lower = tighter clusters)
        linkage: 'average', 'complete', 'single', or 'ward' (ward requires euclidean)
    """
    embeddings_norm = normalize(embeddings)

    hac = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage=linkage,
    )
    labels = hac.fit_predict(embeddings_norm)

    clusters = []
    unclustered = []

    # Group by cluster label
    label_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        cluster_emb = embeddings_norm[indices]

        # Extract articles
        articles = []
        for idx in indices:
            row = df.iloc[idx]
            link = row.get("link", "")
            date_str = row.get("date", "")
            parsed_dt = parse_datetime(date_str)
            articles.append(Article(
                title=row["title"],
                source=row["source"],
                link=link,
                description=row.get("description", ""),
                author=row.get("author", ""),
                date=date_str,
                domain=extract_domain(link),
                time_display=format_time_display(parsed_dt, reference_date),
                parsed_date=parsed_dt,
            ))

        # Sort articles by date (most recent first)
        # Use timestamp for comparison to avoid timezone-aware vs naive issues
        def get_sort_key(a: Article) -> float:
            if a.parsed_date:
                return a.parsed_date.timestamp()
            return 0.0
        articles.sort(key=get_sort_key, reverse=True)

        # Single articles go to unclustered
        if len(articles) == 1:
            unclustered.append(articles[0])
            continue

        # Get medoid (representative article)
        medoid_title, medoid_desc, _ = get_cluster_medoid(cluster_emb, indices, df)

        # Calculate metrics
        sources = [a.source for a in articles]
        n_sources = len(set(sources))
        diversity = n_sources / len(articles)
        importance = len(articles) * (0.5 + 0.5 * diversity)

        # Get unique domains for favicon display (one per source, preserve order)
        seen_sources = set()
        source_domains = []
        for a in articles:
            if a.source not in seen_sources and a.domain:
                seen_sources.add(a.source)
                source_domains.append(a.domain)

        clusters.append(Cluster(
            id=label,
            articles=articles,
            medoid_title=medoid_title,
            medoid_description=medoid_desc,
            source_count=n_sources,
            source_diversity=diversity,
            importance_score=importance,
            source_domains=source_domains,
        ))

    # Sort clusters by importance
    clusters.sort(key=lambda c: c.importance_score, reverse=True)

    # Sort unclustered by source name for readability
    unclustered.sort(key=lambda a: (a.source, a.title))

    return clusters, unclustered


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ date }} - Rassegna Stampa</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.7;
            max-width: 860px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #faf9f7;
            color: #2d2d2d;
        }
        h1 {
            font-family: 'Newsreader', Georgia, serif;
            font-weight: 700;
            font-size: 2.2em;
            color: #1a1a1a;
            border-bottom: 2px solid #1a1a1a;
            padding-bottom: 15px;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }
        h2 {
            font-family: 'Newsreader', Georgia, serif;
            font-weight: 400;
            font-size: 1.4em;
            color: #1a1a1a;
            margin-top: 50px;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 1px solid #ddd;
        }
        .stats {
            background: #fff;
            padding: 12px 18px;
            border: 1px solid #e0e0e0;
            margin-bottom: 30px;
            font-size: 13px;
            color: #555;
        }
        .stats span {
            margin-right: 25px;
        }
        .stats strong {
            color: #1a1a1a;
            font-weight: 600;
        }

        /* Cluster cards */
        .cluster {
            background: #fff;
            border: 1px solid #e0e0e0;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            transition: box-shadow 0.2s;
        }
        .cluster:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }
        .cluster-header {
            padding: 14px;
            cursor: pointer;
            display: flex;
            align-items: flex-start;
            gap: 0;
        }
        .cluster-rank {
            font-family: 'Newsreader', Georgia, serif;
            font-size: 18px;
            font-weight: 700;
            color: #999;
            width: 55px;
            flex-shrink: 0;
            text-align: center;
        }
        .cluster-info {
            flex: 1;
            min-width: 0;
        }
        .cluster-title {
            font-family: 'Newsreader', Georgia, serif;
            font-weight: 700;
            font-size: 17px;
            line-height: 1.4;
            margin-bottom: 8px;
            color: #1a1a1a;
        }
        .cluster-meta {
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        .cluster-meta-text {
            font-size: 13px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .source-icons {
            display: flex;
            align-items: center;
        }
        .source-icon {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #d0d0d0;
            border: 2px solid #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2), inset 0 0 0 1px rgba(0,0,0,0.05);
            object-fit: contain;
            margin-left: -6px;
            position: relative;
        }
        .source-icon:first-child {
            margin-left: 0;
        }
        .source-icon:nth-child(1) { z-index: 7; }
        .source-icon:nth-child(2) { z-index: 6; }
        .source-icon:nth-child(3) { z-index: 5; }
        .source-icon:nth-child(4) { z-index: 4; }
        .source-icon:nth-child(5) { z-index: 3; }
        .source-icon:nth-child(6) { z-index: 2; }
        .source-icon-more {
            font-size: 11px;
            font-weight: 600;
            color: #666;
            background: #f0f0f0;
            border-radius: 50%;
            width: 22px;
            height: 22px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-left: -6px;
            border: 2px solid #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            position: relative;
            z-index: 1;
        }
        .expand-icon {
            color: #bbb;
            font-size: 12px;
            transition: transform 0.2s;
            margin-top: 4px;
        }
        .cluster.expanded .expand-icon {
            transform: rotate(180deg);
        }
        .cluster-articles {
            display: none;
            border-top: 1px solid #eee;
            padding: 0;
            background: #fcfcfc;
        }
        .cluster.expanded .cluster-articles {
            display: block;
        }

        /* Article items - timeline layout */
        .article {
            display: flex;
            gap: 0;
            padding: 12px 14px;
            border-bottom: 1px solid #e8e8e8;
        }
        .article:last-child {
            border-bottom: none;
        }
        .article-time-col {
            width: 55px;
            flex-shrink: 0;
            text-align: center;
            padding-top: 3px;
        }
        .article-time {
            font-size: 11px;
            font-weight: 600;
            color: #888;
            font-variant-numeric: tabular-nums;
            line-height: 1.3;
        }
        .article-time-date {
            font-size: 10px;
            color: #aaa;
        }
        .article-content {
            flex: 1;
            min-width: 0;
        }
        .article-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }
        .article-source-icon {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #d0d0d0;
            border: 1px solid #bbb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            object-fit: contain;
        }
        .article-source {
            font-size: 12px;
            color: #666;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .article-title {
            margin: 0;
            font-size: 15px;
            line-height: 1.45;
        }
        .article-title a {
            color: #1a1a1a;
            text-decoration: none;
        }
        .article-title a:hover {
            color: #0066cc;
            text-decoration: underline;
        }
        .article-desc {
            font-size: 13px;
            color: #666;
            margin-top: 6px;
            line-height: 1.5;
        }

        /* Unclustered section */
        .unclustered {
            margin-top: 60px;
        }
        .unclustered h2 {
            color: #666;
        }
        .unclustered-intro {
            color: #888;
            font-size: 13px;
            margin-bottom: 20px;
        }
        .source-group {
            background: #fff;
            border: 1px solid #e8e8e8;
            margin-bottom: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .source-group-header {
            padding: 10px 15px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .source-group-header:hover {
            background: #fafafa;
        }
        .source-group-icon {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #d0d0d0;
            border: 1px solid #bbb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            object-fit: contain;
        }
        .source-name {
            font-weight: 600;
            color: #333;
            flex: 1;
        }
        .source-count {
            color: #999;
            font-size: 12px;
        }
        .source-articles {
            display: none;
            padding: 8px 15px 12px 15px;
            background: #fafafa;
            border-top: 1px solid #e8e8e8;
        }
        .source-group.expanded .source-articles {
            display: block;
        }
        .source-article {
            display: flex;
            gap: 0;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .source-article:last-child {
            border-bottom: none;
        }
        .source-article-time {
            width: 55px;
            flex-shrink: 0;
            font-size: 11px;
            font-weight: 600;
            color: #888;
            text-align: center;
            padding-top: 2px;
            line-height: 1.3;
        }
        .source-article-time-date {
            font-size: 10px;
            color: #aaa;
        }
        .source-article-content {
            flex: 1;
            min-width: 0;
        }
        .source-article-title {
            color: #1a1a1a;
            text-decoration: none;
            font-size: 14px;
            line-height: 1.4;
        }
        .source-article-title:hover {
            color: #0066cc;
            text-decoration: underline;
        }
        .source-article-desc {
            font-size: 13px;
            color: #888;
            margin-top: 3px;
            line-height: 1.4;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Config panel - subtle */
        .config {
            font-size: 11px;
            color: #999;
            margin-bottom: 25px;
        }
        .config span {
            margin-right: 15px;
        }
    </style>
</head>
<body>
    <h1>Rassegna Stampa</h1>
    <div class="subtitle">Generato {{ generated_at }} · {{ total_articles }} articoli da {{ source_count }} testate</div>

    <div class="config">
        <span>HAC clustering ({{ linkage }}, threshold={{ threshold }})</span>
        <span>{{ cluster_count }} storie · {{ clustered_pct }}% raggruppati</span>
    </div>

    <h2>Storie Principali</h2>

    {% for cluster in clusters %}
    <div class="cluster" onclick="this.classList.toggle('expanded')">
        <div class="cluster-header">
            <div class="cluster-rank">{{ loop.index }}.</div>
            <div class="cluster-info">
                <div class="cluster-title">{{ cluster.medoid_title }}</div>
                <div class="cluster-meta">
                    <div class="source-icons">
                        {% for domain in cluster.source_domains[:6] %}
                        <img class="source-icon" src="https://www.google.com/s2/favicons?domain={{ domain }}&sz=32" alt="" loading="lazy">
                        {% endfor %}
                        {% if cluster.source_domains|length > 6 %}
                        <span class="source-icon-more">+{{ cluster.source_domains|length - 6 }}</span>
                        {% endif %}
                    </div>
                    <span class="cluster-meta-text">{{ cluster.articles|length }} articoli · {{ cluster.source_count }} testate</span>
                </div>
            </div>
            <span class="expand-icon">▼</span>
        </div>
        <div class="cluster-articles">
            {% for article in cluster.articles %}
            <div class="article">
                <div class="article-time-col">
                    {% if article.time_display %}
                    {% if ' ' in article.time_display %}
                    <div class="article-time-date">{{ article.time_display.split(' ')[:-1]|join(' ') }}</div>
                    <div class="article-time">{{ article.time_display.split(' ')[-1] }}</div>
                    {% else %}
                    <div class="article-time">{{ article.time_display }}</div>
                    {% endif %}
                    {% endif %}
                </div>
                <div class="article-content">
                    <div class="article-header">
                        <img class="article-source-icon" src="https://www.google.com/s2/favicons?domain={{ article.domain }}&sz=32" alt="" loading="lazy">
                        <span class="article-source">{{ article.source }}</span>
                    </div>
                    <div class="article-title">
                        {% if article.link %}
                        <a href="{{ article.link }}" target="_blank">{{ article.title }}</a>
                        {% else %}
                        {{ article.title }}
                        {% endif %}
                    </div>
                    {% if article.description %}
                    <div class="article-desc">{{ article.description[:180] }}{% if article.description|length > 180 %}...{% endif %}</div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endfor %}

    <div class="unclustered">
        <h2>Altre Notizie ({{ unclustered|length }})</h2>
        <p class="unclustered-intro">Articoli non raggruppati in storie principali.</p>

        {% for source, articles in unclustered_by_source.items() %}
        <div class="source-group" onclick="this.classList.toggle('expanded')">
            <div class="source-group-header">
                <img class="source-group-icon" src="https://www.google.com/s2/favicons?domain={{ articles[0].domain }}&sz=32" alt="" loading="lazy">
                <span class="source-name">{{ source }}</span>
                <span class="source-count">{{ articles|length }}</span>
            </div>
            <div class="source-articles">
                {% for article in articles %}
                <div class="source-article">
                    <div class="source-article-time">
                        {% if article.time_display %}
                        {% if ' ' in article.time_display %}
                        <div class="source-article-time-date">{{ article.time_display.split(' ')[:-1]|join(' ') }}</div>
                        {% endif %}
                        {{ article.time_display.split(' ')[-1] if ' ' in article.time_display else article.time_display }}
                        {% endif %}
                    </div>
                    <div class="source-article-content">
                        {% if article.link %}
                        <a class="source-article-title" href="{{ article.link }}" target="_blank">{{ article.title }}</a>
                        {% else %}
                        <span class="source-article-title">{{ article.title }}</span>
                        {% endif %}
                        {% if article.description %}
                        <div class="source-article-desc">{{ article.description }}</div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


def format_italian_datetime(dt: datetime) -> str:
    """Format datetime in Italian style: 'Martedì 21 Gennaio 18:45'."""
    days = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
    months = [
        "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
        "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
    ]
    day_name = days[dt.weekday()]
    month_name = months[dt.month - 1]
    return f"{day_name} {dt.day} {month_name} {dt.strftime('%H:%M')}"


def generate_html(
    clusters: list[Cluster],
    unclustered: list[Article],
    date: str,
    threshold: float,
    linkage: str,
    total_articles: int,
    unique_sources: int,
    generated_at: str,
) -> str:
    """Generate HTML from clusters and unclustered articles."""

    # Group unclustered by source
    unclustered_by_source = {}
    for article in unclustered:
        if article.source not in unclustered_by_source:
            unclustered_by_source[article.source] = []
        unclustered_by_source[article.source].append(article)

    # Sort articles within each source by date (most recent first)
    def get_sort_key(a: Article) -> float:
        if a.parsed_date:
            return a.parsed_date.timestamp()
        return 0.0
    for source in unclustered_by_source:
        unclustered_by_source[source].sort(key=get_sort_key, reverse=True)

    # Sort sources by article count (descending)
    unclustered_by_source = dict(
        sorted(unclustered_by_source.items(), key=lambda x: -len(x[1]))
    )

    # Calculate stats
    clustered_count = sum(len(c.articles) for c in clusters)
    clustered_pct = round(100 * clustered_count / total_articles) if total_articles > 0 else 0

    template = Template(HTML_TEMPLATE)
    return template.render(
        date=date,
        threshold=threshold,
        linkage=linkage,
        total_articles=total_articles,
        source_count=unique_sources,
        cluster_count=len(clusters),
        clustered_count=clustered_count,
        clustered_pct=clustered_pct,
        unclustered_count=len(unclustered),
        clusters=clusters,
        unclustered=unclustered,
        unclustered_by_source=unclustered_by_source,
        generated_at=generated_at,
    )


def generate_sources_view(
    embeddings_path: Path,
    output_path: Path,
    reference_date: str,
    threshold: float = 0.40,
    linkage: str = "average",
) -> Path:
    """
    Generate interactive HTML view of news sources grouped by topic clusters.

    Args:
        embeddings_path: Path to the parquet file with embeddings
        output_path: Path where the HTML file will be saved
        reference_date: Date string (YYYY-MM-DD) for the issue
        threshold: Distance threshold for HAC clustering (default 0.40)
        linkage: Linkage method for HAC ('average', 'complete', 'single')

    Returns:
        Path to the generated HTML file
    """
    df, embeddings = load_embeddings(embeddings_path)
    clusters, unclustered = run_hac_clustering(
        df, embeddings, reference_date, threshold, linkage
    )

    unique_sources = df["source"].nunique()
    generated_at = format_italian_datetime(datetime.now())

    html = generate_html(
        clusters=clusters,
        unclustered=unclustered,
        date=reference_date,
        threshold=threshold,
        linkage=linkage,
        total_articles=len(df),
        unique_sources=unique_sources,
        generated_at=generated_at,
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path
