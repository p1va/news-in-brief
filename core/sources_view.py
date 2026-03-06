"""
Sources View - Generate interactive HTML view of news sources grouped by topic clusters.

Uses HAC (Hierarchical Agglomerative Clustering) to group similar articles,
then renders an HTML page with:
- Top stories (clusters) that can be expanded to see all articles
- Unclustered sources at the bottom
"""

import json
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
        "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601
        "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO 8601 with microseconds
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
            articles.append(
                Article(
                    title=row["title"],
                    source=row["source"],
                    link=link,
                    description=row.get("description", ""),
                    author=row.get("author", ""),
                    date=date_str,
                    domain=extract_domain(link),
                    time_display=format_time_display(parsed_dt, reference_date),
                    parsed_date=parsed_dt,
                )
            )

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

        clusters.append(
            Cluster(
                id=label,
                articles=articles,
                medoid_title=medoid_title,
                medoid_description=medoid_desc,
                source_count=n_sources,
                source_diversity=diversity,
                importance_score=importance,
                source_domains=source_domains,
            )
        )

    # Sort clusters by importance
    clusters.sort(key=lambda c: c.importance_score, reverse=True)

    # Sort unclustered by source name for readability
    unclustered.sort(key=lambda a: (a.source, a.title))

    return clusters, unclustered




def format_english_datetime(dt: datetime) -> str:
    """Format datetime in English style: 'Tue 21 Jan 18:45'."""
    return dt.strftime("%a %d %b %H:%M")


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
    clustered_pct = (
        round(100 * clustered_count / total_articles) if total_articles > 0 else 0
    )

    # Load template from file
    template_path = Path("templates/sources_view.html.j2")
    template_content = template_path.read_text(encoding="utf-8")
    template = Template(template_content)

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


def save_top_stories_json(
    clusters: list[Cluster],
    date: str,
    generated_at: str,
    output_path: Path,
    max_stories: int = 10,
) -> Path:
    """
    Save a JSON summary of the top story clusters for use by the landing page.

    Args:
        clusters: Sorted list of Cluster objects (by importance)
        date: Reference date string (YYYY-MM-DD)
        generated_at: Italian-formatted generation timestamp
        output_path: Path where the JSON file will be saved
        max_stories: Maximum number of stories to include

    Returns:
        Path to the generated JSON file
    """
    stories = []
    for rank, cluster in enumerate(clusters[:max_stories], start=1):
        stories.append(
            {
                "rank": rank,
                "title": cluster.medoid_title,
                "source_count": cluster.source_count,
                "article_count": len(cluster.articles),
                "source_domains": cluster.source_domains,
                "links": [a.link for a in cluster.articles if a.link][:5],
            }
        )

    data = {
        "date": date,
        "generated_at": generated_at,
        "stories": stories,
    }

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


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
    generated_at = format_english_datetime(datetime.now())

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

    # Emit top-stories JSON sidecar for the landing page
    json_path = output_path.parent / f"{reference_date}-top-stories.json"
    save_top_stories_json(clusters, reference_date, generated_at, json_path)

    return output_path
