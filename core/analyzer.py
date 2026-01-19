from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize


@dataclass
class TopicCluster:
    id: int
    size: int
    label: str  # Medoid title
    description: str  # Medoid description
    articles: List[dict]  # {title, source, link...}
    sources: List[str]
    diversity_score: float  # 0-1, unique sources / total articles
    importance_score: float = 0.0  # size * (0.5 + 0.5 * diversity)
    is_junk: bool = False
    junk_reason: str = ""


@dataclass
class DailyAnalysis:
    top_stories: List[TopicCluster]
    niche_stories: List[TopicCluster]
    junk_stories: List[TopicCluster]
    total_articles: int
    total_clusters: int
    singletons: int


class NewsAnalyzer:
    """
    Analyzes news articles using HAC clustering.
    """

    def __init__(
        self,
        threshold: float = 0.25,
    ):
        """
        Args:
            threshold: HAC distance threshold (0.25 = articles must be ~0.75 similar)
        """
        self.threshold = threshold

    def analyze(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> DailyAnalysis:
        """
        Perform HAC clustering and analyze the results.

        Args:
            df: DataFrame with title, description, source, link columns
            embeddings: Numpy array of article embeddings

        Returns:
            DailyAnalysis with categorized story clusters
        """
        if df.empty or len(embeddings) == 0:
            return DailyAnalysis([], [], [], 0, 0, 0)

        # Step 1: Normalize for cosine metric
        embeddings_norm = normalize(embeddings)

        # Step 2: Run HAC clustering
        hac = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.threshold,
            metric="cosine",
            linkage="average",
        )
        labels = hac.fit_predict(embeddings_norm)

        # Step 3: Build cluster objects
        clusters = []
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            mask = labels == label
            cluster = self._build_cluster(label, df, embeddings_norm, mask)
            clusters.append(cluster)

        # Step 4: Sort by importance score
        clusters.sort(key=lambda x: x.importance_score, reverse=True)

        # Step 5: Categorize clusters
        top_stories = []
        niche_stories = []
        junk_stories = []  # Single-source spam clusters

        for c in clusters:
            # Single-source spam detection (3+ articles from 1 source)
            if c.size > 2 and len(c.sources) == 1:
                c.is_junk = True
                c.junk_reason = f"Single source ({c.sources[0]})"
                junk_stories.append(c)
            elif c.size >= 3 and c.diversity_score >= 0.3:
                # Big story covered by multiple sources
                top_stories.append(c)
            else:
                # Valid but small/niche or singleton
                niche_stories.append(c)

        return DailyAnalysis(
            top_stories=top_stories,
            niche_stories=niche_stories,
            junk_stories=junk_stories,
            total_articles=len(df),
            total_clusters=len(unique_labels),
            singletons=sum(1 for c in clusters if c.size == 1),
        )

    def _build_cluster(
        self,
        label: int,
        df: pd.DataFrame,
        embeddings_norm: np.ndarray,
        mask: np.ndarray,
    ) -> TopicCluster:
        """Build a TopicCluster from masked data."""
        # Extract cluster data using positional indexing
        indices = np.where(mask)[0]
        cluster_titles = df["title"].iloc[indices].tolist()
        cluster_descs = df["description"].iloc[indices].tolist()
        cluster_sources = df["source"].iloc[indices].tolist()

        # Get links if available
        cluster_links = []
        if "link" in df.columns:
            cluster_links = df["link"].iloc[indices].tolist()
        else:
            cluster_links = [""] * len(indices)

        # Build article list
        articles = []
        for t, d, s, link in zip(
            cluster_titles, cluster_descs, cluster_sources, cluster_links
        ):
            articles.append(
                {
                    "title": t,
                    "description": d if pd.notna(d) else "",
                    "source": s,
                    "link": link,
                }
            )

        # Find medoid (article closest to cluster center)
        cluster_emb = embeddings_norm[mask]
        medoid_idx = self._find_medoid_index(cluster_emb)
        medoid_title = cluster_titles[medoid_idx]
        medoid_desc = (
            cluster_descs[medoid_idx] if pd.notna(cluster_descs[medoid_idx]) else ""
        )

        # Calculate metrics
        n_sources = len(set(cluster_sources))
        diversity = n_sources / len(cluster_sources)
        importance = len(cluster_titles) * (0.5 + 0.5 * diversity)

        return TopicCluster(
            id=int(label),
            size=len(cluster_titles),
            label=medoid_title,
            description=medoid_desc,
            articles=articles,
            sources=list(set(cluster_sources)),
            diversity_score=diversity,
            importance_score=importance,
        )

    def _find_medoid_index(self, embeddings: np.ndarray) -> int:
        """Find index of the article closest to the cluster centroid."""
        if len(embeddings) == 0:
            return 0
        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        dists = cosine_distances(embeddings, centroid).flatten()
        return int(np.argmin(dists))

def generate_stories_markdown(
    analysis: DailyAnalysis,
    date_str: str,
    max_stories: int = 20,
) -> str:
    """
    Generate a markdown summary of the top stories.

    Args:
        analysis: DailyAnalysis from NewsAnalyzer.analyze()
        date_str: Date string for the header (e.g., "2026-01-18")
        max_stories: Maximum number of stories to include

    Returns:
        Markdown string with clustered stories
    """
    # Setup Jinja2 environment
    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("stories.md.j2")

    return template.render(analysis=analysis, date=date_str, max_stories=max_stories)


def save_stories_markdown(
    analysis: DailyAnalysis,
    output_path: Path,
    date_str: str,
) -> None:
    """
    Generate and save stories markdown file.
    """
    markdown = generate_stories_markdown(analysis, date_str)
    output_path.write_text(markdown)
    print(f"Saved stories to {output_path}")
