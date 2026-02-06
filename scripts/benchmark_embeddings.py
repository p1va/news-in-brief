#!/usr/bin/env python3
"""
Embedding model benchmark: compare clustering quality across embedding models.

Sweeps HAC thresholds, computes intrinsic + task-specific metrics, suggests
an optimal threshold per model, and optionally compares two models side-by-side.

Usage:
    # Single model: find optimal threshold
    uv run python scripts/benchmark_embeddings.py <parquet_with_embeddings>

    # Compare two models
    uv run python scripts/benchmark_embeddings.py <parquet_a> <parquet_b>

    # Custom sweep range
    uv run python scripts/benchmark_embeddings.py <parquet_a> --sweep-min 0.15 --sweep-max 0.60 --sweep-step 0.01
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from scripts/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize

from core.analyzer import NewsAnalyzer
from core.sources_view import (
    generate_sources_view,
    load_embeddings,
)


# ---------------------------------------------------------------------------
# Distance distribution
# ---------------------------------------------------------------------------

def analyze_distances(embeddings: np.ndarray, n_sample: int = 500) -> dict:
    """Sample pairwise cosine distances, return percentiles."""
    n = min(n_sample, len(embeddings))
    idx = np.random.default_rng(42).choice(len(embeddings), n, replace=False)
    sample = normalize(embeddings[idx])
    dists = cosine_distances(sample).ravel()
    dists = dists[dists > 0]  # drop self-distances
    return {
        "min": float(np.min(dists)),
        "p5": float(np.percentile(dists, 5)),
        "p10": float(np.percentile(dists, 10)),
        "p25": float(np.percentile(dists, 25)),
        "median": float(np.median(dists)),
        "p75": float(np.percentile(dists, 75)),
        "p90": float(np.percentile(dists, 90)),
        "max": float(np.max(dists)),
    }


def print_distances(stats: dict, name: str) -> None:
    print(f"\n  Distance distribution ({name}):")
    for k, v in stats.items():
        print(f"    {k:>6s}: {v:.4f}")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Run HAC at each threshold and collect metrics."""
    embeddings_norm = normalize(embeddings)
    dist_matrix = cosine_distances(embeddings_norm)

    rows = []
    for t in thresholds:
        hac = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=t,
            metric="cosine",
            linkage="average",
        )
        labels = hac.fit_predict(embeddings_norm)

        n_clusters = len(set(labels))
        n_singletons = sum(1 for l in set(labels) if np.sum(labels == l) == 1)
        pct_clustered = 100 * (1 - n_singletons / len(labels))

        # Silhouette (needs >= 2 clusters with >= 2 members)
        multi_mask = np.array([np.sum(labels == l) > 1 for l in labels])
        n_multi_clusters = len(set(labels[multi_mask]))
        if n_multi_clusters >= 2 and multi_mask.sum() > n_multi_clusters:
            sil = silhouette_score(dist_matrix[np.ix_(multi_mask, multi_mask)],
                                   labels[multi_mask], metric="precomputed")
        else:
            sil = float("nan")

        # Mean intra-cluster cosine similarity
        intra_sims = []
        for l in set(labels):
            members = np.where(labels == l)[0]
            if len(members) < 2:
                continue
            sim_block = cosine_similarity(embeddings_norm[members])
            # upper triangle only
            triu = sim_block[np.triu_indices(len(members), k=1)]
            intra_sims.append(float(np.mean(triu)))
        mean_intra_sim = float(np.mean(intra_sims)) if intra_sims else float("nan")

        # Task metrics via NewsAnalyzer
        analyzer = NewsAnalyzer(threshold=t)
        analysis = analyzer.analyze(df, embeddings)
        n_top = len(analysis.top_stories)
        n_junk = len(analysis.junk_stories)
        if n_top > 0:
            mean_top_div = float(np.mean([s.diversity_score for s in analysis.top_stories]))
        else:
            mean_top_div = 0.0
        task_score = n_top * mean_top_div

        rows.append({
            "threshold": t,
            "n_clusters": n_clusters,
            "n_singletons": n_singletons,
            "pct_clustered": round(pct_clustered, 1),
            "silhouette": round(sil, 4) if not np.isnan(sil) else None,
            "mean_intra_sim": round(mean_intra_sim, 4) if not np.isnan(mean_intra_sim) else None,
            "n_top_stories": n_top,
            "n_junk": n_junk,
            "mean_top_diversity": round(mean_top_div, 3),
            "task_score": round(task_score, 2),
        })

    return pd.DataFrame(rows)


def suggest_threshold(sweep_df: pd.DataFrame) -> tuple[float, str]:
    """
    Find the optimal threshold.

    1. Filter to thresholds where meaningful clustering occurs (pct_clustered >= 20%).
    2. Find silhouette peak within that range.
    3. Define quality band: silhouette >= 90% of peak.
    4. Within that band, pick threshold maximizing task_score.
    """
    valid = sweep_df.dropna(subset=["silhouette"])

    # Filter out degenerate thresholds where barely anything clusters
    valid = valid[valid["pct_clustered"] >= 20.0]

    if valid.empty:
        # Fallback: just max task_score
        best_idx = sweep_df["task_score"].idxmax()
        t = sweep_df.loc[best_idx, "threshold"]
        return t, f"fallback (no valid silhouette with pct_clustered>=20%); max task_score={sweep_df.loc[best_idx, 'task_score']}"

    peak_sil = valid["silhouette"].max()
    band = valid[valid["silhouette"] >= 0.75 * peak_sil]

    if band.empty:
        band = valid

    best_idx = band["task_score"].idxmax()
    t = band.loc[best_idx, "threshold"]
    sil = band.loc[best_idx, "silhouette"]
    ts = band.loc[best_idx, "task_score"]
    reason = (
        f"silhouette peak={peak_sil:.4f} (pct_clustered>=20%), "
        f"quality band=[{band['threshold'].min():.2f}, {band['threshold'].max():.2f}], "
        f"chosen silhouette={sil:.4f}, task_score={ts:.2f}"
    )
    return t, reason


def print_sweep(sweep_df: pd.DataFrame, name: str) -> None:
    print(f"\n{'='*100}")
    print(f"  Threshold sweep: {name}")
    print(f"{'='*100}")
    print(sweep_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Comparison metrics (two-model mode)
# ---------------------------------------------------------------------------

def get_labels_at_threshold(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Return HAC labels at a given threshold."""
    embeddings_norm = normalize(embeddings)
    hac = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    return hac.fit_predict(embeddings_norm)


def compare_clusterings(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    df: pd.DataFrame,
    name_a: str,
    name_b: str,
) -> None:
    """Print ARI, NMI, and disagreement analysis."""
    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)

    print(f"\n{'='*80}")
    print(f"  Clustering agreement: {name_a} vs {name_b}")
    print(f"{'='*80}")
    print(f"  Adjusted Rand Index (ARI):            {ari:.4f}  (1.0 = identical)")
    print(f"  Normalized Mutual Information (NMI):   {nmi:.4f}  (1.0 = identical)")

    # ------- Disagreement analysis -------
    # Build cluster -> article-indices maps
    def cluster_map(labels):
        m = defaultdict(set)
        for i, l in enumerate(labels):
            m[l].add(i)
        return m

    map_a = cluster_map(labels_a)
    map_b = cluster_map(labels_b)

    # Find splits: A groups them, B separates them
    splits = []
    for cluster_id_a, members_a in map_a.items():
        if len(members_a) < 2:
            continue
        b_labels_for_a = set(labels_b[i] for i in members_a)
        if len(b_labels_for_a) > 1:
            splits.append((cluster_id_a, members_a, b_labels_for_a))

    # Find merges: B groups them, A separates them
    merges = []
    for cluster_id_b, members_b in map_b.items():
        if len(members_b) < 2:
            continue
        a_labels_for_b = set(labels_a[i] for i in members_b)
        if len(a_labels_for_b) > 1:
            merges.append((cluster_id_b, members_b, a_labels_for_b))

    # Sort by size of disagreement
    splits.sort(key=lambda x: len(x[1]), reverse=True)
    merges.sort(key=lambda x: len(x[1]), reverse=True)

    titles = df["title"].tolist()

    print(f"\n  Top splits ({name_a} groups together, {name_b} separates):")
    if not splits:
        print("    (none)")
    for cluster_id_a, members, b_labels in splits[:5]:
        print(f"    {name_a} cluster {cluster_id_a} ({len(members)} articles) -> {len(b_labels)} clusters in {name_b}")
        sample_indices = sorted(members)[:3]
        for idx in sample_indices:
            print(f"      [{name_b} c{labels_b[idx]}] {titles[idx][:80]}")

    print(f"\n  Top merges ({name_b} groups together, {name_a} separates):")
    if not merges:
        print("    (none)")
    for cluster_id_b, members, a_labels in merges[:5]:
        print(f"    {name_b} cluster {cluster_id_b} ({len(members)} articles) -> {len(a_labels)} clusters in {name_a}")
        sample_indices = sorted(members)[:3]
        for idx in sample_indices:
            print(f"      [{name_a} c{labels_a[idx]}] {titles[idx][:80]}")


def print_top_stories_side_by_side(
    df: pd.DataFrame,
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    name_a: str,
    name_b: str,
    n: int = 10,
) -> None:
    """Print top N stories from each model for quick comparison."""
    analyzer_a = NewsAnalyzer(threshold=threshold_a)
    analysis_a = analyzer_a.analyze(df, embeddings_a)

    analyzer_b = NewsAnalyzer(threshold=threshold_b)
    analysis_b = analyzer_b.analyze(df, embeddings_b)

    print(f"\n{'='*80}")
    print(f"  Top {n} stories: {name_a} (threshold={threshold_a:.2f})")
    print(f"{'='*80}")
    for i, story in enumerate(analysis_a.top_stories[:n], 1):
        print(f"  {i:>2}. [{story.size} articles, {len(story.sources)} sources, div={story.diversity_score:.0%}]")
        print(f"      {story.label[:80]}")

    print(f"\n{'='*80}")
    print(f"  Top {n} stories: {name_b} (threshold={threshold_b:.2f})")
    print(f"{'='*80}")
    for i, story in enumerate(analysis_b.top_stories[:n], 1):
        print(f"  {i:>2}. [{story.size} articles, {len(story.sources)} sources, div={story.diversity_score:.0%}]")
        print(f"      {story.label[:80]}")


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_benchmark_html(
    parquet_path: Path,
    output_dir: Path,
    threshold: float,
    model_name: str,
) -> Path:
    """Generate sources-view HTML at the suggested threshold."""
    # Infer date from filename (e.g. 2026-02-05-sources-with-embeddings.parquet)
    stem = parquet_path.stem
    date_str = stem[:10]  # YYYY-MM-DD

    output_path = output_dir / f"{date_str}-benchmark-{model_name}.html"
    generate_sources_view(
        embeddings_path=parquet_path,
        output_path=output_path,
        reference_date=date_str,
        threshold=threshold,
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def infer_model_name(parquet_path: Path) -> str:
    """Infer model name from filename."""
    stem = parquet_path.stem.lower()
    if "voyage" in stem:
        return "voyage"
    if "openai" in stem:
        return "openai"
    # Default: the part after "-with-embeddings-" or just "openai"
    marker = "-with-embeddings-"
    if marker in stem:
        return stem.split(marker, 1)[1]
    return "openai"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models for news clustering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("parquet_a", type=Path, help="Parquet with embeddings (model A)")
    parser.add_argument("parquet_b", type=Path, nargs="?", default=None,
                        help="Optional second parquet (model B) for comparison")
    parser.add_argument("--name-a", type=str, default=None, help="Display name for model A")
    parser.add_argument("--name-b", type=str, default=None, help="Display name for model B")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for HTML output (default: parquet_a's parent)")
    parser.add_argument("--sweep-min", type=float, default=0.10)
    parser.add_argument("--sweep-max", type=float, default=0.70)
    parser.add_argument("--sweep-step", type=float, default=0.02)
    args = parser.parse_args()

    if not args.parquet_a.exists():
        print(f"Error: {args.parquet_a} not found")
        sys.exit(1)
    if args.parquet_b and not args.parquet_b.exists():
        print(f"Error: {args.parquet_b} not found")
        sys.exit(1)

    output_dir = args.output_dir or args.parquet_a.parent
    name_a = args.name_a or infer_model_name(args.parquet_a)
    name_b: str = args.name_b or (infer_model_name(args.parquet_b) if args.parquet_b else "model_b")

    thresholds = [
        round(args.sweep_min + i * args.sweep_step, 4)
        for i in range(int((args.sweep_max - args.sweep_min) / args.sweep_step) + 1)
    ]

    # ---- Model A ----
    print(f"Loading {args.parquet_a} ...")
    df_a, emb_a = load_embeddings(args.parquet_a)
    print(f"  {len(df_a)} articles, embedding dim={emb_a.shape[1]}")

    dist_a = analyze_distances(emb_a)
    print_distances(dist_a, name_a)

    print(f"\n  Running threshold sweep ({len(thresholds)} steps) ...")
    sweep_a = sweep_thresholds(df_a, emb_a, thresholds)
    print_sweep(sweep_a, name_a)

    thresh_a, reason_a = suggest_threshold(sweep_a)
    print(f"\n  Suggested threshold for {name_a}: {thresh_a:.2f}")
    print(f"    ({reason_a})")

    html_a = generate_benchmark_html(args.parquet_a, output_dir, thresh_a, name_a)
    print(f"  Generated HTML: {html_a}")

    # ---- Model B (optional) ----
    if args.parquet_b:
        print(f"\nLoading {args.parquet_b} ...")
        df_b, emb_b = load_embeddings(args.parquet_b)
        print(f"  {len(df_b)} articles, embedding dim={emb_b.shape[1]}")

        # Sanity check: same articles
        if len(df_a) != len(df_b):
            print(f"WARNING: different article counts ({len(df_a)} vs {len(df_b)})")

        dist_b = analyze_distances(emb_b)
        print_distances(dist_b, name_b)

        print(f"\n  Running threshold sweep ({len(thresholds)} steps) ...")
        sweep_b = sweep_thresholds(df_b, emb_b, thresholds)
        print_sweep(sweep_b, name_b)

        thresh_b, reason_b = suggest_threshold(sweep_b)
        print(f"\n  Suggested threshold for {name_b}: {thresh_b:.2f}")
        print(f"    ({reason_b})")

        html_b = generate_benchmark_html(args.parquet_b, output_dir, thresh_b, name_b)
        print(f"  Generated HTML: {html_b}")

        # ---- Comparison ----
        labels_a = get_labels_at_threshold(emb_a, thresh_a)
        labels_b = get_labels_at_threshold(emb_b, thresh_b)

        compare_clusterings(labels_a, labels_b, df_a, name_a, name_b)

        print_top_stories_side_by_side(
            df_a, emb_a, emb_b, thresh_a, thresh_b, name_a, name_b,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
