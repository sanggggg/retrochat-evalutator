"""Visualization utilities for training results."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ..models.rubric import Rubric

logger = logging.getLogger(__name__)


def save_clustering_visualization(
    result_folder: Path,
    embeddings: np.ndarray,
    clusters: np.ndarray,
    all_rubrics: list[Rubric],
    cluster_info: Optional[dict[int, dict]] = None,
    umap_embeddings: Optional[np.ndarray] = None,
) -> None:
    """Generate and save 2D visualization of clustering results.

    Args:
        result_folder: Output directory for the visualization.
        embeddings: Original high-dimensional embedding vectors.
        clusters: Cluster labels for each rubric.
        all_rubrics: List of all rubrics.
        cluster_info: Optional cluster information dictionary.
        umap_embeddings: Optional UMAP-reduced embeddings (preferred for visualization if available).
    """
    logger.info("Generating clustering visualization...")

    # Prefer UMAP embeddings if available (already dimensionality-reduced and structure-preserving)
    source_embeddings = umap_embeddings if umap_embeddings is not None else embeddings
    method_name = "UMAP" if umap_embeddings is not None else "original"

    logger.info(
        f"Using {method_name} embeddings (shape: {source_embeddings.shape}) for visualization"
    )

    # Reduce dimensions to 2D using t-SNE
    if len(source_embeddings) < 2:
        logger.warning("Not enough data points for visualization (need at least 2)")
        return

    # Use t-SNE for dimensionality reduction
    # For small datasets, use smaller perplexity
    perplexity = min(30, max(5, len(source_embeddings) - 1))

    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        embeddings_2d = tsne.fit_transform(source_embeddings)
        reduction_method = "t-SNE"
    except Exception as e:
        logger.warning(f"t-SNE failed, trying PCA: {e}")
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(source_embeddings)
        reduction_method = "PCA"

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique clusters and assign colors
    unique_clusters = sorted(set(clusters))
    n_clusters = len(unique_clusters)

    # Use a colormap for distinct colors
    colormap_name = "tab20" if n_clusters <= 20 else "tab20b"
    try:
        # Try new matplotlib API (3.5+)
        cmap = plt.colormaps[colormap_name]
    except (AttributeError, KeyError):
        # Fallback to old API
        cmap = plt.cm.get_cmap(colormap_name)
    colors = [cmap(i / max(n_clusters, 1)) for i in range(n_clusters)]

    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        cluster_points = embeddings_2d[mask]

        # Plot points
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[i]],
            label=f"Cluster {cluster_id} (n={len(cluster_points)})",
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidths=0.5,
        )

        # Add cluster label at centroid
        centroid = cluster_points.mean(axis=0)
        ax.annotate(
            f"C{cluster_id}",
            xy=centroid,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
        )

    # Customize plot
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(
        f"Semantic Clustering Visualization (UMAP + HDBSCAN)\n"
        f"{len(all_rubrics)} rubrics grouped into {n_clusters} clusters\n"
        f"Dimensionality reduction: {method_name} → {reduction_method}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # Save figure
    plt.tight_layout()
    visualization_path = result_folder / "clustering_visualization.png"
    fig.savefig(visualization_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved clustering visualization to {visualization_path}")
