"""Semantic clustering based rubric summarizer using Google AI embeddings."""

import logging
import os
from typing import Optional

import hdbscan
import numpy as np
import umap.umap_ as umap
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from ..models.rubric import Rubric

logger = logging.getLogger(__name__)


class SemanticClusteringSummarizer:
    """Consolidate rubrics using semantic embedding and clustering.

    This approach uses:
    1. Text embedding via Google AI (gemini-embedding-001) to convert rubrics to vectors
    2. UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
    3. HDBSCAN (Hierarchical Density-Based Spatial Clustering) to group similar rubrics
    4. Cluster size as frequency metric
    5. Centroid-based representative selection
    """

    def __init__(
        self,
        embedding_model: str = "models/gemini-embedding-001",
        umap_n_neighbors: int = 15,
        umap_n_components: int = 5,
        umap_metric: str = "cosine",
        min_cluster_size: int = 2,
        min_rubrics: int = 5,
        max_rubrics: int = 10,
        api_key: Optional[str] = None,
    ):
        """Initialize semantic clustering summarizer.

        Args:
            embedding_model: Google AI embedding model name.
            umap_n_neighbors: Number of neighbors for UMAP. Controls local vs global structure.
                Lower values focus on local structure, higher on global.
            umap_n_components: Target dimensionality for UMAP reduction.
            umap_metric: Distance metric for UMAP (default: cosine for semantic similarity).
            min_cluster_size: Minimum cluster size for HDBSCAN.
            min_rubrics: Minimum number of rubrics to output.
            max_rubrics: Maximum number of rubrics to output.
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
        """
        self.embedding_model = embedding_model
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.umap_metric = umap_metric
        self.min_cluster_size = min_cluster_size
        self.min_rubrics = min_rubrics
        self.max_rubrics = max_rubrics
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None

        # Store clustering results for visualization
        self.last_embeddings: Optional[np.ndarray] = None
        self.last_umap_embeddings: Optional[np.ndarray] = None
        self.last_clusters: Optional[np.ndarray] = None
        self.last_all_rubrics: Optional[list[Rubric]] = None
        self.last_cluster_info: Optional[dict[int, dict]] = None

    def _get_embeddings_client(self) -> GoogleGenerativeAIEmbeddings:
        """Lazy load the Google AI embeddings client."""
        if self._embeddings is None:
            logger.info(f"Initializing Google AI embeddings: {self.embedding_model}")
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=self.api_key,
            )
        return self._embeddings

    async def summarize(self, rubric_lists: list[list[Rubric]]) -> tuple[list[Rubric], str]:
        """Consolidate rubrics using semantic clustering.

        Args:
            rubric_lists: List of rubric lists from multiple sessions.

        Returns:
            Tuple of (final rubrics list, consolidation notes).
        """
        # Flatten all rubrics
        all_rubrics = self._flatten_rubrics(rubric_lists)

        if not all_rubrics:
            logger.warning("No rubrics to cluster")
            return [], "No rubrics provided for clustering."

        total_rubrics = len(all_rubrics)
        logger.info(f"Clustering {total_rubrics} rubrics from {len(rubric_lists)} sessions")

        if total_rubrics < self.min_rubrics:
            logger.info(f"Not enough rubrics to cluster ({total_rubrics}), returning all")
            return (
                self._deduplicate_rubrics(all_rubrics),
                f"Only {total_rubrics} rubrics, no clustering needed.",
            )

        # Step 1: Generate embeddings
        texts = [self._rubric_to_text(r) for r in all_rubrics]
        embeddings = await self._generate_embeddings(texts)

        # Step 2: Perform clustering
        clusters, umap_embeddings = self._cluster_embeddings(embeddings)

        # Step 3: Calculate cluster statistics
        cluster_info = self._analyze_clusters(clusters, all_rubrics, embeddings)

        # Step 4: Select top k clusters and get representatives
        final_rubrics = self._select_representatives(cluster_info, all_rubrics, embeddings)

        # Store clustering results for visualization
        self.last_embeddings = embeddings
        self.last_umap_embeddings = umap_embeddings
        self.last_clusters = clusters
        self.last_all_rubrics = all_rubrics
        self.last_cluster_info = cluster_info

        # Generate consolidation notes
        notes = self._generate_notes(total_rubrics, len(final_rubrics), cluster_info)

        logger.info(f"Clustered {total_rubrics} rubrics into {len(final_rubrics)} final rubrics")
        return final_rubrics, notes

    def _flatten_rubrics(self, rubric_lists: list[list[Rubric]]) -> list[Rubric]:
        """Flatten nested rubric lists into single list."""
        return [rubric for rubrics in rubric_lists for rubric in rubrics if rubrics]

    def _rubric_to_text(self, rubric: Rubric) -> str:
        """Convert rubric to text for embedding.

        Combines name, description, and scoring criteria into a single text.
        """
        parts = [
            f"Name: {rubric.name}",
            f"Description: {rubric.description}",
        ]
        if rubric.scoring_criteria:
            parts.append(f"Scoring: {rubric.scoring_criteria}")
        return " ".join(parts)

    async def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for all texts using Google AI."""
        client = self._get_embeddings_client()
        logger.info(f"Generating embeddings for {len(texts)} rubrics via Google AI")

        # Use embed_documents for batch embedding
        embeddings = await client.aembed_documents(texts)
        return np.array(embeddings)

    def _cluster_embeddings(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cluster embeddings using UMAP + HDBSCAN.

        First reduces dimensionality with UMAP, then clusters with HDBSCAN.

        Returns:
            Tuple of (cluster_labels, umap_embeddings)
        """
        n_samples = len(embeddings)

        # Step 1: Dimensionality reduction with UMAP
        logger.info(
            f"Reducing dimensionality with UMAP: "
            f"n_neighbors={self.umap_n_neighbors}, "
            f"n_components={self.umap_n_components}, "
            f"metric={self.umap_metric}"
        )

        # Adjust n_neighbors if we have fewer samples
        n_neighbors = min(self.umap_n_neighbors, n_samples - 1)
        if n_neighbors < 2:
            n_neighbors = 2

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=self.umap_n_components,
            metric=self.umap_metric,
            random_state=42,
        )
        umap_embeddings = reducer.fit_transform(embeddings)

        # Step 2: Clustering with HDBSCAN
        logger.info(f"Clustering with HDBSCAN: " f"min_cluster_size={self.min_cluster_size}")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        clusters = clusterer.fit_predict(umap_embeddings)

        # Count clusters (excluding noise points labeled as -1)
        unique_clusters = set(clusters)
        n_clusters = len([c for c in unique_clusters if c >= 0])
        n_noise = np.sum(clusters == -1)

        logger.info(
            f"Found {n_clusters} clusters and {n_noise} noise points from {n_samples} rubrics"
        )

        return clusters, umap_embeddings

    def _analyze_clusters(
        self,
        clusters: np.ndarray,
        rubrics: list[Rubric],
        embeddings: np.ndarray,
    ) -> dict[int, dict]:
        """Analyze each cluster for size and centroid.

        Returns:
            Dictionary mapping cluster_id to cluster info including:
            - size: number of rubrics in cluster
            - indices: list of rubric indices in cluster
            - centroid: mean embedding of cluster
            - representative_idx: index of rubric closest to centroid

        Note: DBSCAN noise points (cluster_id == -1) are treated as individual clusters.
        """
        cluster_info = {}
        unique_clusters = set(clusters)

        # Assign new cluster IDs to noise points (-1)
        # Each noise point becomes its own cluster
        max_cluster_id = (
            max(c for c in unique_clusters if c >= 0)
            if any(c >= 0 for c in unique_clusters)
            else -1
        )
        noise_indices = np.where(clusters == -1)[0]
        for i, noise_idx in enumerate(noise_indices):
            new_cluster_id = max_cluster_id + 1 + i
            cluster_info[new_cluster_id] = {
                "size": 1,
                "indices": [noise_idx],
                "centroid": embeddings[noise_idx],
                "representative_idx": noise_idx,
                "max_similarity": 1.0,
            }

        for cluster_id in unique_clusters:
            # Skip noise points (already handled above)
            if cluster_id == -1:
                continue
            # Get indices of rubrics in this cluster
            indices = np.where(clusters == cluster_id)[0].tolist()

            # Calculate centroid
            cluster_embeddings = embeddings[indices]
            centroid = np.mean(cluster_embeddings, axis=0)

            # Find representative (closest to centroid)
            similarities = cosine_similarity([centroid], cluster_embeddings)[0]
            representative_local_idx = np.argmax(similarities)
            representative_idx = indices[representative_local_idx]

            cluster_info[cluster_id] = {
                "size": len(indices),
                "indices": indices,
                "centroid": centroid,
                "representative_idx": representative_idx,
                "max_similarity": similarities[representative_local_idx],
            }

        return cluster_info

    def _select_representatives(
        self,
        cluster_info: dict[int, dict],
        rubrics: list[Rubric],
        embeddings: np.ndarray,
    ) -> list[Rubric]:
        """Select representative rubrics from top clusters.

        Selects the top k clusters by size, then returns the representative
        rubric from each cluster.
        """
        # Sort clusters by size (descending)
        sorted_clusters = sorted(
            cluster_info.items(),
            key=lambda x: x[1]["size"],
            reverse=True,
        )

        # Determine number of clusters to select
        n_select = min(self.max_rubrics, len(sorted_clusters))
        n_select = max(n_select, min(self.min_rubrics, len(sorted_clusters)))

        logger.info(f"Selecting top {n_select} clusters by frequency")

        final_rubrics = []
        for i, (cluster_id, info) in enumerate(sorted_clusters[:n_select]):
            rep_idx = info["representative_idx"]
            original_rubric = rubrics[rep_idx]

            # Create new rubric with updated ID
            new_rubric = Rubric(
                id=f"rubric_{i+1:03d}",
                name=original_rubric.name,
                description=original_rubric.description,
                scoring_criteria=original_rubric.scoring_criteria,
                weight=1.0,
                evidence=None,
            )
            final_rubrics.append(new_rubric)

            logger.debug(
                f"Cluster {cluster_id}: size={info['size']}, "
                f"representative='{original_rubric.name}'"
            )

        return final_rubrics

    def _deduplicate_rubrics(self, rubrics: list[Rubric]) -> list[Rubric]:
        """Simple deduplication for small rubric sets based on name."""
        seen_names = set()
        unique_rubrics = []

        for rubric in rubrics:
            normalized_name = rubric.name.lower().strip()
            if normalized_name not in seen_names:
                seen_names.add(normalized_name)
                new_rubric = Rubric(
                    id=f"rubric_{len(unique_rubrics)+1:03d}",
                    name=rubric.name,
                    description=rubric.description,
                    scoring_criteria=rubric.scoring_criteria,
                    weight=1.0,
                    evidence=None,
                )
                unique_rubrics.append(new_rubric)

        return unique_rubrics[: self.max_rubrics]

    def _generate_notes(
        self,
        total_input: int,
        total_output: int,
        cluster_info: dict[int, dict],
    ) -> str:
        """Generate consolidation notes describing the clustering process."""
        # Get top clusters by size
        sorted_clusters = sorted(
            cluster_info.items(),
            key=lambda x: x[1]["size"],
            reverse=True,
        )

        cluster_sizes = [info["size"] for _, info in sorted_clusters[:total_output]]

        notes_parts = [
            f"Semantic clustering consolidation:",
            f"- Input: {total_input} rubrics from extraction",
            f"- Output: {total_output} clustered rubrics",
            f"- Clustering method: UMAP + HDBSCAN",
            f"- UMAP parameters: n_neighbors={self.umap_n_neighbors}, n_components={self.umap_n_components}, metric={self.umap_metric}",
            f"- HDBSCAN min_cluster_size: {self.min_cluster_size}",
            f"- Embedding model: {self.embedding_model}",
            f"- Total clusters found: {len(cluster_info)}",
            f"- Selected cluster sizes: {cluster_sizes}",
        ]

        return "\n".join(notes_parts)
