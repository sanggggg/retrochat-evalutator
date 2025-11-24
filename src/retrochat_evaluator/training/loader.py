"""Dataset loader for training module."""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..models.chat_session import ChatSession

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a single session from manifest."""

    file: str
    scores: dict[str, float]
    metadata: dict[str, Any]
    split: Optional[str] = None  # "training" or "validation"
    score_percentiles: Optional[dict[str, float]] = None

    def get_score(self, score_name: str) -> float | None:
        """Get a specific score by name.

        Args:
            score_name: Name of the score to retrieve.

        Returns:
            The score value, or None if not found.
        """
        return self.scores.get(score_name)

    def get_percentile(self, score_name: str) -> float | None:
        """Get percentile rank (0-100] for the given score if available."""
        if not self.score_percentiles:
            return None
        return self.score_percentiles.get(score_name)


@dataclass
class DatasetManifest:
    """Parsed dataset manifest."""

    sessions: list[SessionInfo]

    @classmethod
    def from_json(cls, path: Path) -> "DatasetManifest":
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sessions = []
        for session_data in data.get("sessions", []):
            # Support both legacy "score" (float) and new "scores" (dict) format
            scores_data = session_data.get("scores")
            if scores_data is None:
                # Legacy format: convert single score to dict with "default" key
                legacy_score = session_data.get("score")
                scores_data = {"default": legacy_score} if legacy_score is not None else {}

            sessions.append(
                SessionInfo(
                    file=session_data["file"],
                    scores=scores_data,
                    metadata=session_data.get("metadata", {}),
                    split=session_data.get("split"),  # "training" or "validation"
                    score_percentiles=session_data.get("score_percentiles"),
                )
            )

        return cls(sessions=sessions)


class DatasetLoader:
    """Load and filter chat sessions from dataset manifest."""

    def __init__(self, dataset_dir: Path, manifest_path: Path):
        """Initialize dataset loader.

        Args:
            dataset_dir: Directory containing JSONL session files.
            manifest_path: Path to the dataset manifest JSON file.
        """
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = Path(manifest_path)
        self._manifest: Optional[DatasetManifest] = None

    def load_manifest(self) -> DatasetManifest:
        """Load and cache the dataset manifest."""
        if self._manifest is None:
            self._manifest = DatasetManifest.from_json(self.manifest_path)
            logger.info(f"Loaded manifest with {len(self._manifest.sessions)} sessions")
        return self._manifest

    def filter_by_score(
        self, threshold: float, score_name: str = "default", split: Optional[str] = None
    ) -> list[SessionInfo]:
        """Return sessions with score >= threshold for the specified score type.

        Args:
            threshold: Minimum score threshold.
            score_name: Name of the score to filter by (default: "default").
            split: Optional split filter ("training" or "validation"). If None, includes all splits.

        Returns:
            List of SessionInfo for qualified sessions.
        """
        manifest = self.load_manifest()
        qualified = []
        for s in manifest.sessions:
            # Filter by split if specified
            if split is not None and s.split != split:
                continue

            score_value = s.get_score(score_name)
            if score_value is not None and score_value >= threshold:
                qualified.append(s)

        split_info = f" (split={split})" if split else ""
        logger.info(
            f"Filtered {len(qualified)}/{len(manifest.sessions)} sessions "
            f"with {score_name} score >= {threshold}{split_info}"
        )
        return qualified

    def filter_by_percentile(
        self, top_percentile: float, score_name: str = "default", split: Optional[str] = None
    ) -> list[SessionInfo]:
        """Return sessions that fall within the top percentile for the specified score."""
        manifest = self.load_manifest()
        if top_percentile <= 0:
            logger.warning("top_percentile must be > 0 to filter by percentile")
            return []

        capped_percentile = min(top_percentile, 100.0)
        candidate_scores: list[tuple[SessionInfo, float]] = []
        for session in manifest.sessions:
            if split is not None and session.split != split:
                continue
            score_value = session.get_score(score_name)
            if score_value is not None:
                candidate_scores.append((session, float(score_value)))

        if not candidate_scores:
            logger.warning(
                f"No sessions found with score '{score_name}' for percentile filtering (split={split})"
            )
            return []

        candidate_scores.sort(key=lambda item: item[1], reverse=True)
        total = len(candidate_scores)
        cutoff_count = max(1, math.ceil(total * (capped_percentile / 100.0)))
        cutoff_score = candidate_scores[cutoff_count - 1][1]

        qualified = [session for session, value in candidate_scores if value >= cutoff_score]

        split_info = f" (split={split})" if split else ""
        logger.info(
            f"Filtered {len(qualified)}/{len(manifest.sessions)} sessions "
            f"within top {capped_percentile}% for {score_name}{split_info}"
        )
        return qualified

    def filter_by_split(self, split: str) -> list[SessionInfo]:
        """Return sessions for the specified split.

        Args:
            split: Split name ("training" or "validation").

        Returns:
            List of SessionInfo for the specified split.
        """
        manifest = self.load_manifest()
        filtered = [s for s in manifest.sessions if s.split == split]
        logger.info(f"Filtered {len(filtered)}/{len(manifest.sessions)} sessions for split={split}")
        return filtered

    def load_session(self, session_info: SessionInfo) -> ChatSession:
        """Parse session file into ChatSession model.

        Supports both JSON (.json) and JSONL (.jsonl) formats.

        Args:
            session_info: Session info from manifest.

        Returns:
            Parsed ChatSession object.
        """
        session_path = self.dataset_dir / session_info.file
        logger.debug(f"Loading session from {session_path}")

        if session_path.suffix == ".json":
            return ChatSession.from_json(session_path)
        else:
            return ChatSession.from_jsonl(session_path)

    def load_sessions(
        self,
        session_infos: list[SessionInfo],
        max_sessions: Optional[int] = None,
    ) -> list[ChatSession]:
        """Load multiple sessions.

        Args:
            session_infos: List of session info objects.
            max_sessions: Maximum number of sessions to load.

        Returns:
            List of parsed ChatSession objects.
        """
        if max_sessions:
            session_infos = session_infos[:max_sessions]

        sessions = []
        for info in session_infos:
            try:
                session = self.load_session(info)
                sessions.append(session)
            except Exception as e:
                logger.warning(f"Failed to load session {info.file}: {e}")

        logger.info(f"Successfully loaded {len(sessions)} sessions")
        return sessions

    def get_session_count(self) -> int:
        """Get total number of sessions in manifest."""
        manifest = self.load_manifest()
        return len(manifest.sessions)
