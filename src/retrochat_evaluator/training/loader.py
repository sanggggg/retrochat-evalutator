"""Dataset loader for training module."""

import json
import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

from ..models.chat_session import ChatSession

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a single session from manifest."""

    file: str
    scores: dict[str, float]
    metadata: dict[str, Any]

    def get_score(self, score_name: str) -> float | None:
        """Get a specific score by name.

        Args:
            score_name: Name of the score to retrieve.

        Returns:
            The score value, or None if not found.
        """
        return self.scores.get(score_name)


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

    def filter_by_score(self, threshold: float, score_name: str = "default") -> list[SessionInfo]:
        """Return sessions with score >= threshold for the specified score type.

        Args:
            threshold: Minimum score threshold.
            score_name: Name of the score to filter by (default: "default").

        Returns:
            List of SessionInfo for qualified sessions.
        """
        manifest = self.load_manifest()
        qualified = []
        for s in manifest.sessions:
            score_value = s.get_score(score_name)
            if score_value is not None and score_value >= threshold:
                qualified.append(s)

        logger.info(
            f"Filtered {len(qualified)}/{len(manifest.sessions)} sessions "
            f"with {score_name} score >= {threshold}"
        )
        return qualified

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
