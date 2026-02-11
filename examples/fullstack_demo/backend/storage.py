"""Storage abstraction layer for game runs.

Supports both local JSON storage and Supabase backend.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def save_run(
        self,
        run_id: str,
        user_id: str,
        mode: str,
        guidance_frequency: Optional[int],
        final_reward: float,
        log_text: str,
    ) -> bool:
        """Save a completed game run.

        Args:
            run_id: Unique run identifier
            user_id: User identifier (UUID string)
            mode: Game mode (Mode 1 or Mode 2)
            guidance_frequency: Guidance frequency for Mode 2 (if applicable)
            final_reward: Final reward/profit from the game
            log_text: Full transcript/log of the game

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    async def get_user_runs(self, user_id: str) -> list[Dict[str, Any]]:
        """Get all runs for a specific user.

        Args:
            user_id: User identifier

        Returns:
            List of run records
        """
        pass


class JSONStorage(StorageBackend):
    """Local JSON file storage backend."""

    def __init__(self, data_dir: Path = None):
        """Initialize JSON storage.

        Args:
            data_dir: Directory to store game_runs.json (defaults to ./data)
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "game_runs.json"

        # Initialize file if it doesn't exist
        if not self.file_path.exists():
            self.file_path.write_text(json.dumps([], indent=2))
            logger.info(f"Created new game_runs.json at {self.file_path}")

    def _load_data(self) -> list[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            content = self.file_path.read_text()
            return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not load {self.file_path}, returning empty list")
            return []

    def _save_data(self, data: list[Dict[str, Any]]) -> None:
        """Save data to JSON file."""
        self.file_path.write_text(json.dumps(data, indent=2))

    async def save_run(
        self,
        run_id: str,
        user_id: str,
        mode: str,
        guidance_frequency: Optional[int],
        final_reward: float,
        log_text: str,
    ) -> bool:
        """Save a game run to JSON file."""
        try:
            runs = self._load_data()

            run_record = {
                "id": run_id,
                "user_id": user_id,
                "mode": mode,
                "guidance_frequency": guidance_frequency,
                "final_reward": float(final_reward),
                "log_text": log_text,
                "created_at": str(__import__("datetime").datetime.utcnow().isoformat()),
            }

            runs.append(run_record)
            self._save_data(runs)

            logger.info(f"Saved run {run_id} to JSON storage")
            return True
        except Exception as e:
            logger.error(f"Error saving run to JSON: {e}")
            return False

    async def get_user_runs(self, user_id: str) -> list[Dict[str, Any]]:
        """Get all runs for a user from JSON file."""
        try:
            runs = self._load_data()
            user_runs = [run for run in runs if run.get("user_id") == user_id]
            return user_runs
        except Exception as e:
            logger.error(f"Error retrieving user runs from JSON: {e}")
            return []


class SupabaseStorage(StorageBackend):
    """Supabase cloud storage backend."""

    def __init__(self):
        """Initialize Supabase storage.

        Requires environment variables:
            - SUPABASE_URL
            - SUPABASE_SERVICE_ROLE_KEY
        """
        try:
            from supabase import create_client
        except ImportError:
            raise ImportError(
                "supabase-py is required for Supabase storage. "
                "Install with: pip install supabase"
            )

        supabase_url = os.getenv("SUPABASE_URL")
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not service_role_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables required"
            )

        self.client = create_client(supabase_url, service_role_key)
        logger.info("Initialized Supabase storage")

    async def save_run(
        self,
        run_id: str,
        user_id: str,
        mode: str,
        guidance_frequency: Optional[int],
        final_reward: float,
        log_text: str,
    ) -> bool:
        """Save a game run to Supabase."""
        try:
            data = {
                "run_id": run_id,
                "user_id": user_id,
                "mode": mode,
                "guidance_frequency": guidance_frequency,
                "final_reward": float(final_reward),
                "log_text": log_text,
            }

            response = self.client.table("game_runs").insert(data).execute()
            logger.info(f"Saved run {run_id} to Supabase")
            return True
        except Exception as e:
            logger.error(f"Error saving run to Supabase: {e}")
            return False

    async def get_user_runs(self, user_id: str) -> list[Dict[str, Any]]:
        """Get all runs for a user from Supabase."""
        try:
            response = (
                self.client.table("game_runs")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"Error retrieving user runs from Supabase: {e}")
            return []


def get_storage_backend() -> StorageBackend:
    """Factory function to get the appropriate storage backend.

    Uses environment variables to determine which backend to use:
        - USE_LOCAL_STORAGE=true → JSONStorage
        - Otherwise → SupabaseStorage (requires Supabase env vars)

    Returns:
        StorageBackend instance

    Raises:
        ValueError: If configuration is invalid
    """
    use_local = os.getenv("USE_LOCAL_STORAGE", "").lower() in ("true", "1", "yes")

    if use_local:
        logger.info("Using local JSON storage")
        return JSONStorage()
    else:
        logger.info("Using Supabase storage")
        return SupabaseStorage()
