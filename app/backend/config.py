"""Application configuration management."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Application configuration from environment variables."""

    # Server configuration
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    RELOAD: bool = os.getenv("FULLSTACK_DEMO_RELOAD", "1") not in ("0", "false", "False")

    # OpenAI configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Storage configuration
    USE_LOCAL_STORAGE: bool = os.getenv("USE_LOCAL_STORAGE", "").lower() in (
        "true",
        "1",
        "yes",
    )
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))

    # Supabase configuration (optional)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")

    # Demand data configuration
    DEMAND_DATA_PATH: str = os.getenv(
        "DEMAND_DATA_PATH", "app/data/datasets/demand.csv"
    )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        if not self.USE_LOCAL_STORAGE:
            # Validate Supabase configuration
            if not self.SUPABASE_URL or not self.SUPABASE_SERVICE_ROLE_KEY:
                raise ValueError(
                    "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required "
                    "when USE_LOCAL_STORAGE is not set"
                )

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        lines = [
            "=== Application Configuration ===",
            f"Server: {self.HOST}:{self.PORT}",
            f"Debug: {self.DEBUG}",
            f"Reload: {self.RELOAD}",
            f"OpenAI Model: {self.OPENAI_MODEL}",
            f"Storage: {'Local JSON' if self.USE_LOCAL_STORAGE else 'Supabase'}",
            f"Data Directory: {self.DATA_DIR}",
        ]
        return "\n".join(lines)


# Global configuration instance
config = Config()
