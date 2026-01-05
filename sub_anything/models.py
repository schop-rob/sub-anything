"""Data models for sub-anything."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TranscriptSegment:
    """A single transcribed segment with timing."""
    start_time: float  # seconds
    end_time: float  # seconds
    text: str
    confidence: float = 1.0
    speaker: Optional[str] = None
    language: Optional[str] = None


@dataclass
class Config:
    """Persistent configuration."""
    gcs_bucket: str = ""
    project_id: str = ""
    chirp_location: str = "us"

    @classmethod
    def load(cls, config_file: Path) -> "Config":
        if config_file.exists():
            with open(config_file) as f:
                data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in ["gcs_bucket", "project_id", "chirp_location"]})
        return cls()

    def save(self, config_file: Path):
        with open(config_file, "w") as f:
            json.dump(
                {"gcs_bucket": self.gcs_bucket, "project_id": self.project_id, "chirp_location": self.chirp_location},
                f,
                indent=2,
            )

    def is_complete(self) -> bool:
        return bool(self.gcs_bucket and self.project_id)
