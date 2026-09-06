import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_backend: str
    models_dir: Path


def get_settings() -> Settings:
    return Settings(
        model_backend=os.environ.get("MODEL_BACKEND", "sklearn"),
        models_dir=Path(os.environ.get("MODELS_DIR", "models")),
    )
