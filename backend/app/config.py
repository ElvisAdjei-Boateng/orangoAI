from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    episodes: int = 500
    user_type: str = "neutral"
    seed: int = 7
    hidden_dim: int = 64
    learning_rate: float = 0.001
    discount: float = 0.95
    epsilon_decay: float = 0.995
    output_dir: str = "artifacts"

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        values: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
