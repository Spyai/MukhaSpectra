"""Versioned YAML configuration (QC thresholds, analysis parameters, triage rules).

Every result records the name, version and SHA-256 of each config file it used, so a report can
always be traced back to the exact thresholds and weights that produced it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mukha_pipeline.schema.models import ConfigRef

CONFIG_DIR = Path(__file__).parent

QC_CONFIG = "qc.v1.yaml"
ANALYSIS_CONFIG = "analysis.v1.yaml"
TRIAGE_CONFIG = "triage_rules.v1.yaml"


@dataclass(frozen=True)
class LoadedConfig:
    data: dict[str, Any]
    ref: ConfigRef


def load_config(name_or_path: str | Path) -> LoadedConfig:
    """Load a bundled config by file name, or any YAML file by path."""
    path = Path(name_or_path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    raw = path.read_bytes()
    data = yaml.safe_load(raw) or {}
    ref = ConfigRef(
        name=str(data.get("name", path.stem)),
        version=str(data.get("version", "0")),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    return LoadedConfig(data=data, ref=ref)


@dataclass(frozen=True)
class PipelineConfig:
    qc: LoadedConfig
    analysis: LoadedConfig
    triage: LoadedConfig

    @classmethod
    def default(cls) -> PipelineConfig:
        return cls(
            qc=load_config(QC_CONFIG),
            analysis=load_config(ANALYSIS_CONFIG),
            triage=load_config(TRIAGE_CONFIG),
        )

    @classmethod
    def from_paths(
        cls,
        qc: str | Path | None = None,
        analysis: str | Path | None = None,
        triage: str | Path | None = None,
    ) -> PipelineConfig:
        return cls(
            qc=load_config(qc or QC_CONFIG),
            analysis=load_config(analysis or ANALYSIS_CONFIG),
            triage=load_config(triage or TRIAGE_CONFIG),
        )
