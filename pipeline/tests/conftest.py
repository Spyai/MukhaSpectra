"""Shared fixtures for pipeline tests. Module-specific helpers live in <module>_helpers.py."""

from datetime import UTC, datetime

import numpy as np
import pytest

from mukha_pipeline.config import PipelineConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def pipeline_config() -> PipelineConfig:
    return PipelineConfig.default()


@pytest.fixture(scope="session")
def qc_cfg(pipeline_config: PipelineConfig) -> dict:
    return pipeline_config.qc.data


@pytest.fixture(scope="session")
def analysis_cfg(pipeline_config: PipelineConfig) -> dict:
    return pipeline_config.analysis.data


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 10, 14, 9, 30, tzinfo=UTC)
