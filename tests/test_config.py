from pathlib import Path

import pytest

from more.config import ModelConfig, MoREConfig
from more.core import ProjectConfig, load_config, load_experiment_config


def test_model_config_validates_attention_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=10, n_heads=3)


def test_yaml_config_and_legacy_config_are_both_available(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
project:
  name: Test
routing:
  threshold: 0.4
  max_depth: 3
model:
  d_model: 32
  n_heads: 4
  max_seq_len: 16
""",
        encoding="utf-8",
    )
    legacy = load_config(path)
    experiment = load_experiment_config(path)
    assert isinstance(legacy, ProjectConfig)
    assert legacy.project_name == "Test"
    assert legacy.routing_threshold == 0.4
    assert experiment.model.d_model == 32
    assert experiment.model.max_seq_len == 16


def test_overrides_round_trip() -> None:
    config = MoREConfig.from_dict({}).with_overrides(**{"train.max_steps": 3})
    assert config.train.max_steps == 3
