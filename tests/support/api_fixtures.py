"""Fixtures for API tests (TASK-023).

``build_promoted_baseline`` trains a baseline on the synthetic sample, registers
and promotes it, and returns ``Settings`` pointing at ``tmp_path``.
"""

from pathlib import Path

from pydantic import SecretStr

from support.sample_data import prepare_sample
from ticket_classifier.registry import ModelRegistry
from ticket_classifier.settings import Settings
from ticket_classifier.training import train_and_register

TEST_API_KEY = "test-key"


def build_promoted_baseline(tmp_path: Path) -> Settings:
    """Prepare the sample, train and promote a baseline, and return test settings.

    ``MLFLOW_TRACKING_URI`` must point to a location under ``tmp_path`` (set by the
    caller) so that tracking never writes into the repository.
    """
    config = prepare_sample(tmp_path / "sample")
    settings = Settings(
        api_key=SecretStr(TEST_API_KEY),
        artifacts_dir=tmp_path / "artifacts",
        db_path=tmp_path / "db" / "tickets.db",
    )
    outcome = train_and_register("baseline", config, settings)
    ModelRegistry(settings.artifacts_dir).promote(outcome.version.version_id)
    return settings
