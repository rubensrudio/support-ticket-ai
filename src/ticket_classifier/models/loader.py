"""Load the classifier of a registered model version from local artifacts (CT-16).

Must not import ``training`` or ``tracking``: it is used by the API at runtime.
"""

from pathlib import Path

from ticket_classifier.errors import PipelineError
from ticket_classifier.models.base import TicketClassifier
from ticket_classifier.models.baseline import BaselineClassifier
from ticket_classifier.models.transformer import TransformerClassifier
from ticket_classifier.registry import ModelVersion


def _version_directory(version: ModelVersion, artifacts_dir: Path) -> Path:
    root = Path(artifacts_dir).resolve()
    directory = (root / version.artifact_dir).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise PipelineError(f"Invalid artifact directory for version {version.version_id}.")
    if not directory.is_dir():
        raise PipelineError(f"Model artifacts not found for version {version.version_id}.")
    return directory


def load_classifier(version: ModelVersion, artifacts_dir: Path) -> TicketClassifier:
    """Load the classifier of ``version`` from ``artifacts_dir``, dispatching on ``kind``."""
    directory = _version_directory(version, artifacts_dir)
    try:
        if version.kind == "baseline":
            return BaselineClassifier.load(directory)
        if version.kind == "transformer":
            return TransformerClassifier.load(directory)
    except (OSError, ValueError) as exc:
        raise PipelineError(
            f"Model artifacts of version {version.version_id} could not be loaded."
        ) from exc
    raise PipelineError(f"Unknown model kind for version {version.version_id}.")
