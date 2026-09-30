"""File-based registry of model versions (``artifacts/registry.json``).

Writes are serialized with ``fcntl.flock`` on ``registry.lock`` and made atomic with a
temporary file plus ``os.replace``. Reads take no lock: they always see a complete file.
A missing ``registry.json`` is an empty registry.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ticket_classifier.errors import PipelineError

REGISTRY_FILE = "registry.json"
LOCK_FILE = "registry.lock"
MODELS_DIR = "models"

KINDS = ("baseline", "transformer")
STATUS_REGISTERED = "registered"
STATUS_PROMOTED = "promoted"
STATUS_RETIRED = "retired"
STATUSES = (STATUS_REGISTERED, STATUS_PROMOTED, STATUS_RETIRED)

_VERSION_ID_RE = re.compile(r"^(baseline|transformer)-\d{8}T\d{6}Z-[0-9a-f]{8}$")

Metrics = dict[str, dict[str, dict[str, Any]]]


@dataclass
class ModelVersion:
    """One trained model version as stored in the registry (plan section 7.3)."""

    version_id: str
    kind: str
    status: str
    created_at: str
    splits_version: str
    seed: int
    hyperparameters: dict[str, Any]
    metrics: Metrics
    training_rows: int
    artifact_dir: str
    feedback_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVersion:
        if not isinstance(data, dict) or data.get("status") not in STATUSES:
            raise PipelineError("Invalid model version entry in registry.")
        try:
            return cls(
                version_id=str(data["version_id"]),
                kind=str(data["kind"]),
                status=str(data["status"]),
                created_at=str(data["created_at"]),
                splits_version=str(data["splits_version"]),
                seed=int(data["seed"]),
                hyperparameters=dict(data["hyperparameters"]),
                metrics=dict(data["metrics"]),
                training_rows=int(data["training_rows"]),
                artifact_dir=str(data["artifact_dir"]),
                feedback_rows=int(data.get("feedback_rows", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError("Invalid model version entry in registry.") from exc

    def test_macro_f1(self, target: str) -> float | None:
        """Macro F1 of ``target`` on the frozen test set, if recorded."""
        value = self.metrics.get("test", {}).get(target, {}).get("macro_f1")
        return float(value) if isinstance(value, int | float) else None


def _created_at_key(version: ModelVersion) -> datetime:
    try:
        parsed = datetime.fromisoformat(version.created_at)
    except ValueError as exc:
        raise PipelineError(f"Invalid created_at for version {version.version_id}.") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class ModelRegistry:
    """Registry of model versions stored under ``artifacts_dir``."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)

    @property
    def registry_path(self) -> Path:
        return self.artifacts_dir / REGISTRY_FILE

    def new_version_id(self, kind: str) -> str:
        _check_kind(kind)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{kind}-{timestamp}-{secrets.token_hex(4)}"

    def model_dir(self, version_id: str) -> Path:
        _check_version_id(version_id)
        return self.artifacts_dir / MODELS_DIR / version_id

    def register(self, version: ModelVersion) -> None:
        _check_version_id(version.version_id)
        _check_kind(version.kind)
        if version.status != STATUS_REGISTERED:
            raise PipelineError(
                f"New model version must have status '{STATUS_REGISTERED}', got '{version.status}'."
            )
        _created_at_key(version)

        def apply(versions: list[ModelVersion]) -> None:
            if any(v.version_id == version.version_id for v in versions):
                raise PipelineError(f"Model version already registered: {version.version_id}.")
            versions.append(version)

        self._update(apply)

    def promote(self, version_id: str) -> None:
        def apply(versions: list[ModelVersion]) -> None:
            target = _find(versions, version_id)
            if target is None:
                raise PipelineError(f"Model version not found: {version_id}.")
            if target.status != STATUS_REGISTERED:
                raise PipelineError(
                    f"Model version {version_id} cannot be promoted from status '{target.status}'."
                )
            for other in versions:
                if other.status == STATUS_PROMOTED:
                    other.status = STATUS_RETIRED
            target.status = STATUS_PROMOTED

        self._update(apply)

    def get(self, version_id: str) -> ModelVersion:
        found = _find(self._read(), version_id)
        if found is None:
            raise PipelineError(f"Model version not found: {version_id}.")
        return found

    def get_promoted(self) -> ModelVersion | None:
        for version in self._read():
            if version.status == STATUS_PROMOTED:
                return version
        return None

    def latest(self, kind: str) -> ModelVersion | None:
        _check_kind(kind)
        candidates = [v for v in self.list_versions() if v.kind == kind]
        return candidates[-1] if candidates else None

    def list_versions(self) -> list[ModelVersion]:
        return sorted(self._read(), key=_created_at_key)

    def _read(self) -> list[ModelVersion]:
        try:
            raw = self.registry_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise PipelineError("Cannot read the model registry.") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PipelineError("Model registry file is corrupted.") from exc
        entries = data.get("versions") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise PipelineError("Model registry file is corrupted.")
        return [ModelVersion.from_dict(entry) for entry in entries]

    def _update(self, apply: Callable[[list[ModelVersion]], None]) -> None:
        with self._lock():
            versions = self._read()
            apply(versions)
            self._write(versions)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(self.artifacts_dir / LOCK_FILE, "a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write(self, versions: list[ModelVersion]) -> None:
        payload = json.dumps({"versions": [v.to_dict() for v in versions]}, indent=2)
        fd, tmp_name = tempfile.mkstemp(prefix=".registry-", suffix=".tmp", dir=self.artifacts_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, self.registry_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


def _find(versions: list[ModelVersion], version_id: str) -> ModelVersion | None:
    return next((v for v in versions if v.version_id == version_id), None)


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise PipelineError(f"Unknown model kind; expected one of: {', '.join(KINDS)}.")


def _check_version_id(version_id: str) -> None:
    if not _VERSION_ID_RE.match(version_id):
        raise PipelineError("Invalid model version id.")
