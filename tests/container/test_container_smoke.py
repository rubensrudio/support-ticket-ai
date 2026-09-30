"""Container smoke test for the CPU-only API image (OPS-02, TASK-028).

Builds ``docker/Dockerfile``, promotes a baseline trained on the synthetic sample
under ``tmp_path``, starts the container on port 18080 with the artifacts mounted
read-only and ``TICKET_API_KEY`` set, and checks ``/health`` and ``/predict``.
The container is always removed at the end.

Marked ``container``: excluded from the default ``uv run pytest`` run.
"""

import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from support.api_fixtures import TEST_API_KEY, build_promoted_baseline

pytestmark = pytest.mark.container

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE = "support-ticket-ai"
HOST_PORT = 18080
BASE_URL = f"http://localhost:{HOST_PORT}"
STARTUP_TIMEOUT_SECONDS = 120.0
BUILD_TIMEOUT_SECONDS = 1800
VALID_TICKET = {"title": "Cannot log in", "description": "My password reset link does not work."}


def _docker(*args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "null")


def _make_world_readable(root: Path) -> None:
    # The registry is written with owner-only permissions (mkstemp) and the container
    # runs as UID 10001, so grant read access like the README instructs.
    for path in [root, *root.rglob("*")]:
        extra = 0o555 if path.is_dir() else 0o444
        path.chmod(path.stat().st_mode | extra)


def _wait_until_ready(container: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            status, _ = _request("GET", "/health")
            if status in (200, 503):
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    logs = _docker("logs", container).stderr[-2000:]
    pytest.fail(f"API container did not become ready in time. Logs:\n{logs}")


@pytest.fixture(scope="module")
def image() -> str:
    result = _docker(
        "build",
        "-f",
        "docker/Dockerfile",
        "-t",
        IMAGE,
        str(REPO_ROOT),
        timeout=BUILD_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return IMAGE


@pytest.fixture
def running_api(image: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    settings = build_promoted_baseline(tmp_path)
    _make_world_readable(settings.artifacts_dir)
    container = f"support-ticket-ai-smoke-{uuid.uuid4().hex[:8]}"
    result = _docker(
        "run",
        "-d",
        "--rm",
        "--name",
        container,
        "-p",
        f"{HOST_PORT}:8000",
        "-v",
        f"{settings.artifacts_dir.resolve()}:/app/artifacts:ro",
        "-e",
        f"TICKET_API_KEY={TEST_API_KEY}",
        image,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    try:
        _wait_until_ready(container)
        yield container
    finally:
        _docker("rm", "-f", container)


def test_image_is_cpu_only_without_mlflow(image: str) -> None:
    result = _docker(
        "run",
        "--rm",
        image,
        "python",
        "-c",
        "import torch, importlib.util; assert torch.version.cuda is None; "
        "assert importlib.util.find_spec('mlflow') is None",
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_ops02_health_and_predict_in_container(running_api: str) -> None:
    status, health = _request("GET", "/health")
    assert status == 200, health
    assert health["status"] == "ok"
    assert health["model_version"]

    status, prediction = _request("POST", "/predict", VALID_TICKET)
    assert status == 200, prediction
    assert prediction["prediction_id"]
    assert prediction["model_version"] == health["model_version"]
