"""Smoke tests for Docker artifacts: compose config syntax + Dockerfile lint (best-effort)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _has_docker() -> bool:
    return shutil.which("docker") is not None


# ---------------------------------------------------------------------------
# docker compose config — validates compose YAML syntax (no Docker daemon needed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_docker(), reason="docker not available in this environment")
def test_compose_config_valid():
    result = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"docker compose config failed:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Dockerfile lint — hadolint best-effort (skip if not installed)
# ---------------------------------------------------------------------------

def _has_hadolint() -> bool:
    return shutil.which("hadolint") is not None


DOCKERFILES = [
    REPO_ROOT / "Dockerfile",
    REPO_ROOT / "dashboard" / "Dockerfile",
]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_dockerfile_exists(dockerfile: Path):
    assert dockerfile.exists(), f"Dockerfile missing: {dockerfile}"


@pytest.mark.skipif(not _has_hadolint(), reason="hadolint not available in this environment")
@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_dockerfile_lint(dockerfile: Path):
    result = subprocess.run(
        ["hadolint", str(dockerfile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"hadolint found issues in {dockerfile.relative_to(REPO_ROOT)}:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Static checks that don't require Docker or hadolint
# ---------------------------------------------------------------------------

def test_bot_dockerfile_has_platform():
    content = (REPO_ROOT / "Dockerfile").read_text()
    assert "linux/amd64" in content, "Bot Dockerfile must target linux/amd64"


def test_bot_dockerfile_has_cmd_main():
    content = (REPO_ROOT / "Dockerfile").read_text()
    assert "main.py" in content, "Bot Dockerfile CMD must run main.py"


def test_dashboard_dockerfile_has_platform():
    content = (REPO_ROOT / "dashboard" / "Dockerfile").read_text()
    assert "linux/amd64" in content, "Dashboard Dockerfile must target linux/amd64"


def test_dashboard_dockerfile_frozen_lockfile():
    content = (REPO_ROOT / "dashboard" / "Dockerfile").read_text()
    assert "--frozen-lockfile" in content, "Dashboard Dockerfile must use --frozen-lockfile"


def test_dashboard_dockerfile_production_build():
    content = (REPO_ROOT / "dashboard" / "Dockerfile").read_text()
    assert "run build" in content, "Dashboard Dockerfile must run a production build"


def test_compose_has_soros_data_volume():
    content = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "soros_data" in content, "compose must declare soros_data volume"


def test_compose_has_restart_policy():
    content = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "unless-stopped" in content, "compose services must have restart: unless-stopped"


def test_compose_dashboard_port_3000():
    content = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "3000" in content, "compose must expose dashboard on port 3000"


def test_env_example_has_oauth_token():
    content = (REPO_ROOT / ".env.example").read_text()
    assert "CLAUDE_CODE_OAUTH_TOKEN" in content, ".env.example must document CLAUDE_CODE_OAUTH_TOKEN"


def test_env_example_crypto_live_default_false():
    content = (REPO_ROOT / ".env.example").read_text()
    assert "CRYPTO_LIVE=false" in content, ".env.example must default CRYPTO_LIVE to false"


def test_zimaos_runbook_exists():
    runbook = REPO_ROOT / "deploy" / "ZIMAOS.md"
    assert runbook.exists(), "ZimaOS runbook must exist at deploy/ZIMAOS.md"


def test_zimaos_runbook_covers_backup():
    content = (REPO_ROOT / "deploy" / "ZIMAOS.md").read_text()
    assert "backup" in content.lower(), "ZimaOS runbook must cover volume backup"


def test_zimaos_runbook_covers_git_pull_rebuild():
    content = (REPO_ROOT / "deploy" / "ZIMAOS.md").read_text()
    assert "git pull" in content, "ZimaOS runbook must document git pull + rebuild loop"
    assert "--build" in content, "ZimaOS runbook must document docker compose up --build"
