"""Behavioral tests for the repository's public-push guard.

The hook is an allowlist: only the published branch and its ``v*`` tags may
leave this machine, and their history must contain no internal-only path.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOOK = PROJECT_ROOT / "scripts" / "git-hooks" / "pre-push"


def _git(repo: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _repo_with_commit(tmp_path: Path, relative_path: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    assert _git(repo, "init", "-b", "main").returncode == 0
    assert _git(repo, "config", "user.email", "test@example.com").returncode == 0
    assert _git(repo, "config", "user.name", "Push Guard Test").returncode == 0

    target = repo / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("internal test content\n", encoding="utf-8")
    assert _git(repo, "add", "-f", relative_path).returncode == 0
    assert _git(repo, "commit", "-m", "test: create candidate commit").returncode == 0
    return repo


def _run_pre_push(repo: Path, old_sha: str = "0" * 40) -> subprocess.CompletedProcess[str]:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    ref_update = f"refs/heads/main {head} refs/heads/main {old_sha}\n"
    environment = {**os.environ, "PATH": os.environ["PATH"]}
    return subprocess.run(
        [str(HOOK), "origin", "git@example.com:example/public.git"],
        cwd=repo,
        input=ref_update,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


def test_pre_push_rejects_internal_directories_even_when_force_added(tmp_path: Path) -> None:
    for relative_path in (
        ".claude/settings.local.json",
        ".agents/skills/example.md",
        ".codex/example.md",
        ".superpowers/example.md",
        "docs/ARCHITECTURE.md",
        "openspec/config.yaml",
        ".env",
        "src/prompt_workbench/__pycache__/app.pyc",
        ".DS_Store",
    ):
        result = _run_pre_push(
            _repo_with_commit(tmp_path / relative_path.replace("/", "_"), relative_path)
        )
        assert result.returncode != 0, relative_path
        assert relative_path in result.stderr


def test_pre_push_rejects_internal_planning_files_even_when_force_added(tmp_path: Path) -> None:
    for relative_path in ("CLAUDE.md", "AGENTS.md", "PLAN.md"):
        result = _run_pre_push(
            _repo_with_commit(tmp_path / relative_path.replace("/", "_"), relative_path)
        )
        assert result.returncode != 0, relative_path
        assert relative_path in result.stderr


def test_pre_push_allows_public_application_files(tmp_path: Path) -> None:
    repo = _repo_with_commit(tmp_path, "src/prompt_workbench/app.py")

    result = _run_pre_push(repo)

    assert result.returncode == 0, result.stderr


def test_pre_push_allows_vscode_configuration(tmp_path: Path) -> None:
    repo = _repo_with_commit(tmp_path, ".vscode/settings.json")

    result = _run_pre_push(repo)

    assert result.returncode == 0, result.stderr


def test_pre_push_rejects_internal_path_deleted_from_tip(tmp_path: Path) -> None:
    """History, not just the tip: a removed internal file is still in the pack."""
    repo = _repo_with_commit(tmp_path, "openspec/config.yaml")
    (repo / "openspec/config.yaml").unlink()
    assert _git(repo, "add", "-u").returncode == 0
    assert _git(repo, "commit", "-m", "test: delete internal file").returncode == 0

    result = _run_pre_push(repo)

    assert result.returncode != 0
    assert "openspec/config.yaml" in result.stderr


def test_pre_push_refuses_any_branch_other_than_the_published_one(tmp_path: Path) -> None:
    repo = _repo_with_commit(tmp_path, "src/prompt_workbench/app.py")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()

    result = subprocess.run(
        [str(HOOK), "origin", "git@example.com:example/public.git"],
        cwd=repo,
        input=f"refs/heads/development {head} refs/heads/development {'0' * 40}\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refs/heads/development" in result.stderr
