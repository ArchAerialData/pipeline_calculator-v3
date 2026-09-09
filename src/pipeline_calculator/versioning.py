"""Deterministic Git versions and embedded build metadata (stdlib only)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

BASELINE = "fc4cc05108dda7ae2f61be4a763eb34f5f1ebb0e"
MAJOR = 4


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ).strip()


def derive_version(repo: Path, *, baseline: str = BASELINE, env=None) -> dict:
    env = os.environ if env is None else env
    if git(repo, "rev-parse", "--is-shallow-repository") == "true":
        raise ValueError("Full Git history required: git fetch --unshallow origin")
    git(repo, "rev-parse", "--verify", baseline + "^{commit}")
    head = git(repo, "rev-parse", "HEAD")
    history = git(repo, "rev-list", "--first-parent", "HEAD").splitlines()
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=normal"))
    ref = env.get("GITHUB_REF", "")
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    # CI context takes precedence: PR synthetic merges must remain previews.
    release = ref == "refs/heads/main" if ref else branch == "main"
    if ref.startswith("refs/tags/") or (not ref and branch == "HEAD"):
        main_history = git(repo, "rev-list", "--first-parent", "origin/main").splitlines()
        release = head in main_history
    if release and baseline not in history:
        raise ValueError("Release baseline must be on main's first-parent history")
    if release and dirty and ref:
        raise ValueError("CI release builds require a clean working tree")
    # Old branches may fork before the baseline; keep these explicit previews.
    count = history.index(baseline) if baseline in history else int(
        git(repo, "rev-list", "--first-parent", "--count", f"{baseline}..HEAD")
    )
    numeric = f"{MAJOR}.{count}"
    version = numeric if release and not dirty else f"{numeric}-dev.{head[:12]}"
    if dirty:
        version += ".dirty"
    if ref.startswith("refs/tags/") and (not release or ref != f"refs/tags/v{version}"):
        raise ValueError(f"Release tag must match v{version} and point to main history")
    return {"version": version, "numeric_version": numeric, "commit": head, "dirty": dirty}


def get_version() -> str:
    metadata = Path(__file__).with_name("version.json")
    if metadata.is_file():
        return json.loads(metadata.read_text(encoding="utf-8"))["version"]
    if not getattr(sys, "frozen", False):
        try:
            return derive_version(Path(__file__).resolve().parents[2])["version"]
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return f"{MAJOR}.0-dev.unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        metadata = derive_version(Path(__file__).resolve().parents[2])
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.exit(1, f"Cannot determine build version: {exc}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata["version"])


if __name__ == "__main__":
    main()
