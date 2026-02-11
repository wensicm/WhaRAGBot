#!/usr/bin/env python3.12
"""Fail commits when likely secrets or notebook outputs are present."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI API key", re.compile(r"sk-(?:proj|live|test)-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "AWS secret key assignment",
        re.compile(r"aws_secret_access_key\s*[:=]\s*[A-Za-z0-9/+_=]{30,}", re.IGNORECASE),
    ),
    (
        "Private key block",
        re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY-----"),
    ),
]

BLOCKED_FILENAMES = {"AGENTS.md", "agents.md"}


def git_root() -> Path:
    cp = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(cp.stdout.strip())


def list_git_files(repo_root: Path) -> list[Path]:
    cp = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=False,
        cwd=repo_root,
    )
    raw = cp.stdout.decode("utf-8", errors="ignore")
    return [repo_root / p for p in raw.split("\x00") if p]


def iter_targets(paths: list[str], repo_root: Path) -> Iterable[Path]:
    if paths:
        for p in paths:
            path = Path(p)
            if not path.is_absolute():
                path = repo_root / path
            if path.exists() and path.is_file():
                yield path
        return

    for path in list_git_files(repo_root):
        if path.exists() and path.is_file():
            yield path


def line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_blocked_path(path: Path) -> list[str]:
    issues: list[str] = []
    name = path.name
    lowered = name.lower()

    if name in BLOCKED_FILENAMES:
        issues.append(f"{path}:1: blocked file name '{name}'")

    if lowered.startswith(".env") and name != ".env.example":
        issues.append(f"{path}:1: '.env*' files are not allowed in git (except .env.example)")

    if path.suffix.lower() in {".pem", ".key", ".p12", ".crt"}:
        issues.append(f"{path}:1: certificate/key files are not allowed in git")

    return issues


def scan_text(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            line = line_for_offset(text, match.start())
            issues.append(f"{path}:{line}: possible {label}")
    return issues


def scan_notebook(path: Path) -> list[str]:
    issues: list[str] = []
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}:1: cannot parse notebook JSON: {exc}"]

    cells = nb.get("cells", [])
    for idx, cell in enumerate(cells, start=1):
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            issues.append(f"{path}:cell{idx}: notebook outputs must be cleared before commit")

        source = "".join(cell.get("source", []))
        for label, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(source):
                line = line_for_offset(source, match.start())
                issues.append(f"{path}:cell{idx}:line{line}: possible {label}")

    return issues


def scan_file(path: Path) -> list[str]:
    issues = check_blocked_path(path)

    if path.suffix.lower() == ".ipynb":
        issues.extend(scan_notebook(path))
        return issues

    try:
        data = path.read_bytes()
    except Exception as exc:
        return issues + [f"{path}:1: cannot read file: {exc}"]

    if b"\x00" in data:
        return issues

    text = data.decode("utf-8", errors="ignore")
    issues.extend(scan_text(path, text))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    repo_root = git_root()
    issues: list[str] = []
    for path in iter_targets(args.paths, repo_root):
        issues.extend(scan_file(path))

    if issues:
        print("[repo-safety] commit blocked:\n")
        for issue in issues:
            print(f"- {issue}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
