#!/usr/bin/env python3
"""Review Voxy custom_terms candidates against current config."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tomllib
from pathlib import Path


def load_extractor(repo_root: Path):
    script_path = repo_root / "contrib" / "extract_custom_terms.py"
    spec = importlib.util.spec_from_file_location("voxy_extract_terms", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extractor from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_terms(config_path: Path) -> dict[str, str]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return config.get("llm", {}).get("custom_terms", {})


def is_partial_destination(dst: str, current_terms: dict[str, str]) -> bool:
    values = set(current_terms.values())
    return any(value != dst and value.startswith(dst) for value in values)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path.home() / ".local" / "share" / "voxy" / "history.json",
        help="Path to history.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".config" / "voxy" / "config.toml",
        help="Path to config.toml",
    )
    parser.add_argument("--min-count", type=int, default=2, help="Minimum candidate count")
    parser.add_argument("--limit", type=int, default=20, help="Max rows per section")
    args = parser.parse_args()

    history_path = args.history.expanduser()
    config_path = args.config.expanduser()

    records = json.loads(history_path.read_text(encoding="utf-8"))
    current_terms = load_terms(config_path)
    extractor = load_extractor(repo_root)
    candidates = extractor.extract_candidates(records, args.min_count)

    new_candidates: list[tuple[str, str, int]] = []
    existing_matches: list[tuple[str, str, int]] = []
    conflicts: list[tuple[str, str, str, int]] = []
    skipped_partial: list[tuple[str, str, int]] = []

    for src, dst, count in candidates:
        if is_partial_destination(dst, current_terms):
            skipped_partial.append((src, dst, count))
            continue
        existing = current_terms.get(src)
        if existing is None:
            new_candidates.append((src, dst, count))
        elif existing == dst:
            existing_matches.append((src, dst, count))
        else:
            conflicts.append((src, existing, dst, count))

    print(f"history_records: {len(records)}")
    print(f"current_custom_terms: {len(current_terms)}")
    print(f"candidate_pairs: {len(candidates)}")
    print()

    print("[new_candidates]")
    for src, dst, count in new_candidates[: args.limit]:
        print(f"# count={count}")
        print(f'"{src}" = "{dst}"')
    if not new_candidates:
        print("# none")
    print()

    print("[conflicts]")
    for src, existing, proposed, count in conflicts[: args.limit]:
        print(f"# count={count}")
        print(f'"{src}" current="{existing}" proposed="{proposed}"')
    if not conflicts:
        print("# none")
    print()

    print("[skipped_partial_destinations]")
    for src, dst, count in skipped_partial[: args.limit]:
        print(f"# count={count}")
        print(f'"{src}" = "{dst}"')
    if not skipped_partial:
        print("# none")
    print()

    print("[existing_matches]")
    for src, dst, count in existing_matches[: args.limit]:
        print(f"# count={count}")
        print(f'"{src}" = "{dst}"')
    if not existing_matches:
        print("# none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
