#!/usr/bin/env python3
"""从 voxy history.json 粗筛自定义词典候选项。"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path


PUNCT = " ，。！？；：,.!?:;\"“”'()（）[]【】<>《》\n\t"
TERM_RE = re.compile(r"(?:[A-Za-z][A-Za-z0-9.+#:/_-]*)(?: [A-Za-z0-9.+#:/_-]+)*$")


def _clean(text: str) -> str:
    return text.strip(PUNCT)


def _is_interesting(src: str, dst: str) -> bool:
    if not src or not dst or src == dst:
        return False
    if len(src) > 24 or len(dst) > 24:
        return False
    if len(dst) < 4:
        return False
    if len(src) < 2 and len(dst) < 2:
        return False

    if not TERM_RE.fullmatch(dst):
        return False

    has_ascii = bool(re.search(r"[A-Za-z0-9]", src + dst))
    has_mixed = bool(re.search(r"[\u4e00-\u9fff]", src + dst) and has_ascii)
    has_case_term = bool(re.search(r"[A-Z][A-Za-z0-9.+#:-]{2,}", dst))
    has_dev_term = bool(re.search(r"[A-Za-z][A-Za-z0-9.+#:/_-]{2,}", dst))
    return has_mixed or has_case_term or has_dev_term


def extract_candidates(records: list[dict], min_count: int) -> list[tuple[str, str, int]]:
    counter: Counter[tuple[str, str]] = Counter()

    for rec in records:
        raw = rec.get("raw", "").strip()
        polished = rec.get("polished", "").strip()
        if not raw or not polished:
            continue

        ratio = len(polished) / max(len(raw), 1)
        if ratio < 0.75 or ratio > 1.25:
            continue

        matcher = difflib.SequenceMatcher(a=raw, b=polished, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue

            src = _clean(raw[i1:i2])
            dst = _clean(polished[j1:j2])
            if not _is_interesting(src, dst):
                continue

            counter[(src, dst)] += 1

    items = []
    for (src, dst), count in counter.items():
        if count >= min_count:
            items.append((src, dst, count))

    items.sort(key=lambda item: (-item[2], item[1].lower(), item[0].lower()))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "history",
        nargs="?",
        default=str(Path.home() / ".local" / "share" / "voxy" / "history.json"),
        help="history.json 路径",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="至少出现多少次才输出",
    )
    args = parser.parse_args()

    history_path = Path(args.history).expanduser()
    records = json.loads(history_path.read_text(encoding="utf-8"))

    print("[llm.custom_terms]")
    for src, dst, count in extract_candidates(records, args.min_count):
        print(f'# count={count}')
        print(f'"{src}" = "{dst}"')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
