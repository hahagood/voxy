---
name: voxy-dictionary-maintainer
description: Maintain Voxy's ASR correction dictionary from history.json and config.toml. Use this skill when the user wants to review `~/.local/share/voxy/history.json`, extract `llm.custom_terms` candidates, merge or prune dictionary entries, resolve conflicts in `~/.config/voxy/config.toml`, or turn repeated raw/polished corrections into reusable Voxy term mappings.
---

# Voxy Dictionary Maintainer

Use this skill for Voxy-specific dictionary maintenance, not for general text cleanup.

## When To Use

Trigger this skill when the user asks to:

- inspect or summarize `~/.local/share/voxy/history.json`
- extract new `llm.custom_terms` candidates
- merge confirmed mappings into `~/.config/voxy/config.toml`
- review conflicts, stale mappings, or noisy dictionary entries
- derive a reusable correction dictionary from repeated `raw` / `polished` pairs

## Files To Check First

Read only what is needed:

1. `~/.local/share/voxy/history.json`
2. `~/.config/voxy/config.toml`
3. `contrib/extract_custom_terms.py`

Use `scripts/review_terms.py` for a deterministic first pass before making manual decisions.

## Workflow

### 1. Collect current state

- Confirm that `history.json` exists and estimate its record count.
- Read the current `[llm.custom_terms]` table from `config.toml`.
- If the user wants repo-facing changes, also inspect `README.md` and `config.example.toml`.

### 2. Generate candidate mappings

Run:

```bash
python .codex/skills/voxy-dictionary-maintainer/scripts/review_terms.py --min-count 2
```

This script compares extracted candidates against the current dictionary and separates:

- `new_candidates`
- `existing_matches`
- `conflicts`

If the history is small or the user wants a broader sweep, lower `--min-count` to `1`. Prefer `2+` for anything you intend to merge.

### 3. Manually filter candidates

Prefer entries that are stable across contexts:

- product names: `OpenRouter`, `Ollama`, `Anthropic`
- project names: `TS-GPT`
- tools and files: `GitHub`, `Notion`, `README.md`
- technical terms: `token`, `OpenAI`, `iOS`

Reject or review carefully:

- summary-like rewrites
- punctuation-only changes
- phrase rewrites that depend on sentence context
- partial fragments that are not valid standalone terms

If a candidate conflicts with an existing mapping, do not overwrite it automatically. Show the conflict and explain why one side should win.

## Apply Rules

- Only edit `~/.config/voxy/config.toml` when the user asked to apply changes, not when they only asked for review.
- Preserve existing mappings unless there is direct evidence they are wrong or obsolete.
- Keep the table readable; group related terms loosely and avoid duplicate variants when one canonical form is enough.
- After edits, validate the TOML by loading it with `tomllib`.

## Useful Commands

Review candidates:

```bash
python .codex/skills/voxy-dictionary-maintainer/scripts/review_terms.py --min-count 2
```

Show raw extractor output:

```bash
python contrib/extract_custom_terms.py --min-count 2
```

Validate config:

```bash
python - <<'PY'
import tomllib, pathlib
path = pathlib.Path.home() / ".config" / "voxy" / "config.toml"
tomllib.loads(path.read_text(encoding="utf-8"))
print("OK")
PY
```

## Output Expectations

When reviewing, produce:

- a short count of history records inspected
- the best new candidates
- any conflicts with current `custom_terms`
- a note about what was intentionally skipped

When applying, also:

- edit `config.toml`
- validate the file
- mention the exact file changed
