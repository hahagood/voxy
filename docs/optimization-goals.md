# Voxy Optimization Goals

Date: 2026-03-31

This document distills the comparison with `ZhangHanDong/vox` into concrete optimization goals for Voxy.

Scope: algorithm, process, and structure only.

Non-goals:
- Do not optimize for rich GUI or menu-bar style UX.
- Do not copy code or implementation details from AGPL projects.
- Do not trade away Voxy's existing strengths in backend flexibility and automation.

## Current Position

Voxy is already ahead in these areas:
- Multiple STT backends with a common abstraction.
- Daemon mode for warm model reuse.
- Voice commands and context-aware command mapping.
- Long-audio transcription and long-text polishing flow.

The main improvement opportunities are not "better models", but cleaner execution flow and stronger internal boundaries.

## Optimization Goals

### 1. Treat speech input as a transaction

Model one speech action as an explicit transaction:

`capture -> transcribe -> command_match -> polish -> inject/output -> restore -> fallback`

Target outcomes:
- Side effects are explicit and reversible where possible.
- Platform-specific output logic can prepare state before injection and restore it after.
- Failures have a defined rollback or fallback path instead of ad hoc subprocess behavior.

Why:
- This makes text output more robust across Wayland, X11, terminals, and future adapters.
- It prevents the current "do one command and hope it worked" structure from spreading.

### 2. Introduce first-class profiles above raw config

Add a profile layer that compiles to low-level config.

Examples:
- `dictation`
- `raw-transcribe`
- `translate-to-english`
- `polish-chinese`
- `command-mode`

A profile should be able to define:
- STT backend and language behavior
- LLM enablement and prompt strategy
- output mode
- command matching policy

Why:
- Users should think in tasks, not in provider/model wiring.
- This keeps the current flexibility while giving the pipeline a more stable semantic layer.

### 3. Make the daemon the single execution surface

Move toward a model where the daemon is the canonical execution engine and the CLI is mainly a client.

Direction:
- CLI submits requests to daemon when available.
- Daemon owns backend lifecycle, model warmup, and execution routing.
- Local model backends and cloud backends share the same request path.

Why:
- Reduces branching between direct mode and daemon mode.
- Makes future integrations easier because all frontends can target one stable RPC boundary.
- Centralizes observability, caching, and retries.

### 4. Isolate platform adapters from business logic

Create a clearer platform boundary for:
- active window detection
- clipboard access
- typed text injection
- terminal/browser/editor-specific output behavior

The pipeline should depend on platform interfaces, not on `hyprctl`, `wtype`, or `xdotool` directly.

Why:
- Current Linux integration works, but platform details are spread across business modules.
- Cleaner adapters will make it easier to support more environments without leaking shell commands upward.

### 5. Split orchestration out of the CLI entrypoint

`cli.py` should stop accumulating business orchestration.

Target structure:
- CLI: argument parsing and response rendering
- application service / pipeline: request orchestration
- backends: STT, LLM, commands, output, platform adapters

Why:
- The current entrypoint already coordinates history, daemon fallback, command matching, polishing, segmentation, and output.
- Continued growth in `cli.py` will make behavior harder to test and reason about.

### 6. Formalize output-state handling

Add an explicit output-state model for injection flows.

Examples of state that may need handling:
- clipboard preservation
- input mode preservation
- focus/window targeting
- delayed restore
- fallback from paste to direct type

Why:
- Even in a minimal product, correctness of output matters more than appearance.
- This is the structural lesson most worth carrying forward from native voice-input apps.

### 7. Keep prompt strategy modular

Preserve and strengthen prompt modularity rather than baking language behavior into one fixed path.

Direction:
- separate base cleanup prompt from profile-specific behavior
- allow profile-specific prompt overlays
- keep custom terms and segmentation as composable layers

Why:
- Voxy's LLM path is already more flexible than the compared project.
- The right next step is better composition, not simplification into one prompt.

### 8. Add design notes for major subsystems

For subsystems with growing complexity, keep short design docs in-repo.

Suggested docs:
- transcription pipeline
- daemon protocol
- platform adapter boundary
- command execution model

Why:
- README is not enough once behavior becomes multi-stage.
- Short design notes reduce future architectural drift.

## Near-Term Priority

Priority 1:
- Define a pipeline service outside `cli.py`
- Introduce transactional output stages: `prepare`, `inject`, `restore`, `fallback`
- Start extracting Linux platform adapters

Priority 2:
- Unify execution around daemon requests
- Define profile abstraction above raw TOML

Priority 3:
- Add design docs for pipeline and daemon protocol
- Improve observability around stage timing and failures

## Guardrails

When optimizing Voxy, keep these constraints:
- Minimal UX is acceptable; brittle behavior is not.
- Backend flexibility is a competitive advantage and should be preserved.
- Voice commands remain a core differentiator, not a side feature.
- Structural simplification should reduce branching, not remove capabilities.

## Success Criteria

These changes are successful if:
- adding a new output target or platform requires a new adapter, not edits across unrelated modules
- CLI commands mostly describe requests instead of containing workflow logic
- daemon and non-daemon execution produce the same functional behavior
- profiles express common tasks without exposing low-level model wiring every time
- failures during output or platform interaction have predictable fallback behavior
