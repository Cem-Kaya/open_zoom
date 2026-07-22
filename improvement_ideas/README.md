# Improvement Ideas

Actionable improvement backlog for OpenZoom, written for AI agents (or humans) picking up
future work. Produced from a full-codebase analysis on 2026-07-21 at commit `9e069d9`
("Add assistive runtime and harden Windows builds"). All `file:line` references are
relative to that commit — re-locate by symbol name if the files have since changed.

## How to use this directory

- Each file groups related ideas by theme. Ideas are self-contained: problem, evidence,
  concrete fix, priority, effort.
- **Verification status matters.** Ideas marked `Confirmed` were verified against the
  source during this analysis. Ideas marked `Reported` came from subsystem analysis and
  are plausible but should be re-verified against the code before implementing.
- Read [`verified-non-issues.md`](verified-non-issues.md) **before** starting your own
  analysis — it lists plausible-looking "bugs" that were investigated and refuted, so you
  don't waste time rediscovering them.
- Follow `agents.md` at the repo root: update `docs/code_reference.md`, `CHANGELOG.md`,
  and related docs in the same commit as any code change.
- When you implement an idea, delete it from its file (or mark it `DONE <date>` with a
  commit reference) so this backlog stays truthful.

## Files

| File | Theme | Highest priority item |
|---|---|---|
| [`01-stability-threading.md`](01-stability-threading.md) | Crashes, races, COM/CUDA lifetime | Capture/UI threading audit and fence failure recovery |
| [`02-architecture-app-decomposition.md`](02-architecture-app-decomposition.md) | God-object breakup, state machines | Decompose 2,233-line `app.cpp` |
| [`03-performance-gpu-cpu.md`](03-performance-gpu-cpu.md) | Frame latency and throughput | Shared-memory Gaussian blur and frame timing |
| [`04-accessibility-ux.md`](04-accessibility-ux.md) | Accessibility (the product mission) | Screen-reader labels on all controls |
| [`05-robustness-validation.md`](05-robustness-validation.md) | Input validation, error handling, I/O | Settings versioning/migration; stride overflow checks |
| [`06-build-tooling-docs.md`](06-build-tooling-docs.md) | Build system, CI, tests, docs hygiene | Compiler warnings + first test target |
| [`verified-non-issues.md`](verified-non-issues.md) | Refuted findings — do not "fix" these | — |

## Suggested implementation order

1. **Stability first** (`01`): the camera-switch lifetime bug, COM leak, and
   mid-stream format handling are complete; verify the remaining capture/UI
   threading and fence-failure recovery items before broad refactors.
2. **Cheap guardrails** (`06`): enable `/W4` warnings and add a first unit-test target for
   `image_processing.cpp` — both make every later refactor safer.
3. **Performance** (`03`): presentation and recording readback are now
   asynchronous; measure frame timing before optimizing blur and CPU copies.
4. **Accessibility** (`04`): small effort, directly serves the low-vision mission.
5. **Architecture** (`02`): do the `app.cpp` decomposition *after* tests exist, ideally in
   several small extractions rather than one big-bang refactor.
6. **Robustness** (`05`): fold these into whichever module you touch along the way.
