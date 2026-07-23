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
| [`02-architecture-app-decomposition.md`](02-architecture-app-decomposition.md) | God-object breakup, state machines | Decompose 3,396-line `app.cpp` |
| [`03-performance-gpu-cpu.md`](03-performance-gpu-cpu.md) | Frame latency and throughput | Pinned-host upload staging ring (P11) |
| [`04-accessibility-ux.md`](04-accessibility-ux.md) | Accessibility (the product mission) | Respect system High Contrast / palette (U2) |
| [`05-robustness-validation.md`](05-robustness-validation.md) | Input validation, error handling, I/O | Settings versioning/migration (V1) |
| [`06-build-tooling-docs.md`](06-build-tooling-docs.md) | Build system, CI, tests, docs hygiene | Compiler warnings + first test target |
| [`07-text-clarity-plan.md`](07-text-clarity-plan.md) | GPU document and text enhancement | Adaptive binarization and background flattening |
| [`08-ml-text-sr-options.md`](08-ml-text-sr-options.md) | ML text super-resolution research | Choose a practical runtime/model path |
| [`09-maxine-superres-plan.md`](09-maxine-superres-plan.md) | NVIDIA Video Effects SuperRes | Runtime-loaded, benchmark-gated integration |
| [`10-vendor-independent-gpu.md`](10-vendor-independent-gpu.md) | Cross-vendor GPU direction | Portable processing backend plan |
| [`11-hardening-refactor-plan.md`](11-hardening-refactor-plan.md) | Stability, performance, and refactor waves | Measured pipeline hardening |
| [`12-color-picker-redesign.md`](12-color-picker-redesign.md) | Accessible display-color selection | Visual two-color scheme picker |
| [`13-handoff-batches-bcd.md`](13-handoff-batches-bcd.md) | Ordered implementation handoff | Batches B, C, then D |
| [`14-stabilization-v2.md`](14-stabilization-v2.md) | Robust camera stabilization | Similarity tracking and screen lock |
| [`15-aspect-safe-high-refresh-viewport.md`](15-aspect-safe-high-refresh-viewport.md) | Viewport geometry and motion | Aspect-safe Fill plus 120 FPS navigation |
| [`16-review-findings-2026-07-23.md`](16-review-findings-2026-07-23.md) | Batch C/D + plan 15 review verdict | Fence fixes done; P1/P2 follow-ups |
| [`17-project-rename-plan.md`](17-project-rename-plan.md) | Project rename (FrontRow blocked) | Clear a name, migrate settings, then mechanical rename |
| [`verified-non-issues.md`](verified-non-issues.md) | Refuted findings — do not "fix" these | — |

## Suggested implementation order

1. **Stability first** (`01`): the camera-switch lifetime bug, COM leak, and
   mid-stream format handling are complete; verify the remaining capture/UI
   threading and fence-failure recovery items before broad refactors.
2. **Cheap guardrails** (`06`): enable `/W4` warnings and add a first unit-test target for
   `image_processing.cpp` — both make every later refactor safer.
3. **Performance** (`03`): presentation and recording readback are now
   asynchronous; add frame timing (P8), then take the biggest remaining win —
   the pinned-host upload staging ring (P11) — before optimizing blur and CPU
   copies.
4. **Accessibility** (`04`): small effort, directly serves the low-vision mission.
5. **Architecture** (`02`): do the `app.cpp` decomposition *after* tests exist, ideally in
   several small extractions rather than one big-bang refactor.
6. **Robustness** (`05`): fold these into whichever module you touch along the way.
