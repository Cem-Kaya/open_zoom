# Review Findings — Batch C/D + Plan 15 Drop (2026-07-23)

Independent review of the working tree as of the 18:12 build (the large drop
containing the app decomposition, the include/openzoom header tree, the tests,
and the plan-15 aspect-safe high-refresh viewport). Four review passes ran over
(1) app decomposition fidelity, (2) UI/shutdown safety, (3) the GPU/fence
architecture, (4) tests/CMake/docs/hygiene, plus independent crash forensics on
the two 2026-07-23 evening minidumps and six launch/run/close smoke cycles.

Overall verdict: **high-quality drop; accepted with two GPU-sync defects to fix
before the next release build.** The decomposition is behavior-preserving (no
dropped logic found), the capture-thread contract and teardown discipline
survived, tests are genuinely behavioral, docs match the new layout, and the
shutdown crashes are explained and fixed. The items below are ordered by
priority for the implementing agent.

## Implementation follow-up — 2026-07-23

- **F1 fixed:** graphics reservations now seed from the presenter's latest
  signal, `GraphicsSignaled(...)` adopts the actual internal frame-slot value
  after every successful present, and the presenter defensively chooses a
  value above its current signal and queued CUDA wait. A regression test covers
  camera, graphics, viewport-only graphics, and next-camera ordering.
- **F2 fixed:** `ReadbackTexture(...)` accepts a CUDA wait value and queues
  that shared-fence dependency before its copy. Both on-demand OCR/Explain and
  Assistant frame attachment pass `LastCudaSignal()`.
- **F3/F4 fixed in the same pass:** the no-fence CUDA fallback clamps active
  presentation to camera rate, and elapsed interaction timing uses
  nanoseconds.
- The Simple-mode performance regression found during validation was separate
  from the GPU races: duplicate Qt/native mouse activity repeatedly recomputed,
  moved, and raised every floating chrome window. Already-visible chrome now
  takes a throttled idle-deadline-only path.

---

## P0 — Fence protocol gap between the camera and viewport clocks (F1)

**Verified in code by two independent readers. This is the one serious defect
in the drop.**

- `D3D12Presenter::PresentSceneTexture` signals the shared fence **twice** per
  present: the caller's `fenceSync->signalValue`, then an internal slot-pacing
  value `slotSignal = ++fenceValue_` (src/d3d12/presenter.cpp:398-410).
- `FenceSequencer::BeginGraphicsFrame()` is const and reserves nothing;
  `GraphicsSignaled()` advances `nextValue_` by only 1
  (include/openzoom/app/pipeline_orchestrator.hpp:52-58). The sequencer never
  learns about the slot signal unless `ReadbackObserved(GetLastSignaledFenceValue())`
  runs — and that call sits inside `if (readbackRequestId != 0)`
  (src/app/app_pipeline_runtime.cpp:555-568).
- Consequence 1 (write-after-read): during plain pan/zoom with no recording,
  assistive capture, or photo pending, a viewport-only re-present is fenced by
  a value the presenter **already signaled** for the previous present's slot
  pacing. The next `BeginCudaFrame` then computes a wait that completed before
  the re-present's draw even executed, so the CUDA kernel chain may overwrite
  the shared texture (and the SuperRes ROI cache) while that draw still samples
  it. Expect rare one-frame tearing/garbage under fast panning; worse on slower
  GPUs where draws retire later.
- Consequence 2 (fence rewind): D3D12 `Signal` sets the value with no
  max-semantics; the duplicate signal can execute after CUDA has device-side
  signaled a higher value, transiently rewinding the fence and un-signaling
  values other waits depend on — mis-ordered waits or stalls.
- This is a plan-15 regression: the double-signal existed at HEAD but presents
  happened exactly once per CUDA frame, so no value was ever pre-signaled.

**Fix (small, localized — pick one):**
1. Call `Fence().ReadbackObserved(presenter_->GetLastSignaledFenceValue())`
   after **every** successful `PresentSceneTexture`, not only when a readback
   was requested (hoist it out of the `readbackRequestId != 0` block); or
2. Make `BeginGraphicsFrame` reserve from the presenter's
   `GetLastSignaledFenceValue()` the way `BeginCudaFrame` already does, so
   graphics signal values are always fresh.
Option 1 is a two-line change and restores both strict monotonicity and
coverage of the newest draw.

## P0 — On-demand assistive readback races CUDA writes (F2)

`AssistiveFeatureManager`'s on-demand path calls
`presenter_->ReadbackTexture(cudaSharedTexture_.Get(), ...)`
(src/app/app_assistant.cpp:49, 108). `ReadbackTexture`
(src/d3d12/presenter.cpp:836-940) enqueues `CopyTextureRegion` and then
`WaitForGpu()` — which waits for the *copy*, but never queues a
`commandQueue_->Wait(fence_, LastCudaSignal())` **before** the copy. With fence
interop active, `ProcessFrame` returns while kernels are still queued, so the
copy can capture a half-written frame → torn OCR/VLM input. The periodic
assistive path through the fenced readback ring is correct; only this blocking
path is exposed. Fix: plumb a wait value into `ReadbackTexture` and queue the
Wait before the copy.

## P1 — Robustness items worth doing in the same pass

- **AssistiveFeatureManager teardown (M2):** `OverlayUpdated` connects a lambda
  with context `overlay_` but capturing raw `this`
  (src/app/assistive_feature_manager.cpp:29-34). During
  `assistiveManager_.reset()`, `~AssistiveRuntime` kills the OCR process with
  `waitForFinished`, whose synchronous handlers can re-emit `OverlayUpdated`
  into the half-destroyed manager. Works today by luck of member layout. Fix:
  first line of `~AssistiveFeatureManager`:
  `QObject::disconnect(runtime_.get(), nullptr, overlay_, nullptr);`
- **Dual ownership (M3):** three objects are both `unique_ptr`-owned and
  Qt-parented: `AssistiveRuntime` (manager + app parent,
  assistive_feature_manager.cpp:27), `codexClient_` and `ocrProcess_` (runtime
  unique_ptrs + runtime parent, assistive_runtime.cpp:148-149). Safe only
  because explicit resets always precede the parent sweep. Drop one owner in
  each pair, or add a loud comment at each site (pattern of
  verified-non-issues #1).
- **Release-build widget-bind guards:** `UIStateManager` binds ~85 widget
  pointers with `Q_ASSERT_X` only (src/app/ui_state_manager.cpp:31-33) and
  call sites dropped the old null checks (e.g. app_bootstrap.cpp:90-91). Add a
  release-mode fail-fast (log + abort init) on any null bind.
- **Degraded-fence mode now stalls per viewport tick (F3):** with fence interop
  off (unsupported driver, or permanently after a 3-strike resync),
  `PresentSceneTexture` takes the full `WaitForGpu()` drain path
  (presenter.cpp:418-423) at up to 120 Hz on the UI thread. Clamp the viewport
  rate to camera rate when `FenceInteropEnabled()` is false.
- **Swallowed photo capture (decomposition finding 3):** if
  `PrepareOriginalFrame` fails or the camera is inactive, a pending photo
  request retries forever with no user feedback (app_pipeline_runtime.cpp:521-523,
  877-882). Old code fell back to the last presentation buffer. Add a timeout +
  status message, or restore the fallback.

## P2 — Infrastructure and polish

- **Test presets are false-green no-ops:** `msvc-debug-tests` /
  `msvc-release-tests` point at configure presets with
  `OPENZOOM_ENABLE_TESTS=OFF` (cmake/CMakePresets.json:19, 33, 67-73) — ctest
  finds zero tests and exits 0. The test TUs are CUDA-free, so just flip
  `OPENZOOM_ENABLE_TESTS=ON` in all three configure presets (also fixes the
  "tests live only in the deprecated CPU preset" tension) and add
  `"execution": {"noTestsAction": "error"}` to every test preset.
- **Missing tests where the CHANGELOG claims most:** legacy color modes 2-16 →
  scheme migration (`LegacyColorScheme` fallback,
  src/app/settings_store.cpp:56-93) has zero coverage; `BuildColorLut`
  endpoints/stepped boundaries untested; stop-count edges (>8 truncated,
  1-stop fallback, malformed `#rrggbb`) untested; the
  `mlTextSuperResolution*` key aliasing untested. All are cheap QtTest cases in
  the existing binaries.
- **Stale backlog docs:** improvement_ideas/02 still says app.cpp "is 3,396
  lines and keeps growing"; improvement_ideas/README still lists the
  decomposition as open; 11-hardening-refactor-plan.md line ~650 leaves the
  settings-tests checkbox unchecked though done. Mark them DONE — the README's
  own rule requires it.
- **Viewport tick timing (F4):** `viewportTickTimer_.restart()` returns ms;
  at 120 Hz that alternates 8/9 ms → ±6% pan-speed jitter. Use
  `nsecsElapsed()` (src/app/pipeline_orchestrator.cpp:295-298).
- **No re-present when camera is off (F6):** `RunFrameTick` returns before the
  re-present branch when `cameraActive_` is false (app_pipeline_runtime.cpp:841-843),
  so resizing after stopping the camera can leave stale/undefined back
  buffers — violates plan 15's own resize acceptance for that state.
- **SuperRes ROI `generation` tag is set but never compared**
  (src/cuda/cuda_interop.cpp:1344). Staleness is currently prevented
  structurally; either wire the check into `PresentLatestCudaScene` or delete
  the field so it can't rot into a false guarantee.
- **Stale comment:** "S6b contract in app.hpp" (app_pipeline_runtime.cpp:406) —
  the contract lives in pipeline_orchestrator.hpp now.
- `settings_store.cpp` is wrapped in `#ifdef _WIN32` though fully portable —
  drop the guard so the test suite stays host-independent.
- CPU-fallback re-present resamples the full viewport per tick on the UI
  thread (app_pipeline_runtime.cpp:859-871) — fine to leave if the CPU path is
  being deprecated; otherwise clamp its viewport rate.

## Resolved by this drop (for the record)

- **The 2026-07-23 crash cluster is explained.** Minidump forensics (WER
  ReportQueue dumps, 17:59 and 18:04): both were access violations through a
  **null object pointer inside a slot invoked from `QMetaObject::activate`** —
  signals delivered synchronously into objects mid-destruction, the concrete
  case being `QProcess::waitForFinished` in `~CodexAppServerClient` pumping the
  final `finished`/`error` callbacks (at HEAD this null-deref'd
  `assistiveRuntime_->IsCodexTurnActive()` on every exit with a running Codex
  child). The fix set — `QSignalBlocker` across `Shutdown()`
  (codex_app_server_client.cpp:108-115) plus the disconnect-all sweep in
  `~OpenZoomApp` (app_bootstrap.cpp:730-739) — is correct and triple-redundant.
  Six launch/run/close smoke cycles on the 18:12 build: clean exit 0, zero new
  WER events.
- The 0xc0000374 heap-corruption events (03:57, 04:13-04:18) are the same
  family if they occurred at close (dying-object signal writes are exactly the
  class that corrupts heap metadata); no reproduction on the current build.
  Keep half an eye open: if a 0xc0000374 recurs **mid-session** (not at exit),
  the next suspect is the frame pipeline (S1 successor paths: presenter
  readback ring teardown, recorder finalize), ideally under AppVerifier.

## Positive findings (don't re-litigate)

- Decomposition fidelity: every old `OpenZoomApp::` method accounted for;
  constructor/Initialize sequence preserved ~1:1; capture callback still
  touches only `latestFrame_` under `cameraMutex_` + a queued marshal
  (verified-non-issues #6 still holds); FenceSequencer encodes the old inline
  fence dance exactly (3-strike resync is the documented S6b policy).
- Teardown invariants hold: orchestrator stopped first, `WaitForIdle` before
  releasing shared textures, `SynchronizeStream` before destroying CUDA
  surfaces/external memory.
- view_transform math (Fill/Fit, pan clamping, rotation via swapped extents,
  SuperRes ROI containment remap) verified correct; its unit tests are the
  strongest in the suite (pixel-level property tests across five aspect
  ratios, ROI accept and reject paths).
- LUT invariant holds: 256-entry `__constant__` upload only on generation
  change; legacy modes 2-16 are 2-stop smooth duotones; modes 0/1 bypass.
- Temporal purity holds: viewport re-present is a pure resample; all stateful
  stages are gated on `newCameraFrame`. Presentation is single-threaded (one
  UI-thread timer drives both clocks) — no concurrent Present.
- Accessibility preserved and extended: announcement events (assertive, no
  TTS), full `setAccessibleName` sweep incl. new controls, wheel-safe
  subclasses everywhere (zero plain QComboBox/QSlider/QSpinBox in src/ui +
  src/app), keyboard grid navigation in the picker, TTS still strictly
  user-triggered.
- CHANGELOG spot-checks: every verifiable claim backed by code.
- Hygiene: no secrets in new files; NVIDIA-EULA material confined to the
  gitignored `redist/`; lucide icons carry their license.

## Reviewer-applied changes in this pass (already in the tree)

1. `src/ui/color_scheme_picker.cpp` — popup no longer auto-hides (and Escape
   no longer double-fires) while its own modal color dialog is open (M1).
2. `.gitignore` — `Maxine-VFX-SDK/samples/` + `resources/` excluded so the
   repo keeps only the MIT nvvfx header snapshot (~130 KB) instead of 97 MB of
   sample apps/media.
3. `CHANGELOG.md` — entry for the M1 fix.
