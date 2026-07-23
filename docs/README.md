# OpenZoom Documentation Guide

OpenZoom is a Windows-only live magnification application that combines:
- Qt 6 for the desktop shell and input handling
- Media Foundation for camera discovery and frame capture
- Direct3D 12 for presentation and GPU texture management
- CUDA for optional GPU processing through D3D12 external-memory interop

## Architecture At A Glance
The current frame flow is:

1. `MediaCapture` enumerates cameras and streams frames from the selected device, preferring NV12, then YUY2, before BGRA formats. Frame ownership moves into the application handoff instead of copying the full camera buffer at each boundary.
2. NV12/YUY2 frames upload their compact raw planes directly; color conversion and rotation run in CUDA. All formats first pack into a two-slot page-locked host ring, whose per-slot CUDA event makes the H2D copies genuinely asynchronous and guards reuse. `CpuFramePipeline` converts/rotates only the remaining formats, the debug composite view, GPU-unavailable passthrough, and the per-frame fallback.
3. `CudaInteropSurface` runs the GPU effect chain when the interop surface is
   valid and publishes the completed camera generation as a persistent scene
   texture. Stateful effects, SuperRes inference, recording, OCR, and
   assistive scheduling advance only on this camera clock.
4. `PipelineOrchestrator` runs a separate viewport clock. During pan, zoom,
   focus animation, or resize it can re-present the latest complete scene up
   to 120 FPS or the active display rate; while idle it reduces to the camera
   rate. This improves navigation smoothness without synthesizing camera
   frames or rerunning temporal effects.
5. `D3D12Presenter` keeps its swap chain at the render window's native pixel
   size and draws the persistent scene through one canonical `ViewTransform`.
   Fill crops uniformly and Fit letterboxes uniformly, so camera frames are
   never independently stretched on X or Y. Its frame-slot signal is folded
   back into the same strictly increasing CUDA/D3D12 fence timeline after
   every present, including viewport-only motion. Recording and periodic assistive
   grabs use the asynchronous readback ring (`RequestReadback` /
   `TryGetCompletedReadback`), while photos and on-demand analysis use
   synchronous readback that queues a wait for the latest CUDA completion
   before copying. Request ids match every processed recording frame to its
   original camera frame.
6. Two `VideoRecorder` instances encode synchronized original/processed output live through Media Foundation. Both probe AV1 first, fall back together to H.264, and write fragmented MP4 with free-disk-space guards.

CUDA is the processing path and the CPU effects pipeline is deprecated: when the GPU pipeline is unavailable the app presents unprocessed passthrough video with a persistent "GPU required" notice instead of running effects on the CPU. The debug composite view remains CPU-only as a diagnostic.

The UI now has two states:
- Simple: a full-size live view with three flush, auto-fading primary clusters,
  contextual keystone history controls, a numbered quick-mode grid, and large
  visual/accessibility mode announcements
- Advanced: the same live view beside a narrow inspector with separate `Image`
  and `Assistant` tabs, wrapping section arrows, a full-width AI Settings row
  below the tabs, and Image-side pipeline diagnostics; Assistant provides subscription
  status, camera-aware chat, and OpenZoom-owned history

## Module Map
- `src/app` / `include/openzoom/app`: composition root plus focused pipeline,
  recording, settings, UI-state, assistive, and interaction managers. The
  `OpenZoomApp` implementation is split by responsibility across `app_*`
  translation units.
- `src/capture` / `include/openzoom/capture`: Media Foundation camera enumeration, mode discovery, and capture.
- `src/common` / `include/openzoom/common`: CPU image conversion/effects,
  canonical aspect/view transforms, frame pipeline, and media writing.
- `src/d3d12` / `include/openzoom/d3d12`: swap chain, upload, presentation, and texture readback.
- `src/cuda` / `include/openzoom/cuda`: CUDA interop surface, kernels, and fence synchronization.
- `src/ui` / `include/openzoom/ui`: Qt widgets, overlays, and event routing.

## Build Matrix
- `scripts/build_and_run.bat`: default local Windows build and launch helper.
- `scripts/build_release_bundle.bat`: packages a distributable `dist/OpenZoom`
  folder and explicitly enables CUDA plus the runtime-loaded Text-SR adapter
  unless either option is overridden in the environment.
- `scripts/run_minimal_test.bat`: builds the app without launching it, then runs the DX12/CUDA sandbox harness when its `CMakeLists.txt` is present; otherwise it reports a successful optional-harness skip.
- `cmake/CMakePresets.json`: includes `msvc-debug`, `msvc-release`, and `msvc-cpu`.

Core CMake options:
- `OPENZOOM_ENABLE_CUDA=ON|OFF`
- `OPENZOOM_ENABLE_TESTS=ON|OFF`
- `OPENZOOM_ENABLE_TEXT_SR=ON|OFF` (runtime-only NVIDIA Maxine SuperRes adapter;
  enabled by CUDA presets and disabled by the CPU preset)

When operating from the WSL/Linux agent shell, invoke Windows-side tooling with
PowerShell 7 via `pwsh.exe -NoProfile -Command '...'`, for example
`pwsh.exe -NoProfile -Command 'Get-Date'`. This PowerShell 7 bridge can also
run the Windows build helpers, such as
`pwsh.exe -NoProfile -Command '& .\scripts\build_and_run.bat'`; do not use the
legacy `powershell.exe` bridge.

## Runtime Behavior
- Camera modes are listed in the UI for the selected device.
- Global Advanced viewport controls select aspect-safe Fill/crop or
  Fit/letterbox plus Auto-up-to-120, fixed 60/90/120, or Match-display motion.
  Explicit rates clamp to the monitor and are reported once. Diagnostics
  distinguish camera FPS, measured/target viewport FPS, display Hz, viewport
  pixels, scene pixels, framing mode, and missed presents.
- Camera mode probes release their Media Foundation activation session before capture starts, so camera restarts and virtual-camera sources receive a fresh media source.
- Mid-stream device loss triggers an automatic reconnect state machine (2s/4s/8s backoff for about 30 seconds) that re-finds the same physical device by symbolic link; no modal dialogs appear while it runs. Unsupported dynamic format changes and startup failures are reported in plain language in the status label.
- The Simple interaction surface is a bottom-left preset carousel and temporary
  tile grid; each preset maps to a full advanced configuration. Keys `1`-`9`
  activate the first nine entries, including the built-in `Projector Screen`
  and `Whiteboard` modes.
- Simple chrome fades after about five seconds idle and returns on mouse,
  keyboard, focus, or application activity. Mode changes produce a centered
  toast plus a Qt accessibility announcement; they do not start speech.
  Qt and native activity notifications share a fast path while chrome is
  already visible, so high-frequency pointer input does not repeatedly move
  or raise its owned tool windows and starve viewport presentation.
- Advanced edits update a live config and can be promoted into user-defined quick modes without hiding the camera.
- Advanced places the global Virtual Joystick control near the top, provides a
  question-mark Help window ordered as Controls then Features, and offers
  Reset Tuning for restoring profile-owned values without changing global
  device or interaction choices.
- Keystone correction retains up to 32 accepted warps. Previous freezes and
  restores older history, Stop/Continue controls live detection, and Next
  restores newer history or requests one fresh correction while stopped.
- Text Clarity runs after stabilization/keystone and before legacy BW/spatial
  sharpening. A shared pitched GPU workspace supplies local luma mean and
  variance to background flattening, Sauvola, glare suppression, focus
  scoring, and the stroke mask. The `Document` preset enables the complete
  classical stack. Simple auto mode classifies paper/board/mixed content from
  the device histogram, replacing paper/board with the reading mask while
  preserving natural color for mixed scenes.
- At zoom levels of 1.33x or more, profile-owned NVIDIA Super Resolution can
  replace the NIS/FSR stage. The wrapper loads NVIDIA Video Effects only at
  runtime, resolves models from the runtime's `models` subdirectory, and runs
  a supported 4/3x pass on the shared CUDA stream; additional magnification is
  applied afterward. It performs no host frame readback and returns to NIS
  when unavailable or when its 60-frame steady-state average exceeds the 24 ms
  latency target. Ten warmup samples are excluded. Advanced shows the source
  crop, viewport target, final zoom, and measured average. Maximum source
  detail is the default. Optional `Faster 2x mode (narrower view)` uses at
  least a 2x stage, changing a 1280x720 target's visible source crop from
  960x540 to 640x360. This is explicitly a speed/field-of-view tradeoff. When
  latency is the only problem, a compact checkbox can keep SuperRes active
  until the feature is turned off. Optional `Ultra quality (full frame, up to
  1440p)` creates a distinct high-resolution D3D12/CUDA cache and applies
  viewport zoom/crop only after full-frame inference: 720p -> 1440p uses 2x,
  1080p -> 1440p uses 4/3x, and native 1440p input bypasses unnecessary AI
  enlargement. The ordinary scene and cache dimensions follow the negotiated
  post-rotation camera mode; 720p is not an internal fixed resolution.
- Focus scoring reduces Laplacian statistics on-device and asynchronously
  copies only two floats about every 15 frames; no image readback or render
  stall is introduced. A low score suppresses OCR submission and shows a
  refocus prompt.
- Camera selection and orientation are global. Stabilization, display colors,
  contrast, sharpening, zoom, and other image treatment are profile-owned.
  Display Colors uses an accessible compact swatch grid and a 256-entry GPU
  luma LUT; custom 2-8 stop gradients/posterize schemes persist globally for
  reuse by profiles. Its native popover has an opaque backing surface, so
  content beneath it never shows through the swatch and editor controls. Wheel
  scrolling never edits selectors or sliders.
- Orientation is applied before the rest of the processing pipeline.
- Settings persist to `%APPDATA%\OpenZoom\OpenZoom\settings.json`.
- Snapshots are saved as timestamp-matched `_original.jpg` and `_processed.jpg`
  pairs under `output/img/`.
- Recordings are saved as timestamp-matched `_original.mp4` and
  `_processed.mp4` pairs under `output/vid/`; encoding is live AV1 when
  available and otherwise live H.264.
- The processing status label under Advanced Image diagnostics distinguishes CPU, GPU, fallback, debug-view, recording, OCR, and VLM states without covering the Simple camera view.
- The Advanced inspector uses a draggable high-contrast splitter and persists
  its width. Text-clarity and display sliders reflow beneath their labels when
  the inspector is narrow, and feature status labels wrap within the panel.
- OCR runs locally through configured, Setup Assistant-managed, PATH, or
  standard-install Tesseract discovery. The first-run Setup Assistant is
  non-blocking, verifies pinned vendor downloads, manages Tesseract removal,
  installs or updates the official Codex CLI, and hides its NVIDIA row on
  unsupported hardware. Tesseract uses the UB Mannheim
  GitHub release asset first; a failed Qt transfer is retried with Windows
  `curl.exe` and then the vendor's alternate host, with the same mandatory
  SHA-256 check on every path. The release bundle contains neither Tesseract
  nor NVIDIA Video Effects binaries.
- Scene Explain defaults to a native Qt JSON-RPC client for `codex app-server`, reusing a ChatGPT-managed Codex login. Simple Explain threads are ephemeral and always restricted. Advanced Assistant threads are persistent and can opt into internet access or workspace-scoped coding; only OpenZoom-created thread ids are indexed in settings. OpenAI-compatible HTTP servers remain an optional fallback.
- The streamed result panel is an owned floating tool window with native
  move/resize handling over the D3D camera surface. Streamed fragments update
  its text without reapplying geometry. Its camera-relative geometry persists
  across restarts, and its first-use position clears the top Simple controls.
  Its follow-up field remains editable during a streamed answer while Ask and
  Enter submission remain blocked; once ready, it attaches the current view
  and sends questions into the shared persistent Advanced Assistant
  conversation.
- Lecture notes are valid per-session HTML documents under `output/notes/`.
  They collect timestamped OCR text, scene explanations, and relative captured
  image links that render in a browser and remain portable with `output/`.
- AI Settings uses a bounded, vertically scrollable dialog with distinct Codex,
  OpenAI-compatible VLM, OCR, speech, and notes sections. It displays the
  built-in OpenZoom Codex instruction read-only and persists separate user
  preferences for response language, tone, and detail. Those preferences are
  added to Codex developer instructions without weakening its permission
  policy and become a system message for the OpenAI-compatible fallback. The
  image-capable model list and per-model reasoning efforts are populated
  dynamically from Codex app-server `model/list`; a configured model missing
  from the current catalog remains identified as unavailable.
- Read Aloud is manual-only and uses Qt TextToSpeech over the Windows Runtime
  backend when available. AI Settings lists all voices exposed to desktop apps
  by that backend across installed languages, then persists the selected voice
  and speed. Windows 11 Narrator/Magnifier Natural voice packages are not
  exposed by the public Windows Runtime speech API and are not selectable here.
- `Setup & Downloads` in Advanced reopens the dependency assistant at any time.
  Dismissing its automatic first-run prompt is persisted independently of
  manually reopening it.

## Documentation Index
- [`README.md`](../README.md): top-level project overview and usage.
- [`docs/code_reference.md`](code_reference.md): authoritative file/class map.
- [`docs/ui_modes_design.md`](ui_modes_design.md): Simple/Advanced layout and settings-ownership contract.
- [`docs/hardcoded_paths.md`](hardcoded_paths.md): machine defaults and magic values.
- [`docs/progress.md`](progress.md): implementation tracker.
- [`docs/ai_upscaling_todo.md`](ai_upscaling_todo.md): future GPU upscaling plan.
- [`docs/THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md): third-party attribution and redistribution notes.

## Current Gaps
- Automated tests are still limited.
- CUDA interop still needs broader hardware validation across more driver/toolkit combinations.
- OCR quality depends on a user-installed Tesseract runtime and the quality of
  the processed frame fed into it.
- Maxine SuperRes requires a supported NVIDIA GPU and the user-installed
  NVIDIA Video Effects runtime; hardware/runtime and visual-quality validation
  remains necessary across Turing, Ampere, Ada, and Blackwell systems.
- SuperRes inference follows NVIDIA's synchronous sample path on OpenZoom's
  CUDA stream. Its enhanced frame is the sole zoom result rather than a layer
  blended over a separately timed conventional zoom frame. Additional zoom
  uses the live focus point mapped into the clamped 4/3x source crop.
- Subscription-backed AI depends on a compatible installed Codex CLI and
  available account usage. Setup can install or update the per-user official
  CLI from a pinned, verified OpenAI bootstrap script; authentication remains
  an explicit `Connect ChatGPT` action. The fallback VLM mode depends on a
  user-provided OpenAI-compatible `chat/completions` server.
