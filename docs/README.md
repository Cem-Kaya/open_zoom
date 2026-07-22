# OpenZoom Documentation Guide

OpenZoom is a Windows-only live magnification application that combines:
- Qt 6 for the desktop shell and input handling
- Media Foundation for camera discovery and frame capture
- Direct3D 12 for presentation and GPU texture management
- CUDA for optional GPU processing through D3D12 external-memory interop

## Architecture At A Glance
The current frame flow is:

1. `MediaCapture` enumerates cameras and streams frames from the selected device.
2. `CpuFramePipeline` converts camera frames into BGRA, applies rotation, and either builds the CPU output directly or prepares input for CUDA.
3. `CudaInteropSurface` runs GPU effects when the interop surface is valid.
4. `D3D12Presenter` presents the final frame and can read back GPU textures for recording.
5. `VideoRecorder` writes processed output to H.264 MP4 through Media Foundation.

The app is intentionally resilient: debug view is CPU-only, and the normal view falls back to the CPU path whenever the CUDA path cannot run.

The UI now has two states:
- Simple: a full-size live view with three flush, auto-fading corner clusters,
  a numbered quick-mode grid, and large visual/spoken mode announcements
- Advanced: the same live view beside a narrow inspector with separate `Image`
  and `Assistant` tabs, wrapping section arrows, a labeled top-level AI Settings
  button, and Image-side pipeline diagnostics; Assistant provides subscription
  status, camera-aware chat, and OpenZoom-owned history

## Module Map
- `src/app` / `include/openzoom/app`: application lifecycle, settings persistence, and interaction control.
- `src/capture` / `include/openzoom/capture`: Media Foundation camera enumeration, mode discovery, and capture.
- `src/common` / `include/openzoom/common`: CPU image conversion/effects, frame pipeline, and media writing.
- `src/d3d12` / `include/openzoom/d3d12`: swap chain, upload, presentation, and texture readback.
- `src/cuda` / `include/openzoom/cuda`: CUDA interop surface, kernels, and fence synchronization.
- `src/ui` / `include/openzoom/ui`: Qt widgets, overlays, and event routing.

## Build Matrix
- `scripts/build_and_run.bat`: default local Windows build and launch helper.
- `scripts/build_release_bundle.bat`: packages a distributable `dist/OpenZoom` folder.
- `scripts/run_minimal_test.bat`: builds the app without launching it, then runs the DX12/CUDA sandbox harness when its `CMakeLists.txt` is present; otherwise it reports a successful optional-harness skip.
- `cmake/CMakePresets.json`: includes `msvc-debug`, `msvc-release`, and `msvc-cpu`.

Core CMake options:
- `OPENZOOM_ENABLE_CUDA=ON|OFF`
- `OPENZOOM_ENABLE_TESTS=ON|OFF`

When operating from the WSL/Linux agent shell, invoke Windows-side tooling with
PowerShell 7 via `pwsh.exe -NoProfile -Command '...'`, for example
`pwsh.exe -NoProfile -Command 'Get-Date'`. This PowerShell 7 bridge can also
run the Windows build helpers, such as
`pwsh.exe -NoProfile -Command '& .\scripts\build_and_run.bat'`; do not use the
legacy `powershell.exe` bridge.

## Runtime Behavior
- Camera modes are listed in the UI for the selected device.
- Camera mode probes release their Media Foundation activation session before capture starts, so camera restarts and virtual-camera sources receive a fresh media source.
- Mid-stream camera failures and unsupported dynamic format changes stop the affected session and remain visible in the status label and camera error dialog.
- The Simple interaction surface is a bottom-left preset carousel and temporary
  tile grid; each preset maps to a full advanced configuration. Keys `1`-`7`
  activate the first seven entries.
- Simple chrome fades after about five seconds idle and returns on mouse,
  keyboard, focus, or application activity. Mode changes produce a centered
  toast plus a Qt accessibility announcement; they do not start speech.
- Advanced edits update a live config and can be promoted into user-defined quick modes without hiding the camera.
- Camera selection and orientation are global. Stabilization, display colors, contrast, sharpening, zoom, and other image treatment are profile-owned.
- Orientation is applied before the rest of the processing pipeline.
- Settings persist to `%APPDATA%\OpenZoom\OpenZoom\settings.json`.
- Snapshots are saved to `output/img/`.
- Recordings are saved to `output/vid/`.
- The processing status label under Advanced Image diagnostics distinguishes CPU, GPU, fallback, debug-view, recording, OCR, and VLM states without covering the Simple camera view.
- OCR runs locally through configured, bundled, PATH, or standard-install Tesseract discovery. The Windows release bundler includes an installed Tesseract runtime and language data.
- Scene Explain defaults to a native Qt JSON-RPC client for `codex app-server`, reusing a ChatGPT-managed Codex login. Simple Explain threads are ephemeral and always restricted. Advanced Assistant threads are persistent and can opt into internet access or workspace-scoped coding; only OpenZoom-created thread ids are indexed in settings. OpenAI-compatible HTTP servers remain an optional fallback.
- The streamed result panel is an owned floating tool window with native
  move/resize handling over the D3D camera surface. Streamed fragments update
  its text without reapplying geometry. Its follow-up field attaches the current
  view and sends questions into the shared persistent Advanced Assistant
  conversation.
- AI Settings persists shared Assistant Instructions for response language,
  tone, and detail. These preferences are added to Codex developer instructions
  without weakening its permission policy and become a system message for the
  OpenAI-compatible fallback. Codex defaults to `gpt-5.5` with `xhigh`
  reasoning, then uses the app-server's image-capable default when unavailable.
- Read Aloud is manual-only and uses Qt TextToSpeech over the Windows Runtime
  backend when available. AI Settings lists all voices exposed to desktop apps
  by that backend across installed languages, then persists the selected voice
  and speed. Windows 11 Narrator/Magnifier Natural voice packages are not
  exposed by the public Windows Runtime speech API and are not selectable here.

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
- OCR quality depends on an external Tesseract installation and the quality of the processed frame fed into it.
- Subscription-backed AI depends on a compatible installed Codex CLI and available account usage. The fallback VLM mode depends on a user-provided OpenAI-compatible `chat/completions` server.
