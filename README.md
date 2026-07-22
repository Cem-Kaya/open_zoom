# OpenZoom

OpenZoom is a Windows-only camera magnifier built around Qt 6, Media Foundation, Direct3D 12, and an optional CUDA processing path. The current codebase already supports live camera capture, CPU and GPU frame processing, a two-stage preset/advanced UI, rotation-aware presentation, persistent settings, photo snapshots, and H.264 MP4 recording.

## Current Capabilities
- Media Foundation camera enumeration with per-device mode listing (`width x height @ fps`), restart-safe device activation, plain-language failure reporting, and automatic reconnection: when a camera drops mid-lecture, OpenZoom quietly retries the same physical device for about 30 seconds (2s/4s/8s backoff) without any modal dialogs, and only reports failure if the device never comes back.
- CPU frame pipeline for format conversion and rotation of formats the GPU path does not consume, plus the legacy debug composite view. NV12 and YUY2 camera frames bypass it entirely: color conversion and rotation run in CUDA.
- Direct3D 12 presenter for swap-chain output plus GPU texture readback. Recording and the periodic assistive grab use an asynchronous readback ring that never blocks the render loop; photos and on-demand analysis use the synchronous path.
- CUDA external-memory interop path with GPU color conversion (NV12/YUY2) and rotation, video stabilization, automatic keystone correction for projected screens, black-and-white, zoom, Gaussian blur, temporal smoothing, auto contrast (percentile level stretch), low-vision display color modes with contrast/brightness, focus marker, and spatial sharpening via NVIDIA NIS or AMD FSR 1.0 style kernels.
- GPU video stabilization (projection-profile motion estimation, fully on-device) to counter phone-on-laptop mount vibration.
- Two-speed UI: Simple mode gives the full client area to the live view and overlays three auto-fading corner control clusters; Advanced keeps the camera visible beside a narrow inspector containing every parameter and pipeline diagnostics. The chosen mode persists.
- Stage-1 quick modes backed by full stage-2 advanced configurations, including promotion of advanced tuning into user-defined quick options.
- Local OCR via Tesseract plus scene explanations through either a signed-in Codex CLI/ChatGPT subscription or an OpenAI-compatible HTTP endpoint. Results stream into a focusable assistive panel and Advanced Assistant, can be spoken aloud, and can be written to `output/notes/`. Release bundles include the installed Tesseract runtime and English data when available.
- Session persistence in `%APPDATA%\OpenZoom\OpenZoom\settings.json`.
- Processed output capture to `output/img/IMG_*.jpg` and `output/vid/VID_*.mp4` next to the executable.
- A branded, high-contrast magnifier icon embedded in the Windows executable at multiple display sizes.

## Status
CUDA is the processing path; the CPU effects pipeline is deprecated. When the D3D12/CUDA interop surface cannot initialize, the app presents unprocessed passthrough video and shows a persistent "GPU required — processing disabled (showing raw video)" notice instead of silently degrading. The CPU debug composite view remains available as a diagnostic.

## Prerequisites
- Windows 10 or Windows 11.
- Visual Studio 2022 with the Desktop C++ workload and Windows SDK.
- Qt 6.9.3 for `msvc2022_64`, or matching overrides via `QT_PREFIX` / `Qt6_DIR`.
- CMake 3.23 or newer.
- NVIDIA GPU plus CUDA Toolkit 13.x if you want the CUDA path.
- Optional: Codex CLI for subscription-backed Explain and Assistant features. OpenZoom auto-detects `codex.exe`; use AI Settings or `OPENZOOM_CODEX_PATH` to override it.

## Build And Run
From a Visual Studio x64 developer prompt or a PowerShell 7 session (`pwsh.exe`) with MSVC, Qt, and optionally CUDA on `PATH`:

```bat
scripts\build_and_run.bat
```

From the WSL/Linux agent shell, run Windows-side commands through PowerShell 7
with `pwsh.exe -NoProfile -Command '...'`, for example
`pwsh.exe -NoProfile -Command 'Get-Date'`.

The helper script:
- configures `build\` with the Visual Studio 2022 generator,
- clears stale CMake cache entries when the source path changes,
- builds `open_zoom`,
- prefers `build\cmake\Release\open_zoom.exe` when launching,
- runs `windeployqt` automatically when it can find the Qt runtime.

If Windows reports missing `Qt6*.dll` files, add the Qt `bin` directory to `PATH` or point `QT_PREFIX` / `Qt6_DIR` at the correct installation.

## Alternative Builds
### CMake presets

```powershell
cmake --preset msvc-release
cmake --build --preset msvc-release-build
```

Available presets live in [`cmake/CMakePresets.json`](cmake/CMakePresets.json):
- `msvc-debug`
- `msvc-release`
- `msvc-cpu`

### CPU-only build

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:/Qt/6.9.3/msvc2022_64" -DOPENZOOM_ENABLE_CUDA=OFF
cmake --build build-cpu
```

### Release bundle

```bat
scripts\build_release_bundle.bat
```

This produces `dist\OpenZoom\` with `open_zoom.exe`, Qt runtime files, `LICENSE`, `README.txt`, `THIRD_PARTY_LICENSES.md`, and the bundled Lucide icon notice. CUDA redistributables are copied when available. An installed Tesseract runtime is copied to `tools\tesseract\` with its language data and notices so local OCR works without a separate install on the destination machine.

## Runtime Controls
- `Simple` / `Advanced` switches between a full-view overlay UI and a right-side inspector. The live camera remains visible in both states.
- In Simple mode, the switch, profile carousel, and Photo/Record/Explain/Read actions occupy three flush view corners. Pipeline status stays in Advanced diagnostics so it never covers the camera. The solid, high-contrast chrome fades after about five seconds idle and returns on mouse, keyboard, focus, or application activity.
- `Ctrl+H` pins the Simple controls on screen or restores automatic hiding; `Esc` closes the quick-mode grid.
- The grid button or current profile opens all quick modes as large tiles. Built-in modes use plain-language labels such as `Read a Page`, `High Contrast`, `Sharpen Text`, `Keep It Steady`, `See in Low Light`, `Projector Screen`, and `Whiteboard`.
- Number keys `1` through `9` apply the first nine quick modes. A mode change shows a large centered announcement and notifies screen readers without starting speech.
- Advanced has separate `Image` and `Assistant` tabs with wrapping previous/next navigation arrows. `Image` contains device, per-profile tuning, and pipeline status; `Assistant` contains a persistent camera-aware chat plus an OpenZoom-only history list. The labeled `AI Settings` pop-out button sits in this top navigation rather than inside the Image form.
- `Advanced Tuning` expands the per-profile controls inside the Advanced Image inspector.
- `Save As Quick Option` promotes the current advanced setup into a reusable stage-1 preset.
- `Camera` selects the active Media Foundation device from Advanced. Camera selection and orientation are global rather than part of a quick profile.
- `Modes` shows the discovered capture modes for the selected camera.
- `Rotation` rotates the pipeline in 90 degree clockwise steps before downstream processing.
- `Black & White` applies thresholded monochrome conversion.
- `Zoom` enables the magnifier and focus-point controls.
- `Gaussian Blur` applies the CUDA blur stage with configurable sigma and supported discrete radii.
- `Temporal Smooth` applies an exponential running average.
- `Stabilize Image` enables GPU video stabilization; the strength slider controls how aggressively the camera path is smoothed.
- `Straighten Screen (Keystone)` automatically detects a projected slide or screen viewed at an angle and warps it fronto-parallel. Used by the `Projector Screen` and `Whiteboard` quick modes.
- `Auto Contrast` stretches washed-out projector colors using a percentile level analysis; its strength slider blends toward the full stretch.
- `Display Colors` selects a low-vision color scheme (Normal, Inverted, White on Black, Yellow on Black, Black on Yellow); `Contrast` and `Brightness` fine-tune legibility.
- `OCR Assist`, `Scene Explain`, and `Assistive Overlay` drive asynchronous assistive analysis and on-screen text overlays.
- `Read Text`, identified by a speaker icon, runs local OCR. `Explain` sends one temporary, non-history camera question and changes to `Stop` while Codex is working.
- The assistive result panel updates as an answer streams. It is an owned floating tool window: native window movement keeps dragging responsive over the D3D camera surface, and streamed text does not reset its geometry. Drag its header to move it and an edge or corner to resize it. Its text can be focused, selected, and read by a screen reader; the question field sends follow-ups into the shared persistent Assistant conversation with the current view attached. Speech starts only when `Read Aloud` is clicked and omits visible section labels such as `Scene Explain` and `OCR`, while the high-contrast Close control or `Esc` dismisses the panel.
- The Advanced Assistant can attach the current processed view, stream answers, stop a response, and manage persistent OpenZoom conversations with resume, rename, export, and delete actions.
- The Advanced Assistant subscription label reports the percentage left in the current Codex usage window.
- `Connect ChatGPT` uses the Codex app-server browser login flow. Existing Codex CLI sign-in is reused automatically.
- `AI Settings` selects Codex subscription or an OpenAI-compatible server and configures the model, reasoning level, shared Assistant Instructions, scene prompt, Tesseract, OCR language, installed Windows Read Aloud voice and speed, and lecture notes. Assistant Instructions can set a preferred response language, tone, or level of detail for both providers without changing Codex security permissions. Codex defaults to the installed app-server's current image-capable default, `gpt-5.5`, with Extra high reasoning, and falls back to the server default if that model is unavailable. OpenZoom lists every voice exposed to desktop applications by Windows Runtime speech synthesis, across installed languages. Windows 11 Narrator/Magnifier Natural voice packages are not currently exposed by that public API and therefore do not appear in OpenZoom. Advanced Assistant internet and coding permissions are separate opt-ins; coding also requires a workspace folder.
- `Open Notes` opens the current session's lecture notes markdown file from `output/notes/`.
- `Spatial Sharpen` enables the CUDA sharpening/upscaling stage and lets you choose NIS or FSR-style processing when the GPU path is active.
- `Debug View` switches to the CPU composite grid so intermediate stages can be inspected; the nearby pipeline status reports CPU/GPU, fallback, recording, OCR, and VLM state.
- `Show Focus Point` overlays the current zoom center on the presented output.
- `Capture Photo` saves the current processed frame to `output/img/`.
- `Start Recording` writes processed video to `output/vid/` as H.264 in a fragmented MP4 container, with a 12 hour cap per file. Fragments are flushed while recording, so the file stays playable up to the last completed fragment even if the app or PC dies mid-lecture. Recording refuses to start with less than 500 MB free and finalizes itself cleanly (keeping everything recorded so far) if free space drops below 200 MB; both cases are reported in the pipeline status instead of an error dialog.

## Navigation And Interaction
- `Ctrl + mouse wheel`: zoom around the cursor.
- `Mouse wheel`: pan while zoomed.
- `Middle mouse drag`: pan the zoom focus.
- Arrow keys: nudge the zoom focus.
- `1` through `9` in Simple mode: apply the corresponding numbered quick mode.
- `Tab` / `Shift+Tab` in Simple mode: move through the corner controls.
- `Virtual Joystick`: shows an on-canvas joystick overlay for panning.
- Wheel, arrow-key, joystick, and middle-drag panning keep the active quick mode selected; changing an actual Advanced processing control still creates a custom setup.

## Persistence And Output Paths
- Settings persist to `%APPDATA%\OpenZoom\OpenZoom\settings.json`. Camera and orientation are global; stabilization, colors, contrast, sharpening, zoom, and the other image treatments are stored in the current profile and user-created quick options. UI and assistive/AI configuration also persist. OpenZoom stores only an index of the persistent Codex conversations it created; Codex stores their transcripts. Temporary Simple explanations are ephemeral and do not enter history.
- Snapshots are written under `output/img/` relative to the executable.
- Recordings are written under `output/vid/` relative to the executable.
- Lecture notes are written under `output/notes/` relative to the executable.
- Manually exported Assistant transcripts default to `output/assistant/` relative to the executable.

## Assistive Runtime Configuration
The in-app `AI Settings` dialog is the primary way to configure the assistive
features; values are stored in `settings.json`. The default provider starts the
local `codex app-server` process and reuses Codex's ChatGPT-managed login. The
fallback provider accepts any OpenAI-compatible `chat/completions` server,
including local ones (LM Studio, Ollama, llama.cpp server), so image-to-text can
run fully offline. Local servers do not require an API key.

OpenZoom's Codex client starts in restricted mode: read-only sandboxing, no
approvals, no tool network access, and vision-only instructions. AI Settings
can independently allow internet access and coding for persistent Advanced
Assistant conversations. Coding requires an existing workspace folder and uses
Codex `workspaceWrite` sandboxing with that folder as the only writable root.
Simple Explain always stays restricted. MCP, dynamic, and collaboration tools
remain blocked in every mode, and unexpected approval requests are denied.
Codex app-server is still a local coding-agent process, so users should install
and update it from the official Codex distribution.

Environment variables remain as fallback for any field left empty in the
dialog:
- OCR first uses the configured path, then `OPENZOOM_TESSERACT_PATH`, a bundled `tools/tesseract/tesseract.exe`, `PATH`, or the standard Windows install directories. The release bundler copies an installed Tesseract runtime automatically; override its source with `OPENZOOM_TESSERACT_DIR` or `TESSERACT_PREFIX`.
- Codex first uses the configured CLI path, then `OPENZOOM_CODEX_PATH`, `PATH`, or `%LOCALAPPDATA%\Microsoft\WinGet\Links\codex.exe`.
- Configure VLM with:
  - `OPENZOOM_VLM_API_URL`
  - `OPENZOOM_VLM_API_KEY`
  - `OPENZOOM_VLM_MODEL`
  - optional `OPENZOOM_VLM_PROMPT`

## Repository Layout
- `src/app/` - application lifecycle, settings persistence, and interaction wiring.
- `src/capture/` - Media Foundation camera discovery and capture loop.
- `src/common/` - CPU frame pipeline, image processing helpers, and Media Foundation recording wrapper.
- `src/d3d12/` - Direct3D 12 presenter, swap chain, upload, and readback logic.
- `src/cuda/` - CUDA interop surface and kernels.
- `src/ui/` - Qt widgets, overlays, and event routing.
- `include/openzoom/` - public headers mirroring the source layout.
- `docs/` - architecture notes, code reference, progress tracking, and licensing docs.
- `scripts/` - build, bundle, and validation helpers.

## Documentation Map
- [`docs/README.md`](docs/README.md) for the architecture overview.
- [`docs/code_reference.md`](docs/code_reference.md) for the current class and file map.
- [`docs/hardcoded_paths.md`](docs/hardcoded_paths.md) for machine-specific defaults.
- [`docs/progress.md`](docs/progress.md) for implementation status.
- [`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md) for redistribution notes.

## License

This project is dual-licensed:

- GPL-3.0 via [`LICENSE`](LICENSE)
- Commercial licensing by direct arrangement with the project owner

By contributing, you agree that your changes may be distributed under both terms. Third-party notices are summarized in [`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md).
