# OpenZoom

OpenZoom is a Windows-only camera magnifier built around Qt 6, Media Foundation, Direct3D 12, and an optional CUDA processing path. The current codebase already supports live camera capture, GPU (CUDA) frame processing with CPU format-conversion support paths, a two-stage preset/advanced UI, rotation-aware presentation, persistent settings, paired photo snapshots, and live AV1/H.264 recording in fragmented MP4 containers.

## Current Capabilities
- Media Foundation camera enumeration with per-device mode listing (`width x height @ fps`), restart-safe device activation, plain-language failure reporting, and automatic reconnection: when a camera drops mid-lecture, OpenZoom quietly retries the same physical device for about 30 seconds (2s/4s/8s backoff) without any modal dialogs, and only reports failure if the device never comes back.
- CPU frame pipeline for format conversion and rotation of formats the GPU path does not consume, plus the legacy debug composite view. NV12 and YUY2 camera frames bypass it entirely: color conversion and rotation run in CUDA.
- Direct3D 12 presenter for swap-chain output plus GPU texture readback. Recording and the periodic assistive grab use an asynchronous readback ring that never blocks the render loop; photos and on-demand analysis use the synchronous path.
- CUDA external-memory interop path with GPU color conversion (NV12/YUY2) and rotation, video stabilization, automatic keystone correction for projected screens, black-and-white, zoom, Gaussian blur, temporal smoothing, auto contrast (percentile level stretch), low-vision display color modes with contrast/brightness, focus marker, and spatial sharpening via NVIDIA NIS or AMD FSR 1.0 style kernels.
- GPU video stabilization (projection-profile motion estimation, fully on-device) to counter phone-on-laptop mount vibration.
- CUDA Text Clarity for camera text: background flattening, soft adaptive
  Sauvola thresholding, automatic text polarity, stroke weight, smart
  sharpening, CLAHE, two-color reading, anti-shimmer hysteresis, selective
  text-edge sharpening, glare suppression, and asynchronous focus detection.
- Optional NVIDIA Maxine SuperRes replaces NIS/FSR for zoomed text when the
  separately installed Video Effects runtime is available. It runs on the
  existing CUDA stream, discards 10 warmup samples, and automatically falls
  back if its next 60-frame average exceeds the 24 ms latency target. Advanced
  reports the measured average and offers a compact checkbox override when
  latency is the only failure.
- Two-speed UI: Simple mode gives the full client area to the live view and overlays three auto-fading primary clusters plus contextual screen-correction controls; Advanced keeps the camera visible beside a narrow inspector containing every parameter and pipeline diagnostics. The chosen mode persists.
- Stage-1 quick modes backed by full stage-2 advanced configurations, including promotion of advanced tuning into user-defined quick options.
- Local OCR via Tesseract plus scene explanations through either a signed-in Codex CLI/ChatGPT subscription or an OpenAI-compatible HTTP endpoint. Results stream into a focusable assistive panel and Advanced Assistant, can be spoken aloud, and can be written to `output/notes/`. The non-blocking Setup Assistant can install Tesseract, Codex CLI, and NVIDIA Video Effects without putting those optional runtimes in an OpenZoom release bundle.
- Session persistence in `%APPDATA%\OpenZoom\OpenZoom\settings.json`.
- Synchronized original/processed capture pairs under `output/img/` and
  `output/vid/` next to the executable.
- A branded, high-contrast magnifier icon embedded in the Windows executable
  at multiple display sizes and assigned through Qt for the title bar,
  Alt-Tab, and taskbar.

## Status
CUDA is the processing path; the CPU effects pipeline is deprecated. When the D3D12/CUDA interop surface cannot initialize, the app presents unprocessed passthrough video and shows a persistent "GPU required — processing disabled (showing raw video)" notice instead of silently degrading. The CPU debug composite view remains available as a diagnostic.

## Prerequisites
- Windows 10 or Windows 11.
- Visual Studio 2022 with the Desktop C++ workload and Windows SDK.
- Qt 6.9.3 for `msvc2022_64`, or matching overrides via `QT_PREFIX` / `Qt6_DIR`.
- CMake 3.23 or newer.
- NVIDIA GPU plus CUDA Toolkit 13.x if you want the CUDA path.
- Optional: NVIDIA Video Effects runtime for Maxine SuperRes, Tesseract OCR
  for local text recognition, and Codex CLI for subscription-backed Explain
  and Assistant features. OpenZoom offers verified vendor downloads for all
  three from `Setup & Downloads`.

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

`OPENZOOM_ENABLE_TEXT_SR=ON` compiles the runtime-only NVIDIA Maxine SuperRes
adapter and its controls. CUDA presets and the release-bundle script enable it
by default; the CPU preset disables it. Set the environment override explicitly
to `OFF` only for a bundle that intentionally omits the adapter. OpenZoom links
no Maxine import library and ships no proprietary runtime or model files.

### Release bundle

```bat
scripts\build_release_bundle.bat
```

This produces `dist\OpenZoom\` with `open_zoom.exe`, Qt runtime files,
`LICENSE`, `README.txt`, `THIRD_PARTY_LICENSES.md`, and the bundled Lucide icon
notice. The CUDA runtime is linked statically. NVIDIA Video Effects, Tesseract,
and Codex CLI binaries are never copied into the bundle; users obtain them from
their vendors through the Setup Assistant. The Tesseract installer and
OpenAI's Codex bootstrap script are pinned and SHA-256 verified before
execution. The verified Codex bootstrap then verifies the selected official
release package against OpenAI's checksum manifest.
If Qt's transfer fails, Setup retries with the Windows downloader and then the
vendor's alternate host without weakening verification.

## Runtime Controls
- `Simple` / `Advanced` switches between a full-view overlay UI and a right-side inspector. The live camera remains visible in both states.
- `Viewport framing` under `Advanced > Image > Global device` is global:
  `Fill (crop)` fills the camera area without distorting it, while `Fit (show
  all)` preserves the complete frame with symmetric black bars. Resizing,
  rotating, or dragging the Advanced divider always preserves the camera
  aspect ratio.
- `Viewport motion rate` controls how smoothly pan and animated zoom move over
  the latest processed scene: Auto (up to 120 FPS), 60, 90, 120, or Match
  display. The active monitor clamps unsupported choices and OpenZoom reports
  the effective rate. This does not invent camera frames: a 30 FPS camera is
  still 30 FPS; only navigation over its newest completed frame is refreshed
  more often. Auto reduces to the camera rate while idle.
- In Simple mode, the switch, profile carousel, and Photo/Record/Explain/Read actions occupy three flush view corners. Keystone profiles add a separate Previous/Stop/Next correction strip beside the carousel. Pipeline status stays in Advanced diagnostics so it never covers the camera. The solid, high-contrast chrome fades after about five seconds idle and returns on mouse, keyboard, focus, or application activity. Repeated pointer activity only extends that deadline; it does not relayout the floating controls or reduce viewport motion rate.
- `Ctrl+H` pins the Simple controls on screen or restores automatic hiding; `Esc` closes the quick-mode grid.
- The grid button or current profile opens all quick modes as large tiles. Built-in modes use plain-language labels such as `Read a Page`, `High Contrast`, `Sharpen Text`, `Keep It Steady`, `See in Low Light`, `Projector Screen`, and `Whiteboard`.
- Number keys `1` through `9` apply the first nine quick modes except while an
  editable text field has focus. A mode change shows a large centered
  announcement and notifies screen readers without starting speech.
- `Text Clarity` in Simple mode automatically selects a paper, board, or
  mixed-content treatment. The `Document` quick mode applies the full
  page-reading stack; use the carousel/grid to reach it after the numbered
  first nine modes.
- `NVIDIA Super Resolution` and its strength slider are profile-owned Advanced
  controls under `Advanced > Image > Advanced Tuning > Text clarity`. Enabling
  it restores a useful strength when needed and raises zoom to the 1.33x
  minimum. Maxine runs its supported 4/3x AI pass, with any additional zoom
  applied afterward on the GPU. The result is consumed synchronously on the
  application CUDA stream and is not blended with a separately timed base
  frame; residual zoom maps the live pan focus into the AI crop. This prevents
  moving ghost layers. The status row distinguishes the source crop,
  viewport target, final magnification, and measured inference time.
  Maximum source detail is the default. The optional `Faster 2x mode (narrower
  view)` raises the minimum magnification to 2x; for a 1280x720 viewport this
  changes the visible source crop from 960x540 to 640x360. It is a speed and
  field-of-view tradeoff, not a higher-quality mode. `Ultra quality (full
  frame, up to 1440p)` instead allocates a separate high-resolution scene
  cache, runs SuperRes over the complete processed camera frame, and applies
  viewport zoom/cropping afterward. A 1280x720 camera runs 2x into
  2560x1440; a 1920x1080 camera runs 4/3x into 2560x1440; an already-1440p
  camera remains native. The regular scene texture always follows the actual
  post-rotation camera resolution and is never fixed at 720p. Ultra and Faster
  2x are mutually exclusive profile choices. Unavailable or failed runtimes
  fall back to NIS/FSR automatically and report the reason; a slow run can be
  kept on with the compact performance-limit checkbox until SuperRes is
  turned off.
- The bottom-left quick-mode carousel and its full preset grid remain available
  in Advanced mode, so presets can be changed without returning to Simple.
- Advanced has separate `Image` and `Assistant` tabs with wrapping previous/next navigation arrows. `Image` contains device, per-profile tuning, and pipeline status; `Assistant` contains a persistent camera-aware chat plus an OpenZoom-only history list. Each page places the labeled `AI Settings` pop-out in a full-width row directly below the tab strip. Drag the high-contrast divider at the inspector's left edge to resize it; OpenZoom remembers the width. Long setting rows place their slider on a second line instead of clipping text or making part of the track unreachable.
- `Advanced Tuning` expands the per-profile controls inside the Advanced Image inspector.
- `Save As Quick Option` promotes the current advanced setup into a reusable stage-1 preset.
- `Reset Tuning` restores profile-owned image and assistive controls to their
  defaults after confirmation. It deliberately keeps the selected camera,
  orientation, viewport motion/framing, and Virtual Joystick preference.
- The question-mark button in the Advanced tab header opens a compact guide
  with Controls first and Features second.
- `Camera` selects the active Media Foundation device from Advanced. Camera selection and orientation are global rather than part of a quick profile.
- `Modes` shows the discovered capture modes for the selected camera.
- `Rotation` rotates the pipeline in 90 degree clockwise steps before downstream processing.
- `Black & White` applies thresholded monochrome conversion.
- `Zoom` enables the magnifier and focus-point controls.
- `Gaussian Blur` applies the CUDA blur stage with configurable sigma and supported discrete radii.
- `Temporal Smooth` applies an exponential running average.
- `Stabilize Image` enables GPU video stabilization; the strength slider controls how aggressively the camera path is smoothed.
- `Straighten Screen (Keystone)` automatically detects a projected slide or screen viewed at an angle and warps it fronto-parallel. Its Previous control freezes tracking and restores an earlier accepted correction; Stop/Continue holds or resumes the live detector; Next restores newer history or samples exactly one fresh correction while stopped. Used by the `Projector Screen` and `Whiteboard` quick modes.
- `Auto Contrast` stretches washed-out projector colors using a percentile level analysis; its strength slider blends toward the full stretch.
- `Text clarity` in Advanced exposes profile-owned controls for background
  flattening, adaptive text, edge softness, text polarity, stroke weight,
  smart sharpen, CLAHE, two-color reading, steady text edges, selective
  sharpening, focus warnings, and glare suppression.
- `Display Colors` opens a compact accessible swatch grid. Built-in reading
  pairs and effects share one GPU luma-LUT model, while the Custom editor can
  create and persist 2-8 stop gradients or stepped posterize schemes. Selection
  changes are announced; merely hovering never changes the camera view.
- Mouse-wheel input scrolls the Advanced panel without changing selectors or
  sliders. Click, drag, and keyboard input still edit those controls.
- `OCR Assist`, `Scene Explain`, and `Assistive Overlay` drive asynchronous assistive analysis and on-screen text overlays.
- `Read Text`, identified by a speaker icon, runs local OCR. `Explain` sends one temporary, non-history camera question and changes to `Stop` while Codex is working.
- The assistive result panel updates as an answer streams. It is an owned floating tool window: native window movement keeps dragging responsive over the D3D camera surface, and streamed text does not reset its geometry. Drag its header to move it and an edge or corner to resize it. The first placement clears the top Simple controls; later position and size changes persist relative to the camera view and are restored across restarts. Its text can be focused, selected, and read by a screen reader; the question field remains editable while an answer is streaming, but Ask and Enter submission stay blocked until that answer finishes. Follow-ups enter the shared persistent Assistant conversation with the current view attached. Speech starts only when `Read Aloud` is clicked and omits visible section labels such as `Scene Explain` and `OCR`, while the high-contrast Close control or `Esc` dismisses the panel.
- The Advanced Assistant can attach the current processed view, stream answers, stop a response, and manage persistent OpenZoom conversations with resume, rename, export, and delete actions.
- The Advanced Assistant subscription label reports the percentage left in the current Codex usage window.
- `Connect ChatGPT` uses the Codex app-server browser login flow. Existing Codex CLI sign-in is reused automatically.
- `AI Settings` is vertically scrollable and separates Codex subscription,
  OpenAI-compatible vision server, OCR, Read Aloud, and lecture-note controls.
  It shows OpenZoom's built-in Codex prompt read-only beside editable user
  instructions. The Codex model dropdown and its reasoning dropdown are
  populated from the signed-in app-server's `model/list` response, including
  each model's supported efforts and default; the saved model remains visible
  as unavailable when it is absent from the current catalog. Assistant
  Instructions can set a preferred response language, tone, or level of detail
  for both providers without changing Codex security permissions. OpenZoom
  lists every voice exposed to desktop applications by Windows Runtime speech
  synthesis, across installed languages. Windows 11 Narrator/Magnifier Natural
  voice packages are not currently exposed by that public API and therefore do
  not appear in OpenZoom. Advanced Assistant internet and coding permissions
  are separate opt-ins; coding also requires a workspace folder.
- `Open Notes` opens the current session's accessible HTML lecture notes from `output/notes/`. Notes contain timestamped OCR and scene explanations plus browser-renderable relative links to captured images.
- `Spatial Sharpen` enables the CUDA sharpening/upscaling stage and lets you choose NIS or FSR-style processing when the GPU path is active.
- `Debug View` switches to the CPU composite grid so intermediate stages can be inspected; the nearby pipeline status reports CPU/GPU, fallback, recording, OCR, and VLM state.
- `Show Focus Point` overlays the current zoom center on the presented output.
- `Capture Photo` waits for the next complete camera frame and saves
  `IMG_<timestamp>_original.jpg` plus `IMG_<timestamp>_processed.jpg` to
  `output/img/`.
- `Start Recording` writes synchronized `VID_<timestamp>_original.mp4` and
  `VID_<timestamp>_processed.mp4` files to `output/vid/`. Both are encoded live:
  OpenZoom tries AV1 first and falls back to H.264 when AV1 is unavailable. It
  does not transcode after Stop. Fragmented MP4 keeps each file playable up to
  its last completed fragment, the duration cap is 12 hours, and disk-space
  guards stop both files together. Paired capture approximately doubles output
  storage and adds CPU camera-format conversion while recording.

## Navigation And Interaction
- `Ctrl + mouse wheel`: zoom around the cursor.
- `Mouse wheel`: pan while zoomed.
- `Middle mouse drag`: pan the zoom focus.
- Arrow keys: nudge the zoom focus.
- `1` through `9` in Simple mode: apply the corresponding numbered quick mode.
- `Tab` / `Shift+Tab` in Simple mode: move through the corner controls.
- `Virtual Joystick`, near the top of Advanced Image with the other global
  controls, shows an on-canvas joystick overlay for panning.
- Wheel, arrow-key, joystick, and middle-drag panning keep the active quick mode selected; changing an actual Advanced processing control still creates a custom setup.

## Persistence And Output Paths
- Settings persist to `%APPDATA%\OpenZoom\OpenZoom\settings.json`. Camera and orientation are global; stabilization, colors, contrast, sharpening, zoom, and the other image treatments are stored in the current profile and user-created quick options. UI and assistive/AI configuration also persist. OpenZoom stores only an index of the persistent Codex conversations it created; Codex stores their transcripts. Temporary Simple explanations are ephemeral and do not enter history.
- Snapshot pairs are written under `output/img/` relative to the executable.
- Original/processed recording pairs are written under `output/vid/` relative
  to the executable.
- Lecture notes are written as `NOTES_<timestamp>.html` under `output/notes/`
  relative to the executable. Image references are relative to the notes file,
  so the complete `output/` directory remains portable.
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
Codex app-server is still a local coding-agent process. `Setup & Downloads`
can install or update the official per-user distribution; ChatGPT sign-in is
then completed separately through `Connect ChatGPT` in AI Settings.

Environment variables remain as fallback for any field left empty in the
dialog:
- OCR first uses the configured path, then `OPENZOOM_TESSERACT_PATH`, the
  Setup Assistant-managed `%LOCALAPPDATA%\OpenZoom\tools\tesseract\tesseract.exe`,
  `PATH`, or standard Windows install directories. `Setup & Downloads` installs
  or removes the managed copy after verifying its pinned installer hash.
- Maxine discovery checks `OPENZOOM_MAXINE_PATH`, `NV_VIDEO_EFFECTS_PATH`, the
  standard Program Files location, and the NVIDIA Video Effects uninstall
  registry entry. The bottom of Advanced displays the required
  `SuperRes powered by NVIDIA Maxine™` attribution; this is attribution only
  and does not imply NVIDIA endorsement.
- Codex first uses the configured CLI path, then `OPENZOOM_CODEX_PATH`, `PATH`,
  the official standalone
  `%LOCALAPPDATA%\Programs\OpenAI\Codex\bin\codex.exe`, or
  `%LOCALAPPDATA%\Microsoft\WinGet\Links\codex.exe`. New configurations default
  to `gpt-5.6-tera` with `low` reasoning; explicit saved choices are preserved.
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

- GPL-3.0-only via [`LICENSE`](LICENSE), with an additional GPL §7 permission
  allowing linking with NVIDIA proprietary runtime libraries (CUDA runtime,
  Optical Flow SDK, Video Effects/Maxine SDK, TensorRT) — see the notice at
  the top of [`LICENSE`](LICENSE)
- Commercial licensing, support/SLA contracts, and sponsored development by
  direct arrangement with the project owner — see [`COMMERCIAL.md`](COMMERCIAL.md)

Contributions are accepted under the contributor license agreement in
[`CLA.md`](CLA.md): you keep ownership of your work, it is licensed for both
the GPL and commercial distributions, and it always remains available under
the GPL. Third-party notices are summarized in
[`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md).
