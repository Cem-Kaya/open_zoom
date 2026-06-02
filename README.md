# OpenZoom

OpenZoom is a Windows-only camera magnifier built around Qt 6, Media Foundation, Direct3D 12, and an optional CUDA processing path. The current codebase already supports live camera capture, CPU and GPU frame processing, a two-stage preset/advanced UI, rotation-aware presentation, persistent settings, photo snapshots, and H.264 MP4 recording.

## Current Capabilities
- Media Foundation camera enumeration with per-device mode listing (`width x height @ fps`).
- CPU frame pipeline for format conversion, rotation, black-and-white thresholding, zoom, Gaussian blur, temporal smoothing, and debug compositing.
- Direct3D 12 presenter for swap-chain output plus GPU texture readback for recording and snapshots.
- CUDA external-memory interop path with black-and-white, zoom, Gaussian blur, temporal smoothing, focus marker, and spatial sharpening via NVIDIA NIS or AMD FSR 1.0 style kernels.
- Stage-1 quick modes backed by full stage-2 advanced configurations, including promotion of advanced tuning into user-defined quick options.
- CPU fallback when CUDA interop is unavailable or debug view is enabled.
- OCR via `tesseract.exe`, VLM via an OpenAI-compatible HTTP endpoint, and an in-app assistive overlay.
- Session persistence in `%APPDATA%\OpenZoom\OpenZoom\settings.json`.
- Processed output capture to `output/img/IMG_*.jpg` and `output/vid/VID_*.mp4` next to the executable.

## Status
OpenZoom is no longer just a CPU-only shell. The CPU path is the most deterministic debugging path, while the CUDA path is active when the D3D12/CUDA interop surface initializes successfully. When GPU processing cannot be used, the app falls back to the CPU pipeline automatically and exposes that state in the UI.

## Prerequisites
- Windows 10 or Windows 11.
- Visual Studio 2022 with the Desktop C++ workload and Windows SDK.
- Qt 6.9.3 for `msvc2022_64`, or matching overrides via `QT_PREFIX` / `Qt6_DIR`.
- CMake 3.23 or newer.
- NVIDIA GPU plus CUDA Toolkit 13.x if you want the CUDA path.

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

This produces `dist\OpenZoom\` with `open_zoom.exe`, Qt runtime files, `LICENSE`, `README.txt`, and `THIRD_PARTY_LICENSES.md`. CUDA redistributables are copied when they are available on the machine.

## Runtime Controls
- `Quick Modes` is the primary stage-1 UI: choose task-oriented presets like reading, high contrast, OCR assist, or scene explain.
- `Advanced Tuning` opens the stage-2 panel with the full set of low-level controls.
- `Save As Quick Option` promotes the current advanced setup into a reusable stage-1 preset.
- `Camera` selects the active Media Foundation device.
- `Modes` shows the discovered capture modes for the selected camera.
- `Rotation` rotates the pipeline in 90 degree clockwise steps before downstream processing.
- `Black & White` applies thresholded monochrome conversion.
- `Zoom` enables the magnifier and focus-point controls.
- `Gaussian Blur` applies the CPU or CUDA blur stage with configurable sigma and supported discrete radii.
- `Temporal Smooth` applies an exponential running average.
- `OCR Assist`, `Scene Explain`, and `Assistive Overlay` drive asynchronous assistive analysis and on-screen text overlays.
- `Spatial Sharpen` enables the CUDA sharpening/upscaling stage and lets you choose NIS or FSR-style processing when the GPU path is active.
- `Debug View` switches to the CPU composite grid so intermediate stages can be inspected.
- `Show Focus Point` overlays the current zoom center on the presented output.
- `Capture Photo` saves the current processed frame to `output/img/`.
- `Start Recording` writes processed video to `output/vid/` as H.264 MP4, with a 12 hour cap per file.

## Navigation And Interaction
- `Ctrl + mouse wheel`: zoom around the cursor.
- `Mouse wheel`: pan while zoomed.
- `Middle mouse drag`: pan the zoom focus.
- Arrow keys: nudge the zoom focus.
- `Virtual Joystick`: shows an on-canvas joystick overlay for panning.

## Persistence And Output Paths
- Settings persist to `%APPDATA%\OpenZoom\OpenZoom\settings.json`, including the selected quick mode, current advanced configuration, and user-created quick options.
- Snapshots are written under `output/img/` relative to the executable.
- Recordings are written under `output/vid/` relative to the executable.

## Assistive Runtime Configuration
- OCR uses `tesseract.exe`. If it is not on `PATH`, set `OPENZOOM_TESSERACT_PATH`.
- VLM mode uses an OpenAI-compatible `chat/completions` endpoint.
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
