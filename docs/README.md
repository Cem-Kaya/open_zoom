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

The UI now has two layers:
- stage 1: quick modes for common low-vision tasks
- stage 2: advanced tuning for power users and preset creation

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
- `scripts/run_minimal_test.bat`: builds the app without launching it, then runs the DX12/CUDA sandbox harness if available.
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
- The main interaction surface is a preset list; each preset maps to a full advanced configuration.
- Advanced edits update a live config and can be promoted into user-defined quick modes.
- Rotation is applied before the rest of the processing pipeline.
- Settings persist to `%APPDATA%\OpenZoom\OpenZoom\settings.json`.
- Snapshots are saved to `output/img/`.
- Recordings are saved to `output/vid/`.
- The processing status label distinguishes CPU, GPU, fallback, debug-view, and recording states.
- OCR runs through `tesseract.exe` and VLM requests run through a configurable OpenAI-compatible HTTP endpoint, with an in-app assistive overlay rendering the results.

## Documentation Index
- [`README.md`](../README.md): top-level project overview and usage.
- [`docs/code_reference.md`](code_reference.md): authoritative file/class map.
- [`docs/hardcoded_paths.md`](hardcoded_paths.md): machine defaults and magic values.
- [`docs/progress.md`](progress.md): implementation tracker.
- [`docs/ai_upscaling_todo.md`](ai_upscaling_todo.md): future GPU upscaling plan.
- [`docs/THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md): third-party attribution and redistribution notes.

## Current Gaps
- Automated tests are still limited.
- CUDA interop still needs broader hardware validation across more driver/toolkit combinations.
- OCR quality depends on an external Tesseract installation and the quality of the processed frame fed into it.
- VLM mode depends on user-provided endpoint credentials and currently targets OpenAI-compatible `chat/completions` payloads.
