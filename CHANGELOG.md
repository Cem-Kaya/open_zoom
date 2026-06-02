# Changelog

## [Unreleased]
- Fixed several build/runtime hardening issues: CPU-only CMake definitions now
  honor `OPENZOOM_ENABLE_CUDA=OFF`, missing direct includes were added, CUDA-off
  stubs match the app API, Qt teardown destroys widgets before `QApplication`,
  `AssistiveRuntime` now generates Qt moc metadata, BGRA frame wrappers use the
  Qt-supported `QImage::Format_ARGB32`, and the CUDA shared texture now matches
  the BGRA presenter/readback path.
- Added a working assistive-analysis runtime: OCR now shells out to `tesseract.exe`, VLM requests can be sent to an OpenAI-compatible endpoint, and results render in an in-app overlay.
- Added a two-stage UI model with quick modes for everyday use, advanced tuning for power users, and a path to promote advanced setups into reusable quick options.
- Refactored settings persistence around live advanced configs plus user-defined preset libraries instead of a single flat settings blob.
- Added OCR/VLM assistive-mode scaffolding in the UI and persistence layer so future overlays can slot into the preset model cleanly.
- Added a rotation button that cycles the capture orientation in 90° steps and rotates frames at the start of the CPU/CUDA pipeline so downstream filters and focus controls respect the new orientation.
- Expanded Gaussian blur radius options to cover discrete values from 0 through 50, with the slider snapping to those radii while keeping CPU and CUDA pipelines in sync.
- Added a persistent settings store (`settings.json`) so camera selection and tuning controls carry over between sessions, with graceful handling of missing or legacy fields.
- Refreshed the Markdown documentation set to match the current module split, active CUDA fallback behavior, camera mode listing, snapshot/recording outputs, and public API surface.

## [v0.1] - 2025-10-15
- Added `scripts/build_release_bundle.bat` to produce a self-contained `dist/OpenZoom` folder with Qt and CUDA runtime DLLs.
- Enabled optional CUDA builds via `OPENZOOM_ENABLE_CUDA` and introduced new CMake presets for MSVC debug/release and CPU-only configurations.
- Guarded CUDA headers and added temporal smoothing pipeline along with FSR/NIS backend selection.
- Updated documentation: new `docs/hardcoded_paths.md`, refreshed README quick-start, third-party license summary, and dual-license notes.
- Consolidated licensing into a single `LICENSE` file (GPL-3.0 + commercial notice).

> Upload `dist/OpenZoom/OpenZoom.zip` to GitHub Releases when publishing v0.1.
