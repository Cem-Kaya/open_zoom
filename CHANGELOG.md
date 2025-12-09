# Changelog

## [Unreleased]
- Added a rotation button that cycles the capture orientation in 90° steps and rotates frames at the start of the CPU/CUDA pipeline so downstream filters and focus controls respect the new orientation.
- Expanded Gaussian blur radius options to cover discrete values from 0 through 50, with the slider snapping to those radii while keeping CPU and CUDA pipelines in sync.
- Added a persistent settings store (`settings.json`) so camera selection and tuning controls carry over between sessions, with graceful handling of missing or legacy fields.

## [v0.1] - 2025-10-15
- Added `scripts/build_release_bundle.bat` to produce a self-contained `dist/OpenZoom` folder with Qt and CUDA runtime DLLs.
- Enabled optional CUDA builds via `OPENZOOM_ENABLE_CUDA` and introduced new CMake presets for MSVC debug/release and CPU-only configurations.
- Guarded CUDA headers and added temporal smoothing pipeline along with FSR/NIS backend selection.
- Updated documentation: new `docs/hardcoded_paths.md`, refreshed README quick-start, third-party license summary, and dual-license notes.
- Consolidated licensing into a single `LICENSE` file (GPL-3.0 + commercial notice).

> Upload `dist/OpenZoom/OpenZoom.zip` to GitHub Releases when publishing v0.1.
