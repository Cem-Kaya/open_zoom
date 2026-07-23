# OpenZoom Project Tracker

## Current Snapshot
Updated after viewport and architecture work on 2026-07-23.

## Completed
- [x] Split the codebase into `app`, `capture`, `common`, `d3d12`, `cuda`, and `ui` modules with mirrored public headers.
- [x] Implement Media Foundation camera enumeration and per-camera mode discovery.
- [x] Build the CPU frame pipeline for conversion, rotation, zoom, blur, temporal smoothing, and debug compositing.
- [x] Bring up the Direct3D 12 presenter, including GPU texture readback.
- [x] Enable the CUDA interop processing path with CPU fallback.
- [x] Add persistent settings storage in `%APPDATA%\OpenZoom\OpenZoom\settings.json`.
- [x] Add synchronized original/processed photo and live AV1/H.264 fragmented
  MP4 recording pairs.
- [x] Add rotation-aware focus controls, joystick navigation, mouse pan, and wheel zoom/pan.
- [x] Add release-bundle scripting and a minimal validation harness entry point.
- [x] Introduce a two-stage quick-mode plus advanced-tuning UI model with promotable custom presets.
- [x] Add OCR/VLM assistive-mode scaffolding to the config model and UI.
- [x] Add a working assistive overlay plus OCR/VLM runtime hooks for live analysis.
- [x] Add the classical CUDA Text Clarity stack, focus-aware OCR gating, Simple
  master toggle, Advanced controls, and the built-in Document quick mode.
- [x] Add runtime-loaded NVIDIA Maxine SuperRes with per-profile strength,
  NIS fallback, CUDA-event latency guard, Setup Assistant, and required
  attribution without redistributing proprietary runtime files.
- [x] Add canonical aspect-safe Fill/Fit viewport geometry, native-client
  D3D12 presentation, a separate high-refresh navigation clock, and cached
  registered SuperRes ROI presentation.
- [x] Add CPU settings-store and viewport-transform regression tests behind
  `OPENZOOM_ENABLE_TESTS`.
- [x] Decompose the application into focused pipeline, recording, settings,
  UI-state, assistive, and interaction responsibilities with standalone
  complex UI widgets.
- [x] Verify the rebuilt CUDA application through a 45-second live-camera
  startup/runtime/normal-shutdown smoke run.

## In Progress
- [ ] Broaden CUDA interop validation across more GPUs, drivers, and toolkit versions.
- [ ] Broaden automated tests to cover capture/reconnect, recording, and
  hardware presentation paths.
- [ ] Tighten documentation and release hygiene as the feature set expands.
- [ ] Improve OCR quality, add ROI selection, and harden VLM request/response handling across more providers.

## Upcoming
- [ ] Add richer overlays for diagnostics, accessibility, and future AI assistance.
- [ ] Expand recording and capture validation with regression fixtures.
- [ ] Benchmark Maxine SuperRes against NIS on real slide footage across
  Turing, Ampere, Ada, and Blackwell GPUs; pursue the vendor-independent tier
  in `improvement_ideas/10-vendor-independent-gpu.md` if visual gains are weak.
