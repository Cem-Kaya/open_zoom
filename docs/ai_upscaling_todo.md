# AI Upscaling Implementation Plan

> Drafted 2025-11-??. Refine as work progresses; keep tasks scoped to GPU execution path first, CPU fallback second.

## Global Prep
- [ ] Audit current CUDA pipeline entry points (`OpenZoomApp::ProcessFrameWithCuda`, `CudaInteropSurface`) to verify hook points for new upscalers.
- [ ] Define a clean pipeline configuration object once DL/TensorRT tiers arrive (current UI uses independent toggles for spatial sharpen and Gaussian blur).
- [ ] Extend settings UI (dropdown) and persistence so the selected mode survives restarts.
- [ ] Introduce a shared staging buffer format (RGBA8 or FP16) to pass between kernels consistently.
- [ ] Decide how to gate experimental backends (warning label, telemetry logging, debug toggles).

## Mode 0 – Gaussian Blur (Baseline)
- [ ] Keep existing separable Gaussian path as-is; ensure it is selectable via the new dropdown.
- [ ] Refactor configuration so blur radius/sigma map cleanly when mode 0 is active and are hidden/disabled otherwise.
- [ ] Write regression test notes: capture before/after frames to validate baseline still works.

## Mode 1 – Spatial Upscaler (NVIDIA NIS default)
- [x] Research minimal licensing/attribution requirements for AMD FSR1 and NVIDIA NIS.
- [x] Implement initial CUDA kernels (Lanczos + RCAS-style sharpen) for FSR-like and NIS-like paths with shared sharpness slider.
- [x] Expose a dedicated “Spatial Sharpen” toggle (FSR/NIS + sharpness) independent of other stages.
- [x] Default to NVIDIA NIS when the spatial upscaler is enabled; leave FSR evaluation as future optional work.
- [ ] Benchmark latency on 720p→1080p and 1080p→1440p paths; target <2 ms per frame.
- [ ] Log full configuration per frame when debug logging enabled (scales, sharpness, adapter).

## Mode 2 – Lightweight DL SR (OpenCV DNN)
- [ ] Package/verify OpenCV DNN with CUDA backend; ensure binaries load in deployed app.
- [ ] Integrate FSRCNN, ESPCN, EDSR, and LapSRN models (choose 2–3 for launch) with runtime selection.
- [ ] Manage model caching (load once per session, reuse across frames); add progress logs.
- [ ] Ensure asynchronous CUDA stream interop between Media Foundation upload and DNN inference.
- [ ] Provide fallback to CPU inference if CUDA backend unavailable (with warning banner).
- [ ] Implement quality/perf toggles (e.g., half/quarter tiling, precision control).

## Mode 3 – Real-ESRGAN (TensorRT/Torch-TensorRT)
- [ ] Export Real-ESRGAN weights to ONNX; verify compatibility with TensorRT 10.x.
- [ ] Build TensorRT engine(s) ahead of time or on first launch; cache serialized plans per GPU model.
- [ ] Integrate slight deblocking pre-pass (e.g., Bilateral) and subtle unsharp mask post-pass.
- [ ] Tune execution to stay under ~8 ms at 1080p input on RTX 4060/4090.
- [ ] Add watchdog for engine build failures; surface actionable errors with hint to fall back.
- [ ] Extend logging UI with per-frame timing (capture, inference, post-process).

## Mode 4 – Text-Aware SR (TensorRT)
- [ ] Evaluate STT/TBSRN/TextSR/TADiSR models; select 1–2 with manageable latency.
- [ ] Convert chosen model(s) to TensorRT (FP16) with dynamic shape support.
- [ ] Add optional text-region detection heuristic to prioritize high-value areas (optional first-iteration skip).
- [ ] Expose "Experimental" badge in UI when mode 4 selected; warn about GPU load.
- [ ] Gather subjective readability metrics (test on UI/IDE screenshots) to confirm benefit.
- [ ] Instrument fallback path to auto-drop to mode 1 if inference exceeds pre-set latency budget.

## Cross-Cutting Concerns
- [ ] Update telemetry/logging to record chosen backend, average processing time, and failure counts.
- [ ] Ensure fence synchronization and shared texture ownership remain correct across new kernels.
- [ ] Document build/runtime dependencies (OpenCV, TensorRT, model downloads) in `docs/README`.
- [ ] Prepare automated smoke-test script: run minimal harness per backend to validate startup.
- [ ] Coordinate with deployment pipeline to bundle necessary DLLs/engine files.
- [ ] Migrate debug 2×2 view to GPU textures (optional copies per stage) so we can inspect CUDA path without CPU fallbacks.

## Stretch Goals
- [ ] Add ImGui debug overlay showing per-backend perf stats and quality toggles.
- [ ] Implement dynamic quality scaling (auto swap to faster backend when FPS drops).
- [ ] Investigate multi-frame SR (temporal) once single-frame path stable.

## Repo Hygiene / Architecture (feedback backlog)
- [ ] Add top-level LICENSE, description, topics, and releases metadata.
- [ ] Introduce CI (GitHub Actions Windows build with CUDA/Qt cache) and package artifacts.
- [ ] Wire `OPENZOOM_ENABLE_TESTS`, add CMake options/presets, and provide infrastructure for CPU golden-image tests.
- [ ] Refactor source tree into feature modules (`src/app`, `src/capture`, `src/d3d12`, `src/cuda`, `src/ui`, `src/common`) with mirrored headers under `include/openzoom/`.
- [ ] Replace ad-hoc logging with structured logger (spdlog or custom) capturing negotiated formats and GPU path health.
- [ ] Provide tooling (editorconfig/clang-format, pre-commit hook, vcpkg/CPM story) for consistent DevX.
- [ ] Author docs/architecture.md, CONTRIBUTING.md, issue/PR templates; add testing strategy (golden images, synthetic MF source, interop guards).
- [ ] Address Qt moc hygiene (no `app.moc` in non-Q_OBJECT files; break out QObject subclasses per widget).
- [ ] Encapsulate Media Foundation `IMF*` pointers behind PImpl and normalize BGRA at boundary with explicit format enum.
- [ ] Build FeatureFlags struct to gate CUDA path, ensure device-loss handling, validate CUDA external memory size/pitch.
- [ ] Explore optional OpenCL backend to mirror CUDA pipeline for broader GPU support (future stretch goal; evaluate feasibility).
