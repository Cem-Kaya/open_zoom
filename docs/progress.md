# OpenZoom Project Tracker

## Completed
- [x] Establish initial repository structure
- [x] Draft high-level build configuration strategy
- [x] Implement CUDA/D3D11 interop demo pipeline skeleton (gradient fill)
- [x] Refactor UI shell to Qt Widgets with Direct3D12 CPU presentation path
- [x] Add DX12-CUDA minimal sandbox verifying external memory import flags
- [x] Integrate multi-frame history manager for temporal kernels
- [x] Implement example GPU kernels (temporal average, stabilization stub)

## In Progress
- [ ] Adapt CUDA interop path to D3D12 resources and restore GPU processing

## Upcoming
- [ ] Build real-time presentation layer enhancements (UI overlays, controls)
- [ ] Add still image capture path and encoding
- [ ] Set up automated validation/tests for kernel outputs

## Today's Focus (2025-10-14)
- [x] Document external dependency setup in `docs/README.md`.
- [x] Lock spatial upscaling to NVIDIA NIS and remove dormant FSR heuristics.
- [x] Introduce configurable staging buffer format flag (default RGBA8, optional FP16).
- [x] Harden camera switching by catching CUDA surface reinitialization failures.
- [x] Implement temporal history manager with exponential smoothing kernel.
- [x] Add sample stabilization/temporal CUDA kernels wired into the pipeline.
