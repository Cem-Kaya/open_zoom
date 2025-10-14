# OpenZoom

OpenZoom is a Windows-only camera magnification playground that experiments with real-time capture, GPU/CPU image processing, and live presentation. The current milestone runs entirely on the CPU for stability while keeping the CUDA + Direct3D12 pathway wired up for future kernels.

## Highlights
- Media Foundation capture thread that speaks BGRA32, RGB32, NV12, and YUY2 and normalises everything into BGRA for rendering.
- Qt 6 widget shell hosting a Direct3D 12 swap chain, including resize-aware layout and clean shutdown handling.
- Debug-friendly 2×2 compositor that shows raw input, grayscale, zoom, and combined output so filter stages are easy to inspect.
- CUDA external-memory interop path now active with optional spatial sharpening (FSR 1.0-style or NVIDIA NIS) plus the legacy Gaussian blur toggle for clearer zoomed text.
- Batch script that bootstraps a Visual Studio 2022 build, scrubs stale CMake caches, and launches the app when compilation succeeds.

## Project Status
> CPU presentation path is production-ready for debugging. CUDA kernels and zero-copy plumbing are staged but inactive. Next major push is to restore GPU processing inside the Qt/D3D12 architecture.

See `docs/progress.md` for a live task board and `chatgpt_future_readme.txt` for long-term design notes.

## Prerequisites
- Windows 10/11 with a CUDA-capable NVIDIA GPU (SM 7.5, 8.6, or 8.9 tested).
- Visual Studio 2022 with Desktop C++ workload and the Windows 10/11 SDK.
- CUDA Toolkit 13.x (ensure `nvcc` and headers are on PATH/INCLUDE).
- Qt 6.9.3 (MSVC 2022 64-bit build) or adjust `QT_PREFIX`/`Qt6_DIR` to match your install.
- CMake ≥ 3.23; Ninja is optional but recommended for non-MSBuild workflows.

## Quick Start (Visual Studio generator)
1. Open a "x64 Native Tools Command Prompt for VS 2022" or a PowerShell session with VS, Qt, and CUDA on PATH.
2. Run the helper script:
   ```bat
   scripts\build_and_run.bat
   ```
3. If Windows reports missing `Qt6*.dll` files, append the Qt `bin` directory (for example `C:\Qt\6.9.3\msvc2022_64\bin`) to `PATH` or copy those DLLs next to `open_zoom.exe`.

### Alternative: Ninja build
```powershell
cmake -S . -B build-ninja -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:/Qt/6.9.3/msvc2022_64" -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
cmake --build build-ninja
```
Run the produced `open_zoom.exe` from `build-ninja` (ensure Qt DLLs are discoverable).

## Runtime Controls
- Camera picker combo box selects the active Media Foundation device.
- `Black & White` checkbox toggles grayscale processing.
- `Zoom` checkbox enables the magnifier.
- The 2×2 debug view displays: top-left raw feed, top-right grayscale, bottom-left zoom, bottom-right combined result.

## Repository Layout
- `src/` – Qt application, Media Foundation capture, Direct3D12 presentation, and CUDA stubs.
- `include/` – Public headers for the application shell and interop helpers.
- `docs/` – Supplemental design notes and progress tracking.
- `scripts/` – Local tooling (currently `build_and_run.bat`).
- `ref/` – External references and scratch assets (ignored by git).
- `build/` – Default out-of-source build directory (generated).

## Development Notes
- The CPU path keeps frame processing deterministic for debugging; CUDA kernels will re-use the same pipeline once interop validation is complete.
- Stale build trees (e.g., when switching between \wsl$ and local paths) can confuse CMake; the batch script auto-cleans mismatched caches.
- When experimenting with CUDA interop, ensure `CudaInteropSurface::IsValid` gates presentation to avoid presenting invalid resources.

## Roadmap
- Restore CUDA external-memory interop and port temporal filters (averaging, stabilization scaffolding).
- Add still-frame capture and encoding once GPU path returns.
- Improve presentation overlays (FPS, format diagnostics, histogram views).
- Expand automated checks for kernel outputs and add regression tests via `OPENZOOM_ENABLE_TESTS`.

For deeper architectural context and historical notes, start with `docs/README.md`.
