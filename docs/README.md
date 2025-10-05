# OpenZoom

> Notes for future assistant: keep this file updated whenever the architecture evolves.

OpenZoom is a Windows-only experimental camera magnification playground combining:
- **Qt 6** for the UI shell and event loop.
- **Direct3D 12** for presentation and swap-chain management.
- **Media Foundation** for camera capture.
- **CUDA** (optional) for GPU-based image processing via the external-memory interop path.

## Building
1. Install CUDA Toolkit 13.x, Qt 6.9.3 (MSVC 2019/2022 64-bit), and MSVC 2022.
2. From a VS Developer Command Prompt or PowerShell with the toolchain on PATH:
   ```
   scripts\build_and_run.bat
   ```
3. If you see "Qt6*.dll was not found", add `C:\Qt\6.9.3\msvc2022_64\bin` to `PATH` or copy the DLLs next to the executable.

## Runtime Controls
- Toggle cameras via the combo box.
- Enable grayscale via **Black & White**.
- Enable zoom via **Zoom**.
- Both effects can run simultaneously; the bottom-right quadrant shows the combined output.

## Next Steps
- Validate CUDA interop on systems with the full external-memory headers.
- Re-enable the CUDA processing path in the presentation loop once a GPU device is available.
- Port temporal filters and stabilization kernels from the original design.
