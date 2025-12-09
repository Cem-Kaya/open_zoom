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

## External Dependencies
- **Visual Studio 2022** with Desktop C++ workload and Windows 10/11 SDK (minimum 10.0.19041).
- **Qt 6.9.3 MSVC 64-bit** binaries with matching debug/release builds; ensure `Qt6_DIR` resolves to the `lib/cmake/Qt6` folder.
- **CUDA Toolkit 13.x** (tested with 13.4); add `bin`, `lib/x64`, and `include` to PATH/LIB/INCLUDE or configure `CUDA_PATH`.
- **NVIDIA drivers** supporting CUDA external-memory interop (R555+ recommended).
- **CMake ≥ 3.23** and **Ninja** (optional but faster for incremental builds).
- (Planned) **OpenCV DNN + CUDA** and **TensorRT 10.x** for advanced upscalers—documented install scripts will land alongside their integration.

## Runtime Controls
- Toggle cameras via the combo box.
- Enable grayscale via **Black & White**.
- Enable zoom via **Zoom**.
- Enable **Temporal Smooth** to run an exponential moving average across frames (slider controls how much of the new frame is blended in).
- Use **Rotation** to rotate the live feed clockwise in 90° increments; the frame is rotated at the start of the pipeline so CUDA, zooming, and focus aids all track the new orientation.
- OpenZoom remembers the last-used camera, effect toggles, and tuning values in `settings.json` under `%APPDATA%\OpenZoom\OpenZoom`; remove the file if you need to return to factory defaults.
- Enable **Spatial Sharpen** to choose between AMD FSR 1.0 or NVIDIA NIS (NIS is selected by default).
- Both effects can run simultaneously; the bottom-right quadrant shows the combined output.

## Runtime Flags
- `--cuda-buffer-format=<rgba8|fp16>` or `OPENZOOM_CUDA_BUFFER_FORMAT` (defaults to `rgba8`; `fp16` is reserved for upcoming deep-learning kernels).

See `docs/hardcoded_paths.md` for a list of hard-coded toolchain paths and
defaults that may need adjustment on new machines. License attributions live
in `docs/THIRD_PARTY_LICENSES.md`.

## Next Steps
- Validate CUDA interop on systems with the full external-memory headers.
- Re-enable the CUDA processing path in the presentation loop once a GPU device is available.
- Port temporal filters and stabilization kernels from the original design.
