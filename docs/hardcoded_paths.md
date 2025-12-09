# Hard-Coded Paths and Defaults

This file lists the current hard-coded paths, generator defaults, and similar magic values so they can be reviewed and updated when the toolchain changes.

## Qt installation
- **`C:\Qt\6.9.3\msvc2022_64`** – default Qt prefix used in:
  - `scripts/build_and_run.bat`
  - `scripts/build_release_bundle.bat`
  - `README.md` quick-start examples
- These scripts fall back to the default if `QT_PREFIX`/`Qt6_DIR` is not set. Update all references if you move to another Qt version or install location.

## CUDA
- **Default installation guess**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` via `CUDA_PREFIX_DEFAULT`.
- **`%CUDA_PATH%`** is probed first. If not set, the bundler script scans each `CUDA_PATH_V*` environment variable and finally falls back to the default path.
- **DLL search order** in `scripts/build_release_bundle.bat`:
  1. `%CUDA_PATH%\redistributable_bin`
  2. `%CUDA_PATH%\bin\x64`
  3. `%CUDA_PATH%\bin`
- **Bundled DLLs (hard-coded for CUDA 13.x)**: `cudart64_13.dll`, `nvrtc64_130_0.dll`, `nvrtc64_130_0.alt.dll`, `nvrtc-builtins64_130.dll`, `nvJitLink_130_0.dll`, `nvfatbin_130_0.dll`. Wildcard copies (`cudart64*.dll`, `nvrtc64*.dll`) run afterwards for compatibility with future point releases.
- **CUDA architectures** hard-coded in `CMakeLists.txt`: `75;86;89`. Adjust when adding support for newer GPUs.

## CMake / Generators
- **Default generator** for the bundle script: `Visual Studio 17 2022`. Override via `CMAKE_GENERATOR` environment variable if you prefer Ninja.
- **Build directories**:
  - Regular builds default to `build/` or `build/msvc-*` (from `CMakePresets.json`).
  - The release bundle uses `build/release-bundle` and outputs to `dist/OpenZoom`.

## Runtime bundling output
- **Distribution folder**: `dist\OpenZoom\open_zoom.exe` plus copied DLLs. Delete this folder before packaging a new release to avoid stale assets.

Keep this document up to date whenever paths, versions, or magic defaults change.
