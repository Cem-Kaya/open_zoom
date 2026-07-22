# Hard-Coded Paths and Defaults

This document tracks machine-specific defaults, generated output locations, and other fixed values that should be reviewed when the toolchain or deployment story changes.

## Qt
- Default Qt prefix: `C:\Qt\6.9.3\msvc2022_64`
- Referenced by:
  - `scripts/build_and_run.bat`
  - `scripts/build_release_bundle.bat`
  - `cmake/CMakePresets.json`
  - `README.md` examples
- Override with `QT_PREFIX` or `Qt6_DIR`.

## CUDA
- CMake default: `OPENZOOM_ENABLE_CUDA=ON`
- Default fallback install root in the release bundle script: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- Preferred environment variable: `%CUDA_PATH%`
- Secondary probe: `%CUDA_PATH_V*%`
- Hard-coded CUDA architectures in `cmake/CMakeLists.txt`: `75;86;89`
- Supported CUDA staging buffer flag values:
  - CLI: `--cuda-buffer-format=rgba8|fp16`
  - Environment: `OPENZOOM_CUDA_BUFFER_FORMAT=rgba8|fp16`
  - Current implementation still resolves `fp16` back to `rgba8`

## Build Directories
- `scripts/build_and_run.bat` configures `build\`
- The helper prefers launching:
  1. `build\cmake\Release\open_zoom.exe`
  2. `build\Release\open_zoom.exe`
  3. `build\open_zoom.exe`
- CMake presets build into:
  - `build/msvc-debug`
  - `build/msvc-release`
  - `build/msvc-cpu`
- Release bundle build directory: `build/release-bundle`

## Output Directories
- Release bundle output: `dist\OpenZoom\`
- Snapshot output: `output\img\IMG_*.jpg` relative to the executable
- Recording output: `output\vid\VID_*.mp4` relative to the executable
- Assistant export default: `output\assistant\OpenZoom_Assistant_*.txt` relative to the executable

## Runtime Defaults
- Settings path: `%APPDATA%\OpenZoom\OpenZoom\settings.json`
- Recording frame rate target: `30 FPS`
- Recording duration cap per file: `12 hours`
- Rotation options: `0`, `90`, `180`, `270` degrees clockwise
- Assistive analysis cadence: roughly `1600 ms` between OCR/VLM submissions
- Codex scene explanations are on-demand only; periodic assistive polling does not spend Codex subscription usage.
- Default Codex model: `gpt-5.5`; the client uses the current image-capable
  app-server default when that id is unavailable.
- Default Codex reasoning effort: `xhigh` (shown as Extra high in AI Settings).
- Default Assistant Instructions: reply in the request's language unless asked
  otherwise, using concise and easy-to-understand answers. The field is editable
  and shared by Codex and the OpenAI-compatible provider.
- Read Aloud engine preference: Qt `winrt`, with `sapi` used only after a WinRT
  engine error; default speech rate is `0.0` (normal) on Qt's `-1.0..1.0` scale.
- The voice picker enumerates every locale exposed by the selected public
  Windows speech engine. Narrator/Magnifier Natural voice AppX packages are not
  exposed to desktop apps through `Windows.Media.SpeechSynthesis`.
- Speech never starts automatically; only the result panel's Read Aloud button
  or the AI Settings Preview button invokes text-to-speech.

## Assistive Runtime Environment
- Optional OCR executable override: `OPENZOOM_TESSERACT_PATH`
- Release-bundle Tesseract source overrides: `OPENZOOM_TESSERACT_DIR` or `TESSERACT_PREFIX`
- Release-bundle fallback Tesseract root: `C:\Program Files\Tesseract-OCR`
- Runtime bundled OCR location: `tools\tesseract\tesseract.exe` relative to `open_zoom.exe`
- Optional Codex executable override: `OPENZOOM_CODEX_PATH`
- Codex discovery fallback: `%LOCALAPPDATA%\Microsoft\WinGet\Links\codex.exe`
- Codex app-server working directory: `%TEMP%\OpenZoom\assistant\`
- Optional Advanced Assistant coding workspace: user-selected in AI Settings; no default folder is granted.
- Attached Codex frames use temporary `openzoom_codex_*.jpg` files and are removed when the turn completes.
- Optional VLM endpoint configuration:
  - `OPENZOOM_VLM_API_URL`
  - `OPENZOOM_VLM_API_KEY`
  - `OPENZOOM_VLM_MODEL`
  - `OPENZOOM_VLM_PROMPT`

## Redistributed Runtime Files
- The bundle script copies `LICENSE`, `README.md` as `README.txt`, `docs/THIRD_PARTY_LICENSES.md`, and an available Tesseract runtime under `tools\tesseract\`.
- CUDA redistributables are copied from:
  1. `%CUDA_PATH%\redistributable_bin`
  2. `%CUDA_PATH%\bin\x64`
  3. `%CUDA_PATH%\bin`
- Explicit DLL names still target CUDA 13.x:
  - `cudart64_13.dll`
  - `nvrtc64_130_0.dll`
  - `nvrtc64_130_0.alt.dll`
  - `nvrtc-builtins64_130.dll`
  - `nvJitLink_130_0.dll`
  - `nvfatbin_130_0.dll`

Update this file whenever these defaults move or new fixed paths are introduced.
