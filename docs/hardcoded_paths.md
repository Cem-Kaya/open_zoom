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
- Text-SR adapter default: `OPENZOOM_ENABLE_TEXT_SR=ON` for CUDA presets and
  release bundles, and `OFF` for the CPU preset. The bundle script passes the
  value explicitly so a stale CMake cache cannot silently disable it; set the
  environment variable to override this default. It runtime-loads NVIDIA Video
  Effects and never links or redistributes its proprietary libraries or model
  files.
- Preferred environment variable: `%CUDA_PATH%`
- Secondary probe: `%CUDA_PATH_V*%`
- Hard-coded CUDA architectures in `cmake/CMakeLists.txt`: `75;86;89`
- The Windows target uses the static CUDA runtime, so release bundling does not
  copy `cudart`, `nvrtc`, `nvJitLink`, or other CUDA Toolkit DLLs.
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
- Snapshot output: paired `output\img\IMG_*_original.jpg` and
  `IMG_*_processed.jpg` files relative to the executable
- Recording output: paired `output\vid\VID_*_original.mp4` and
  `VID_*_processed.mp4` files relative to the executable
- Lecture notes output: `output\notes\NOTES_*.html` relative to the executable;
  captured-image URLs are stored relative to each notes file
- Assistant export default: `output\assistant\OpenZoom_Assistant_*.txt` relative to the executable

## Runtime Defaults
- Settings path: `%APPDATA%\OpenZoom\OpenZoom\settings.json`
- Settings schema version: `7` (field-tolerant additions include the Setup
  Assistant decline preference and structured `colorScheme`; older Text SR
  names migrate to Maxine names, and legacy `displayColorMode` integers derive
  an equivalent built-in LUT scheme without requiring a version bump)
- Recording frame rate target: `30 FPS`
- Recording codec order: live AV1 first, then live H.264 fallback; the same
  codec must initialize for both original and processed files
- Recording duration cap per file: `12 hours`
- Rotation options: `0`, `90`, `180`, `270` degrees clockwise
- Viewport framing default: `Fill (crop)`; the global alternative is
  `Fit (show all)` with symmetric black bars.
- Viewport motion default: `Auto (up to 120 FPS)`. Explicit choices are
  `60`, `90`, `120`, and `Match display`; all are clamped to the active
  monitor's current refresh rate without changing camera FPS.
- Viewport motion remains in the high-rate state for `150 ms` after the last
  input update, then idles at the negotiated camera rate. Native-window resize
  requests are coalesced for `16 ms` before resizing the swap chain.
- Assistive analysis cadence: roughly `1600 ms` between OCR/VLM submissions
- Codex scene explanations are on-demand only; periodic assistive polling does not spend Codex subscription usage.
- Default Codex model: `gpt-5.6-tera`; the client uses the current image-capable
  app-server default when that id is unavailable.
- Default Codex reasoning effort: `low`.
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
- Text local-statistics radius: `clamp(frame width / 32, 8, 128)`, producing a
  local window approximately 1/16 of the frame width.
- Text scene classes: paper at mean luma above `0.62`, board below `0.34`, and
  mixed content otherwise. Automatic polarity follows luma-histogram skew,
  with mean luma below `0.46` used only for near-symmetric histograms.
- Focus statistics are sampled every fourth pixel and copied asynchronously
  every 15 frames; default variance threshold is `0.012`.
- CLAHE uses an `8 x 8` tile grid and 256-bin clipped histograms.
- Display Colors contains `17` stable persisted indices (`0..16`).
- Keystone correction history retains at most `32` accepted corner sets.
- Viewport-target Maxine SuperRes activates at zoom `>= 1.33x`; its smallest
  vendor-supported exact ratio is `4/3x`. The mutually exclusive, profile-owned
  faster mode raises minimum zoom to `2.0x`, changing a `1280x720` target's
  crop from `960x540` to `640x360`. Ultra full-frame mode does not require
  viewport zoom and caps its separate cache at landscape `2560x1440` (or
  portrait `1440x2560`): `1280x720` uses `2x`, `1920x1080` uses `4/3x`, and
  an input already at the cap remains native. All modes read models from the
  runtime's `models` subdirectory. After 60 timed runs SuperRes disables
  itself for the current CUDA surface when average runtime exceeds `24 ms`,
  then uses NIS/FSR. Ten warmup runs are excluded.
- NVIDIA architecture mapping for Setup Assistant downloads: compute 7.5 =
  Turing, 8.9 = Ada, other 8.x = Ampere, and 10.x or newer = Blackwell.

## Assistive Runtime Environment
- Optional OCR executable override: `OPENZOOM_TESSERACT_PATH`
- Setup Assistant-managed OCR root:
  `%LOCALAPPDATA%\OpenZoom\tools\tesseract\`; removal is restricted to this
  OpenZoom-owned directory.
- Tesseract discovery also checks `PATH` and `C:\Program Files\Tesseract-OCR`.
- Optional Maxine runtime directory override: `OPENZOOM_MAXINE_PATH` (with
  `NV_VIDEO_EFFECTS_PATH` supported for NVIDIA SDK compatibility).
- Default Maxine runtime root:
  `%ProgramFiles%\NVIDIA Corporation\NVIDIA Video Effects\`, followed by the
  matching 32-bit/64-bit uninstall registry entry.
- Setup Assistant URLs and SHA-256 values are pinned in one table in
  `src/app/setup_assistant.cpp`: Tesseract 5.4.0.20240606, the official Codex
  Windows bootstrap script, and NVIDIA Video Effects 0.7.6
  Turing/Ampere/Ada/Blackwell installers. Tesseract's primary
  source is the UB Mannheim GitHub release asset; the Mannheim download host is
  retained as an alternate. Qt downloads time out after 60 seconds without
  activity, then automatically retry through Windows
  `%SystemRoot%\System32\curl.exe`; all successful transfers use the same pinned
  SHA-256 verification before execution. Final failures offer the GitHub
  release, OpenAI Codex setup guide, or NVIDIA vendor page. The verified Codex
  bootstrap runs with `CODEX_NON_INTERACTIVE=1`; it resolves the latest
  official release and verifies that package against OpenAI's checksum
  manifest before installing it.
- Optional Codex executable override: `OPENZOOM_CODEX_PATH`
- Codex discovery fallbacks:
  `%LOCALAPPDATA%\Programs\OpenAI\Codex\bin\codex.exe` and
  `%LOCALAPPDATA%\Microsoft\WinGet\Links\codex.exe`
- Codex app-server working directory: `%TEMP%\OpenZoom\assistant\`
- Optional Advanced Assistant coding workspace: user-selected in AI Settings; no default folder is granted.
- Attached Codex frames use temporary `openzoom_codex_*.jpg` files and are removed when the turn completes.
- Optional VLM endpoint configuration:
  - `OPENZOOM_VLM_API_URL`
  - `OPENZOOM_VLM_API_KEY`
  - `OPENZOOM_VLM_MODEL`
  - `OPENZOOM_VLM_PROMPT`

## Redistributed Runtime Files
- The bundle script copies the executable, Qt deployment output, `LICENSE`,
  `README.md` as `README.txt`, `docs/THIRD_PARTY_LICENSES.md`, and the Lucide
  notice.
- It explicitly removes Qt's optional `opengl32sw.dll` and copies no NVIDIA,
  CUDA Toolkit, Maxine, Tesseract, model, or language-data binaries.

Update this file whenever these defaults move or new fixed paths are introduced.
