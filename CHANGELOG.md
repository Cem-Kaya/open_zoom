# Changelog

## [Unreleased]
- Added Codex CLI to first-run and Advanced `Setup & Downloads`. OpenZoom
  detects official standalone, PATH, and WinGet installs; can install or update
  the per-user CLI from a pinned SHA-256-verified OpenAI bootstrap; persists
  the resolved executable immediately; and keeps ChatGPT sign-in explicit.
  The upstream bootstrap also verifies the official release package checksum.
- Changed new/empty Codex settings to `gpt-5.6-tera` with `low` reasoning while
  preserving explicit saved choices.
- Prevented Simple `1`-`9` preset shortcuts from firing while an editable
  Assistant or other text field has focus.
- Fixed the Virtual Joystick being covered by the Simple Photo, Record,
  Explain, and Read cluster. It now follows that cluster's real window
  geometry and stays one margin above it across mode, resize, and DPI changes.
- Added profile-owned `Ultra quality (full frame, up to 1440p)` for NVIDIA
  SuperRes. It uses a separate arbitrary-size D3D12/CUDA cache instead of
  downsampling into the camera-sized cache, then applies viewport pan, zoom,
  and crop afterward. 720p input runs full-frame 2x to 1440p, 1080p runs
  4/3x to 1440p, and native 1440p remains native. The normal scene texture
  continues to follow the negotiated post-rotation camera resolution; Ultra
  and Faster 2x are mutually exclusive and persist per profile.
- Formalized the licensing structure: the LICENSE notice now carries a GPL
  section 7 additional permission for linking with NVIDIA proprietary runtime
  libraries (CUDA runtime, Optical Flow SDK, Video Effects/Maxine SDK,
  TensorRT), COMMERCIAL.md describes the paid offerings (support/SLA,
  sponsored development, commercial licensing, prebuilt convenience) with an
  explicit everything-stays-GPL promise, and CLA.md defines the contributor
  agreement (contributors keep ownership, grant dual-licensing rights, work
  remains GPL forever). README license section updated to match.
- Made AI Settings usable at constrained window heights with a vertically
  scrollable content area and fixed OK/Cancel row. Codex subscription,
  OpenAI-compatible VLM, OCR, Read Aloud, and notes now have distinct sections;
  the built-in OpenZoom Codex prompt is visible read-only, while user response
  preferences remain editable. Model and reasoning selectors now update from
  Codex app-server `model/list`, including each model's supported effort set.
- Made the NVIDIA SuperRes latency decision inspectable and overridable. The
  guard excludes 10 warmup frames, reports its measured 60-frame average
  against a 24 ms target, and exposes a compact checkbox only for a
  latency-only fallback. Turning SuperRes off clears the override. Added a
  default-off, profile-owned `Faster 2x mode (narrower view)` option that uses
  a 640x360 visible crop for a 1280x720 target instead of the 960x540 4/3x
  crop, and removed a redundant destination clear before the full-frame output
  transfer.
- Fixed release bundles silently inheriting a stale
  `OPENZOOM_ENABLE_TEXT_SR=OFF` CMake cache value. CUDA bundles now explicitly
  compile the runtime-loaded NVIDIA Super Resolution adapter by default while
  still allowing an environment override, and the control reflects the
  installed Video Effects runtime instead of appearing permanently disabled.
- Made the Display Colors popover use an opaque native backing surface so the
  Advanced inspector and camera can never show through its controls.
- Moved the global Virtual Joystick toggle to the top of Advanced Image, added
  a compact top-bar Help dialog with Controls and Features sections, and added
  Reset Tuning for restoring only profile-owned settings while preserving
  camera, orientation, viewport, and joystick preferences.
- Fixed two D3D12/CUDA synchronization races introduced by the separate
  viewport clock. Every present now adopts the presenter's internal frame-slot
  fence signal into the shared monotonic sequence, and blocking OCR/Assistant
  texture copies queue a GPU wait for the latest CUDA completion before
  reading. Plain pan/zoom can no longer reuse an already-signaled value, and
  on-demand AI input cannot capture a half-written frame.
- Restored high-refresh pan and zoom in Simple mode. Repeated Qt/native mouse
  activity now only extends the chrome idle deadline while controls are
  visible instead of recomputing, moving, and raising every owned tool window
  twice per input sample. Viewport elapsed time now uses nanosecond precision,
  and the no-fence fallback clamps to camera rate because it must drain the GPU
  for each present.
- Fixed the display-color popover dismissing itself when the modal color-stop
  dialog opened (and Escape in that dialog closing the popover underneath it):
  the popup's auto-hide and Escape handling now stand down while one of its
  own modal children is active, so custom-stop editing keeps its context.
- Added an aspect-safe, high-refresh viewport architecture. The camera clock
  publishes one completed CPU/GPU scene generation, while a separate viewport
  clock can re-present it during pan, zoom, and resize at Auto-up-to-120,
  60/90/120, or Match-display rates without advancing temporal effects,
  SuperRes, recording, OCR, or assistive analysis. The D3D12 swap chain now
  stays at the native render-window size and uses one canonical shader/CPU
  transform for Fill/crop or Fit/letterbox, eliminating narrow/wide/rotated
  axis stretching. Diagnostics report camera, viewport, display, scene, and
  missed-present data; unsupported explicit rates are clamped and reported.
- Registered NVIDIA SuperRes output as a cached generation-tagged ROI. The
  viewport uses it only while the current aspect-safe crop is contained in
  that ROI, otherwise immediately showing the identically registered
  conventional scene until a later camera frame refreshes the AI crop. This
  prevents fixed-center ghost layers during high-refresh panning.
- Continued the application/UI disaggregation: `OpenZoomApp` is now a
  composition root with fallible `Initialize()`, focused pipeline, recording,
  settings, UI-state, assistive, and interaction managers, and responsibility
  split across `app_*` translation units. Render, joystick, assistive overlay,
  color picker, and responsive slider-row widgets now have their own source
  and header files.
- Hardened shutdown ordering so Codex process termination and final widget
  signals cannot call back into already-released UI state.
- Modernized the display-color picker: rounded "squircle" swatches with soft
  borders, circular custom color wells, a rounded translucent popover with
  refined section headers and hover states, and a restyled, enlarged color-stop
  dialog (dark theme, bigger controls) for low-vision usability.
- Replaced the legacy 17-row Display Colors combo with an accessible compact
  swatch picker backed by a unified 256-entry luma-LUT color model. Legacy
  modes migrate without visible change, custom 2-8 stop gradient/posterize
  schemes persist, text-clarity compositing uses the active scheme endpoints,
  and LUT uploads occur only when the selected scheme changes.
- Prevented mouse-wheel scrolling from changing settings selectors or sliders.
  Camera, rotation, color/custom controls, image tuning, zoom, and speech rate
  remain editable by click, drag, and keyboard while wheel input continues to
  the surrounding panel or camera navigation.
- Fixed a startup crash in the Display Colors trigger caused by passing a null
  painter to Qt's dropdown-arrow style primitive.
- Completed plan 11 Wave 3 / Batch B performance work. Camera frames now move
  from the capture thread instead of being copied twice, then upload through a
  two-slot page-locked CUDA staging ring for BGRA, NV12, and YUY2; per-slot
  events guard reuse without waiting for the full frame. Gaussian parameter
  changes use stream-ordered symbol copies with the dead size symbol and
  device-wide synchronization removed, and stabilization projections merge
  shared-memory partial sums instead of issuing two global atomics per pixel.
  The benchmark-gated three-box blur was rejected and removed because it was
  about 11.7x slower than the exact radius-25 Gaussian on the reference RTX
  4090 Laptop GPU; P6 remains deferred because cross-surface allocation reuse
  is not a trivial resolution-change fix.
- Added always-on frame timing instrumentation (plan 11 Wave 1 / P8): a
  60-frame rolling wall-clock average of the frame tick plus a CUDA-event
  sample of the GPU kernel chain every 30th frame, both shown in the
  processing status tooltip ("NN ms/frame - CUDA|passthrough - GPU NN ms"),
  with a one-shot warning when the average stays above 40 ms for 3 seconds.
- Encoded the D3D12/CUDA fence contract in a single `FenceSequencer` struct
  (plan 11 Wave 2 / S6b): fence values strictly increase, a failed CUDA frame
  rolls its reserved signal value back so nothing ever waits on a value that
  will not be signaled, and three consecutive ProcessFrame failures trigger a
  full resync (queue drain + fence re-seed) with one status message instead of
  per-frame spam. An S4 audit of capture-thread vs UI-thread state found no
  unguarded shared access (findings recorded in improvement_ideas/01).
- NVIDIA Super Resolution now snaps to the full set of supported scale factors
  (4/3, 1.5, 2, 3, 4), choosing the largest factor at or below the current
  zoom whose source crop maps exactly onto the viewport; the remaining zoom is
  applied by the GPU sampler around the same crop center. Setup failures latch
  per (crop, factor) configuration and retry only when the configuration
  changes or the toggle is re-enabled, and the runtime falls back to the DLL
  directory when no `models` subdirectory exists beside it. The status row
  reports the actually chosen AI factor.
- Fixed NVIDIA Super Resolution showing a stale moving layer over the current
  camera image. The wrapper now validates crop pitches and buffer bounds, uses
  NVIDIA's synchronous execution path, fully transfers the destination, and
  consumes one SuperRes result instead of blending it over a separately timed
  conventional zoom frame. Residual zoom also maps the live pan focus into the
  clamped AI crop rather than assuming the crop center.
- Made the Advanced inspector horizontally resizable with a visible drag
  handle and persistent width. Long Text Clarity/display rows now put sliders
  on a full-width second line when necessary, while SuperRes status text uses
  short wrapping lines and cannot force controls beyond the panel edge.
- Fixed NVIDIA Super Resolution appearing enabled without changing the image.
  The runtime now receives its actual `models` subdirectory and a supported
  exact 4/3x AI stage instead of the invalid 1.25x ratio. Turning it on restores
  nonzero strength, enforces the 1.33x minimum, applies additional zoom after
  the AI pass, latches failures instead of retrying every frame, and shows the
  source crop, viewport target, final magnification, and fallback state.
- Fixed Advanced Image controls spreading vertically when Advanced Tuning is
  collapsed. Remaining controls stay packed at the top, while the bottom-left
  quick-mode carousel and grid remain available for preset changes in Advanced.
- Hardened Windows identity and icon setup with an explicit AppUserModelID and
  native large/small window icons in addition to Qt and executable resources.
- Setup Assistant dependency rows now use large green check/red X indicators.
  System-wide Tesseract installations are labelled as Windows-managed and
  provide an enabled `Open Windows Apps` action instead of an unexplained
  disabled Remove button; OpenZoom-managed copies retain direct removal.
- Fixed Tesseract Setup downloads failing when the Mannheim file host returns
  `Forbidden`. Setup now uses UB Mannheim's GitHub release asset first,
  retries failed Qt transfers through Windows `curl.exe`, can try the Mannheim
  host as an alternate, and requires the same pinned SHA-256 digest on every
  path. The failure action now opens the working GitHub release page.
- Fixed NVIDIA Video Effects installation failing with Windows error 740. The
  verified installer now launches through the native `runas` shell verb,
  displays the UAC consent flow, remains monitored without blocking the UI,
  tells the user to continue in the separate vendor installer window, and
  refreshes dependency status when it exits.
- Added a GPL-clean NVIDIA Maxine SuperRes tier for zoomed text. The adapter
  resolves the separately installed Video Effects 0.7.6 runtime dynamically,
  runs device-only conversion and inference on OpenZoom's existing CUDA
  stream, persists enable/strength per profile, and falls back to NIS/FSR when
  unavailable, failed, or slower than the configured steady-state guard.
- Added a non-blocking, screen-reader-labelled Setup Assistant on first run and
  through Advanced `Setup & Downloads`. It detects supported NVIDIA GPU
  generations, downloads the matching NVIDIA or Tesseract installer from a
  pinned vendor URL with SHA-256 verification and timeout/cancel handling,
  supports removal, and persists `Don't ask again` independently.
- Removed CUDA Toolkit, NVIDIA Video Effects, Tesseract, language-data, and
  software-OpenGL binaries from release bundling. CUDA now uses its static
  runtime and optional dependencies are obtained by the user through Setup.
- Added the mandatory `SuperRes powered by NVIDIA Maxine™` attribution at the
  bottom of Advanced and documented the separate MIT-header and proprietary
  runtime licensing boundaries.
- Assistive View follow-up drafts now remain editable while an answer streams;
  Ask and Enter submission stay disabled until the active response finishes,
  and the draft is preserved for immediate submission afterward.
- Replaced per-session Markdown lecture notes with valid accessible HTML under
  `output/notes/`. Notes remain complete after every entry, escape generated
  text, and use portable relative image links that render directly in browsers.
- Fixed the generic Windows taskbar icon by registering the branded magnifier
  as a Qt resource and assigning it to the application and main window at
  startup. The existing multi-size executable `.ico` remains in place for
  Explorer and shortcuts.
- Photo and video capture now save synchronized `_original` and `_processed`
  pairs under one timestamp. Recording encodes both streams live into
  fragmented MP4, probes AV1 first, and falls back to H.264 when the installed
  Media Foundation path cannot initialize AV1. Async processed readbacks carry
  request ids so each frame is matched to its exact original source frame.
- Assistive View now opens below the top Simple controls on first use and
  persists its camera-relative position and size across restarts. Restored
  geometry is clamped to the current camera area when the window changes.
- Expanded `Display Colors` from five text-only entries to a 17-choice visual
  palette. The closed selector and popup now paint the actual foreground and
  background colors while retaining accessible names and tooltips; every new
  cyan, green, amber, blue, yellow, red, black, and white pairing is backed by
  the CUDA display-grade and two-color text paths.
- Added interactive Projector Screen correction history. Keystone profiles now
  expose Previous, Stop/Continue, and Next controls in Advanced plus a compact
  contextual Simple strip; Previous freezes and restores an accepted warp,
  while Next replays newer history or requests one fresh detection. History is
  bounded to 32 corrections.
- Fixed the Advanced-mode Alt-Tab trap: application reactivation restores and
  raises the persistent Simple/Advanced switch, so focus loss cannot leave the
  user without a route back to the full-view Simple UI. AI Settings also moved
  out of the crowded tab strip into a full-width row below the tabs.
- Added the CUDA Text Clarity pipeline: shared local luma statistics,
  illumination flattening, adaptive soft Sauvola binarization, automatic text
  polarity, stroke thinning/boldening, bilateral halo-clamped smart sharpen,
  8x8 CLAHE, two-color reading, temporal mask hysteresis, stroke-selective
  sharpening, bounded glare suppression, and asynchronous Laplacian-variance
  focus detection that blocks blurry OCR captures. Added a profile-owned
  Advanced control group, a compact Simple `Text Clarity` master toggle, and a
  built-in `Document` quick mode. Mixed-content auto mode preserves natural
  images; paper/board scenes use the reading mask.
- Added automatic keystone correction ("Straighten Screen") and auto contrast
  (percentile level stretch with a strength slider) as CUDA pipeline stages,
  with Advanced "Screen fix" controls, per-preset persistence, and two new
  built-in quick modes: `Projector Screen` (keystone + auto contrast +
  stabilization + gentle temporal smoothing) and `Whiteboard` (strong auto
  contrast + keystone + sharpening + stabilization). Simple-mode number
  shortcuts now cover quick modes `1` through `9`.
- Moved camera color conversion and rotation onto the GPU: capture now prefers
  NV12, then YUY2, before falling back to BGRA formats, so NV12 and YUY2
  frames upload their compact raw planes and are converted/rotated in CUDA,
  removing the last per-pixel CPU work on the processing path. The CPU
  converter remains for the debug view, other camera formats, GPU-unavailable
  passthrough, and as an automatic per-frame fallback.
- Recording and the periodic assistive frame grab now use an asynchronous
  D3D12 readback ring (`RequestReadback` / `TryGetCompletedReadback`), so
  recording no longer stalls the render loop each frame. A shared-fence wait
  keeps CUDA from overwriting the shared texture while a readback copy is
  still in flight. Photo capture and on-demand analysis keep the synchronous
  readback.
- Camera failures now speak plain language: startup errors explain busy,
  missing, or permission-blocked devices in full sentences, and mid-stream
  device loss triggers an automatic reconnect state machine (2s/4s/8s backoff
  for ~30 seconds, driven from the frame tick) that re-finds the same physical
  device by symbolic link. No modal error dialogs appear while reconnecting.
- Recording now writes fragmented MP4, so an interrupted recording stays
  playable up to the last flushed fragment. Recording refuses to start with
  under 500 MB free and finalizes itself cleanly below 200 MB; the disk-full
  stop is surfaced as a non-modal status message ("…recording so far was
  saved") instead of an error dialog.
- Read Aloud now omits the visible `OCR` and `Scene Explain` section headers
  and speaks only the recognized or generated result text.
- Assistive View now uses an owned floating tool window with native move/resize
  handling, and streamed AI text no longer reapplies its geometry on every
  fragment. The Advanced header also labels the pop-out control `AI Settings`.
- Refined the camera-first UI to three primary Simple clusters: processing
  status now lives under Advanced Image diagnostics instead of covering the
  live view. A fourth contextual strip appears only for live keystone history.
  AI Settings sits in a dedicated row below the Advanced tab header, while
  wrapping previous/next section arrows remain in the header.
- Upgraded the assistive result panel into a movable, edge-resizable floating
  Assistant. It keeps streamed, screen-reader-readable output, adds an
  accessible follow-up question field backed by the shared persistent
  Assistant conversation, and uses a large white-on-black Close control.
- Replaced the Simple Photo, Record, Explain, and Read platform icons with
  camera, record-circle, question-message, and speaker icons from Lucide so
  each action has a direct and consistent visual meaning.
- Fixed preset equivalence ignoring rotation: `AreConfigsEquivalent` compares
  `rotationQuarterTurns` again (rotation is per-preset), so preset highlighting
  and dirty-state detection respect rotation differences.
- Added a 60-second timeout to Codex app-server JSON-RPC requests with a
  sweeper timer; pending request handlers now also receive an error (instead
  of being silently dropped) when Codex stops or restarts.
- Simple-mode floating chrome (corner panels, mode grid, toast) now hides when
  OpenZoom loses focus so the always-on-top tool windows no longer float over
  other applications; it reappears on reactivation.
- Closing the mode grid (Esc or toggle) returns keyboard focus to the current
  mode button instead of leaving focus stranded.
- Hardened assistive frame validation against integer overflow on oversized
  camera dimensions, and malformed Codex delta notifications are now logged
  and ignored instead of silently producing empty updates.
- Hardened CPU camera conversion against invalid dimensions, undersized
  strides/buffers, size arithmetic overflow, and odd-width YUY2 output writes.
- Added a shared Assistant Instructions section to AI Settings so response
  language, tone, and detail can be changed independently of the scene prompt.
  Instructions persist and apply to Codex and OpenAI-compatible providers
  without overriding Codex permission limits. Codex now defaults to the
  installed catalog's `gpt-5.6-tera` image model with Low (`low`) reasoning
  and falls back to the app-server default when necessary.
- Changed the Advanced Assistant usage label to show the percentage remaining
  in the current Codex window instead of the percentage already consumed.
- Fixed Codex turn finalization so a completed streamed answer is retained for
  the result overlay, lecture notes, and manual Read Aloud action instead of
  being replaced by a false empty-description error.
- Made speech strictly user-triggered: OCR, Explain, and quick-mode changes no
  longer speak automatically. AI Settings now lists all voices and locales
  exposed to desktop apps by Windows Runtime speech synthesis, saves a voice
  and reading speed, and provides an explicit Preview action. WinRT no longer
  falls back to SAPI merely because its asynchronous initialization has not
  completed; fallback now requires an actual engine error.
- Replaced the passive assistive text overlay with a solid, focusable result
  panel that updates incrementally while answers stream. Result text is now
  keyboard-selectable and exposed to screen readers, with explicit Read Aloud
  and Close controls plus Escape-to-close behavior.
- Added opt-in Advanced Assistant permissions for internet access and coding.
  Coding requires an existing workspace folder and uses Codex workspace-write
  sandboxing limited to that writable root; both permissions remain disabled
  by default and never apply to ephemeral Simple Explain turns.
- Added a native Qt client for `codex app-server` over stdio JSON-RPC. OpenZoom
  can reuse a user's ChatGPT-managed Codex login for image-aware explanations
  without an API key, discovers compatible image models, shows subscription
  usage, streams answers, and supports cancellation.
- Added an Advanced Assistant surface with camera attachment, persistent
  OpenZoom-only conversation history, resume, rename, export, and delete.
  Simple Explain uses ephemeral threads and does not add history.
- Defaulted Codex integration to read-only/no-network vision turns and
  automatic interruption of ungranted command, file-edit, or web-search items;
  MCP, dynamic, and collaboration tools remain blocked in every mode. Requests
  for additional permissions receive an explicit empty grant.
  OpenAI-compatible local/cloud VLM servers remain an optional provider, and
  local servers no longer require a dummy API key.
- Added a custom multi-resolution OpenZoom magnifier icon to the Windows
  executable for Explorer, shortcuts, and taskbar presentation.
- Replaced the Simple-mode Read action's document icon with the native speaker
  icon so its read-aloud behavior is immediately recognizable.
- Added GPU video stabilization (CUDA projection-profile motion estimation with
  strength-controlled path smoothing) as the first pipeline stage, with a
  checkbox and strength slider, per-preset persistence, and state reset on
  camera switch, camera stop, and rotation changes.
- Added low-vision display color modes (Normal / Inverted / White on Black /
  Yellow on Black / Black on Yellow) plus contrast and brightness controls,
  applied on-GPU after temporal smoothing and persisted per preset.
- Reworked the main window into a two-speed Simple/Advanced UI around one
  persistent camera surface. Simple now gives the full client area to the live
  image and overlays four solid, high-contrast corner control clusters. The
  clusters fade after five seconds idle, restore on activity, expose a
  numbered quick-mode grid with `1`-`7` shortcuts, and announce mode changes
  through a large toast and Qt accessibility. Advanced opens a
  narrow, scrollable right-side inspector without hiding the camera.
- Made mouse and focus activity over the native D3D render surface reliably
  restore auto-hidden Simple controls.
- Added `Esc` dismissal for the quick-mode grid and `Ctrl+H` to pin or unpin
  Simple controls, with an assertive accessibility announcement.
- Clarified Simple labels so High Contrast keeps its technical plain-language
  name while Sharp Text is presented as `Sharpen Text`.
- Kept the active quick mode selected while wheel, keyboard, joystick, or
  middle-drag navigation changes zoom focus; true Advanced edits still become
  a custom setup.
- Added standard-install and app-managed Tesseract discovery plus TESSDATA
  setup for local OCR. Tesseract is now installed or removed separately
  through Setup rather than copied into release bundles.
- Fixed the release bundler's executable lookup for Visual Studio builds that
  place `open_zoom.exe` under the `cmake\Release` subdirectory.
- Shortened the visible processing state to fit its corner cluster while
  retaining full GPU/backend and camera-error detail in the status tooltip.
- Fixed quick modes with half-step values being mislabeled as a custom setup
  after application by comparing persisted configs at each UI control's real
  slider precision.
- Separated global device state from profile processing: camera selection and
  orientation persist globally, while stabilization, display colors, contrast,
  sharpening, zoom, and other image treatment remain part of each quick
  profile. Existing profile rotation values migrate to the global setting.
- Added an AI settings dialog (VLM base URL / API key / model / prompt,
  tesseract path and OCR language, TTS, lecture notes) stored in
  `settings.json`; works with OpenAI-compatible local servers (LM Studio,
  Ollama, llama.cpp server) so image-to-text can run fully offline.
  Environment variables remain as fallback.
- Added lecture notes: a per-session HTML file under `output/notes/` that
  collects timestamped OCR text and scene explanations and references captured
  photos, plus an "Open Notes" button.
- Added on-demand analysis: "Read Text" (OCR) and "Explain Now" (VLM) buttons
  analyze the currently displayed frame immediately, independent of the
  periodic assistive loop.
- Added user-triggered text-to-speech of OCR/VLM results via Qt TextToSpeech
  (linked when the Qt module is available).
- Deprecated the CPU effects pipeline: CUDA is the processing path. Without a
  working GPU pipeline the app now presents unprocessed passthrough video with
  a persistent "GPU required" notice instead of running effects on the CPU.
  CPU code remains for camera-format conversion/rotation feeding the GPU and
  for the legacy debug composite view.
- Fixed the camera-switch crash: `CudaInteropSurface` now synchronizes its CUDA
  stream before destroying the surface object, external memory, and device
  buffers, so in-flight kernels can no longer touch freed resources during
  surface reinitialization.
- Fixed a COM reference leak that leaked one `IMFActivate` per camera on every
  device enumeration (redundant `AddRef` on a `ComPtr` assignment).
- Fixed camera startup failing with `MF_E_SHUTDOWN` (`0xC00D3E85`) after the
  mode list probed a device: temporary and live `IMFActivate` sessions now call
  `ShutdownObject()` so restarts create a fresh media source. Camera errors now
  use real hexadecimal HRESULTs with system details.
- Added session-aware reporting for mid-stream camera failures, balanced COM
  initialization on the capture thread, and refreshed frame metadata when a
  camera changes its current media type.
- Fixed `scripts/run_minimal_test.bat` failing after a successful application
  build when the optional `dx12_cuda_minimal` sandbox is not present; the
  missing optional harness is now reported as a clean skip.
- Pipelined the D3D12 presenter: `Present`/`PresentFromTexture` no longer stall
  the CPU until the GPU drains each frame. Per-back-buffer command allocators
  and upload buffers are paced by per-slot fence values; the CUDA path signals
  the shared fence as soon as the copy is queued. The app re-seeds its shared
  fence counter from the presenter to keep fence values monotonic when CPU and
  GPU presentation paths interleave, and drains the queue before releasing the
  shared CUDA texture.
- Added screen-reader metadata (accessible names and descriptions) to all
  interactive controls in the main window.
- Fixed several build/runtime hardening issues: CPU-only CMake definitions now
  honor `OPENZOOM_ENABLE_CUDA=OFF`, missing direct includes were added, CUDA-off
  stubs match the app API, Qt teardown destroys widgets before `QApplication`,
  `AssistiveRuntime` now generates Qt moc metadata, BGRA frame wrappers use the
  Qt-supported `QImage::Format_ARGB32`, and the CUDA shared texture now matches
  the BGRA presenter/readback path.
- Added a working assistive-analysis runtime: OCR now shells out to `tesseract.exe`, VLM requests can be sent to an OpenAI-compatible endpoint, and results render in an in-app overlay.
- Added a two-stage UI model with quick modes for everyday use, advanced tuning for power users, and a path to promote advanced setups into reusable quick options.
- Refactored settings persistence around live advanced configs plus user-defined preset libraries instead of a single flat settings blob.
- Added OCR/VLM assistive-mode scaffolding in the UI and persistence layer so future overlays can slot into the preset model cleanly.
- Added a rotation button that cycles the capture orientation in 90° steps and rotates frames at the start of the CPU/CUDA pipeline so downstream filters and focus controls respect the new orientation.
- Expanded Gaussian blur radius options to cover discrete values from 0 through 50, with the slider snapping to those radii while keeping CPU and CUDA pipelines in sync.
- Added a persistent settings store (`settings.json`) so camera selection and tuning controls carry over between sessions, with graceful handling of missing or legacy fields.
- Refreshed the Markdown documentation set to match the current module split, active CUDA fallback behavior, camera mode listing, snapshot/recording outputs, and public API surface.

## [v0.1] - 2025-10-15
- Added `scripts/build_release_bundle.bat` to produce a self-contained `dist/OpenZoom` folder with Qt and CUDA runtime DLLs.
- Enabled optional CUDA builds via `OPENZOOM_ENABLE_CUDA` and introduced new CMake presets for MSVC debug/release and CPU-only configurations.
- Guarded CUDA headers and added temporal smoothing pipeline along with FSR/NIS backend selection.
- Updated documentation: new `docs/hardcoded_paths.md`, refreshed README quick-start, third-party license summary, and dual-license notes.
- Consolidated licensing into a single `LICENSE` file (GPL-3.0 + commercial notice).

> Upload `dist/OpenZoom/OpenZoom.zip` to GitHub Releases when publishing v0.1.
