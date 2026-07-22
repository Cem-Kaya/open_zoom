# Changelog

## [Unreleased]
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
- Refined the camera-first UI from four Simple clusters to three: processing
  status now lives under Advanced Image diagnostics instead of covering the
  live view. AI Settings moved into the Advanced tab header beside wrapping
  previous/next section arrows.
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
  installed catalog's `gpt-5.5` image model with Extra high (`xhigh`) reasoning
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
- Added bundled and standard-install Tesseract discovery, TESSDATA setup, and
  Windows release packaging for the local OCR runtime, language data, and
  upstream notices.
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
- Added lecture notes: a per-session markdown file under `output/notes/` that
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
