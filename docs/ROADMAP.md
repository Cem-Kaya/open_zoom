# OpenZoom 2.0 Roadmap — The Lecture Assistant

Mission: the best possible tool for a legally blind university student to follow
live lectures using a phone camera clamped to a laptop. Everything below serves
that scenario: shaky mount, projected slides, whiteboards, variable lighting,
and the need to capture material for later study.

## Product pillars

1. **Stable, readable image** — GPU stabilization kills mount vibration;
   low-vision color schemes and contrast controls make slides readable.
2. **AI that takes notes for you** — OCR and vision-language models turn what
   the camera sees into text, explanations, and a timestamped notes file.
3. **Two-speed UI** — Simple mode: a handful of huge, high-contrast buttons.
   Advanced mode: every parameter. Nothing in between to learn.
4. **GPU-first** — CUDA is the processing path. The CPU effects path is
   deprecated: without a supported GPU the app shows unprocessed passthrough
   video and says so clearly, rather than silently degrading.

## Phase 1 (implemented in this pass)

### Video stabilization (CUDA, zero readback)
Projection-profile global motion estimation, fully on-GPU so the pipelined
presenter never stalls:
1. Downsample current frame to a small luma image (≤320×180).
2. Row and column projection profiles (sum of luma per row / per column).
3. 1D correlation against the previous frame's profiles over ±16 px finds the
   frame-to-frame translation (dx, dy) — one tiny kernel, result stays in
   device memory.
4. Motion filtering in device memory: accumulate the camera path, low-pass it
   (strength-controlled exponential smoothing), and derive a correction =
   smoothed path − actual path, clamped to a crop margin.
5. A warp kernel applies the correction with bilinear sampling as the first
   pipeline stage.
Projections use *every pixel*, which makes the estimate robust to noise and
compression artifacts — well suited to small-amplitude tremble from a phone
resting on a laptop hinge. State (previous luma, path accumulators) lives in
`CudaInteropSurface` and resets on camera/resolution change.

### Low-vision display modes (CUDA)
A display-color stage: None / Invert / White-on-black / Yellow-on-black /
Black-on-yellow, plus contrast and brightness controls. Schemes map luma onto
foreground/background colors — the classic high-legibility combinations for
low-vision reading.

### Screen fix: keystone correction + auto contrast (CUDA)
- **Straighten Screen (keystone)**: automatic detection of the projected
  slide/screen quad viewed at an angle, low-passed corner tracking, and a
  bilinear homography warp that makes it fronto-parallel. Used by the
  `Projector Screen` and `Whiteboard` built-in quick modes.
- **Auto contrast**: 256-bin luma histogram + 2nd/98th-percentile level
  stretch with a strength slider, applied before the display color grade to
  rescue washed-out projector output.

### Text clarity (CUDA)
- Shared local luma mean/variance drives illumination flattening, adaptive soft
  Sauvola thresholding, automatic text polarity, stroke weight, and bounded
  glare suppression.
- Bilateral denoise plus halo-clamped unsharp masking can be restricted to
  detected stroke edges. Optional 8x8 CLAHE and temporal mask hysteresis cover
  mixed lighting and edge shimmer.
- Two-color output reuses the low-vision color schemes. A `Document` preset and
  compact Simple `Text Clarity` master expose the stack without requiring the
  individual controls.
- Asynchronous Laplacian-variance focus scoring copies only two floats every
  15 frames and gates blurry OCR captures.
- NVIDIA Maxine SuperRes is integrated behind `OPENZOOM_ENABLE_TEXT_SR` as a
  runtime-loaded optional tier. It shares the CUDA stream, falls back to NIS
  when unavailable or too slow, and ships no proprietary runtime or weights.

### GPU pipeline completion
- GPU color-space conversion: capture prefers NV12, then YUY2, before BGRA;
  raw planes upload compactly and are converted to BGRA in CUDA. The CPU
  converter remains only for the debug view, other camera formats,
  GPU-unavailable passthrough, and per-frame fallback.
- Rotation on GPU for the raw NV12/YUY2 formats (applied after conversion).
- Async readback: recording and the periodic assistive grab use a two-slot
  asynchronous D3D12 readback ring that never blocks the render loop; photos
  and on-demand analysis keep the synchronous path.

### Robust capture and recording
- Camera reconnect: mid-stream device loss triggers an automatic same-device
  reconnect state machine (2s/4s/8s backoff, ~30 s) with no modal dialogs.
- Fragmented MP4 recording with free-disk-space guards (no start under
  500 MB, clean finalize under 200 MB), so an interrupted recording stays
  playable.

### AI expansion
- VLM (image→text) configuration moves into the app: base URL, API key, model,
  prompt — persisted in settings, editable in a dialog. Works with OpenAI's API
  and any OpenAI-compatible local server (LM Studio, Ollama, llama.cpp server),
  so image-to-text can run fully locally.
- **Lecture notes**: a per-session HTML file (`output/notes/`) that
  automatically collects timestamped OCR text and scene explanations, and
  embeds portable relative references to captured photos. The browser-ready
  format preserves selectable text and displays images without Markdown
  preview quirks.
- On-demand analysis ("Explain now") in addition to the periodic loop.
- Shared, persisted Assistant Instructions for response language, tone, and
  detail across Codex and OpenAI-compatible providers, with a separate scene
  prompt and configurable Codex reasoning effort.
- Manual text-to-speech of OCR/VLM results with installed Windows voice and
  speed selection (Qt TextToSpeech, if available).
- Track a supported Microsoft API for Narrator/Magnifier Natural voices. Their
  AppX model packages are currently private to Windows accessibility clients
  and do not appear in public `Windows.Media.SpeechSynthesis::AllVoices`.
- OCR hardening: configurable tesseract path + language, process watchdog.

### UI overhaul
- `Simple / Advanced` as two states around one persistent render surface.
  Simple uses three auto-fading corner clusters over the full camera view, plus
  a temporary numbered quick-mode grid (number keys `1`–`9` apply the first
  nine quick modes, now including `Projector Screen` and `Whiteboard`).
  Advanced opens a narrow right-side inspector with global camera/orientation
  and all profile-owned tuning controls, including stabilization and display
  colors.
- AI settings dialog (URL/key/model/prompt, Tesseract path/language, manual
  Read Aloud voice/speed, notes) backed by `settings.json`; environment
  variables remain as fallback.
- Larger default fonts and visible focus indicators app-wide.

### CPU path deprecation
The CPU effects pipeline (zoom/BW/blur on CPU) is deprecated. CUDA is the
processing path; NV12/YUY2 conversion and rotation also run in CUDA now, so
CPU conversion code remains only for the debug composite view, other camera
formats, GPU-unavailable passthrough, and per-frame fallback. Without CUDA the
app presents unprocessed frames with a persistent "GPU required" notice.

## Phase 2 (next)
- Reading mode: OCR layout awareness — detect text lines, auto-follow the
  lecturer's laser pointer / current bullet, TTS reads line by line.
- Slide-change detection (frame difference after stabilization) → auto-OCR
  each new slide exactly once into the notes file, instead of on a timer.
- Rotation/scale terms in stabilization (projections handle translation only).

## Phase 3 (ideas)
- Second GPU backend: D3D12 compute shaders for non-NVIDIA hardware (the
  `ProcessingSettings` contract is backend-neutral by design).
- Local Whisper for lecture audio → merged transcript + slides in one notes
  timeline.
- Export notes to accessible formats (large-print PDF, HTML with semantic
  headings).
- Multi-camera: slides camera + board camera, switchable instantly.
