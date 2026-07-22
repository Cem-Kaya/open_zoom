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

### AI expansion
- VLM (image→text) configuration moves into the app: base URL, API key, model,
  prompt — persisted in settings, editable in a dialog. Works with OpenAI's API
  and any OpenAI-compatible local server (LM Studio, Ollama, llama.cpp server),
  so image-to-text can run fully locally.
- **Lecture notes**: a per-session markdown file (`output/notes/`) that
  automatically collects timestamped OCR text and scene explanations, and
  references captured photos. Study material writes itself during class.
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
  a temporary numbered quick-mode grid.
  Advanced opens a narrow right-side inspector with global camera/orientation
  and all profile-owned tuning controls, including stabilization and display
  colors.
- AI settings dialog (URL/key/model/prompt, tesseract path/language, manual
  Read Aloud voice/speed, notes) backed by `settings.json`; environment
  variables remain as fallback.
- Larger default fonts and visible focus indicators app-wide.

### CPU path deprecation
The CPU effects pipeline (zoom/BW/blur on CPU) is deprecated. CUDA is the
processing path; CPU code remains only for camera-format conversion and
rotation that feed the GPU, and for the legacy debug composite view. Without
CUDA the app presents unprocessed frames with a persistent "GPU required"
notice.

## Phase 2 (next)
- GPU color-space conversion (NV12/YUY2 → BGRA in CUDA) to retire the last CPU
  per-pixel work.
- Rotation on GPU; remove the CPU resample stage.
- Async readback (P3) so recording never touches the render loop.
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
