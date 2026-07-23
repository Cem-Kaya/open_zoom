# OpenZoom Code Reference

Authoritative code map for the current repository state as of 2026-07-23. Update this file whenever classes, public structs, or significant functions change.

## App Module

### `include/openzoom/app/app.hpp`
`openzoom::OpenZoomApp`
- Composition root for Qt/COM/Media Foundation startup and the application
  services. It delegates scheduling, settings/presets, widget synchronization,
  assistive behavior, interaction, and paired recording to focused manager
  classes instead of implementing those policies directly.
- Public API:
  - `OpenZoomApp(int& argc, char** argv)`
  - `~OpenZoomApp()`
  - `bool Initialize()` — performs fallible platform, UI, presenter, camera,
    settings, and service initialization after construction
  - `int Run()`
- Implementation is separated by responsibility:
  - `src/app/app_bootstrap.cpp` — lifecycle, service construction, signal wiring
  - `src/app/app_pipeline_runtime.cpp` — camera-clock processing, persistent
    scene publication, viewport-only presentation, readback, photos
  - `src/app/app_interaction.cpp` — input routing, status, focus mapping
  - `src/app/app_controls.cpp` — processing-control slots
  - `src/app/app_settings.cpp` — presets and persistent settings
  - `src/app/app_assistant.cpp` — Assistant and assistive actions
  - `src/app/app.cpp` — intentionally empty compatibility translation unit
- The low-level CUDA/presentation entry points remain private `OpenZoomApp`
  methods; `PipelineOrchestrator` owns their scheduling and synchronization
  policy through callbacks.

### `include/openzoom/app/pipeline_orchestrator.hpp`
`openzoom::FenceSequencer`
- Owns the single monotonic D3D12/CUDA fence timeline. CUDA reservations are
  committed only after submission; failed submissions roll back so no queue
  waits on a value that will never be signaled.
- `BeginGraphicsFrame(presenterSignaledValue)` reserves above the presenter's
  latest internal frame-slot signal. `GraphicsSignaled(actualValue)` adopts
  the actual last value after every successful present, so viewport-only draws
  and camera processing cannot reuse or rewind a fence value.

`openzoom::PipelineOrchestrator`
- Owns the two-clock scheduler, active/idle viewport rate policy, elapsed-time
  tick timing at nanosecond precision, dirty/presentation generation state, display-rate clamp
  reporting, camera reconnect backoff, fence sequencing, and repeated CUDA
  failure count.
- Camera-clock work runs only when `tick(...)` consumes a fresh capture frame.
  Viewport motion can present the last completed CPU or GPU scene at the
  effective display rate without advancing temporal effects, recording, OCR,
  or SuperRes.
- Public controls include `Start()`, `Stop()`, `UpdateTimerPolicy()`,
  viewport rate/fit setters, dirty/motion/present notifications, measured rate
  and timing accessors, reconnect methods, and fence/failure methods.
- `Callbacks` supplies fresh-frame processing, motion/presenter/camera state,
  active display refresh discovery, and a one-shot explicit-rate clamp notice.

### `include/openzoom/app/recording_manager.hpp`
`openzoom::CapturedFrame`, `openzoom::RecordingState`,
`openzoom::RecordingManager`
- Own synchronized original/processed recording state, live AV1-to-H.264
  negotiation through two `VideoRecorder` instances, matching async readbacks
  by request id, UI button state, elapsed time, and paired teardown.

### `include/openzoom/app/settings_controller.hpp`
`openzoom::SettingsController`
- Owns the persisted settings document and settings path, built-in/user preset
  lookup, live-config decoration/matching, quick-option promotion, and saving.

### `include/openzoom/app/ui_state_manager.hpp`
`openzoom::UIStateManager`
- Caches the `MainWindow` widget accessors in one UI-only object and translates
  between controls and `settings::AdvancedConfig`. It centralizes signal
  blocking during programmatic configuration changes.

### `include/openzoom/app/assistive_feature_manager.hpp`
`openzoom::AssistiveFeatureManager`
- Owns `AssistiveRuntime` plus the floating `AssistiveOverlay`, periodic
  analysis cadence, OCR/VLM mode state, focus warnings, TTS/result routing,
  and persisted camera-relative overlay geometry.

### `include/openzoom/app/suspend_guard.hpp`
`openzoom::SuspendGuard`
- Small non-copyable RAII latch used while applying settings/UI state. It
  restores the referenced suspension flag even when a scope exits early.

### `include/openzoom/app/interaction_controller.hpp`
`openzoom::InteractionController`
- Converts keyboard, wheel, mouse-drag, and virtual joystick input into zoom-center updates.
- Public API:
  - `explicit InteractionController(OpenZoomApp& app)`
  - `bool HandlePanKey(int key, bool pressed)`
  - `bool HandlePanScroll(const QWheelEvent* wheelEvent)`
  - `void HandleZoomWheel(int delta, const QPointF& localPos)`
  - `bool ApplyInputForces(double elapsedSeconds)` — integrates motion as
    normalized units per second, so 60 and 120 FPS move at the same speed
  - `bool HasContinuousMotion() const`
  - `void BeginMousePan(const QPointF& pos, const QSize& widgetSize)`
  - `bool UpdateMousePan(const QPointF& pos)`
  - `void EndMousePan()`
  - `bool IsMousePanActive() const`
  - `void ResetJoystick()`
  - `void SetJoystickAxes(float x, float y)`

### `include/openzoom/app/setup_assistant.hpp`
`openzoom::SetupAssistantDialog`
- Nonmodal, accessible dependency manager shown after first paint when an
  optional dependency is missing and the user has not declined automatic
  prompting. Advanced `Setup & Downloads` can reopen it at any time.
- Streams pinned Tesseract, official Codex bootstrap, and
  architecture-specific NVIDIA Video Effects downloads through
  `QNetworkAccessManager`, cancels after 60 seconds without activity, and
  verifies the complete file with SHA-256 before execution.
  Failed Qt transfers automatically retry asynchronously through Windows
  `curl.exe`; Tesseract can then retry its alternate vendor host. Every
  transport uses the same pinned digest, and final failures offer the vendor
  release page.
- NVIDIA's verified installer is launched with `ShellExecuteExW` and the
  `runas` verb so administrator-required manifests receive a real UAC consent
  flow instead of failing through `QProcess` with Windows error 740. A timer
  monitors the returned process handle and refreshes status on completion; the
  in-progress row directs the user to continue in the separate vendor
  installer window.
- Tesseract installs under the current user's OpenZoom data directory and can
  be removed only from that managed directory. NVIDIA installation/removal is
  delegated to the vendor installer or registered uninstaller.
- Codex installs or updates through OpenAI's verified per-user bootstrap.
  OpenZoom detects its standalone and WinGet paths, persists the resolved
  executable, and keeps ChatGPT authentication as a separate AI Settings
  action. Because the official installer exposes no uninstall action, the row
  opens the install folder or official setup guide instead.
- Dependency rows pair explicit text with large green check/red X status
  indicators. A system-wide Tesseract copy exposes `Open Windows Apps`; only
  the OpenZoom-managed copy exposes direct `Remove`.
- Public API:
  - `SetupAssistantDialog(const QString& configuredTesseractPath, const QString& configuredCodexPath, bool declined, QWidget* parent = nullptr)`
  - `~SetupAssistantDialog()`
  - `static bool NeedsSetup(const QString& configuredTesseractPath, const QString& configuredCodexPath)`
  - `static QString FindTesseractExecutable(const QString& configuredPath = {})`
  - `static QString FindCodexExecutable(const QString& configuredPath = {})`
  - `static QString ManagedTesseractDirectory()`
- Signals: `TesseractPathChanged(...)`, `CodexPathChanged(...)`,
  `DeclinePreferenceChanged(...)`, and `DependenciesChanged()`.
- `closeEvent(...)` cancels an in-flight download, but keeps the dialog alive
  while a launched vendor installer is running so Qt cannot terminate it by
  destroying the child `QProcess`.

### `include/openzoom/app/color_schemes.hpp`
Namespace `openzoom::color_schemes`
- `enum class SchemeMode { Duotone, Posterize, Gradient }` selects two-stop,
  hard-banded, or smoothly interpolated luma mapping.
- `struct ColorScheme` is the single display-color model: stable id, visible and
  accessible names, mode, 2-8 `QColor` stops, stepped/text-polarity flags,
  optional legacy mode, and effect classification.
- `using ColorLut = std::array<std::uint32_t, 256>` stores packed BGRA results.
- `BuiltInColorSchemes()`, `LegacyColorScheme(int)`, and
  `FindBuiltInColorScheme(...)` expose the authoritative table and legacy
  migration.
- `NormalizeColorScheme(...)` and `BuildColorLut(...)` validate a scheme and
  generate its GPU-ready table.
- `TextForegroundBgra(...)` / `TextBackgroundBgra(...)` supply the same active
  scheme endpoints to Text Clarity compositing.
- `PackBgra(...)`, `UnpackBgra(...)`, `ModeToken(...)`, `ModeFromToken(...)`,
  and `SchemesEquivalent(...)` support persistence and comparison.

### `include/openzoom/app/settings_store.hpp`
Namespace `openzoom::settings`
- `enum class ViewportRateMode`
  - global viewport navigation policy: `AutoUpTo120`, fixed 60/90/120 FPS, or
    `MatchDisplay`; explicit choices clamp to the active monitor
- `enum class ViewportFitModeSetting`
  - global aspect policy: default crop-to-fill or show-all fit with black bars
- `struct AdvancedConfig`
  - full profile-owned tuning payload including image-processing and focus
    state, OCR/VLM mode flags, stabilization, structured `colorScheme` plus the
    migration-only `displayColorMode`, screen fix, and Text Clarity
    (`autoTextClarityEnabled`, background flatten, adaptive
    binarization/Sauvola/softness, polarity, stroke weight, smart sharpen,
    CLAHE, two-color output, mask hysteresis, selective sharpen, focus
    detection, glare suppression, `mlSuperResEnabled`, and
    `mlSuperResStrength`; optional, default-off `mlSuperResPrefer2x` raises the
    profile's minimum magnification to 2x for a smaller, faster, narrower-view
    AI source crop; mutually exclusive `mlSuperResUltra1440p` uses the complete
    processed camera frame and a separate cache capped at 1440p-class output)
  - legacy `mlTextSuperResolution*` JSON keys migrate to the current Maxine
    keys on load; new settings are written only under `mlSuperRes*`
  - `rotationQuarterTurns` is read from old profile JSON only for backward migration, is no longer written into profiles, and is ignored by current profile comparisons
- `struct AssistiveSettings`
  - AI/assistive configuration: `aiProvider`, `codexExecutablePath`, `codexModel`, `codexReasoningEffort`, `codexInternetEnabled`, `codexCodingEnabled`, `codexWorkspaceDirectory`, `assistantInstructions`, `vlmApiUrl`, `vlmApiKey`, `vlmModel`, `vlmPrompt`, `tesseractPath`, `ocrLanguage`, `ttsEngine`, `ttsVoiceName`, `ttsVoiceLocale`, `ttsRate`, and `lectureNotesEnabled`; non-empty values take precedence over matching environment fallbacks
- `struct CodexConversation`
  - OpenZoom-owned thread index entry: `threadId`, `title`, `preview`, `createdAt`, and `updatedAt`; transcripts remain in the Codex thread store
- `struct PresetDefinition`
  - stage-1 quick-mode metadata: preset id, name, description, target config id, built-in flag
- `struct PersistentSettings`
  - persists global `cameraIndex` and `rotationQuarterTurns`, UI state,
    `simpleUiMode`, resizable `advancedPanelWidth`, camera-relative `assistiveOverlayGeometry`,
    `viewportRateMode` and `viewportFitMode`,
    `setupAssistantDeclined`, selected preset id, current live advanced config,
    reusable `customColorScheme`,
    the `assistive` settings block,
    OpenZoom-created `codexConversations`, and user-created configs/presets
- Functions:
  - `QString ResolveSettingsPath()`
  - `void EnsureSettingsDirectory(const QString& path)`
  - `std::optional<PersistentSettings> Load(const QString& path)`
  - `bool Save(const QString& path, const PersistentSettings& settings)`
  - `const std::vector<AdvancedConfig>& BuiltInConfigs()`
  - `const std::vector<PresetDefinition>& BuiltInPresets()`
  - `QString DefaultPresetId()`
  - `const AdvancedConfig* FindAdvancedConfigById(...)`
  - `const PresetDefinition* FindPresetById(...)`
  - `std::optional<AdvancedConfig> ResolveConfigForPreset(...)`
  - `bool AreConfigsEquivalent(...)` — compares slider-backed values at their
    representable UI precision so applied presets remain selected after
    control quantization

### `include/openzoom/app/constants.hpp`
Namespace `openzoom::app_constants`
- UI scaling and step constants for zoom, panning, and blur controls
- Helper functions:
  - `SliderValueToSigma(int sliderValue)`
  - `SnapBlurRadius(int value)`

## Capture Module

### `include/openzoom/capture/media_capture.hpp`
Types:
- `struct MediaFrame`
  - `data`, `subtype`, `width`, `height`, `stride`, `dataSize`
- `using FrameCallback = std::function<void(MediaFrame&& frame)>` — transfers
  ownership from the capture thread into the app's mutex-protected latest-frame
  slot; the Qt frame tick moves it back out, avoiding two full-frame copies
- `using CaptureErrorCallback = std::function<void(const std::string& message)>`
- `struct CameraDescriptor`
  - `name`, `symbolicLink`, `activation`
- `struct VideoFormat`
  - `width`, `height`, `numerator`, `denominator`
- `enum class CameraFailureKind`
  - `None`, `DeviceBusy`, `DeviceMissing`, `AccessDenied`, `Other` — plain-language classification of the most recent capture failure

`openzoom::MediaCapture`
- Media Foundation camera enumeration and threaded source-reader capture.
- Public API:
  - `MediaCapture()`
  - `~MediaCapture()`
  - `bool Initialize()`
  - `void Shutdown()`
  - `std::vector<CameraDescriptor> EnumerateCameras()`
  - `std::vector<VideoFormat> EnumerateFormats(const CameraDescriptor& descriptor)`
  - `bool StartCapture(const CameraDescriptor& descriptor, FrameCallback callback, GUID preferredSubtype = MFVideoFormat_NV12, CaptureErrorCallback errorCallback = {})` — requests the compact NV12 format first so CUDA can perform color conversion, then falls back through YUY2/BGRA formats; retries transient busy/resource errors internally; on failure `LastError()` holds a full plain-language sentence for busy/missing/access-denied kinds
  - `void StopCapture()`
  - `const std::string& LastError() const`
  - `CameraFailureKind LastFailureKind() const`
  - `double CurrentFrameRate() const` — reports the negotiated capture rate;
    diagnostics do not claim viewport re-presentation as camera FPS
  - `bool ConsumeDeviceLost()` — atomically returns and clears the mid-stream device-loss flag; polled from the app's frame tick to drive reconnection
  - `const std::wstring& LastSymbolicLink() const` — symbolic link of the most recently started device, kept across loss/stop so reconnection can re-find the same physical camera
- Internal helpers:
  - `ConfigureReader(...)`
  - `ReadCurrentFormat(IMFSourceReader* reader, FrameFormat& outFormat)`
  - `CaptureLoop(FrameCallback callback, CaptureErrorCallback errorCallback)`
  - `ExtractFormats(IMFSourceReader* reader)`
  - `HrToString(HRESULT hr)`
- Capture ownership:
  - `activeActivation_` retains the activation object for the live session and `StopCapture()` calls `ShutdownObject()` before releasing it
  - temporary mode enumeration uses the same balanced activation/shutdown contract
  - the capture callback moves each completed `MediaFrame`; it does not retain
    or access the moved buffer after invoking the consumer

## Common Module

### `include/openzoom/common/codex_app_server_client.hpp`
`openzoom::CodexAppServerClient`
- Native Qt `QProcess`/JSON-RPC client for the local `codex app-server` stdio transport.
- The public surface covers server lifecycle, account/login state,
  model/rate-limit discovery, assistant turns, cancellation, and OpenZoom
  conversation load/rename/delete operations.
- Restricted mode uses read-only sandboxing, approval policy `never`, disabled
  network access, and vision-only developer instructions. Persistent Advanced
  Assistant turns can opt into web/network access and workspace-scoped command
  execution/file changes. Simple Explain never inherits those permissions.
  MCP, dynamic, and collaboration tool items always trigger interruption.
  Server-initiated command/file approvals are declined, and permission-profile
  requests receive an empty grant so no additional filesystem or network
  capability is accidentally approved.
- Public API:
  - `Configure(const QString& executablePath, const QString& preferredModel, const QString& reasoningEffort, const QString& assistantInstructions, bool internetEnabled, bool codingEnabled, const QString& workspaceDirectory)`
  - `Start()` / `Shutdown()`
  - `IsReady()`, `IsSignedIn()`, `IsTurnActive()`, `SelectedModel()`
  - `BuiltInAssistantInstructions()` returns the read-only OpenZoom identity
    prompt shown in AI Settings
  - `RefreshAccount()` / `StartChatGptLogin()`
  - `RequestVisionTurn(...)` / `InterruptTurn()`
  - `LoadConversation(...)`, `RenameConversation(...)`, `DeleteConversation(...)`
- Signals cover server/account/model/rate-limit state, login URL, conversation
  lifecycle/transcript loading, and streamed turn start/delta/completion.
  `ModelCatalogChanged(const QJsonArray&, ...)` preserves app-server display
  names, defaults, modalities, and supported reasoning efforts while the
  compatibility `ModelsChanged(QStringList, ...)` signal remains available.
  Destruction suppresses outward lifecycle notifications while synchronously
  stopping the child process, so owners cannot receive callbacks from
  partially destroyed service state.

### `include/openzoom/common/assistive_runtime.hpp`
Supporting types:
- `struct AssistiveRuntimeConfig`
  - same provider/Codex/VLM/OCR/speech fields as `settings::AssistiveSettings`, including Codex internet/coding/workspace permissions, plus `QString notesDirectory` (absolute directory for lecture notes; empty disables notes)

`openzoom::AssistiveRuntime`
- Asynchronous assistive-analysis runtime owned by `OpenZoomApp`.
- OCR path:
  - exports the current frame to a temporary PNG
  - resolves configured/environment, Setup Assistant-managed, PATH, and
    standard Windows Tesseract locations, including sibling `tessdata`
  - runs `tesseract.exe` asynchronously
  - returns extracted text to the overlay
- VLM path:
  - default: saves a temporary JPEG and submits it as `localImage` through `CodexAppServerClient`; Codex explanations are on-demand rather than periodic
  - fallback: JPEG-encodes the frame and posts an OpenAI-compatible `chat/completions` request; API keys are optional for local servers
  - streams/finalizes text into the overlay, notes, and Assistant signals; TTS is invoked separately by `ReadAloud(...)`
- Public API:
  - `AssistiveRuntime(QObject* parent = nullptr)`
  - `~AssistiveRuntime()`
  - `void SetConfig(const AssistiveRuntimeConfig& config)`
  - `void SetModes(bool ocrEnabled, bool vlmEnabled)`
  - `bool WantsAnalysis() const`
  - `bool IsBusy() const`
  - `bool IsCodexTurnActive() const`
  - `void SubmitFrame(const uint8_t* bgraData, int width, int height)`
  - `void SubmitFrameForced(const uint8_t* bgraData, int width, int height, bool runOcr, bool runVlm)` — runs the requested analyses immediately regardless of enabled modes
  - `void ReadAloud(const QString& text)` — speaks a result only after an explicit user request, using the configured Windows voice and speed
  - `void DismissOverlay()` — hides the current result panel until a new forced OCR, Explain, or Assistant request begins
  - Codex/Assistant control: `StartCodexLogin()`, `StopAssistant()`, `SubmitAssistantPrompt(...)`, `LoadAssistantConversation(...)`, `RenameAssistantConversation(...)`, `DeleteAssistantConversation(...)`
  - `void NoteCapturedPhoto(const QString& filePath)` — appends a browser-renderable relative photo reference to the HTML lecture notes
  - `QString notesFilePath() const` — absolute path of the current notes file (empty until something is written)
- Manual-only text-to-speech of results via Qt TextToSpeech when built with `OPENZOOM_HAS_TTS=1`; the runtime prefers the `winrt` engine and falls back to Qt's default engine.
- Lecture notes: a valid per-session HTML document in the configured notes directory collecting escaped timestamped OCR text, scene explanations, and portable relative photo references. Updates use `QSaveFile` replacement around an insertion marker so the document remains complete after every entry.
- Signals:
  - `OverlayUpdated(const QString& title, const QString& body, bool visible)`
  - Codex server/account/model/rate-limit/login state
  - Assistant conversation lifecycle, transcript, and streamed turn events

### `include/openzoom/common/image_processing.hpp`
Namespace `openzoom::processing`
- Format conversion helpers:
  - `CopyArgbToBgra(...)`
  - `CopyRgbxToBgra(...)`
  - `ConvertNv12ToBgra(...)`
  - `ConvertYuy2ToBgra(...)`
  - CPU pipeline dispatch plus the NV12/YUY2 helpers reject dimensions above
    16384 per axis, undersized strides/source buffers, overflowed size
    calculations, and unsafe odd-width packed layouts before writing output
- CPU effect helpers:
  - `ApplyBlackWhite(...)`
  - `ApplyZoom(...)`
  - `ApplyGaussianBlur(...)`
  - `ApplyTemporalSmoothCpu(...)`

### `include/openzoom/common/frame_pipeline.hpp`
Namespace `openzoom::processing`
- `struct CpuPipelineConfig`
  - `enableBlackWhite`, `blackWhiteThreshold`
  - `enableZoom`, `zoomAmount`, `zoomCenterX`, `zoomCenterY`
  - `enableBlur`, `blurRadius`, `blurSigma`
  - `enableTemporalSmooth`, `temporalSmoothAlpha`
- `struct CpuPipelineOutput`
  - `data`, `width`, `height`, `isComposite`

`CpuFramePipeline`
- Owns the CPU fallback pipeline and the intermediate stage buffers.
- Public API:
  - `bool ConvertFrameToBgra(...)`
  - `bool RotateRawBuffer(int quarterTurns, UINT& width, UINT& height)`
  - `CpuPipelineOutput BuildStages(UINT width, UINT height, const CpuPipelineConfig& config, bool debugViewEnabled)`
  - `bool ResampleToFill(UINT targetWidth, UINT targetHeight, float centerXNorm, float centerYNorm)`
  - `void ResetTemporalHistory()`
  - `const std::vector<uint8_t>& StageRaw() const`
  - `UINT RawWidth() const`
  - `UINT RawHeight() const`

### `include/openzoom/common/media_writer.hpp`
`openzoom::VideoRecorder`
- Media Foundation sink-writer wrapper for live AV1 or H.264 output. The
  container is fragmented MP4 (fMP4): fragments flush to disk while recording,
  so the file stays playable up to the last completed fragment even if the
  process dies before finalization. Files keep the `.mp4` extension.
- `enum class Codec` — `Av1`, `H264`
- `enum class StopReason` — `None`, `Manual`, `DiskFull`, `WriteFailed`; why
  the recorder last transitioned from recording to stopped.
- Public API:
  - `VideoRecorder()`
  - `~VideoRecorder()`
  - `bool Start(const std::wstring& filePath, UINT width, UINT height, UINT fps, Codec codec)` — starts the requested live encoder and refuses to start with under 500 MB free on the target volume (`LastError()` explains why)
  - `void Stop()`
  - `bool IsRecording() const`
  - `bool AddFrame(const uint8_t* bgraData, size_t strideBytes)` — free space is re-checked every ~5 seconds; below 200 MB the recording is finalized cleanly and `AddFrame` returns false with `StopReason::DiskFull` (the file is already intact on disk)
  - `double DurationSeconds() const`
  - `const std::string& LastError() const`
  - `StopReason LastStopReason() const`
  - `Codec ActiveCodec() const`
  - `static const char* CodecName(Codec codec)`
- Internal helpers:
  - `InitializeSink(...)`
  - `FinalizeAndStop(StopReason reason)`
  - `SetError(const std::string& err)`

### `include/openzoom/common/maxine_superres.hpp`
`openzoom::MaxineSuperRes`
- GPL-clean runtime-only adapter for NVIDIA Video Effects SuperRes. The
  implementation includes the staged MIT headers but resolves every
  proprietary entry point from `NVCVImage.dll` and `NVVideoEffects.dll` with
  `LoadLibraryExW`/`GetProcAddress`; it has no import-library dependency.
- Discovery checks an explicit override, `OPENZOOM_MAXINE_PATH`,
  `NV_VIDEO_EFFECTS_PATH`, the standard Program Files directory, and uninstall
  registry views. Models are loaded from the detected runtime's `models`
  subdirectory. A missing runtime is cached as unavailable without crashing.
- Converts BGRA8 device memory to/from planar BGR float using
  `NvCVImage_Transfer`, runs the effect on the caller's CUDA stream, and never
  reads a frame back to the host.
- Public API:
  - `MaxineSuperRes()` / `~MaxineSuperRes()`
  - `bool Ensure(...)` for explicit source/destination dimensions or integer scale
  - `bool Run(...)` validates pitched crop views and follows NVIDIA's
    synchronous inference path before copying the complete result into the
    device destination, preventing previous-frame output from appearing as a
    moving ghost layer
  - `void SetStrength(float strength)` / `void Teardown()`
  - `bool IsReady() const` / `bool IsAvailable()`
  - `const std::string& LastError() const`
  - `const std::wstring& RuntimeDirectory() const`
  - `static std::wstring FindRuntimeDirectory(...)`
  - `static bool IsRuntimeInstalled(...)`

### `include/openzoom/common/view_transform.hpp`
`openzoom::ViewportFitMode`, `openzoom::ViewTransform`,
`openzoom::NormalizedSourceRect`, `openzoom::PixelViewMapping`
- Define the single camera-to-viewport geometry contract used by the D3D12
  shader, CPU fallback, pointer mapping, focus marker, and cached SuperRes ROI.
  The mapping always uses one uniform scale.
- `ComputeViewTransform(...)` computes aspect-safe Fill/crop or Fit/letterbox
  source and destination rectangles from rotated scene size, native viewport
  pixels, zoom, and focus.
- `RemapViewTransformToSourceRect(...)` expresses the requested scene view
  inside a cached ROI and rejects views that fall outside it.
- `ComputePixelViewMapping(...)` derives integer destination bounds and source
  sampling increments for the CPU fallback from the same transform.

## D3D12 Module

### `include/openzoom/d3d12/presenter.hpp`
`openzoom::ViewportPresentationOptions`
- Optional per-present focus marker and readback request used by the viewport
  shader pass.

`openzoom::D3D12Presenter`
- Manages the D3D12 device, native-client-sized swap chain, viewport shader,
  per-frame upload buffers, shared fence, and readback rings.
- `PresentSceneTexture(...)` samples a persistent processed scene through
  `ViewTransform` using a full-screen triangle and bilinear sampler. The back
  buffer remains the render HWND's native pixel size; camera texture dimensions
  never resize it, so DXGI cannot stretch one camera axis independently.
- Presentation uses two frame slots plus the swap chain frame-latency waitable
  object. A frame-slot timeout drops and counts the present instead of queuing
  unbounded work. `NeedsScenePresent()`, viewport dimensions, and
  `MissedPresentCount()` expose scheduler/diagnostic state.
- Shared-fence presentation chooses values above the presenter's current
  value, the queued CUDA wait, and the caller reservation; the actual
  post-Present frame-slot value is the authoritative value returned by
  `GetLastSignaledFenceValue()`.
- Native resize requests are coalesced by `RenderWidget`. `Resize(...)` drains
  in-flight back buffers before `ResizeBuffers`, while CPU upload buffers are
  recreated lazily only if the CPU path next presents at the new size.
- Public API:
  - `D3D12Presenter()`
  - `~D3D12Presenter()`
  - `void Initialize(HWND hwnd, UINT width, UINT height)`
  - `bool IsInitialized() const`
  - `void Resize(UINT width, UINT height)`
  - `void Present(const uint8_t* data, UINT width, UINT height)`
  - `void PresentFromTexture(ID3D12Resource* texture, UINT width, UINT height, const FenceSyncParams* fenceSync = nullptr)`
  - `bool PresentSceneTexture(ID3D12Resource*, UINT sourceWidth,
    UINT sourceHeight, const ViewTransform&, const FenceSyncParams*,
    const ViewportPresentationOptions*, UINT64* outReadbackRequestId)`
  - `bool ReadbackTexture(ID3D12Resource* texture, UINT width, UINT height,
    std::vector<uint8_t>& outBgra, UINT64 waitFenceValue = 0)` — optionally
    queues a shared-fence wait before the blocking GPU copy, used by on-demand
    OCR and Assistant frame attachment to avoid reading active CUDA writes
  - `bool RequestReadback(ID3D12Resource* texture, UINT width, UINT height, UINT64* outRequestId = nullptr)` — enqueues an async copy into a two-slot ring, optionally returning its fence-backed request id; returns false when both slots are in flight
  - `bool TryGetCompletedReadback(std::vector<uint8_t>& outBgra, UINT& outWidth, UINT& outHeight, UINT64* outRequestId = nullptr)` — moves the oldest completed request's tightly packed BGRA8 pixels out and optionally returns the matching request id. Pending requests are silently dropped by `Resize`
  - `ID3D12Device* GetDevice() const`
  - `ID3D12Fence* GetFence() const`
  - `UINT64 GetLastSignaledFenceValue() const`
  - `void WaitForIdle()` — drains all submitted GPU work; must be called before
    releasing resources in-flight frames may still reference (e.g. the shared
    CUDA texture)

## CUDA Module

### `include/openzoom/cuda/cuda_interop.hpp`
Supporting types:
- `struct FenceSyncParams`
  - `enable`, `waitValue`, `signalValue`
- `struct KeystoneTrackingState`
  - `paused`, `canStepBack`, `canStepForward`, `stepPending`, `position`, and
    `count` describe the bounded accepted-correction history exposed to Qt
- `struct SuperResRoiMetadata`
  - identifies the generation, normalized source rectangle, output extent, and
    supported NVIDIA scale of the cached SuperRes texture. Viewport
    presentation uses it only when the current `ViewTransform` is contained in
    that ROI; otherwise it immediately presents the registered conventional
    scene rather than blending mismatched images.
- `enum class SpatialUpscaler`
  - `kFsrEasuRcas`
  - `kNis`
- `enum class CudaBufferFormat`
  - `kRgba8`
  - `kRgba16F`
- `enum class DisplayColorTransform`
  - `kNone` preserves full color and keeps the identity fast path
  - `kInvert` preserves the legacy per-channel inversion path
  - `kLumaLut` maps byte luma through the active 256-entry table
- `struct ProcessingSettings`
  - toggles and parameters for BW, zoom, blur, focus marker, spatial sharpening, temporal smoothing, and staging format
  - stabilization: `enableStabilization`, `stabilizationStrength` (0..1, higher = stronger path smoothing)
  - display grading: `displayColorTransform`, host-owned `displayColorLut`, and
    monotonic `displayColorLutGeneration`; `textForegroundBgra` and
    `textBackgroundBgra` keep Text Clarity on the same scheme endpoints;
    `contrast` (0.25..4.0) and `brightness` (-1..1) remain independent
  - screen fix: `enableKeystone` (auto-detect the projected slide quad and warp it fronto-parallel), `enableAutoContrast` (percentile level stretch before contrast/brightness), `autoContrastStrength` (0..1 blend toward the full stretch)
  - text clarity: master/individual toggles and strength values for background
    flattening, adaptive Sauvola, soft edges, polarity, stroke weight, smart
    sharpening, CLAHE, two-color mapping, hysteresis, selective sharpening,
    focus detection and glare suppression
  - Maxine SuperRes: `enableMlSuperRes`, `mlSuperResStrength` (0..1), and
    `mlSuperResUltra1440p` for full-frame inference into the separate
    high-resolution cache
- `struct ProcessingInput`
  - `hostPixels`, `hostStrideBytes`, `pixelSizeBytes`, `width`, `height` — `width`/`height` always describe the host pixel layout (pre-rotation)
  - `inputFormat` — 0=BGRA8 (existing CPU-converted path), 1=NV12 (`hostPixels` = Y plane, `hostPlane2` = interleaved UV plane with `hostPlane2StrideBytes`), 2=YUY2 (packed in `hostPixels`)
  - `rotationQuarterTurns` — 0..3 clockwise, applied on the GPU after conversion for raw formats only (ignored for BGRA with a one-shot warning). For odd turns the interop surface must be created at the post-rotation extent (height x width); `ProcessFrame` validates and returns false on mismatch

`openzoom::CudaInteropSurface`
- Imports a D3D12 texture into CUDA and runs the GPU effect chain.
- All teardown paths (destructor, device-buffer and temporal-history release)
  synchronize the CUDA stream first, since `ProcessFrame` returns with kernels
  still queued when fence sync is active.
- Host upload uses two `cudaMallocHost` staging slots shared by BGRA, NV12,
  and YUY2. Only the Qt tick writes/rotates the slots. A per-slot CUDA event,
  recorded immediately after the final H2D copy, guards the next host write;
  shared D3D/CUDA fence values cannot protect host-side memory reuse.
- Public API:
  - `explicit CudaInteropSurface(ID3D12Resource* texture, ID3D12Fence* sharedFence = nullptr)`
  - `~CudaInteropSurface()`
  - `bool IsValid() const`
  - `bool HasExternalSemaphore() const`
  - `void RunGradientDemoKernel(unsigned int width, unsigned int height, float timeSeconds)`
  - `bool ProcessFrame(const ProcessingInput& input, const ProcessingSettings& settings, const FenceSyncParams& fenceSync)`
  - `const std::string& LastError() const`
  - `void ResetTemporalHistory()`
  - `void ResetStabilization()` — clears stabilization state (previous luma profiles, camera-path accumulators); the app calls it on camera switch/stop and rotation changes
  - `void ResetKeystone()` — clears the keystone detection state (smoothed source-quad corners, pending luma snapshot); the app calls it alongside every `ResetStabilization()` and when the keystone toggle changes
  - `void SetKeystoneTrackingPaused(bool paused)` — freezes or resumes periodic
    projected-quad detection without changing the current warp
  - `bool StepKeystoneCorrection(int direction)` — freezes tracking, restores
    the previous/next accepted corner set, or requests one fresh detection when
    stepping forward from the newest entry
  - `KeystoneTrackingState GetKeystoneTrackingState() const` — returns the
    current history and pending-step state; history is capped at 32 entries
  - `void ResetTextClarityHistory()` — invalidates the previous binary mask
  - `bool HasFocusScore() const` / `float LatestFocusScore() const` /
    `bool IsFocusAcceptable(float threshold) const`
  - `const std::string& SuperResStatus() const`, `bool IsSuperResActive() const`,
    and `void ResetSuperRes()` expose/reset the optional runtime tier;
    `SuperResSourceWidth()/SuperResSourceHeight()/SuperResFactor()` report the
    crop and the fixed scale factor of the active AI stage
  - `IsSuperResPerformanceLimited()` and `SuperResAverageMs()` report a latency
    guard decision against the 24 ms target;
    `SetSuperResPerformanceOverride(bool)` explicitly enables or restores the
    guard
  - `float LastGpuFrameMs() const` — P8 GPU timing: duration of the last
    sampled ProcessFrame kernel chain (cudaEvent pair recorded every 30th
    frame, polled non-blockingly by `ConsumeProcessTiming()`); negative until
    the first sample completes
- Profile and Advanced Text Clarity changes synchronize and reset SuperRes so
  the SDK's load-time strength/mode selectors are applied on the next frame.
  Enabling viewport-target mode restores the default 0.65 strength when zero
  and enforces its 1.33x minimum zoom before the next frame. The CUDA path
  snaps to the largest supported Maxine factor (4/3, 1.5, 2, 3 or 4) at or
  below the requested zoom whose source crop maps exactly onto the viewport,
  crops around the actual clamped zoom center, and applies residual
  magnification with the GPU sampler around the same mapped center. Ultra mode
  instead imports a second D3D12/CUDA texture whose extent may differ from the
  camera-sized primary scene, allocates a matching pitched CUDA output, runs
  the full source frame to at most 2560x1440 (or 1440x2560 after rotation), and
  lets the same presenter apply pan/zoom afterward. 720p uses 2x, 1080p uses
  4/3x, and 1440p remains native. Setup failures latch per
  (crop, factor) key and retry only when the key changes or the toggle is
  re-enabled. The model directory resolves to the runtime's `models`
  subdirectory, falling back to the DLL directory itself.
- Stage order: upload/convert/rotate → stabilization → keystone → Text Clarity
  analysis/flatten/CLAHE/mask/smart-sharpen → legacy BW → Maxine SuperRes or
  NIS/FSR zoom/sharpen → blur → temporal → auto-contrast → color grade → focus
  marker → interop copy. Maxine discards 10 warmup timings, averages the next
  60 effect runs with CUDA events, and disables itself for the surface above
  the 24 ms target unless the user explicitly overrides the guard.
- The luma LUT is copied to CUDA constant memory asynchronously only when
  `displayColorLutGeneration` changes. Contrast and brightness edits reuse the
  resident table and do not trigger uploads.

### `include/openzoom/cuda/cuda_kernels.hpp`
Kernel launch wrappers:
- `LaunchGradientKernel(...)`
- `LaunchBlackWhiteKernel(...)`
- `LaunchZoomKernel(...)`
- `LaunchBlackWhiteLinear(...)`
- `LaunchZoomLinear(...)`
- `LaunchGaussianBlurLinear(...)`
- `LaunchFocusMarkerLinear(...)`
- `LaunchFsrEasuRcasLinear(...)`
- `LaunchNisLinear(...)`
- `LaunchTemporalSmoothLinear(...)`
- `LaunchStabilizationLumaDownsample(...)`
- `LaunchStabilizationProjections(...)` — accumulates 16x16 block row/column
  partials with shared-memory atomics, then merges one partial per bin into
  global memory
- `LaunchStabilizationEstimate(...)`
- `LaunchStabilizationWarp(...)`
- `LaunchDisplayColorGradeLinear(...)` — takes the display transform plus
  optional device `autoContrastLevels` (float2 lo/hi) and
  `autoContrastStrength`; the LUT branch indexes constant memory by byte luma
- `LaunchNv12ToBgraLinear(...)` / `LaunchYuy2ToBgraLinear(...)` — BT.601 limited-range YUV to BGRA, integer math identical to the CPU converters
- `LaunchRotateQuarterLinear(...)` — rotate by 1/2/3 clockwise quarter turns (destination is srcHeight x srcWidth for odd turns)
- `LaunchKeystoneWarp(...)` — bilinear homography warp (output pixel to source pixel, row-major 3x3), black outside the source rect
- `LaunchAutoContrastHistogram(...)` / `LaunchAutoContrastAnalysis(...)` — 256-bin luma histogram plus 2nd/98th-percentile level derivation low-passed into device-resident levels
- `LaunchTextLocalStatistics(...)`, `LaunchTextSceneAnalysis(...)`,
  `LaunchBackgroundFlattenLinear(...)`, `LaunchClaheLinear(...)`,
  `LaunchSauvolaMask(...)`, `LaunchStrokeWeight(...)`,
  `LaunchTextMaskHysteresis(...)`, `LaunchTextMaskComposite(...)`,
  `LaunchSmartSharpenLinear(...)`, and `LaunchFocusMetric(...)`
- `bool UploadGaussianKernel(int radius, float sigma, cudaStream_t stream)` —
  normalizes exact Gaussian weights and queues the live weight/radius symbols
  on `stream`; no unused size symbol or device-wide synchronization remains
- `bool UploadDisplayColorLut(const std::uint32_t* lut256, cudaStream_t)` queues
  one 256-entry BGRA table upload on the processing stream

## UI Module

### `include/openzoom/ui/render_widget.hpp`
`openzoom::RenderWidget`
- Native widget that hosts the D3D12 presenter. Show and resize events create
  or coalesce native-client-pixel presenter resizes so splitter/window motion
  cannot make camera-frame dimensions take over the swap chain.
- Public API:
  - `explicit RenderWidget(QWidget* parent = nullptr)`
  - `QPaintEngine* paintEngine() const override`
  - `void setPresenter(D3D12Presenter* presenter)`
  - `bool isPresenterReady() const`

### `include/openzoom/ui/joystick_overlay.hpp`
`openzoom::JoystickOverlay`
- Circular on-canvas joystick overlay.
- Anchors to the render viewport's bottom-right corner, but detects the Simple
  action cluster's native window rectangle and moves one margin above it.
  Placement refreshes when either the viewport or action cluster changes
  geometry or visibility, so the joystick remains reachable in both UI modes.
- Public API:
  - `explicit JoystickOverlay(QWidget* parent = nullptr)`
  - `void ResetKnob()`
- Signal:
  - `JoystickChanged(float normX, float normY)`

### `include/openzoom/ui/main_window.hpp`
`openzoom::MainWindow`
- Builds the UI shell and exposes widget accessors used by `OpenZoomApp`.
- Installs both Qt and Win32-native event filters for reliable activity
  detection across the native swap-chain surface and its owned control windows.
- Two-speed UI around one persistent render widget: Simple keeps the render
  widget full-size and places three solid, high-contrast primary control
  windows plus a contextual keystone strip flush to its edges; Advanced keeps
  the bottom-left quick-mode carousel available and opens a 420-580 pixel
  tabbed inspector to the right of the camera. `Image`
  contains scrollable device/tuning controls and pipeline status; `Assistant`
  contains Chat and History views. Wrapping section arrows remain in the tab
  header alongside a compact Help button whose dialog presents Controls before
  Features; a full-width AI Settings pop-out row appears below it. The top-left
  mode switch is restored on application activation in both UI modes. The
  collapsed tuning panel keeps its remaining controls packed at the top.
  - `void setSimpleMode(bool simple)` / `bool isSimpleMode() const`
  - `int advancedPanelWidth() const` / `void setAdvancedPanelWidth(int width)`
    expose the persisted splitter width while preserving minimum camera and
    inspector widths
  - `QAbstractButton* simpleModeButton() const` / `QAbstractButton* advancedModeButton() const` — checkable and mutually exclusive; state switching is wired internally, while the app connects only for persistence
- Public API includes getters for:
  - camera selection and mode list
  - quick-mode preset list, preset description label, quick-option promotion,
    and Reset Tuning; reset emits `resetCurrentProfileRequested()` so the app
    can restore profile-owned defaults without altering global controls
  - BW, zoom, blur, temporal smoothing, and spatial sharpening controls
  - stabilization checkbox and strength slider (`stabilizationCheckbox()`, `stabilizationStrengthSlider()`, 0–98)
  - screen-fix controls: `keystoneCheckbox()` ("Straighten Screen (Keystone)"), `autoContrastCheckbox()` ("Auto Contrast"), and `autoContrastStrengthSlider()` (0–100, default 70, enabled with its checkbox); `setKeystoneTrackingControls(...)` updates the shared Simple/Advanced Previous, Stop/Continue, and Next controls and their accessible state
  - Simple and Advanced Text Clarity master controls plus component checkboxes,
    sliders, polarity selector, and profile-owned NVIDIA Super Resolution
    enable/strength, Ultra full-frame cache, and Faster 2x choices; Ultra and
    Faster 2x are mutually exclusive and all SuperRes controls stay disabled
    when its build option is off
  - display color picker (`displayColorPicker()`) backed by structured
    `ColorScheme` values and accessible swatches, contrast slider (25-400 =
    contrast x100), brightness slider (-100..100)
  - OCR/VLM scaffolding checkboxes plus assistive overlay toggle
  - on-demand analysis buttons (`explainNowButton()`, `readTextButton()`) and
    assistive buttons (`aiSettingsButton()`, `openNotesButton()`,
    `setupAssistantButton()`)
  - Assistant status, sign-in, transcript, prompt, camera attachment, send/stop/new, and history resume/rename/export/delete widgets
  - focus sliders, rotation combo, debug toggle, focus marker, and the global
    joystick toggle near the top of Advanced Image
  - capture and recording buttons
  - processing status label
  - the static `SuperRes powered by NVIDIA Maxine™` attribution at the bottom
    of the Advanced inspector; `setMaxineRuntimeInstalled(...)` enables or
    disables the compiled control from actual runtime detection and keeps its
    tooltip and accessible description current after Setup changes;
    `isMaxineRuntimeInstalled()` exposes the cached state to dependent controls
  - `setSuperResStatus(...)` updates the dedicated status row with runtime or
    fallback state plus compact wrapping source-crop, viewport-target,
    final-zoom, and measured-latency details; a latency-only failure exposes a
    compact `Ignore 24 ms performance limit` checkbox
- Signals `keystoneStepBackRequested()`, `keystonePauseResumeRequested()`,
  `keystoneStepForwardRequested()`, `resetCurrentProfileRequested()`, and
  `superResPerformanceOverrideChanged(bool)` bridge UI commands to
  `OpenZoomApp`.
- Event handling:
  - arrow-key routing for panning
  - global Qt activity detection plus native render-window mouse/focus
    detection that reveals chrome, restarts the five-second idle timer, and
    preserves focused controls; while chrome is already visible this is a
    deadline-only fast path and never recomputes or raises tool-window geometry
  - number keys `1`-`9` for the first nine quick modes (grid tiles show matching number badges)
  - `Esc` closes the quick-mode grid and `Ctrl+H` toggles pinned chrome
  - explicit `Tab` / `Shift+Tab` traversal across the separate corner windows
  - previous/next wrapping profile activation and a temporary numbered tile grid
  - large centered mode toast plus assertive `QAccessibleAnnouncementEvent`
  - event filter on the render widget for Ctrl+wheel zoom, plain-wheel pan,
    middle-button drag pan, and corner-window repositioning on resize/move

### `include/openzoom/ui/color_scheme_picker.hpp`
`openzoom::ColorSchemePicker`
- Replaces the old full-width combo list with a compact trigger and owned tool
  popover containing six-column reading-color/effect grids and a Custom editor.
- The frameless native popover uses an opaque backing store and solid dark
  palette rather than translucent composition over the D3D/inspector surface.
- Custom schemes support 2-8 stops, duotone/posterize/gradient modes, stepped
  output, per-stop `QColorDialog` wells, a live tile preview, and a persistent
  pencil-badged reusable tile.
- Selection is explicit: hover only updates normal button feedback and never
  changes the camera. `schemeChanged()` is emitted after a chosen built-in or
  custom scheme is applied.
- Every tile/well is a real focusable control with accessible names/tooltips.
  Arrow/Home/End navigate the grid; Esc closes and restores trigger focus;
  selection emits an assertive accessibility announcement.
- Public API: `currentScheme()`, `customScheme()`, `setCurrentScheme(...)`,
  `setCustomScheme(...)`, and `hasCustomScheme()`.

### `include/openzoom/ui/wheel_safe_combo_box.hpp`
`openzoom::WheelSafeComboBox` / `openzoom::WheelSafeSlider`
- Ignore wheel edits so events continue to the surrounding scroll area or
  camera navigation. Click, drag (sliders), and keyboard editing remain
  unchanged. All settings combos and sliders use these subclasses.

### `include/openzoom/ui/ai_settings_dialog.hpp`
`openzoom::AiSettingsDialog`
- Modal editor for AI provider selection and assistive configuration. Supports
  Codex subscription mode with executable/model overrides, explicit internet
  and workspace-scoped coding permissions, and an
  OpenAI-compatible server mode, including local servers without an API key.
  Its fixed action row surrounds a vertically scrollable content area with
  separate Codex, OpenAI-compatible VLM, OCR, Read Aloud, and notes sections.
  It shows the built-in Codex prompt read-only and exposes separate user
  instructions for response language, tone, and detail. `SetCodexModelCatalog`
  fills the model selector and model-specific reasoning selector from the live
  app-server catalog while keeping an unavailable saved selection visible.
  It also enumerates all voices and locales exposed by Qt's selected Windows
  speech backend, persists voice/locale/rate selection, and previews speech
  only on request. It falls back from WinRT to SAPI only after an actual engine
  error, not while asynchronous WinRT initialization is still in progress.
- Public API:
  - `explicit AiSettingsDialog(const settings::AssistiveSettings& initial, QWidget* parent = nullptr)`
  - `SetCodexModelCatalog(const QJsonArray& models, const QString& selectedModel)`
  - `settings::AssistiveSettings result() const` — the edited settings after the dialog is accepted

### `include/openzoom/ui/assistive_overlay.hpp`
`openzoom::AssistiveOverlay`
- Solid owned floating Assistant tool window drawn on top of the render
  surface. Its header and edges use the native window-system move/resize path;
  streamed result updates never reapply window geometry. A
  read-only `QTextBrowser` exposes incrementally streamed text to screen
  readers and keyboard selection; the question field stays editable while a
  response streams, while `SetBusy(true)` blocks Ask and Enter submission
  without discarding the draft. When ready, it sends the current view to the
  shared persistent Assistant conversation. Read Aloud remains manual
  and strips the visible `OCR` / `Scene Explain` section labels from its speech
  payload, while the Close control uses a high-contrast white icon and border.
- Public API:
  - `explicit AssistiveOverlay(QWidget* parent = nullptr)`
  - `void SetContent(const QString& title, const QString& body, bool visible)`
  - `void SetBusy(bool busy)`
  - `void RestoreRelativeGeometry(const QRect& geometry)`
  - `QRect RelativeGeometry() const`
  - `std::array<QWidget*, 5> FocusTargets() const` — returns result text, question field, Ask, Read Aloud, and Close for the Simple-mode focus loop
- Signals:
  - `Dismissed()`
  - `ReadAloudRequested(const QString& text)`
  - `QuestionSubmitted(const QString& question)`

### `include/openzoom/ui/responsive_slider_row.hpp`
`openzoom::ResponsiveSliderRow`
- Reusable settings row that moves its slider to a full-width second line when
  the inspector is too narrow. This keeps labels wrapped and every point of
  the slider track reachable during live splitter resizing.

## Entry Point

### `src/app/main.cpp`
- On Windows, constructs `OpenZoomApp`, calls fallible `Initialize()`, then
  enters `Run()` inside a `try`/`catch`.
- On non-Windows platforms, exits with an unsupported-platform message.
