# OpenZoom Code Reference

Authoritative code map for the current repository state as of 2026-07-22. Update this file whenever classes, public structs, or significant functions change.

## App Module

### `include/openzoom/app/app.hpp`
`openzoom::OpenZoomApp`
- Owns the Qt application lifecycle, UI wiring, camera control, CPU/GPU processing selection, output capture, and settings persistence.
- Public API:
  - `OpenZoomApp(int& argc, char** argv)`
  - `~OpenZoomApp()`
  - `int Run()`
- Core slots:
  - `OnFrameTick()`
  - `OnPresetSelectionChanged(QListWidgetItem* current, QListWidgetItem* previous)`
  - `OnCameraSelectionChanged(int index)`
  - `OnBlackWhiteToggled(bool checked)`
  - `OnBlackWhiteThresholdChanged(int value)`
  - `OnZoomToggled(bool checked)`
  - `OnZoomAmountChanged(int value)`
  - `OnDebugViewToggled(bool checked)`
  - `OnZoomCenterXChanged(int value)`
  - `OnZoomCenterYChanged(int value)`
  - `OnRotationSelectionChanged(int index)`
  - `OnControlsCollapsedToggled(bool checked)`
  - `OnVirtualJoystickToggled(bool checked)`
  - `OnBlurToggled(bool checked)`
  - `OnBlurSigmaChanged(int value)`
  - `OnBlurRadiusChanged(int value)`
  - `OnFocusMarkerToggled(bool checked)`
  - `OnSpatialSharpenToggled(bool checked)`
  - `OnSpatialUpscalerChanged(int index)`
  - `OnSpatialSharpnessChanged(int value)`
  - `OnTemporalSmoothToggled(bool checked)`
  - `OnTemporalSmoothStrengthChanged(int value)`
  - `OnOcrAssistToggled(bool checked)`
  - `OnVlmAssistToggled(bool checked)`
  - `OnAssistiveOverlayToggled(bool checked)`
  - `OnAssistiveOverlayUpdated(const QString& title, const QString& body, bool visible)`
  - `OnStabilizationToggled(bool checked)`
  - `OnStabilizationStrengthChanged(int value)`
  - `OnKeystoneToggled(bool checked)`
  - `OnAutoContrastToggled(bool checked)`
  - `OnAutoContrastStrengthChanged(int value)`
  - `OnDisplayColorModeChanged(int index)`
  - `OnContrastChanged(int value)`
  - `OnBrightnessChanged(int value)`
- Important helpers:
  - preset/config workflow: `CaptureCurrentAdvancedConfig()`, `ApplyAdvancedConfig(...)`, `PopulatePresetList()`, `RefreshPresetSelection(bool preserveCurrentSelection = false)`, `UpdatePresetDescription()`, `SyncCurrentConfigToPersistence(bool preservePresetSelection = false)`, `PromoteCurrentConfigToPreset()`
  - assistive workflow: `UpdateAssistiveRuntimeState()`, `MaybeRequestAssistiveAnalysis(...)`, `BuildAssistiveRuntimeConfig()`, `ApplyAssistiveSettingsToRuntime()`, `SubmitOnDemandAnalysis(bool runOcr, bool runVlm)`, `SubmitAssistantPrompt()`, `PopulateAssistantHistory()`, `LoadSelectedAssistantConversation()`, `SetAssistantBusy(bool)`, `AppendAssistantMessage(...)`, `OpenAiSettingsDialog()`, `OpenNotesFile()`
  - camera lifecycle: `EnumerateCameras()`, `PopulateCameraCombo()`, `RefreshCameraModesList(size_t)`, `StartCameraCapture(size_t, bool interactive = true)` (returns success; `interactive = false` suppresses the failure dialog for reconnect attempts), `StopCameraCapture()`, `HandleCameraStartFailure(...)`, `HandleCameraRuntimeFailure(...)`
  - camera reconnect: `BeginCameraReconnect()` / `DriveCameraReconnect()` — a frame-tick-driven state machine entered on `MediaCapture::ConsumeDeviceLost()`; re-enumerates, matches `LastSymbolicLink()`, retries with 2s/4s/8s backoff (QDateTime-based, no sleeps) and gives up after ~30 seconds; no modal dialogs while it runs
  - processing/presentation: `BuildCompositeAndPresent(UINT, UINT)`, `PresentFitted(...)`, `EnsureCudaSurface(UINT, UINT)`, `ProcessFrameWithCuda(UINT, UINT)` (CPU-converted BGRA input), `TryProcessRawFrameWithCuda(const MediaFrame&)` (NV12/YUY2 raw planes straight to CUDA at the post-rotation extent; falls back to the CPU path when it returns false), `RunCudaPipeline(const ProcessingInput&, UINT presentWidth, UINT presentHeight)` (shared settings/fence/present tail)
  - async readback: `HandleGpuFramePresented(UINT, UINT)` — single drain/request point for the presenter's async readback ring; routes completed frames to the recorder and the periodic assistive grab, and records `lastReadbackSignalValue_` so the next CUDA pass waits GPU-side for in-flight copies; `AssistiveAnalysisDue()` gates the assistive request cadence
  - input/focus: `SetZoomCenter(float, float, bool, bool preservePresetSelection = false)`, `ApplyInputForces()`, `MapViewToSource(...)`, `RotateNormalizedPoint(...)`; wheel, keyboard, joystick, and drag navigation preserve the selected quick-mode identity while true Advanced edits still become custom
  - output: `CaptureSnapshot(...)`, `MaybeRecordFrame(const uint8_t*, UINT, UINT)` (lazy recorder start at the delivered frame's dimensions; disk-full stops surface as info, not errors), `StopRecordingUi()`, `EnsureOutputSubdir(const QString&)`
  - persistence/status: `ApplyPersistentSettings(...)`, `SavePersistentSettings()`, `UpdateProcessingStatusLabel()`, `ShowStatusMessage(const QString&, int durationMs = 10000)` — transient non-modal notification shown in the pipeline status label
- Key state:
  - UI widget pointers from `MainWindow`
  - preset/config persistence via `settings::PersistentSettings persistentSettings_`
  - `MediaCapture mediaCapture_`
  - `cameraSessionId_` invalidates queued failure callbacks from older capture sessions
  - `processing::CpuFramePipeline cpuPipeline_`
  - `VideoRecorder videoRecorder_`
  - `std::unique_ptr<D3D12Presenter> presenter_`
  - `std::unique_ptr<CudaInteropSurface> cudaSurface_`
  - profile-owned tuning values for BW, zoom, blur, temporal smoothing, stabilization, display color mode with contrast/brightness, sharpening, focus marker, and debug view
  - global device/UI values for camera, orientation, joystick, Simple/Advanced mode, and recording

### `include/openzoom/app/interaction_controller.hpp`
`openzoom::InteractionController`
- Converts keyboard, wheel, mouse-drag, and virtual joystick input into zoom-center updates.
- Public API:
  - `explicit InteractionController(OpenZoomApp& app)`
  - `bool HandlePanKey(int key, bool pressed)`
  - `bool HandlePanScroll(const QWheelEvent* wheelEvent)`
  - `void HandleZoomWheel(int delta, const QPointF& localPos)`
  - `void ApplyInputForces()`
  - `void BeginMousePan(const QPointF& pos, const QSize& widgetSize)`
  - `bool UpdateMousePan(const QPointF& pos)`
  - `void EndMousePan()`
  - `bool IsMousePanActive() const`
  - `void ResetJoystick()`
  - `void SetJoystickAxes(float x, float y)`

### `include/openzoom/app/settings_store.hpp`
Namespace `openzoom::settings`
- `struct AdvancedConfig`
  - full profile-owned tuning payload including image-processing and focus state, OCR/VLM mode flags, `stabilizationEnabled`, `stabilizationStrength` (0..1), `displayColorMode` (0..4), `contrast`, `brightness`, `keystoneEnabled`, `autoContrastEnabled`, and `autoContrastStrength` (0..1)
  - `rotationQuarterTurns` is read from old profile JSON only for backward migration, is no longer written into profiles, and is ignored by current profile comparisons
- `struct AssistiveSettings`
  - AI/assistive configuration: `aiProvider`, `codexExecutablePath`, `codexModel`, `codexReasoningEffort`, `codexInternetEnabled`, `codexCodingEnabled`, `codexWorkspaceDirectory`, `assistantInstructions`, `vlmApiUrl`, `vlmApiKey`, `vlmModel`, `vlmPrompt`, `tesseractPath`, `ocrLanguage`, `ttsEngine`, `ttsVoiceName`, `ttsVoiceLocale`, `ttsRate`, and `lectureNotesEnabled`; non-empty values take precedence over matching environment fallbacks
- `struct CodexConversation`
  - OpenZoom-owned thread index entry: `threadId`, `title`, `preview`, `createdAt`, and `updatedAt`; transcripts remain in the Codex thread store
- `struct PresetDefinition`
  - stage-1 quick-mode metadata: preset id, name, description, target config id, built-in flag
- `struct PersistentSettings`
  - persists global `cameraIndex` and `rotationQuarterTurns`, UI state, `simpleUiMode`, selected preset id, current live advanced config, the `assistive` settings block, OpenZoom-created `codexConversations`, and user-created configs/presets
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
- `using FrameCallback = std::function<void(const MediaFrame& frame)>`
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
  - `RefreshAccount()` / `StartChatGptLogin()`
  - `RequestVisionTurn(...)` / `InterruptTurn()`
  - `LoadConversation(...)`, `RenameConversation(...)`, `DeleteConversation(...)`
- Signals cover server/account/model/rate-limit state, login URL, conversation
  lifecycle/transcript loading, and streamed turn start/delta/completion.

### `include/openzoom/common/assistive_runtime.hpp`
Supporting types:
- `struct AssistiveRuntimeConfig`
  - same provider/Codex/VLM/OCR/speech fields as `settings::AssistiveSettings`, including Codex internet/coding/workspace permissions, plus `QString notesDirectory` (absolute directory for lecture notes; empty disables notes)

`openzoom::AssistiveRuntime`
- Asynchronous assistive-analysis runtime owned by `OpenZoomApp`.
- OCR path:
  - exports the current frame to a temporary PNG
  - resolves configured/environment, bundled, PATH, and standard Windows
    Tesseract locations, including sibling `tessdata`
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
  - `void NoteCapturedPhoto(const QString& filePath)` — appends a photo reference to the lecture notes
  - `QString notesFilePath() const` — absolute path of the current notes file (empty until something is written)
- Manual-only text-to-speech of results via Qt TextToSpeech when built with `OPENZOOM_HAS_TTS=1`; the runtime prefers the `winrt` engine and falls back to Qt's default engine.
- Lecture notes: a per-session markdown file in the configured notes directory collecting timestamped OCR text, scene explanations, and photo references.
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
- Media Foundation sink-writer wrapper for processed H.264 output. The
  container is fragmented MP4 (fMP4): fragments flush to disk while recording,
  so the file stays playable up to the last completed fragment even if the
  process dies before finalization. Files keep the `.mp4` extension.
- `enum class StopReason` — `None`, `Manual`, `DiskFull`, `WriteFailed`; why
  the recorder last transitioned from recording to stopped.
- Public API:
  - `VideoRecorder()`
  - `~VideoRecorder()`
  - `bool Start(const std::wstring& filePath, UINT width, UINT height, UINT fps)` — refuses to start with under 500 MB free on the target volume (`LastError()` explains why)
  - `void Stop()`
  - `bool IsRecording() const`
  - `bool AddFrame(const uint8_t* bgraData, size_t strideBytes)` — free space is re-checked every ~5 seconds; below 200 MB the recording is finalized cleanly and `AddFrame` returns false with `StopReason::DiskFull` (the file is already intact on disk)
  - `double DurationSeconds() const`
  - `const std::string& LastError() const`
  - `StopReason LastStopReason() const`
- Internal helpers:
  - `InitializeSink(...)`
  - `FinalizeAndStop(StopReason reason)`
  - `SetError(const std::string& err)`

## D3D12 Module

### `include/openzoom/d3d12/presenter.hpp`
`openzoom::D3D12Presenter`
- Manages the D3D12 device, swap chain, per-frame upload buffers, shared fence, and readback buffer.
- Presentation is pipelined: each of the two frame slots has its own command
  allocator, upload buffer, and fence value, and `Present`/`PresentFromTexture`
  only wait until the GPU has finished the frame that previously used the
  current slot instead of draining the queue. `ReadbackTexture` remains fully
  blocking and uses a dedicated command allocator.
- Public API:
  - `D3D12Presenter()`
  - `~D3D12Presenter()`
  - `void Initialize(HWND hwnd, UINT width, UINT height)`
  - `bool IsInitialized() const`
  - `void Resize(UINT width, UINT height)`
  - `void Present(const uint8_t* data, UINT width, UINT height)`
  - `void PresentFromTexture(ID3D12Resource* texture, UINT width, UINT height, const FenceSyncParams* fenceSync = nullptr)`
  - `bool ReadbackTexture(ID3D12Resource* texture, UINT width, UINT height, std::vector<uint8_t>& outBgra)`
  - `bool RequestReadback(ID3D12Resource* texture, UINT width, UINT height)` — enqueues an async copy into a two-slot ring; returns false when both slots are in flight (caller skips this frame's readback); never blocks the CPU
  - `bool TryGetCompletedReadback(std::vector<uint8_t>& outBgra, UINT& outWidth, UINT& outHeight)` — moves the OLDEST completed request's pixels out (tightly packed BGRA8), one result per call; poll every tick to drain. Pending requests are silently dropped by `Resize`
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
- `enum class SpatialUpscaler`
  - `kFsrEasuRcas`
  - `kNis`
- `enum class CudaBufferFormat`
  - `kRgba8`
  - `kRgba16F`
- `struct ProcessingSettings`
  - toggles and parameters for BW, zoom, blur, focus marker, spatial sharpening, temporal smoothing, and staging format
  - stabilization: `enableStabilization`, `stabilizationStrength` (0..1, higher = stronger path smoothing)
  - display grading: `displayColorMode` (0=Normal, 1=Inverted, 2=WhiteOnBlack, 3=YellowOnBlack, 4=BlackOnYellow), `contrast` (0.25..4.0), `brightness` (-1..1)
  - screen fix: `enableKeystone` (auto-detect the projected slide quad and warp it fronto-parallel), `enableAutoContrast` (percentile level stretch before contrast/brightness), `autoContrastStrength` (0..1 blend toward the full stretch)
- `struct ProcessingInput`
  - `hostPixels`, `hostStrideBytes`, `pixelSizeBytes`, `width`, `height` — `width`/`height` always describe the host pixel layout (pre-rotation)
  - `inputFormat` — 0=BGRA8 (existing CPU-converted path), 1=NV12 (`hostPixels` = Y plane, `hostPlane2` = interleaved UV plane with `hostPlane2StrideBytes`), 2=YUY2 (packed in `hostPixels`)
  - `rotationQuarterTurns` — 0..3 clockwise, applied on the GPU after conversion for raw formats only (ignored for BGRA with a one-shot warning). For odd turns the interop surface must be created at the post-rotation extent (height x width); `ProcessFrame` validates and returns false on mismatch

`openzoom::CudaInteropSurface`
- Imports a D3D12 texture into CUDA and runs the GPU effect chain.
- All teardown paths (destructor, device-buffer and temporal-history release)
  synchronize the CUDA stream first, since `ProcessFrame` returns with kernels
  still queued when fence sync is active.
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
- Stage order: upload/convert/rotate → stabilization → keystone → BW → sharpen → zoom → blur → temporal → auto-contrast histogram+analysis → color grade → focus marker → interop copy.

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
- `LaunchStabilizationProjections(...)`
- `LaunchStabilizationEstimate(...)`
- `LaunchStabilizationWarp(...)`
- `LaunchDisplayColorGradeLinear(...)` — takes optional device `autoContrastLevels` (float2 lo/hi) plus `autoContrastStrength` for the auto-contrast remap
- `LaunchNv12ToBgraLinear(...)` / `LaunchYuy2ToBgraLinear(...)` — BT.601 limited-range YUV to BGRA, integer math identical to the CPU converters
- `LaunchRotateQuarterLinear(...)` — rotate by 1/2/3 clockwise quarter turns (destination is srcHeight x srcWidth for odd turns)
- `LaunchKeystoneWarp(...)` — bilinear homography warp (output pixel to source pixel, row-major 3x3), black outside the source rect
- `LaunchAutoContrastHistogram(...)` / `LaunchAutoContrastAnalysis(...)` — 256-bin luma histogram plus 2nd/98th-percentile level derivation low-passed into device-resident levels
- `bool UploadGaussianKernel(int radius, float sigma, cudaStream_t stream)`

## UI Module

### `include/openzoom/ui/main_window.hpp`
`openzoom::RenderWidget`
- Native widget that hosts the D3D12 presenter.
- Public API:
  - `explicit RenderWidget(QWidget* parent = nullptr)`
  - `QPaintEngine* paintEngine() const override`
  - `void setPresenter(D3D12Presenter* presenter)`
  - `bool isPresenterReady() const`

`openzoom::JoystickOverlay`
- Circular on-canvas joystick overlay.
- Public API:
  - `explicit JoystickOverlay(QWidget* parent = nullptr)`
  - `void ResetKnob()`
- Signal:
  - `JoystickChanged(float normX, float normY)`

`openzoom::MainWindow`
- Builds the UI shell and exposes widget accessors used by `OpenZoomApp`.
- Installs both Qt and Win32-native event filters for reliable activity
  detection across the native swap-chain surface and its owned control windows.
- Two-speed UI around one persistent render widget: Simple keeps the render
  widget full-size and places three solid, high-contrast frameless control
  windows flush to its corners; Advanced hides the lower Simple controls and
  opens a 420-580 pixel tabbed inspector to the right of the camera. `Image`
  contains scrollable device/tuning controls and pipeline status; `Assistant`
  contains Chat and History views. Wrapping section arrows and the top-level
  AI Settings pop-out control live in the tab header.
  - `void setSimpleMode(bool simple)` / `bool isSimpleMode() const`
  - `QAbstractButton* simpleModeButton() const` / `QAbstractButton* advancedModeButton() const` — checkable and mutually exclusive; state switching is wired internally, while the app connects only for persistence
- Public API includes getters for:
  - camera selection and mode list
  - quick-mode preset list, preset description label, and quick-option promotion button
  - BW, zoom, blur, temporal smoothing, and spatial sharpening controls
  - stabilization checkbox and strength slider (`stabilizationCheckbox()`, `stabilizationStrengthSlider()`, 0–98)
  - screen-fix controls: `keystoneCheckbox()` ("Straighten Screen (Keystone)"), `autoContrastCheckbox()` ("Auto Contrast"), and `autoContrastStrengthSlider()` (0–100, default 70, enabled with its checkbox)
  - display color combo (`displayColorCombo()`, indices match `AdvancedConfig::displayColorMode`), contrast slider (25–400 = contrast x100), brightness slider (-100..100)
  - OCR/VLM scaffolding checkboxes plus assistive overlay toggle
  - on-demand analysis buttons (`explainNowButton()`, `readTextButton()`) and assistive buttons (`aiSettingsButton()`, `openNotesButton()`)
  - Assistant status, sign-in, transcript, prompt, camera attachment, send/stop/new, and history resume/rename/export/delete widgets
  - focus sliders, rotation combo, debug toggle, focus marker, joystick toggle
  - capture and recording buttons
  - processing status label
- Event handling:
  - arrow-key routing for panning
  - global Qt activity detection plus native render-window mouse/focus
    detection that reveals chrome, restarts the five-second idle timer, and
    preserves focused controls
  - number keys `1`-`9` for the first nine quick modes (grid tiles show matching number badges)
  - `Esc` closes the quick-mode grid and `Ctrl+H` toggles pinned chrome
  - explicit `Tab` / `Shift+Tab` traversal across the separate corner windows
  - previous/next wrapping profile activation and a temporary numbered tile grid
  - large centered mode toast plus assertive `QAccessibleAnnouncementEvent`
  - event filter on the render widget for Ctrl+wheel zoom, plain-wheel pan,
    middle-button drag pan, and corner-window repositioning on resize/move

### `include/openzoom/ui/ai_settings_dialog.hpp`
`openzoom::AiSettingsDialog`
- Modal editor for AI provider selection and assistive configuration. Supports
  Codex subscription mode with executable/model overrides, explicit internet
  and workspace-scoped coding permissions, and an
  OpenAI-compatible server mode, including local servers without an API key.
  It exposes Codex reasoning effort plus shared Assistant Instructions for
  response language, tone, and detail while keeping scene-specific prompting
  separate.
  It also enumerates all voices and locales exposed by Qt's selected Windows
  speech backend, persists voice/locale/rate selection, and previews speech
  only on request. It falls back from WinRT to SAPI only after an actual engine
  error, not while asynchronous WinRT initialization is still in progress.
- Public API:
  - `explicit AiSettingsDialog(const settings::AssistiveSettings& initial, QWidget* parent = nullptr)`
  - `settings::AssistiveSettings result() const` — the edited settings after the dialog is accepted

`openzoom::AssistiveOverlay`
- Solid owned floating Assistant tool window drawn on top of the render
  surface. Its header and edges use the native window-system move/resize path;
  streamed result updates never reapply window geometry. A
  read-only `QTextBrowser` exposes incrementally streamed text to screen
  readers and keyboard selection; the question field sends the current view
  to the shared persistent Assistant conversation. Read Aloud remains manual
  and strips the visible `OCR` / `Scene Explain` section labels from its speech
  payload, while the Close control uses a high-contrast white icon and border.
- Public API:
  - `explicit AssistiveOverlay(QWidget* parent = nullptr)`
  - `void SetContent(const QString& title, const QString& body, bool visible)`
  - `void SetBusy(bool busy)`
  - `std::array<QWidget*, 5> FocusTargets() const` — returns result text, question field, Ask, Read Aloud, and Close for the Simple-mode focus loop
- Signals:
  - `Dismissed()`
  - `ReadAloudRequested(const QString& text)`
  - `QuestionSubmitted(const QString& question)`

## Entry Point

### `src/app/main.cpp`
- On Windows, constructs `OpenZoomApp` and runs it inside a `try`/`catch`.
- On non-Windows platforms, exits with an unsupported-platform message.
