# OpenZoom Code Reference

Authoritative code map for the current repository state as of 2026-06-02. Update this file whenever classes, public structs, or significant functions change.

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
- Important helpers:
  - preset/config workflow: `CaptureCurrentAdvancedConfig()`, `ApplyAdvancedConfig(...)`, `PopulatePresetList()`, `RefreshPresetSelection()`, `UpdatePresetDescription()`, `SyncCurrentConfigToPersistence()`, `PromoteCurrentConfigToPreset()`
  - assistive workflow: `UpdateAssistiveRuntimeState()`, `MaybeRequestAssistiveAnalysis(...)`
  - camera lifecycle: `EnumerateCameras()`, `PopulateCameraCombo()`, `RefreshCameraModesList(size_t)`, `StartCameraCapture(size_t)`, `StopCameraCapture()`
  - processing/presentation: `BuildCompositeAndPresent(UINT, UINT)`, `PresentFitted(...)`, `EnsureCudaSurface(UINT, UINT)`, `ProcessFrameWithCuda(UINT, UINT)`
  - input/focus: `SetZoomCenter(float, float, bool)`, `ApplyInputForces()`, `MapViewToSource(...)`, `RotateNormalizedPoint(...)`
  - output: `CaptureSnapshot(...)`, `MaybeRecordFrame(...)`, `EnsureOutputSubdir(const QString&)`
  - persistence/status: `ApplyPersistentSettings(...)`, `SavePersistentSettings()`, `UpdateProcessingStatusLabel()`
- Key state:
  - UI widget pointers from `MainWindow`
  - preset/config persistence via `settings::PersistentSettings persistentSettings_`
  - `MediaCapture mediaCapture_`
  - `processing::CpuFramePipeline cpuPipeline_`
  - `VideoRecorder videoRecorder_`
  - `std::unique_ptr<D3D12Presenter> presenter_`
  - `std::unique_ptr<CudaInteropSurface> cudaSurface_`
  - per-feature booleans and tuning values for BW, zoom, blur, temporal smoothing, sharpening, focus marker, joystick, debug view, rotation, and recording

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
  - full stage-2 tuning payload including image-processing flags, focus/rotation state, and OCR/VLM scaffolding flags
- `struct PresetDefinition`
  - stage-1 quick-mode metadata: preset id, name, description, target config id, built-in flag
- `struct PersistentSettings`
  - persists camera/UI state, selected preset id, current live advanced config, and user-created configs/presets
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
  - `bool AreConfigsEquivalent(...)`

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
- `struct CameraDescriptor`
  - `name`, `symbolicLink`, `activation`
- `struct VideoFormat`
  - `width`, `height`, `numerator`, `denominator`

`openzoom::MediaCapture`
- Media Foundation camera enumeration and threaded source-reader capture.
- Public API:
  - `MediaCapture()`
  - `~MediaCapture()`
  - `bool Initialize()`
  - `void Shutdown()`
  - `std::vector<CameraDescriptor> EnumerateCameras()`
  - `std::vector<VideoFormat> EnumerateFormats(const CameraDescriptor& descriptor)`
  - `bool StartCapture(const CameraDescriptor& descriptor, FrameCallback callback, GUID preferredSubtype = MFVideoFormat_ARGB32)`
  - `void StopCapture()`
  - `const std::string& LastError() const`
- Internal helpers:
  - `ConfigureReader(...)`
  - `CaptureLoop(FrameCallback callback)`
  - `ExtractFormats(IMFSourceReader* reader)`
  - `HrToString(HRESULT hr)`

## Common Module

### `include/openzoom/common/assistive_runtime.hpp`
`openzoom::AssistiveRuntime`
- Asynchronous assistive-analysis runtime owned by `OpenZoomApp`.
- OCR path:
  - exports the current frame to a temporary PNG
  - runs `tesseract.exe` asynchronously
  - returns extracted text to the overlay
- VLM path:
  - JPEG-encodes the current frame
  - posts an OpenAI-compatible `chat/completions` request to a configured endpoint
  - returns the text response to the overlay
- Public API:
  - `AssistiveRuntime(QObject* parent = nullptr)`
  - `~AssistiveRuntime()`
  - `void SetModes(bool ocrEnabled, bool vlmEnabled)`
  - `bool WantsAnalysis() const`
  - `bool IsBusy() const`
  - `void SubmitFrame(const uint8_t* bgraData, int width, int height)`
- Signal:
  - `OverlayUpdated(const QString& title, const QString& body, bool visible)`

### `include/openzoom/common/image_processing.hpp`
Namespace `openzoom::processing`
- Format conversion helpers:
  - `CopyArgbToBgra(...)`
  - `CopyRgbxToBgra(...)`
  - `ConvertNv12ToBgra(...)`
  - `ConvertYuy2ToBgra(...)`
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
- Thin Media Foundation sink-writer wrapper for processed MP4 output.
- Public API:
  - `VideoRecorder()`
  - `~VideoRecorder()`
  - `bool Start(const std::wstring& filePath, UINT width, UINT height, UINT fps)`
  - `void Stop()`
  - `bool IsRecording() const`
  - `bool AddFrame(const uint8_t* bgraData, size_t strideBytes)`
  - `double DurationSeconds() const`
  - `const std::string& LastError() const`
- Internal helpers:
  - `InitializeSink(...)`
  - `SetError(const std::string& err)`

## D3D12 Module

### `include/openzoom/d3d12/presenter.hpp`
`openzoom::D3D12Presenter`
- Manages the D3D12 device, swap chain, upload buffer, shared fence, and readback buffer.
- Public API:
  - `D3D12Presenter()`
  - `~D3D12Presenter()`
  - `void Initialize(HWND hwnd, UINT width, UINT height)`
  - `bool IsInitialized() const`
  - `void Resize(UINT width, UINT height)`
  - `void Present(const uint8_t* data, UINT width, UINT height)`
  - `void PresentFromTexture(ID3D12Resource* texture, UINT width, UINT height, const FenceSyncParams* fenceSync = nullptr)`
  - `bool ReadbackTexture(ID3D12Resource* texture, UINT width, UINT height, std::vector<uint8_t>& outBgra)`
  - `ID3D12Device* GetDevice() const`
  - `ID3D12Fence* GetFence() const`
  - `UINT64 GetLastSignaledFenceValue() const`

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
- `struct ProcessingInput`
  - `hostPixels`, `hostStrideBytes`, `pixelSizeBytes`, `width`, `height`

`openzoom::CudaInteropSurface`
- Imports a D3D12 texture into CUDA and runs the GPU effect chain.
- Public API:
  - `explicit CudaInteropSurface(ID3D12Resource* texture, ID3D12Fence* sharedFence = nullptr)`
  - `~CudaInteropSurface()`
  - `bool IsValid() const`
  - `bool HasExternalSemaphore() const`
  - `void RunGradientDemoKernel(unsigned int width, unsigned int height, float timeSeconds)`
  - `bool ProcessFrame(const ProcessingInput& input, const ProcessingSettings& settings, const FenceSyncParams& fenceSync)`
  - `const std::string& LastError() const`
  - `void ResetTemporalHistory()`

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
- Public API includes getters for:
  - camera selection and mode list
  - quick-mode preset list, preset description label, and quick-option promotion button
  - BW, zoom, blur, temporal smoothing, and spatial sharpening controls
  - OCR/VLM scaffolding checkboxes plus assistive overlay toggle
  - focus sliders, rotation combo, debug toggle, focus marker, joystick toggle
  - capture and recording buttons
  - processing status label
- Event handling:
  - arrow-key routing for panning
  - event filter on the render widget for Ctrl+wheel zoom, plain-wheel pan, and middle-button drag pan

`openzoom::AssistiveOverlay`
- Semi-transparent overlay widget drawn on top of the render surface.
- Public API:
  - `explicit AssistiveOverlay(QWidget* parent = nullptr)`
  - `void SetContent(const QString& title, const QString& body, bool visible)`

## Entry Point

### `src/app/main.cpp`
- On Windows, constructs `OpenZoomApp` and runs it inside a `try`/`catch`.
- On non-Windows platforms, exits with an unsupported-platform message.
