#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QObject>
#include <QWidget>
#include <QPointF>
#include <QRectF>
#include <QSize>
#include <QString>
#include <QElapsedTimer>

#include "openzoom/app/settings_store.hpp"
#include "openzoom/capture/media_capture.hpp"
#include "openzoom/common/frame_pipeline.hpp"
#include "openzoom/common/media_writer.hpp"
#include "openzoom/common/assistive_runtime.hpp"
#include "openzoom/cuda/cuda_interop.hpp"

#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

QT_BEGIN_NAMESPACE
class QApplication;
class QTimer;
class QComboBox;
class QCheckBox;
class QSlider;
class QPushButton;
class QToolButton;
class QLabel;
class QListWidget;
class QListWidgetItem;
class QEvent;
class QShowEvent;
class QPaintEvent;
class QMouseEvent;
class QPlainTextEdit;
class QResizeEvent;
class QTextBrowser;
class QWheelEvent;
QT_END_NAMESPACE

struct ID3D12Resource;

namespace openzoom {

class RenderWidget;
class AssistiveOverlay;
class JoystickOverlay;
class MainWindow;
class InteractionController;

class D3D12Presenter;
class CudaInteropSurface;

class OpenZoomApp : public QObject {
    Q_OBJECT
    friend class MainWindow;
    friend class InteractionController;
public:
    OpenZoomApp(int& argc, char** argv);
    ~OpenZoomApp() override;

    int Run();

private slots:
    void OnFrameTick();
    void OnPresetSelectionChanged(QListWidgetItem* current, QListWidgetItem* previous);
    void OnCameraSelectionChanged(int index);
    void OnBlackWhiteToggled(bool checked);
    void OnBlackWhiteThresholdChanged(int value);
    void OnZoomToggled(bool checked);
    void OnZoomAmountChanged(int value);
    void OnDebugViewToggled(bool checked);
    void OnZoomCenterXChanged(int value);
    void OnZoomCenterYChanged(int value);
    void OnRotationSelectionChanged(int index);
    void OnControlsCollapsedToggled(bool checked);
    void OnVirtualJoystickToggled(bool checked);
    void OnBlurToggled(bool checked);
    void OnBlurSigmaChanged(int value);
    void OnBlurRadiusChanged(int value);
    void OnFocusMarkerToggled(bool checked);
    void OnSpatialSharpenToggled(bool checked);
    void OnSpatialUpscalerChanged(int index);
    void OnSpatialSharpnessChanged(int value);
    void OnTemporalSmoothToggled(bool checked);
    void OnTemporalSmoothStrengthChanged(int value);
    void OnOcrAssistToggled(bool checked);
    void OnVlmAssistToggled(bool checked);
    void OnAssistiveOverlayToggled(bool checked);
    void OnAssistiveOverlayUpdated(const QString& title, const QString& body, bool visible);
    void OnStabilizationToggled(bool checked);
    void OnStabilizationStrengthChanged(int value);
    void OnKeystoneToggled(bool checked);
    void OnAutoContrastToggled(bool checked);
    void OnAutoContrastStrengthChanged(int value);
    void OnDisplayColorModeChanged(int index);
    void OnContrastChanged(int value);
    void OnBrightnessChanged(int value);

private:
    settings::AdvancedConfig CaptureCurrentAdvancedConfig() const;
    void ApplyAdvancedConfig(const settings::AdvancedConfig& config);
    void PopulatePresetList();
    void RefreshPresetSelection(bool preserveCurrentSelection = false);
    void UpdatePresetDescription();
    void SyncCurrentConfigToPersistence(bool preservePresetSelection = false);
    void PromoteCurrentConfigToPreset();
    void UpdateAssistiveRuntimeState();
    void MaybeRequestAssistiveAnalysis(const uint8_t* data, UINT width, UINT height);
    AssistiveRuntimeConfig BuildAssistiveRuntimeConfig() const;
    void ApplyAssistiveSettingsToRuntime();
    void OpenAiSettingsDialog();
    void OpenNotesFile();
    void SubmitOnDemandAnalysis(bool runOcr, bool runVlm);
    void SubmitAssistantPrompt();
    void SubmitAssistantPromptText(const QString& prompt,
                                   bool clearAdvancedEditor,
                                   bool forceAttachFrame);
    void SubmitFloatingAssistantPrompt(const QString& prompt);
    void PopulateAssistantHistory();
    void LoadSelectedAssistantConversation();
    void SetAssistantBusy(bool busy);
    void AppendAssistantMessage(const QString& speaker, const QString& text);
    void InitializePlatform();
    void EnumerateCameras();
    void PopulateCameraCombo();
    void RefreshCameraModesList(size_t index);
    bool StartCameraCapture(size_t index, bool interactive = true);
    void StopCameraCapture();
    void BeginCameraReconnect();
    void DriveCameraReconnect();
    void BuildCompositeAndPresent(UINT width, UINT height);
    void PresentFitted(const uint8_t* data,
                       UINT srcWidth,
                       UINT srcHeight,
                       bool cropToFill,
                       float centerXNorm,
                       float centerYNorm);
    void SetZoomCenter(float normX, float normY, bool syncUi,
                       bool preservePresetSelection = false);
    void ApplyInputForces();
    void UpdateJoystickVisibility();
    bool HandlePanKey(int key, bool pressed);
    bool HandlePanScroll(const QWheelEvent* wheelEvent);
    void HandleZoomWheel(int delta, const QPointF& localPos);
    void UpdateBlurUiLabels();
    void UpdateProcessingStatusLabel();
    void UpdateSpatialSharpenUi();
    void UpdateTemporalSmoothUi();
    void BeginMousePan(const QPointF& pos, const QSize& widgetSize);
    bool UpdateMousePan(const QPointF& pos);
    void EndMousePan();
    bool IsMousePanActive() const;
    bool MapViewToSource(const QPointF& pos, float& outX, float& outY) const;
    bool EnsureCudaSurface(UINT width, UINT height);
    bool ProcessFrameWithCuda(UINT width, UINT height);
    bool TryProcessRawFrameWithCuda(const MediaFrame& frame);
    bool RunCudaPipeline(const ProcessingInput& input, UINT presentWidth, UINT presentHeight);
    void HandleGpuFramePresented(UINT width, UINT height);
    bool AssistiveAnalysisDue() const;
    void CaptureSnapshot(const uint8_t* data, UINT width, UINT height);
    void MaybeRecordFrame(const uint8_t* data, UINT width, UINT height);
    void StopRecordingUi();
    void ShowStatusMessage(const QString& message, int durationMs = 10000);
    QString EnsureOutputSubdir(const QString& subdir);
    void ResetCudaFenceState();
    void ResolveCudaBufferFormatFromOptions();
    void HandleCameraStartFailure(const QString& message);
    void HandleCameraRuntimeFailure(uint64_t captureSession, const QString& message);
    void UpdateRotationUi();
    static void RotateNormalizedPoint(float inX, float inY, int quarterTurns, float& outX, float& outY);
    void ApplyPersistentSettings(const settings::PersistentSettings& settings);
    void SavePersistentSettings();

    QApplication* qtApp_{};
    std::unique_ptr<MainWindow> mainWindow_;
    QTimer* frameTimer_{};
    RenderWidget* renderWidget_{};
    QComboBox* cameraCombo_{};
    QListWidget* presetList_{};
    QLabel* presetDescriptionLabel_{};
    QPushButton* promotePresetButton_{};
    QCheckBox* bwCheckbox_{};
    QSlider* bwSlider_{};
    QCheckBox* zoomCheckbox_{};
    QSlider* zoomSlider_{};
    QPushButton* debugButton_{};
    QPushButton* capturePhotoButton_{};
    QPushButton* recordButton_{};
    QCheckBox* focusMarkerCheckbox_{};
    QSlider* zoomCenterXSlider_{};
    QSlider* zoomCenterYSlider_{};
    QComboBox* rotationCombo_{};
    QListWidget* cameraModesList_{};
    QCheckBox* joystickCheckbox_{};
    QToolButton* collapseButton_{};
    QWidget* controlsContainer_{};
    QCheckBox* blurCheckbox_{};
    QSlider* blurSigmaSlider_{};
    QSlider* blurRadiusSlider_{};
    QLabel* blurSigmaValueLabel_{};
    QLabel* blurRadiusValueLabel_{};
    QCheckBox* temporalSmoothCheckbox_{};
    QSlider* temporalSmoothSlider_{};
    QLabel* temporalSmoothValueLabel_{};
    QCheckBox* ocrAssistCheckbox_{};
    QCheckBox* vlmAssistCheckbox_{};
    QCheckBox* assistiveOverlayCheckbox_{};
    QCheckBox* spatialSharpenCheckbox_{};
    QComboBox* spatialBackendCombo_{};
    QSlider* spatialSharpnessSlider_{};
    QLabel* spatialSharpnessValueLabel_{};
    QLabel* processingStatusLabel_{};
    QCheckBox* stabilizationCheckbox_{};
    QSlider* stabilizationStrengthSlider_{};
    QCheckBox* keystoneCheckbox_{};
    QCheckBox* autoContrastCheckbox_{};
    QSlider* autoContrastStrengthSlider_{};
    QComboBox* displayColorCombo_{};
    QSlider* contrastSlider_{};
    QSlider* brightnessSlider_{};
    QPushButton* explainNowButton_{};
    QPushButton* readTextButton_{};
    QPushButton* aiSettingsButton_{};
    QPushButton* openNotesButton_{};
    QLabel* assistantConnectionLabel_{};
    QLabel* assistantUsageLabel_{};
    QPushButton* assistantConnectButton_{};
    QTextBrowser* assistantTranscript_{};
    QPlainTextEdit* assistantPromptEdit_{};
    QCheckBox* assistantAttachFrameCheckbox_{};
    QPushButton* assistantSendButton_{};
    QPushButton* assistantStopButton_{};
    QPushButton* assistantNewButton_{};
    QListWidget* assistantHistoryList_{};
    QPushButton* assistantRenameButton_{};
    QPushButton* assistantExportButton_{};
    QPushButton* assistantDeleteButton_{};
    QString currentAssistantThreadId_;
    QString pendingAssistantPrompt_;
    bool assistantResponseOpen_{false};
    bool assistantResponseReceivedText_{false};
    bool codexReady_{false};
    bool codexSignedIn_{false};

    std::unique_ptr<D3D12Presenter> presenter_;

    std::vector<CameraDescriptor> cameras_;
    MediaCapture mediaCapture_;
    std::mutex cameraMutex_;
    MediaFrame latestFrame_;
    bool cameraActive_{};
    uint64_t cameraSessionId_{};
    int selectedCameraIndex_{-1};
    UINT processedFrameWidth_{};
    UINT processedFrameHeight_{};

    bool comInitialized_{};
    bool mfInitialized_{};

    bool blackWhiteEnabled_{};
    float blackWhiteThreshold_{0.5f};
    bool zoomEnabled_{};
    float zoomAmount_{1.0f};
    bool debugViewEnabled_{};
    bool focusMarkerEnabled_{};
    float zoomCenterX_{0.5f};
    float zoomCenterY_{0.5f};
    bool controlsCollapsed_{};
    bool virtualJoystickEnabled_{};
    bool suspendControlSync_{};
    bool blurEnabled_{};
    float blurSigma_{1.0f};
    int blurRadius_{3};
    bool temporalSmoothEnabled_{};
    float temporalSmoothAlpha_{0.25f};
    bool ocrAssistEnabled_{};
    bool vlmAssistEnabled_{};
    bool assistiveOverlayEnabled_{true};
    bool spatialSharpenEnabled_{};
    SpatialUpscaler spatialUpscaler_{SpatialUpscaler::kNis};
    float spatialSharpness_{0.25f};
    int rotationQuarterTurns_{0};
    bool stabilizationEnabled_{};
    float stabilizationStrength_{0.85f};
    int displayColorMode_{0};
    float contrast_{1.0f};
    float brightness_{0.0f};
    bool keystoneEnabled_{};
    bool autoContrastEnabled_{};
    float autoContrastStrength_{0.7f};
    bool simpleUiMode_{true};
    std::vector<uint8_t> presentationBuffer_;
    std::vector<uint8_t> recordingBuffer_;
    std::vector<uint8_t> assistiveBuffer_;
    std::vector<uint8_t> asyncReadbackBuffer_;
    processing::CpuFramePipeline cpuPipeline_;
    VideoRecorder videoRecorder_;
    bool recording_{false};
    QElapsedTimer recordingTimer_;
    uint64_t recordingFrameCount_{0};
    UINT recordingWidth_{0};
    UINT recordingHeight_{0};

    AssistiveOverlay* assistiveOverlay_{};
    JoystickOverlay* joystickOverlay_{};
    std::unique_ptr<AssistiveRuntime> assistiveRuntime_;
    std::unique_ptr<InteractionController> interactionController_;
    QElapsedTimer assistiveAnalysisTimer_;

    Microsoft::WRL::ComPtr<ID3D12Resource> cudaSharedTexture_;
    std::unique_ptr<CudaInteropSurface> cudaSurface_;
    UINT cudaSurfaceWidth_{};
    UINT cudaSurfaceHeight_{};
    bool cudaPipelineAvailable_{};
    bool usingCudaLastFrame_{};
    uint64_t sharedFenceCounter_{1};
    uint64_t lastCudaSignalValue_{};
    uint64_t lastGraphicsSignalValue_{};
    uint64_t lastReadbackSignalValue_{};
    bool cudaFenceInteropEnabled_{};
    CudaBufferFormat cudaBufferFormat_{CudaBufferFormat::kRgba8};
    bool rawCudaPathWarned_{false};
    bool cameraReconnectPending_{false};
    int cameraReconnectAttempt_{0};
    qint64 cameraReconnectStartedMs_{0};
    qint64 cameraReconnectNextAttemptMs_{0};
    QString transientStatusMessage_;
    qint64 transientStatusUntilMs_{0};
    QString lastCameraError_;
    QString settingsPath_;
    settings::PersistentSettings persistentSettings_{};
    bool presetSelectionSyncSuspended_{false};
    bool configTrackingSuspended_{false};
    UINT presentationWidth_{0};
    UINT presentationHeight_{0};
};

} // namespace openzoom

#endif // _WIN32
