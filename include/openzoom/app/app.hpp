#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QObject>
#include <QWidget>
#include <QPointF>
#include <QRectF>
#include <QSize>
#include <QString>

#include "openzoom/cuda/cuda_interop.hpp"

#include <wrl/client.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <optional>

QT_BEGIN_NAMESPACE
class QApplication;
class QTimer;
class QComboBox;
class QCheckBox;
class QSlider;
class QPushButton;
class QToolButton;
class QLabel;
class QEvent;
class QShowEvent;
class QPaintEvent;
class QMouseEvent;
class QResizeEvent;
class QWheelEvent;
QT_END_NAMESPACE

struct ID3D12Resource;

namespace openzoom {

class D3D12Presenter;
class RenderWidget;
class MainWindow;
class JoystickOverlay;
class CudaInteropSurface;
class JoystickOverlay : public QWidget {
    Q_OBJECT
public:
    explicit JoystickOverlay(QWidget* parent = nullptr);

    void ResetKnob();

signals:
    void JoystickChanged(float normX, float normY);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    QRectF KnobRect() const;
    void UpdatePlacement();
    void UpdateFromPosition(const QPointF& pos);
    void UpdateMask();

    bool dragging_{};
    QPointF knobPos_{};
};

class OpenZoomApp : public QObject {
    Q_OBJECT
    friend class MainWindow;
public:
    OpenZoomApp(int& argc, char** argv);
    ~OpenZoomApp() override;

    int Run();

private slots:
    void OnFrameTick();
    void OnCameraSelectionChanged(int index);
    void OnBlackWhiteToggled(bool checked);
    void OnBlackWhiteThresholdChanged(int value);
    void OnZoomToggled(bool checked);
    void OnZoomAmountChanged(int value);
    void OnDebugViewToggled(bool checked);
    void OnZoomCenterXChanged(int value);
    void OnZoomCenterYChanged(int value);
    void OnRotateButtonClicked();
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

private:
    void InitializePlatform();
    void EnumerateCameras();
    void PopulateCameraCombo();
    void StartCameraCapture(size_t index);
    void StopCameraCapture();
    void CameraCaptureLoop();
    bool ConvertFrameToBgra(const std::vector<uint8_t>& frame,
                            const GUID& subtype,
                            UINT width,
                            UINT height,
                            UINT stride,
                            UINT dataSize);
    void BuildCompositeAndPresent(UINT width, UINT height);
    void PresentFitted(const uint8_t* data,
                       UINT srcWidth,
                       UINT srcHeight,
                       bool cropToFill,
                       float centerXNorm,
                       float centerYNorm);
    void SetZoomCenter(float normX, float normY, bool syncUi);
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
    bool IsMousePanActive() const { return middlePanActive_; }
    bool MapViewToSource(const QPointF& pos, float& outX, float& outY) const;
    bool EnsureCudaSurface(UINT width, UINT height);
    bool ProcessFrameWithCuda(UINT width, UINT height);
    void ResetCudaFenceState();
    void ResolveCudaBufferFormatFromOptions();
    void HandleCameraStartFailure(const QString& message);
    void ApplyTemporalSmoothCpu(std::vector<uint8_t>& frame, UINT width, UINT height);
    void UpdateRotateButtonLabel();
    void ApplyRotationToStageRaw(UINT& width, UINT& height);
    static void RotateNormalizedPoint(float inX, float inY, int quarterTurns, float& outX, float& outY);
    struct PersistentSettings {
        int cameraIndex{-1};
        bool blackWhiteEnabled{false};
        float blackWhiteThreshold{0.5f};
        bool zoomEnabled{false};
        float zoomAmount{1.0f};
        float zoomCenterX{0.5f};
        float zoomCenterY{0.5f};
        bool blurEnabled{false};
        float blurSigma{1.0f};
        int blurRadius{3};
        bool temporalSmoothEnabled{false};
        float temporalSmoothAlpha{0.25f};
        bool spatialSharpenEnabled{false};
        SpatialUpscaler spatialUpscaler{SpatialUpscaler::kNis};
        float spatialSharpness{0.25f};
        bool debugView{false};
        bool focusMarker{false};
        bool virtualJoystick{false};
        bool controlsCollapsed{false};
        int rotationQuarterTurns{0};
    };
    std::optional<PersistentSettings> LoadPersistentSettingsFromDisk();
    void ApplyPersistentSettings(const PersistentSettings& settings);
    void SavePersistentSettings();
    QString ResolveSettingsPath() const;
    void EnsureSettingsDirectory(const QString& path) const;

    QApplication* qtApp_{};
    std::unique_ptr<MainWindow> mainWindow_;
    QTimer* frameTimer_{};
    RenderWidget* renderWidget_{};
    QComboBox* cameraCombo_{};
    QCheckBox* bwCheckbox_{};
    QSlider* bwSlider_{};
    QCheckBox* zoomCheckbox_{};
    QSlider* zoomSlider_{};
    QPushButton* debugButton_{};
    QPushButton* rotateButton_{};
    QCheckBox* focusMarkerCheckbox_{};
    QSlider* zoomCenterXSlider_{};
    QSlider* zoomCenterYSlider_{};
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
    QCheckBox* spatialSharpenCheckbox_{};
    QComboBox* spatialBackendCombo_{};
    QSlider* spatialSharpnessSlider_{};
    QLabel* spatialSharpnessValueLabel_{};
    QLabel* processingStatusLabel_{};

    std::unique_ptr<D3D12Presenter> presenter_;

    struct CameraInfo {
        std::wstring name;
        std::wstring symbolicLink;
        Microsoft::WRL::ComPtr<IMFActivate> activation;
    };

    std::vector<CameraInfo> cameras_;

    Microsoft::WRL::ComPtr<IMFMediaSource> cameraMediaSource_;
    Microsoft::WRL::ComPtr<IMFSourceReader> cameraReader_;
    std::thread cameraThread_;
    std::mutex cameraMutex_;
    std::vector<uint8_t> cameraFrameBuffer_;
    UINT cameraFrameWidth_{};
    UINT cameraFrameHeight_{};
    UINT cameraFrameStride_{};
    size_t cameraFrameDataSize_{};
    GUID cameraFrameSubtype_{GUID_NULL};
    bool cameraFrameValid_{};
    std::atomic<bool> cameraCaptureRunning_{false};
    bool cameraActive_{};
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
    bool panLeftPressed_{};
    bool panRightPressed_{};
    bool panUpPressed_{};
    bool panDownPressed_{};
    float joystickPanX_{};
    float joystickPanY_{};
    bool suspendControlSync_{};
    bool blurEnabled_{};
    float blurSigma_{1.0f};
    int blurRadius_{3};
    bool temporalSmoothEnabled_{};
    float temporalSmoothAlpha_{0.25f};
    bool spatialSharpenEnabled_{};
    SpatialUpscaler spatialUpscaler_{SpatialUpscaler::kNis};
    float spatialSharpness_{0.25f};
    int rotationQuarterTurns_{0};
    bool middlePanActive_{};
    QPointF middlePanLastPos_{};
    QSize middlePanWidgetSize_{};

    std::vector<uint8_t> stageRaw_;
    std::vector<uint8_t> stageBw_;
    std::vector<uint8_t> stageZoom_;
    std::vector<uint8_t> stageFinal_;
    std::vector<uint8_t> compositeBuffer_;
    std::vector<uint8_t> presentationBuffer_;
    std::vector<uint8_t> stageBlur_;
    std::vector<uint8_t> blurScratch_;
    std::vector<float> temporalHistoryCpu_;
    bool temporalHistoryValid_{};
    std::vector<uint8_t> rotatedStageBuffer_;

    JoystickOverlay* joystickOverlay_{};

    Microsoft::WRL::ComPtr<ID3D12Resource> cudaSharedTexture_;
    std::unique_ptr<CudaInteropSurface> cudaSurface_;
    UINT cudaSurfaceWidth_{};
    UINT cudaSurfaceHeight_{};
    bool cudaPipelineAvailable_{};
    bool usingCudaLastFrame_{};
    uint64_t sharedFenceCounter_{1};
    uint64_t lastCudaSignalValue_{};
    uint64_t lastGraphicsSignalValue_{};
    bool cudaFenceInteropEnabled_{};
    CudaBufferFormat cudaBufferFormat_{CudaBufferFormat::kRgba8};
    QString lastCameraError_;
    QString settingsPath_;
};

} // namespace openzoom

#endif // _WIN32
