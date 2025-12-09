#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QObject>
#include <QWidget>
#include <QPointF>
#include <QRectF>
#include <QSize>
#include <QString>

#include "openzoom/app/settings_store.hpp"
#include "openzoom/capture/media_capture.hpp"
#include "openzoom/common/frame_pipeline.hpp"
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
class QEvent;
class QShowEvent;
class QPaintEvent;
class QMouseEvent;
class QResizeEvent;
class QWheelEvent;
QT_END_NAMESPACE

struct ID3D12Resource;

namespace openzoom {

class RenderWidget;
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

private:
    void InitializePlatform();
    void EnumerateCameras();
    void PopulateCameraCombo();
    void StartCameraCapture(size_t index);
    void StopCameraCapture();
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
    bool IsMousePanActive() const;
    bool MapViewToSource(const QPointF& pos, float& outX, float& outY) const;
    bool EnsureCudaSurface(UINT width, UINT height);
    bool ProcessFrameWithCuda(UINT width, UINT height);
    void ResetCudaFenceState();
    void ResolveCudaBufferFormatFromOptions();
    void HandleCameraStartFailure(const QString& message);
    void UpdateRotationUi();
    static void RotateNormalizedPoint(float inX, float inY, int quarterTurns, float& outX, float& outY);
    void ApplyPersistentSettings(const settings::PersistentSettings& settings);
    void SavePersistentSettings();

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
    QCheckBox* focusMarkerCheckbox_{};
    QSlider* zoomCenterXSlider_{};
    QSlider* zoomCenterYSlider_{};
    QComboBox* rotationCombo_{};
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

    std::vector<CameraDescriptor> cameras_;
    MediaCapture mediaCapture_;
    std::mutex cameraMutex_;
    MediaFrame latestFrame_;
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
    std::vector<uint8_t> presentationBuffer_;
    processing::CpuFramePipeline cpuPipeline_;

    JoystickOverlay* joystickOverlay_{};
    std::unique_ptr<InteractionController> interactionController_;

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
