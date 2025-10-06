#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QObject>
#include <QWidget>
#include <QPointF>
#include <QRectF>

#include <wrl/client.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <atomic>
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
QT_END_NAMESPACE

namespace openzoom {

class D3D12Presenter;
class RenderWidget;
class MainWindow;
class JoystickOverlay;

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
    void OnControlsCollapsedToggled(bool checked);
    void OnVirtualJoystickToggled(bool checked);
    void OnBlurToggled(bool checked);
    void OnBlurSigmaChanged(int value);
    void OnBlurRadiusChanged(int value);

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
    void HandleZoomWheel(int delta);
    void UpdateBlurUiLabels();

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

    bool comInitialized_{};
    bool mfInitialized_{};

    bool blackWhiteEnabled_{};
    float blackWhiteThreshold_{0.5f};
    bool zoomEnabled_{};
    float zoomAmount_{1.0f};
    bool debugViewEnabled_{};
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

    std::vector<uint8_t> stageRaw_;
    std::vector<uint8_t> stageBw_;
    std::vector<uint8_t> stageZoom_;
    std::vector<uint8_t> stageFinal_;
    std::vector<uint8_t> compositeBuffer_;
    std::vector<uint8_t> presentationBuffer_;
    std::vector<uint8_t> stageBlur_;
    std::vector<uint8_t> blurScratch_;

    JoystickOverlay* joystickOverlay_{};
};

} // namespace openzoom

#endif // _WIN32
