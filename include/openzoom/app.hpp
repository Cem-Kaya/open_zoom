#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QObject>

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
QT_END_NAMESPACE

namespace openzoom {

class D3D12Presenter;
class RenderWidget;
class MainWindow;

class OpenZoomApp : public QObject {
    Q_OBJECT
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

    QApplication* qtApp_{};
    std::unique_ptr<MainWindow> mainWindow_;
    QTimer* frameTimer_{};
    RenderWidget* renderWidget_{};
    QComboBox* cameraCombo_{};
    QCheckBox* bwCheckbox_{};
    QSlider* bwSlider_{};
    QCheckBox* zoomCheckbox_{};
    QSlider* zoomSlider_{};

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

    std::vector<uint8_t> stageRaw_;
    std::vector<uint8_t> stageBw_;
    std::vector<uint8_t> stageZoom_;
    std::vector<uint8_t> stageFinal_;
    std::vector<uint8_t> compositeBuffer_;
};

} // namespace openzoom

#endif // _WIN32
