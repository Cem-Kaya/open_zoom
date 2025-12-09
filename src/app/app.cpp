#ifdef _WIN32

#include "openzoom/app/app.hpp"
#include "openzoom/cuda/cuda_interop.hpp"
#include "openzoom/d3d12/presenter.hpp"
#include "openzoom/app/constants.hpp"
#include "openzoom/app/interaction_controller.hpp"
#include "openzoom/ui/main_window.hpp"
#include <QApplication>
#include <QCoreApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QKeyEvent>
#include <QPainter>
#include <QEvent>
#include <QResizeEvent>
#include <QRegion>
#include <QWheelEvent>
#include <QTimer>
#include <QToolButton>
#include <QString>
#include <QSizePolicy>
#include <QPaintEngine>
#include <QResizeEvent>
#include <QShowEvent>
#include <QDebug>
#include <QMessageBox>

#include <windows.h>
#include <combaseapi.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <shlwapi.h>

#include <cwchar>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <wrl/client.h>

namespace openzoom {

namespace {

void ThrowIfFailed(HRESULT hr, const char* message)
{
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(message) + " (hr=0x" + std::to_string(static_cast<unsigned long>(hr)) + ")");
    }
}

namespace processing = openzoom::processing;
using namespace openzoom::app_constants;

CudaBufferFormat ParseCudaBufferFormatToken(const QString& token, bool* ok)
{
    const QString normalized = token.trimmed().toLower();
    if (normalized == QStringLiteral("rgba8") ||
        normalized == QStringLiteral("bgra8") ||
        normalized == QStringLiteral("uint8") ||
        normalized == QStringLiteral("8bit")) {
        if (ok) {
            *ok = true;
        }
        return CudaBufferFormat::kRgba8;
    }

    if (normalized == QStringLiteral("rgba16f") ||
        normalized == QStringLiteral("fp16") ||
        normalized == QStringLiteral("half") ||
        normalized == QStringLiteral("16f")) {
        if (ok) {
            *ok = true;
        }
        return CudaBufferFormat::kRgba16F;
    }

    if (ok) {
        *ok = false;
    }
    return CudaBufferFormat::kRgba8;
}

} // namespace




OpenZoomApp::OpenZoomApp(int& argc, char** argv)
    : QObject(nullptr) {
    qtApp_ = new QApplication(argc, argv);
    QCoreApplication::setOrganizationName(QStringLiteral("OpenZoom"));
    QCoreApplication::setApplicationName(QStringLiteral("OpenZoom"));
    ResolveCudaBufferFormatFromOptions();
    InitializePlatform();

    presenter_ = std::make_unique<D3D12Presenter>();
    ResetCudaFenceState();

    mainWindow_ = std::make_unique<MainWindow>();
    mainWindow_->setApp(this);
    renderWidget_ = mainWindow_->renderWidget();
    renderWidget_->setPresenter(presenter_.get());
    interactionController_ = std::make_unique<InteractionController>(*this);
    settingsPath_ = settings::ResolveSettingsPath();
    connect(qtApp_, &QCoreApplication::aboutToQuit, this, [this]() { SavePersistentSettings(); });
    cameraCombo_ = mainWindow_->cameraCombo();
    bwCheckbox_ = mainWindow_->blackWhiteCheckbox();
    bwSlider_ = mainWindow_->blackWhiteSlider();
    zoomCheckbox_ = mainWindow_->zoomCheckbox();
    zoomSlider_ = mainWindow_->zoomSlider();
    debugButton_ = mainWindow_->debugButton();
    rotationCombo_ = mainWindow_->rotationCombo();
    if (rotationCombo_) {
        qInfo() << "Rotation combo ready with" << rotationCombo_->count() << "items";
    } else {
        qWarning() << "Rotation combo not found";
    }
    focusMarkerCheckbox_ = mainWindow_->focusMarkerCheckbox();
    zoomCenterXSlider_ = mainWindow_->zoomCenterXSlider();
    zoomCenterYSlider_ = mainWindow_->zoomCenterYSlider();
    joystickCheckbox_ = mainWindow_->joystickCheckbox();
    collapseButton_ = mainWindow_->controlsToggleButton();
    controlsContainer_ = mainWindow_->controlsContainer();
    blurCheckbox_ = mainWindow_->blurCheckbox();
    blurSigmaSlider_ = mainWindow_->blurSigmaSlider();
    blurRadiusSlider_ = mainWindow_->blurRadiusSlider();
    blurSigmaValueLabel_ = mainWindow_->blurSigmaValueLabel();
    blurRadiusValueLabel_ = mainWindow_->blurRadiusValueLabel();
    temporalSmoothCheckbox_ = mainWindow_->temporalSmoothCheckbox();
    temporalSmoothSlider_ = mainWindow_->temporalSmoothSlider();
    temporalSmoothValueLabel_ = mainWindow_->temporalSmoothValueLabel();
    spatialSharpenCheckbox_ = mainWindow_->spatialSharpenCheckbox();
    spatialBackendCombo_ = mainWindow_->spatialBackendCombo();
    spatialSharpnessSlider_ = mainWindow_->spatialSharpnessSlider();
    spatialSharpnessValueLabel_ = mainWindow_->spatialSharpnessValueLabel();
    processingStatusLabel_ = mainWindow_->processingStatusLabel();

    joystickOverlay_ = new JoystickOverlay(renderWidget_);
    connect(joystickOverlay_, &JoystickOverlay::JoystickChanged,
            this, [this](float x, float y) {
                if (interactionController_) {
                    interactionController_->SetJoystickAxes(x, y);
                }
            });
    UpdateJoystickVisibility();

    if (zoomCenterXSlider_) {
        zoomCenterX_ = std::clamp(static_cast<float>(zoomCenterXSlider_->value()) / static_cast<float>(kZoomFocusSliderScale), 0.0f, 1.0f);
    }
    if (zoomCenterYSlider_) {
        zoomCenterY_ = std::clamp(static_cast<float>(zoomCenterYSlider_->value()) / static_cast<float>(kZoomFocusSliderScale), 0.0f, 1.0f);
    }

    SetZoomCenter(zoomCenterX_, zoomCenterY_, true);

    if (blurSigmaSlider_) {
        blurSigma_ = SliderValueToSigma(blurSigmaSlider_->value());
    }
    if (blurRadiusSlider_) {
        const int sliderValue = blurRadiusSlider_->value();
        const int snapped = SnapBlurRadius(sliderValue);
        blurRadius_ = snapped;
        if (sliderValue != snapped) {
            QSignalBlocker block(blurRadiusSlider_);
            blurRadiusSlider_->setValue(snapped);
        }
    }
    if (blurRadiusValueLabel_) {
        blurRadiusValueLabel_->setText(QString::number(blurRadius_));
    }
    blurEnabled_ = blurCheckbox_ ? blurCheckbox_->isChecked() : false;
    temporalSmoothEnabled_ = temporalSmoothCheckbox_ ? temporalSmoothCheckbox_->isChecked() : false;
    if (temporalSmoothSlider_) {
        temporalSmoothAlpha_ = std::clamp(static_cast<float>(temporalSmoothSlider_->value()) / 100.0f, 0.0f, 1.0f);
    }
    spatialSharpenEnabled_ = spatialSharpenCheckbox_ ? spatialSharpenCheckbox_->isChecked() : false;
    if (spatialBackendCombo_) {
        QSignalBlocker block(spatialBackendCombo_);
        spatialBackendCombo_->setCurrentIndex(static_cast<int>(SpatialUpscaler::kNis));
    }
    spatialUpscaler_ = SpatialUpscaler::kNis;
    if (spatialSharpnessSlider_) {
        spatialSharpness_ = std::clamp(static_cast<float>(spatialSharpnessSlider_->value()) / 100.0f, 0.0f, 1.0f);
    }
    if (focusMarkerCheckbox_) {
        focusMarkerEnabled_ = focusMarkerCheckbox_->isChecked();
    }

    PopulateCameraCombo();

    auto loadedSettings = settings::Load(settingsPath_);

    connect(cameraCombo_, &QComboBox::currentIndexChanged,
            this, &OpenZoomApp::OnCameraSelectionChanged);
    connect(bwCheckbox_, &QCheckBox::toggled,
            this, &OpenZoomApp::OnBlackWhiteToggled);
    connect(bwSlider_, &QSlider::valueChanged,
            this, &OpenZoomApp::OnBlackWhiteThresholdChanged);
    connect(zoomCheckbox_, &QCheckBox::toggled,
            this, &OpenZoomApp::OnZoomToggled);
    connect(zoomSlider_, &QSlider::valueChanged,
            this, &OpenZoomApp::OnZoomAmountChanged);
    if (debugButton_) {
        connect(debugButton_, &QPushButton::toggled,
                this, &OpenZoomApp::OnDebugViewToggled);
    }
    if (rotationCombo_) {
        connect(rotationCombo_, &QComboBox::currentIndexChanged,
                this, &OpenZoomApp::OnRotationSelectionChanged);
    }
    if (zoomCenterXSlider_) {
        connect(zoomCenterXSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnZoomCenterXChanged);
    }
    if (zoomCenterYSlider_) {
        connect(zoomCenterYSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnZoomCenterYChanged);
    }
    if (collapseButton_) {
        connect(collapseButton_, &QToolButton::toggled,
                this, &OpenZoomApp::OnControlsCollapsedToggled);
        OnControlsCollapsedToggled(collapseButton_->isChecked());
    }
    if (joystickCheckbox_) {
        joystickCheckbox_->setChecked(false);
        connect(joystickCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnVirtualJoystickToggled);
    }
    if (blurCheckbox_) {
        QSignalBlocker block(blurCheckbox_);
        blurCheckbox_->setChecked(blurEnabled_);
        connect(blurCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnBlurToggled);
    }
    if (blurSigmaSlider_) {
        connect(blurSigmaSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnBlurSigmaChanged);
    }
    if (blurRadiusSlider_) {
        connect(blurRadiusSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnBlurRadiusChanged);
    }
    if (temporalSmoothCheckbox_) {
        connect(temporalSmoothCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnTemporalSmoothToggled);
    }
    if (temporalSmoothSlider_) {
        connect(temporalSmoothSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnTemporalSmoothStrengthChanged);
    }
    if (spatialSharpenCheckbox_) {
        connect(spatialSharpenCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnSpatialSharpenToggled);
    }
    if (spatialBackendCombo_) {
        connect(spatialBackendCombo_, &QComboBox::currentIndexChanged,
                this, &OpenZoomApp::OnSpatialUpscalerChanged);
    }
    if (spatialSharpnessSlider_) {
        connect(spatialSharpnessSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnSpatialSharpnessChanged);
    }
    if (focusMarkerCheckbox_) {
        connect(focusMarkerCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnFocusMarkerToggled);
    }

    if (loadedSettings) {
        ApplyPersistentSettings(*loadedSettings);
    }

    UpdateBlurUiLabels();
    UpdateTemporalSmoothUi();
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
    UpdateRotationUi();

    frameTimer_ = new QTimer(this);
    connect(frameTimer_, &QTimer::timeout, this, &OpenZoomApp::OnFrameTick);
    frameTimer_->start(16);

    mainWindow_->show();

    int initialCameraIndex = 0;
    if (loadedSettings) {
        const int candidate = loadedSettings->cameraIndex;
        if (candidate >= 0 && static_cast<size_t>(candidate) < cameras_.size()) {
            initialCameraIndex = candidate;
        }
    }

    if (!cameras_.empty()) {
        initialCameraIndex = std::clamp(initialCameraIndex, 0, static_cast<int>(cameras_.size()) - 1);
        {
            QSignalBlocker blocker(cameraCombo_);
            cameraCombo_->setCurrentIndex(initialCameraIndex);
        }
        StartCameraCapture(static_cast<size_t>(initialCameraIndex));
    }
}

OpenZoomApp::~OpenZoomApp() {
    SavePersistentSettings();
    if (frameTimer_) {
        frameTimer_->stop();
    }
    StopCameraCapture();
    mediaCapture_.Shutdown();

    if (mfInitialized_) {
        MFShutdown();
        mfInitialized_ = false;
    }

    if (comInitialized_) {
        CoUninitialize();
        comInitialized_ = false;
    }

    delete qtApp_;
    qtApp_ = nullptr;
}

int OpenZoomApp::Run() {
    return qtApp_->exec();
}

void OpenZoomApp::InitializePlatform() {
    const HRESULT coInit = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (SUCCEEDED(coInit) || coInit == RPC_E_CHANGED_MODE) {
        comInitialized_ = SUCCEEDED(coInit);
    } else {
        ThrowIfFailed(coInit, "Failed to initialize COM");
    }

    const HRESULT mfInit = MFStartup(MF_VERSION);
    if (FAILED(mfInit)) {
        ThrowIfFailed(mfInit, "Failed to initialize Media Foundation");
    }
    mfInitialized_ = true;

    if (!mediaCapture_.Initialize()) {
        qWarning() << "MediaCapture initialization failed";
    }
    EnumerateCameras();
}

void OpenZoomApp::ResolveCudaBufferFormatFromOptions() {
    cudaBufferFormat_ = CudaBufferFormat::kRgba8;

    bool ok = false;
    const QByteArray envValue = qgetenv("OPENZOOM_CUDA_BUFFER_FORMAT");
    if (!envValue.isEmpty()) {
        const auto parsed = ParseCudaBufferFormatToken(QString::fromUtf8(envValue), &ok);
        if (ok) {
            cudaBufferFormat_ = parsed;
        } else {
            qWarning() << "Ignoring unknown OPENZOOM_CUDA_BUFFER_FORMAT value" << envValue;
        }
    }

    const QStringList args = qtApp_->arguments();
    for (const QString& arg : args) {
        if (arg.startsWith(QStringLiteral("--cuda-buffer-format="), Qt::CaseInsensitive)) {
            const QString value = arg.section('=', 1);
            const auto parsed = ParseCudaBufferFormatToken(value, &ok);
            if (ok) {
                cudaBufferFormat_ = parsed;
            } else {
                qWarning() << "Ignoring unknown --cuda-buffer-format option" << value;
            }
        } else if (arg.compare(QStringLiteral("--cuda-buffer-fp16"), Qt::CaseInsensitive) == 0) {
            cudaBufferFormat_ = CudaBufferFormat::kRgba16F;
        } else if (arg.compare(QStringLiteral("--cuda-buffer-rgba8"), Qt::CaseInsensitive) == 0) {
            cudaBufferFormat_ = CudaBufferFormat::kRgba8;
        }
    }

    const char* label = (cudaBufferFormat_ == CudaBufferFormat::kRgba16F) ? "RGBA16F" : "RGBA8";
    qInfo() << "CUDA staging buffer format set to" << label;
}

void OpenZoomApp::EnumerateCameras() {
    cameras_ = mediaCapture_.EnumerateCameras();
    selectedCameraIndex_ = cameras_.empty() ? -1 : 0;
}

void OpenZoomApp::PopulateCameraCombo() {
    if (!cameraCombo_) {
        return;
    }

    cameraCombo_->clear();
    for (const auto& camera : cameras_) {
        cameraCombo_->addItem(QString::fromWCharArray(camera.name.c_str()));
    }
}

void OpenZoomApp::OnCameraSelectionChanged(int index) {
    if (index < 0 || static_cast<size_t>(index) >= cameras_.size()) {
        return;
    }

    StartCameraCapture(static_cast<size_t>(index));
}

void OpenZoomApp::OnBlackWhiteToggled(bool checked) {
    blackWhiteEnabled_ = checked;
    if (bwSlider_) {
        bwSlider_->setEnabled(checked);
    }
}

void OpenZoomApp::OnBlackWhiteThresholdChanged(int value) {
    blackWhiteThreshold_ = std::clamp(static_cast<float>(value) / 255.0f, 0.0f, 1.0f);
}

void OpenZoomApp::OnZoomToggled(bool checked) {
    zoomEnabled_ = checked;
    if (zoomSlider_) {
        zoomSlider_->setEnabled(checked);
    }
}

void OpenZoomApp::OnZoomAmountChanged(int value) {
    zoomAmount_ = std::max(1.0f, static_cast<float>(value) / static_cast<float>(kZoomSliderScale));
}

void OpenZoomApp::OnDebugViewToggled(bool checked) {
    debugViewEnabled_ = checked;
    if (focusMarkerCheckbox_) {
        focusMarkerCheckbox_->setEnabled(!checked);
    }
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnZoomCenterXChanged(int value) {
    if (suspendControlSync_) {
        return;
    }
    const float norm = std::clamp(static_cast<float>(value) / static_cast<float>(kZoomFocusSliderScale), 0.0f, 1.0f);
    SetZoomCenter(norm, zoomCenterY_, false);
}

void OpenZoomApp::OnZoomCenterYChanged(int value) {
    if (suspendControlSync_) {
        return;
    }
    const float norm = std::clamp(static_cast<float>(value) / static_cast<float>(kZoomFocusSliderScale), 0.0f, 1.0f);
    SetZoomCenter(zoomCenterX_, norm, false);
}

void OpenZoomApp::OnRotationSelectionChanged(int index) {
    if (!rotationCombo_) {
        return;
    }

    const int clamped = std::clamp(index, 0, 3);
    const int previous = ((rotationQuarterTurns_ % 4) + 4) % 4;
    if (clamped == previous) {
        return;
    }

    const int delta = (clamped - previous + 4) % 4;
    rotationQuarterTurns_ = clamped;

    float rotatedX = zoomCenterX_;
    float rotatedY = zoomCenterY_;
    RotateNormalizedPoint(zoomCenterX_, zoomCenterY_, delta, rotatedX, rotatedY);
    SetZoomCenter(rotatedX, rotatedY, true);

    cpuPipeline_.ResetTemporalHistory();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
    }
    ResetCudaFenceState();

    processedFrameWidth_ = 0;
    processedFrameHeight_ = 0;

    UpdateRotationUi();
}

void OpenZoomApp::OnControlsCollapsedToggled(bool checked) {
    controlsCollapsed_ = !checked;
    if (controlsContainer_) {
        controlsContainer_->setVisible(checked);
    }
    if (collapseButton_) {
        collapseButton_->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        collapseButton_->setText(checked ? "Hide Controls" : "Show Controls");
    }
}

void OpenZoomApp::OnVirtualJoystickToggled(bool checked) {
    virtualJoystickEnabled_ = checked;
    if (!virtualJoystickEnabled_) {
        if (interactionController_) {
            interactionController_->ResetJoystick();
        }
        if (joystickOverlay_) {
            joystickOverlay_->ResetKnob();
        }
    } else {
        if (interactionController_) {
            interactionController_->ResetJoystick();
        }
    }
    UpdateJoystickVisibility();
}

void OpenZoomApp::OnBlurToggled(bool checked) {
    blurEnabled_ = checked;
    UpdateBlurUiLabels();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnBlurSigmaChanged(int value) {
    blurSigma_ = SliderValueToSigma(value);
    UpdateBlurUiLabels();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnBlurRadiusChanged(int value) {
    const int snapped = SnapBlurRadius(value);
    if (blurRadiusSlider_ && snapped != value) {
        QSignalBlocker blocker(blurRadiusSlider_);
        blurRadiusSlider_->setValue(snapped);
    }
    blurRadius_ = snapped;
    UpdateBlurUiLabels();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnFocusMarkerToggled(bool checked) {
    focusMarkerEnabled_ = checked;
}

void OpenZoomApp::OnSpatialSharpenToggled(bool checked) {
    spatialSharpenEnabled_ = checked;
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnSpatialUpscalerChanged(int index) {
    const int clamped = std::clamp(index, 0, 1);
    spatialUpscaler_ = static_cast<SpatialUpscaler>(clamped);
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnSpatialSharpnessChanged(int value) {
    spatialSharpness_ = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
    if (spatialSharpnessValueLabel_) {
        spatialSharpnessValueLabel_->setText(QString::number(spatialSharpness_, 'f', 2));
    }
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnTemporalSmoothToggled(bool checked) {
    temporalSmoothEnabled_ = checked;
    if (temporalSmoothSlider_) {
        temporalSmoothSlider_->setEnabled(checked);
    }
    cpuPipeline_.ResetTemporalHistory();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
    }
    UpdateTemporalSmoothUi();
}

void OpenZoomApp::OnTemporalSmoothStrengthChanged(int value) {
    const int sliderMin = temporalSmoothSlider_ ? temporalSmoothSlider_->minimum() : 1;
    const int sliderMax = temporalSmoothSlider_ ? temporalSmoothSlider_->maximum() : 100;
    const int clamped = std::clamp(value, sliderMin, sliderMax);
    if (temporalSmoothSlider_ && clamped != value) {
        QSignalBlocker block(temporalSmoothSlider_);
        temporalSmoothSlider_->setValue(clamped);
    }
    temporalSmoothAlpha_ = std::clamp(static_cast<float>(clamped) / 100.0f, 0.0f, 1.0f);
    if (temporalSmoothValueLabel_) {
        temporalSmoothValueLabel_->setText(QString::number(temporalSmoothAlpha_, 'f', 2));
    }
    cpuPipeline_.ResetTemporalHistory();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
    }
    UpdateTemporalSmoothUi();
}

void OpenZoomApp::SetZoomCenter(float normX, float normY, bool syncUi) {
    const float clampedX = std::clamp(normX, 0.0f, 1.0f);
    const float clampedY = std::clamp(normY, 0.0f, 1.0f);
    zoomCenterX_ = clampedX;
    zoomCenterY_ = clampedY;

    if (syncUi) {
        suspendControlSync_ = true;
        if (zoomCenterXSlider_) {
            QSignalBlocker blockX(zoomCenterXSlider_);
            zoomCenterXSlider_->setValue(static_cast<int>(std::round(clampedX * kZoomFocusSliderScale)));
        }
        if (zoomCenterYSlider_) {
            QSignalBlocker blockY(zoomCenterYSlider_);
            zoomCenterYSlider_->setValue(static_cast<int>(std::round(clampedY * kZoomFocusSliderScale)));
        }
        suspendControlSync_ = false;
    }
}

bool OpenZoomApp::HandlePanKey(int key, bool pressed) {
    if (!interactionController_) {
        return false;
    }
    return interactionController_->HandlePanKey(key, pressed);
}

bool OpenZoomApp::HandlePanScroll(const QWheelEvent* wheelEvent) {
    if (!interactionController_) {
        return false;
    }
    return interactionController_->HandlePanScroll(wheelEvent);
}

void OpenZoomApp::ApplyInputForces() {
    if (!interactionController_) {
        return;
    }
    interactionController_->ApplyInputForces();
}

void OpenZoomApp::UpdateJoystickVisibility() {
    if (!joystickOverlay_) {
        return;
    }
    if (virtualJoystickEnabled_) {
        joystickOverlay_->ResetKnob();
        joystickOverlay_->show();
    } else {
        joystickOverlay_->hide();
    }
}

void OpenZoomApp::UpdateBlurUiLabels() {
    const QString sigmaText = QString::number(blurSigma_, 'f', 1);
    const bool blurActive = blurEnabled_;
    if (blurSigmaValueLabel_) {
        blurSigmaValueLabel_->setText(sigmaText);
        blurSigmaValueLabel_->setEnabled(blurActive);
    }
    if (blurRadiusValueLabel_) {
        blurRadiusValueLabel_->setText(QString::number(blurRadius_));
        blurRadiusValueLabel_->setEnabled(blurActive);
    }
    if (blurSigmaSlider_) {
        blurSigmaSlider_->setEnabled(blurActive);
    }
    if (blurRadiusSlider_) {
        QSignalBlocker block(blurRadiusSlider_);
        blurRadiusSlider_->setValue(blurRadius_);
        blurRadiusSlider_->setEnabled(blurActive);
    }
    if (blurCheckbox_) {
        QSignalBlocker block(blurCheckbox_);
        blurCheckbox_->setChecked(blurEnabled_);
        blurCheckbox_->setEnabled(true);
    }
}

void OpenZoomApp::UpdateTemporalSmoothUi() {
    if (temporalSmoothCheckbox_) {
        QSignalBlocker block(temporalSmoothCheckbox_);
        temporalSmoothCheckbox_->setChecked(temporalSmoothEnabled_);
    }
    if (temporalSmoothSlider_) {
        temporalSmoothSlider_->setEnabled(temporalSmoothEnabled_);
        const int sliderValue = static_cast<int>(std::round(temporalSmoothAlpha_ * 100.0f));
        QSignalBlocker block(temporalSmoothSlider_);
        temporalSmoothSlider_->setValue(std::clamp(sliderValue,
                                                  temporalSmoothSlider_->minimum(),
                                                  temporalSmoothSlider_->maximum()));
    }
    if (temporalSmoothValueLabel_) {
        temporalSmoothValueLabel_->setEnabled(temporalSmoothEnabled_);
        temporalSmoothValueLabel_->setText(QString::number(temporalSmoothAlpha_, 'f', 2));
    }
}

void OpenZoomApp::RotateNormalizedPoint(float inX, float inY, int quarterTurns, float& outX, float& outY) {
    const int turnsRaw = quarterTurns % 4;
    const int turns = turnsRaw < 0 ? turnsRaw + 4 : turnsRaw;
    switch (turns) {
    case 0:
        outX = inX;
        outY = inY;
        break;
    case 1:
        outX = 1.0f - inY;
        outY = std::clamp(inX, 0.0f, 1.0f);
        break;
    case 2:
        outX = 1.0f - inX;
        outY = 1.0f - inY;
        break;
    case 3:
        outX = std::clamp(inY, 0.0f, 1.0f);
        outY = 1.0f - inX;
        break;
    default:
        outX = inX;
        outY = inY;
        break;
    }
    outX = std::clamp(outX, 0.0f, 1.0f);
    outY = std::clamp(outY, 0.0f, 1.0f);
}






void OpenZoomApp::UpdateRotationUi() {
    const int turnsRaw = rotationQuarterTurns_ % 4;
    const int turns = turnsRaw < 0 ? turnsRaw + 4 : turnsRaw;
    if (rotationCombo_) {
        QSignalBlocker block(rotationCombo_);
        if (turns >= 0 && turns < rotationCombo_->count()) {
            rotationCombo_->setCurrentIndex(turns);
        }
    }
}

void OpenZoomApp::UpdateSpatialSharpenUi() {
    const bool enabled = spatialSharpenEnabled_;

    if (spatialSharpenCheckbox_) {
        QSignalBlocker block(spatialSharpenCheckbox_);
        spatialSharpenCheckbox_->setChecked(enabled);
    }
    if (spatialBackendCombo_) {
        QSignalBlocker block(spatialBackendCombo_);
        spatialBackendCombo_->setCurrentIndex(static_cast<int>(spatialUpscaler_));
        spatialBackendCombo_->setEnabled(enabled);
    }
    if (spatialSharpnessSlider_) {
        spatialSharpnessSlider_->setEnabled(enabled);
        if (enabled) {
            QSignalBlocker block(spatialSharpnessSlider_);
            spatialSharpnessSlider_->setValue(static_cast<int>(std::round(spatialSharpness_ * 100.0f)));
        }
    }
    if (spatialSharpnessValueLabel_) {
        spatialSharpnessValueLabel_->setEnabled(enabled);
        spatialSharpnessValueLabel_->setText(QString::number(spatialSharpness_, 'f', 2));
    }
}

void OpenZoomApp::UpdateProcessingStatusLabel() {
    if (!processingStatusLabel_) {
        return;
    }

    QString text;
    QString color;

    auto backendLabel = [this]() -> QString {
        if (spatialSharpenEnabled_) {
            if (spatialUpscaler_ == SpatialUpscaler::kNis) {
                return QStringLiteral("Spatial Sharpen (NIS)");
            }
            return QStringLiteral("Spatial Sharpen (FSR)");
        }
        if (blurEnabled_) {
            return QStringLiteral("Gaussian Blur");
        }
        return QStringLiteral("None");
    };

    if (!cameraActive_) {
        if (lastCameraError_.isEmpty()) {
            text = QStringLiteral("Processing: Idle (camera offline)");
        } else {
            text = QStringLiteral("Processing: Idle (camera offline — %1)").arg(lastCameraError_);
        }
        color = QStringLiteral("#c0392b");
    } else if (debugViewEnabled_) {
        text = QStringLiteral("Processing: CPU (debug view)");
        color = QStringLiteral("#d17c00");
    } else if (usingCudaLastFrame_ && cudaPipelineAvailable_) {
        const QString backend = backendLabel();
        if (cudaFenceInteropEnabled_) {
            text = QStringLiteral("Processing: GPU (fence interop, %1)").arg(backend);
        } else {
            text = QStringLiteral("Processing: GPU (%1)").arg(backend);
        }
        color = QStringLiteral("#1c9c3e");
    } else if (cudaPipelineAvailable_) {
        text = QStringLiteral("Processing: CPU (fallback, %1)").arg(backendLabel());
        color = QStringLiteral("#c0392b");
    } else {
        text = QStringLiteral("Processing: CPU (%1)").arg(backendLabel());
        color = QStringLiteral("#c0392b");
    }

    processingStatusLabel_->setText(text);
    processingStatusLabel_->setStyleSheet(QStringLiteral("color: %1;").arg(color));
}

void OpenZoomApp::HandleZoomWheel(int delta, const QPointF& localPos) {
    if (!interactionController_) {
        return;
    }
    interactionController_->HandleZoomWheel(delta, localPos);
}

bool OpenZoomApp::MapViewToSource(const QPointF& pos, float& outX, float& outY) const {
    if (!renderWidget_ || processedFrameWidth_ == 0 || processedFrameHeight_ == 0) {
        return false;
    }

    const int targetWidth = std::max(1, renderWidget_->width());
    const int targetHeight = std::max(1, renderWidget_->height());

    const float srcWidth = static_cast<float>(processedFrameWidth_);
    const float srcHeight = static_cast<float>(processedFrameHeight_);

    bool cropToFill = !debugViewEnabled_;
    float cropWidth = srcWidth;
    float cropHeight = srcHeight;

    if (cropToFill) {
        const float targetAspect = static_cast<float>(targetWidth) / static_cast<float>(targetHeight);
        const float srcAspect = srcWidth / srcHeight;
        if (targetAspect > srcAspect) {
            cropHeight = srcWidth / targetAspect;
            cropHeight = std::min(cropHeight, srcHeight);
        } else {
            cropWidth = srcHeight * targetAspect;
            cropWidth = std::min(cropWidth, srcWidth);
        }
    }

    cropWidth = std::clamp(cropWidth, 1.0f, srcWidth);
    cropHeight = std::clamp(cropHeight, 1.0f, srcHeight);

    float centerX = zoomCenterX_ * (srcWidth - 1.0f);
    float centerY = zoomCenterY_ * (srcHeight - 1.0f);

    const float halfCropWidth = cropWidth * 0.5f;
    const float halfCropHeight = cropHeight * 0.5f;

    const float minCenterX = std::max(0.0f, halfCropWidth - 0.5f);
    const float maxCenterX = std::max(minCenterX, (srcWidth - 1.0f) - (halfCropWidth - 0.5f));
    const float minCenterY = std::max(0.0f, halfCropHeight - 0.5f);
    const float maxCenterY = std::max(minCenterY, (srcHeight - 1.0f) - (halfCropHeight - 0.5f));

    centerX = std::clamp(centerX, minCenterX, maxCenterX);
    centerY = std::clamp(centerY, minCenterY, maxCenterY);

    float startX = centerX - halfCropWidth + 0.5f;
    float startY = centerY - halfCropHeight + 0.5f;
    startX = std::clamp(startX, 0.0f, srcWidth - cropWidth);
    startY = std::clamp(startY, 0.0f, srcHeight - cropHeight);

    float scaleFactor;
    if (cropToFill) {
        scaleFactor = static_cast<float>(targetWidth) / cropWidth;
    } else {
        const float widthScale = static_cast<float>(targetWidth) / cropWidth;
        const float heightScale = static_cast<float>(targetHeight) / cropHeight;
        scaleFactor = std::min(widthScale, heightScale);
    }
    if (!(scaleFactor > 0.0f) || !std::isfinite(scaleFactor)) {
        scaleFactor = 1.0f;
    }

    UINT activeWidth = static_cast<UINT>(std::round(cropWidth * scaleFactor));
    UINT activeHeight = static_cast<UINT>(std::round(cropHeight * scaleFactor));
    activeWidth = std::clamp(activeWidth, 1u, static_cast<UINT>(targetWidth));
    activeHeight = std::clamp(activeHeight, 1u, static_cast<UINT>(targetHeight));

    const UINT offsetX = (targetWidth > static_cast<int>(activeWidth)) ?
                         static_cast<UINT>((targetWidth - static_cast<int>(activeWidth)) / 2) : 0;
    const UINT offsetY = (targetHeight > static_cast<int>(activeHeight)) ?
                         static_cast<UINT>((targetHeight - static_cast<int>(activeHeight)) / 2) : 0;

    float localX = static_cast<float>(pos.x()) - static_cast<float>(offsetX);
    float localY = static_cast<float>(pos.y()) - static_cast<float>(offsetY);

    localX = std::clamp(localX, 0.0f, static_cast<float>(activeWidth - 1));
    localY = std::clamp(localY, 0.0f, static_cast<float>(activeHeight - 1));

    const float stepX = cropWidth / static_cast<float>(activeWidth);
    const float stepY = cropHeight / static_cast<float>(activeHeight);

    const float sampleX = startX + localX * stepX;
    const float sampleY = startY + localY * stepY;

    const float denomX = std::max(srcWidth - 1.0f, 1.0f);
    const float denomY = std::max(srcHeight - 1.0f, 1.0f);
    outX = std::clamp(sampleX / denomX, 0.0f, 1.0f);
    outY = std::clamp(sampleY / denomY, 0.0f, 1.0f);
    return true;
}

void OpenZoomApp::ResetCudaFenceState() {
    const UINT64 baseValue = presenter_ ? presenter_->GetLastSignaledFenceValue() : 0;
    sharedFenceCounter_ = baseValue + 1;
    lastCudaSignalValue_ = 0;
    lastGraphicsSignalValue_ = baseValue;
    cudaFenceInteropEnabled_ = false;
}

bool OpenZoomApp::EnsureCudaSurface(UINT width, UINT height) {
    if (!presenter_ || !renderWidget_ || !renderWidget_->isPresenterReady()) {
        qWarning() << "CUDA surface unavailable: presenter or render widget not ready";
        return false;
    }

    if (cudaSurface_ && cudaSurface_->IsValid() &&
        cudaSurfaceWidth_ == width && cudaSurfaceHeight_ == height) {
        return true;
    }

    cudaSurface_.reset();
    cudaSharedTexture_.Reset();
    cudaSurfaceWidth_ = 0;
    cudaSurfaceHeight_ = 0;
    ResetCudaFenceState();

    ID3D12Device* device = presenter_->GetDevice();
    if (!device) {
        qWarning() << "CUDA surface unavailable: presenter returned null device";
        return false;
    }

    try {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Alignment = 0;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;

        ThrowIfFailed(device->CreateCommittedResource(&heapProps,
                                                      D3D12_HEAP_FLAG_SHARED,
                                                      &desc,
                                                      D3D12_RESOURCE_STATE_COMMON,
                                                      nullptr,
                                                      IID_PPV_ARGS(&cudaSharedTexture_)),
                      "Failed to create CUDA shared texture");

        auto surface = std::make_unique<CudaInteropSurface>(cudaSharedTexture_.Get(), presenter_->GetFence());
        if (!surface || !surface->IsValid()) {
            if (surface) {
                const std::string& err = surface->LastError();
                if (!err.empty()) {
                    qWarning() << "CUDA surface detail:" << err.c_str();
                }
            }
            cudaSharedTexture_.Reset();
            qWarning() << "CUDA surface initialization failed: surface invalid"
                       << "(requested" << width << "x" << height << ")";
            return false;
        }

        cudaSurface_ = std::move(surface);
        cudaSurfaceWidth_ = width;
        cudaSurfaceHeight_ = height;
        cudaPipelineAvailable_ = true;
        cudaFenceInteropEnabled_ = cudaSurface_->HasExternalSemaphore();
        if (cudaFenceInteropEnabled_) {
            lastGraphicsSignalValue_ = presenter_->GetLastSignaledFenceValue();
            sharedFenceCounter_ = lastGraphicsSignalValue_ + 1;
            lastCudaSignalValue_ = 0;
            qInfo() << "CUDA fence interop enabled; base fence value"
                    << static_cast<unsigned long long>(lastGraphicsSignalValue_);
        } else {
            qInfo() << "CUDA surface ready without fence interop";
        }
        qInfo() << "CUDA surface ready for" << width << "x" << height;
        return true;
    } catch (...) {
        cudaSurface_.reset();
        cudaSharedTexture_.Reset();
        cudaSurfaceWidth_ = 0;
        cudaSurfaceHeight_ = 0;
        cudaPipelineAvailable_ = false;
        ResetCudaFenceState();
        qWarning() << "CUDA surface creation exception triggered fallback";
        return false;
    }
}

bool OpenZoomApp::ProcessFrameWithCuda(UINT width, UINT height) {
    const auto& stageRaw = cpuPipeline_.StageRaw();
    if (stageRaw.empty()) {
        qWarning() << "CUDA pipeline skipped: stage raw empty";
        return false;
    }

    if (!EnsureCudaSurface(width, height)) {
        qWarning() << "CUDA pipeline disabled: failed to ensure CUDA surface";
        if (cudaSurface_) {
            const std::string& err = cudaSurface_->LastError();
            if (!err.empty()) {
                qWarning() << "CUDA surface detail:" << err.c_str();
            }
        }
        usingCudaLastFrame_ = false;
        return false;
    }

    if (!cudaSurface_) {
        qWarning() << "CUDA pipeline disabled: surface not available";
        usingCudaLastFrame_ = false;
        return false;
    }

    ProcessingInput input{};
    input.hostPixels = stageRaw.data();
    input.hostStrideBytes = width * 4;
    input.pixelSizeBytes = static_cast<unsigned int>(sizeof(uint32_t));
    input.width = width;
    input.height = height;

    ProcessingSettings settings{};
    settings.enableBlackWhite = blackWhiteEnabled_;
    settings.blackWhiteThreshold = blackWhiteThreshold_;
    settings.enableZoom = zoomEnabled_;
    settings.zoomAmount = zoomAmount_;
    settings.zoomCenterX = zoomCenterX_;
    settings.zoomCenterY = zoomCenterY_;
    settings.enableBlur = blurEnabled_;
    settings.blurRadius = std::max(blurRadius_, 0);
    settings.blurSigma = blurSigma_;
    settings.drawFocusMarker = focusMarkerEnabled_ && !debugViewEnabled_;
    settings.enableSpatialSharpen = spatialSharpenEnabled_;
    settings.spatialUpscaler = spatialUpscaler_;
    settings.spatialSharpness = spatialSharpness_;
    settings.stagingFormat = cudaBufferFormat_;
    settings.enableTemporalSmoothing = temporalSmoothEnabled_;
    settings.temporalSmoothingAlpha = temporalSmoothAlpha_;

    FenceSyncParams cudaSyncParams{};
    uint64_t cudaSignalCandidate = 0;
    if (cudaFenceInteropEnabled_) {
        cudaSyncParams.enable = true;
        cudaSyncParams.waitValue = lastGraphicsSignalValue_;
        cudaSignalCandidate = sharedFenceCounter_;
        cudaSyncParams.signalValue = cudaSignalCandidate;
    }

    if (!cudaSurface_->ProcessFrame(input, settings, cudaSyncParams)) {
        cudaPipelineAvailable_ = false;
        qWarning() << "CUDA pipeline processing failed, falling back to CPU";
        ResetCudaFenceState();
        usingCudaLastFrame_ = false;
        return false;
    }

    if (cudaSyncParams.enable) {
        lastCudaSignalValue_ = cudaSignalCandidate;
        sharedFenceCounter_ = cudaSignalCandidate + 1;
    }

    cudaPipelineAvailable_ = true;
    FenceSyncParams presentSync{};
    uint64_t graphicsSignalCandidate = 0;
    if (cudaFenceInteropEnabled_) {
        presentSync.enable = true;
        presentSync.waitValue = lastCudaSignalValue_;
        graphicsSignalCandidate = sharedFenceCounter_;
        presentSync.signalValue = graphicsSignalCandidate;
    }

    presenter_->PresentFromTexture(cudaSharedTexture_.Get(),
                                   width,
                                   height,
                                   presentSync.enable ? &presentSync : nullptr);

    if (presentSync.enable) {
        lastGraphicsSignalValue_ = graphicsSignalCandidate;
        sharedFenceCounter_ = graphicsSignalCandidate + 1;
    }

    usingCudaLastFrame_ = true;
    return true;
}

void OpenZoomApp::BeginMousePan(const QPointF& pos, const QSize& widgetSize) {
    if (!interactionController_) {
        return;
    }
    interactionController_->BeginMousePan(pos, widgetSize);
}

bool OpenZoomApp::UpdateMousePan(const QPointF& pos) {
    if (!interactionController_) {
        return false;
    }
    return interactionController_->UpdateMousePan(pos);
}

void OpenZoomApp::EndMousePan() {
    if (!interactionController_) {
        return;
    }
    interactionController_->EndMousePan();
}

bool OpenZoomApp::IsMousePanActive() const
{
    return interactionController_ && interactionController_->IsMousePanActive();
}

void OpenZoomApp::StartCameraCapture(size_t index) {
    if (index >= cameras_.size()) {
        return;
    }

    selectedCameraIndex_ = static_cast<int>(index);
    StopCameraCapture();
    cpuPipeline_.ResetTemporalHistory();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
    }

    const CameraDescriptor& descriptor = cameras_[index];
    FrameCallback callback = [this](const MediaFrame& frame) {
        std::scoped_lock lock(cameraMutex_);
        latestFrame_ = frame;
    };

    if (!mediaCapture_.StartCapture(descriptor, std::move(callback))) {
        HandleCameraStartFailure(QStringLiteral("Failed to start camera capture"));
        return;
    }

    processedFrameWidth_ = 0;
    processedFrameHeight_ = 0;
    cameraActive_ = true;
    lastCameraError_.clear();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::StopCameraCapture() {
    mediaCapture_.StopCapture();
    cameraActive_ = false;

    {
        std::scoped_lock lock(cameraMutex_);
        latestFrame_ = MediaFrame{};
    }

    cpuPipeline_.ResetTemporalHistory();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
    }
    processedFrameWidth_ = 0;
    processedFrameHeight_ = 0;
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::HandleCameraStartFailure(const QString& message) {
    qWarning() << "Camera start failed:" << message;
    lastCameraError_ = message;
    StopCameraCapture();
    UpdateProcessingStatusLabel();
    if (mainWindow_) {
        QMessageBox::warning(mainWindow_.get(), QStringLiteral("Camera Error"), message);
    }
}



void OpenZoomApp::ApplyPersistentSettings(const settings::PersistentSettings& settings) {
    if (bwCheckbox_) {
        QSignalBlocker block(bwCheckbox_);
        bwCheckbox_->setChecked(settings.blackWhiteEnabled);
    }
    blackWhiteEnabled_ = settings.blackWhiteEnabled;
    blackWhiteThreshold_ = std::clamp(settings.blackWhiteThreshold, 0.0f, 1.0f);
    if (bwSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(blackWhiteThreshold_ * 255.0f)),
                                           bwSlider_->minimum(), bwSlider_->maximum());
        QSignalBlocker block(bwSlider_);
        bwSlider_->setValue(sliderValue);
        OnBlackWhiteThresholdChanged(sliderValue);
    }
    OnBlackWhiteToggled(blackWhiteEnabled_);

    if (zoomCheckbox_) {
        QSignalBlocker block(zoomCheckbox_);
        zoomCheckbox_->setChecked(settings.zoomEnabled);
    }
    zoomEnabled_ = settings.zoomEnabled;
    zoomAmount_ = std::clamp(settings.zoomAmount, 1.0f, static_cast<float>(kZoomSliderMaxMultiplier));
    if (zoomSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(zoomAmount_ * kZoomSliderScale)),
                                           zoomSlider_->minimum(), zoomSlider_->maximum());
        QSignalBlocker block(zoomSlider_);
        zoomSlider_->setValue(sliderValue);
        OnZoomAmountChanged(sliderValue);
    }
    SetZoomCenter(settings.zoomCenterX, settings.zoomCenterY, true);
    OnZoomToggled(zoomEnabled_);

    if (blurCheckbox_) {
        QSignalBlocker block(blurCheckbox_);
        blurCheckbox_->setChecked(settings.blurEnabled);
    }
    blurEnabled_ = settings.blurEnabled;
    blurSigma_ = std::clamp(settings.blurSigma, kBlurSigmaStep, static_cast<float>(kBlurSigmaSliderMax) * kBlurSigmaStep);
    blurRadius_ = SnapBlurRadius(settings.blurRadius);
    if (blurSigmaSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(blurSigma_ / kBlurSigmaStep)),
                                           blurSigmaSlider_->minimum(), blurSigmaSlider_->maximum());
        QSignalBlocker block(blurSigmaSlider_);
        blurSigmaSlider_->setValue(sliderValue);
        OnBlurSigmaChanged(sliderValue);
    }
    if (blurRadiusSlider_) {
        QSignalBlocker block(blurRadiusSlider_);
        blurRadiusSlider_->setValue(blurRadius_);
        OnBlurRadiusChanged(blurRadius_);
    }
    OnBlurToggled(blurEnabled_);

    if (temporalSmoothCheckbox_) {
        QSignalBlocker block(temporalSmoothCheckbox_);
        temporalSmoothCheckbox_->setChecked(settings.temporalSmoothEnabled);
    }
    temporalSmoothEnabled_ = settings.temporalSmoothEnabled;
    temporalSmoothAlpha_ = std::clamp(settings.temporalSmoothAlpha, 0.0f, 1.0f);
    if (temporalSmoothSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(temporalSmoothAlpha_ * 100.0f)),
                                           temporalSmoothSlider_->minimum(), temporalSmoothSlider_->maximum());
        QSignalBlocker block(temporalSmoothSlider_);
        temporalSmoothSlider_->setValue(sliderValue);
        OnTemporalSmoothStrengthChanged(sliderValue);
    }
    OnTemporalSmoothToggled(temporalSmoothEnabled_);

    if (spatialSharpenCheckbox_) {
        QSignalBlocker block(spatialSharpenCheckbox_);
        spatialSharpenCheckbox_->setChecked(settings.spatialSharpenEnabled);
    }
    spatialSharpenEnabled_ = settings.spatialSharpenEnabled;
    spatialUpscaler_ = settings.spatialUpscaler == 0 ? SpatialUpscaler::kFsrEasuRcas : SpatialUpscaler::kNis;
    spatialSharpness_ = std::clamp(settings.spatialSharpness, 0.0f, 1.0f);
    if (spatialBackendCombo_) {
        QSignalBlocker block(spatialBackendCombo_);
        spatialBackendCombo_->setCurrentIndex(static_cast<int>(spatialUpscaler_));
        OnSpatialUpscalerChanged(spatialBackendCombo_->currentIndex());
    }
    if (spatialSharpnessSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(spatialSharpness_ * 100.0f)),
                                           spatialSharpnessSlider_->minimum(), spatialSharpnessSlider_->maximum());
        QSignalBlocker block(spatialSharpnessSlider_);
        spatialSharpnessSlider_->setValue(sliderValue);
        OnSpatialSharpnessChanged(sliderValue);
    }
    OnSpatialSharpenToggled(spatialSharpenEnabled_);

    if (debugButton_) {
        QSignalBlocker block(debugButton_);
        debugButton_->setChecked(settings.debugView);
    }
    debugViewEnabled_ = settings.debugView;
    OnDebugViewToggled(debugViewEnabled_);

    if (focusMarkerCheckbox_) {
        QSignalBlocker block(focusMarkerCheckbox_);
        focusMarkerCheckbox_->setChecked(settings.focusMarker);
    }
    focusMarkerEnabled_ = settings.focusMarker;
    OnFocusMarkerToggled(focusMarkerEnabled_);

    if (joystickCheckbox_) {
        QSignalBlocker block(joystickCheckbox_);
        joystickCheckbox_->setChecked(settings.virtualJoystick);
    }
    virtualJoystickEnabled_ = settings.virtualJoystick;
    OnVirtualJoystickToggled(virtualJoystickEnabled_);

    if (collapseButton_) {
        QSignalBlocker block(collapseButton_);
        collapseButton_->setChecked(!settings.controlsCollapsed);
    }
    controlsCollapsed_ = settings.controlsCollapsed;
    OnControlsCollapsedToggled(collapseButton_ ? collapseButton_->isChecked() : !controlsCollapsed_);

    rotationQuarterTurns_ = settings.rotationQuarterTurns % 4;
    if (rotationQuarterTurns_ < 0) {
        rotationQuarterTurns_ += 4;
    }
    UpdateRotationUi();
}

void OpenZoomApp::SavePersistentSettings() {
    settings::PersistentSettings settings;
    settings.cameraIndex = selectedCameraIndex_;
    settings.blackWhiteEnabled = blackWhiteEnabled_;
    settings.blackWhiteThreshold = blackWhiteThreshold_;
    settings.zoomEnabled = zoomEnabled_;
    settings.zoomAmount = zoomAmount_;
    settings.zoomCenterX = zoomCenterX_;
    settings.zoomCenterY = zoomCenterY_;
    settings.blurEnabled = blurEnabled_;
    settings.blurSigma = blurSigma_;
    settings.blurRadius = blurRadius_;
    settings.temporalSmoothEnabled = temporalSmoothEnabled_;
    settings.temporalSmoothAlpha = temporalSmoothAlpha_;
    settings.spatialSharpenEnabled = spatialSharpenEnabled_;
    settings.spatialUpscaler = static_cast<int>(spatialUpscaler_);
    settings.spatialSharpness = spatialSharpness_;
    settings.debugView = debugViewEnabled_;
    settings.focusMarker = focusMarkerEnabled_;
    settings.virtualJoystick = virtualJoystickEnabled_;
    settings.controlsCollapsed = controlsCollapsed_;
    settings.rotationQuarterTurns = rotationQuarterTurns_;

    settings::Save(settingsPath_, settings);
}

void OpenZoomApp::OnFrameTick() {
    if (!cameraActive_) {
        return;
    }

    ApplyInputForces();
    MediaFrame frame;
    {
        std::scoped_lock lock(cameraMutex_);
        frame = latestFrame_;
    }

    if (frame.data.empty() || frame.width == 0 || frame.height == 0) {
        return;
    }

    if (!cpuPipeline_.ConvertFrameToBgra(frame.data,
                                         frame.subtype,
                                         frame.width,
                                         frame.height,
                                         frame.stride,
                                         frame.dataSize)) {
        return;
    }

    UINT width = frame.width;
    UINT height = frame.height;
    cpuPipeline_.RotateRawBuffer(rotationQuarterTurns_, width, height);
    if ((rotationQuarterTurns_ % 2) != 0 && renderWidget_) {
        const int targetWidth = std::max(1, renderWidget_->width());
        const int targetHeight = std::max(1, renderWidget_->height());
        if (cpuPipeline_.ResampleToFill(static_cast<UINT>(targetWidth),
                                        static_cast<UINT>(targetHeight),
                                        zoomCenterX_,
                                        zoomCenterY_)) {
            width = static_cast<UINT>(targetWidth);
            height = static_cast<UINT>(targetHeight);
        }
    }
    processedFrameWidth_ = width;
    processedFrameHeight_ = height;

    BuildCompositeAndPresent(width, height);
}
void OpenZoomApp::BuildCompositeAndPresent(UINT width, UINT height) {
    processedFrameWidth_ = width;
    processedFrameHeight_ = height;
    usingCudaLastFrame_ = false;
    if (!debugViewEnabled_ && ProcessFrameWithCuda(width, height)) {
        usingCudaLastFrame_ = true;
        UpdateProcessingStatusLabel();
        return;
    }
    processing::CpuPipelineConfig config{};
    config.enableBlackWhite = blackWhiteEnabled_;
    config.blackWhiteThreshold = blackWhiteThreshold_;
    config.enableZoom = zoomEnabled_;
    config.zoomAmount = zoomAmount_;
    config.zoomCenterX = zoomCenterX_;
    config.zoomCenterY = zoomCenterY_;
    config.enableBlur = blurEnabled_;
    config.blurRadius = std::max(0, blurRadius_);
    config.blurSigma = blurSigma_;
    config.enableTemporalSmooth = temporalSmoothEnabled_;
    config.temporalSmoothAlpha = temporalSmoothAlpha_;

    const processing::CpuPipelineOutput output =
        cpuPipeline_.BuildStages(width, height, config, debugViewEnabled_);

    if (!output.data || output.width == 0 || output.height == 0) {
        UpdateProcessingStatusLabel();
        return;
    }

    const bool cropToFill = !debugViewEnabled_;
    const float centerX = cropToFill ? zoomCenterX_ : 0.5f;
    const float centerY = cropToFill ? zoomCenterY_ : 0.5f;
    PresentFitted(output.data,
                  output.width,
                  output.height,
                  cropToFill,
                  centerX,
                  centerY);
}

void OpenZoomApp::PresentFitted(const uint8_t* data,
                                UINT srcWidth,
                                UINT srcHeight,
                                bool cropToFill,
                                float centerXNorm,
                                float centerYNorm) {
    if (!data || srcWidth == 0 || srcHeight == 0) {
        return;
    }

    if (!renderWidget_ || !renderWidget_->isPresenterReady()) {
        return;
    }

    const int targetWidthInt = std::max(1, renderWidget_->width());
    const int targetHeightInt = std::max(1, renderWidget_->height());
    const UINT targetWidth = static_cast<UINT>(targetWidthInt);
    const UINT targetHeight = static_cast<UINT>(targetHeightInt);

    presentationBuffer_.assign(static_cast<size_t>(targetWidth) * targetHeight * 4, 0);

    const float srcWidthF = static_cast<float>(srcWidth);
    const float srcHeightF = static_cast<float>(srcHeight);
    const float targetAspect = static_cast<float>(targetWidth) / static_cast<float>(targetHeight);
    const float srcAspect = srcWidthF / srcHeightF;

    float cropWidth = srcWidthF;
    float cropHeight = srcHeightF;

    if (cropToFill) {
        if (targetAspect > srcAspect) {
            cropHeight = srcWidthF / targetAspect;
            cropHeight = std::min(cropHeight, srcHeightF);
        } else {
            cropWidth = srcHeightF * targetAspect;
            cropWidth = std::min(cropWidth, srcWidthF);
        }
    }

    cropWidth = std::clamp(cropWidth, 1.0f, srcWidthF);
    cropHeight = std::clamp(cropHeight, 1.0f, srcHeightF);

    float centerX = std::clamp(centerXNorm, 0.0f, 1.0f) * (srcWidthF - 1.0f);
    float centerY = std::clamp(centerYNorm, 0.0f, 1.0f) * (srcHeightF - 1.0f);

    const float halfCropWidth = cropWidth * 0.5f;
    const float halfCropHeight = cropHeight * 0.5f;

    const float minCenterX = std::max(0.0f, halfCropWidth - 0.5f);
    const float maxCenterX = std::max(minCenterX, (srcWidthF - 1.0f) - (halfCropWidth - 0.5f));
    const float minCenterY = std::max(0.0f, halfCropHeight - 0.5f);
    const float maxCenterY = std::max(minCenterY, (srcHeightF - 1.0f) - (halfCropHeight - 0.5f));

    if (minCenterX <= maxCenterX) {
        centerX = std::clamp(centerX, minCenterX, maxCenterX);
    } else {
        centerX = (srcWidthF - 1.0f) * 0.5f;
    }

    if (minCenterY <= maxCenterY) {
        centerY = std::clamp(centerY, minCenterY, maxCenterY);
    } else {
        centerY = (srcHeightF - 1.0f) * 0.5f;
    }

    float startX = centerX - halfCropWidth + 0.5f;
    float startY = centerY - halfCropHeight + 0.5f;
    startX = std::clamp(startX, 0.0f, srcWidthF - cropWidth);
    startY = std::clamp(startY, 0.0f, srcHeightF - cropHeight);

    float scaleFactor;
    if (cropToFill) {
        scaleFactor = static_cast<float>(targetWidth) / cropWidth;
    } else {
        const float widthScale = static_cast<float>(targetWidth) / cropWidth;
        const float heightScale = static_cast<float>(targetHeight) / cropHeight;
        scaleFactor = std::min(widthScale, heightScale);
    }

    if (!(scaleFactor > 0.0f) || !std::isfinite(scaleFactor)) {
        scaleFactor = 1.0f;
    }

    UINT activeWidth = static_cast<UINT>(std::roundf(cropWidth * scaleFactor));
    UINT activeHeight = static_cast<UINT>(std::roundf(cropHeight * scaleFactor));
    activeWidth = std::clamp(activeWidth, 1u, targetWidth);
    activeHeight = std::clamp(activeHeight, 1u, targetHeight);

    const UINT offsetX = (targetWidth > activeWidth) ? (targetWidth - activeWidth) / 2 : 0;
    const UINT offsetY = (targetHeight > activeHeight) ? (targetHeight - activeHeight) / 2 : 0;

    const float stepX = cropWidth / static_cast<float>(activeWidth);
    const float stepY = cropHeight / static_cast<float>(activeHeight);
    const UINT srcStride = srcWidth * 4;
    const UINT dstStride = targetWidth * 4;

    for (UINT y = 0; y < activeHeight; ++y) {
        const float sampleY = startY + static_cast<float>(y) * stepY;
        int srcYIndex = static_cast<int>(std::lroundf(sampleY));
        srcYIndex = std::clamp(srcYIndex, 0, static_cast<int>(srcHeight) - 1);
        uint8_t* dstRow = presentationBuffer_.data() +
                          (static_cast<size_t>(offsetY + y) * dstStride) +
                          offsetX * 4;
        const uint8_t* srcRow = data + static_cast<size_t>(srcYIndex) * srcStride;
        for (UINT x = 0; x < activeWidth; ++x) {
            const float sampleX = startX + static_cast<float>(x) * stepX;
            int srcXIndex = static_cast<int>(std::lroundf(sampleX));
            srcXIndex = std::clamp(srcXIndex, 0, static_cast<int>(srcWidth) - 1);
            const uint8_t* srcPixel = srcRow + srcXIndex * 4;
            uint8_t* dstPixel = dstRow + x * 4;
            dstPixel[0] = srcPixel[0];
            dstPixel[1] = srcPixel[1];
            dstPixel[2] = srcPixel[2];
            dstPixel[3] = srcPixel[3];
        }
    }

    if (focusMarkerEnabled_ && cropToFill) {
        const float localX = (centerX - startX) / stepX;
        const float localY = (centerY - startY) / stepY;
        const float markerX = static_cast<float>(offsetX) + localX;
        const float markerY = static_cast<float>(offsetY) + localY;

        auto drawFilledCircle = [&](float cx, float cy, float radius,
                                    uint8_t b, uint8_t g, uint8_t r, uint8_t a) {
            const int minX = std::max(0, static_cast<int>(std::floor(cx - radius)));
            const int maxX = std::min(static_cast<int>(targetWidth) - 1,
                                      static_cast<int>(std::ceil(cx + radius)));
            const int minY = std::max(0, static_cast<int>(std::floor(cy - radius)));
            const int maxY = std::min(static_cast<int>(targetHeight) - 1,
                                      static_cast<int>(std::ceil(cy + radius)));
            const float radiusSq = radius * radius;
            for (int py = minY; py <= maxY; ++py) {
                const float dy = (static_cast<float>(py) + 0.5f) - cy;
                for (int px = minX; px <= maxX; ++px) {
                    const float dx = (static_cast<float>(px) + 0.5f) - cx;
                    if (dx * dx + dy * dy <= radiusSq) {
                        uint8_t* pixel = presentationBuffer_.data() +
                                         (static_cast<size_t>(py) * targetWidth + px) * 4;
                        pixel[0] = b;
                        pixel[1] = g;
                        pixel[2] = r;
                        pixel[3] = a;
                    }
                }
            }
        };

        constexpr float kMarkerRadius = 18.0f;
        drawFilledCircle(markerX, markerY, kMarkerRadius, 0, 0, 255, 255);
        constexpr float kInnerRadius = 6.0f;
        drawFilledCircle(markerX, markerY, kInnerRadius, 255, 255, 255, 255);
    }

    presenter_->Present(presentationBuffer_.data(), targetWidth, targetHeight);
    UpdateProcessingStatusLabel();
}

} // namespace openzoom

#endif // _WIN32
