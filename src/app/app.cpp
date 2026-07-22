#ifdef _WIN32

#include "openzoom/app/app.hpp"
#include "openzoom/cuda/cuda_interop.hpp"
#include "openzoom/d3d12/presenter.hpp"
#include "openzoom/app/constants.hpp"
#include "openzoom/app/interaction_controller.hpp"
#include "openzoom/ui/main_window.hpp"
#include "openzoom/ui/ai_settings_dialog.hpp"
#include <QAbstractButton>
#include <QApplication>
#include <QDesktopServices>
#include <QUrl>
#include <QByteArray>
#include <QCoreApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
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
#include <QListWidget>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QDir>
#include <QDateTime>
#include <QImage>
#include <QString>
#include <QStringList>
#include <QSizePolicy>
#include <QPaintEngine>
#include <QResizeEvent>
#include <QShowEvent>
#include <QDebug>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonObject>
#include <QMessageBox>
#include <QMetaObject>
#include <QPlainTextEdit>
#include <QTextBrowser>
#include <QTextCursor>

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

struct ViewMapping {
    UINT targetWidth{};
    UINT targetHeight{};
    UINT activeWidth{};
    UINT activeHeight{};
    UINT offsetX{};
    UINT offsetY{};
    float startX{};
    float startY{};
    float stepX{};
    float stepY{};
    float centerX{};
    float centerY{};
    float cropWidth{};
    float cropHeight{};
};

bool ComputeViewMapping(UINT srcWidth,
                        UINT srcHeight,
                        int targetWidthInt,
                        int targetHeightInt,
                        float centerXNorm,
                        float centerYNorm,
                        bool cropToFill,
                        ViewMapping& out)
{
    if (srcWidth == 0 || srcHeight == 0 || targetWidthInt <= 0 || targetHeightInt <= 0) {
        return false;
    }

    const float srcWidthF = static_cast<float>(srcWidth);
    const float srcHeightF = static_cast<float>(srcHeight);
    out.targetWidth = static_cast<UINT>(std::max(1, targetWidthInt));
    out.targetHeight = static_cast<UINT>(std::max(1, targetHeightInt));

    const float targetAspect = static_cast<float>(out.targetWidth) / static_cast<float>(out.targetHeight);
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
        scaleFactor = static_cast<float>(out.targetWidth) / cropWidth;
    } else {
        const float widthScale = static_cast<float>(out.targetWidth) / cropWidth;
        const float heightScale = static_cast<float>(out.targetHeight) / cropHeight;
        scaleFactor = std::min(widthScale, heightScale);
    }

    if (!(scaleFactor > 0.0f) || !std::isfinite(scaleFactor)) {
        scaleFactor = 1.0f;
    }

    UINT activeWidth = static_cast<UINT>(std::roundf(cropWidth * scaleFactor));
    UINT activeHeight = static_cast<UINT>(std::roundf(cropHeight * scaleFactor));
    activeWidth = std::clamp(activeWidth, 1u, out.targetWidth);
    activeHeight = std::clamp(activeHeight, 1u, out.targetHeight);

    const UINT offsetX = (out.targetWidth > activeWidth) ? (out.targetWidth - activeWidth) / 2 : 0;
    const UINT offsetY = (out.targetHeight > activeHeight) ? (out.targetHeight - activeHeight) / 2 : 0;

    const float stepX = cropWidth / static_cast<float>(activeWidth);
    const float stepY = cropHeight / static_cast<float>(activeHeight);

    out.activeWidth = activeWidth;
    out.activeHeight = activeHeight;
    out.offsetX = offsetX;
    out.offsetY = offsetY;
    out.startX = startX;
    out.startY = startY;
    out.stepX = stepX;
    out.stepY = stepY;
    out.centerX = centerX;
    out.centerY = centerY;
    out.cropWidth = cropWidth;
    out.cropHeight = cropHeight;
    return true;
}

} // namespace

constexpr int kPresetIdRole = Qt::UserRole + 1;

QString MakeCustomEntityId(const QString& prefix)
{
    return QStringLiteral("%1-%2").arg(prefix).arg(QDateTime::currentMSecsSinceEpoch());
}




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
    assistiveOverlay_ = new AssistiveOverlay(renderWidget_);
    assistiveRuntime_ = std::make_unique<AssistiveRuntime>(this);
    connect(assistiveRuntime_.get(), &AssistiveRuntime::OverlayUpdated,
            this, &OpenZoomApp::OnAssistiveOverlayUpdated);
    connect(assistiveOverlay_, &AssistiveOverlay::Dismissed,
            assistiveRuntime_.get(), &AssistiveRuntime::DismissOverlay);
    connect(assistiveOverlay_, &AssistiveOverlay::ReadAloudRequested,
            assistiveRuntime_.get(), &AssistiveRuntime::ReadAloud);
    connect(assistiveOverlay_, &AssistiveOverlay::QuestionSubmitted,
            this, &OpenZoomApp::SubmitFloatingAssistantPrompt);
    interactionController_ = std::make_unique<InteractionController>(*this);
    settingsPath_ = settings::ResolveSettingsPath();
    if (auto loadedSettings = settings::Load(settingsPath_)) {
        persistentSettings_ = *loadedSettings;
    } else {
        persistentSettings_.selectedPresetId = settings::DefaultPresetId();
        if (auto defaultConfig = settings::ResolveConfigForPreset(persistentSettings_.selectedPresetId,
                                                                  persistentSettings_.customConfigs,
                                                                  persistentSettings_.customPresets)) {
            persistentSettings_.currentConfig = *defaultConfig;
        } else if (!settings::BuiltInConfigs().empty()) {
            persistentSettings_.currentConfig = settings::BuiltInConfigs().front();
        }
    }
    configTrackingSuspended_ = true;
    connect(qtApp_, &QCoreApplication::aboutToQuit, this, [this]() { SavePersistentSettings(); });
    cameraCombo_ = mainWindow_->cameraCombo();
    presetList_ = mainWindow_->presetList();
    presetDescriptionLabel_ = mainWindow_->presetDescriptionLabel();
    promotePresetButton_ = mainWindow_->promotePresetButton();
    cameraModesList_ = mainWindow_->cameraModesList();
    bwCheckbox_ = mainWindow_->blackWhiteCheckbox();
    bwSlider_ = mainWindow_->blackWhiteSlider();
    zoomCheckbox_ = mainWindow_->zoomCheckbox();
    zoomSlider_ = mainWindow_->zoomSlider();
    debugButton_ = mainWindow_->debugButton();
    capturePhotoButton_ = mainWindow_->capturePhotoButton();
    recordButton_ = mainWindow_->recordButton();
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
    ocrAssistCheckbox_ = mainWindow_->ocrAssistCheckbox();
    vlmAssistCheckbox_ = mainWindow_->vlmAssistCheckbox();
    assistiveOverlayCheckbox_ = mainWindow_->assistiveOverlayCheckbox();
    spatialSharpenCheckbox_ = mainWindow_->spatialSharpenCheckbox();
    spatialBackendCombo_ = mainWindow_->spatialBackendCombo();
    spatialSharpnessSlider_ = mainWindow_->spatialSharpnessSlider();
    spatialSharpnessValueLabel_ = mainWindow_->spatialSharpnessValueLabel();
    processingStatusLabel_ = mainWindow_->processingStatusLabel();
    stabilizationCheckbox_ = mainWindow_->stabilizationCheckbox();
    stabilizationStrengthSlider_ = mainWindow_->stabilizationStrengthSlider();
    keystoneCheckbox_ = mainWindow_->keystoneCheckbox();
    autoContrastCheckbox_ = mainWindow_->autoContrastCheckbox();
    autoContrastStrengthSlider_ = mainWindow_->autoContrastStrengthSlider();
    displayColorCombo_ = mainWindow_->displayColorCombo();
    contrastSlider_ = mainWindow_->contrastSlider();
    brightnessSlider_ = mainWindow_->brightnessSlider();
    explainNowButton_ = mainWindow_->explainNowButton();
    readTextButton_ = mainWindow_->readTextButton();
    aiSettingsButton_ = mainWindow_->aiSettingsButton();
    openNotesButton_ = mainWindow_->openNotesButton();
    assistantConnectionLabel_ = mainWindow_->assistantConnectionLabel();
    assistantUsageLabel_ = mainWindow_->assistantUsageLabel();
    assistantConnectButton_ = mainWindow_->assistantConnectButton();
    assistantTranscript_ = mainWindow_->assistantTranscript();
    assistantPromptEdit_ = mainWindow_->assistantPromptEdit();
    assistantAttachFrameCheckbox_ = mainWindow_->assistantAttachFrameCheckbox();
    assistantSendButton_ = mainWindow_->assistantSendButton();
    assistantStopButton_ = mainWindow_->assistantStopButton();
    assistantNewButton_ = mainWindow_->assistantNewButton();
    assistantHistoryList_ = mainWindow_->assistantHistoryList();
    assistantRenameButton_ = mainWindow_->assistantRenameButton();
    assistantExportButton_ = mainWindow_->assistantExportButton();
    assistantDeleteButton_ = mainWindow_->assistantDeleteButton();

    joystickOverlay_ = new JoystickOverlay(renderWidget_);
    connect(joystickOverlay_, &JoystickOverlay::JoystickChanged,
            this, [this](float x, float y) {
                if (interactionController_) {
                    interactionController_->SetJoystickAxes(x, y);
                }
            });
    UpdateJoystickVisibility();
    assistiveAnalysisTimer_.invalidate();

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
    stabilizationEnabled_ = stabilizationCheckbox_ ? stabilizationCheckbox_->isChecked() : false;
    if (stabilizationStrengthSlider_) {
        stabilizationStrength_ = std::clamp(static_cast<float>(stabilizationStrengthSlider_->value()) / 100.0f, 0.0f, 0.98f);
    }
    keystoneEnabled_ = keystoneCheckbox_ ? keystoneCheckbox_->isChecked() : false;
    autoContrastEnabled_ = autoContrastCheckbox_ ? autoContrastCheckbox_->isChecked() : false;
    if (autoContrastStrengthSlider_) {
        autoContrastStrength_ = std::clamp(static_cast<float>(autoContrastStrengthSlider_->value()) / 100.0f, 0.0f, 1.0f);
    }
    if (displayColorCombo_) {
        displayColorMode_ = std::clamp(displayColorCombo_->currentIndex(), 0, 4);
    }
    if (contrastSlider_) {
        contrast_ = std::clamp(static_cast<float>(contrastSlider_->value()) / 100.0f, 0.25f, 4.0f);
    }
    if (brightnessSlider_) {
        brightness_ = std::clamp(static_cast<float>(brightnessSlider_->value()) / 100.0f, -1.0f, 1.0f);
    }

    PopulateCameraCombo();
    PopulatePresetList();

    connect(cameraCombo_, &QComboBox::currentIndexChanged,
            this, &OpenZoomApp::OnCameraSelectionChanged);
    if (presetList_) {
        connect(presetList_, &QListWidget::currentItemChanged,
                this, &OpenZoomApp::OnPresetSelectionChanged);
    }
    if (promotePresetButton_) {
        connect(promotePresetButton_, &QPushButton::clicked,
                this, [this]() { PromoteCurrentConfigToPreset(); });
    }
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
    if (capturePhotoButton_) {
        connect(capturePhotoButton_, &QPushButton::clicked, this, [this]() {
            if (usingCudaLastFrame_ && cudaSharedTexture_ && presenter_) {
                if (presenter_->ReadbackTexture(cudaSharedTexture_.Get(),
                                                processedFrameWidth_,
                                                processedFrameHeight_,
                                                recordingBuffer_)) {
                    CaptureSnapshot(recordingBuffer_.data(), processedFrameWidth_, processedFrameHeight_);
                    return;
                }
            }
            if (!presentationBuffer_.empty() && presentationWidth_ > 0 && presentationHeight_ > 0) {
                CaptureSnapshot(presentationBuffer_.data(), presentationWidth_, presentationHeight_);
            } else {
                qWarning() << "Capture skipped: no frame available";
            }
        });
    }
    if (recordButton_) {
        recordButton_->setCheckable(true);
        connect(recordButton_, &QPushButton::toggled, this, [this](bool checked) {
            if (checked) {
                // Start
                recordingFrameCount_ = 0;
                recordingTimer_.restart();
                recording_ = true;
                recordButton_->setText(QStringLiteral("Stop"));
            } else {
                recording_ = false;
                videoRecorder_.Stop();
                recordButton_->setText(QStringLiteral("Record"));
            }
        });
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
    if (ocrAssistCheckbox_) {
        connect(ocrAssistCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnOcrAssistToggled);
    }
    if (vlmAssistCheckbox_) {
        connect(vlmAssistCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnVlmAssistToggled);
    }
    if (assistiveOverlayCheckbox_) {
        connect(assistiveOverlayCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnAssistiveOverlayToggled);
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
    if (stabilizationCheckbox_) {
        connect(stabilizationCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnStabilizationToggled);
    }
    if (stabilizationStrengthSlider_) {
        connect(stabilizationStrengthSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnStabilizationStrengthChanged);
    }
    if (keystoneCheckbox_) {
        connect(keystoneCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnKeystoneToggled);
    }
    if (autoContrastCheckbox_) {
        connect(autoContrastCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnAutoContrastToggled);
    }
    if (autoContrastStrengthSlider_) {
        connect(autoContrastStrengthSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnAutoContrastStrengthChanged);
    }
    if (displayColorCombo_) {
        connect(displayColorCombo_, &QComboBox::currentIndexChanged,
                this, &OpenZoomApp::OnDisplayColorModeChanged);
    }
    if (contrastSlider_) {
        connect(contrastSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnContrastChanged);
    }
    if (brightnessSlider_) {
        connect(brightnessSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnBrightnessChanged);
    }
    // Simple/Advanced mode buttons: the MainWindow wires the page switch
    // internally; here we only track the state for persistence and expand the
    // advanced tuning panel when the advanced page is entered.
    if (QAbstractButton* simpleButton = mainWindow_->simpleModeButton()) {
        connect(simpleButton, &QAbstractButton::toggled, this, [this](bool checked) {
            if (checked) {
                simpleUiMode_ = true;
                persistentSettings_.simpleUiMode = true;
            }
        });
    }
    if (QAbstractButton* advancedButton = mainWindow_->advancedModeButton()) {
        connect(advancedButton, &QAbstractButton::toggled, this, [this](bool checked) {
            if (checked) {
                simpleUiMode_ = false;
                persistentSettings_.simpleUiMode = false;
                if (collapseButton_ && !collapseButton_->isChecked()) {
                    collapseButton_->setChecked(true);
                }
            }
        });
    }
    if (explainNowButton_) {
        connect(explainNowButton_, &QPushButton::clicked,
                this, [this]() {
                    if (assistiveRuntime_ && assistiveRuntime_->IsCodexTurnActive()) {
                        assistiveRuntime_->StopAssistant();
                    } else {
                        SubmitOnDemandAnalysis(false, true);
                    }
                });
    }
    if (readTextButton_) {
        connect(readTextButton_, &QPushButton::clicked,
                this, [this]() { SubmitOnDemandAnalysis(true, false); });
    }
    if (aiSettingsButton_) {
        connect(aiSettingsButton_, &QPushButton::clicked,
                this, [this]() { OpenAiSettingsDialog(); });
    }
    if (openNotesButton_) {
        connect(openNotesButton_, &QPushButton::clicked,
                this, [this]() { OpenNotesFile(); });
    }
    if (assistantConnectButton_) {
        connect(assistantConnectButton_, &QPushButton::clicked,
                this, [this]() { assistiveRuntime_->StartCodexLogin(); });
    }
    if (assistantSendButton_) {
        connect(assistantSendButton_, &QPushButton::clicked,
                this, [this]() { SubmitAssistantPrompt(); });
    }
    if (assistantStopButton_) {
        connect(assistantStopButton_, &QPushButton::clicked,
                this, [this]() { assistiveRuntime_->StopAssistant(); });
    }
    if (assistantNewButton_) {
        connect(assistantNewButton_, &QPushButton::clicked, this, [this]() {
            if (assistiveRuntime_->IsCodexTurnActive()) {
                return;
            }
            currentAssistantThreadId_.clear();
            pendingAssistantPrompt_.clear();
            assistantResponseOpen_ = false;
            assistantResponseReceivedText_ = false;
            assistantTranscript_->clear();
            assistantHistoryList_->clearSelection();
            assistantPromptEdit_->setFocus();
        });
    }
    if (assistantHistoryList_) {
        connect(assistantHistoryList_, &QListWidget::itemDoubleClicked,
                this, [this](QListWidgetItem*) { LoadSelectedAssistantConversation(); });
        connect(assistantHistoryList_, &QListWidget::itemActivated,
                this, [this](QListWidgetItem*) { LoadSelectedAssistantConversation(); });
    }
    if (assistantRenameButton_) {
        connect(assistantRenameButton_, &QPushButton::clicked, this, [this]() {
            QListWidgetItem* item = assistantHistoryList_->currentItem();
            if (!item) {
                return;
            }
            const QString threadId = item->data(Qt::UserRole).toString();
            const QString oldName = item->data(Qt::UserRole + 1).toString();
            bool accepted = false;
            const QString name = QInputDialog::getText(mainWindow_.get(),
                                                       QStringLiteral("Rename Conversation"),
                                                       QStringLiteral("Name:"),
                                                       QLineEdit::Normal,
                                                       oldName,
                                                       &accepted).trimmed();
            if (accepted && !name.isEmpty()) {
                assistiveRuntime_->RenameAssistantConversation(threadId, name);
            }
        });
    }
    if (assistantExportButton_) {
        connect(assistantExportButton_, &QPushButton::clicked, this, [this]() {
            if (!assistantTranscript_ || assistantTranscript_->toPlainText().trimmed().isEmpty()) {
                return;
            }
            const QString suggested = QDir(EnsureOutputSubdir(QStringLiteral("assistant")))
                                          .filePath(QStringLiteral("OpenZoom_Assistant_%1.txt")
                                                        .arg(QDateTime::currentDateTime().toString(
                                                            QStringLiteral("yyyyMMdd_HHmmss"))));
            const QString path = QFileDialog::getSaveFileName(mainWindow_.get(),
                                                               QStringLiteral("Export Conversation"),
                                                               suggested,
                                                               QStringLiteral("Text Files (*.txt);;All Files (*)"));
            if (path.isEmpty()) {
                return;
            }
            QFile file(path);
            if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                file.write(assistantTranscript_->toPlainText().toUtf8());
            }
        });
    }
    if (assistantDeleteButton_) {
        connect(assistantDeleteButton_, &QPushButton::clicked, this, [this]() {
            QListWidgetItem* item = assistantHistoryList_->currentItem();
            if (!item) {
                return;
            }
            const QString threadId = item->data(Qt::UserRole).toString();
            if (QMessageBox::question(mainWindow_.get(),
                                      QStringLiteral("Delete Conversation"),
                                      QStringLiteral("Permanently delete this OpenZoom assistant conversation?"))
                == QMessageBox::Yes) {
                assistiveRuntime_->DeleteAssistantConversation(threadId);
            }
        });
    }

    connect(assistiveRuntime_.get(), &AssistiveRuntime::CodexServerStateChanged,
            this, [this](bool ready, const QString& status) {
                codexReady_ = ready;
                assistantConnectionLabel_->setText(status);
                SetAssistantBusy(assistiveRuntime_->IsCodexTurnActive());
            });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::CodexAccountChanged,
            this, [this](bool signedIn, const QString& label, const QString& planType) {
                codexSignedIn_ = signedIn;
                const QString plan = planType.trimmed();
                assistantConnectionLabel_->setText(plan.isEmpty()
                                                       ? label
                                                       : QStringLiteral("%1 (%2)").arg(label, plan));
                assistantConnectButton_->setText(signedIn ? QStringLiteral("Reconnect ChatGPT")
                                                          : QStringLiteral("Connect ChatGPT"));
                SetAssistantBusy(assistiveRuntime_->IsCodexTurnActive());
            });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::CodexModelsChanged,
            this, [this](const QStringList&, const QString& selectedModel) {
                if (!selectedModel.isEmpty()) {
                    assistantConnectionLabel_->setToolTip(
                        QStringLiteral("Vision model: %1").arg(selectedModel));
                }
            });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::CodexRateLimitChanged,
            this, [this](const QString& summary) { assistantUsageLabel_->setText(summary); });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::CodexLoginUrlReady,
            this, [](const QUrl& url) { QDesktopServices::openUrl(url); });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::AssistantConversationCreated,
            this, [this](const QJsonObject& thread) {
                settings::CodexConversation conversation;
                conversation.threadId = thread.value(QStringLiteral("id")).toString();
                conversation.preview = pendingAssistantPrompt_.left(160);
                conversation.title = pendingAssistantPrompt_.simplified().left(60);
                if (conversation.title.isEmpty()) {
                    conversation.title = QStringLiteral("OpenZoom Assistant");
                }
                conversation.createdAt = thread.value(QStringLiteral("createdAt")).toInteger(
                    QDateTime::currentSecsSinceEpoch());
                conversation.updatedAt = conversation.createdAt;
                const auto existing = std::find_if(
                    persistentSettings_.codexConversations.begin(),
                    persistentSettings_.codexConversations.end(),
                    [&conversation](const settings::CodexConversation& candidate) {
                        return candidate.threadId == conversation.threadId;
                    });
                if (existing == persistentSettings_.codexConversations.end()) {
                    persistentSettings_.codexConversations.push_back(conversation);
                }
                currentAssistantThreadId_ = conversation.threadId;
                PopulateAssistantHistory();
                SavePersistentSettings();
            });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::AssistantTranscriptLoaded,
            this, [this](const QString& threadId, const QJsonArray& messages) {
                currentAssistantThreadId_ = threadId;
                assistantTranscript_->clear();
                for (const QJsonValue& value : messages) {
                    const QJsonObject message = value.toObject();
                    const QString speaker = message.value(QStringLiteral("role")).toString()
                                                    == QStringLiteral("user")
                                                ? QStringLiteral("You")
                                                : QStringLiteral("OpenZoom Assistant");
                    AppendAssistantMessage(speaker, message.value(QStringLiteral("text")).toString());
                }
                assistantResponseOpen_ = false;
                assistantResponseReceivedText_ = false;
            });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::AssistantConversationRenamed,
            this, [this](const QString& threadId, const QString& name) {
                for (settings::CodexConversation& conversation : persistentSettings_.codexConversations) {
                    if (conversation.threadId == threadId) {
                        conversation.title = name;
                        conversation.updatedAt = QDateTime::currentSecsSinceEpoch();
                        break;
                    }
                }
                PopulateAssistantHistory();
                SavePersistentSettings();
            });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::AssistantConversationDeleted,
            this, [this](const QString& threadId) {
                std::erase_if(persistentSettings_.codexConversations,
                              [&threadId](const settings::CodexConversation& conversation) {
                                  return conversation.threadId == threadId;
                              });
                if (currentAssistantThreadId_ == threadId) {
                    currentAssistantThreadId_.clear();
                    assistantTranscript_->clear();
                }
                PopulateAssistantHistory();
                SavePersistentSettings();
            });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::AssistantTurnStarted,
            this, [this](const QString&, const QString&, bool persistent) {
                SetAssistantBusy(true);
                if (!persistent && explainNowButton_) {
                    explainNowButton_->setText(QStringLiteral("Stop"));
                }
            });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::AssistantTextDelta,
            this, [this](const QString& threadId, const QString&, const QString& delta) {
                if (!assistantResponseOpen_ ||
                    (!currentAssistantThreadId_.isEmpty() && threadId != currentAssistantThreadId_)) {
                    return;
                }
                QTextCursor cursor = assistantTranscript_->textCursor();
                cursor.movePosition(QTextCursor::End);
                cursor.insertText(delta);
                assistantTranscript_->setTextCursor(cursor);
                assistantTranscript_->ensureCursorVisible();
                assistantResponseReceivedText_ = true;
            });
    connect(assistiveRuntime_.get(), &AssistiveRuntime::AssistantTurnFinished,
            this,
            [this](const QString& threadId,
                   const QString&,
                   const QString& text,
                   const QString& error,
                   bool interrupted,
                   bool persistent) {
                SetAssistantBusy(false);
                if (explainNowButton_) {
                    explainNowButton_->setText(QStringLiteral("Explain"));
                }
                if (!persistent) {
                    return;
                }
                if (assistantResponseOpen_) {
                    QTextCursor cursor = assistantTranscript_->textCursor();
                    cursor.movePosition(QTextCursor::End);
                    if (!assistantResponseReceivedText_ && !text.trimmed().isEmpty()) {
                        cursor.insertText(text.trimmed());
                    }
                    if (!error.trimmed().isEmpty()) {
                        cursor.insertText(QStringLiteral("\n%1").arg(error.trimmed()));
                    } else if (interrupted) {
                        cursor.insertText(QStringLiteral("\nStopped."));
                    }
                    cursor.insertText(QStringLiteral("\n\n"));
                    assistantTranscript_->setTextCursor(cursor);
                }
                assistantResponseOpen_ = false;
                assistantResponseReceivedText_ = false;
                for (settings::CodexConversation& conversation : persistentSettings_.codexConversations) {
                    if (conversation.threadId == threadId) {
                        conversation.updatedAt = QDateTime::currentSecsSinceEpoch();
                        break;
                    }
                }
                PopulateAssistantHistory();
                SavePersistentSettings();
            });

    ApplyPersistentSettings(persistentSettings_);
    PopulateAssistantHistory();

    UpdateBlurUiLabels();
    UpdateTemporalSmoothUi();
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
    UpdateRotationUi();
    UpdatePresetDescription();
    UpdateAssistiveRuntimeState();

    frameTimer_ = new QTimer(this);
    connect(frameTimer_, &QTimer::timeout, this, &OpenZoomApp::OnFrameTick);
    frameTimer_->start(16);

    mainWindow_->show();

    int initialCameraIndex = 0;
    const int candidate = persistentSettings_.cameraIndex;
    if (candidate >= 0 && static_cast<size_t>(candidate) < cameras_.size()) {
        initialCameraIndex = candidate;
    }

    if (!cameras_.empty()) {
        initialCameraIndex = std::clamp(initialCameraIndex, 0, static_cast<int>(cameras_.size()) - 1);
        {
            QSignalBlocker blocker(cameraCombo_);
            cameraCombo_->setCurrentIndex(initialCameraIndex);
        }
        RefreshCameraModesList(static_cast<size_t>(initialCameraIndex));
        StartCameraCapture(static_cast<size_t>(initialCameraIndex));
    }
}

OpenZoomApp::~OpenZoomApp() {
    SavePersistentSettings();
    videoRecorder_.Stop();
    if (frameTimer_) {
        frameTimer_->stop();
        delete frameTimer_;
        frameTimer_ = nullptr;
    }
    StopCameraCapture();
    mediaCapture_.Shutdown();

    interactionController_.reset();
    assistiveRuntime_.reset();
    mainWindow_.reset();
    renderWidget_ = nullptr;
    assistiveOverlay_ = nullptr;
    joystickOverlay_ = nullptr;
    if (presenter_) {
        presenter_->WaitForIdle();
    }
    cudaSurface_.reset();
    cudaSharedTexture_.Reset();
    presenter_.reset();

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

settings::AdvancedConfig OpenZoomApp::CaptureCurrentAdvancedConfig() const
{
    settings::AdvancedConfig config;
    config.id = persistentSettings_.currentConfig.id.isEmpty() ? QStringLiteral("current-live") : persistentSettings_.currentConfig.id;
    config.name = persistentSettings_.currentConfig.name.isEmpty() ? QStringLiteral("Current Setup") : persistentSettings_.currentConfig.name;
    config.description = persistentSettings_.currentConfig.description.isEmpty()
                             ? QStringLiteral("Live configuration derived from quick mode and advanced tuning.")
                             : persistentSettings_.currentConfig.description;
    config.blackWhiteEnabled = blackWhiteEnabled_;
    config.blackWhiteThreshold = blackWhiteThreshold_;
    config.zoomEnabled = zoomEnabled_;
    config.zoomAmount = zoomAmount_;
    config.zoomCenterX = zoomCenterX_;
    config.zoomCenterY = zoomCenterY_;
    config.blurEnabled = blurEnabled_;
    config.blurSigma = blurSigma_;
    config.blurRadius = blurRadius_;
    config.temporalSmoothEnabled = temporalSmoothEnabled_;
    config.temporalSmoothAlpha = temporalSmoothAlpha_;
    config.spatialSharpenEnabled = spatialSharpenEnabled_;
    config.spatialUpscaler = static_cast<int>(spatialUpscaler_);
    config.spatialSharpness = spatialSharpness_;
    config.debugView = debugViewEnabled_;
    config.focusMarker = focusMarkerEnabled_;
    config.ocrAssistEnabled = ocrAssistEnabled_;
    config.vlmAssistEnabled = vlmAssistEnabled_;
    config.assistiveOverlayEnabled = assistiveOverlayEnabled_;
    config.stabilizationEnabled = stabilizationEnabled_;
    config.stabilizationStrength = stabilizationStrength_;
    config.displayColorMode = displayColorMode_;
    config.contrast = contrast_;
    config.brightness = brightness_;
    config.keystoneEnabled = keystoneEnabled_;
    config.autoContrastEnabled = autoContrastEnabled_;
    config.autoContrastStrength = autoContrastStrength_;
    return config;
}

void OpenZoomApp::PopulatePresetList()
{
    if (!presetList_) {
        return;
    }

    presetSelectionSyncSuspended_ = true;
    presetList_->clear();

    auto appendPreset = [this](const settings::PresetDefinition& preset) {
        auto* item = new QListWidgetItem(preset.name);
        item->setData(kPresetIdRole, preset.id);
        item->setToolTip(preset.description);
        presetList_->addItem(item);
    };

    for (const settings::PresetDefinition& preset : settings::BuiltInPresets()) {
        appendPreset(preset);
    }
    for (const settings::PresetDefinition& preset : persistentSettings_.customPresets) {
        appendPreset(preset);
    }

    presetSelectionSyncSuspended_ = false;
}

void OpenZoomApp::RefreshPresetSelection(bool preserveCurrentSelection)
{
    QString matchedPresetId;
    const settings::AdvancedConfig current = CaptureCurrentAdvancedConfig();

    if (preserveCurrentSelection &&
        settings::ResolveConfigForPreset(persistentSettings_.selectedPresetId,
                                         persistentSettings_.customConfigs,
                                         persistentSettings_.customPresets)) {
        matchedPresetId = persistentSettings_.selectedPresetId;
    }

    auto maybeMatch = [&](const settings::PresetDefinition& preset) {
        auto config = settings::ResolveConfigForPreset(preset.id,
                                                       persistentSettings_.customConfigs,
                                                       persistentSettings_.customPresets);
        if (config && settings::AreConfigsEquivalent(current, *config)) {
            matchedPresetId = preset.id;
            return true;
        }
        return false;
    };

    if (matchedPresetId.isEmpty()) {
        for (const settings::PresetDefinition& preset : settings::BuiltInPresets()) {
            if (maybeMatch(preset)) {
                break;
            }
        }
    }
    if (matchedPresetId.isEmpty()) {
        for (const settings::PresetDefinition& preset : persistentSettings_.customPresets) {
            if (maybeMatch(preset)) {
                break;
            }
        }
    }

    persistentSettings_.selectedPresetId = matchedPresetId;
    if (!presetList_) {
        return;
    }

    presetSelectionSyncSuspended_ = true;
    QListWidgetItem* matchedItem = nullptr;
    for (int row = 0; row < presetList_->count(); ++row) {
        QListWidgetItem* item = presetList_->item(row);
        if (item && item->data(kPresetIdRole).toString() == matchedPresetId) {
            matchedItem = item;
            break;
        }
    }
    if (matchedItem) {
        presetList_->setCurrentItem(matchedItem);
    } else {
        presetList_->clearSelection();
        presetList_->setCurrentItem(nullptr);
    }
    presetSelectionSyncSuspended_ = false;
}

void OpenZoomApp::UpdatePresetDescription()
{
    if (!presetDescriptionLabel_) {
        return;
    }

    QString text;
    const QString presetId = persistentSettings_.selectedPresetId;
    if (!presetId.isEmpty()) {
        if (const settings::PresetDefinition* preset =
                settings::FindPresetById(presetId, persistentSettings_.customPresets)) {
            text = QStringLiteral("%1\n%2").arg(preset->name, preset->description);
        }
    }

    if (text.isEmpty()) {
        text = QStringLiteral("Custom configuration from Advanced Tuning. Save it as a quick option when it feels right.");
    }

    QString assistiveText = QStringLiteral("Assistive hooks: off");
    if (ocrAssistEnabled_ && vlmAssistEnabled_) {
        assistiveText = QStringLiteral("Assistive hooks: OCR + Scene Explain");
    } else if (ocrAssistEnabled_) {
        assistiveText = QStringLiteral("Assistive hooks: OCR");
    } else if (vlmAssistEnabled_) {
        assistiveText = QStringLiteral("Assistive hooks: Scene Explain");
    }
    if ((ocrAssistEnabled_ || vlmAssistEnabled_) && assistiveOverlayEnabled_) {
        assistiveText.append(QStringLiteral(" with overlay"));
    }

    presetDescriptionLabel_->setText(text + QStringLiteral("\n") + assistiveText);
}

void OpenZoomApp::SyncCurrentConfigToPersistence(bool preservePresetSelection)
{
    if (configTrackingSuspended_) {
        return;
    }
    persistentSettings_.currentConfig = CaptureCurrentAdvancedConfig();
    if (!preservePresetSelection) {
        RefreshPresetSelection();
    }
    if (persistentSettings_.selectedPresetId.isEmpty()) {
        persistentSettings_.currentConfig.id = QStringLiteral("current-live");
        persistentSettings_.currentConfig.name = QStringLiteral("Current Setup");
        persistentSettings_.currentConfig.description =
            QStringLiteral("Live configuration derived from quick mode and advanced tuning.");
    } else if (auto config = settings::ResolveConfigForPreset(persistentSettings_.selectedPresetId,
                                                              persistentSettings_.customConfigs,
                                                              persistentSettings_.customPresets)) {
        persistentSettings_.currentConfig.id = config->id;
        persistentSettings_.currentConfig.name = config->name;
        persistentSettings_.currentConfig.description = config->description;
    }
    UpdatePresetDescription();
}

void OpenZoomApp::ApplyAdvancedConfig(const settings::AdvancedConfig& config)
{
    configTrackingSuspended_ = true;

    persistentSettings_.currentConfig = config;

    if (bwCheckbox_) {
        QSignalBlocker block(bwCheckbox_);
        bwCheckbox_->setChecked(config.blackWhiteEnabled);
    }
    blackWhiteEnabled_ = config.blackWhiteEnabled;
    blackWhiteThreshold_ = std::clamp(config.blackWhiteThreshold, 0.0f, 1.0f);
    if (bwSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(blackWhiteThreshold_ * 255.0f)),
                                           bwSlider_->minimum(), bwSlider_->maximum());
        QSignalBlocker block(bwSlider_);
        bwSlider_->setValue(sliderValue);
    }
    OnBlackWhiteToggled(blackWhiteEnabled_);
    OnBlackWhiteThresholdChanged(static_cast<int>(std::round(blackWhiteThreshold_ * 255.0f)));

    if (zoomCheckbox_) {
        QSignalBlocker block(zoomCheckbox_);
        zoomCheckbox_->setChecked(config.zoomEnabled);
    }
    zoomEnabled_ = config.zoomEnabled;
    zoomAmount_ = std::clamp(config.zoomAmount, 1.0f, static_cast<float>(kZoomSliderMaxMultiplier));
    if (zoomSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(zoomAmount_ * kZoomSliderScale)),
                                           zoomSlider_->minimum(), zoomSlider_->maximum());
        QSignalBlocker block(zoomSlider_);
        zoomSlider_->setValue(sliderValue);
    }
    OnZoomAmountChanged(static_cast<int>(std::round(zoomAmount_ * kZoomSliderScale)));
    SetZoomCenter(config.zoomCenterX, config.zoomCenterY, true);
    OnZoomToggled(zoomEnabled_);

    if (blurCheckbox_) {
        QSignalBlocker block(blurCheckbox_);
        blurCheckbox_->setChecked(config.blurEnabled);
    }
    blurEnabled_ = config.blurEnabled;
    blurSigma_ = std::clamp(config.blurSigma, kBlurSigmaStep, static_cast<float>(kBlurSigmaSliderMax) * kBlurSigmaStep);
    blurRadius_ = SnapBlurRadius(config.blurRadius);
    if (blurSigmaSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(blurSigma_ / kBlurSigmaStep)),
                                           blurSigmaSlider_->minimum(), blurSigmaSlider_->maximum());
        QSignalBlocker block(blurSigmaSlider_);
        blurSigmaSlider_->setValue(sliderValue);
    }
    if (blurRadiusSlider_) {
        QSignalBlocker block(blurRadiusSlider_);
        blurRadiusSlider_->setValue(blurRadius_);
    }
    OnBlurToggled(blurEnabled_);
    OnBlurSigmaChanged(static_cast<int>(std::round(blurSigma_ / kBlurSigmaStep)));
    OnBlurRadiusChanged(blurRadius_);

    if (temporalSmoothCheckbox_) {
        QSignalBlocker block(temporalSmoothCheckbox_);
        temporalSmoothCheckbox_->setChecked(config.temporalSmoothEnabled);
    }
    temporalSmoothEnabled_ = config.temporalSmoothEnabled;
    temporalSmoothAlpha_ = std::clamp(config.temporalSmoothAlpha, 0.0f, 1.0f);
    if (temporalSmoothSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(temporalSmoothAlpha_ * 100.0f)),
                                           temporalSmoothSlider_->minimum(), temporalSmoothSlider_->maximum());
        QSignalBlocker block(temporalSmoothSlider_);
        temporalSmoothSlider_->setValue(sliderValue);
    }
    OnTemporalSmoothToggled(temporalSmoothEnabled_);
    OnTemporalSmoothStrengthChanged(static_cast<int>(std::round(temporalSmoothAlpha_ * 100.0f)));

    if (stabilizationCheckbox_) {
        QSignalBlocker block(stabilizationCheckbox_);
        stabilizationCheckbox_->setChecked(config.stabilizationEnabled);
    }
    stabilizationEnabled_ = config.stabilizationEnabled;
    stabilizationStrength_ = std::clamp(config.stabilizationStrength, 0.0f, 0.98f);
    if (stabilizationStrengthSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(stabilizationStrength_ * 100.0f)),
                                           stabilizationStrengthSlider_->minimum(), stabilizationStrengthSlider_->maximum());
        QSignalBlocker block(stabilizationStrengthSlider_);
        stabilizationStrengthSlider_->setValue(sliderValue);
    }
    OnStabilizationToggled(stabilizationEnabled_);
    OnStabilizationStrengthChanged(static_cast<int>(std::round(stabilizationStrength_ * 100.0f)));

    if (keystoneCheckbox_) {
        QSignalBlocker block(keystoneCheckbox_);
        keystoneCheckbox_->setChecked(config.keystoneEnabled);
    }
    keystoneEnabled_ = config.keystoneEnabled;
    OnKeystoneToggled(keystoneEnabled_);

    if (autoContrastCheckbox_) {
        QSignalBlocker block(autoContrastCheckbox_);
        autoContrastCheckbox_->setChecked(config.autoContrastEnabled);
    }
    autoContrastEnabled_ = config.autoContrastEnabled;
    autoContrastStrength_ = std::clamp(config.autoContrastStrength, 0.0f, 1.0f);
    if (autoContrastStrengthSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(autoContrastStrength_ * 100.0f)),
                                           autoContrastStrengthSlider_->minimum(), autoContrastStrengthSlider_->maximum());
        QSignalBlocker block(autoContrastStrengthSlider_);
        autoContrastStrengthSlider_->setValue(sliderValue);
    }
    OnAutoContrastToggled(autoContrastEnabled_);
    OnAutoContrastStrengthChanged(static_cast<int>(std::round(autoContrastStrength_ * 100.0f)));

    displayColorMode_ = std::clamp(config.displayColorMode, 0, 4);
    if (displayColorCombo_) {
        QSignalBlocker block(displayColorCombo_);
        displayColorCombo_->setCurrentIndex(displayColorMode_);
    }
    contrast_ = std::clamp(config.contrast, 0.25f, 4.0f);
    if (contrastSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(contrast_ * 100.0f)),
                                           contrastSlider_->minimum(), contrastSlider_->maximum());
        QSignalBlocker block(contrastSlider_);
        contrastSlider_->setValue(sliderValue);
    }
    brightness_ = std::clamp(config.brightness, -1.0f, 1.0f);
    if (brightnessSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(brightness_ * 100.0f)),
                                           brightnessSlider_->minimum(), brightnessSlider_->maximum());
        QSignalBlocker block(brightnessSlider_);
        brightnessSlider_->setValue(sliderValue);
    }
    OnDisplayColorModeChanged(displayColorMode_);
    OnContrastChanged(static_cast<int>(std::round(contrast_ * 100.0f)));
    OnBrightnessChanged(static_cast<int>(std::round(brightness_ * 100.0f)));

    if (ocrAssistCheckbox_) {
        QSignalBlocker block(ocrAssistCheckbox_);
        ocrAssistCheckbox_->setChecked(config.ocrAssistEnabled);
    }
    ocrAssistEnabled_ = config.ocrAssistEnabled;
    if (vlmAssistCheckbox_) {
        QSignalBlocker block(vlmAssistCheckbox_);
        vlmAssistCheckbox_->setChecked(config.vlmAssistEnabled);
    }
    vlmAssistEnabled_ = config.vlmAssistEnabled;
    if (assistiveOverlayCheckbox_) {
        QSignalBlocker block(assistiveOverlayCheckbox_);
        assistiveOverlayCheckbox_->setChecked(config.assistiveOverlayEnabled);
    }
    assistiveOverlayEnabled_ = config.assistiveOverlayEnabled;
    UpdateAssistiveRuntimeState();

    if (spatialSharpenCheckbox_) {
        QSignalBlocker block(spatialSharpenCheckbox_);
        spatialSharpenCheckbox_->setChecked(config.spatialSharpenEnabled);
    }
    spatialSharpenEnabled_ = config.spatialSharpenEnabled;
    spatialUpscaler_ = config.spatialUpscaler == 0 ? SpatialUpscaler::kFsrEasuRcas : SpatialUpscaler::kNis;
    spatialSharpness_ = std::clamp(config.spatialSharpness, 0.0f, 1.0f);
    if (spatialBackendCombo_) {
        QSignalBlocker block(spatialBackendCombo_);
        spatialBackendCombo_->setCurrentIndex(static_cast<int>(spatialUpscaler_));
    }
    if (spatialSharpnessSlider_) {
        const int sliderValue = std::clamp(static_cast<int>(std::round(spatialSharpness_ * 100.0f)),
                                           spatialSharpnessSlider_->minimum(), spatialSharpnessSlider_->maximum());
        QSignalBlocker block(spatialSharpnessSlider_);
        spatialSharpnessSlider_->setValue(sliderValue);
    }
    OnSpatialSharpenToggled(spatialSharpenEnabled_);
    OnSpatialUpscalerChanged(static_cast<int>(spatialUpscaler_));
    OnSpatialSharpnessChanged(static_cast<int>(std::round(spatialSharpness_ * 100.0f)));

    if (debugButton_) {
        QSignalBlocker block(debugButton_);
        debugButton_->setChecked(config.debugView);
    }
    debugViewEnabled_ = config.debugView;
    OnDebugViewToggled(debugViewEnabled_);

    if (focusMarkerCheckbox_) {
        QSignalBlocker block(focusMarkerCheckbox_);
        focusMarkerCheckbox_->setChecked(config.focusMarker);
    }
    focusMarkerEnabled_ = config.focusMarker;
    OnFocusMarkerToggled(focusMarkerEnabled_);

    configTrackingSuspended_ = false;
    SyncCurrentConfigToPersistence();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::PromoteCurrentConfigToPreset()
{
    if (!mainWindow_) {
        return;
    }

    QString defaultName = QStringLiteral("Custom Quick Option");
    if (const settings::PresetDefinition* preset =
            settings::FindPresetById(persistentSettings_.selectedPresetId, persistentSettings_.customPresets)) {
        defaultName = preset->name + QStringLiteral(" Copy");
    }

    bool ok = false;
    const QString name = QInputDialog::getText(mainWindow_.get(),
                                               QStringLiteral("Save As Quick Option"),
                                               QStringLiteral("Quick option name:"),
                                               QLineEdit::Normal,
                                               defaultName,
                                               &ok).trimmed();
    if (!ok || name.isEmpty()) {
        return;
    }

    settings::AdvancedConfig config = CaptureCurrentAdvancedConfig();
    config.id = MakeCustomEntityId(QStringLiteral("custom-config"));
    config.name = name;
    config.description = QStringLiteral("Custom quick option created from Advanced Tuning.");

    settings::PresetDefinition preset;
    preset.id = MakeCustomEntityId(QStringLiteral("custom-preset"));
    preset.name = name;
    preset.description = config.description;
    preset.configId = config.id;
    preset.isBuiltIn = false;

    persistentSettings_.customConfigs.push_back(config);
    persistentSettings_.customPresets.push_back(preset);
    persistentSettings_.currentConfig = config;
    persistentSettings_.selectedPresetId = preset.id;

    PopulatePresetList();
    RefreshPresetSelection(true);
    UpdatePresetDescription();
}

void OpenZoomApp::OnPresetSelectionChanged(QListWidgetItem* current, QListWidgetItem* /*previous*/)
{
    if (presetSelectionSyncSuspended_ || !current) {
        return;
    }

    const QString presetId = current->data(kPresetIdRole).toString();
    auto config = settings::ResolveConfigForPreset(presetId,
                                                   persistentSettings_.customConfigs,
                                                   persistentSettings_.customPresets);
    if (!config) {
        return;
    }

    persistentSettings_.selectedPresetId = presetId;
    ApplyAdvancedConfig(*config);
}

void OpenZoomApp::UpdateAssistiveRuntimeState()
{
    if (assistiveRuntime_) {
        assistiveRuntime_->SetModes(ocrAssistEnabled_ && assistiveOverlayEnabled_,
                                    vlmAssistEnabled_ && assistiveOverlayEnabled_);
    }
    if (assistiveOverlay_) {
        assistiveOverlay_->setVisible(assistiveOverlayEnabled_ && (ocrAssistEnabled_ || vlmAssistEnabled_));
    }
}

void OpenZoomApp::MaybeRequestAssistiveAnalysis(const uint8_t* data, UINT width, UINT height)
{
    if (!assistiveRuntime_ || !assistiveRuntime_->WantsAnalysis() || !assistiveOverlayEnabled_) {
        return;
    }
    if (debugViewEnabled_) {
        return;
    }
    if (!data || width == 0 || height == 0) {
        return;
    }
    if (assistiveRuntime_->IsBusy()) {
        return;
    }

    constexpr qint64 kAssistiveIntervalMs = 1600;
    if (assistiveAnalysisTimer_.isValid() && assistiveAnalysisTimer_.elapsed() < kAssistiveIntervalMs) {
        return;
    }
    assistiveAnalysisTimer_.restart();
    assistiveRuntime_->SubmitFrame(data, static_cast<int>(width), static_cast<int>(height));
}

AssistiveRuntimeConfig OpenZoomApp::BuildAssistiveRuntimeConfig() const
{
    const settings::AssistiveSettings& assistive = persistentSettings_.assistive;
    AssistiveRuntimeConfig cfg;
    cfg.aiProvider = assistive.aiProvider;
    cfg.codexExecutablePath = assistive.codexExecutablePath;
    cfg.codexModel = assistive.codexModel;
    cfg.codexReasoningEffort = assistive.codexReasoningEffort;
    cfg.codexInternetEnabled = assistive.codexInternetEnabled;
    cfg.codexCodingEnabled = assistive.codexCodingEnabled;
    cfg.codexWorkspaceDirectory = assistive.codexWorkspaceDirectory;
    cfg.assistantInstructions = assistive.assistantInstructions;
    cfg.vlmApiUrl = assistive.vlmApiUrl;
    cfg.vlmApiKey = assistive.vlmApiKey;
    cfg.vlmModel = assistive.vlmModel;
    cfg.vlmPrompt = assistive.vlmPrompt;
    cfg.tesseractPath = assistive.tesseractPath;
    cfg.ocrLanguage = assistive.ocrLanguage;
    cfg.ttsEngine = assistive.ttsEngine;
    cfg.ttsVoiceName = assistive.ttsVoiceName;
    cfg.ttsVoiceLocale = assistive.ttsVoiceLocale;
    cfg.ttsRate = assistive.ttsRate;
    cfg.lectureNotesEnabled = assistive.lectureNotesEnabled;
    cfg.notesDirectory = QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("output/notes"));
    return cfg;
}

void OpenZoomApp::ApplyAssistiveSettingsToRuntime()
{
    if (assistiveRuntime_) {
        assistiveRuntime_->SetConfig(BuildAssistiveRuntimeConfig());
    }
}

void OpenZoomApp::OpenAiSettingsDialog()
{
    if (!mainWindow_) {
        return;
    }
    AiSettingsDialog dialog(persistentSettings_.assistive, mainWindow_.get());
    if (dialog.exec() == QDialog::Accepted) {
        persistentSettings_.assistive = dialog.result();
        ApplyAssistiveSettingsToRuntime();
        SavePersistentSettings();
    }
}

void OpenZoomApp::OpenNotesFile()
{
    const QString path = assistiveRuntime_ ? assistiveRuntime_->notesFilePath() : QString();
    if (path.isEmpty()) {
        if (processingStatusLabel_) {
            processingStatusLabel_->setText(
                QStringLiteral("No lecture notes yet — notes appear once OCR or Explain produces text."));
        }
        qInfo() << "Open notes skipped: no notes file written yet";
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

void OpenZoomApp::SubmitOnDemandAnalysis(bool runOcr, bool runVlm)
{
    if (!assistiveRuntime_) {
        return;
    }

    // Prefer the processed GPU output, using the same readback path as the
    // periodic assistive loop.
    if (usingCudaLastFrame_ && cudaSharedTexture_ && presenter_ &&
        processedFrameWidth_ > 0 && processedFrameHeight_ > 0 &&
        presenter_->ReadbackTexture(cudaSharedTexture_.Get(),
                                    processedFrameWidth_,
                                    processedFrameHeight_,
                                    assistiveBuffer_)) {
        assistiveRuntime_->SubmitFrameForced(assistiveBuffer_.data(),
                                             static_cast<int>(processedFrameWidth_),
                                             static_cast<int>(processedFrameHeight_),
                                             runOcr, runVlm);
        return;
    }

    // Fall back to the CPU-converted presentation frame when GPU readback is
    // unavailable (passthrough or debug view).
    if (!presentationBuffer_.empty() && presentationWidth_ > 0 && presentationHeight_ > 0) {
        assistiveRuntime_->SubmitFrameForced(presentationBuffer_.data(),
                                             static_cast<int>(presentationWidth_),
                                             static_cast<int>(presentationHeight_),
                                             runOcr, runVlm);
        return;
    }

    qWarning() << "On-demand analysis skipped: no frame available";
}

void OpenZoomApp::SubmitAssistantPrompt()
{
    if (!assistiveRuntime_ || !assistantPromptEdit_ || assistiveRuntime_->IsCodexTurnActive()) {
        return;
    }
    const QString prompt = assistantPromptEdit_->toPlainText().trimmed();
    if (prompt.isEmpty()) {
        assistantPromptEdit_->setFocus();
        return;
    }
    SubmitAssistantPromptText(prompt, true, false);
}

void OpenZoomApp::SubmitFloatingAssistantPrompt(const QString& prompt)
{
    if (prompt.trimmed().isEmpty()) {
        return;
    }
    SubmitAssistantPromptText(prompt.trimmed(), false, true);
}

void OpenZoomApp::SubmitAssistantPromptText(const QString& prompt,
                                            bool clearAdvancedEditor,
                                            bool forceAttachFrame)
{
    if (!assistiveRuntime_ || prompt.trimmed().isEmpty() || assistiveRuntime_->IsCodexTurnActive()) {
        return;
    }
    const bool attachFrame = forceAttachFrame ||
                             (assistantAttachFrameCheckbox_ && assistantAttachFrameCheckbox_->isChecked());
    const uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    if (attachFrame && usingCudaLastFrame_ && cudaSharedTexture_ && presenter_ &&
        processedFrameWidth_ > 0 && processedFrameHeight_ > 0 &&
        presenter_->ReadbackTexture(cudaSharedTexture_.Get(),
                                    processedFrameWidth_,
                                    processedFrameHeight_,
                                    assistiveBuffer_)) {
        data = assistiveBuffer_.data();
        width = static_cast<int>(processedFrameWidth_);
        height = static_cast<int>(processedFrameHeight_);
    } else if (attachFrame && !presentationBuffer_.empty() &&
               presentationWidth_ > 0 && presentationHeight_ > 0) {
        data = presentationBuffer_.data();
        width = static_cast<int>(presentationWidth_);
        height = static_cast<int>(presentationHeight_);
    }

    const QString submittedPrompt = prompt.trimmed();
    pendingAssistantPrompt_ = submittedPrompt;
    AppendAssistantMessage(QStringLiteral("You"), submittedPrompt);
    QTextCursor cursor = assistantTranscript_->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertText(QStringLiteral("OpenZoom Assistant\n"));
    assistantTranscript_->setTextCursor(cursor);
    assistantResponseOpen_ = true;
    assistantResponseReceivedText_ = false;
    if (clearAdvancedEditor && assistantPromptEdit_) {
        assistantPromptEdit_->clear();
    }
    SetAssistantBusy(true);
    assistiveRuntime_->SubmitAssistantPrompt(submittedPrompt,
                                             currentAssistantThreadId_,
                                             data,
                                             width,
                                             height,
                                             attachFrame);
}

void OpenZoomApp::PopulateAssistantHistory()
{
    if (!assistantHistoryList_) {
        return;
    }
    QSignalBlocker blocker(assistantHistoryList_);
    assistantHistoryList_->clear();
    std::vector<const settings::CodexConversation*> conversations;
    conversations.reserve(persistentSettings_.codexConversations.size());
    for (const settings::CodexConversation& conversation : persistentSettings_.codexConversations) {
        conversations.push_back(&conversation);
    }
    std::sort(conversations.begin(), conversations.end(),
              [](const settings::CodexConversation* lhs, const settings::CodexConversation* rhs) {
                  return lhs->updatedAt > rhs->updatedAt;
              });
    for (const settings::CodexConversation* conversation : conversations) {
        const qint64 timestamp = conversation->updatedAt > 0
                                     ? conversation->updatedAt
                                     : conversation->createdAt;
        const QString timeText = timestamp > 0
                                     ? QDateTime::fromSecsSinceEpoch(timestamp).toString(
                                           QStringLiteral("yyyy-MM-dd  HH:mm"))
                                     : QString();
        const QString title = conversation->title.trimmed().isEmpty()
                                  ? QStringLiteral("OpenZoom Assistant")
                                  : conversation->title.trimmed();
        const QString preview = conversation->preview.simplified().left(110);
        auto* item = new QListWidgetItem(
            QStringLiteral("%1\n%2%3")
                .arg(title,
                     timeText,
                     preview.isEmpty() ? QString() : QStringLiteral("\n%1").arg(preview)),
            assistantHistoryList_);
        item->setData(Qt::UserRole, conversation->threadId);
        item->setData(Qt::UserRole + 1, title);
        item->setToolTip(preview);
        if (conversation->threadId == currentAssistantThreadId_) {
            assistantHistoryList_->setCurrentItem(item);
        }
    }
}

void OpenZoomApp::LoadSelectedAssistantConversation()
{
    if (!assistantHistoryList_ || !assistiveRuntime_ || assistiveRuntime_->IsCodexTurnActive()) {
        return;
    }
    QListWidgetItem* item = assistantHistoryList_->currentItem();
    if (!item) {
        return;
    }
    const QString threadId = item->data(Qt::UserRole).toString();
    if (threadId.isEmpty()) {
        return;
    }
    currentAssistantThreadId_ = threadId;
    assistantTranscript_->setPlainText(QStringLiteral("Loading conversation..."));
    assistiveRuntime_->LoadAssistantConversation(threadId);
}

void OpenZoomApp::SetAssistantBusy(bool busy)
{
    if (assistiveOverlay_) {
        assistiveOverlay_->SetBusy(busy);
    }
    if (assistantSendButton_) {
        assistantSendButton_->setEnabled(!busy && codexReady_ && codexSignedIn_);
    }
    if (assistantStopButton_) {
        assistantStopButton_->setEnabled(busy);
    }
    if (assistantNewButton_) {
        assistantNewButton_->setEnabled(!busy);
    }
    if (assistantConnectButton_) {
        assistantConnectButton_->setEnabled(!busy);
    }
    if (assistantPromptEdit_) {
        assistantPromptEdit_->setEnabled(!busy);
    }
    if (assistantHistoryList_) {
        assistantHistoryList_->setEnabled(!busy);
    }
    for (QPushButton* button : {assistantRenameButton_, assistantExportButton_, assistantDeleteButton_}) {
        if (button) {
            button->setEnabled(!busy);
        }
    }
}

void OpenZoomApp::AppendAssistantMessage(const QString& speaker, const QString& text)
{
    if (!assistantTranscript_ || text.trimmed().isEmpty()) {
        return;
    }
    QTextCursor cursor = assistantTranscript_->textCursor();
    cursor.movePosition(QTextCursor::End);
    if (!assistantTranscript_->document()->isEmpty()) {
        cursor.insertText(QStringLiteral("\n"));
    }
    cursor.insertText(QStringLiteral("%1\n%2\n").arg(speaker, text.trimmed()));
    assistantTranscript_->setTextCursor(cursor);
    assistantTranscript_->ensureCursorVisible();
}

void OpenZoomApp::OnOcrAssistToggled(bool checked)
{
    ocrAssistEnabled_ = checked;
    UpdateAssistiveRuntimeState();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnVlmAssistToggled(bool checked)
{
    vlmAssistEnabled_ = checked;
    UpdateAssistiveRuntimeState();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnAssistiveOverlayToggled(bool checked)
{
    assistiveOverlayEnabled_ = checked;
    UpdateAssistiveRuntimeState();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnAssistiveOverlayUpdated(const QString& title, const QString& body, bool visible)
{
    if (!assistiveOverlay_) {
        return;
    }
    assistiveOverlay_->SetContent(title, body, visible && assistiveOverlayEnabled_);
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

void OpenZoomApp::RefreshCameraModesList(size_t index) {
    if (!cameraModesList_) {
        return;
    }
    cameraModesList_->clear();
    if (index >= cameras_.size()) {
        return;
    }

    const auto formats = mediaCapture_.EnumerateFormats(cameras_[index]);
    if (formats.empty()) {
        const std::string& detail = mediaCapture_.LastError();
        if (!detail.empty()) {
            cameraModesList_->addItem(QStringLiteral("Modes unavailable (%1)").arg(QString::fromStdString(detail)));
        } else {
            cameraModesList_->addItem(QStringLiteral("No modes reported"));
        }
        return;
    }

    std::vector<VideoFormat> sorted = formats;
    std::sort(sorted.begin(), sorted.end(), [](const VideoFormat& a, const VideoFormat& b) {
        const unsigned int pixelsA = a.width * a.height;
        const unsigned int pixelsB = b.width * b.height;
        if (pixelsA != pixelsB) {
            return pixelsA > pixelsB;
        }
        const double fpsA = (a.denominator == 0) ? 0.0 : static_cast<double>(a.numerator) / static_cast<double>(a.denominator);
        const double fpsB = (b.denominator == 0) ? 0.0 : static_cast<double>(b.numerator) / static_cast<double>(b.denominator);
        if (std::abs(fpsA - fpsB) > 0.01) {
            return fpsA > fpsB;
        }
        if (a.width != b.width) {
            return a.width > b.width;
        }
        return a.height > b.height;
    });

    for (const auto& fmt : sorted) {
        QString fpsText;
        if (fmt.numerator == 0 || fmt.denominator == 0) {
            fpsText = QStringLiteral("?");
        } else {
            const double fps = static_cast<double>(fmt.numerator) / static_cast<double>(fmt.denominator);
            if (std::abs(fps - std::round(fps)) < 0.01) {
                fpsText = QString::number(static_cast<int>(std::round(fps)));
            } else {
                fpsText = QString::number(fps, 'f', 2);
            }
        }
        const QString line = QStringLiteral("%1x%2@%3")
                                 .arg(fmt.width)
                                 .arg(fmt.height)
                                 .arg(fpsText);
        cameraModesList_->addItem(line);
    }
}

void OpenZoomApp::OnCameraSelectionChanged(int index) {
    if (index < 0 || static_cast<size_t>(index) >= cameras_.size()) {
        return;
    }

    // A manual camera pick always wins over an in-flight automatic reconnect.
    cameraReconnectPending_ = false;
    persistentSettings_.cameraIndex = index;
    RefreshCameraModesList(static_cast<size_t>(index));
    StartCameraCapture(static_cast<size_t>(index));
}

void OpenZoomApp::OnBlackWhiteToggled(bool checked) {
    blackWhiteEnabled_ = checked;
    if (bwSlider_) {
        bwSlider_->setEnabled(checked);
    }
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnBlackWhiteThresholdChanged(int value) {
    blackWhiteThreshold_ = std::clamp(static_cast<float>(value) / 255.0f, 0.0f, 1.0f);
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnZoomToggled(bool checked) {
    zoomEnabled_ = checked;
    if (zoomSlider_) {
        zoomSlider_->setEnabled(checked);
    }
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnZoomAmountChanged(int value) {
    zoomAmount_ = std::max(1.0f, static_cast<float>(value) / static_cast<float>(kZoomSliderScale));
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnDebugViewToggled(bool checked) {
    debugViewEnabled_ = checked;
    if (focusMarkerCheckbox_) {
        focusMarkerCheckbox_->setEnabled(!checked);
    }
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
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
    persistentSettings_.rotationQuarterTurns = rotationQuarterTurns_;

    float rotatedX = zoomCenterX_;
    float rotatedY = zoomCenterY_;
    RotateNormalizedPoint(zoomCenterX_, zoomCenterY_, delta, rotatedX, rotatedY);
    SetZoomCenter(rotatedX, rotatedY, true, true);

    cpuPipeline_.ResetTemporalHistory();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
        cudaSurface_->ResetStabilization();
        cudaSurface_->ResetKeystone();
    }
    ResetCudaFenceState();

    processedFrameWidth_ = 0;
    processedFrameHeight_ = 0;

    UpdateRotationUi();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnControlsCollapsedToggled(bool checked) {
    controlsCollapsed_ = !checked;
    if (controlsContainer_) {
        controlsContainer_->setVisible(checked);
    }
    if (collapseButton_) {
        collapseButton_->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        collapseButton_->setText(checked ? "Hide Advanced Tuning" : "Advanced Tuning");
    }
    persistentSettings_.controlsCollapsed = controlsCollapsed_;
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
    persistentSettings_.virtualJoystick = virtualJoystickEnabled_;
}

void OpenZoomApp::OnBlurToggled(bool checked) {
    blurEnabled_ = checked;
    UpdateBlurUiLabels();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnBlurSigmaChanged(int value) {
    blurSigma_ = SliderValueToSigma(value);
    UpdateBlurUiLabels();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
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
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnFocusMarkerToggled(bool checked) {
    focusMarkerEnabled_ = checked;
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnSpatialSharpenToggled(bool checked) {
    spatialSharpenEnabled_ = checked;
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnSpatialUpscalerChanged(int index) {
    const int clamped = std::clamp(index, 0, 1);
    spatialUpscaler_ = static_cast<SpatialUpscaler>(clamped);
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnSpatialSharpnessChanged(int value) {
    spatialSharpness_ = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
    if (spatialSharpnessValueLabel_) {
        spatialSharpnessValueLabel_->setText(QString::number(spatialSharpness_, 'f', 2));
    }
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
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
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
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
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnStabilizationToggled(bool checked) {
    stabilizationEnabled_ = checked;
    if (stabilizationStrengthSlider_) {
        stabilizationStrengthSlider_->setEnabled(checked);
    }
    if (cudaSurface_) {
        cudaSurface_->ResetStabilization();
        cudaSurface_->ResetKeystone();
    }
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnStabilizationStrengthChanged(int value) {
    const int sliderMin = stabilizationStrengthSlider_ ? stabilizationStrengthSlider_->minimum() : 0;
    const int sliderMax = stabilizationStrengthSlider_ ? stabilizationStrengthSlider_->maximum() : 98;
    const int clamped = std::clamp(value, sliderMin, sliderMax);
    if (stabilizationStrengthSlider_ && clamped != value) {
        QSignalBlocker block(stabilizationStrengthSlider_);
        stabilizationStrengthSlider_->setValue(clamped);
    }
    stabilizationStrength_ = std::clamp(static_cast<float>(clamped) / 100.0f, 0.0f, 0.98f);
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnKeystoneToggled(bool checked) {
    keystoneEnabled_ = checked;
    if (cudaSurface_) {
        cudaSurface_->ResetKeystone();
    }
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnAutoContrastToggled(bool checked) {
    autoContrastEnabled_ = checked;
    if (autoContrastStrengthSlider_) {
        autoContrastStrengthSlider_->setEnabled(checked);
    }
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnAutoContrastStrengthChanged(int value) {
    autoContrastStrength_ = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnDisplayColorModeChanged(int index) {
    displayColorMode_ = std::clamp(index, 0, 4);
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnContrastChanged(int value) {
    contrast_ = std::clamp(static_cast<float>(value) / 100.0f, 0.25f, 4.0f);
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnBrightnessChanged(int value) {
    brightness_ = std::clamp(static_cast<float>(value) / 100.0f, -1.0f, 1.0f);
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::SetZoomCenter(float normX, float normY, bool syncUi,
                                bool preservePresetSelection) {
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
    SyncCurrentConfigToPersistence(preservePresetSelection);
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
    QString detail;
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

    if (!cameraActive_ && cameraReconnectPending_) {
        text = QStringLiteral("Reconnecting to camera…");
        detail = QStringLiteral("Camera connection lost - reconnecting automatically");
        color = QStringLiteral("#d17c00");
    } else if (!cameraActive_) {
        text = QStringLiteral("Camera Offline");
        if (lastCameraError_.isEmpty()) {
            detail = QStringLiteral("Processing: Idle (camera offline)");
        } else {
            detail = QStringLiteral("Processing: Idle (camera offline - %1)").arg(lastCameraError_);
        }
        color = QStringLiteral("#c0392b");
    } else if (debugViewEnabled_) {
        text = QStringLiteral("CPU Debug");
        detail = QStringLiteral("Processing: CPU (debug view)");
        color = QStringLiteral("#d17c00");
    } else if (usingCudaLastFrame_ && cudaPipelineAvailable_) {
        text = QStringLiteral("GPU Ready");
        const QString backend = backendLabel();
        if (cudaFenceInteropEnabled_) {
            detail = QStringLiteral("Processing: GPU (fence interop, %1)").arg(backend);
        } else {
            detail = QStringLiteral("Processing: GPU (%1)").arg(backend);
        }
        color = QStringLiteral("#1c9c3e");
    } else {
        // CPU effects path is deprecated: without the GPU pipeline the app
        // shows unprocessed passthrough video.
        text = QStringLiteral("GPU Required");
        detail = QStringLiteral("GPU required - processing disabled (showing raw video)");
        color = QStringLiteral("#c0392b");
    }

    if (recording_) {
        text.append(QStringLiteral(" [REC]"));
        detail.append(QStringLiteral(" [REC]"));
    }

    if (ocrAssistEnabled_) {
        text.append(QStringLiteral(" [OCR]"));
        detail.append(QStringLiteral(" [OCR]"));
    }
    if (vlmAssistEnabled_) {
        text.append(QStringLiteral(" [VLM]"));
        detail.append(QStringLiteral(" [VLM]"));
    }

    // Transient, non-modal notifications (disk-full recording stop, reconnect
    // give-up, ...) override the pipeline summary for a few seconds. The label
    // is refreshed every presented frame, so the message clears on its own.
    if (!transientStatusMessage_.isEmpty()) {
        if (QDateTime::currentMSecsSinceEpoch() < transientStatusUntilMs_) {
            detail = text + QStringLiteral(" - ") + transientStatusMessage_;
            text = transientStatusMessage_;
            color = QStringLiteral("#1c6dd0");
        } else {
            transientStatusMessage_.clear();
        }
    }

    processingStatusLabel_->setText(text);
    processingStatusLabel_->setToolTip(detail);
    processingStatusLabel_->setStyleSheet(QStringLiteral("color: %1;").arg(color));
}

void OpenZoomApp::ShowStatusMessage(const QString& message, int durationMs)
{
    transientStatusMessage_ = message;
    transientStatusUntilMs_ = QDateTime::currentMSecsSinceEpoch() + durationMs;
    UpdateProcessingStatusLabel();
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

    ViewMapping mapping{};
    const bool cropToFill = !debugViewEnabled_;
    if (!ComputeViewMapping(processedFrameWidth_,
                            processedFrameHeight_,
                            renderWidget_->width(),
                            renderWidget_->height(),
                            zoomCenterX_,
                            zoomCenterY_,
                            cropToFill,
                            mapping)) {
        return false;
    }

    float localX = static_cast<float>(pos.x()) - static_cast<float>(mapping.offsetX);
    float localY = static_cast<float>(pos.y()) - static_cast<float>(mapping.offsetY);

    localX = std::clamp(localX, 0.0f, static_cast<float>(std::max<UINT>(1, mapping.activeWidth) - 1));
    localY = std::clamp(localY, 0.0f, static_cast<float>(std::max<UINT>(1, mapping.activeHeight) - 1));

    const float sampleX = mapping.startX + localX * mapping.stepX;
    const float sampleY = mapping.startY + localY * mapping.stepY;

    const float denomX = std::max(static_cast<float>(processedFrameWidth_ - 1), 1.0f);
    const float denomY = std::max(static_cast<float>(processedFrameHeight_ - 1), 1.0f);
    outX = std::clamp(sampleX / denomX, 0.0f, 1.0f);
    outY = std::clamp(sampleY / denomY, 0.0f, 1.0f);
    return true;
}

void OpenZoomApp::ResetCudaFenceState() {
    const UINT64 baseValue = presenter_ ? presenter_->GetLastSignaledFenceValue() : 0;
    sharedFenceCounter_ = baseValue + 1;
    lastCudaSignalValue_ = 0;
    lastGraphicsSignalValue_ = baseValue;
    lastReadbackSignalValue_ = 0;
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

    // Drain the graphics queue first: pipelined presents may still be copying
    // from the shared texture we are about to release.
    presenter_->WaitForIdle();
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
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
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

    ProcessingInput input{};
    input.hostPixels = stageRaw.data();
    input.hostStrideBytes = width * 4;
    input.pixelSizeBytes = static_cast<unsigned int>(sizeof(uint32_t));
    input.width = width;
    input.height = height;
    // inputFormat 0 (BGRA): the CPU already converted and rotated, so
    // rotationQuarterTurns stays 0.

    return RunCudaPipeline(input, width, height);
}

// Feeds raw NV12/YUY2 camera frames straight to the CUDA pipeline: color
// conversion and rotation both run on the GPU and the per-frame CPU
// convert/rotate work is skipped entirely. Returns false whenever anything is
// off (unsupported layout, surface failure, ProcessFrame failure) so the
// caller can fall back to the CPU-converted BGRA path for that frame.
bool OpenZoomApp::TryProcessRawFrameWithCuda(const MediaFrame& frame) {
    const bool isNv12 = IsEqualGUID(frame.subtype, MFVideoFormat_NV12);
    const bool isYuy2 = IsEqualGUID(frame.subtype, MFVideoFormat_YUY2);
    if (!isNv12 && !isYuy2) {
        return false;
    }

    const UINT width = frame.width;
    const UINT height = frame.height;
    if (width == 0 || height == 0 || frame.data.empty()) {
        return false;
    }

    UINT stride = frame.stride;
    const uint8_t* plane2 = nullptr;
    UINT plane2Stride = 0;
    if (isNv12) {
        if (stride < width) {
            stride = width;
        }
        // Media Foundation NV12 buffers are contiguous: Y rows followed by
        // interleaved UV rows, both at the Y stride.
        const size_t yBytes = static_cast<size_t>(stride) * height;
        const size_t uvBytes = static_cast<size_t>(stride) * ((height + 1) / 2);
        if (frame.dataSize < yBytes + uvBytes) {
            return false;
        }
        plane2 = frame.data.data() + yBytes;
        plane2Stride = stride;
    } else {
        if (stride < width * 2) {
            stride = width * 2;
        }
        if (frame.dataSize < static_cast<size_t>(stride) * height) {
            return false;
        }
    }

    // Rotation happens on the GPU after conversion, so the interop surface and
    // every processing stage run at the post-rotation extent.
    const int turns = ((rotationQuarterTurns_ % 4) + 4) % 4;
    const UINT outWidth = ((turns & 1) != 0) ? height : width;
    const UINT outHeight = ((turns & 1) != 0) ? width : height;

    if (!EnsureCudaSurface(outWidth, outHeight)) {
        usingCudaLastFrame_ = false;
        return false;
    }

    ProcessingInput input{};
    input.hostPixels = frame.data.data();
    input.hostStrideBytes = stride;
    input.pixelSizeBytes = isNv12 ? 1u : 2u;
    input.width = width;    // pre-rotation host layout
    input.height = height;
    input.inputFormat = isNv12 ? 1 : 2;
    input.hostPlane2 = plane2;
    input.hostPlane2StrideBytes = plane2Stride;
    input.rotationQuarterTurns = turns;

    if (!RunCudaPipeline(input, outWidth, outHeight)) {
        if (!rawCudaPathWarned_) {
            qWarning() << "GPU raw-format path failed; falling back to CPU conversion for"
                       << (isNv12 ? "NV12" : "YUY2") << "frames";
            rawCudaPathWarned_ = true;
        }
        return false;
    }

    processedFrameWidth_ = outWidth;
    processedFrameHeight_ = outHeight;
    usingCudaLastFrame_ = true;
    HandleGpuFramePresented(outWidth, outHeight);
    UpdateProcessingStatusLabel();
    return true;
}

// Shared tail of the CUDA path: builds ProcessingSettings from the live UI
// state, runs ProcessFrame with the fence dance, and presents the result.
// The interop surface must already exist at presentWidth x presentHeight.
bool OpenZoomApp::RunCudaPipeline(const ProcessingInput& input, UINT presentWidth, UINT presentHeight) {
    if (!cudaSurface_) {
        qWarning() << "CUDA pipeline disabled: surface not available";
        usingCudaLastFrame_ = false;
        return false;
    }

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
    settings.enableStabilization = stabilizationEnabled_;
    settings.stabilizationStrength = stabilizationStrength_;
    settings.displayColorMode = displayColorMode_;
    settings.contrast = contrast_;
    settings.brightness = brightness_;
    settings.enableKeystone = keystoneEnabled_;
    settings.enableAutoContrast = autoContrastEnabled_;
    settings.autoContrastStrength = autoContrastStrength_;

    FenceSyncParams cudaSyncParams{};
    uint64_t cudaSignalCandidate = 0;
    if (cudaFenceInteropEnabled_) {
        // Re-seed from the presenter so our values stay above any internal
        // fence signals issued by CPU-path frames in between; signaling the
        // shared fence with a lower value would move it backwards and break
        // the wait/signal ordering below.
        sharedFenceCounter_ = std::max(sharedFenceCounter_,
                                       presenter_->GetLastSignaledFenceValue() + 1);
        cudaSyncParams.enable = true;
        // Async readbacks copy from the shared texture on the graphics queue;
        // CUDA must not write the next frame until both the present and the
        // newest readback copy have retired (GPU-side wait only).
        cudaSyncParams.waitValue = std::max(lastGraphicsSignalValue_, lastReadbackSignalValue_);
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
                                   presentWidth,
                                   presentHeight,
                                   presentSync.enable ? &presentSync : nullptr);

    if (presentSync.enable) {
        lastGraphicsSignalValue_ = graphicsSignalCandidate;
        sharedFenceCounter_ = graphicsSignalCandidate + 1;
    }

    usingCudaLastFrame_ = true;
    return true;
}

// Post-present GPU housekeeping: pump the presenter's async readback ring for
// recording and the periodic assistive grab. One request per frame at most and
// a single drain point, so the two consumers never steal each other's results.
// Nothing here blocks: requests are skip-on-busy and completed copies surface
// one or more frames later with their own dimensions (results are silently
// dropped on resize).
void OpenZoomApp::HandleGpuFramePresented(UINT width, UINT height) {
    if (!presenter_ || !cudaSharedTexture_) {
        return;
    }

    // Drain first so a fully occupied ring frees up before the new request.
    UINT readbackWidth = 0;
    UINT readbackHeight = 0;
    while (presenter_->TryGetCompletedReadback(asyncReadbackBuffer_, readbackWidth, readbackHeight)) {
        if (recording_) {
            MaybeRecordFrame(asyncReadbackBuffer_.data(), readbackWidth, readbackHeight);
        }
        MaybeRequestAssistiveAnalysis(asyncReadbackBuffer_.data(), readbackWidth, readbackHeight);
    }

    const bool assistiveWanted = assistiveRuntime_ && assistiveRuntime_->WantsAnalysis() &&
                                 assistiveOverlayEnabled_ && !debugViewEnabled_ &&
                                 !assistiveRuntime_->IsBusy() && AssistiveAnalysisDue();
    if (recording_ || assistiveWanted) {
        if (presenter_->RequestReadback(cudaSharedTexture_.Get(), width, height)) {
            // The readback copy runs on the graphics queue after the present;
            // next frame's CUDA pass must wait for it before writing the
            // shared texture again.
            lastReadbackSignalValue_ = presenter_->GetLastSignaledFenceValue();
        }
    }
}

bool OpenZoomApp::AssistiveAnalysisDue() const {
    constexpr qint64 kAssistiveIntervalMs = 1600;
    return !assistiveAnalysisTimer_.isValid() ||
           assistiveAnalysisTimer_.elapsed() >= kAssistiveIntervalMs;
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

bool OpenZoomApp::StartCameraCapture(size_t index, bool interactive) {
    if (index >= cameras_.size()) {
        return false;
    }

    selectedCameraIndex_ = static_cast<int>(index);
    StopCameraCapture();
    const uint64_t captureSession = cameraSessionId_;
    cpuPipeline_.ResetTemporalHistory();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
        cudaSurface_->ResetStabilization();
        cudaSurface_->ResetKeystone();
    }

    const CameraDescriptor& descriptor = cameras_[index];
    FrameCallback callback = [this](const MediaFrame& frame) {
        std::scoped_lock lock(cameraMutex_);
        latestFrame_ = frame;
    };
    CaptureErrorCallback errorCallback = [this, captureSession](const std::string& detail) {
        const QString message = QString::fromStdString(detail);
        QMetaObject::invokeMethod(this,
                                  [this, captureSession, message]() {
                                      HandleCameraRuntimeFailure(captureSession, message);
                                  },
                                  Qt::QueuedConnection);
    };

    if (!mediaCapture_.StartCapture(descriptor,
                                    std::move(callback),
                                    MFVideoFormat_NV12,
                                    std::move(errorCallback))) {
        const std::string detail = mediaCapture_.LastError();
        const CameraFailureKind kind = mediaCapture_.LastFailureKind();
        QString message;
        if (!detail.empty() && (kind == CameraFailureKind::DeviceBusy ||
                                kind == CameraFailureKind::DeviceMissing ||
                                kind == CameraFailureKind::AccessDenied)) {
            // LastError() is already a full plain-language sentence for these
            // failure kinds; show it verbatim.
            message = QString::fromStdString(detail);
        } else {
            message = QStringLiteral("Failed to start camera capture");
            if (!detail.empty()) {
                message += QStringLiteral(" (%1)").arg(QString::fromStdString(detail));
            }
        }
        if (interactive) {
            HandleCameraStartFailure(message);
        } else {
            qWarning() << "Camera start failed (silent):" << message;
            lastCameraError_ = message;
            UpdateProcessingStatusLabel();
        }
        return false;
    }

    processedFrameWidth_ = 0;
    processedFrameHeight_ = 0;
    cameraActive_ = true;
    lastCameraError_.clear();
    UpdateProcessingStatusLabel();
    return true;
}

void OpenZoomApp::StopCameraCapture() {
    ++cameraSessionId_;
    mediaCapture_.StopCapture();
    cameraActive_ = false;

    {
        std::scoped_lock lock(cameraMutex_);
        latestFrame_ = MediaFrame{};
    }

    cpuPipeline_.ResetTemporalHistory();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
        cudaSurface_->ResetStabilization();
        cudaSurface_->ResetKeystone();
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

void OpenZoomApp::HandleCameraRuntimeFailure(uint64_t captureSession, const QString& message) {
    // Never surface a modal dialog while the automatic reconnect is running:
    // popping one up mid-lecture would steal focus from the student.
    if (cameraReconnectPending_) {
        return;
    }
    if (captureSession != cameraSessionId_ || !cameraActive_) {
        return;
    }

    qWarning() << "Camera runtime failure:" << message;
    lastCameraError_ = message;

    // Mid-stream device loss is handled by the reconnect state machine instead
    // of an error dialog; the flag is also polled from OnFrameTick in case the
    // tick sees it first.
    if (mediaCapture_.ConsumeDeviceLost()) {
        BeginCameraReconnect();
        return;
    }

    StopCameraCapture();
    UpdateProcessingStatusLabel();
    if (mainWindow_) {
        QMessageBox::warning(mainWindow_.get(), QStringLiteral("Camera Error"), message);
    }
}

// Camera reconnect state machine. Entered on mid-stream device loss; driven
// from OnFrameTick with QDateTime-based backoff (2s/4s/8s), no blocking
// sleeps, no modal dialogs. Gives up after ~30 seconds.
void OpenZoomApp::BeginCameraReconnect() {
    if (cameraReconnectPending_) {
        return;
    }
    qWarning() << "Camera connection lost; reconnecting automatically";
    cameraReconnectPending_ = true;
    cameraReconnectAttempt_ = 0;
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    cameraReconnectStartedMs_ = now;
    cameraReconnectNextAttemptMs_ = now + 2000;
    lastCameraError_ = QStringLiteral("Reconnecting to camera…");
    StopCameraCapture();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::DriveCameraReconnect() {
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now < cameraReconnectNextAttemptMs_) {
        return;
    }

    // Re-enumerate and look for the same physical device again.
    const std::wstring targetLink = mediaCapture_.LastSymbolicLink();
    cameras_ = mediaCapture_.EnumerateCameras();
    {
        QSignalBlocker blocker(cameraCombo_);
        PopulateCameraCombo();
    }

    int matchIndex = -1;
    if (!targetLink.empty()) {
        for (size_t i = 0; i < cameras_.size(); ++i) {
            if (cameras_[i].symbolicLink == targetLink) {
                matchIndex = static_cast<int>(i);
                break;
            }
        }
    }

    if (matchIndex >= 0 && StartCameraCapture(static_cast<size_t>(matchIndex), false)) {
        cameraReconnectPending_ = false;
        qInfo() << "Camera reconnected after" << (cameraReconnectAttempt_ + 1) << "attempt(s)";
        if (cameraCombo_) {
            QSignalBlocker blocker(cameraCombo_);
            cameraCombo_->setCurrentIndex(matchIndex);
        }
        RefreshCameraModesList(static_cast<size_t>(matchIndex));
        ShowStatusMessage(QStringLiteral("Camera reconnected."), 5000);
        return;
    }

    ++cameraReconnectAttempt_;
    constexpr qint64 kReconnectWindowMs = 30000;
    if (now - cameraReconnectStartedMs_ >= kReconnectWindowMs) {
        cameraReconnectPending_ = false;
        const std::string detail = mediaCapture_.LastError();
        lastCameraError_ = !detail.empty()
            ? QString::fromStdString(detail)
            : QStringLiteral("The camera did not come back. Check the connection, then pick it "
                             "again from the camera list.");
        qWarning() << "Camera reconnect gave up:" << lastCameraError_;
        ShowStatusMessage(lastCameraError_, 15000);
        return;
    }

    // Backoff: 2s after the loss, then 4s, then 8s between attempts.
    const qint64 delayMs = std::min<qint64>(2000ll << std::min(cameraReconnectAttempt_, 2), 8000ll);
    cameraReconnectNextAttemptMs_ = now + delayMs;
}



void OpenZoomApp::ApplyPersistentSettings(const settings::PersistentSettings& settings) {
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
    simpleUiMode_ = settings.simpleUiMode;
    if (mainWindow_) {
        mainWindow_->setSimpleMode(settings.simpleUiMode);
    }
    ApplyAdvancedConfig(settings.currentConfig);
    rotationQuarterTurns_ = ((settings.rotationQuarterTurns % 4) + 4) % 4;
    UpdateRotationUi();
    persistentSettings_.selectedPresetId = settings.selectedPresetId;
    RefreshPresetSelection(true);
    UpdatePresetDescription();
    ApplyAssistiveSettingsToRuntime();
}

void OpenZoomApp::SavePersistentSettings() {
    persistentSettings_.cameraIndex = selectedCameraIndex_;
    persistentSettings_.rotationQuarterTurns = rotationQuarterTurns_;
    persistentSettings_.virtualJoystick = virtualJoystickEnabled_;
    persistentSettings_.controlsCollapsed = controlsCollapsed_;
    persistentSettings_.simpleUiMode = simpleUiMode_;
    persistentSettings_.currentConfig = CaptureCurrentAdvancedConfig();
    settings::Save(settingsPath_, persistentSettings_);
}

void OpenZoomApp::OnFrameTick() {
    // Camera loss / reconnect state machine. ConsumeDeviceLost() is polled
    // here in addition to the capture error callback so the reconnect starts
    // no matter which side notices the loss first.
    if (mediaCapture_.ConsumeDeviceLost() && !cameraReconnectPending_) {
        BeginCameraReconnect();
    }
    if (cameraReconnectPending_) {
        DriveCameraReconnect();
    }

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

    // GPU fast path: NV12/YUY2 frames go straight to CUDA (conversion and
    // rotation on the GPU), skipping the per-pixel CPU work below. The CPU
    // path remains for the debug view, other subtypes, GPU-unavailable
    // passthrough, and any frame the raw path rejects.
    if (!debugViewEnabled_ && TryProcessRawFrameWithCuda(frame)) {
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
        // Recording and the periodic assistive grab both use the async
        // readback ring; nothing on this path blocks on the GPU anymore.
        HandleGpuFramePresented(width, height);
        UpdateProcessingStatusLabel();
        return;
    }

    if (!debugViewEnabled_) {
        // The CPU effects pipeline is deprecated: without CUDA the app
        // presents the unprocessed converted frame and reports that the GPU
        // is required. Recording, snapshots, and assistive readback continue
        // to work from the presentation buffer inside PresentFitted.
        const std::vector<uint8_t>& raw = cpuPipeline_.StageRaw();
        if (raw.empty()) {
            UpdateProcessingStatusLabel();
            return;
        }
        PresentFitted(raw.data(), width, height, true, zoomCenterX_, zoomCenterY_);
        return;
    }

    // Legacy CPU composite, kept as a diagnostic for the debug view only.
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

    ViewMapping mapping{};
    if (!ComputeViewMapping(srcWidth,
                            srcHeight,
                            renderWidget_->width(),
                            renderWidget_->height(),
                            centerXNorm,
                            centerYNorm,
                            cropToFill,
                            mapping)) {
        return;
    }

    presentationBuffer_.assign(static_cast<size_t>(mapping.targetWidth) * mapping.targetHeight * 4, 0);
    presentationWidth_ = mapping.targetWidth;
    presentationHeight_ = mapping.targetHeight;

    const UINT srcStride = srcWidth * 4;
    const UINT dstStride = mapping.targetWidth * 4;

    for (UINT y = 0; y < mapping.activeHeight; ++y) {
        const float sampleY = mapping.startY + static_cast<float>(y) * mapping.stepY;
        int srcYIndex = static_cast<int>(std::lroundf(sampleY));
        srcYIndex = std::clamp(srcYIndex, 0, static_cast<int>(srcHeight) - 1);
        uint8_t* dstRow = presentationBuffer_.data() +
                          (static_cast<size_t>(mapping.offsetY + y) * dstStride) +
                          mapping.offsetX * 4;
        const uint8_t* srcRow = data + static_cast<size_t>(srcYIndex) * srcStride;
        for (UINT x = 0; x < mapping.activeWidth; ++x) {
            const float sampleX = mapping.startX + static_cast<float>(x) * mapping.stepX;
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
        const float localX = (mapping.centerX - mapping.startX) / mapping.stepX;
        const float localY = (mapping.centerY - mapping.startY) / mapping.stepY;
        const float markerX = static_cast<float>(mapping.offsetX) + localX;
        const float markerY = static_cast<float>(mapping.offsetY) + localY;

        auto drawFilledCircle = [&](float cx, float cy, float radius,
                                    uint8_t b, uint8_t g, uint8_t r, uint8_t a) {
            const int minX = std::max(0, static_cast<int>(std::floor(cx - radius)));
            const int maxX = std::min(static_cast<int>(mapping.targetWidth) - 1,
                                      static_cast<int>(std::ceil(cx + radius)));
            const int minY = std::max(0, static_cast<int>(std::floor(cy - radius)));
            const int maxY = std::min(static_cast<int>(mapping.targetHeight) - 1,
                                      static_cast<int>(std::ceil(cy + radius)));
            const float radiusSq = radius * radius;
            for (int py = minY; py <= maxY; ++py) {
                const float dy = (static_cast<float>(py) + 0.5f) - cy;
                for (int px = minX; px <= maxX; ++px) {
                    const float dx = (static_cast<float>(px) + 0.5f) - cx;
                    if (dx * dx + dy * dy <= radiusSq) {
                        uint8_t* pixel = presentationBuffer_.data() +
                                         (static_cast<size_t>(py) * mapping.targetWidth + px) * 4;
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

    MaybeRequestAssistiveAnalysis(presentationBuffer_.data(), mapping.targetWidth, mapping.targetHeight);
    MaybeRecordFrame(presentationBuffer_.data(), mapping.targetWidth, mapping.targetHeight);
    presenter_->Present(presentationBuffer_.data(), mapping.targetWidth, mapping.targetHeight);
    UpdateProcessingStatusLabel();
}

QString OpenZoomApp::EnsureOutputSubdir(const QString& subdir)
{
    QDir base(QCoreApplication::applicationDirPath());
    QString outDirPath = base.filePath(QStringLiteral("output/%1").arg(subdir));
    QDir outDir(outDirPath);
    if (!outDir.exists()) {
        outDir.mkpath(QStringLiteral("."));
    }
    return outDir.absolutePath();
}

void OpenZoomApp::CaptureSnapshot(const uint8_t* data, UINT width, UINT height)
{
    if (!data || width == 0 || height == 0) {
        qWarning() << "Snapshot skipped: invalid buffer";
        return;
    }
    const QString dirPath = EnsureOutputSubdir(QStringLiteral("img"));
    const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"));
    const QString filename = QStringLiteral("IMG_%1.jpg").arg(timestamp);
    const QString fullPath = QDir(dirPath).filePath(filename);

    QImage image(data,
                 static_cast<int>(width),
                 static_cast<int>(height),
                 static_cast<int>(width) * 4,
                 QImage::Format_ARGB32);
    if (!image.save(fullPath, "JPG", 90)) {
        qWarning() << "Failed to save snapshot to" << fullPath;
    } else {
        qInfo() << "Saved snapshot to" << fullPath;
        if (assistiveRuntime_) {
            assistiveRuntime_->NoteCapturedPhoto(fullPath);
        }
    }
}

void OpenZoomApp::StopRecordingUi()
{
    recording_ = false;
    if (recordButton_) {
        QSignalBlocker block(recordButton_);
        recordButton_->setChecked(false);
        recordButton_->setText(QStringLiteral("Record"));
    }
}

// Delivers one BGRA frame to the recorder. On the GPU path the frames arrive
// from the async readback ring (one or more presents after they were
// requested) with their own dimensions; on the CPU passthrough path they come
// straight from the presentation buffer.
void OpenZoomApp::MaybeRecordFrame(const uint8_t* data, UINT width, UINT height)
{
    if (!recording_ || !data || width == 0 || height == 0) {
        return;
    }

    // Start recorder lazily on first frame so we have dimensions.
    if (!videoRecorder_.IsRecording()) {
        const QString dirPath = EnsureOutputSubdir(QStringLiteral("vid"));
        const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
        const QString filename = QStringLiteral("VID_%1.mp4").arg(timestamp);
        const QString fullPath = QDir(dirPath).filePath(filename);
        const std::wstring filePathW = fullPath.toStdWString();
        const UINT fps = 30;
        if (!videoRecorder_.Start(filePathW, width, height, fps)) {
            // Start() refuses when the disk is nearly full; surface its
            // explanation without a modal dialog.
            const QString reason = QString::fromStdString(videoRecorder_.LastError());
            qWarning() << "Failed to start recording:" << reason;
            StopRecordingUi();
            if (!reason.isEmpty()) {
                ShowStatusMessage(reason);
            }
            return;
        }
        recordingWidth_ = width;
        recordingHeight_ = height;
        recordingTimer_.restart();
        recordingFrameCount_ = 0;
        qInfo() << "Recording started:" << fullPath;
    }

    if (width != recordingWidth_ || height != recordingHeight_) {
        // The processed frame size changed mid-recording (rotation or window
        // change on the passthrough path). The sink cannot switch dimensions,
        // so finalize the file cleanly instead of feeding it garbage.
        qInfo() << "Recording stopped: frame size changed from"
                << recordingWidth_ << "x" << recordingHeight_ << "to" << width << "x" << height;
        videoRecorder_.Stop();
        StopRecordingUi();
        ShowStatusMessage(QStringLiteral("Recording stopped: the video size changed. "
                                         "The recording so far was saved."));
        return;
    }

    const size_t stride = static_cast<size_t>(width) * 4;
    if (!videoRecorder_.AddFrame(data, stride)) {
        const VideoRecorder::StopReason reason = videoRecorder_.LastStopReason();
        const QString message = QString::fromStdString(videoRecorder_.LastError());
        StopRecordingUi();
        if (reason == VideoRecorder::StopReason::DiskFull) {
            // The recorder already finalized the file; everything up to this
            // point is safe on disk. Informational, not an error dialog.
            qInfo() << "Recording finalized (disk full):" << message;
            ShowStatusMessage(message);
        } else {
            qWarning() << "Recording error:" << message;
            videoRecorder_.Stop();
            if (!message.isEmpty()) {
                ShowStatusMessage(message);
            }
        }
        return;
    }

    ++recordingFrameCount_;

    constexpr double kMaxSeconds = 12.0 * 3600.0;
    if (videoRecorder_.DurationSeconds() >= kMaxSeconds) {
        qInfo() << "Recording stopped: reached 12-hour limit";
        videoRecorder_.Stop();
        StopRecordingUi();
    }
}

} // namespace openzoom

#endif // _WIN32
