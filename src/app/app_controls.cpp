#ifdef _WIN32

#include "app_internal.hpp"

namespace openzoom {

void OpenZoomApp::OnCameraSelectionChanged(int index) {
    if (index < 0 || static_cast<size_t>(index) >= cameras_.size()) {
        return;
    }

    // A manual camera pick always wins over an in-flight automatic reconnect.
    pipelineOrchestrator_->CancelCameraReconnect();
    settingsController_->MutableSettings().cameraIndex = index;
    RefreshCameraModesList(static_cast<size_t>(index));
    StartCameraCapture(static_cast<size_t>(index));
}

void OpenZoomApp::OnBlackWhiteToggled(bool checked) {
    blackWhiteEnabled_ = checked;
    if (uiState_->bwSlider_) {
        uiState_->bwSlider_->setEnabled(checked);
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
    pipelineOrchestrator_->MarkViewportDirty();
    if (uiState_->zoomSlider_) {
        uiState_->zoomSlider_->setEnabled(checked);
    }
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnZoomAmountChanged(int value) {
    zoomAmount_ = std::max(1.0f, static_cast<float>(value) / static_cast<float>(kZoomSliderScale));
    pipelineOrchestrator_->MarkViewportDirty();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnDebugViewToggled(bool checked) {
    debugViewEnabled_ = checked;
    if (uiState_->focusMarkerCheckbox_) {
        uiState_->focusMarkerCheckbox_->setEnabled(!checked);
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
    if (!uiState_->rotationCombo_) {
        return;
    }

    const int clamped = std::clamp(index, 0, 3);
    const int previous = ((rotationQuarterTurns_ % 4) + 4) % 4;
    if (clamped == previous) {
        return;
    }

    const int delta = (clamped - previous + 4) % 4;
    rotationQuarterTurns_ = clamped;
    settingsController_->MutableSettings().rotationQuarterTurns = rotationQuarterTurns_;

    float rotatedX = zoomCenterX_;
    float rotatedY = zoomCenterY_;
    RotateNormalizedPoint(zoomCenterX_, zoomCenterY_, delta, rotatedX, rotatedY);
    SetZoomCenter(rotatedX, rotatedY, true, true);

    cpuPipeline_.ResetTemporalHistory();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
        cudaSurface_->ResetStabilization();
        cudaSurface_->ResetKeystone();
        cudaSurface_->ResetTextClarityHistory();
    }
    UpdateKeystoneTrackingUi();
    ResetCudaFenceState();

    processedFrameWidth_ = 0;
    processedFrameHeight_ = 0;
    cpuSceneBuffer_.clear();
    cpuSceneWidth_ = 0;
    cpuSceneHeight_ = 0;
    cpuSceneReady_ = false;

    UpdateRotationUi();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnControlsCollapsedToggled(bool checked) {
    controlsCollapsed_ = !checked;
    if (uiState_->controlsContainer_) {
        uiState_->controlsContainer_->setVisible(checked);
    }
    if (uiState_->collapseButton_) {
        uiState_->collapseButton_->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        uiState_->collapseButton_->setText(checked ? "Hide Advanced Tuning" : "Advanced Tuning");
    }
    settingsController_->MutableSettings().controlsCollapsed = controlsCollapsed_;
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
    settingsController_->MutableSettings().virtualJoystick = virtualJoystickEnabled_;
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
    if (uiState_->blurRadiusSlider_ && snapped != value) {
        auto blocker = uiState_->BlockSignals(uiState_->blurRadiusSlider_);
        uiState_->blurRadiusSlider_->setValue(snapped);
    }
    blurRadius_ = snapped;
    UpdateBlurUiLabels();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OnFocusMarkerToggled(bool checked) {
    focusMarkerEnabled_ = checked;
    pipelineOrchestrator_->MarkViewportDirty();
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
    if (uiState_->spatialSharpnessValueLabel_) {
        uiState_->spatialSharpnessValueLabel_->setText(QString::number(spatialSharpness_, 'f', 2));
    }
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnTemporalSmoothToggled(bool checked) {
    temporalSmoothEnabled_ = checked;
    if (uiState_->temporalSmoothSlider_) {
        uiState_->temporalSmoothSlider_->setEnabled(checked);
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
    const int sliderMin = uiState_->temporalSmoothSlider_ ? uiState_->temporalSmoothSlider_->minimum() : 1;
    const int sliderMax = uiState_->temporalSmoothSlider_ ? uiState_->temporalSmoothSlider_->maximum() : 100;
    const int clamped = std::clamp(value, sliderMin, sliderMax);
    if (uiState_->temporalSmoothSlider_ && clamped != value) {
        auto block = uiState_->BlockSignals(uiState_->temporalSmoothSlider_);
        uiState_->temporalSmoothSlider_->setValue(clamped);
    }
    temporalSmoothAlpha_ = std::clamp(static_cast<float>(clamped) / 100.0f, 0.0f, 1.0f);
    if (uiState_->temporalSmoothValueLabel_) {
        uiState_->temporalSmoothValueLabel_->setText(QString::number(temporalSmoothAlpha_, 'f', 2));
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
    if (uiState_->stabilizationStrengthSlider_) {
        uiState_->stabilizationStrengthSlider_->setEnabled(checked);
    }
    if (cudaSurface_) {
        cudaSurface_->ResetStabilization();
        cudaSurface_->ResetKeystone();
        cudaSurface_->ResetTextClarityHistory();
    }
    UpdateKeystoneTrackingUi();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnStabilizationStrengthChanged(int value) {
    const int sliderMin = uiState_->stabilizationStrengthSlider_ ? uiState_->stabilizationStrengthSlider_->minimum() : 0;
    const int sliderMax = uiState_->stabilizationStrengthSlider_ ? uiState_->stabilizationStrengthSlider_->maximum() : 98;
    const int clamped = std::clamp(value, sliderMin, sliderMax);
    if (uiState_->stabilizationStrengthSlider_ && clamped != value) {
        auto block = uiState_->BlockSignals(uiState_->stabilizationStrengthSlider_);
        uiState_->stabilizationStrengthSlider_->setValue(clamped);
    }
    stabilizationStrength_ = std::clamp(static_cast<float>(clamped) / 100.0f, 0.0f, 0.98f);
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnKeystoneToggled(bool checked) {
    keystoneEnabled_ = checked;
    if (cudaSurface_) {
        cudaSurface_->ResetKeystone();
        cudaSurface_->ResetTextClarityHistory();
    }
    UpdateKeystoneTrackingUi();
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::OpenSetupAssistant()
{
    if (!mainWindow_) {
        return;
    }
    if (setupAssistantDialog_) {
        setupAssistantDialog_->show();
        setupAssistantDialog_->raise();
        setupAssistantDialog_->activateWindow();
        return;
    }

    setupAssistantDialog_ = new SetupAssistantDialog(
        settingsController_->MutableSettings().assistive.tesseractPath,
        settingsController_->MutableSettings().assistive.codexExecutablePath,
        settingsController_->MutableSettings().setupAssistantDeclined,
        mainWindow_.get());
    connect(setupAssistantDialog_, &QObject::destroyed, this, [this]() {
        setupAssistantDialog_ = nullptr;
    });
    connect(setupAssistantDialog_, &SetupAssistantDialog::TesseractPathChanged,
            this, [this](const QString& path) {
                settingsController_->MutableSettings().assistive.tesseractPath = path;
                assistiveManager_->ApplySettings(
                    settingsController_->MutableSettings().assistive);
                SavePersistentSettings();
            });
    connect(setupAssistantDialog_, &SetupAssistantDialog::CodexPathChanged,
            this, [this](const QString& path) {
                settingsController_->MutableSettings().assistive.codexExecutablePath = path;
                assistiveManager_->ApplySettings(
                    settingsController_->MutableSettings().assistive);
                SavePersistentSettings();
            });
    connect(setupAssistantDialog_, &SetupAssistantDialog::DeclinePreferenceChanged,
            this, [this](bool declined) {
                settingsController_->MutableSettings().setupAssistantDeclined = declined;
                SavePersistentSettings();
            });
    connect(setupAssistantDialog_, &SetupAssistantDialog::DependenciesChanged,
            this, [this]() {
                if (cudaSurface_) {
                    cudaSurface_->ResetSuperRes();
                }
                mainWindow_->setMaxineRuntimeInstalled(MaxineSuperRes::IsRuntimeInstalled());
                assistiveManager_->ApplySettings(
                    settingsController_->MutableSettings().assistive);
                UpdateProcessingStatusLabel();
            });
    setupAssistantDialog_->show();
}
void OpenZoomApp::OnKeystoneStepBack() {
    if (!keystoneEnabled_ || !cudaSurface_) {
        return;
    }
    if (cudaSurface_->StepKeystoneCorrection(-1)) {
        cudaSurface_->ResetTextClarityHistory();
        ShowStatusMessage(QStringLiteral("Screen correction moved back and tracking stopped."), 3500);
    }
    UpdateKeystoneTrackingUi();
}
void OpenZoomApp::OnKeystonePauseResume() {
    if (!keystoneEnabled_ || !cudaSurface_) {
        return;
    }
    const bool pause = !cudaSurface_->GetKeystoneTrackingState().paused;
    cudaSurface_->SetKeystoneTrackingPaused(pause);
    ShowStatusMessage(pause ? QStringLiteral("Automatic screen correction stopped.")
                            : QStringLiteral("Automatic screen correction continuing."),
                      3500);
    UpdateKeystoneTrackingUi();
}
void OpenZoomApp::OnKeystoneStepForward() {
    if (!keystoneEnabled_ || !cudaSurface_) {
        return;
    }
    if (cudaSurface_->StepKeystoneCorrection(1)) {
        cudaSurface_->ResetTextClarityHistory();
        const bool pending = cudaSurface_->GetKeystoneTrackingState().stepPending;
        ShowStatusMessage(pending ? QStringLiteral("Finding one new screen correction...")
                                  : QStringLiteral("Using the next screen correction."),
                          3500);
    }
    UpdateKeystoneTrackingUi();
}
void OpenZoomApp::UpdateKeystoneTrackingUi() {
    if (!mainWindow_) {
        return;
    }
    const bool available = cameraActive_ && cudaSurface_ && cudaSurface_->IsValid() &&
                           cudaPipelineAvailable_;
    const KeystoneTrackingState state = cudaSurface_
                                            ? cudaSurface_->GetKeystoneTrackingState()
                                            : KeystoneTrackingState{};
    mainWindow_->setKeystoneTrackingControls(keystoneEnabled_, available,
                                             state.paused, state.canStepBack,
                                             state.canStepForward, state.stepPending,
                                             state.position, state.count);
}
void OpenZoomApp::OnAutoContrastToggled(bool checked) {
    autoContrastEnabled_ = checked;
    if (uiState_->autoContrastStrengthSlider_) {
        uiState_->autoContrastStrengthSlider_->setEnabled(checked);
    }
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnAutoContrastStrengthChanged(int value) {
    autoContrastStrength_ = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
    SyncCurrentConfigToPersistence();
}
void OpenZoomApp::OnTextClarityControlsChanged() {
    const bool wasSuperResEnabled = mlTextSuperResolutionEnabled_;
    autoTextClarityEnabled_ = uiState_->textClarityCheckbox_ && uiState_->textClarityCheckbox_->isChecked();
    backgroundFlattenEnabled_ = uiState_->backgroundFlattenCheckbox_ && uiState_->backgroundFlattenCheckbox_->isChecked();
    backgroundFlattenStrength_ = uiState_->backgroundFlattenStrengthSlider_
                                     ? std::clamp(uiState_->backgroundFlattenStrengthSlider_->value() / 100.0f, 0.0f, 1.0f) : 0.8f;
    adaptiveBinarizationEnabled_ = uiState_->adaptiveBinarizationCheckbox_ && uiState_->adaptiveBinarizationCheckbox_->isChecked();
    sauvolaStrength_ = uiState_->sauvolaStrengthSlider_
                           ? std::clamp(uiState_->sauvolaStrengthSlider_->value() / 100.0f, 0.1f, 0.5f) : 0.28f;
    binarizationSoftness_ = uiState_->binarizationSoftnessSlider_
                                ? std::clamp(uiState_->binarizationSoftnessSlider_->value() / 100.0f, 0.0f, 0.25f) : 0.06f;
    textPolarityMode_ = uiState_->textPolarityCombo_ ? std::clamp(uiState_->textPolarityCombo_->currentIndex(), 0, 2) : 0;
    strokeWeight_ = uiState_->strokeWeightSlider_ ? std::clamp(uiState_->strokeWeightSlider_->value(), -3, 3) : 0;
    smartSharpenEnabled_ = uiState_->smartSharpenCheckbox_ && uiState_->smartSharpenCheckbox_->isChecked();
    smartSharpenStrength_ = uiState_->smartSharpenStrengthSlider_
                                ? std::clamp(uiState_->smartSharpenStrengthSlider_->value() / 100.0f, 0.0f, 1.0f) : 0.45f;
    claheEnabled_ = uiState_->claheCheckbox_ && uiState_->claheCheckbox_->isChecked();
    claheClipLimit_ = uiState_->claheClipLimitSlider_
                          ? std::clamp(uiState_->claheClipLimitSlider_->value() / 10.0f, 1.0f, 8.0f) : 2.0f;
    twoColorTextEnabled_ = uiState_->twoColorTextCheckbox_ && uiState_->twoColorTextCheckbox_->isChecked();
    textHysteresisEnabled_ = uiState_->textHysteresisCheckbox_ && uiState_->textHysteresisCheckbox_->isChecked();
    textHysteresisStrength_ = uiState_->textHysteresisStrengthSlider_
                                  ? std::clamp(uiState_->textHysteresisStrengthSlider_->value() / 100.0f, 0.0f, 0.25f) : 0.08f;
    selectiveSharpenEnabled_ = uiState_->selectiveSharpenCheckbox_ && uiState_->selectiveSharpenCheckbox_->isChecked();
    focusDetectionEnabled_ = uiState_->focusDetectionCheckbox_ && uiState_->focusDetectionCheckbox_->isChecked();
    focusThreshold_ = uiState_->focusThresholdSlider_
                          ? std::clamp(uiState_->focusThresholdSlider_->value() / 1000.0f, 0.001f, 0.1f) : 0.012f;
    glareSuppressionEnabled_ = uiState_->glareSuppressionCheckbox_ && uiState_->glareSuppressionCheckbox_->isChecked();
    glareSuppressionStrength_ = uiState_->glareSuppressionStrengthSlider_
                                    ? std::clamp(uiState_->glareSuppressionStrengthSlider_->value() / 100.0f, 0.0f, 1.0f) : 0.5f;
#if OPENZOOM_ENABLE_TEXT_SR
    if (uiState_->mlTextSuperResolutionCheckbox_ && uiState_->mlTextSuperResolutionCheckbox_->isChecked()) {
        mlTextSuperResolutionUltra1440p_ =
            uiState_->mlTextSuperResolutionUltra1440pCheckbox_ &&
            uiState_->mlTextSuperResolutionUltra1440pCheckbox_->isChecked();
        if (mlTextSuperResolutionUltra1440p_ &&
            uiState_->mlTextSuperResolutionPrefer2xCheckbox_ &&
            uiState_->mlTextSuperResolutionPrefer2xCheckbox_->isChecked()) {
            auto block = uiState_->BlockSignals(
                uiState_->mlTextSuperResolutionPrefer2xCheckbox_);
            uiState_->mlTextSuperResolutionPrefer2xCheckbox_->setChecked(false);
        }
        mlTextSuperResolutionPrefer2x_ =
            !mlTextSuperResolutionUltra1440p_ &&
            uiState_->mlTextSuperResolutionPrefer2xCheckbox_ &&
            uiState_->mlTextSuperResolutionPrefer2xCheckbox_->isChecked();
        if (uiState_->mlTextSuperResolutionStrengthSlider_ &&
            uiState_->mlTextSuperResolutionStrengthSlider_->value() <= 0) {
            auto block = uiState_->BlockSignals(uiState_->mlTextSuperResolutionStrengthSlider_);
            uiState_->mlTextSuperResolutionStrengthSlider_->setValue(65);
        }
        if (!mlTextSuperResolutionUltra1440p_ &&
            uiState_->zoomCheckbox_ && !uiState_->zoomCheckbox_->isChecked()) {
            auto block = uiState_->BlockSignals(uiState_->zoomCheckbox_);
            uiState_->zoomCheckbox_->setChecked(true);
        }
        if (!mlTextSuperResolutionUltra1440p_) {
            zoomEnabled_ = true;
        }
        if (!mlTextSuperResolutionUltra1440p_ && uiState_->zoomSlider_) {
            uiState_->zoomSlider_->setEnabled(true);
            // NVIDIA's smallest supported SuperRes ratio is 4/3. The slider
            // stores hundredths, so 1.33 is the closest user-facing value;
            // the CUDA stage itself uses the exact 4/3 ratio.
            const int minimumZoom =
                mlTextSuperResolutionPrefer2x_ ? 200 : 133;
            if (uiState_->zoomSlider_->value() < minimumZoom) {
                auto block = uiState_->BlockSignals(uiState_->zoomSlider_);
                uiState_->zoomSlider_->setValue(minimumZoom);
            }
            zoomAmount_ = std::max(
                static_cast<float>(minimumZoom) /
                    static_cast<float>(kZoomSliderScale),
                static_cast<float>(uiState_->zoomSlider_->value()) /
                    static_cast<float>(kZoomSliderScale));
        } else if (!mlTextSuperResolutionUltra1440p_) {
            zoomAmount_ = std::max(
                zoomAmount_, mlTextSuperResolutionPrefer2x_ ? 2.0f : 1.33f);
        }
    } else {
        mlTextSuperResolutionUltra1440p_ =
            uiState_->mlTextSuperResolutionUltra1440pCheckbox_ &&
            uiState_->mlTextSuperResolutionUltra1440pCheckbox_->isChecked();
        mlTextSuperResolutionPrefer2x_ =
            !mlTextSuperResolutionUltra1440p_ &&
            uiState_->mlTextSuperResolutionPrefer2xCheckbox_ &&
            uiState_->mlTextSuperResolutionPrefer2xCheckbox_->isChecked();
    }
    mlTextSuperResolutionEnabled_ = uiState_->mlTextSuperResolutionCheckbox_ && uiState_->mlTextSuperResolutionCheckbox_->isChecked();
    mlTextSuperResolutionStrength_ = uiState_->mlTextSuperResolutionStrengthSlider_
                                         ? std::clamp(uiState_->mlTextSuperResolutionStrengthSlider_->value() / 100.0f,
                                                      0.0f, 1.0f)
                                         : 0.65f;
#else
    mlTextSuperResolutionEnabled_ = false;
#endif
    if (wasSuperResEnabled && !mlTextSuperResolutionEnabled_) {
        superResPerformanceOverride_ = false;
        if (mainWindow_) {
            mainWindow_->setSuperResPerformanceOverrideChecked(false);
        }
    }

    if (uiState_->backgroundFlattenStrengthSlider_) uiState_->backgroundFlattenStrengthSlider_->setEnabled(backgroundFlattenEnabled_ || autoTextClarityEnabled_);
    if (uiState_->sauvolaStrengthSlider_) uiState_->sauvolaStrengthSlider_->setEnabled(adaptiveBinarizationEnabled_ || autoTextClarityEnabled_);
    if (uiState_->binarizationSoftnessSlider_) uiState_->binarizationSoftnessSlider_->setEnabled(adaptiveBinarizationEnabled_ || autoTextClarityEnabled_);
    if (uiState_->smartSharpenStrengthSlider_) uiState_->smartSharpenStrengthSlider_->setEnabled(smartSharpenEnabled_ || autoTextClarityEnabled_);
    if (uiState_->claheClipLimitSlider_) uiState_->claheClipLimitSlider_->setEnabled(claheEnabled_);
    if (uiState_->textHysteresisStrengthSlider_) uiState_->textHysteresisStrengthSlider_->setEnabled(textHysteresisEnabled_ || autoTextClarityEnabled_);
    if (uiState_->focusThresholdSlider_) uiState_->focusThresholdSlider_->setEnabled(focusDetectionEnabled_ || autoTextClarityEnabled_);
    if (uiState_->glareSuppressionStrengthSlider_) uiState_->glareSuppressionStrengthSlider_->setEnabled(glareSuppressionEnabled_ || autoTextClarityEnabled_);
    if (uiState_->mlTextSuperResolutionStrengthSlider_) {
        uiState_->mlTextSuperResolutionStrengthSlider_->setEnabled(
            mlTextSuperResolutionEnabled_ && mainWindow_ &&
            mainWindow_->isMaxineRuntimeInstalled());
    }
    if (uiState_->mlTextSuperResolutionPrefer2xCheckbox_) {
        uiState_->mlTextSuperResolutionPrefer2xCheckbox_->setEnabled(
            mlTextSuperResolutionEnabled_ && mainWindow_ &&
            mainWindow_->isMaxineRuntimeInstalled());
    }
    if (uiState_->mlTextSuperResolutionUltra1440pCheckbox_) {
        uiState_->mlTextSuperResolutionUltra1440pCheckbox_->setEnabled(
            mlTextSuperResolutionEnabled_ && mainWindow_ &&
            mainWindow_->isMaxineRuntimeInstalled());
    }
    if (cudaSurface_) {
        cudaSurface_->ResetTextClarityHistory();
        // SuperRes strength/mode are load-time SDK parameters. Recreate the
        // effect under the surface's stream-synchronization discipline after
        // profile or Advanced Text Clarity changes.
        cudaSurface_->ResetSuperRes();
        cudaSurface_->SetSuperResPerformanceOverride(superResPerformanceOverride_);
    }
    UpdateProcessingStatusLabel();
    SyncCurrentConfigToPersistence();
}

void OpenZoomApp::SetSuperResPerformanceOverride(bool enabled) {
#if OPENZOOM_ENABLE_TEXT_SR
    if (!mlTextSuperResolutionEnabled_ || !cudaSurface_) {
        return;
    }
    superResPerformanceOverride_ = enabled;
    cudaSurface_->SetSuperResPerformanceOverride(enabled);
    ShowStatusMessage(enabled
                          ? QStringLiteral("NVIDIA Super Resolution performance limit ignored.")
                          : QStringLiteral("NVIDIA Super Resolution performance limit restored."));
#else
    Q_UNUSED(enabled);
#endif
}
void OpenZoomApp::OnDisplayColorSchemeChanged() {
    if (uiState_->displayColorPicker_) {
        displayColorScheme_ = color_schemes::NormalizeColorScheme(
            uiState_->displayColorPicker_->currentScheme(), displayColorMode_);
    }
    displayColorMode_ = displayColorScheme_.legacyMode >= 0
                            ? displayColorScheme_.legacyMode
                            : 0;
    displayColorLut_ = color_schemes::BuildColorLut(displayColorScheme_);
    ++displayColorLutGeneration_;
    if (displayColorLutGeneration_ == 0) {
        displayColorLutGeneration_ = 1;
    }
    if (uiState_->displayColorPicker_ && uiState_->displayColorPicker_->hasCustomScheme()) {
        settingsController_->MutableSettings().customColorScheme = uiState_->displayColorPicker_->customScheme();
    }
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

} // namespace openzoom

#endif // _WIN32
