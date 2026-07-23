#ifdef _WIN32

#include "app_internal.hpp"

namespace openzoom {

void OpenZoomApp::SetZoomCenter(float normX, float normY, bool syncUi,
                                bool preservePresetSelection,
                                bool persist) {
    const float clampedX = std::clamp(normX, 0.0f, 1.0f);
    const float clampedY = std::clamp(normY, 0.0f, 1.0f);
    zoomCenterX_ = clampedX;
    zoomCenterY_ = clampedY;
    pipelineOrchestrator_->MarkViewportDirty();

    if (syncUi) {
        SuspendGuard suspendControlSync(suspendControlSync_);
        if (uiState_->zoomCenterXSlider_) {
            auto blockX = uiState_->BlockSignals(uiState_->zoomCenterXSlider_);
            uiState_->zoomCenterXSlider_->setValue(static_cast<int>(std::round(clampedX * kZoomFocusSliderScale)));
        }
        if (uiState_->zoomCenterYSlider_) {
            auto blockY = uiState_->BlockSignals(uiState_->zoomCenterYSlider_);
            uiState_->zoomCenterYSlider_->setValue(static_cast<int>(std::round(clampedY * kZoomFocusSliderScale)));
        }
    }
    if (persist) {
        SyncCurrentConfigToPersistence(preservePresetSelection);
    }
}

bool OpenZoomApp::HandlePanKey(int key, bool pressed) {
    if (!interactionController_) {
        return false;
    }
    const bool handled = interactionController_->HandlePanKey(key, pressed);
    if (handled) {
        pipelineOrchestrator_->NotifyViewportMotion();
        pipelineOrchestrator_->UpdateTimerPolicy();
    }
    return handled;
}

bool OpenZoomApp::HandlePanScroll(const QWheelEvent* wheelEvent) {
    if (!interactionController_) {
        return false;
    }
    return interactionController_->HandlePanScroll(wheelEvent);
}

bool OpenZoomApp::ApplyInputForces(double elapsedSeconds) {
    if (!interactionController_) {
        return false;
    }
    return interactionController_->ApplyInputForces(elapsedSeconds);
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
    if (uiState_->blurSigmaValueLabel_) {
        uiState_->blurSigmaValueLabel_->setText(sigmaText);
        uiState_->blurSigmaValueLabel_->setEnabled(blurActive);
    }
    if (uiState_->blurRadiusValueLabel_) {
        uiState_->blurRadiusValueLabel_->setText(QString::number(blurRadius_));
        uiState_->blurRadiusValueLabel_->setEnabled(blurActive);
    }
    if (uiState_->blurSigmaSlider_) {
        uiState_->blurSigmaSlider_->setEnabled(blurActive);
    }
    if (uiState_->blurRadiusSlider_) {
        auto block = uiState_->BlockSignals(uiState_->blurRadiusSlider_);
        uiState_->blurRadiusSlider_->setValue(blurRadius_);
        uiState_->blurRadiusSlider_->setEnabled(blurActive);
    }
    if (uiState_->blurCheckbox_) {
        auto block = uiState_->BlockSignals(uiState_->blurCheckbox_);
        uiState_->blurCheckbox_->setChecked(blurEnabled_);
        uiState_->blurCheckbox_->setEnabled(true);
    }
}

void OpenZoomApp::UpdateTemporalSmoothUi() {
    if (uiState_->temporalSmoothCheckbox_) {
        auto block = uiState_->BlockSignals(uiState_->temporalSmoothCheckbox_);
        uiState_->temporalSmoothCheckbox_->setChecked(temporalSmoothEnabled_);
    }
    if (uiState_->temporalSmoothSlider_) {
        uiState_->temporalSmoothSlider_->setEnabled(temporalSmoothEnabled_);
        const int sliderValue = static_cast<int>(std::round(temporalSmoothAlpha_ * 100.0f));
        auto block = uiState_->BlockSignals(uiState_->temporalSmoothSlider_);
        uiState_->temporalSmoothSlider_->setValue(std::clamp(sliderValue,
                                                  uiState_->temporalSmoothSlider_->minimum(),
                                                  uiState_->temporalSmoothSlider_->maximum()));
    }
    if (uiState_->temporalSmoothValueLabel_) {
        uiState_->temporalSmoothValueLabel_->setEnabled(temporalSmoothEnabled_);
        uiState_->temporalSmoothValueLabel_->setText(QString::number(temporalSmoothAlpha_, 'f', 2));
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
    if (uiState_->rotationCombo_) {
        auto block = uiState_->BlockSignals(uiState_->rotationCombo_);
        if (turns >= 0 && turns < uiState_->rotationCombo_->count()) {
            uiState_->rotationCombo_->setCurrentIndex(turns);
        }
    }
}

void OpenZoomApp::UpdateSpatialSharpenUi() {
    const bool enabled = spatialSharpenEnabled_;

    if (uiState_->spatialSharpenCheckbox_) {
        auto block = uiState_->BlockSignals(uiState_->spatialSharpenCheckbox_);
        uiState_->spatialSharpenCheckbox_->setChecked(enabled);
    }
    if (uiState_->spatialBackendCombo_) {
        auto block = uiState_->BlockSignals(uiState_->spatialBackendCombo_);
        uiState_->spatialBackendCombo_->setCurrentIndex(static_cast<int>(spatialUpscaler_));
        uiState_->spatialBackendCombo_->setEnabled(enabled);
    }
    if (uiState_->spatialSharpnessSlider_) {
        uiState_->spatialSharpnessSlider_->setEnabled(enabled);
        if (enabled) {
            auto block = uiState_->BlockSignals(uiState_->spatialSharpnessSlider_);
            uiState_->spatialSharpnessSlider_->setValue(static_cast<int>(std::round(spatialSharpness_ * 100.0f)));
        }
    }
    if (uiState_->spatialSharpnessValueLabel_) {
        uiState_->spatialSharpnessValueLabel_->setEnabled(enabled);
        uiState_->spatialSharpnessValueLabel_->setText(QString::number(spatialSharpness_, 'f', 2));
    }
}

void OpenZoomApp::UpdateProcessingStatusLabel() {
    if (!uiState_->processingStatusLabel_) {
        return;
    }

    QString text;
    QString detail;
    QString color;

    auto backendLabel = [this]() -> QString {
        if (autoTextClarityEnabled_ || backgroundFlattenEnabled_ ||
            adaptiveBinarizationEnabled_ || smartSharpenEnabled_ || claheEnabled_) {
            return QStringLiteral("Text Clarity");
        }
        if (cudaSurface_ && cudaSurface_->IsSuperResActive() &&
            superResPresentedLastFrame_) {
            return QStringLiteral("NVIDIA SuperRes");
        }
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

    if (!cameraActive_ &&
        pipelineOrchestrator_->IsCameraReconnectPending()) {
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
        if (pipelineOrchestrator_->FenceInteropEnabled()) {
            detail = QStringLiteral("Processing: GPU (fence interop, %1)").arg(backend);
        } else {
            detail = QStringLiteral("Processing: GPU (%1)").arg(backend);
        }
        if ((focusDetectionEnabled_ || autoTextClarityEnabled_) && cudaSurface_ &&
            cudaSurface_->HasFocusScore()) {
            detail.append(QStringLiteral(", focus %1")
                              .arg(cudaSurface_->LatestFocusScore(), 0, 'f', 4));
        }
        if (cudaSurface_ && !cudaSurface_->SuperResStatus().empty()) {
            detail.append(QStringLiteral(", %1")
                              .arg(QString::fromStdString(cudaSurface_->SuperResStatus())));
        }
        color = QStringLiteral("#1c9c3e");
    } else {
        // CPU effects path is deprecated: without the GPU pipeline the app
        // shows unprocessed passthrough video.
        text = QStringLiteral("GPU Required");
        detail = QStringLiteral("GPU required - processing disabled (showing raw video)");
        color = QStringLiteral("#c0392b");
    }

    if (recordingManager_ && recordingManager_->IsActive()) {
        text.append(QStringLiteral(" [REC]"));
        if (recordingManager_->CodecName().isEmpty()) {
            detail.append(QStringLiteral(" [REC original + processed, starting encoder]"));
        } else {
            detail.append(QStringLiteral(" [REC original + processed, %1]")
                              .arg(recordingManager_->CodecName()));
        }
    }

    if (ocrAssistEnabled_) {
        text.append(QStringLiteral(" [OCR]"));
        detail.append(QStringLiteral(" [OCR]"));
    }
    if (vlmAssistEnabled_) {
        text.append(QStringLiteral(" [VLM]"));
        detail.append(QStringLiteral(" [VLM]"));
    }

    // P8 frame timing (plan 11 Wave 1): rolling CPU tick average and the
    // sampled GPU kernel-chain ms, tooltip only (corner label stays short).
    const float frameTickAverageMs =
        pipelineOrchestrator_->FrameTickAverageMs();
    if (cameraActive_ && frameTickAverageMs >= 0.0f) {
        QString timing = QStringLiteral("\n%1 ms/frame - %2")
                             .arg(static_cast<double>(frameTickAverageMs), 0, 'f', 1)
                             .arg(usingCudaLastFrame_ ? QStringLiteral("CUDA")
                                                      : QStringLiteral("passthrough"));
        const float gpuMs = cudaSurface_ ? cudaSurface_->LastGpuFrameMs() : -1.0f;
        if (usingCudaLastFrame_ && gpuMs >= 0.0f) {
            timing.append(QStringLiteral(" - GPU %1 ms")
                              .arg(static_cast<double>(gpuMs), 0, 'f', 1));
        }
        detail.append(timing);
    }
    if (cameraActive_) {
        const double cameraFps = mediaCapture_.CurrentFrameRate();
        const QString cameraRate =
            cameraFps > 0.0
                ? QString::number(cameraFps, 'f',
                                  std::abs(cameraFps - std::round(cameraFps)) < 0.01
                                      ? 0
                                      : 1)
                : QStringLiteral("?");
        const UINT viewportWidth = presenter_ ? presenter_->ViewportWidth() : 0;
        const UINT viewportHeight = presenter_ ? presenter_->ViewportHeight() : 0;
        detail.append(
            QStringLiteral("\nCamera %1 FPS | Viewport %2/%3 FPS | Display %4 Hz"
                           "\nViewport %5x%6 | Scene %7x%8 | %9 | Missed %10")
                .arg(cameraRate)
                .arg(static_cast<double>(
                         pipelineOrchestrator_->MeasuredViewportRate()),
                     0,
                     'f',
                     0)
                .arg(pipelineOrchestrator_->EffectiveViewportRate())
                .arg(pipelineOrchestrator_->DisplayRefreshRate())
                .arg(viewportWidth)
                .arg(viewportHeight)
                .arg(processedFrameWidth_)
                .arg(processedFrameHeight_)
                .arg(pipelineOrchestrator_->ViewportFitMode() ==
                             settings::ViewportFitModeSetting::Fit
                         ? QStringLiteral("Fit")
                         : QStringLiteral("Fill"))
                .arg(presenter_ ? presenter_->MissedPresentCount() : 0));
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

    uiState_->processingStatusLabel_->setText(text);
    uiState_->processingStatusLabel_->setToolTip(detail);
    uiState_->processingStatusLabel_->setStyleSheet(QStringLiteral("color: %1;").arg(color));

    QString superResStatus = QStringLiteral("Off");
    bool superResActive = false;
    if (mlTextSuperResolutionEnabled_) {
        if (!mlTextSuperResolutionUltra1440p_ &&
            (!zoomEnabled_ || zoomAmount_ < 1.33f)) {
            superResStatus = QStringLiteral(
                "Waiting for 1.33x zoom; NVIDIA SuperRes supports 4/3x and above");
        } else {
            QString pipelineState = QStringLiteral("Starting NVIDIA Super Resolution");
            if (cudaSurface_ && !cudaSurface_->SuperResStatus().empty()) {
                pipelineState = QString::fromStdString(cudaSurface_->SuperResStatus());
                superResActive =
                    cudaSurface_->IsSuperResActive() &&
                    superResPresentedLastFrame_;
            }

            if (superResActive && cudaSurface_) {
                // Report the factor the stage actually snapped to (4/3, 1.5,
                // 2, 3 or 4); the residual zoom is applied by the GPU sampler.
                const SuperResRoiMetadata roi = cudaSurface_->SuperResRoi();
                const QString geometryDescription =
                    mlTextSuperResolutionUltra1440p_
                        ? QStringLiteral("Full camera frame -> Ultra scene cache")
                        : QStringLiteral("Source crop -> viewport target");
                superResStatus = QStringLiteral(
                                     "%1\nAI stage: %2x%3 -> %4x%5 (%6x)"
                                     "\n%7"
                                     "\nFinal magnification: %8x")
                                     .arg(pipelineState)
                                     .arg(cudaSurface_->SuperResSourceWidth())
                                     .arg(cudaSurface_->SuperResSourceHeight())
                                     .arg(roi.outputWidth)
                                     .arg(roi.outputHeight)
                                     .arg(static_cast<double>(cudaSurface_->SuperResFactor()), 0, 'f', 2)
                                     .arg(geometryDescription)
                                     .arg(zoomEnabled_ ? zoomAmount_ : 1.0f, 0, 'f', 2);
            } else if (cudaSurface_ && cudaSurface_->IsSuperResActive()) {
                superResStatus = QStringLiteral(
                    "NVIDIA SuperRes ROI cached; using the conventional "
                    "view while the AI crop catches up");
            } else {
                superResStatus = pipelineState;
            }
        }
    }
    if (mainWindow_) {
        mainWindow_->setSuperResStatus(
            superResStatus,
            superResActive,
            cudaSurface_ && cudaSurface_->IsSuperResPerformanceLimited());
    }
    if (superResStatus != lastSuperResStatus_) {
        qInfo().noquote() << "NVIDIA Super Resolution:" << superResStatus;
        lastSuperResStatus_ = superResStatus;
    }
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
    if (!uiState_->renderWidget_ || processedFrameWidth_ == 0 || processedFrameHeight_ == 0) {
        return false;
    }

    const float viewportWidth =
        static_cast<float>(std::max(1, uiState_->renderWidget_->width()));
    const float viewportHeight =
        static_cast<float>(std::max(1, uiState_->renderWidget_->height()));
    ViewTransform transform =
        ComputeViewTransform(processedFrameWidth_,
                             processedFrameHeight_,
                             static_cast<UINT>(viewportWidth),
                             static_cast<UINT>(viewportHeight),
                             zoomEnabled_ ? zoomAmount_ : 1.0f,
                             zoomCenterX_,
                             zoomCenterY_,
                             pipelineOrchestrator_->ViewportFitMode() ==
                                     settings::ViewportFitModeSetting::Fit
                                 ? ViewportFitMode::kFit
                                 : ViewportFitMode::kFill);
    if (!transform.valid) {
        return false;
    }

    const float viewportU =
        std::clamp(static_cast<float>(pos.x()) / viewportWidth, 0.0f, 1.0f);
    const float viewportV =
        std::clamp(static_cast<float>(pos.y()) / viewportHeight, 0.0f, 1.0f);
    if (viewportU < transform.destinationX ||
        viewportV < transform.destinationY ||
        viewportU > transform.destinationX + transform.destinationWidth ||
        viewportV > transform.destinationY + transform.destinationHeight) {
        return false;
    }
    const float localU =
        (viewportU - transform.destinationX) / transform.destinationWidth;
    const float localV =
        (viewportV - transform.destinationY) / transform.destinationHeight;
    outX = std::clamp(
        transform.sourceX + localU * transform.sourceWidth, 0.0f, 1.0f);
    outY = std::clamp(
        transform.sourceY + localV * transform.sourceHeight, 0.0f, 1.0f);
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


} // namespace openzoom

#endif // _WIN32
