#ifdef _WIN32

#include "app_internal.hpp"

namespace openzoom {

namespace {

struct SuperResCacheExtent {
    UINT width{};
    UINT height{};
};

SuperResCacheExtent ComputeSuperResCacheExtent(UINT sceneWidth,
                                               UINT sceneHeight,
                                               bool ultra1440p) {
    SuperResCacheExtent extent{sceneWidth, sceneHeight};
    if (!ultra1440p || sceneWidth == 0 || sceneHeight == 0) {
        return extent;
    }

    const UINT maxWidth = sceneWidth >= sceneHeight ? 2560u : 1440u;
    const UINT maxHeight = sceneWidth >= sceneHeight ? 1440u : 2560u;
    struct Scale {
        UINT numerator;
        UINT denominator;
    };
    constexpr Scale kScales[] = {
        {4u, 1u}, {3u, 1u}, {2u, 1u}, {3u, 2u}, {4u, 3u}};
    for (const Scale scale : kScales) {
        const std::uint64_t scaledWidth =
            static_cast<std::uint64_t>(sceneWidth) * scale.numerator;
        const std::uint64_t scaledHeight =
            static_cast<std::uint64_t>(sceneHeight) * scale.numerator;
        if (scaledWidth % scale.denominator != 0u ||
            scaledHeight % scale.denominator != 0u) {
            continue;
        }
        const std::uint64_t candidateWidth =
            scaledWidth / scale.denominator;
        const std::uint64_t candidateHeight =
            scaledHeight / scale.denominator;
        if (candidateWidth <= maxWidth && candidateHeight <= maxHeight) {
            extent = {static_cast<UINT>(candidateWidth),
                      static_cast<UINT>(candidateHeight)};
            break;
        }
    }
    return extent;
}

} // namespace

void OpenZoomApp::EnumerateCameras() {
    cameras_ = mediaCapture_.EnumerateCameras();
    selectedCameraIndex_ = cameras_.empty() ? -1 : 0;
}

void OpenZoomApp::PopulateCameraCombo() {
    if (!uiState_->cameraCombo_) {
        return;
    }

    uiState_->cameraCombo_->clear();
    for (const auto& camera : cameras_) {
        uiState_->cameraCombo_->addItem(QString::fromWCharArray(camera.name.c_str()));
    }
}

void OpenZoomApp::RefreshCameraModesList(size_t index) {
    if (!uiState_->cameraModesList_) {
        return;
    }
    uiState_->cameraModesList_->clear();
    if (index >= cameras_.size()) {
        return;
    }

    const auto formats = mediaCapture_.EnumerateFormats(cameras_[index]);
    if (formats.empty()) {
        const std::string& detail = mediaCapture_.LastError();
        if (!detail.empty()) {
            uiState_->cameraModesList_->addItem(QStringLiteral("Modes unavailable (%1)").arg(QString::fromStdString(detail)));
        } else {
            uiState_->cameraModesList_->addItem(QStringLiteral("No modes reported"));
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
        uiState_->cameraModesList_->addItem(line);
    }
}

void OpenZoomApp::ResetCudaFenceState() {
    const UINT64 baseValue = presenter_ ? presenter_->GetLastSignaledFenceValue() : 0;
    pipelineOrchestrator_->ResetFence(baseValue);
    if (recordingManager_) {
        recordingManager_->ClearPendingReadbacks();
    }
}

// S6b recovery policy: one failed ProcessFrame only rolls the fence
// reservation back (RunCudaPipeline calls CudaFailed() before this). Three
// consecutive failures trigger a full resync — drain the graphics queue and
// re-seed the fence timeline — plus a single status message, so a persistent
// CUDA failure can neither wedge the present loop nor spam the user.
void OpenZoomApp::HandleCudaProcessingFailure() {
    if (pipelineOrchestrator_->RecordCudaFailure() == 3) {
        if (presenter_) {
            presenter_->WaitForIdle();
        }
        ResetCudaFenceState();
        qWarning() << "CUDA processing failed 3 times in a row; fence state resynced";
        ShowStatusMessage(QStringLiteral(
            "GPU processing keeps failing - showing unprocessed video."));
    }
}

bool OpenZoomApp::EnsureCudaSurface(UINT width, UINT height) {
    if (!presenter_ || !uiState_->renderWidget_ || !uiState_->renderWidget_->isPresenterReady()) {
        qWarning() << "CUDA surface unavailable: presenter or render widget not ready";
        return false;
    }

    const SuperResCacheExtent superResExtent =
        ComputeSuperResCacheExtent(
            width,
            height,
            mlTextSuperResolutionEnabled_ &&
                mlTextSuperResolutionUltra1440p_);
    if (cudaSurface_ && cudaSurface_->IsValid() &&
        cudaSurfaceWidth_ == width && cudaSurfaceHeight_ == height &&
        cudaSuperResWidth_ == superResExtent.width &&
        cudaSuperResHeight_ == superResExtent.height) {
        return true;
    }

    // Drain the graphics queue first: pipelined presents may still be copying
    // from the shared texture we are about to release.
    presenter_->WaitForIdle();
    cudaSurface_.reset();
    cudaSharedTexture_.Reset();
    cudaSuperResTexture_.Reset();
    cudaSceneReady_ = false;
    cudaSurfaceWidth_ = 0;
    cudaSurfaceHeight_ = 0;
    cudaSuperResWidth_ = 0;
    cudaSuperResHeight_ = 0;
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
        D3D12_RESOURCE_DESC superResDesc = desc;
        superResDesc.Width = superResExtent.width;
        superResDesc.Height = superResExtent.height;
        ThrowIfFailed(device->CreateCommittedResource(&heapProps,
                                                      D3D12_HEAP_FLAG_SHARED,
                                                      &superResDesc,
                                                      D3D12_RESOURCE_STATE_COMMON,
                                                      nullptr,
                                                      IID_PPV_ARGS(&cudaSuperResTexture_)),
                      "Failed to create CUDA SuperRes cache texture");

        auto surface = std::make_unique<CudaInteropSurface>(
            cudaSharedTexture_.Get(),
            cudaSuperResTexture_.Get(),
            presenter_->GetFence());
        if (!surface || !surface->IsValid()) {
            if (surface) {
                const std::string& err = surface->LastError();
                if (!err.empty()) {
                    qWarning() << "CUDA surface detail:" << err.c_str();
                }
            }
            cudaSharedTexture_.Reset();
            cudaSuperResTexture_.Reset();
            qWarning() << "CUDA surface initialization failed: surface invalid"
                       << "(requested" << width << "x" << height << ")";
            return false;
        }

        surface->SetSuperResPerformanceOverride(superResPerformanceOverride_);
        cudaSurface_ = std::move(surface);
        cudaSurfaceWidth_ = width;
        cudaSurfaceHeight_ = height;
        cudaSuperResWidth_ = superResExtent.width;
        cudaSuperResHeight_ = superResExtent.height;
        cudaPipelineAvailable_ = true;
        UpdateKeystoneTrackingUi();
        pipelineOrchestrator_->SetFenceInteropEnabled(
            cudaSurface_->HasExternalSemaphore());
        if (pipelineOrchestrator_->FenceInteropEnabled()) {
            const UINT64 baseValue = presenter_->GetLastSignaledFenceValue();
            pipelineOrchestrator_->ResetFence(baseValue);
            pipelineOrchestrator_->SetFenceInteropEnabled(true);
            qInfo() << "CUDA fence interop enabled; base fence value"
                    << static_cast<unsigned long long>(baseValue);
        } else {
            qInfo() << "CUDA surface ready without fence interop";
        }
        qInfo() << "CUDA surface ready for" << width << "x" << height
                << "; SuperRes cache" << superResExtent.width << "x"
                << superResExtent.height;
        return true;
    } catch (...) {
        cudaSurface_.reset();
        cudaSharedTexture_.Reset();
        cudaSuperResTexture_.Reset();
        cudaSceneReady_ = false;
        cudaSurfaceWidth_ = 0;
        cudaSurfaceHeight_ = 0;
        cudaSuperResWidth_ = 0;
        cudaSuperResHeight_ = 0;
        cudaPipelineAvailable_ = false;
        UpdateKeystoneTrackingUi();
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
bool OpenZoomApp::TryProcessRawFrameWithCuda(const MediaFrame& frame,
                                             CapturedFrame* originalFrame) {
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
    PresentLatestCudaScene(true, originalFrame);
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
    // Viewport zoom is a presentation transform. Stateful CUDA stages process
    // each camera frame once at full-scene geometry.
    settings.enableZoom = false;
    // The requested geometry is still supplied so the camera-clock SuperRes
    // cache can build an ROI without applying zoom to the full scene.
    settings.zoomAmount = zoomEnabled_ ? zoomAmount_ : 1.0f;
    settings.zoomCenterX = zoomCenterX_;
    settings.zoomCenterY = zoomCenterY_;
    settings.enableBlur = blurEnabled_;
    settings.blurRadius = std::max(blurRadius_, 0);
    settings.blurSigma = blurSigma_;
    settings.drawFocusMarker = false;
    settings.enableSpatialSharpen = spatialSharpenEnabled_;
    settings.spatialUpscaler = spatialUpscaler_;
    settings.spatialSharpness = spatialSharpness_;
    settings.stagingFormat = cudaBufferFormat_;
    settings.enableTemporalSmoothing = temporalSmoothEnabled_;
    settings.temporalSmoothingAlpha = temporalSmoothAlpha_;
    settings.enableStabilization = stabilizationEnabled_;
    settings.stabilizationStrength = stabilizationStrength_;
    if (displayColorScheme_.id == QStringLiteral("normal")) {
        settings.displayColorTransform = DisplayColorTransform::kNone;
    } else if (displayColorScheme_.id == QStringLiteral("inverted")) {
        settings.displayColorTransform = DisplayColorTransform::kInvert;
    } else {
        settings.displayColorTransform = DisplayColorTransform::kLumaLut;
    }
    settings.displayColorLut = displayColorLut_.data();
    settings.displayColorLutGeneration = displayColorLutGeneration_;
    settings.textForegroundBgra = color_schemes::TextForegroundBgra(displayColorScheme_);
    settings.textBackgroundBgra = color_schemes::TextBackgroundBgra(displayColorScheme_);
    settings.contrast = contrast_;
    settings.brightness = brightness_;
    settings.enableKeystone = keystoneEnabled_;
    settings.enableAutoContrast = autoContrastEnabled_;
    settings.autoContrastStrength = autoContrastStrength_;
    settings.enableAutoTextClarity = autoTextClarityEnabled_;
    settings.enableBackgroundFlatten = backgroundFlattenEnabled_;
    settings.backgroundFlattenStrength = backgroundFlattenStrength_;
    settings.enableAdaptiveBinarization = adaptiveBinarizationEnabled_;
    settings.sauvolaStrength = sauvolaStrength_;
    settings.binarizationSoftness = binarizationSoftness_;
    settings.textPolarityMode = textPolarityMode_;
    settings.strokeWeight = strokeWeight_;
    settings.enableSmartSharpen = smartSharpenEnabled_;
    settings.smartSharpenStrength = smartSharpenStrength_;
    settings.enableClahe = claheEnabled_;
    settings.claheClipLimit = claheClipLimit_;
    settings.enableTwoColorText = twoColorTextEnabled_;
    settings.enableTextHysteresis = textHysteresisEnabled_;
    settings.textHysteresisStrength = textHysteresisStrength_;
    settings.enableSelectiveSharpen = selectiveSharpenEnabled_;
    settings.enableFocusDetection = focusDetectionEnabled_;
    settings.focusThreshold = focusThreshold_;
    settings.enableGlareSuppression = glareSuppressionEnabled_;
    settings.glareSuppressionStrength = glareSuppressionStrength_;
    settings.enableMlSuperRes = mlTextSuperResolutionEnabled_;
    settings.mlSuperResStrength = mlTextSuperResolutionStrength_;
    settings.mlSuperResUltra1440p = mlTextSuperResolutionUltra1440p_;

    // Fence choreography is owned by FenceSequencer (S6b contract in
    // pipeline_orchestrator.hpp):
    // BeginCudaFrame() re-seeds from the presenter and reserves the CUDA
    // signal value; the reservation is committed only after ProcessFrame
    // actually enqueued the signal.
    FenceSyncParams cudaSyncParams{};
    if (pipelineOrchestrator_->FenceInteropEnabled()) {
        const FenceSequencer::CudaTicket ticket =
            pipelineOrchestrator_->Fence().BeginCudaFrame(
                presenter_->GetLastSignaledFenceValue());
        cudaSyncParams.enable = true;
        // Async readbacks copy from the shared texture on the graphics queue;
        // CUDA must not write the next frame until both the present and the
        // newest readback copy have retired (GPU-side wait only).
        cudaSyncParams.waitValue = ticket.waitValue;
        cudaSyncParams.signalValue = ticket.signalValue;
    }

    if (!cudaSurface_->ProcessFrame(input, settings, cudaSyncParams)) {
        pipelineOrchestrator_->Fence().CudaFailed();
        cudaPipelineAvailable_ = false;
        UpdateKeystoneTrackingUi();
        qWarning() << "CUDA pipeline processing failed, falling back to CPU";
        HandleCudaProcessingFailure();
        usingCudaLastFrame_ = false;
        cudaSceneReady_ = false;
        return false;
    }
    pipelineOrchestrator_->ResetCudaFailures();

    if (cudaSyncParams.enable) {
        pipelineOrchestrator_->Fence().CudaSignaled();
    }

    cudaPipelineAvailable_ = true;
    UpdateKeystoneTrackingUi();
    Q_UNUSED(presentWidth);
    Q_UNUSED(presentHeight);
    usingCudaLastFrame_ = true;
    cudaSceneReady_ = true;
    pipelineOrchestrator_->MarkViewportDirty();
    return true;
}

void OpenZoomApp::PresentLatestCudaScene(bool newCameraFrame,
                                         CapturedFrame* originalFrame) {
    if (!cudaSceneReady_ || !cudaSharedTexture_ || !presenter_ ||
        !presenter_->IsInitialized() ||
        processedFrameWidth_ == 0 || processedFrameHeight_ == 0) {
        return;
    }

    const UINT viewportWidth = presenter_->ViewportWidth();
    const UINT viewportHeight = presenter_->ViewportHeight();
    ViewTransform transform =
        ComputeViewTransform(processedFrameWidth_,
                             processedFrameHeight_,
                             viewportWidth,
                             viewportHeight,
                             zoomEnabled_ ? zoomAmount_ : 1.0f,
                             zoomCenterX_,
                             zoomCenterY_,
                             pipelineOrchestrator_->ViewportFitMode() ==
                                     settings::ViewportFitModeSetting::Fit
                                 ? ViewportFitMode::kFit
                                 : ViewportFitMode::kFill);
    if (!transform.valid) {
        return;
    }

    ID3D12Resource* presentationTexture = cudaSharedTexture_.Get();
    UINT presentationSourceWidth = processedFrameWidth_;
    UINT presentationSourceHeight = processedFrameHeight_;
    float presentationFocusX = zoomCenterX_;
    float presentationFocusY = zoomCenterY_;
    const SuperResRoiMetadata superResRoi =
        cudaSurface_ ? cudaSurface_->SuperResRoi() : SuperResRoiMetadata{};
    const NormalizedSourceRect roiRect{
        superResRoi.sourceX,
        superResRoi.sourceY,
        superResRoi.sourceWidth,
        superResRoi.sourceHeight};
    ViewTransform roiTransform;
    bool presentingSuperRes = false;
    if (superResRoi.valid && cudaSuperResTexture_ &&
        RemapViewTransformToSourceRect(transform, roiRect, roiTransform)) {
        transform = roiTransform;
        presentationFocusX =
            (zoomCenterX_ - superResRoi.sourceX) /
            superResRoi.sourceWidth;
        presentationFocusY =
            (zoomCenterY_ - superResRoi.sourceY) /
            superResRoi.sourceHeight;
        presentationTexture = cudaSuperResTexture_.Get();
        presentationSourceWidth = superResRoi.outputWidth;
        presentationSourceHeight = superResRoi.outputHeight;
        presentingSuperRes = true;
    }
    presentationFocusX = std::clamp(presentationFocusX, 0.0f, 1.0f);
    presentationFocusY = std::clamp(presentationFocusY, 0.0f, 1.0f);

    DrainCompletedGpuReadbacks();
    if (pendingPhotoReadbackId_ != 0 &&
        pendingPhotoReadbackTimer_.isValid() &&
        pendingPhotoReadbackTimer_.elapsed() > 1000) {
        // Resize drops old-dimension presenter readbacks by contract. Retry a
        // photo whose request never returned instead of leaving capture stuck.
        pendingPhotoReadbackId_ = 0;
        pendingPhotoOriginal_ = {};
        pendingPhotoReadbackTimer_.invalidate();
        photoCapturePending_ = true;
    }
    const bool recordingActive =
        newCameraFrame && recordingManager_ && recordingManager_->IsActive();
    const bool assistiveWanted =
        newCameraFrame && assistiveManager_->WantsPeriodicReadback(debugViewEnabled_);
    const bool photoWanted =
        newCameraFrame && photoCapturePending_ &&
        originalFrame && originalFrame->IsValid();

    FenceSyncParams presentSync{};
    if (pipelineOrchestrator_->FenceInteropEnabled()) {
        presentSync.enable = true;
        presentSync.waitValue =
            pipelineOrchestrator_->Fence().LastCudaSignal();
        presentSync.signalValue =
            pipelineOrchestrator_->Fence().BeginGraphicsFrame(
                presenter_->GetLastSignaledFenceValue());
    }
    ViewportPresentationOptions presentationOptions{};
    presentationOptions.drawFocusMarker = focusMarkerEnabled_;
    presentationOptions.focusX = presentationFocusX;
    presentationOptions.focusY = presentationFocusY;
    presentationOptions.requestReadback =
        recordingActive || assistiveWanted || photoWanted;
    UINT64 readbackRequestId = 0;
    const bool presented = presenter_->PresentSceneTexture(
        presentationTexture,
        presentationSourceWidth,
        presentationSourceHeight,
        transform,
        presentSync.enable ? &presentSync : nullptr,
        &presentationOptions,
        &readbackRequestId);
    if (!presented) {
        pipelineOrchestrator_->MarkViewportDirty();
        return;
    }
    if (presentSync.enable) {
        // PresentSceneTexture also signals its internal frame-slot value.
        // Adopt the actual newest value after every present so viewport-only
        // draws and the next CUDA frame share one strictly monotonic timeline.
        pipelineOrchestrator_->Fence().GraphicsSignaled(
            presenter_->GetLastSignaledFenceValue());
    }
    if (readbackRequestId != 0) {
        if (recordingActive && originalFrame && originalFrame->IsValid()) {
            recordingManager_->StorePendingOriginal(
                readbackRequestId, *originalFrame);
        }
        if (photoWanted) {
            pendingPhotoReadbackId_ = readbackRequestId;
            pendingPhotoOriginal_ = *originalFrame;
            pendingPhotoReadbackTimer_.restart();
            photoCapturePending_ = false;
        }
        pipelineOrchestrator_->Fence().ReadbackObserved(
            presenter_->GetLastSignaledFenceValue());
    }
    pipelineOrchestrator_->MarkViewportPresented();
    superResPresentedLastFrame_ = presentingSuperRes;

    if (newCameraFrame) {
        UpdateProcessingStatusLabel();
    }
}

// Drain viewport-sized copies produced by PresentLatestCudaScene. Recording,
// photos, and assistive analysis therefore consume the same crop and aspect
// mapping shown on screen rather than the uncropped CUDA scene texture.
void OpenZoomApp::DrainCompletedGpuReadbacks() {
    if (!presenter_) {
        return;
    }

    UINT readbackWidth = 0;
    UINT readbackHeight = 0;
    UINT64 requestId = 0;
    while (presenter_->TryGetCompletedReadback(asyncReadbackBuffer_,
                                               readbackWidth,
                                               readbackHeight,
                                               &requestId)) {
        if (requestId == pendingPhotoReadbackId_ &&
            pendingPhotoOriginal_.IsValid()) {
            SaveCapturedPhotoPair(asyncReadbackBuffer_.data(),
                                  readbackWidth,
                                  readbackHeight,
                                  pendingPhotoOriginal_);
            pendingPhotoReadbackId_ = 0;
            pendingPhotoOriginal_ = {};
            pendingPhotoReadbackTimer_.invalidate();
        }
        if (recordingManager_) {
            recordingManager_->HandleProcessedReadback(requestId,
                                                       asyncReadbackBuffer_.data(),
                                                       readbackWidth,
                                                       readbackHeight);
        }
        const bool focusGateEnabled = focusDetectionEnabled_ || autoTextClarityEnabled_;
        const bool focusAcceptable =
            !focusGateEnabled || !cudaSurface_ ||
            cudaSurface_->IsFocusAcceptable(focusThreshold_);
        assistiveManager_->MaybeRequestAnalysis(asyncReadbackBuffer_.data(),
                                                 readbackWidth,
                                                 readbackHeight,
                                                 debugViewEnabled_,
                                                 focusGateEnabled,
                                                 focusAcceptable);
    }
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
        cudaSurface_->ResetTextClarityHistory();
    }
    UpdateKeystoneTrackingUi();

    const CameraDescriptor& descriptor = cameras_[index];
    FrameCallback callback = [this](MediaFrame&& frame) {
        std::scoped_lock lock(cameraMutex_);
        latestFrame_ = std::move(frame);
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
    cpuSceneBuffer_.clear();
    cpuSceneWidth_ = 0;
    cpuSceneHeight_ = 0;
    cpuSceneReady_ = false;
    cameraActive_ = true;
    lastCameraError_.clear();
    UpdateKeystoneTrackingUi();
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
        cudaSurface_->ResetTextClarityHistory();
    }
    UpdateKeystoneTrackingUi();
    processedFrameWidth_ = 0;
    processedFrameHeight_ = 0;
    cpuSceneBuffer_.clear();
    cpuSceneWidth_ = 0;
    cpuSceneHeight_ = 0;
    cpuSceneReady_ = false;
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
    if (pipelineOrchestrator_->IsCameraReconnectPending()) {
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
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (!pipelineOrchestrator_->BeginCameraReconnect(now)) {
        return;
    }
    qWarning() << "Camera connection lost; reconnecting automatically";
    lastCameraError_ = QStringLiteral("Reconnecting to camera…");
    StopCameraCapture();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::DriveCameraReconnect() {
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (!pipelineOrchestrator_->CameraReconnectDue(now)) {
        return;
    }

    // Re-enumerate and look for the same physical device again.
    const std::wstring targetLink = mediaCapture_.LastSymbolicLink();
    cameras_ = mediaCapture_.EnumerateCameras();
    {
        auto blocker = uiState_->BlockSignals(uiState_->cameraCombo_);
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
        const int attempts =
            pipelineOrchestrator_->CameraReconnectAttempt() + 1;
        pipelineOrchestrator_->CancelCameraReconnect();
        qInfo() << "Camera reconnected after" << attempts << "attempt(s)";
        if (uiState_->cameraCombo_) {
            auto blocker = uiState_->BlockSignals(uiState_->cameraCombo_);
            uiState_->cameraCombo_->setCurrentIndex(matchIndex);
        }
        RefreshCameraModesList(static_cast<size_t>(matchIndex));
        ShowStatusMessage(QStringLiteral("Camera reconnected."), 5000);
        return;
    }

    pipelineOrchestrator_->ScheduleCameraReconnectRetry(now);
    if (pipelineOrchestrator_->CameraReconnectExpired(now)) {
        pipelineOrchestrator_->CancelCameraReconnect();
        const std::string detail = mediaCapture_.LastError();
        lastCameraError_ = !detail.empty()
            ? QString::fromStdString(detail)
            : QStringLiteral("The camera did not come back. Check the connection, then pick it "
                             "again from the camera list.");
        qWarning() << "Camera reconnect gave up:" << lastCameraError_;
        ShowStatusMessage(lastCameraError_, 15000);
        return;
    }

    // The orchestrator schedules 2s/4s/8s retry backoff.
}



bool OpenZoomApp::RunFrameTick(double elapsedSeconds) {
    // Camera loss / reconnect state machine. ConsumeDeviceLost() is polled
    // here in addition to the capture error callback so the reconnect starts
    // no matter which side notices the loss first.
    if (mediaCapture_.ConsumeDeviceLost() &&
        !pipelineOrchestrator_->IsCameraReconnectPending()) {
        BeginCameraReconnect();
    }
    if (pipelineOrchestrator_->IsCameraReconnectPending()) {
        DriveCameraReconnect();
    }

    if (!cameraActive_) {
        return false;
    }

    if (ApplyInputForces(elapsedSeconds)) {
        pipelineOrchestrator_->MarkViewportDirty();
    }
    MediaFrame frame;
    {
        std::scoped_lock lock(cameraMutex_);
        frame = std::move(latestFrame_);
    }

    if (frame.data.empty() || frame.width == 0 || frame.height == 0) {
        if (usingCudaLastFrame_ && cudaSceneReady_ &&
            (pipelineOrchestrator_->IsViewportDirty() ||
             presenter_->NeedsScenePresent())) {
            PresentLatestCudaScene(false, nullptr);
        } else if (!usingCudaLastFrame_ && cpuSceneReady_ &&
                   (pipelineOrchestrator_->IsViewportDirty() ||
                    presenter_->NeedsScenePresent())) {
            PresentFitted(
                cpuSceneBuffer_.data(),
                cpuSceneWidth_,
                cpuSceneHeight_,
                pipelineOrchestrator_->ViewportFitMode() ==
                    settings::ViewportFitModeSetting::Fill,
                zoomCenterX_,
                zoomCenterY_,
                nullptr);
        }
        return false;
    }

    CapturedFrame originalFrame;
    const bool recordingActive = recordingManager_ && recordingManager_->IsActive();
    if ((recordingActive || photoCapturePending_) && !PrepareOriginalFrame(frame, originalFrame)) {
        if (recordingActive) {
            recordingManager_->Stop(QStringLiteral(
                "Recording stopped: the original camera frame could not be converted."));
        }
    }

    // GPU fast path: NV12/YUY2 frames go straight to CUDA (conversion and
    // rotation on the GPU), skipping the per-pixel CPU work below. The CPU
    // path remains for the debug view, other subtypes, GPU-unavailable
    // passthrough, and any frame the raw path rejects.
    if (!debugViewEnabled_ && TryProcessRawFrameWithCuda(frame, &originalFrame)) {
        return true;
    }

    if (!cpuPipeline_.ConvertFrameToBgra(frame.data,
                                         frame.subtype,
                                         frame.width,
                                         frame.height,
                                         frame.stride,
                                         frame.dataSize)) {
        return true;
    }

    UINT width = frame.width;
    UINT height = frame.height;
    cpuPipeline_.RotateRawBuffer(rotationQuarterTurns_, width, height);
    processedFrameWidth_ = width;
    processedFrameHeight_ = height;

    BuildCompositeAndPresent(width, height, &originalFrame);
    if (photoCapturePending_ && !usingCudaLastFrame_) {
        CapturePendingPhoto(originalFrame);
    }
    return true;
}
void OpenZoomApp::BuildCompositeAndPresent(UINT width,
                                           UINT height,
                                           CapturedFrame* originalFrame) {
    processedFrameWidth_ = width;
    processedFrameHeight_ = height;
    usingCudaLastFrame_ = false;
    if (!debugViewEnabled_ && ProcessFrameWithCuda(width, height)) {
        usingCudaLastFrame_ = true;
        // Recording and the periodic assistive grab both use the async
        // readback ring; nothing on this path blocks on the GPU anymore.
        PresentLatestCudaScene(true, originalFrame);
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
        cpuSceneBuffer_ = raw;
        cpuSceneWidth_ = width;
        cpuSceneHeight_ = height;
        cpuSceneReady_ = true;
        PresentFitted(
            cpuSceneBuffer_.data(),
            width,
            height,
            pipelineOrchestrator_->ViewportFitMode() ==
                settings::ViewportFitModeSetting::Fill,
            zoomCenterX_,
            zoomCenterY_,
            originalFrame);
        return;
    }

    // Legacy CPU composite, kept as a diagnostic for the debug view only.
    processing::CpuPipelineConfig config{};
    config.enableBlackWhite = blackWhiteEnabled_;
    config.blackWhiteThreshold = blackWhiteThreshold_;
    // Viewport geometry owns zoom. Keeping it out of the stateful CPU stage
    // lets the cached result be re-presented smoothly without advancing
    // temporal effects or applying magnification twice.
    config.enableZoom = false;
    config.zoomAmount = 1.0f;
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
    cpuSceneBuffer_.assign(
        output.data,
        output.data + static_cast<std::size_t>(output.width) *
                          output.height * 4);
    cpuSceneWidth_ = output.width;
    cpuSceneHeight_ = output.height;
    cpuSceneReady_ = true;

    const bool cropToFill =
        pipelineOrchestrator_->ViewportFitMode() ==
        settings::ViewportFitModeSetting::Fill;
    const float centerX = cropToFill ? zoomCenterX_ : 0.5f;
    const float centerY = cropToFill ? zoomCenterY_ : 0.5f;
    PresentFitted(cpuSceneBuffer_.data(),
                  cpuSceneWidth_,
                  cpuSceneHeight_,
                  cropToFill,
                  centerX,
                  centerY,
                  originalFrame);
}

void OpenZoomApp::PresentFitted(const uint8_t* data,
                                UINT srcWidth,
                                UINT srcHeight,
                                bool cropToFill,
                                float centerXNorm,
                                float centerYNorm,
                                const CapturedFrame* originalFrame) {
    if (!data || srcWidth == 0 || srcHeight == 0) {
        return;
    }

    if (!uiState_->renderWidget_ || !uiState_->renderWidget_->isPresenterReady()) {
        return;
    }

    const UINT viewportWidth = presenter_->ViewportWidth();
    const UINT viewportHeight = presenter_->ViewportHeight();
    if (viewportWidth == 0 || viewportHeight == 0) {
        return;
    }
    const ViewTransform transform =
        ComputeViewTransform(
            srcWidth,
            srcHeight,
            viewportWidth,
            viewportHeight,
            zoomEnabled_ ? zoomAmount_ : 1.0f,
            centerXNorm,
            centerYNorm,
            cropToFill ? ViewportFitMode::kFill : ViewportFitMode::kFit);
    const PixelViewMapping mapping =
        ComputePixelViewMapping(
            transform, srcWidth, srcHeight, viewportWidth, viewportHeight);
    if (!mapping.valid) {
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
        const float focusX =
            std::clamp(centerXNorm, 0.0f, 1.0f) * static_cast<float>(srcWidth);
        const float focusY =
            std::clamp(centerYNorm, 0.0f, 1.0f) * static_cast<float>(srcHeight);
        const float localX = (focusX - mapping.startX) / mapping.stepX;
        const float localY = (focusY - mapping.startY) / mapping.stepY;
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

    const bool focusGateEnabled = focusDetectionEnabled_ || autoTextClarityEnabled_;
    const bool focusAcceptable =
        !focusGateEnabled || !cudaSurface_ ||
        cudaSurface_->IsFocusAcceptable(focusThreshold_);
    if (originalFrame) {
        assistiveManager_->MaybeRequestAnalysis(presentationBuffer_.data(),
                                                 mapping.targetWidth,
                                                 mapping.targetHeight,
                                                 debugViewEnabled_,
                                                 focusGateEnabled,
                                                 focusAcceptable);
    }
    if (originalFrame && originalFrame->IsValid()) {
        if (recordingManager_) {
            recordingManager_->AddFrame(presentationBuffer_.data(),
                                        mapping.targetWidth,
                                        mapping.targetHeight,
                                        *originalFrame);
        }
    }
    presenter_->Present(presentationBuffer_.data(), mapping.targetWidth, mapping.targetHeight);
    pipelineOrchestrator_->MarkViewportPresented();
    if (originalFrame) {
        UpdateProcessingStatusLabel();
    }
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

bool OpenZoomApp::PrepareOriginalFrame(const MediaFrame& source,
                                       CapturedFrame& destination)
{
    destination = {};
    if (!capturePipeline_.ConvertFrameToBgra(source.data,
                                             source.subtype,
                                             source.width,
                                             source.height,
                                             source.stride,
                                             source.dataSize)) {
        return false;
    }

    UINT width = source.width;
    UINT height = source.height;
    capturePipeline_.RotateRawBuffer(rotationQuarterTurns_, width, height);
    const std::vector<uint8_t>& pixels = capturePipeline_.StageRaw();
    if (pixels.empty()) {
        return false;
    }

    destination.pixels = pixels;
    destination.width = width;
    destination.height = height;
    return true;
}

bool OpenZoomApp::SaveSnapshot(const uint8_t* data,
                               UINT width,
                               UINT height,
                               const QString& fullPath)
{
    if (!data || width == 0 || height == 0 || fullPath.isEmpty()) {
        return false;
    }
    QImage image(data,
                 static_cast<int>(width),
                 static_cast<int>(height),
                 static_cast<int>(width) * 4,
                 QImage::Format_ARGB32);
    if (!image.save(fullPath, "JPG", 90)) {
        qWarning() << "Failed to save snapshot to" << fullPath;
        return false;
    }
    qInfo() << "Saved snapshot to" << fullPath;
    return true;
}

void OpenZoomApp::CapturePendingPhoto(const CapturedFrame& originalFrame)
{
    if (!photoCapturePending_) {
        return;
    }
    photoCapturePending_ = false;
    if (!originalFrame.IsValid()) {
        ShowStatusMessage(QStringLiteral(
            "Photo not saved: the original camera frame could not be converted."));
        return;
    }

    if (usingCudaLastFrame_) {
        // CUDA photos are paired with the asynchronous viewport readback in
        // PresentLatestCudaScene so the saved image matches the visible crop.
        photoCapturePending_ = true;
        return;
    }
    if (presentationBuffer_.empty() ||
        presentationWidth_ == 0 || presentationHeight_ == 0) {
        ShowStatusMessage(QStringLiteral("Photo not saved: no processed frame was available."));
        return;
    }
    SaveCapturedPhotoPair(presentationBuffer_.data(),
                          presentationWidth_,
                          presentationHeight_,
                          originalFrame);
}

void OpenZoomApp::SaveCapturedPhotoPair(const uint8_t* processedData,
                                        UINT processedWidth,
                                        UINT processedHeight,
                                        const CapturedFrame& originalFrame)
{
    if (!processedData || processedWidth == 0 || processedHeight == 0 ||
        !originalFrame.IsValid()) {
        ShowStatusMessage(QStringLiteral(
            "Photo not saved: the processed or original frame was unavailable."));
        return;
    }
    const QString dirPath = EnsureOutputSubdir(QStringLiteral("img"));
    const QString timestamp =
        QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"));
    const QString processedPath = QDir(dirPath).filePath(
        QStringLiteral("IMG_%1_processed.jpg").arg(timestamp));
    const QString originalPath = QDir(dirPath).filePath(
        QStringLiteral("IMG_%1_original.jpg").arg(timestamp));

    const bool processedSaved = SaveSnapshot(processedData,
                                             processedWidth,
                                             processedHeight,
                                             processedPath);
    const bool originalSaved = SaveSnapshot(originalFrame.pixels.data(),
                                            originalFrame.width,
                                            originalFrame.height,
                                            originalPath);
    if (processedSaved && originalSaved) {
        assistiveManager_->Runtime().NoteCapturedPhoto(processedPath);
        ShowStatusMessage(QStringLiteral("Saved original and processed photos."), 5000);
    } else {
        ShowStatusMessage(QStringLiteral("One of the paired photos could not be saved."));
    }
}

} // namespace openzoom

#endif // _WIN32
