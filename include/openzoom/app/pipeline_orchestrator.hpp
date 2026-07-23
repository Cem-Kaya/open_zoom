#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include "openzoom/app/settings_store.hpp"

#include <QElapsedTimer>
#include <QTimer>

#include <array>
#include <algorithm>
#include <cstdint>
#include <functional>

namespace openzoom {

// One monotonic D3D12/CUDA fence timeline. Values only become wait targets
// after their corresponding signal has been submitted.
struct FenceSequencer {
    struct CudaTicket {
        std::uint64_t waitValue{0};
        std::uint64_t signalValue{0};
    };

    void Reset(std::uint64_t baseValue)
    {
        nextValue_ = baseValue + 1;
        lastCudaSignal_ = 0;
        lastGraphicsSignal_ = baseValue;
        lastReadbackSignal_ = 0;
        pendingCudaSignal_ = 0;
    }

    CudaTicket BeginCudaFrame(std::uint64_t presenterSignaledValue)
    {
        if (nextValue_ <= presenterSignaledValue) {
            nextValue_ = presenterSignaledValue + 1;
        }
        pendingCudaSignal_ = nextValue_;
        return {std::max(lastGraphicsSignal_, lastReadbackSignal_),
                pendingCudaSignal_};
    }

    void CudaSignaled()
    {
        lastCudaSignal_ = pendingCudaSignal_;
        nextValue_ = pendingCudaSignal_ + 1;
        pendingCudaSignal_ = 0;
    }

    void CudaFailed() { pendingCudaSignal_ = 0; }
    std::uint64_t BeginGraphicsFrame(std::uint64_t presenterSignaledValue)
    {
        nextValue_ = std::max(nextValue_, presenterSignaledValue + 1);
        return nextValue_;
    }

    void GraphicsSignaled(std::uint64_t presenterSignaledValue)
    {
        lastGraphicsSignal_ =
            std::max(lastGraphicsSignal_, presenterSignaledValue);
        nextValue_ = std::max(nextValue_, presenterSignaledValue + 1);
    }

    void ReadbackObserved(std::uint64_t fenceValue)
    {
        lastReadbackSignal_ = std::max(lastReadbackSignal_, fenceValue);
    }

    std::uint64_t LastCudaSignal() const { return lastCudaSignal_; }

private:
    std::uint64_t nextValue_{1};
    std::uint64_t lastCudaSignal_{0};
    std::uint64_t lastGraphicsSignal_{0};
    std::uint64_t lastReadbackSignal_{0};
    std::uint64_t pendingCudaSignal_{0};
};

// Owns the viewport presentation clock and its instrumentation. Camera
// processing remains a distinct callback: it advances only when the callback
// consumes a fresh frame, while viewport-only motion may present the cached
// scene at the active display rate.
class PipelineOrchestrator final {
public:
    struct Callbacks {
        std::function<bool(double)> tick;
        std::function<bool()> hasContinuousMotion;
        std::function<bool()> isMousePanActive;
        std::function<bool()> needsScenePresent;
        std::function<bool()> cameraActive;
        std::function<double()> cameraFrameRate;
        std::function<bool()> usingCuda;
        std::function<int()> queryDisplayRefreshRate;
        std::function<void(int requestedRate, int effectiveRate)>
            viewportRateClamped;
    };

    PipelineOrchestrator(QObject& context, Callbacks callbacks);
    ~PipelineOrchestrator();

    PipelineOrchestrator(const PipelineOrchestrator&) = delete;
    PipelineOrchestrator& operator=(const PipelineOrchestrator&) = delete;

    void Start();
    void Stop();
    void UpdateTimerPolicy();

    void SetViewportRateMode(settings::ViewportRateMode mode);
    settings::ViewportRateMode ViewportRateMode() const;
    void SetViewportFitMode(settings::ViewportFitModeSetting mode);
    settings::ViewportFitModeSetting ViewportFitMode() const;

    void MarkViewportDirty();
    bool IsViewportDirty() const;
    void NotifyViewportMotion();
    void MarkViewportPresented();

    int EffectiveViewportRate() const;
    int DisplayRefreshRate() const;
    float MeasuredViewportRate() const;
    float FrameTickAverageMs() const;

    bool BeginCameraReconnect(qint64 nowMs);
    void CancelCameraReconnect();
    bool IsCameraReconnectPending() const;
    bool CameraReconnectDue(qint64 nowMs) const;
    bool CameraReconnectExpired(qint64 nowMs) const;
    int CameraReconnectAttempt() const;
    void ScheduleCameraReconnectRetry(qint64 nowMs);

    FenceSequencer& Fence();
    const FenceSequencer& Fence() const;
    void ResetFence(std::uint64_t baseValue);
    bool FenceInteropEnabled() const;
    void SetFenceInteropEnabled(bool enabled);
    int RecordCudaFailure();
    void ResetCudaFailures();

private:
    void OnTick();
    void RecordFrameTickSample(qint64 elapsedNanos);
    int RequestedViewportRate() const;

    QTimer timer_;
    Callbacks callbacks_;
    settings::ViewportRateMode viewportRateMode_{
        settings::ViewportRateMode::AutoUpTo120};
    settings::ViewportFitModeSetting viewportFitMode_{
        settings::ViewportFitModeSetting::Fill};
    int displayRefreshHz_{60};
    int effectiveViewportRate_{60};
    int announcedClampedRate_{0};
    float measuredViewportRate_{0.0f};
    int viewportPresentCount_{0};
    bool viewportDirty_{true};
    QElapsedTimer viewportTickTimer_;
    QElapsedTimer viewportMotionTailTimer_;
    QElapsedTimer viewportRateMeasurementTimer_;

    std::array<float, 60> frameTickSamplesMs_{};
    std::size_t frameTickSampleIndex_{0};
    std::size_t frameTickSampleCount_{0};
    float frameTickSampleSumMs_{0.0f};
    float frameTickAverageMs_{-1.0f};
    QElapsedTimer frameTickOverBudgetTimer_;
    bool frameTickOverBudgetWarned_{false};

    bool cameraReconnectPending_{false};
    int cameraReconnectAttempt_{0};
    qint64 cameraReconnectStartedMs_{0};
    qint64 cameraReconnectNextAttemptMs_{0};

    FenceSequencer fenceSequencer_;
    bool fenceInteropEnabled_{false};
    int consecutiveCudaFailures_{0};
};

} // namespace openzoom

#endif // _WIN32
