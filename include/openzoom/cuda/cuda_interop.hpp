#pragma once

#ifdef _WIN32

#include <wrl/client.h>
#include <d3d12.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifndef OPENZOOM_HAS_CUDA_EXT_MEMORY
#  if defined(__has_include)
#    if __has_include(<cuda_runtime.h>)
#      define OPENZOOM_HAS_CUDA_EXT_MEMORY 1
#    else
#      define OPENZOOM_HAS_CUDA_EXT_MEMORY 0
#    endif
#  else
#    define OPENZOOM_HAS_CUDA_EXT_MEMORY 1
#  endif
#endif

#if OPENZOOM_HAS_CUDA_EXT_MEMORY
#  include <cuda.h>
#  include <cuda_runtime_api.h>
#  if defined(__has_include)
#    if __has_include(<cuda_runtime.h>)
#      include <cuda_runtime.h>
#    endif
#  endif
#endif

struct ID3D12Device;
struct ID3D12Resource;
struct ID3D12Fence;

namespace openzoom {

struct FenceSyncParams {
    bool enable{false};
    uint64_t waitValue{0};
    uint64_t signalValue{0};
};

enum class SpatialUpscaler : int {
    kFsrEasuRcas = 0,
    kNis = 1,
};

enum class CudaBufferFormat : int {
    kRgba8 = 0,
    kRgba16F = 1,
};

struct ProcessingSettings {
    bool enableBlackWhite{false};
    float blackWhiteThreshold{0.5f};
    bool enableZoom{false};
    float zoomAmount{1.0f};
    float zoomCenterX{0.5f};
    float zoomCenterY{0.5f};
    bool enableBlur{false};
    int blurRadius{3};
    float blurSigma{1.0f};
    bool drawFocusMarker{false};
    bool enableSpatialSharpen{false};
    SpatialUpscaler spatialUpscaler{SpatialUpscaler::kNis};
    float spatialSharpness{0.2f};
    CudaBufferFormat stagingFormat{CudaBufferFormat::kRgba8};
    bool enableTemporalSmoothing{false};
    float temporalSmoothingAlpha{0.25f};
    bool enableStabilization{false};
    float stabilizationStrength{0.85f};  // 0..1, higher = stronger smoothing
    int displayColorMode{0};             // 0=Normal,1=Inverted,2=WhiteOnBlack,3=YellowOnBlack,4=BlackOnYellow
    float contrast{1.0f};                // 0.25..4.0, 1 = neutral
    float brightness{0.0f};              // -1..1, 0 = neutral
    bool enableKeystone{false};          // auto-detect projected slide quad, warp fronto-parallel
    bool enableAutoContrast{false};      // percentile-based level stretch before contrast/brightness
    float autoContrastStrength{0.7f};    // 0..1, blend toward full stretch
};

// Input frame description.
//
// width/height are ALWAYS the dimensions of the host pixel data as laid out in
// memory (i.e. pre-rotation). When inputFormat != 0 and rotationQuarterTurns is
// odd (90/270), the GPU rotates after conversion, so the interop surface (and
// every processing stage) runs at the POST-rotation dimensions
// (height x width). The caller must create the surface at the post-rotation
// size; ProcessFrame validates that.
struct ProcessingInput {
    const void* hostPixels{nullptr};
    unsigned int hostStrideBytes{0};
    unsigned int pixelSizeBytes{0};
    unsigned int width{0};
    unsigned int height{0};
    int inputFormat{0};                   // 0=BGRA8 (existing), 1=NV12, 2=YUY2
    const void* hostPlane2{nullptr};      // NV12 UV plane (nullptr otherwise)
    unsigned int hostPlane2StrideBytes{0};
    int rotationQuarterTurns{0};          // 0..3 clockwise, applied on GPU after conversion.
                                          // Ignored for inputFormat 0 (CPU already rotated BGRA).
};

#if OPENZOOM_HAS_CUDA_EXT_MEMORY
struct StabilizationState;

class CudaInteropSurface {
public:
    explicit CudaInteropSurface(ID3D12Resource* texture, ID3D12Fence* sharedFence = nullptr);
    ~CudaInteropSurface();

    bool IsValid() const { return valid_; }
    bool HasExternalSemaphore() const { return externalSemaphore_ != nullptr; }

    void RunGradientDemoKernel(unsigned int width, unsigned int height, float timeSeconds);

    bool ProcessFrame(const ProcessingInput& input,
                      const ProcessingSettings& settings,
                      const FenceSyncParams& fenceSync);
    const std::string& LastError() const { return lastError_; }
    void ResetTemporalHistory();
    void ResetStabilization();
    void ResetKeystone();

    CudaInteropSurface(const CudaInteropSurface&) = delete;
    CudaInteropSurface& operator=(const CudaInteropSurface&) = delete;

private:
    void Initialize(ID3D12Resource* texture, ID3D12Fence* sharedFence);

    bool SelectCudaDeviceMatching(LUID adapterLuid);
    bool CreateSurfaceFromResource(ID3D12Device* device, ID3D12Resource* texture);
    void ImportFenceSemaphore(ID3D12Device* device, ID3D12Fence* fence);
    bool EnsureDeviceBuffers(unsigned int width, unsigned int height, CudaBufferFormat format);
    bool EnsureTemporalHistory(unsigned int width, unsigned int height);
    bool EnsureStabilizationBuffers(unsigned int width, unsigned int height);
    bool EnsureRawInputBuffers(unsigned int width, unsigned int height, int format);
    bool EnsurePreRotateBuffer(unsigned int width, unsigned int height);
    bool EnsureKeystoneResources(unsigned int width, unsigned int height);
    bool EnsureAutoContrastBuffers();
    void ReleaseDeviceBuffers();
    void ReleaseTemporalHistory();
    void ReleaseStabilization();
    void ReleaseRawInput();
    void ReleaseKeystone();
    void ReleaseAutoContrast();
    bool EnsureGaussianKernel(int radius, float sigma);
    void RunKeystoneStage(uchar4*& current, uchar4*& alternate,
                          size_t& currentPitch, size_t& alternatePitch);
    void ConsumeKeystoneDetection();
    void ResetKeystoneCornersToIdentity();
    void SynchronizeStream() noexcept;

    cudaExternalMemory_t externalMemory_{};
    cudaMipmappedArray_t mipArray_{};
    cudaArray_t level0Array_{};
    cudaSurfaceObject_t surfaceObject_{0};
    cudaStream_t stream_{};
    cudaExternalSemaphore_t externalSemaphore_{};

    UINT width_{};
    UINT height_{};
    DXGI_FORMAT format_{DXGI_FORMAT_UNKNOWN};
    int cudaDeviceId_{-1};
    bool valid_{false};

    uchar4* deviceBufferA_{};
    uchar4* deviceBufferB_{};
    uchar4* deviceScratch_{};
    size_t devicePitchA_{};
    size_t devicePitchB_{};
    size_t devicePitchScratch_{};
    unsigned int deviceWidth_{};
    unsigned int deviceHeight_{};
    CudaBufferFormat bufferFormat_{CudaBufferFormat::kRgba8};
    float4* deviceTemporalHistory_{};
    size_t devicePitchHistory_{};
    unsigned int historyWidth_{};
    unsigned int historyHeight_{};
    bool temporalHistoryValid_{};
    float* deviceStabLuma_{};
    float* deviceStabColProjCurr_{};
    float* deviceStabRowProjCurr_{};
    float* deviceStabColProjPrev_{};
    float* deviceStabRowProjPrev_{};
    StabilizationState* deviceStabState_{};
    unsigned int stabSmallWidth_{};
    unsigned int stabSmallHeight_{};
    unsigned int stabFactorX_{};
    unsigned int stabFactorY_{};
    unsigned int stabFullWidth_{};
    unsigned int stabFullHeight_{};
    bool stabPrevValid_{};

    // Raw camera input (NV12/YUY2) + GPU rotation staging.
    unsigned char* deviceRawPlane1_{};
    unsigned char* deviceRawPlane2_{};
    size_t rawPlane1Pitch_{};
    size_t rawPlane2Pitch_{};
    unsigned int rawWidth_{};
    unsigned int rawHeight_{};
    int rawFormat_{-1};
    uchar4* devicePreRotate_{};
    size_t devicePitchPreRotate_{};
    unsigned int preRotateWidth_{};
    unsigned int preRotateHeight_{};

    // Keystone: async small-luma snapshot (device -> pinned host) + CPU quad
    // detection. Corners are the temporally smoothed source-quad corners in
    // full-resolution pixels, order TL, TR, BR, BL.
    float* deviceKeystoneLuma_{};
    float* hostKeystoneLuma_{};        // pinned (cudaMallocHost)
    cudaEvent_t keystoneCopyEvent_{};
    bool keystoneCopyPending_{};
    unsigned int keystoneSmallWidth_{};
    unsigned int keystoneSmallHeight_{};
    unsigned int keystoneFactorX_{};
    unsigned int keystoneFactorY_{};
    unsigned int keystoneFullWidth_{};
    unsigned int keystoneFullHeight_{};
    float2 keystoneCorners_[4]{};
    unsigned int keystoneFrameCounter_{};
    unsigned int keystoneFramesSinceDetection_{};

    // Auto contrast: 256-bin histogram + smoothed lo/hi levels, all device-resident.
    unsigned int* deviceHistogram_{};
    float2* deviceAutoLevels_{};
    bool autoLevelsValid_{};

    int cachedKernelRadius_{-1};
    float cachedKernelSigma_{0.0f};
    bool kernelUploaded_{};
    std::string lastError_;
};
#else
class CudaInteropSurface {
public:
    explicit CudaInteropSurface(ID3D12Resource* /*texture*/, ID3D12Fence* /*sharedFence*/ = nullptr) {}
    ~CudaInteropSurface() = default;

    bool IsValid() const { return false; }
    bool HasExternalSemaphore() const { return false; }

    void RunGradientDemoKernel(unsigned int /*width*/, unsigned int /*height*/, float /*timeSeconds*/) {}

    bool ProcessFrame(const ProcessingInput& /*input*/,
                      const ProcessingSettings& /*settings*/,
                      const FenceSyncParams& /*fenceSync*/) { return false; }

    const std::string& LastError() const { static std::string dummy; return dummy; }
    void ResetTemporalHistory() {}
    void ResetStabilization() {}
    void ResetKeystone() {}

    CudaInteropSurface(const CudaInteropSurface&) = delete;
    CudaInteropSurface& operator=(const CudaInteropSurface&) = delete;
};
#endif

} // namespace openzoom

#endif // _WIN32
