#pragma once

#ifdef _WIN32

#include <wrl/client.h>
#include <d3d12.h>
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
};

struct ProcessingInput {
    const void* hostPixels{nullptr};
    unsigned int hostStrideBytes{0};
    unsigned int pixelSizeBytes{0};
    unsigned int width{0};
    unsigned int height{0};
};

#if OPENZOOM_HAS_CUDA_EXT_MEMORY
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

    CudaInteropSurface(const CudaInteropSurface&) = delete;
    CudaInteropSurface& operator=(const CudaInteropSurface&) = delete;

private:
    void Initialize(ID3D12Resource* texture, ID3D12Fence* sharedFence);

    bool SelectCudaDeviceMatching(LUID adapterLuid);
    bool CreateSurfaceFromResource(ID3D12Device* device, ID3D12Resource* texture);
    void ImportFenceSemaphore(ID3D12Device* device, ID3D12Fence* fence);
    bool EnsureDeviceBuffers(unsigned int width, unsigned int height, CudaBufferFormat format);
    bool EnsureTemporalHistory(unsigned int width, unsigned int height);
    void ReleaseDeviceBuffers();
    void ReleaseTemporalHistory();
    bool EnsureGaussianKernel(int radius, float sigma);

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

    CudaInteropSurface(const CudaInteropSurface&) = delete;
    CudaInteropSurface& operator=(const CudaInteropSurface&) = delete;
};
#endif

} // namespace openzoom

#endif // _WIN32
