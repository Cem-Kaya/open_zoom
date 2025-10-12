#pragma once

#ifdef _WIN32

#include <wrl/client.h>
#include <d3d12.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <string>
#include <vector>

#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>) && __has_include(<cuda_surface_types.h>) && __has_include(<surface_functions.h>)
#    include <cuda_runtime.h>
#    include <cuda_surface_types.h>
#    include <surface_functions.h>
#    define OPENZOOM_HAS_CUDA_EXT_MEMORY 1
#  else
#    define OPENZOOM_HAS_CUDA_EXT_MEMORY 0
#  endif
#else
#  define OPENZOOM_HAS_CUDA_EXT_MEMORY 0
#endif

struct ID3D12Device;
struct ID3D12Resource;

namespace openzoom {

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
};

struct ProcessingInput {
    const uint8_t* hostBgra{nullptr};
    unsigned int hostStride{0};
    unsigned int width{0};
    unsigned int height{0};
};

#if OPENZOOM_HAS_CUDA_EXT_MEMORY
class CudaInteropSurface {
public:
    explicit CudaInteropSurface(ID3D12Resource* texture);
    ~CudaInteropSurface();

    bool IsValid() const { return valid_; }

    void RunGradientDemoKernel(unsigned int width, unsigned int height, float timeSeconds);

    bool ProcessFrame(const ProcessingInput& input, const ProcessingSettings& settings);
    const std::string& LastError() const { return lastError_; }

    CudaInteropSurface(const CudaInteropSurface&) = delete;
    CudaInteropSurface& operator=(const CudaInteropSurface&) = delete;

private:
    void Initialize(ID3D12Resource* texture);

    bool SelectCudaDeviceMatching(LUID adapterLuid);
    bool CreateSurfaceFromResource(ID3D12Device* device, ID3D12Resource* texture);
    bool EnsureDeviceBuffers(unsigned int width, unsigned int height);
    void ReleaseDeviceBuffers();
    bool EnsureGaussianKernel(int radius, float sigma);

    cudaExternalMemory_t externalMemory_{};
    cudaMipmappedArray_t mipArray_{};
    cudaArray_t level0Array_{};
    cudaSurfaceObject_t surfaceObject_{0};
    cudaStream_t stream_{};

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
    int cachedKernelRadius_{-1};
    float cachedKernelSigma_{0.0f};
    bool kernelUploaded_{};
    std::string lastError_;
};
#else
class CudaInteropSurface {
public:
    explicit CudaInteropSurface(ID3D12Resource* /*texture*/) {}
    ~CudaInteropSurface() = default;

    bool IsValid() const { return false; }

    void RunGradientDemoKernel(unsigned int /*width*/, unsigned int /*height*/, float /*timeSeconds*/) {}

    bool ProcessFrame(const ProcessingInput& /*input*/, const ProcessingSettings& /*settings*/) { return false; }

    const std::string& LastError() const { static std::string dummy; return dummy; }

    CudaInteropSurface(const CudaInteropSurface&) = delete;
    CudaInteropSurface& operator=(const CudaInteropSurface&) = delete;
};
#endif

} // namespace openzoom

#endif // _WIN32
