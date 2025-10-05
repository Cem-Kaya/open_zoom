#pragma once

#ifdef _WIN32

#include <wrl/client.h>
#include <d3d12.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

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

namespace openzoom {

struct ProcessingSettings {
    bool enableBlackWhite{false};
    float blackWhiteThreshold{0.5f};
    bool enableZoom{false};
    float zoomAmount{1.0f};
};

#if OPENZOOM_HAS_CUDA_EXT_MEMORY
class CudaInteropSurface {
public:
    explicit CudaInteropSurface(ID3D12Resource* texture);
    ~CudaInteropSurface();

    bool IsValid() const { return valid_; }

    void RunGradientDemoKernel(unsigned int width, unsigned int height, float timeSeconds);

    void ProcessFrame(unsigned int /*width*/, unsigned int /*height*/, const ProcessingSettings& /*settings*/) {}

    CudaInteropSurface(const CudaInteropSurface&) = delete;
    CudaInteropSurface& operator=(const CudaInteropSurface&) = delete;

private:
    void Initialize(ID3D12Resource* texture);

    bool SelectCudaDeviceMatching(LUID adapterLuid);
    bool CreateSurfaceFromResource(ID3D12Device* device, ID3D12Resource* texture);

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
};
#else
class CudaInteropSurface {
public:
    explicit CudaInteropSurface(ID3D12Resource* /*texture*/) {}
    ~CudaInteropSurface() = default;

    bool IsValid() const { return false; }

    void RunGradientDemoKernel(unsigned int /*width*/, unsigned int /*height*/, float /*timeSeconds*/) {}

    void ProcessFrame(unsigned int /*width*/, unsigned int /*height*/, const ProcessingSettings& /*settings*/) {}

    CudaInteropSurface(const CudaInteropSurface&) = delete;
    CudaInteropSurface& operator=(const CudaInteropSurface&) = delete;
};
#endif

} // namespace openzoom

#endif // _WIN32
