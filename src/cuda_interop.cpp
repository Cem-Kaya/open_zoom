#ifdef _WIN32

#include "openzoom/cuda_interop.hpp"
#include "openzoom/cuda_kernels.hpp"

#if OPENZOOM_HAS_CUDA_EXT_MEMORY

#include <windows.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <cstring>

namespace openzoom {

namespace {

void ThrowIfFailed(HRESULT hr, const char* message) {
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(message) + " (hr=0x" + std::to_string(static_cast<unsigned long>(hr)) + ")");
    }
}

void ThrowIfCudaFailed(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

class WindowsSecurityAttributes {
public:
    WindowsSecurityAttributes() {
        InitializeSecurityDescriptor(&securityDescriptor_, SECURITY_DESCRIPTOR_REVISION);
        SetSecurityDescriptorDacl(&securityDescriptor_, TRUE, nullptr, FALSE);
        attributes_.nLength = sizeof(attributes_);
        attributes_.lpSecurityDescriptor = &securityDescriptor_;
        attributes_.bInheritHandle = FALSE;
    }

    SECURITY_ATTRIBUTES* get() { return &attributes_; }

private:
    SECURITY_ATTRIBUTES attributes_{};
    SECURITY_DESCRIPTOR securityDescriptor_{};
};

cudaChannelFormatDesc MakeChannelDescForFormat(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return cudaCreateChannelDesc<uchar4>();
    default:
        throw std::runtime_error("Unsupported DXGI format for CUDA interop");
    }
}

} // namespace

CudaInteropSurface::CudaInteropSurface(ID3D12Resource* texture) {
    try {
        Initialize(texture);
        valid_ = true;
    } catch (const std::exception&) {
        valid_ = false;
        if (surfaceObject_ != 0) {
            cudaDestroySurfaceObject(surfaceObject_);
            surfaceObject_ = 0;
        }
        if (externalMemory_ != nullptr) {
            cudaDestroyExternalMemory(externalMemory_);
            externalMemory_ = nullptr;
        }
        mipArray_ = nullptr;
        level0Array_ = nullptr;
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }
}

CudaInteropSurface::~CudaInteropSurface() {
    if (surfaceObject_ != 0) {
        cudaDestroySurfaceObject(surfaceObject_);
        surfaceObject_ = 0;
    }

    if (externalMemory_ != nullptr) {
        cudaDestroyExternalMemory(externalMemory_);
        externalMemory_ = nullptr;
    }

    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

bool CudaInteropSurface::SelectCudaDeviceMatching(LUID adapterLuid) {
    int deviceCount = 0;
    ThrowIfCudaFailed(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount failed");

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp properties{};
        ThrowIfCudaFailed(cudaGetDeviceProperties(&properties, deviceId), "cudaGetDeviceProperties failed");

        if (std::memcmp(&adapterLuid, properties.luid, sizeof(LUID)) == 0) {
            ThrowIfCudaFailed(cudaSetDevice(deviceId), "cudaSetDevice failed");
            cudaDeviceId_ = deviceId;
            return true;
        }
    }

    return false;
}

bool CudaInteropSurface::CreateSurfaceFromResource(ID3D12Device* device, ID3D12Resource* texture) {
    WindowsSecurityAttributes securityAttributes;

    HANDLE sharedHandle = nullptr;
    ThrowIfFailed(device->CreateSharedHandle(texture,
                                             securityAttributes.get(),
                                             GENERIC_ALL,
                                             nullptr,
                                             &sharedHandle),
                  "Failed to create shared handle for D3D12 resource");

    D3D12_RESOURCE_DESC desc = texture->GetDesc();
    width_ = static_cast<UINT>(desc.Width);
    height_ = static_cast<UINT>(desc.Height);
    format_ = desc.Format;

    D3D12_RESOURCE_ALLOCATION_INFO allocationInfo = device->GetResourceAllocationInfo(0, 1, &desc);

    cudaExternalMemoryHandleDesc memoryDesc{};
    memoryDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    memoryDesc.handle.win32.handle = sharedHandle;
    memoryDesc.size = allocationInfo.SizeInBytes;
    memoryDesc.flags = cudaExternalMemoryDedicated;

    auto cleanupHandle = [&sharedHandle]() {
        if (sharedHandle) {
            CloseHandle(sharedHandle);
            sharedHandle = nullptr;
        }
    };

    try {
        ThrowIfCudaFailed(cudaImportExternalMemory(&externalMemory_, &memoryDesc),
                          "cudaImportExternalMemory failed");
        cleanupHandle();

        cudaExternalMemoryMipmappedArrayDesc arrayDesc{};
        arrayDesc.offset = 0;
        arrayDesc.numLevels = 1;
        arrayDesc.extent = make_cudaExtent(width_, height_, 1);
        arrayDesc.formatDesc = MakeChannelDescForFormat(format_);
        arrayDesc.flags = cudaArraySurfaceLoadStore | cudaArrayColorAttachment;

        ThrowIfCudaFailed(cudaExternalMemoryGetMappedMipmappedArray(&mipArray_, externalMemory_, &arrayDesc),
                          "cudaExternalMemoryGetMappedMipmappedArray failed");
        ThrowIfCudaFailed(cudaGetMipmappedArrayLevel(&level0Array_, mipArray_, 0),
                          "cudaGetMipmappedArrayLevel failed");

        cudaResourceDesc resourceDesc{};
        resourceDesc.resType = cudaResourceTypeArray;
        resourceDesc.res.array.array = level0Array_;
        ThrowIfCudaFailed(cudaCreateSurfaceObject(&surfaceObject_, &resourceDesc),
                          "cudaCreateSurfaceObject failed");

        ThrowIfCudaFailed(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
    } catch (...) {
        cleanupHandle();
        throw;
    }

    return true;
}

void CudaInteropSurface::Initialize(ID3D12Resource* texture) {
    if (!texture) {
        throw std::invalid_argument("Cannot initialize CUDA interop with null resource");
    }

    Microsoft::WRL::ComPtr<ID3D12Device> device;
    ThrowIfFailed(texture->GetDevice(IID_PPV_ARGS(&device)), "Failed to query ID3D12Device from resource");

    LUID adapterLuid = device->GetAdapterLuid();
    if (!SelectCudaDeviceMatching(adapterLuid)) {
        throw std::runtime_error("No CUDA device matches the D3D12 adapter LUID");
    }

    CreateSurfaceFromResource(device.Get(), texture);
}

void CudaInteropSurface::RunGradientDemoKernel(unsigned int width, unsigned int height, float timeSeconds) {
    if (!valid_) {
        return;
    }

    if (cudaDeviceId_ >= 0) {
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceId_), "cudaSetDevice failed");
    }

    const unsigned int targetWidth = (width == 0) ? width_ : width;
    const unsigned int targetHeight = (height == 0) ? height_ : height;

    LaunchGradientKernel(surfaceObject_, static_cast<int>(targetWidth), static_cast<int>(targetHeight), timeSeconds);
    ThrowIfCudaFailed(cudaGetLastError(), "Gradient kernel launch failed");
    ThrowIfCudaFailed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed");
}

} // namespace openzoom

#endif // OPENZOOM_HAS_CUDA_EXT_MEMORY

#endif // _WIN32
