#ifdef _WIN32

#include "openzoom/cuda_interop.hpp"
#include "openzoom/cuda_kernels.hpp"

#include <d3d12.h>

#if OPENZOOM_HAS_CUDA_EXT_MEMORY

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <QDebug>

namespace openzoom {

namespace {

void ThrowIfFailed(HRESULT hr, const char* message) {
    if (FAILED(hr)) {
        qWarning() << message << "hr=0x" << Qt::hex << static_cast<unsigned long>(hr);
        throw std::runtime_error(std::string(message) + " (hr=0x" + std::to_string(static_cast<unsigned long>(hr)) + ")");
    }
}

void ThrowIfCudaFailed(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        const char* err = cudaGetErrorString(status);
        qWarning() << message << "cudaError=" << status << err;
        throw std::runtime_error(std::string(message) + ": " + err);
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

constexpr int kMaxCudaBlurRadius = 15;

} // namespace

CudaInteropSurface::CudaInteropSurface(ID3D12Resource* texture) {
    try {
        Initialize(texture);
        valid_ = true;
        lastError_.clear();
    } catch (const std::exception& e) {
        qWarning() << "CudaInteropSurface init failed:" << e.what();
        valid_ = false;
        lastError_ = e.what();
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

    ReleaseDeviceBuffers();
}

bool CudaInteropSurface::SelectCudaDeviceMatching(LUID adapterLuid) {
    int deviceCount = 0;
    ThrowIfCudaFailed(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount failed");

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp properties{};
        ThrowIfCudaFailed(cudaGetDeviceProperties(&properties, deviceId), "cudaGetDeviceProperties failed");

        if (properties.luidSupported &&
            std::memcmp(&adapterLuid, properties.luid, sizeof(LUID)) == 0) {
            qInfo() << "CUDA device" << deviceId << properties.name << "matches DXGI adapter";
            ThrowIfCudaFailed(cudaSetDevice(deviceId), "cudaSetDevice failed");
            cudaDeviceId_ = deviceId;
            return true;
        }
    }

    lastError_ = "No CUDA device matched DXGI adapter LUID";
    qWarning() << "No CUDA device matched DXGI adapter LUID; CUDA interop disabled";
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
    qInfo() << "Created shared D3D12 resource handle for CUDA interop";

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
        arrayDesc.extent = make_cudaExtent(width_, height_, 0);
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
    } catch (const std::exception& e) {
        lastError_ = e.what();
        qWarning() << "CreateSurfaceFromResource exception:" << e.what();
        cleanupHandle();
        throw;
    }

    cachedKernelRadius_ = -1;
    cachedKernelSigma_ = 0.0f;
    kernelUploaded_ = false;
    qInfo() << "CUDA external memory imported successfully (" << width_ << "x" << height_ << ")";
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

bool CudaInteropSurface::EnsureDeviceBuffers(unsigned int width, unsigned int height) {
    if (deviceBufferA_ && deviceBufferB_ && deviceWidth_ == width && deviceHeight_ == height) {
        return true;
    }

    ReleaseDeviceBuffers();

    size_t pitch = 0;
    if (cudaDeviceId_ >= 0) {
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceId_), "cudaSetDevice failed");
    }

    ThrowIfCudaFailed(cudaMallocPitch(reinterpret_cast<void**>(&deviceBufferA_), &devicePitchA_,
                                      static_cast<size_t>(width) * sizeof(uchar4), height),
                      "cudaMallocPitch buffer A failed");
    ThrowIfCudaFailed(cudaMallocPitch(reinterpret_cast<void**>(&deviceBufferB_), &devicePitchB_,
                                      static_cast<size_t>(width) * sizeof(uchar4), height),
                      "cudaMallocPitch buffer B failed");
    ThrowIfCudaFailed(cudaMallocPitch(reinterpret_cast<void**>(&deviceScratch_), &devicePitchScratch_,
                                      static_cast<size_t>(width) * sizeof(uchar4), height),
                      "cudaMallocPitch scratch buffer failed");

    deviceWidth_ = width;
    deviceHeight_ = height;
    return true;
}

void CudaInteropSurface::ReleaseDeviceBuffers() {
    if (deviceBufferA_) {
        cudaFree(deviceBufferA_);
        deviceBufferA_ = nullptr;
    }
    if (deviceBufferB_) {
        cudaFree(deviceBufferB_);
        deviceBufferB_ = nullptr;
    }
    if (deviceScratch_) {
        cudaFree(deviceScratch_);
        deviceScratch_ = nullptr;
    }
    devicePitchA_ = 0;
    devicePitchB_ = 0;
    devicePitchScratch_ = 0;
    deviceWidth_ = 0;
    deviceHeight_ = 0;
    kernelUploaded_ = false;
}

bool CudaInteropSurface::EnsureGaussianKernel(int radius, float sigma) {
    const int clampedRadius = std::min(std::max(radius, 1), kMaxCudaBlurRadius);
    const float clampedSigma = std::max(sigma, 0.001f);

    if (kernelUploaded_ && cachedKernelRadius_ == clampedRadius &&
        std::abs(cachedKernelSigma_ - clampedSigma) < 1e-4f) {
        return true;
    }

    if (cudaDeviceId_ >= 0) {
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceId_), "cudaSetDevice failed");
    }

    if (!UploadGaussianKernel(clampedRadius, clampedSigma, stream_)) {
        kernelUploaded_ = false;
        return false;
    }

    cachedKernelRadius_ = clampedRadius;
    cachedKernelSigma_ = clampedSigma;
    kernelUploaded_ = true;
    return true;
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

bool CudaInteropSurface::ProcessFrame(const ProcessingInput& input, const ProcessingSettings& settings) {
    if (!valid_ || !input.hostBgra || input.width == 0 || input.height == 0) {
        return false;
    }

    if (input.width != width_ || input.height != height_) {
        return false;
    }

    try {
        if (!EnsureDeviceBuffers(input.width, input.height)) {
            return false;
        }

        ThrowIfCudaFailed(cudaMemcpy2DAsync(deviceBufferA_, devicePitchA_,
                                            input.hostBgra, input.hostStride,
                                            static_cast<size_t>(input.width) * sizeof(uchar4), input.height,
                                            cudaMemcpyHostToDevice, stream_),
                          "cudaMemcpy2DAsync host->device failed");

        uchar4* current = deviceBufferA_;
        uchar4* alternate = deviceBufferB_;
        size_t currentPitch = devicePitchA_;
        size_t alternatePitch = devicePitchB_;

        auto swapBuffers = [&]() {
            std::swap(current, alternate);
            std::swap(currentPitch, alternatePitch);
        };

        if (settings.enableBlackWhite) {
            LaunchBlackWhiteLinear(alternate, alternatePitch, current, currentPitch,
                                   static_cast<int>(input.width), static_cast<int>(input.height),
                                   settings.blackWhiteThreshold, stream_);
            swapBuffers();
        }

        if (settings.enableZoom) {
            LaunchZoomLinear(alternate, alternatePitch, current, currentPitch,
                             static_cast<int>(input.width), static_cast<int>(input.height),
                             settings.zoomAmount, settings.zoomCenterX, settings.zoomCenterY, stream_);
            swapBuffers();
        }

        if (settings.enableBlur && settings.blurRadius > 0 && settings.blurSigma > 0.0f) {
            if (!EnsureGaussianKernel(settings.blurRadius, settings.blurSigma)) {
                kernelUploaded_ = false;
                return false;
            }

            LaunchGaussianBlurLinear(alternate, alternatePitch,
                                     deviceScratch_, devicePitchScratch_,
                                     current, currentPitch,
                                     static_cast<int>(input.width), static_cast<int>(input.height),
                                     stream_);
            swapBuffers();
        }

        if (settings.drawFocusMarker) {
            LaunchFocusMarkerLinear(current, currentPitch,
                                    static_cast<int>(input.width), static_cast<int>(input.height),
                                    settings.zoomCenterX, settings.zoomCenterY, stream_);
        }

        ThrowIfCudaFailed(cudaMemcpy2DToArrayAsync(level0Array_, 0, 0,
                                                   current, currentPitch,
                                                   static_cast<size_t>(input.width) * sizeof(uchar4), input.height,
                                                   cudaMemcpyDeviceToDevice, stream_),
                          "cudaMemcpy2DToArrayAsync failed");

        ThrowIfCudaFailed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed");
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace openzoom

#endif // OPENZOOM_HAS_CUDA_EXT_MEMORY

#endif // _WIN32
