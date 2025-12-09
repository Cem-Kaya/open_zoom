#ifdef _WIN32

#include "openzoom/cuda/cuda_interop.hpp"
#include "openzoom/cuda/cuda_kernels.hpp"

#include <d3d12.h>

#if OPENZOOM_HAS_CUDA_EXT_MEMORY

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <mutex>

#include <QDebug>

namespace openzoom {

namespace {

bool gWarnedFp16Unsupported = false;

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

constexpr int kMaxCudaBlurRadius = 50;

bool EnsureCudaDriverInitialized()
{
    static std::once_flag initFlag;
    static CUresult initResult = CUDA_SUCCESS;
    std::call_once(initFlag, []() {
        initResult = cuInit(0);
    });
    if (initResult != CUDA_SUCCESS) {
        qWarning() << "cuInit failed" << static_cast<int>(initResult);
        return false;
    }
    return true;
}

bool QueryDeviceLuid(int deviceId, LUID& luidOut)
{
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11030
    if (!EnsureCudaDriverInitialized()) {
        return false;
    }

    CUdevice cuDevice{};
    if (cuDeviceGet(&cuDevice, deviceId) != CUDA_SUCCESS) {
        return false;
    }

    char luidBuffer[sizeof(LUID)] = {};
    unsigned int nodeMask = 0;
    if (cuDeviceGetLuid(luidBuffer, &nodeMask, cuDevice) == CUDA_SUCCESS) {
        std::memcpy(&luidOut, luidBuffer, sizeof(LUID));
        return true;
    }
#endif
    return false;
}

} // namespace

CudaInteropSurface::CudaInteropSurface(ID3D12Resource* texture, ID3D12Fence* sharedFence) {
    try {
        Initialize(texture, sharedFence);
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
        if (externalSemaphore_ != nullptr) {
            cudaDestroyExternalSemaphore(externalSemaphore_);
            externalSemaphore_ = nullptr;
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

    if (externalSemaphore_ != nullptr) {
        cudaDestroyExternalSemaphore(externalSemaphore_);
        externalSemaphore_ = nullptr;
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

    bool matched = false;
    cudaDeviceProp matchedProps{};

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp properties{};
        ThrowIfCudaFailed(cudaGetDeviceProperties(&properties, deviceId), "cudaGetDeviceProperties failed");

        LUID deviceLuid{};
        if (QueryDeviceLuid(deviceId, deviceLuid) &&
            std::memcmp(&adapterLuid, &deviceLuid, sizeof(LUID)) == 0) {
            matched = true;
            matchedProps = properties;
            ThrowIfCudaFailed(cudaSetDevice(deviceId), "cudaSetDevice failed");
            cudaDeviceId_ = deviceId;
            qInfo() << "CUDA device" << deviceId << properties.name << "matches DXGI adapter";
            break;
        }
    }

    if (!matched) {
        if (deviceCount > 0) {
            qWarning() << "No CUDA device LUID matched; using device 0 as fallback";
            cudaDeviceProp properties{};
            ThrowIfCudaFailed(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties failed");
            ThrowIfCudaFailed(cudaSetDevice(0), "cudaSetDevice failed");
            cudaDeviceId_ = 0;
            matchedProps = properties;
        } else {
            lastError_ = "No CUDA devices available";
            qWarning() << "No CUDA devices reported by runtime";
            return false;
        }
    }

    qInfo() << "Using CUDA device" << cudaDeviceId_ << matchedProps.name;
    return true;
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
        qInfo() << "Importing external memory (size" << static_cast<unsigned long long>(allocationInfo.SizeInBytes)
                << ", flags=cudaExternalMemoryDedicated)";
        ThrowIfCudaFailed(cudaImportExternalMemory(&externalMemory_, &memoryDesc),
                          "cudaImportExternalMemory failed");
        qInfo() << "Imported external memory for CUDA (size" << static_cast<unsigned long long>(allocationInfo.SizeInBytes) << ")";
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

void CudaInteropSurface::Initialize(ID3D12Resource* texture, ID3D12Fence* sharedFence) {
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

    if (sharedFence != nullptr) {
        ImportFenceSemaphore(device.Get(), sharedFence);
    }
}

bool CudaInteropSurface::EnsureDeviceBuffers(unsigned int width, unsigned int height, CudaBufferFormat format) {
    CudaBufferFormat resolvedFormat = format;
    if (format == CudaBufferFormat::kRgba16F) {
        if (!gWarnedFp16Unsupported) {
            qWarning() << "CUDA staging format RGBA16F not yet implemented; falling back to RGBA8";
            gWarnedFp16Unsupported = true;
        }
        resolvedFormat = CudaBufferFormat::kRgba8;
    }

    if (deviceBufferA_ && deviceBufferB_ &&
        deviceWidth_ == width && deviceHeight_ == height &&
        bufferFormat_ == resolvedFormat) {
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
    bufferFormat_ = resolvedFormat;
    return true;
}

void CudaInteropSurface::ImportFenceSemaphore(ID3D12Device* device, ID3D12Fence* fence) {
    if (!device || !fence) {
        return;
    }

    WindowsSecurityAttributes securityAttributes;
    HANDLE fenceHandle = nullptr;
    ThrowIfFailed(device->CreateSharedHandle(fence,
                                            securityAttributes.get(),
                                            GENERIC_ALL,
                                            nullptr,
                                            &fenceHandle),
                  "Failed to create shared handle for D3D12 fence");

    auto cleanupHandle = [&]() {
        if (fenceHandle) {
            CloseHandle(fenceHandle);
            fenceHandle = nullptr;
        }
    };

    cudaExternalSemaphoreHandleDesc semaphoreDesc{};
    semaphoreDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    semaphoreDesc.handle.win32.handle = fenceHandle;
    semaphoreDesc.flags = 0;

    try {
        ThrowIfCudaFailed(cudaImportExternalSemaphore(&externalSemaphore_, &semaphoreDesc),
                          "cudaImportExternalSemaphore failed");
        qInfo() << "Imported shared D3D12 fence into CUDA";
    } catch (...) {
        cleanupHandle();
        throw;
    }

    cleanupHandle();
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
    bufferFormat_ = CudaBufferFormat::kRgba8;
    ReleaseTemporalHistory();
    kernelUploaded_ = false;
}

bool CudaInteropSurface::EnsureTemporalHistory(unsigned int width, unsigned int height) {
    if (deviceTemporalHistory_ && historyWidth_ == width && historyHeight_ == height) {
        return true;
    }

    ReleaseTemporalHistory();

    if (cudaDeviceId_ >= 0) {
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceId_), "cudaSetDevice failed");
    }

    ThrowIfCudaFailed(cudaMallocPitch(reinterpret_cast<void**>(&deviceTemporalHistory_), &devicePitchHistory_,
                                      static_cast<size_t>(width) * sizeof(float4), height),
                      "cudaMallocPitch history buffer failed");
    historyWidth_ = width;
    historyHeight_ = height;
    temporalHistoryValid_ = false;
    return true;
}

void CudaInteropSurface::ReleaseTemporalHistory() {
    if (deviceTemporalHistory_) {
        cudaFree(deviceTemporalHistory_);
        deviceTemporalHistory_ = nullptr;
    }
    devicePitchHistory_ = 0;
    historyWidth_ = 0;
    historyHeight_ = 0;
    temporalHistoryValid_ = false;
}

void CudaInteropSurface::ResetTemporalHistory() {
    temporalHistoryValid_ = false;
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

bool CudaInteropSurface::ProcessFrame(const ProcessingInput& input,
                                      const ProcessingSettings& settings,
                                      const FenceSyncParams& fenceSync) {
    if (!valid_ || !input.hostPixels || input.width == 0 || input.height == 0) {
        return false;
    }

    if (input.width != width_ || input.height != height_) {
        return false;
    }

    const size_t pixelSize = static_cast<size_t>(input.pixelSizeBytes);
    if (pixelSize == 0) {
        qWarning() << "ProcessFrame aborted: invalid pixel size";
        return false;
    }

    CudaBufferFormat resolvedFormat = settings.stagingFormat;
    if (resolvedFormat == CudaBufferFormat::kRgba16F) {
        resolvedFormat = CudaBufferFormat::kRgba8;
    }

    try {
        if (fenceSync.enable && externalSemaphore_ != nullptr) {
            cudaExternalSemaphoreWaitParams waitParams{};
            waitParams.params.fence.value = fenceSync.waitValue;
            waitParams.flags = 0;
            ThrowIfCudaFailed(cudaWaitExternalSemaphoresAsync(&externalSemaphore_, &waitParams, 1, stream_),
                              "cudaWaitExternalSemaphoresAsync failed");
        }

        if (!EnsureDeviceBuffers(input.width, input.height, resolvedFormat)) {
            return false;
        }

        if (pixelSize != sizeof(uchar4)) {
            if (!gWarnedFp16Unsupported) {
                qWarning() << "ProcessFrame RGBA16F upload requested but GPU kernel path expects RGBA8; using RGBA8 upload";
                gWarnedFp16Unsupported = true;
            }
        }

        ThrowIfCudaFailed(cudaMemcpy2DAsync(deviceBufferA_, devicePitchA_,
                                            static_cast<const uint8_t*>(input.hostPixels), input.hostStrideBytes,
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

        if (settings.enableSpatialSharpen) {
            if (settings.spatialUpscaler == SpatialUpscaler::kNis) {
                LaunchNisLinear(alternate, alternatePitch,
                                current, currentPitch,
                                static_cast<int>(input.width), static_cast<int>(input.height),
                                static_cast<int>(input.width), static_cast<int>(input.height),
                                settings.spatialSharpness,
                                stream_);
            } else {
                LaunchFsrEasuRcasLinear(alternate, alternatePitch,
                                        current, currentPitch,
                                        static_cast<int>(input.width), static_cast<int>(input.height),
                                        static_cast<int>(input.width), static_cast<int>(input.height),
                                        settings.spatialSharpness,
                                        stream_);
            }
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

        if (settings.enableTemporalSmoothing && resolvedFormat == CudaBufferFormat::kRgba8) {
            if (!EnsureTemporalHistory(input.width, input.height)) {
                return false;
            }

            const float alpha = std::clamp(settings.temporalSmoothingAlpha, 0.0f, 1.0f);
            LaunchTemporalSmoothLinear(alternate, alternatePitch,
                                       current, currentPitch,
                                       deviceTemporalHistory_, devicePitchHistory_,
                                       static_cast<int>(input.width), static_cast<int>(input.height),
                                       alpha,
                                       temporalHistoryValid_,
                                       stream_);
            temporalHistoryValid_ = true;
            swapBuffers();
        } else {
            temporalHistoryValid_ = false;
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

        if (fenceSync.enable && externalSemaphore_ != nullptr) {
            cudaExternalSemaphoreSignalParams signalParams{};
            signalParams.params.fence.value = fenceSync.signalValue;
            signalParams.flags = 0;
            ThrowIfCudaFailed(cudaSignalExternalSemaphoresAsync(&externalSemaphore_, &signalParams, 1, stream_),
                              "cudaSignalExternalSemaphoresAsync failed");
        } else {
            ThrowIfCudaFailed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed");
        }
        lastError_.clear();
        return true;
    } catch (const std::exception& e) {
        lastError_ = e.what();
        qWarning() << "ProcessFrame exception:" << e.what();
        temporalHistoryValid_ = false;
        return false;
    } catch (...) {
        lastError_ = "Unknown CUDA exception during ProcessFrame";
        qWarning() << "ProcessFrame encountered unknown exception";
        temporalHistoryValid_ = false;
        return false;
    }
}

} // namespace openzoom

#endif // OPENZOOM_HAS_CUDA_EXT_MEMORY

#endif // _WIN32
