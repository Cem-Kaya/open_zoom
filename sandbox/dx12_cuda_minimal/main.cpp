#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>

using Microsoft::WRL::ComPtr;

extern "C" void LaunchFillBuffer(uchar4* destination,
                                 size_t pitch,
                                 unsigned int width,
                                 unsigned int height,
                                 cudaStream_t stream);

namespace {

class WindowsSecurityAttributes {
public:
    WindowsSecurityAttributes()
    {
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

void Log(const std::string& message)
{
    std::cout << "[dx12_cuda_minimal] " << message << std::endl;
}

void LogError(const std::string& message)
{
    std::cerr << "[dx12_cuda_minimal][error] " << message << std::endl;
}

std::string HrToString(HRESULT hr)
{
    char buffer[64]{};
    std::snprintf(buffer, sizeof(buffer), "0x%08lX", static_cast<unsigned long>(hr));
    return std::string(buffer);
}

void ThrowIfFailed(HRESULT hr, const char* message)
{
    if (FAILED(hr)) {
        LogError(std::string(message) + " (hr=" + HrToString(hr) + ")");
        throw std::runtime_error(message);
    }
}

void ThrowIfCudaFailed(cudaError_t status, const char* message)
{
    if (status != cudaSuccess) {
        LogError(std::string(message) + ": " + cudaGetErrorString(status));
        throw std::runtime_error(message);
    }
}

ComPtr<IDXGIFactory6> CreateFactory()
{
    UINT factoryFlags = 0U;
#if defined(_DEBUG)
    ComPtr<IDXGIInfoQueue> infoQueue;
    if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&infoQueue)))) {
        factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }
#endif

    ComPtr<IDXGIFactory6> factory;
    ThrowIfFailed(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2 failed");
    Log("Created DXGI factory");
    return factory;
}

ComPtr<IDXGIAdapter1> PickHardwareAdapter(IDXGIFactory6* factory)
{
    for (UINT adapterIndex = 0;; ++adapterIndex) {
        ComPtr<IDXGIAdapter1> adapter;
        if (DXGI_ERROR_NOT_FOUND == factory->EnumAdapters1(adapterIndex, &adapter)) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }

        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr))) {
            Log("Selected adapter: " + std::string("(LUID hi=" + std::to_string(desc.AdapterLuid.HighPart) +
                                                     " lo=" + std::to_string(desc.AdapterLuid.LowPart) + ")"));
            return adapter;
        }
    }

    return nullptr;
}

bool EnsureCudaDriverInitialized()
{
    static std::once_flag initFlag;
    static CUresult initResult = CUDA_SUCCESS;
    std::call_once(initFlag, []() {
        initResult = cuInit(0);
    });
    if (initResult != CUDA_SUCCESS) {
        LogError(std::string("cuInit failed: ") + std::to_string(static_cast<int>(initResult)));
        return false;
    }
    return true;
}

bool QueryDeviceLuid(int deviceId, LUID& outLuid)
{
#if defined(_WIN32) && defined(CUDA_VERSION) && CUDA_VERSION >= 11030
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
        std::memcpy(&outLuid, luidBuffer, sizeof(LUID));
        return true;
    }
#endif
    return false;
}

bool SelectCudaDeviceMatching(const LUID& luid, int& outDeviceId)
{
    int count = 0;
    ThrowIfCudaFailed(cudaGetDeviceCount(&count), "cudaGetDeviceCount failed");
    Log("CUDA device count: " + std::to_string(count));

    bool matched = false;
    for (int deviceId = 0; deviceId < count; ++deviceId) {
        cudaDeviceProp props{};
        ThrowIfCudaFailed(cudaGetDeviceProperties(&props, deviceId), "cudaGetDeviceProperties failed");

#if defined(_WIN32)
        LUID deviceLuid{};
        if (QueryDeviceLuid(deviceId, deviceLuid) && std::memcmp(&luid, &deviceLuid, sizeof(LUID)) == 0) {
            Log(std::string("Matched CUDA device ") + std::to_string(deviceId) + ": " + props.name);
            ThrowIfCudaFailed(cudaSetDevice(deviceId), "cudaSetDevice failed");
            outDeviceId = deviceId;
            matched = true;
            break;
        }
#endif
    }

    if (matched) {
        return true;
    }

    if (count > 0) {
        LogError("No CUDA device matched the DXGI adapter LUID; falling back to device 0");
        ThrowIfCudaFailed(cudaSetDevice(0), "cudaSetDevice failed");
        outDeviceId = 0;
        return true;
    }

    LogError("No CUDA devices available for interop");
    return false;
}

struct TextureInteropResources {
    cudaExternalMemory_t externalMemory{};
    cudaMipmappedArray_t mipArray{};
    cudaArray_t level0Array{};
    cudaStream_t stream{};

    ~TextureInteropResources()
    {
        if (externalMemory != nullptr) {
            cudaDestroyExternalMemory(externalMemory);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

bool ImportTextureToCuda(ID3D12Device* device,
                         ID3D12Resource* texture,
                         bool dedicatedFlag,
                         const D3D12_RESOURCE_ALLOCATION_INFO& allocationInfo,
                         unsigned width,
                         unsigned height)
{
    HANDLE sharedHandle = nullptr;
    WindowsSecurityAttributes securityAttributes;
    ThrowIfFailed(device->CreateSharedHandle(texture,
                                             securityAttributes.get(),
                                             GENERIC_ALL,
                                             nullptr,
                                             &sharedHandle),
                  "CreateSharedHandle failed");

    cudaExternalMemoryHandleDesc memoryDesc{};
    memoryDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    memoryDesc.handle.win32.handle = sharedHandle;
    memoryDesc.size = allocationInfo.SizeInBytes;
    memoryDesc.flags = dedicatedFlag ? cudaExternalMemoryDedicated : 0;

    TextureInteropResources resources;

    Log(std::string("Attempting cudaImportExternalMemory (flags=") +
        (dedicatedFlag ? "cudaExternalMemoryDedicated" : "0") + ")");

    cudaError_t importStatus = cudaImportExternalMemory(&resources.externalMemory, &memoryDesc);
    if (importStatus != cudaSuccess) {
        LogError(std::string("cudaImportExternalMemory failed: ") + cudaGetErrorString(importStatus));
        CloseHandle(sharedHandle);
        return false;
    }

    CloseHandle(sharedHandle);

    ThrowIfCudaFailed(cudaStreamCreate(&resources.stream), "cudaStreamCreate failed");

    cudaExternalMemoryMipmappedArrayDesc arrayDesc{};
    arrayDesc.offset = 0;
    arrayDesc.numLevels = 1;
    arrayDesc.extent = make_cudaExtent(width, height, 1);
    arrayDesc.formatDesc = cudaCreateChannelDesc<uchar4>();
    arrayDesc.flags = cudaArraySurfaceLoadStore | cudaArrayColorAttachment;

    ThrowIfCudaFailed(cudaExternalMemoryGetMappedMipmappedArray(&resources.mipArray,
                                                                resources.externalMemory,
                                                                &arrayDesc),
                      "cudaExternalMemoryGetMappedMipmappedArray failed");
    ThrowIfCudaFailed(cudaGetMipmappedArrayLevel(&resources.level0Array, resources.mipArray, 0),
                      "cudaGetMipmappedArrayLevel failed");

    uchar4* deviceBuffer = nullptr;
    size_t devicePitch = 0;
    ThrowIfCudaFailed(cudaMallocPitch(reinterpret_cast<void**>(&deviceBuffer),
                                      &devicePitch,
                                      width * sizeof(uchar4),
                                      height),
                      "cudaMallocPitch failed");

    cudaGetLastError();
    LaunchFillBuffer(deviceBuffer, devicePitch, width, height, resources.stream);
    ThrowIfCudaFailed(cudaGetLastError(), "Fill kernel launch failed");

    ThrowIfCudaFailed(cudaMemcpy2DToArrayAsync(resources.level0Array,
                                               0,
                                               0,
                                               deviceBuffer,
                                               devicePitch,
                                               width * sizeof(uchar4),
                                               height,
                                               cudaMemcpyDeviceToDevice,
                                               resources.stream),
                      "cudaMemcpy2DToArrayAsync failed");

    ThrowIfCudaFailed(cudaStreamSynchronize(resources.stream), "cudaStreamSynchronize failed");

    std::vector<uchar4> hostCopy(width * height);
    ThrowIfCudaFailed(cudaMemcpy2DFromArray(hostCopy.data(),
                                            width * sizeof(uchar4),
                                            resources.level0Array,
                                            0,
                                            0,
                                            width * sizeof(uchar4),
                                            height,
                                            cudaMemcpyDeviceToHost),
                      "cudaMemcpy2DFromArray failed");

    cudaFree(deviceBuffer);

    const uchar4 sample = hostCopy.front();
    Log("CUDA surface write succeeded. First pixel = (" +
        std::to_string(static_cast<int>(sample.x)) + ", " +
        std::to_string(static_cast<int>(sample.y)) + ", " +
        std::to_string(static_cast<int>(sample.z)) + ", " +
        std::to_string(static_cast<int>(sample.w)) + ")");

    return true;
}

} // namespace

int main()
{
    try {
        Log("Starting Direct3D12 + CUDA minimal interop test");

        ComPtr<IDXGIFactory6> factory = CreateFactory();
        ComPtr<IDXGIAdapter1> adapter = PickHardwareAdapter(factory.Get());
        if (!adapter) {
            LogError("No suitable hardware adapter found");
            return EXIT_FAILURE;
        }

        ComPtr<ID3D12Device> device;
        ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)),
                      "D3D12CreateDevice failed");
        Log("Created D3D12 device");

        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);

        int cudaDeviceId = -1;
        if (!SelectCudaDeviceMatching(desc.AdapterLuid, cudaDeviceId)) {
            LogError("Failed to select matching CUDA device");
            return EXIT_FAILURE;
        }
        Log("Using CUDA device id " + std::to_string(cudaDeviceId));

        const UINT width = 640;
        const UINT height = 360;

        D3D12_RESOURCE_DESC texDesc{};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Alignment = 0;
        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.DepthOrArraySize = 1;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        texDesc.SampleDesc.Quality = 0;
        texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;

        ComPtr<ID3D12Resource> texture;
        ThrowIfFailed(device->CreateCommittedResource(&heapProps,
                                                      D3D12_HEAP_FLAG_SHARED,
                                                      &texDesc,
                                                      D3D12_RESOURCE_STATE_COMMON,
                                                      nullptr,
                                                      IID_PPV_ARGS(&texture)),
                      "CreateCommittedResource failed");
        Log("Created shared D3D12 texture");

        D3D12_RESOURCE_ALLOCATION_INFO allocationInfo =
            device->GetResourceAllocationInfo(0, 1, &texDesc);

        Log("Testing import without cudaExternalMemoryDedicated flag");
        const bool withoutDedicated = ImportTextureToCuda(device.Get(),
                                                          texture.Get(),
                                                          false,
                                                          allocationInfo,
                                                          width,
                                                          height);
        if (!withoutDedicated) {
            Log("As expected, external memory import failed without cudaExternalMemoryDedicated");
        } else {
            LogError("Import unexpectedly succeeded without dedicated flag");
        }

        Log("Testing import with cudaExternalMemoryDedicated flag");
        const bool withDedicated = ImportTextureToCuda(device.Get(),
                                                       texture.Get(),
                                                       true,
                                                       allocationInfo,
                                                       width,
                                                       height);
        if (!withDedicated) {
            LogError("Import failed even with cudaExternalMemoryDedicated flag");
            return EXIT_FAILURE;
        }

        Log("CUDA/D3D12 interop validated successfully");
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        LogError(std::string("Unhandled exception: ") + e.what());
        return EXIT_FAILURE;
    }
}
