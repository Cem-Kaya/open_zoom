#ifdef _WIN32

#include "openzoom/cuda/cuda_interop.hpp"

#include <d3d12.h>

#if OPENZOOM_HAS_CUDA_EXT_MEMORY

#include "openzoom/cuda/cuda_kernels.hpp"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <mutex>
#include <vector>

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

// Defensive ceiling for driver-supplied frame dimensions; anything above this
// is treated as a corrupt input descriptor (also keeps width*height*4 far away
// from 32-bit overflow).
constexpr unsigned int kMaxInputExtent = 16384;

// Keystone tuning. A snapshot of the small luma image is exported roughly once
// per second (at 30 fps); after kKeystoneStaleFrames without an accepted
// detection the warp eases back to identity.
constexpr unsigned int kKeystoneSnapshotPeriod = 30;
constexpr unsigned int kKeystoneStaleFrames = 150;   // ~5 s at 30 fps
constexpr float kKeystoneCornerLerp = 0.15f;         // per accepted detection
constexpr float kKeystoneIdentityReturnLerp = 0.03f; // per frame once stale
constexpr float kKeystoneThresholdSigmaK = 0.5f;
constexpr float kKeystoneMinAreaFraction = 0.15f;
constexpr float kKeystoneIdentityEpsilonPx = 0.75f;

bool gWarnedBgraRotationIgnored = false;
bool gWarnedInvalidInput = false;

// Shared with stabilization: pick integer downsample factors so the small luma
// image fits in 320x180 (ceil division for the factor, then again for the
// resulting extent).
void ComputeSmallLumaDims(unsigned int width, unsigned int height,
                          unsigned int& factorX, unsigned int& factorY,
                          unsigned int& smallWidth, unsigned int& smallHeight) {
    constexpr unsigned int kTargetSmallWidth = 320;
    constexpr unsigned int kTargetSmallHeight = 180;
    factorX = (width + kTargetSmallWidth - 1) / kTargetSmallWidth;
    factorY = (height + kTargetSmallHeight - 1) / kTargetSmallHeight;
    smallWidth = (width + factorX - 1) / factorX;
    smallHeight = (height + factorY - 1) / factorY;
}

struct KeystoneQuadDetection {
    float2 corners[4]{};  // TL, TR, BR, BL in small-image pixel coordinates
    bool valid{false};
};

float2 LerpPoint(const float2& a, const float2& b, float t) {
    return make_float2(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
}

float PointDistance(const float2& a, const float2& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// TL->TR->BR->BL must turn consistently clockwise (positive cross products in
// y-down image coordinates); near-degenerate quads are rejected too.
bool QuadIsConvexClockwise(const float2 quad[4]) {
    for (int i = 0; i < 4; ++i) {
        const float2& p0 = quad[i];
        const float2& p1 = quad[(i + 1) & 3];
        const float2& p2 = quad[(i + 2) & 3];
        const float cross = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x);
        if (cross <= 1.0f) {
            return false;
        }
    }
    return true;
}

// CPU detection of the bright projected-slide quadrilateral on the tiny
// (<=320x180) luma snapshot. Runs a few times per second on a frame that was
// copied out asynchronously, so it never touches the GPU timeline.
// Steps: threshold at mean + k*sigma, largest 4-connected bright component via
// flood fill, corners from the component's x+y / x-y extremes, then area,
// convexity and aspect validation.
KeystoneQuadDetection DetectProjectedQuad(const float* luma, int width, int height) {
    KeystoneQuadDetection detection{};
    if (luma == nullptr || width <= 0 || height <= 0) {
        return detection;
    }
    const int count = width * height;

    double sum = 0.0;
    double sumSq = 0.0;
    for (int i = 0; i < count; ++i) {
        sum += luma[i];
        sumSq += static_cast<double>(luma[i]) * luma[i];
    }
    const double mean = sum / count;
    const double variance = std::max(sumSq / count - mean * mean, 0.0);
    const double sigma = std::sqrt(variance);
    // k is deliberately small: when the slide dominates the frame the mean sits
    // close to the slide brightness and a large k would push the threshold
    // above it. The area/shape validation below rejects bad segmentations.
    const float threshold =
        static_cast<float>(std::min(mean + kKeystoneThresholdSigmaK * sigma, 250.0));

    // 0 = below threshold, 1 = bright & unvisited, 2 = visited.
    std::vector<uint8_t> state(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        state[static_cast<size_t>(i)] = (luma[i] >= threshold) ? 1u : 0u;
    }

    std::vector<int> stack;
    stack.reserve(1024);

    int bestArea = 0;
    float2 bestCorners[4]{};

    for (int seed = 0; seed < count; ++seed) {
        if (state[static_cast<size_t>(seed)] != 1u) {
            continue;
        }

        int area = 0;
        int minSum = std::numeric_limits<int>::max();
        int maxSum = std::numeric_limits<int>::min();
        int minDiff = std::numeric_limits<int>::max();
        int maxDiff = std::numeric_limits<int>::min();
        float2 tl{}, tr{}, br{}, bl{};

        state[static_cast<size_t>(seed)] = 2u;
        stack.clear();
        stack.push_back(seed);
        while (!stack.empty()) {
            const int idx = stack.back();
            stack.pop_back();
            const int x = idx % width;
            const int y = idx / width;
            ++area;

            // Cheap robust corner picker: extremes of x+y and x-y.
            const int s = x + y;
            const int d = x - y;
            const float2 p = make_float2(static_cast<float>(x), static_cast<float>(y));
            if (s < minSum) { minSum = s; tl = p; }
            if (s > maxSum) { maxSum = s; br = p; }
            if (d > maxDiff) { maxDiff = d; tr = p; }
            if (d < minDiff) { minDiff = d; bl = p; }

            if (x > 0 && state[static_cast<size_t>(idx - 1)] == 1u) {
                state[static_cast<size_t>(idx - 1)] = 2u;
                stack.push_back(idx - 1);
            }
            if (x + 1 < width && state[static_cast<size_t>(idx + 1)] == 1u) {
                state[static_cast<size_t>(idx + 1)] = 2u;
                stack.push_back(idx + 1);
            }
            if (y > 0 && state[static_cast<size_t>(idx - width)] == 1u) {
                state[static_cast<size_t>(idx - width)] = 2u;
                stack.push_back(idx - width);
            }
            if (y + 1 < height && state[static_cast<size_t>(idx + width)] == 1u) {
                state[static_cast<size_t>(idx + width)] = 2u;
                stack.push_back(idx + width);
            }
        }

        if (area > bestArea) {
            bestArea = area;
            bestCorners[0] = tl;
            bestCorners[1] = tr;
            bestCorners[2] = br;
            bestCorners[3] = bl;
        }
    }

    if (bestArea < static_cast<int>(kKeystoneMinAreaFraction * static_cast<float>(count))) {
        return detection;
    }
    if (!QuadIsConvexClockwise(bestCorners)) {
        return detection;
    }

    const float avgWidth = 0.5f * (PointDistance(bestCorners[0], bestCorners[1]) +
                                   PointDistance(bestCorners[3], bestCorners[2]));
    const float avgHeight = 0.5f * (PointDistance(bestCorners[0], bestCorners[3]) +
                                    PointDistance(bestCorners[1], bestCorners[2]));
    if (avgWidth < 0.25f * static_cast<float>(width) ||
        avgHeight < 0.25f * static_cast<float>(height)) {
        return detection;
    }
    const float aspect = avgWidth / std::max(avgHeight, 1.0f);
    if (aspect < 0.3f || aspect > 4.0f) {
        return detection;
    }

    for (int i = 0; i < 4; ++i) {
        detection.corners[i] = bestCorners[i];
    }
    detection.valid = true;
    return detection;
}

// 4-point DLT: homography H with H * (rect corner) ~ quad corner, where the
// rect corners are (0,0), (w-1,0), (w-1,h-1), (0,h-1) and h22 is fixed to 1.
// The 8x8 system is solved with Gauss-Jordan elimination and partial pivoting
// in double precision — 4 correspondences, so this costs microseconds on the
// host and needs no external dependency.
bool SolveRectToQuadHomography(float width, float height, const float2 quad[4], float out[9]) {
    const double rectX[4] = {0.0, static_cast<double>(width) - 1.0,
                             static_cast<double>(width) - 1.0, 0.0};
    const double rectY[4] = {0.0, 0.0,
                             static_cast<double>(height) - 1.0, static_cast<double>(height) - 1.0};

    double a[8][9] = {};
    for (int i = 0; i < 4; ++i) {
        const double x = rectX[i];
        const double y = rectY[i];
        const double u = static_cast<double>(quad[i].x);
        const double v = static_cast<double>(quad[i].y);
        double* row0 = a[2 * i];
        double* row1 = a[2 * i + 1];
        row0[0] = x;  row0[1] = y;  row0[2] = 1.0;
        row0[6] = -u * x;  row0[7] = -u * y;  row0[8] = u;
        row1[3] = x;  row1[4] = y;  row1[5] = 1.0;
        row1[6] = -v * x;  row1[7] = -v * y;  row1[8] = v;
    }

    for (int col = 0; col < 8; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 8; ++row) {
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) {
                pivot = row;
            }
        }
        if (std::abs(a[pivot][col]) < 1e-9) {
            return false;
        }
        if (pivot != col) {
            for (int c = 0; c < 9; ++c) {
                std::swap(a[pivot][c], a[col][c]);
            }
        }
        const double inv = 1.0 / a[col][col];
        for (int c = col; c < 9; ++c) {
            a[col][c] *= inv;
        }
        for (int row = 0; row < 8; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = a[row][col];
            if (factor == 0.0) {
                continue;
            }
            for (int c = col; c < 9; ++c) {
                a[row][c] -= factor * a[col][c];
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<float>(a[i][8]);
    }
    out[8] = 1.0f;
    return true;
}

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
        SynchronizeStream();
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

// ProcessFrame() returns with kernels still queued on stream_ when fence sync is
// active, so every teardown path must drain the stream before releasing resources
// the queued work references (surface object, external memory, device buffers).
void CudaInteropSurface::SynchronizeStream() noexcept {
    if (stream_ == nullptr) {
        return;
    }
    const cudaError_t status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
        qWarning() << "cudaStreamSynchronize during teardown failed:" << cudaGetErrorString(status);
    }
}

CudaInteropSurface::~CudaInteropSurface() {
    SynchronizeStream();

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
    if (deviceBufferA_ || deviceBufferB_ || deviceScratch_ || deviceTemporalHistory_) {
        SynchronizeStream();
    }
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
    ReleaseStabilization();
    ReleaseRawInput();
    ReleaseKeystone();
    ReleaseAutoContrast();
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
        SynchronizeStream();
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

bool CudaInteropSurface::EnsureStabilizationBuffers(unsigned int width, unsigned int height) {
    if (deviceStabState_ && stabFullWidth_ == width && stabFullHeight_ == height) {
        return true;
    }

    ReleaseStabilization();

    if (cudaDeviceId_ >= 0) {
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceId_), "cudaSetDevice failed");
    }

    unsigned int factorX = 0;
    unsigned int factorY = 0;
    unsigned int smallWidth = 0;
    unsigned int smallHeight = 0;
    ComputeSmallLumaDims(width, height, factorX, factorY, smallWidth, smallHeight);

    ThrowIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&deviceStabLuma_),
                                 static_cast<size_t>(smallWidth) * smallHeight * sizeof(float)),
                      "cudaMalloc stabilization luma buffer failed");
    ThrowIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&deviceStabColProjCurr_),
                                 static_cast<size_t>(smallWidth) * sizeof(float)),
                      "cudaMalloc stabilization column projection failed");
    ThrowIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&deviceStabRowProjCurr_),
                                 static_cast<size_t>(smallHeight) * sizeof(float)),
                      "cudaMalloc stabilization row projection failed");
    ThrowIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&deviceStabColProjPrev_),
                                 static_cast<size_t>(smallWidth) * sizeof(float)),
                      "cudaMalloc stabilization previous column projection failed");
    ThrowIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&deviceStabRowProjPrev_),
                                 static_cast<size_t>(smallHeight) * sizeof(float)),
                      "cudaMalloc stabilization previous row projection failed");
    // Allocated last so a partially built set is re-released on the next call.
    ThrowIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&deviceStabState_),
                                 sizeof(StabilizationState)),
                      "cudaMalloc stabilization state failed");

    stabSmallWidth_ = smallWidth;
    stabSmallHeight_ = smallHeight;
    stabFactorX_ = factorX;
    stabFactorY_ = factorY;
    stabFullWidth_ = width;
    stabFullHeight_ = height;
    stabPrevValid_ = false;
    return true;
}

void CudaInteropSurface::ReleaseStabilization() {
    if (deviceStabLuma_ || deviceStabColProjCurr_ || deviceStabRowProjCurr_ ||
        deviceStabColProjPrev_ || deviceStabRowProjPrev_ || deviceStabState_) {
        SynchronizeStream();
    }
    if (deviceStabLuma_) {
        cudaFree(deviceStabLuma_);
        deviceStabLuma_ = nullptr;
    }
    if (deviceStabColProjCurr_) {
        cudaFree(deviceStabColProjCurr_);
        deviceStabColProjCurr_ = nullptr;
    }
    if (deviceStabRowProjCurr_) {
        cudaFree(deviceStabRowProjCurr_);
        deviceStabRowProjCurr_ = nullptr;
    }
    if (deviceStabColProjPrev_) {
        cudaFree(deviceStabColProjPrev_);
        deviceStabColProjPrev_ = nullptr;
    }
    if (deviceStabRowProjPrev_) {
        cudaFree(deviceStabRowProjPrev_);
        deviceStabRowProjPrev_ = nullptr;
    }
    if (deviceStabState_) {
        cudaFree(deviceStabState_);
        deviceStabState_ = nullptr;
    }
    stabSmallWidth_ = 0;
    stabSmallHeight_ = 0;
    stabFactorX_ = 0;
    stabFactorY_ = 0;
    stabFullWidth_ = 0;
    stabFullHeight_ = 0;
    stabPrevValid_ = false;
}

// Host-side flag only: with previousValid false the estimate kernel zeroes the
// motion state and re-seeds the previous projections on the next frame, so no
// device work (and no stream sync) is needed here.
void CudaInteropSurface::ResetStabilization() {
    stabPrevValid_ = false;
}

// Device staging for raw camera formats (NV12 / YUY2). Plane row widths:
//   NV12  plane1 = Y (width bytes), plane2 = interleaved UV (2*ceil(w/2) bytes,
//          ceil(h/2) rows)
//   YUY2  plane1 = packed Y0 U Y1 V (4*ceil(w/2) bytes), plane2 unused
bool CudaInteropSurface::EnsureRawInputBuffers(unsigned int width, unsigned int height, int format) {
    if (deviceRawPlane1_ && rawWidth_ == width && rawHeight_ == height && rawFormat_ == format) {
        return true;
    }

    ReleaseRawInput();

    if (cudaDeviceId_ >= 0) {
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceId_), "cudaSetDevice failed");
    }

    if (format == 1) {
        ThrowIfCudaFailed(cudaMallocPitch(reinterpret_cast<void**>(&deviceRawPlane1_), &rawPlane1Pitch_,
                                          static_cast<size_t>(width), height),
                          "cudaMallocPitch NV12 Y plane failed");
        const size_t uvRowBytes = (static_cast<size_t>(width) + 1) / 2 * 2;
        const size_t uvRows = (static_cast<size_t>(height) + 1) / 2;
        ThrowIfCudaFailed(cudaMallocPitch(reinterpret_cast<void**>(&deviceRawPlane2_), &rawPlane2Pitch_,
                                          uvRowBytes, uvRows),
                          "cudaMallocPitch NV12 UV plane failed");
    } else {
        const size_t rowBytes = (static_cast<size_t>(width) + 1) / 2 * 4;
        ThrowIfCudaFailed(cudaMallocPitch(reinterpret_cast<void**>(&deviceRawPlane1_), &rawPlane1Pitch_,
                                          rowBytes, height),
                          "cudaMallocPitch YUY2 buffer failed");
    }

    rawWidth_ = width;
    rawHeight_ = height;
    rawFormat_ = format;
    return true;
}

// Conversion target at pre-rotation extent; only allocated when the GPU also
// rotates (for turns == 0 the converter writes straight into buffer A).
bool CudaInteropSurface::EnsurePreRotateBuffer(unsigned int width, unsigned int height) {
    if (devicePreRotate_ && preRotateWidth_ == width && preRotateHeight_ == height) {
        return true;
    }

    if (devicePreRotate_) {
        SynchronizeStream();
        cudaFree(devicePreRotate_);
        devicePreRotate_ = nullptr;
    }

    if (cudaDeviceId_ >= 0) {
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceId_), "cudaSetDevice failed");
    }

    ThrowIfCudaFailed(cudaMallocPitch(reinterpret_cast<void**>(&devicePreRotate_), &devicePitchPreRotate_,
                                      static_cast<size_t>(width) * sizeof(uchar4), height),
                      "cudaMallocPitch pre-rotate buffer failed");
    preRotateWidth_ = width;
    preRotateHeight_ = height;
    return true;
}

void CudaInteropSurface::ReleaseRawInput() {
    if (deviceRawPlane1_ || deviceRawPlane2_ || devicePreRotate_) {
        SynchronizeStream();
    }
    if (deviceRawPlane1_) {
        cudaFree(deviceRawPlane1_);
        deviceRawPlane1_ = nullptr;
    }
    if (deviceRawPlane2_) {
        cudaFree(deviceRawPlane2_);
        deviceRawPlane2_ = nullptr;
    }
    if (devicePreRotate_) {
        cudaFree(devicePreRotate_);
        devicePreRotate_ = nullptr;
    }
    rawPlane1Pitch_ = 0;
    rawPlane2Pitch_ = 0;
    rawWidth_ = 0;
    rawHeight_ = 0;
    rawFormat_ = -1;
    devicePitchPreRotate_ = 0;
    preRotateWidth_ = 0;
    preRotateHeight_ = 0;
}

bool CudaInteropSurface::EnsureKeystoneResources(unsigned int width, unsigned int height) {
    if (deviceKeystoneLuma_ && keystoneFullWidth_ == width && keystoneFullHeight_ == height) {
        return true;
    }

    ReleaseKeystone();

    if (cudaDeviceId_ >= 0) {
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceId_), "cudaSetDevice failed");
    }

    unsigned int factorX = 0;
    unsigned int factorY = 0;
    unsigned int smallWidth = 0;
    unsigned int smallHeight = 0;
    ComputeSmallLumaDims(width, height, factorX, factorY, smallWidth, smallHeight);

    const size_t lumaBytes = static_cast<size_t>(smallWidth) * smallHeight * sizeof(float);
    ThrowIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&deviceKeystoneLuma_), lumaBytes),
                      "cudaMalloc keystone luma buffer failed");
    // Pinned so the device->host snapshot copy can run truly asynchronously.
    ThrowIfCudaFailed(cudaMallocHost(reinterpret_cast<void**>(&hostKeystoneLuma_), lumaBytes),
                      "cudaMallocHost keystone snapshot buffer failed");
    ThrowIfCudaFailed(cudaEventCreateWithFlags(&keystoneCopyEvent_, cudaEventDisableTiming),
                      "cudaEventCreate keystone event failed");

    keystoneSmallWidth_ = smallWidth;
    keystoneSmallHeight_ = smallHeight;
    keystoneFactorX_ = factorX;
    keystoneFactorY_ = factorY;
    keystoneFullWidth_ = width;
    keystoneFullHeight_ = height;
    keystoneCopyPending_ = false;
    keystoneFrameCounter_ = 0;
    keystoneFramesSinceDetection_ = 0;
    ResetKeystoneCornersToIdentity();
    return true;
}

void CudaInteropSurface::ReleaseKeystone() {
    if (deviceKeystoneLuma_ || hostKeystoneLuma_ || keystoneCopyEvent_ != nullptr) {
        // Drains the in-flight snapshot copy (if any) before its buffers vanish.
        SynchronizeStream();
    }
    if (deviceKeystoneLuma_) {
        cudaFree(deviceKeystoneLuma_);
        deviceKeystoneLuma_ = nullptr;
    }
    if (hostKeystoneLuma_) {
        cudaFreeHost(hostKeystoneLuma_);
        hostKeystoneLuma_ = nullptr;
    }
    if (keystoneCopyEvent_ != nullptr) {
        cudaEventDestroy(keystoneCopyEvent_);
        keystoneCopyEvent_ = nullptr;
    }
    keystoneCopyPending_ = false;
    keystoneSmallWidth_ = 0;
    keystoneSmallHeight_ = 0;
    keystoneFactorX_ = 0;
    keystoneFactorY_ = 0;
    keystoneFullWidth_ = 0;
    keystoneFullHeight_ = 0;
    keystoneFrameCounter_ = 0;
    keystoneFramesSinceDetection_ = 0;
}

void CudaInteropSurface::ResetKeystoneCornersToIdentity() {
    const float w = static_cast<float>(keystoneFullWidth_);
    const float h = static_cast<float>(keystoneFullHeight_);
    if (keystoneFullWidth_ == 0 || keystoneFullHeight_ == 0) {
        return;
    }
    keystoneCorners_[0] = make_float2(0.0f, 0.0f);
    keystoneCorners_[1] = make_float2(w - 1.0f, 0.0f);
    keystoneCorners_[2] = make_float2(w - 1.0f, h - 1.0f);
    keystoneCorners_[3] = make_float2(0.0f, h - 1.0f);
}

// Host-side state only: corners snap back to identity and any in-flight
// snapshot result is discarded. The stream-ordered device->host copy itself is
// left to finish harmlessly (its pinned buffer stays allocated), so no stream
// sync is needed here.
void CudaInteropSurface::ResetKeystone() {
    keystoneCopyPending_ = false;
    keystoneFrameCounter_ = 0;
    keystoneFramesSinceDetection_ = 0;
    ResetKeystoneCornersToIdentity();
}

bool CudaInteropSurface::EnsureAutoContrastBuffers() {
    if (deviceHistogram_ && deviceAutoLevels_) {
        return true;
    }

    ReleaseAutoContrast();

    if (cudaDeviceId_ >= 0) {
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceId_), "cudaSetDevice failed");
    }

    ThrowIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&deviceHistogram_),
                                 256 * sizeof(unsigned int)),
                      "cudaMalloc auto-contrast histogram failed");
    ThrowIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&deviceAutoLevels_), sizeof(float2)),
                      "cudaMalloc auto-contrast levels failed");
    autoLevelsValid_ = false;
    return true;
}

void CudaInteropSurface::ReleaseAutoContrast() {
    if (deviceHistogram_ || deviceAutoLevels_) {
        SynchronizeStream();
    }
    if (deviceHistogram_) {
        cudaFree(deviceHistogram_);
        deviceHistogram_ = nullptr;
    }
    if (deviceAutoLevels_) {
        cudaFree(deviceAutoLevels_);
        deviceAutoLevels_ = nullptr;
    }
    autoLevelsValid_ = false;
}

// Folds a finished small-luma snapshot into the smoothed corner state. Runs on
// the host, on a frame captured a few frames ago — never blocks the stream.
void CudaInteropSurface::ConsumeKeystoneDetection() {
    const KeystoneQuadDetection detection =
        DetectProjectedQuad(hostKeystoneLuma_,
                            static_cast<int>(keystoneSmallWidth_),
                            static_cast<int>(keystoneSmallHeight_));
    if (!detection.valid) {
        // Keep the previous corners (and thus the previous homography).
        return;
    }

    for (int i = 0; i < 4; ++i) {
        // Small-image cell centers -> full-resolution pixels.
        const float2 full = make_float2(
            (detection.corners[i].x + 0.5f) * static_cast<float>(keystoneFactorX_),
            (detection.corners[i].y + 0.5f) * static_cast<float>(keystoneFactorY_));
        keystoneCorners_[i] = LerpPoint(keystoneCorners_[i], full, kKeystoneCornerLerp);
    }
    keystoneFramesSinceDetection_ = 0;
}

// Keystone stage body. Ordering guarantees the render loop never stalls:
//  - the snapshot copy is enqueued on stream_ and only *queried* (cudaEventQuery)
//    on later frames, never waited on;
//  - detection runs on the pinned host copy;
//  - the per-frame homography upload is 9 floats of kernel arguments.
void CudaInteropSurface::RunKeystoneStage(uchar4*& current, uchar4*& alternate,
                                          size_t& currentPitch, size_t& alternatePitch) {
    ++keystoneFrameCounter_;

    if (keystoneCopyPending_) {
        const cudaError_t status = cudaEventQuery(keystoneCopyEvent_);
        if (status == cudaSuccess) {
            keystoneCopyPending_ = false;
            ConsumeKeystoneDetection();
        } else if (status != cudaErrorNotReady) {
            ThrowIfCudaFailed(status, "cudaEventQuery keystone snapshot failed");
        }
    }

    if (!keystoneCopyPending_ && (keystoneFrameCounter_ % kKeystoneSnapshotPeriod) == 0) {
        // Snapshot the *stabilized* frame so detection sees the same image the
        // warp will run on.
        LaunchStabilizationLumaDownsample(deviceKeystoneLuma_,
                                          static_cast<int>(keystoneSmallWidth_),
                                          static_cast<int>(keystoneSmallHeight_),
                                          current, currentPitch,
                                          static_cast<int>(keystoneFullWidth_),
                                          static_cast<int>(keystoneFullHeight_),
                                          static_cast<int>(keystoneFactorX_),
                                          static_cast<int>(keystoneFactorY_),
                                          stream_);
        const size_t lumaBytes =
            static_cast<size_t>(keystoneSmallWidth_) * keystoneSmallHeight_ * sizeof(float);
        ThrowIfCudaFailed(cudaMemcpyAsync(hostKeystoneLuma_, deviceKeystoneLuma_, lumaBytes,
                                          cudaMemcpyDeviceToHost, stream_),
                          "cudaMemcpyAsync keystone snapshot failed");
        ThrowIfCudaFailed(cudaEventRecord(keystoneCopyEvent_, stream_),
                          "cudaEventRecord keystone snapshot failed");
        keystoneCopyPending_ = true;
    }

    ++keystoneFramesSinceDetection_;
    if (keystoneFramesSinceDetection_ > kKeystoneStaleFrames) {
        // Detection has been failing (scene changed, lights on, ...): ease the
        // warp back to identity instead of holding a stale perspective.
        const float w = static_cast<float>(keystoneFullWidth_);
        const float h = static_cast<float>(keystoneFullHeight_);
        const float2 identity[4] = {make_float2(0.0f, 0.0f), make_float2(w - 1.0f, 0.0f),
                                    make_float2(w - 1.0f, h - 1.0f), make_float2(0.0f, h - 1.0f)};
        for (int i = 0; i < 4; ++i) {
            keystoneCorners_[i] =
                LerpPoint(keystoneCorners_[i], identity[i], kKeystoneIdentityReturnLerp);
        }
    }

    // Skip the warp entirely while the corners sit on the identity rectangle
    // (startup, or eased back) — no kernel launch for a no-op stage.
    const float w = static_cast<float>(keystoneFullWidth_);
    const float h = static_cast<float>(keystoneFullHeight_);
    const float2 identity[4] = {make_float2(0.0f, 0.0f), make_float2(w - 1.0f, 0.0f),
                                make_float2(w - 1.0f, h - 1.0f), make_float2(0.0f, h - 1.0f)};
    float maxDeviation = 0.0f;
    for (int i = 0; i < 4; ++i) {
        maxDeviation = std::max(maxDeviation, PointDistance(keystoneCorners_[i], identity[i]));
    }
    if (maxDeviation < kKeystoneIdentityEpsilonPx) {
        return;
    }

    float homography[9];
    if (!SolveRectToQuadHomography(w, h, keystoneCorners_, homography)) {
        return;  // degenerate corner state; leave the frame unwarped
    }

    LaunchKeystoneWarp(alternate, alternatePitch,
                       current, currentPitch,
                       static_cast<int>(keystoneFullWidth_),
                       static_cast<int>(keystoneFullHeight_),
                       homography,
                       stream_);
    std::swap(current, alternate);
    std::swap(currentPitch, alternatePitch);
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

    // ---- Defensive input validation (dims/strides come from camera drivers).
    const int format = input.inputFormat;
    if (format < 0 || format > 2) {
        if (!gWarnedInvalidInput) {
            qWarning() << "ProcessFrame aborted: unknown inputFormat" << format;
            gWarnedInvalidInput = true;
        }
        return false;
    }
    if (input.width > kMaxInputExtent || input.height > kMaxInputExtent) {
        if (!gWarnedInvalidInput) {
            qWarning() << "ProcessFrame aborted: implausible input dims"
                       << input.width << "x" << input.height;
            gWarnedInvalidInput = true;
        }
        return false;
    }

    size_t minStride = 0;
    switch (format) {
    case 1:  // NV12: hostPixels = Y plane, hostPlane2 = interleaved UV plane
        minStride = static_cast<size_t>(input.width);
        if (input.hostPlane2 == nullptr ||
            static_cast<size_t>(input.hostPlane2StrideBytes) <
                (static_cast<size_t>(input.width) + 1) / 2 * 2) {
            if (!gWarnedInvalidInput) {
                qWarning() << "ProcessFrame aborted: invalid NV12 UV plane";
                gWarnedInvalidInput = true;
            }
            return false;
        }
        break;
    case 2:  // YUY2: 4 bytes per pixel pair
        minStride = (static_cast<size_t>(input.width) + 1) / 2 * 4;
        break;
    default:  // BGRA8
        minStride = static_cast<size_t>(input.width) * 4;
        if (input.pixelSizeBytes == 0) {
            qWarning() << "ProcessFrame aborted: invalid pixel size";
            return false;
        }
        break;
    }
    if (static_cast<size_t>(input.hostStrideBytes) < minStride) {
        if (!gWarnedInvalidInput) {
            qWarning() << "ProcessFrame aborted: stride" << input.hostStrideBytes
                       << "smaller than row size" << static_cast<unsigned long long>(minStride);
            gWarnedInvalidInput = true;
        }
        return false;
    }

    // Rotation runs on the GPU only for raw formats. For BGRA the CPU already
    // converted *and* rotated, so a nonzero request must not rotate twice.
    int turns = 0;
    if (format == 0) {
        if (input.rotationQuarterTurns != 0 && !gWarnedBgraRotationIgnored) {
            qWarning() << "ProcessFrame: rotationQuarterTurns ignored for BGRA input"
                          " (CPU path rotates before upload)";
            gWarnedBgraRotationIgnored = true;
        }
    } else {
        turns = input.rotationQuarterTurns & 3;
    }

    // input.width/height describe the host layout (pre-rotation). All device
    // stages and the interop surface run at the post-rotation extent.
    const unsigned int procWidth = ((turns & 1) != 0) ? input.height : input.width;
    const unsigned int procHeight = ((turns & 1) != 0) ? input.width : input.height;
    if (procWidth != width_ || procHeight != height_) {
        return false;
    }

    const size_t pixelSize = static_cast<size_t>(input.pixelSizeBytes);

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

        if (!EnsureDeviceBuffers(procWidth, procHeight, resolvedFormat)) {
            return false;
        }

        if (format == 0) {
            // BGRA path: identical to the historical behavior.
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
        } else {
            // Raw NV12/YUY2 path: upload the compact planes (1.5-2 B/px instead
            // of 4), convert to BGRA on the GPU, then rotate if requested. With
            // no rotation the converter writes straight into buffer A.
            if (!EnsureRawInputBuffers(input.width, input.height, format)) {
                return false;
            }

            uchar4* convertTarget = deviceBufferA_;
            size_t convertPitch = devicePitchA_;
            if (turns != 0) {
                if (!EnsurePreRotateBuffer(input.width, input.height)) {
                    return false;
                }
                convertTarget = devicePreRotate_;
                convertPitch = devicePitchPreRotate_;
            }

            if (format == 1) {
                ThrowIfCudaFailed(cudaMemcpy2DAsync(deviceRawPlane1_, rawPlane1Pitch_,
                                                    input.hostPixels, input.hostStrideBytes,
                                                    static_cast<size_t>(input.width), input.height,
                                                    cudaMemcpyHostToDevice, stream_),
                                  "cudaMemcpy2DAsync NV12 Y plane failed");
                const size_t uvRowBytes = (static_cast<size_t>(input.width) + 1) / 2 * 2;
                const size_t uvRows = (static_cast<size_t>(input.height) + 1) / 2;
                ThrowIfCudaFailed(cudaMemcpy2DAsync(deviceRawPlane2_, rawPlane2Pitch_,
                                                    input.hostPlane2, input.hostPlane2StrideBytes,
                                                    uvRowBytes, uvRows,
                                                    cudaMemcpyHostToDevice, stream_),
                                  "cudaMemcpy2DAsync NV12 UV plane failed");
                LaunchNv12ToBgraLinear(convertTarget, convertPitch,
                                       deviceRawPlane1_, rawPlane1Pitch_,
                                       deviceRawPlane2_, rawPlane2Pitch_,
                                       static_cast<int>(input.width), static_cast<int>(input.height),
                                       stream_);
            } else {
                const size_t rowBytes = (static_cast<size_t>(input.width) + 1) / 2 * 4;
                ThrowIfCudaFailed(cudaMemcpy2DAsync(deviceRawPlane1_, rawPlane1Pitch_,
                                                    input.hostPixels, input.hostStrideBytes,
                                                    rowBytes, input.height,
                                                    cudaMemcpyHostToDevice, stream_),
                                  "cudaMemcpy2DAsync YUY2 failed");
                LaunchYuy2ToBgraLinear(convertTarget, convertPitch,
                                       deviceRawPlane1_, rawPlane1Pitch_,
                                       static_cast<int>(input.width), static_cast<int>(input.height),
                                       stream_);
            }

            if (turns != 0) {
                LaunchRotateQuarterLinear(deviceBufferA_, devicePitchA_,
                                          devicePreRotate_, devicePitchPreRotate_,
                                          static_cast<int>(input.width), static_cast<int>(input.height),
                                          turns, stream_);
            }
        }

        uchar4* current = deviceBufferA_;
        uchar4* alternate = deviceBufferB_;
        size_t currentPitch = devicePitchA_;
        size_t alternatePitch = devicePitchB_;

        auto swapBuffers = [&]() {
            std::swap(current, alternate);
            std::swap(currentPitch, alternatePitch);
        };

        // Stabilization runs first so every later stage sees the jitter-corrected
        // image. Projections, motion state and the warp correction all stay in
        // device memory; nothing is read back to the host.
        if (settings.enableStabilization) {
            if (!EnsureStabilizationBuffers(procWidth, procHeight)) {
                return false;
            }

            LaunchStabilizationLumaDownsample(deviceStabLuma_,
                                              static_cast<int>(stabSmallWidth_), static_cast<int>(stabSmallHeight_),
                                              current, currentPitch,
                                              static_cast<int>(procWidth), static_cast<int>(procHeight),
                                              static_cast<int>(stabFactorX_), static_cast<int>(stabFactorY_),
                                              stream_);
            LaunchStabilizationProjections(deviceStabLuma_,
                                           static_cast<int>(stabSmallWidth_), static_cast<int>(stabSmallHeight_),
                                           deviceStabColProjCurr_, deviceStabRowProjCurr_,
                                           stream_);

            const float strength = std::clamp(settings.stabilizationStrength, 0.0f, 0.98f);
            LaunchStabilizationEstimate(deviceStabColProjCurr_, deviceStabRowProjCurr_,
                                        deviceStabColProjPrev_, deviceStabRowProjPrev_,
                                        static_cast<int>(stabSmallWidth_), static_cast<int>(stabSmallHeight_),
                                        static_cast<float>(stabFactorX_), static_cast<float>(stabFactorY_),
                                        static_cast<int>(procWidth), static_cast<int>(procHeight),
                                        strength,
                                        stabPrevValid_,
                                        deviceStabState_,
                                        stream_);
            stabPrevValid_ = true;

            LaunchStabilizationWarp(alternate, alternatePitch,
                                    current, currentPitch,
                                    static_cast<int>(procWidth), static_cast<int>(procHeight),
                                    deviceStabState_,
                                    stream_);
            swapBuffers();
        } else {
            stabPrevValid_ = false;
        }

        // Keystone follows stabilization (so the detected quad is jitter-free)
        // and precedes every readability stage, letting the straightened slide
        // fill the frame before black/white, sharpening and auto contrast.
        if (settings.enableKeystone) {
            if (!EnsureKeystoneResources(procWidth, procHeight)) {
                return false;
            }
            RunKeystoneStage(current, alternate, currentPitch, alternatePitch);
        } else if (keystoneFullWidth_ != 0) {
            // Stage is off, so an eased return would be invisible anyway: drop
            // any pending snapshot and re-enable cleanly from identity.
            ResetKeystone();
        }

        if (settings.enableBlackWhite) {
            LaunchBlackWhiteLinear(alternate, alternatePitch, current, currentPitch,
                                   static_cast<int>(procWidth), static_cast<int>(procHeight),
                                   settings.blackWhiteThreshold, stream_);
            swapBuffers();
        }

        if (settings.enableSpatialSharpen) {
            if (settings.spatialUpscaler == SpatialUpscaler::kNis) {
                LaunchNisLinear(alternate, alternatePitch,
                                current, currentPitch,
                                static_cast<int>(procWidth), static_cast<int>(procHeight),
                                static_cast<int>(procWidth), static_cast<int>(procHeight),
                                settings.spatialSharpness,
                                stream_);
            } else {
                LaunchFsrEasuRcasLinear(alternate, alternatePitch,
                                        current, currentPitch,
                                        static_cast<int>(procWidth), static_cast<int>(procHeight),
                                        static_cast<int>(procWidth), static_cast<int>(procHeight),
                                        settings.spatialSharpness,
                                        stream_);
            }
            swapBuffers();
        }

        if (settings.enableZoom) {
            LaunchZoomLinear(alternate, alternatePitch, current, currentPitch,
                             static_cast<int>(procWidth), static_cast<int>(procHeight),
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
                                     static_cast<int>(procWidth), static_cast<int>(procHeight),
                                     stream_);
            swapBuffers();
        }

        if (settings.enableTemporalSmoothing && resolvedFormat == CudaBufferFormat::kRgba8) {
            if (!EnsureTemporalHistory(procWidth, procHeight)) {
                return false;
            }

            const float alpha = std::clamp(settings.temporalSmoothingAlpha, 0.0f, 1.0f);
            LaunchTemporalSmoothLinear(alternate, alternatePitch,
                                       current, currentPitch,
                                       deviceTemporalHistory_, devicePitchHistory_,
                                       static_cast<int>(procWidth), static_cast<int>(procHeight),
                                       alpha,
                                       temporalHistoryValid_,
                                       stream_);
            temporalHistoryValid_ = true;
            swapBuffers();
        } else {
            temporalHistoryValid_ = false;
        }

        // Auto-contrast measurement feeds the grade kernel below within the
        // same frame; both the histogram and the smoothed lo/hi levels live in
        // device memory, so there is no readback and no stall. Runs on the
        // post-keystone image so the straightened slide dominates the
        // statistics.
        const bool autoContrastActive =
            settings.enableAutoContrast && settings.autoContrastStrength > 0.0f;
        if (autoContrastActive) {
            if (!EnsureAutoContrastBuffers()) {
                return false;
            }
            LaunchAutoContrastHistogram(deviceHistogram_,
                                        current, currentPitch,
                                        static_cast<int>(procWidth), static_cast<int>(procHeight),
                                        stream_);
            LaunchAutoContrastAnalysis(deviceHistogram_,
                                       static_cast<int>(procWidth * procHeight),
                                       deviceAutoLevels_,
                                       autoLevelsValid_,
                                       stream_);
            autoLevelsValid_ = true;
        } else {
            autoLevelsValid_ = false;
        }

        if (settings.displayColorMode != 0 ||
            settings.contrast != 1.0f ||
            settings.brightness != 0.0f ||
            autoContrastActive) {
            const float contrast = std::clamp(settings.contrast, 0.25f, 4.0f);
            const float brightness = std::clamp(settings.brightness, -1.0f, 1.0f);
            // In place: the kernel touches only its own pixel, so no swap needed.
            LaunchDisplayColorGradeLinear(current, currentPitch,
                                          static_cast<int>(procWidth), static_cast<int>(procHeight),
                                          settings.displayColorMode, contrast, brightness,
                                          autoContrastActive ? deviceAutoLevels_ : nullptr,
                                          std::clamp(settings.autoContrastStrength, 0.0f, 1.0f),
                                          stream_);
        }

        if (settings.drawFocusMarker) {
            LaunchFocusMarkerLinear(current, currentPitch,
                                    static_cast<int>(procWidth), static_cast<int>(procHeight),
                                    settings.zoomCenterX, settings.zoomCenterY, stream_);
        }

        ThrowIfCudaFailed(cudaMemcpy2DToArrayAsync(level0Array_, 0, 0,
                                                   current, currentPitch,
                                                   static_cast<size_t>(procWidth) * sizeof(uchar4), procHeight,
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
        stabPrevValid_ = false;
        keystoneCopyPending_ = false;
        autoLevelsValid_ = false;
        return false;
    } catch (...) {
        lastError_ = "Unknown CUDA exception during ProcessFrame";
        qWarning() << "ProcessFrame encountered unknown exception";
        temporalHistoryValid_ = false;
        stabPrevValid_ = false;
        keystoneCopyPending_ = false;
        autoLevelsValid_ = false;
        return false;
    }
}

} // namespace openzoom

#endif // OPENZOOM_HAS_CUDA_EXT_MEMORY

#endif // _WIN32
