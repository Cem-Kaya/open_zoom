#ifdef _WIN32

#include "openzoom/app.hpp"
#include "openzoom/cuda_interop.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QKeyEvent>
#include <QPainter>
#include <QMouseEvent>
#include <QCursor>
#include <QEvent>
#include <QResizeEvent>
#include <QRegion>
#include <QWheelEvent>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QString>
#include <QSizePolicy>
#include <QPaintEngine>
#include <QResizeEvent>
#include <QShowEvent>
#include <QDebug>
#include <QMessageBox>

#include <windows.h>
#include <combaseapi.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <shlwapi.h>

#include <cwchar>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <wrl/client.h>

namespace openzoom {

namespace {

constexpr int kZoomSliderScale = 100;
constexpr int kZoomSliderMaxMultiplier = 12;
constexpr int kZoomFocusSliderScale = 100;
constexpr float kPanKeyboardStep = 0.01f;
constexpr float kPanJoystickStep = 0.008f;
constexpr int kBlurSigmaSliderMin = 1;
constexpr int kBlurSigmaSliderMax = 50;
constexpr float kBlurSigmaStep = 0.1f;
constexpr int kBlurRadiusMin = 1;
constexpr int kBlurRadiusMax = 15;

float SliderValueToSigma(int sliderValue) {
    int clamped = std::clamp(sliderValue, kBlurSigmaSliderMin, kBlurSigmaSliderMax);
    return static_cast<float>(clamped) * kBlurSigmaStep;
}

int NormalizeRadiusValue(int sliderValue) {
    int radius = std::clamp(sliderValue, kBlurRadiusMin, kBlurRadiusMax);
    if ((radius & 1) == 0) {
        if (radius >= kBlurRadiusMax) {
            radius -= 1;
        } else {
            radius += 1;
        }
    }
    return radius;
}

CudaBufferFormat ParseCudaBufferFormatToken(const QString& token, bool* ok)
{
    const QString normalized = token.trimmed().toLower();
    if (normalized == QStringLiteral("rgba8") ||
        normalized == QStringLiteral("bgra8") ||
        normalized == QStringLiteral("uint8") ||
        normalized == QStringLiteral("8bit")) {
        if (ok) {
            *ok = true;
        }
        return CudaBufferFormat::kRgba8;
    }

    if (normalized == QStringLiteral("rgba16f") ||
        normalized == QStringLiteral("fp16") ||
        normalized == QStringLiteral("half") ||
        normalized == QStringLiteral("16f")) {
        if (ok) {
            *ok = true;
        }
        return CudaBufferFormat::kRgba16F;
    }

    if (ok) {
        *ok = false;
    }
    return CudaBufferFormat::kRgba8;
}

void ApplyGaussianBlur(const std::vector<uint8_t>& src,
                       std::vector<uint8_t>& scratch,
                       std::vector<uint8_t>& dst,
                       UINT width,
                       UINT height,
                       int radius,
                       float sigma) {
    if (radius <= 0 || sigma <= 0.0f || src.empty() || width == 0 || height == 0) {
        dst = src;
        return;
    }

    const size_t pixelCount = static_cast<size_t>(width) * height;
    scratch.resize(pixelCount * 4);
    dst.resize(pixelCount * 4);

    const int kernelSize = radius * 2 + 1;
    std::vector<float> kernel(static_cast<size_t>(kernelSize));
    const float sigma2 = sigma * sigma;
    const float denom = 2.0f * sigma2;
    float weightSum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float x = static_cast<float>(i);
        const float weight = std::exp(-(x * x) / denom);
        kernel[static_cast<size_t>(i + radius)] = weight;
        weightSum += weight;
    }
    if (weightSum <= 0.0f) {
        dst = src;
        return;
    }
    for (float& weight : kernel) {
        weight /= weightSum;
    }

    auto samplePixel = [&](const std::vector<uint8_t>& buffer, UINT x, UINT y) -> const uint8_t* {
        return buffer.data() + (static_cast<size_t>(y) * width + x) * 4;
    };

    auto writePixel = [](std::vector<uint8_t>& buffer, UINT width, UINT x, UINT y,
                         float b, float g, float r, float a) {
        uint8_t* dstPixel = buffer.data() + (static_cast<size_t>(y) * width + x) * 4;
        dstPixel[0] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(b)), 0, 255));
        dstPixel[1] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(g)), 0, 255));
        dstPixel[2] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(r)), 0, 255));
        dstPixel[3] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(a)), 0, 255));
    };

    // Horizontal pass
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            float accumB = 0.0f;
            float accumG = 0.0f;
            float accumR = 0.0f;
            float accumA = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sampleX = static_cast<int>(x) + k;
                sampleX = std::clamp(sampleX, 0, static_cast<int>(width) - 1);
                const uint8_t* srcPixel = samplePixel(src, static_cast<UINT>(sampleX), y);
                const float weight = kernel[static_cast<size_t>(k + radius)];
                accumB += weight * srcPixel[0];
                accumG += weight * srcPixel[1];
                accumR += weight * srcPixel[2];
                accumA += weight * srcPixel[3];
            }
            writePixel(scratch, width, x, y, accumB, accumG, accumR, accumA);
        }
    }

    // Vertical pass
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            float accumB = 0.0f;
            float accumG = 0.0f;
            float accumR = 0.0f;
            float accumA = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sampleY = static_cast<int>(y) + k;
                sampleY = std::clamp(sampleY, 0, static_cast<int>(height) - 1);
                const uint8_t* srcPixel = samplePixel(scratch, x, static_cast<UINT>(sampleY));
                const float weight = kernel[static_cast<size_t>(k + radius)];
                accumB += weight * srcPixel[0];
                accumG += weight * srcPixel[1];
                accumR += weight * srcPixel[2];
                accumA += weight * srcPixel[3];
            }
            writePixel(dst, width, x, y, accumB, accumG, accumR, accumA);
        }
    }
}


void ThrowIfFailed(HRESULT hr, const char* message) {
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(message) + " (hr=0x" + std::to_string(static_cast<unsigned long>(hr)) + ")");
    }
}

int ClampToByte(int value) {
    if (value < 0) {
        return 0;
    }
    if (value > 255) {
        return 255;
    }
    return value;
}

bool ConvertNv12ToBgra(const uint8_t* src,
                       size_t srcSize,
                       UINT strideY,
                       UINT width,
                       UINT height,
                       std::vector<uint8_t>& dst)
{
    if (!src || width == 0 || height == 0) {
        return false;
    }

    if (strideY == 0) {
        strideY = width;
    }

    const size_t yPlaneSize = static_cast<size_t>(strideY) * height;
    const size_t uvStride = strideY;
    const size_t uvPlaneSize = uvStride * ((height + 1) / 2);

    if (srcSize < yPlaneSize + uvPlaneSize) {
        return false;
    }

    dst.resize(static_cast<size_t>(width) * height * 4);

    const uint8_t* yPlane = src;
    const uint8_t* uvPlane = src + yPlaneSize;

    for (UINT y = 0; y < height; ++y) {
        const uint8_t* yRow = yPlane + static_cast<size_t>(y) * strideY;
        const uint8_t* uvRow = uvPlane + static_cast<size_t>(y / 2) * uvStride;
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * width * 4;

        for (UINT x = 0; x < width; ++x) {
            const int yValue = yRow[x];
            const int uvIndex = (x / 2) * 2;
            const int uValue = uvRow[uvIndex] - 128;
            const int vValue = uvRow[uvIndex + 1] - 128;

            int c = yValue - 16;
            if (c < 0) {
                c = 0;
            }

            const int d = uValue;
            const int e = vValue;

            const int r = (298 * c + 409 * e + 128) >> 8;
            const int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            const int b = (298 * c + 516 * d + 128) >> 8;

            const UINT dstIndex = x * 4;
            dstRow[dstIndex + 0] = static_cast<uint8_t>(ClampToByte(b));
            dstRow[dstIndex + 1] = static_cast<uint8_t>(ClampToByte(g));
            dstRow[dstIndex + 2] = static_cast<uint8_t>(ClampToByte(r));
            dstRow[dstIndex + 3] = 255;
        }
    }

    return true;
}

bool ConvertYuy2ToBgra(const uint8_t* src,
                       size_t srcSize,
                       UINT stride,
                       UINT width,
                       UINT height,
                       std::vector<uint8_t>& dst)
{
    if (!src || width == 0 || height == 0) {
        return false;
    }

    if (stride == 0) {
        stride = width * 2;
    }

    const size_t required = static_cast<size_t>(stride) * height;
    if (srcSize < required) {
        return false;
    }

    dst.resize(static_cast<size_t>(width) * height * 4);

    for (UINT y = 0; y < height; ++y) {
        const uint8_t* srcRow = src + static_cast<size_t>(y) * stride;
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * width * 4;

        for (UINT x = 0; x < width; x += 2) {
            const UINT srcIndex = x * 2;
            if (srcIndex + 3 >= stride) {
                break;
            }

            const int y0 = srcRow[srcIndex + 0];
            const int u = srcRow[srcIndex + 1] - 128;
            const int y1 = srcRow[srcIndex + 2];
            const int v = srcRow[srcIndex + 3] - 128;

            const int c0 = y0 - 16;
            const int c1 = y1 - 16;

            const int r0 = (298 * c0 + 409 * v + 128) >> 8;
            const int g0 = (298 * c0 - 100 * u - 208 * v + 128) >> 8;
            const int b0 = (298 * c0 + 516 * u + 128) >> 8;

            const int r1 = (298 * c1 + 409 * v + 128) >> 8;
            const int g1 = (298 * c1 - 100 * u - 208 * v + 128) >> 8;
            const int b1 = (298 * c1 + 516 * u + 128) >> 8;

            const UINT dstIndex0 = x * 4;
            dstRow[dstIndex0 + 0] = static_cast<uint8_t>(ClampToByte(b0));
            dstRow[dstIndex0 + 1] = static_cast<uint8_t>(ClampToByte(g0));
            dstRow[dstIndex0 + 2] = static_cast<uint8_t>(ClampToByte(r0));
            dstRow[dstIndex0 + 3] = 255;

            dstRow[dstIndex0 + 4] = static_cast<uint8_t>(ClampToByte(b1));
            dstRow[dstIndex0 + 5] = static_cast<uint8_t>(ClampToByte(g1));
            dstRow[dstIndex0 + 6] = static_cast<uint8_t>(ClampToByte(r1));
            dstRow[dstIndex0 + 7] = 255;
        }
    }

    return true;
}

void CopyArgbToBgra(const uint8_t* src,
                    UINT srcStride,
                    UINT width,
                    UINT height,
                    std::vector<uint8_t>& dst)
{
    const UINT dstStride = width * 4;
    dst.resize(static_cast<size_t>(dstStride) * height);
    for (UINT y = 0; y < height; ++y) {
        const uint8_t* srcRow = src + static_cast<size_t>(y) * srcStride;
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * dstStride;
        std::memcpy(dstRow, srcRow, dstStride);
    }
}

void CopyRgbxToBgra(const uint8_t* src,
                    UINT srcStride,
                    UINT width,
                    UINT height,
                    std::vector<uint8_t>& dst)
{
    const UINT dstStride = width * 4;
    dst.resize(static_cast<size_t>(dstStride) * height);
    for (UINT y = 0; y < height; ++y) {
        const uint8_t* srcRow = src + static_cast<size_t>(y) * srcStride;
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * dstStride;
        for (UINT x = 0; x < width; ++x) {
            dstRow[x * 4 + 0] = srcRow[x * 4 + 0];
            dstRow[x * 4 + 1] = srcRow[x * 4 + 1];
            dstRow[x * 4 + 2] = srcRow[x * 4 + 2];
            dstRow[x * 4 + 3] = 255;
        }
    }
}

void ApplyBlackWhite(const std::vector<uint8_t>& src,
                     std::vector<uint8_t>& dst,
                     float threshold)
{
    if (dst.size() != src.size()) {
        dst.resize(src.size());
    }
    const float thresh = std::clamp(threshold, 0.0f, 1.0f);
    for (size_t i = 0; i < src.size(); i += 4) {
        const float r = src[i + 2] / 255.0f;
        const float g = src[i + 1] / 255.0f;
        const float b = src[i + 0] / 255.0f;
        const float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
        const uint8_t value = luminance >= thresh ? 255 : 0;
        dst[i + 0] = value;
        dst[i + 1] = value;
        dst[i + 2] = value;
        dst[i + 3] = src[i + 3];
    }
}

void ApplyZoom(const std::vector<uint8_t>& src,
               std::vector<uint8_t>& dst,
               UINT width,
               UINT height,
               float zoomAmount,
               float centerXNormalized,
               float centerYNormalized)
{
    const float zoom = std::max(1.0f, zoomAmount);
    const UINT stride = width * 4;
    if (dst.size() != src.size()) {
        dst.resize(src.size());
    }

    const float maxIndexX = static_cast<float>(width > 0 ? width - 1 : 0);
    const float maxIndexY = static_cast<float>(height > 0 ? height - 1 : 0);
    const float outputCenterX = maxIndexX * 0.5f;
    const float outputCenterY = maxIndexY * 0.5f;

    float centerX = std::clamp(centerXNormalized, 0.0f, 1.0f) * maxIndexX;
    float centerY = std::clamp(centerYNormalized, 0.0f, 1.0f) * maxIndexY;

    const float halfVisibleWidth = (static_cast<float>(width)) / (zoom * 2.0f);
    const float halfVisibleHeight = (static_cast<float>(height)) / (zoom * 2.0f);

    if (width > 1) {
        const float minCenterX = std::max(0.0f, halfVisibleWidth - 0.5f);
        const float maxCenterX = std::min(maxIndexX, static_cast<float>(width) - 1.0f - (halfVisibleWidth - 0.5f));
        if (minCenterX <= maxCenterX) {
            centerX = std::clamp(centerX, minCenterX, maxCenterX);
        }
    }

    if (height > 1) {
        const float minCenterY = std::max(0.0f, halfVisibleHeight - 0.5f);
        const float maxCenterY = std::min(maxIndexY, static_cast<float>(height) - 1.0f - (halfVisibleHeight - 0.5f));
        if (minCenterY <= maxCenterY) {
            centerY = std::clamp(centerY, minCenterY, maxCenterY);
        }
    }

    for (UINT y = 0; y < height; ++y) {
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * stride;
        for (UINT x = 0; x < width; ++x) {
            const float sx = (static_cast<float>(x) - outputCenterX) / zoom + centerX;
            const float sy = (static_cast<float>(y) - outputCenterY) / zoom + centerY;

            int sampleX = static_cast<int>(std::lroundf(sx));
            int sampleY = static_cast<int>(std::lroundf(sy));
            sampleX = std::clamp(sampleX, 0, static_cast<int>(width) - 1);
            sampleY = std::clamp(sampleY, 0, static_cast<int>(height) - 1);

            const uint8_t* srcPixel = src.data() + static_cast<size_t>(sampleY) * stride + sampleX * 4;
            uint8_t* dstPixel = dstRow + x * 4;
            dstPixel[0] = srcPixel[0];
            dstPixel[1] = srcPixel[1];
            dstPixel[2] = srcPixel[2];
            dstPixel[3] = srcPixel[3];
        }
    }
}

D3D12_RESOURCE_BARRIER TransitionBarrier(ID3D12Resource* resource,
                                         D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return barrier;
}

} // namespace

class D3D12Presenter {
public:
    ~D3D12Presenter() {
        if (uploadBuffer_) {
            uploadBuffer_->Unmap(0, nullptr);
            uploadBuffer_.Reset();
            uploadMappedPtr_ = nullptr;
        }
        if (fenceEvent_) {
            CloseHandle(fenceEvent_);
            fenceEvent_ = nullptr;
        }
    }

    void Initialize(HWND hwnd, UINT width, UINT height) {
        hwnd_ = hwnd;
        CreateDevice();
        CreateCommandObjects();
        CreateFenceObjects();
        CreateSwapChain(width, height);
        EnsureUploadBuffer(width, height);
        initialized_ = true;
    }

    bool IsInitialized() const {
        return initialized_;
    }

    void Resize(UINT width, UINT height) {
        if (!initialized_ || width == 0 || height == 0) {
            return;
        }

        WaitForGpu();
        for (auto& buffer : backBuffers_) {
            buffer.Reset();
        }

        ThrowIfFailed(swapChain_->ResizeBuffers(static_cast<UINT>(backBuffers_.size()),
                                                width,
                                                height,
                                                DXGI_FORMAT_B8G8R8A8_UNORM,
                                                0),
                      "Failed to resize swap chain buffers");

        AcquireBackBuffers();
        width_ = width;
        height_ = height;
        EnsureUploadBuffer(width, height);
    }

    void Present(const uint8_t* data, UINT width, UINT height) {
        if (!initialized_ || !data) {
            return;
        }

        if (width == 0 || height == 0) {
            return;
        }

        if (width != width_ || height != height_) {
            Resize(width, height);
        }

        EnsureUploadBuffer(width, height);
        CopyToUpload(data, width, height);

        ThrowIfFailed(commandAllocator_->Reset(), "Failed to reset command allocator");
        ThrowIfFailed(commandList_->Reset(commandAllocator_.Get(), nullptr), "Failed to reset command list");

        const UINT backIndex = swapChain_->GetCurrentBackBufferIndex();
        Microsoft::WRL::ComPtr<ID3D12Resource> backBuffer = backBuffers_[backIndex];

        auto barrierToCopy = TransitionBarrier(backBuffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        commandList_->ResourceBarrier(1, &barrierToCopy);

        D3D12_TEXTURE_COPY_LOCATION dest{};
        dest.pResource = backBuffer.Get();
        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dest.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = uploadBuffer_.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = uploadFootprint_;

        commandList_->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);

        auto barrierToPresent = TransitionBarrier(backBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        commandList_->ResourceBarrier(1, &barrierToPresent);

        ThrowIfFailed(commandList_->Close(), "Failed to close command list");

        ID3D12CommandList* lists[] = { commandList_.Get() };
        commandQueue_->ExecuteCommandLists(static_cast<UINT>(std::size(lists)), lists);

        ThrowIfFailed(swapChain_->Present(1, 0), "Failed to present swap chain");

        WaitForGpu();
    }

    void PresentFromTexture(ID3D12Resource* texture,
                            UINT width,
                            UINT height,
                            const FenceSyncParams* fenceSync = nullptr) {
        if (!initialized_ || !texture || width == 0 || height == 0) {
            return;
        }

        if (width != width_ || height != height_) {
            Resize(width, height);
        }

        const bool useFenceSync = fenceSync && fenceSync->enable && fence_.Get() != nullptr;

        ThrowIfFailed(commandAllocator_->Reset(), "Failed to reset command allocator");
        ThrowIfFailed(commandList_->Reset(commandAllocator_.Get(), nullptr), "Failed to reset command list");

        const UINT backIndex = swapChain_->GetCurrentBackBufferIndex();
        Microsoft::WRL::ComPtr<ID3D12Resource> backBuffer = backBuffers_[backIndex];

        if (useFenceSync && fenceSync->waitValue > 0) {
            ThrowIfFailed(commandQueue_->Wait(fence_.Get(), fenceSync->waitValue),
                          "Failed to queue wait on shared fence");
        }

        auto transitionSourceToCopy = TransitionBarrier(texture,
                                                        D3D12_RESOURCE_STATE_COMMON,
                                                        D3D12_RESOURCE_STATE_COPY_SOURCE);
        commandList_->ResourceBarrier(1, &transitionSourceToCopy);

        auto transitionDestToCopy = TransitionBarrier(backBuffer.Get(),
                                                      D3D12_RESOURCE_STATE_PRESENT,
                                                      D3D12_RESOURCE_STATE_COPY_DEST);
        commandList_->ResourceBarrier(1, &transitionDestToCopy);

        commandList_->CopyResource(backBuffer.Get(), texture);

        auto transitionDestToPresent = TransitionBarrier(backBuffer.Get(),
                                                         D3D12_RESOURCE_STATE_COPY_DEST,
                                                         D3D12_RESOURCE_STATE_PRESENT);
        commandList_->ResourceBarrier(1, &transitionDestToPresent);

        auto transitionSourceToCommon = TransitionBarrier(texture,
                                                           D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                           D3D12_RESOURCE_STATE_COMMON);
        commandList_->ResourceBarrier(1, &transitionSourceToCommon);

        ThrowIfFailed(commandList_->Close(), "Failed to close command list");

        ID3D12CommandList* lists[] = { commandList_.Get() };
        commandQueue_->ExecuteCommandLists(static_cast<UINT>(std::size(lists)), lists);

        ThrowIfFailed(swapChain_->Present(1, 0), "Failed to present swap chain");

        if (useFenceSync) {
            ThrowIfFailed(commandQueue_->Signal(fence_.Get(), fenceSync->signalValue),
                          "Failed to signal shared fence");
            fenceValue_ = fenceSync->signalValue;
            WaitForFenceValue(fenceSync->signalValue);
        } else {
            WaitForGpu();
        }
    }

    ID3D12Device* GetDevice() const {
        return device_.Get();
    }

    ID3D12Fence* GetFence() const {
        return fence_.Get();
    }

    UINT64 GetLastSignaledFenceValue() const {
        return fenceValue_;
    }

private:
    void CreateDevice() {
        UINT dxgiFactoryFlags = 0;
#if defined(_DEBUG)
        Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
#endif
        Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
        ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)), "Failed to create DXGI factory");
        factory_ = factory;

        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;

        HRESULT hr = factory->EnumAdapterByGpuPreference(0,
                                                         DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                         IID_PPV_ARGS(&adapter));
        if (FAILED(hr) || !adapter) {
            for (UINT adapterIndex = 0;
                 factory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND;
                 ++adapterIndex) {
                DXGI_ADAPTER_DESC1 desc{};
                adapter->GetDesc1(&desc);
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                    continue;
                }
                if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                                _uuidof(ID3D12Device), nullptr))) {
                    qInfo() << "Selected DXGI adapter" << QString::fromWCharArray(desc.Description);
                    break;
                }
            }
        } else {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                adapter.Reset();
            } else {
                qInfo() << "Selected high-performance DXGI adapter" << QString::fromWCharArray(desc.Description);
            }
        }

        if (!adapter) {
            ThrowIfFailed(E_FAIL, "Failed to locate hardware adapter for D3D12");
        }

        ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)),
                      "Failed to create D3D12 device");
    }

    void CreateCommandObjects() {
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        ThrowIfFailed(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue_)),
                      "Failed to create D3D12 command queue");

        ThrowIfFailed(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                      IID_PPV_ARGS(&commandAllocator_)),
                      "Failed to create command allocator");

        ThrowIfFailed(device_->CreateCommandList(0,
                                                 D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 commandAllocator_.Get(),
                                                 nullptr,
                                                 IID_PPV_ARGS(&commandList_)),
                      "Failed to create command list");
        commandList_->Close();
    }

    void CreateFenceObjects() {
        ThrowIfFailed(device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence_)),
                      "Failed to create fence");
        fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!fenceEvent_) {
            ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()), "Failed to create fence event");
        }
    }

    void CreateSwapChain(UINT width, UINT height) {
        DXGI_SWAP_CHAIN_DESC1 scDesc{};
        scDesc.Width = width;
        scDesc.Height = height;
        scDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        scDesc.Stereo = FALSE;
        scDesc.SampleDesc.Count = 1;
        scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scDesc.BufferCount = 2;
        scDesc.Scaling = DXGI_SCALING_STRETCH;
        scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        scDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

        Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
        ThrowIfFailed(factory_->CreateSwapChainForHwnd(commandQueue_.Get(),
                                                       hwnd_,
                                                       &scDesc,
                                                       nullptr,
                                                       nullptr,
                                                       &swapChain1),
                      "Failed to create swap chain");

        ThrowIfFailed(factory_->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER),
                      "Failed to disable Alt+Enter");

        ThrowIfFailed(swapChain1.As(&swapChain_), "Failed to query IDXGISwapChain3");

        backBuffers_.resize(scDesc.BufferCount);
        AcquireBackBuffers();
        width_ = width;
        height_ = height;
    }

    void AcquireBackBuffers() {
        for (UINT i = 0; i < backBuffers_.size(); ++i) {
            ThrowIfFailed(swapChain_->GetBuffer(i, IID_PPV_ARGS(&backBuffers_[i])),
                          "Failed to acquire swap chain buffer");
        }
    }

    void EnsureUploadBuffer(UINT width, UINT height) {
        if (uploadBuffer_ && uploadWidth_ == width && uploadHeight_ == height) {
            return;
        }

        if (uploadBuffer_) {
            uploadBuffer_->Unmap(0, nullptr);
            uploadBuffer_.Reset();
            uploadMappedPtr_ = nullptr;
        }

        D3D12_RESOURCE_DESC textureDesc{};
        textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDesc.Alignment = 0;
        textureDesc.Width = width;
        textureDesc.Height = height;
        textureDesc.DepthOrArraySize = 1;
        textureDesc.MipLevels = 1;
        textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        device_->GetCopyableFootprints(&textureDesc,
                                       0,
                                       1,
                                       0,
                                       &uploadFootprint_,
                                       &uploadNumRows_,
                                       &uploadRowSizeInBytes_,
                                       &uploadTotalBytes_);

        D3D12_RESOURCE_DESC bufferDesc{};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Alignment = 0;
        bufferDesc.Width = uploadTotalBytes_;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

        ThrowIfFailed(device_->CreateCommittedResource(&heapProps,
                                                       D3D12_HEAP_FLAG_NONE,
                                                       &bufferDesc,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ,
                                                       nullptr,
                                                       IID_PPV_ARGS(&uploadBuffer_)),
                      "Failed to create upload buffer");

        ThrowIfFailed(uploadBuffer_->Map(0, nullptr, reinterpret_cast<void**>(&uploadMappedPtr_)),
                      "Failed to map upload buffer");

        uploadWidth_ = width;
        uploadHeight_ = height;
    }

    void CopyToUpload(const uint8_t* data, UINT width, UINT height) {
        const UINT rowPitch = uploadFootprint_.Footprint.RowPitch;
        const UINT srcPitch = width * 4;
        for (UINT row = 0; row < height; ++row) {
            std::memcpy(uploadMappedPtr_ + static_cast<size_t>(row) * rowPitch,
                        data + static_cast<size_t>(row) * srcPitch,
                        srcPitch);
            if (rowPitch > srcPitch) {
                std::memset(uploadMappedPtr_ + static_cast<size_t>(row) * rowPitch + srcPitch,
                            0,
                            rowPitch - srcPitch);
            }
        }
    }

    void WaitForGpu() {
        const UINT64 fenceValue = ++fenceValue_;
        ThrowIfFailed(commandQueue_->Signal(fence_.Get(), fenceValue), "Failed to signal fence");
        if (fence_->GetCompletedValue() < fenceValue) {
            ThrowIfFailed(fence_->SetEventOnCompletion(fenceValue, fenceEvent_),
                          "Failed to set fence completion event");
            WaitForSingleObject(fenceEvent_, INFINITE);
        }
    }

    void WaitForFenceValue(UINT64 value) {
        if (!fence_) {
            return;
        }
        if (fence_->GetCompletedValue() < value) {
            ThrowIfFailed(fence_->SetEventOnCompletion(value, fenceEvent_),
                          "Failed to set fence completion event");
            WaitForSingleObject(fenceEvent_, INFINITE);
        }
    }

    HWND hwnd_{};
    UINT width_{};
    UINT height_{};
    bool initialized_{};

    Microsoft::WRL::ComPtr<IDXGIFactory6> factory_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain_;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> backBuffers_;

    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    UINT64 fenceValue_{};
    HANDLE fenceEvent_{nullptr};

    Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT uploadFootprint_{};
    UINT uploadNumRows_{};
    UINT64 uploadRowSizeInBytes_{};
    UINT64 uploadTotalBytes_{};
    uint8_t* uploadMappedPtr_{};
    UINT uploadWidth_{};
    UINT uploadHeight_{};
};

class RenderWidget : public QWidget {
public:
    explicit RenderWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setAttribute(Qt::WA_NativeWindow);
        setAttribute(Qt::WA_PaintOnScreen);
        setAttribute(Qt::WA_NoSystemBackground);
        setFocusPolicy(Qt::StrongFocus);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }

    QPaintEngine* paintEngine() const override {
        return nullptr;
    }

    void setPresenter(D3D12Presenter* presenter) {
        presenter_ = presenter;
    }

    bool isPresenterReady() const {
        return presenter_ && presenter_->IsInitialized();
    }

protected:
    void showEvent(QShowEvent* event) override {
        QWidget::showEvent(event);
        EnsurePresenter();
    }

    void resizeEvent(QResizeEvent* event) override {
        QWidget::resizeEvent(event);
        if (EnsurePresenter()) {
            presenter_->Resize(std::max(1, width()), std::max(1, height()));
        }
    }

private:
    bool EnsurePresenter() {
        if (!presenter_ || presenter_->IsInitialized()) {
            return presenter_ != nullptr;
        }

        HWND hwnd = reinterpret_cast<HWND>(winId());
        presenter_->Initialize(hwnd, std::max(1, width()), std::max(1, height()));
        return true;
    }

    D3D12Presenter* presenter_{};
};

JoystickOverlay::JoystickOverlay(QWidget* parent)
    : QWidget(parent) {
    setAttribute(Qt::WA_TransparentForMouseEvents, false);
    setAttribute(Qt::WA_NoSystemBackground, true);
    setAttribute(Qt::WA_TranslucentBackground, true);
    setVisible(false);
    if (parent) {
        parent->installEventFilter(this);
    }
    setFixedSize(160, 160);
    UpdateMask();
    ResetKnob();
}

void JoystickOverlay::ResetKnob() {
    knobPos_ = QPointF(width() / 2.0, height() / 2.0);
    update();
}

bool JoystickOverlay::eventFilter(QObject* watched, QEvent* event) {
    if (watched == parentWidget()) {
        if (event->type() == QEvent::Resize) {
            UpdatePlacement();
        }
    }
    return QWidget::eventFilter(watched, event);
}

void JoystickOverlay::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    UpdatePlacement();
}

void JoystickOverlay::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.setCompositionMode(QPainter::CompositionMode_Source);
    painter.fillRect(rect(), Qt::transparent);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

    painter.setBrush(QColor(60, 60, 60, 180));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(rect());

    painter.setBrush(QColor(230, 230, 230, 230));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(KnobRect());
}

void JoystickOverlay::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        dragging_ = true;
        UpdateFromPosition(event->position());
    }
}

void JoystickOverlay::mouseMoveEvent(QMouseEvent* event) {
    if (dragging_) {
        UpdateFromPosition(event->position());
    }
}

void JoystickOverlay::mouseReleaseEvent(QMouseEvent* event) {
    if (dragging_ && event->button() == Qt::LeftButton) {
        dragging_ = false;
        ResetKnob();
        emit JoystickChanged(0.0f, 0.0f);
        update();
    }
}

void JoystickOverlay::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    UpdateMask();
}

QRectF JoystickOverlay::KnobRect() const {
    constexpr qreal knobRadius = 24.0;
    return QRectF(knobPos_.x() - knobRadius,
                  knobPos_.y() - knobRadius,
                  knobRadius * 2.0,
                  knobRadius * 2.0);
}

void JoystickOverlay::UpdatePlacement() {
    if (!parentWidget()) {
        return;
    }
    const int margin = 20;
    const int x = parentWidget()->width() - width() - margin;
    const int y = parentWidget()->height() - height() - margin;
    move(std::max(0, x), std::max(0, y));
}

void JoystickOverlay::UpdateFromPosition(const QPointF& pos) {
    const QPointF center(width() / 2.0, height() / 2.0);
    QPointF delta = pos - center;
    const qreal maxRadius = width() / 2.0;
    if (delta.manhattanLength() < 0.0001) {
        delta = QPointF(0, 0);
    }
    const qreal distance = std::sqrt(delta.x() * delta.x() + delta.y() * delta.y());
    if (distance > maxRadius) {
        delta *= maxRadius / distance;
    }
    knobPos_ = center + delta;
    update();

    float normX = static_cast<float>(delta.x() / maxRadius);
    float normY = static_cast<float>(delta.y() / maxRadius);
    normX = std::clamp(normX, -1.0f, 1.0f);
    normY = std::clamp(normY, -1.0f, 1.0f);

    emit JoystickChanged(normX, -normY);
}

void JoystickOverlay::UpdateMask() {
    if (width() <= 0 || height() <= 0) {
        clearMask();
        return;
    }
    QRegion region(rect(), QRegion::Ellipse);
    setMask(region);
}

class MainWindow : public QMainWindow {
public:
    MainWindow() {
        setWindowTitle("OpenZoom");
        resize(1280, 720);

        auto* central = new QWidget(this);
        auto* rootLayout = new QVBoxLayout(central);
        rootLayout->setContentsMargins(12, 12, 12, 12);
        rootLayout->setSpacing(8);

        controlsToggleButton_ = new QToolButton();
        controlsToggleButton_->setText("Hide Controls");
        controlsToggleButton_->setCheckable(true);
        controlsToggleButton_->setChecked(true);
        controlsToggleButton_->setArrowType(Qt::DownArrow);
        controlsToggleButton_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        joystickCheckbox_ = new QCheckBox("Virtual Joystick");

        auto* headerLayout = new QHBoxLayout();
        headerLayout->setSpacing(8);
        headerLayout->addWidget(controlsToggleButton_);
        headerLayout->addStretch(1);
        headerLayout->addWidget(joystickCheckbox_);
        rootLayout->addLayout(headerLayout);

        controlsContainer_ = new QWidget();
        auto* controlsLayout = new QVBoxLayout(controlsContainer_);
        controlsLayout->setContentsMargins(0, 0, 0, 0);
        controlsLayout->setSpacing(8);

        auto* cameraLayout = new QHBoxLayout();
        cameraLayout->setSpacing(8);
        auto* cameraLabel = new QLabel("Camera:");
        cameraCombo_ = new QComboBox();
        cameraCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        cameraLayout->addWidget(cameraLabel);
        cameraLayout->addWidget(cameraCombo_, 1);
        controlsLayout->addLayout(cameraLayout);

        auto* bwLayout = new QHBoxLayout();
        bwLayout->setSpacing(8);
        bwCheckbox_ = new QCheckBox("Black && White");
        bwSlider_ = new QSlider(Qt::Horizontal);
        bwSlider_->setRange(0, 255);
        bwSlider_->setPageStep(8);
        bwSlider_->setValue(128);
        bwSlider_->setEnabled(false);
        bwLayout->addWidget(bwCheckbox_);
        bwLayout->addWidget(bwSlider_, 1);
        controlsLayout->addLayout(bwLayout);

        auto* zoomLayout = new QHBoxLayout();
        zoomLayout->setSpacing(8);
        zoomCheckbox_ = new QCheckBox("Zoom");
        zoomSlider_ = new QSlider(Qt::Horizontal);
        zoomSlider_->setRange(kZoomSliderScale, kZoomSliderMaxMultiplier * kZoomSliderScale);
        zoomSlider_->setPageStep(10);
        zoomSlider_->setValue(kZoomSliderScale);
        zoomSlider_->setEnabled(false);
        zoomLayout->addWidget(zoomCheckbox_);
        zoomLayout->addWidget(zoomSlider_, 1);
        controlsLayout->addLayout(zoomLayout);

        auto* blurLayout = new QHBoxLayout();
        blurLayout->setSpacing(8);
        blurCheckbox_ = new QCheckBox("Gaussian Blur");
        blurSigmaSlider_ = new QSlider(Qt::Horizontal);
        blurSigmaSlider_->setRange(1, 50); // 0.1 .. 5.0 sigma
        blurSigmaSlider_->setPageStep(2);
        blurSigmaSlider_->setSingleStep(1);
        blurSigmaSlider_->setValue(10);
        blurSigmaSlider_->setEnabled(false);
        blurSigmaValueLabel_ = new QLabel("1.0");
        blurSigmaValueLabel_->setMinimumWidth(40);

        blurRadiusSlider_ = new QSlider(Qt::Horizontal);
        blurRadiusSlider_->setRange(1, 15);
        blurRadiusSlider_->setPageStep(2);
        blurRadiusSlider_->setSingleStep(2);
        blurRadiusSlider_->setValue(3);
        blurRadiusSlider_->setEnabled(false);
        blurRadiusValueLabel_ = new QLabel("3");
        blurRadiusValueLabel_->setMinimumWidth(30);

        blurLayout->addWidget(blurCheckbox_);
        blurLayout->addWidget(new QLabel("Sigma:"));
        blurLayout->addWidget(blurSigmaSlider_, 1);
        blurLayout->addWidget(blurSigmaValueLabel_);
        blurLayout->addSpacing(12);
        blurLayout->addWidget(new QLabel("Radius:"));
        blurLayout->addWidget(blurRadiusSlider_, 1);
        blurLayout->addWidget(blurRadiusValueLabel_);
        controlsLayout->addLayout(blurLayout);

        auto* temporalLayout = new QHBoxLayout();
        temporalLayout->setSpacing(8);
        temporalSmoothCheckbox_ = new QCheckBox("Temporal Smooth");
        temporalSmoothCheckbox_->setChecked(false);
        temporalSmoothSlider_ = new QSlider(Qt::Horizontal);
        temporalSmoothSlider_->setRange(5, 100);
        temporalSmoothSlider_->setPageStep(5);
        temporalSmoothSlider_->setValue(25);
        temporalSmoothSlider_->setEnabled(false);
        temporalSmoothValueLabel_ = new QLabel("0.25");
        temporalSmoothValueLabel_->setMinimumWidth(40);
        temporalLayout->addWidget(temporalSmoothCheckbox_);
        temporalLayout->addSpacing(12);
        temporalLayout->addWidget(new QLabel("Blend:"));
        temporalLayout->addWidget(temporalSmoothSlider_, 1);
        temporalLayout->addWidget(temporalSmoothValueLabel_);
        controlsLayout->addLayout(temporalLayout);

        auto* spatialRow = new QHBoxLayout();
        spatialRow->setSpacing(8);
        spatialSharpenCheckbox_ = new QCheckBox("Spatial Sharpen");
        spatialSharpenCheckbox_->setChecked(false);
        spatialBackendCombo_ = new QComboBox();
        spatialBackendCombo_->addItem("AMD FSR 1.0 (EASU + RCAS)");
        spatialBackendCombo_->addItem("NVIDIA Image Scaling (default)");
        spatialBackendCombo_->setEnabled(false);
        spatialSharpnessSlider_ = new QSlider(Qt::Horizontal);
        spatialSharpnessSlider_->setRange(0, 100);
        spatialSharpnessSlider_->setPageStep(5);
        spatialSharpnessSlider_->setValue(25);
        spatialSharpnessSlider_->setEnabled(false);
        spatialSharpnessValueLabel_ = new QLabel("0.25");
        spatialSharpnessValueLabel_->setMinimumWidth(40);

        spatialRow->addWidget(spatialSharpenCheckbox_);
        spatialRow->addSpacing(12);
        spatialRow->addWidget(new QLabel("Backend:"));
        spatialRow->addWidget(spatialBackendCombo_, 1);
        spatialRow->addSpacing(12);
        spatialRow->addWidget(new QLabel("Sharpness:"));
        spatialRow->addWidget(spatialSharpnessSlider_, 1);
        spatialRow->addWidget(spatialSharpnessValueLabel_);
        controlsLayout->addLayout(spatialRow);

        auto* focusLayout = new QHBoxLayout();
        focusLayout->setSpacing(8);
        auto* focusXLabel = new QLabel("Focus X:");
        zoomCenterXSlider_ = new QSlider(Qt::Horizontal);
        zoomCenterXSlider_->setRange(0, kZoomFocusSliderScale);
        zoomCenterXSlider_->setPageStep(5);
        zoomCenterXSlider_->setValue(kZoomFocusSliderScale / 2);
        focusLayout->addWidget(focusXLabel);
        focusLayout->addWidget(zoomCenterXSlider_, 1);

        auto* focusYLabel = new QLabel("Focus Y:");
        zoomCenterYSlider_ = new QSlider(Qt::Horizontal);
        zoomCenterYSlider_->setRange(0, kZoomFocusSliderScale);
        zoomCenterYSlider_->setPageStep(5);
        zoomCenterYSlider_->setValue(kZoomFocusSliderScale / 2);
        focusLayout->addWidget(focusYLabel);
        focusLayout->addWidget(zoomCenterYSlider_, 1);
        controlsLayout->addLayout(focusLayout);

        auto* debugLayout = new QHBoxLayout();
        debugLayout->setSpacing(8);
        debugButton_ = new QPushButton("Debug View");
        debugButton_->setCheckable(true);
        debugButton_->setChecked(false);
        debugLayout->addWidget(debugButton_);
        focusMarkerCheckbox_ = new QCheckBox("Show Focus Point");
        focusMarkerCheckbox_->setChecked(false);
        focusMarkerCheckbox_->setToolTip("Overlay a red marker at the current zoom focus");
        debugLayout->addWidget(focusMarkerCheckbox_);
        processingStatusLabel_ = new QLabel("Processing: CPU");
        processingStatusLabel_->setObjectName("processingStatusLabel");
        processingStatusLabel_->setMinimumWidth(120);
        debugLayout->addWidget(processingStatusLabel_);
        debugLayout->addStretch(1);
        controlsLayout->addLayout(debugLayout);

        rootLayout->addWidget(controlsContainer_, 0);

        renderWidget_ = new RenderWidget();
        renderWidget_->installEventFilter(this);
        rootLayout->addWidget(renderWidget_, 1);

        setCentralWidget(central);
    }

    void setApp(OpenZoomApp* app) {
        app_ = app;
    }

    RenderWidget* renderWidget() const { return renderWidget_; }
    QComboBox* cameraCombo() const { return cameraCombo_; }
    QCheckBox* blackWhiteCheckbox() const { return bwCheckbox_; }
    QSlider* blackWhiteSlider() const { return bwSlider_; }
    QCheckBox* zoomCheckbox() const { return zoomCheckbox_; }
    QSlider* zoomSlider() const { return zoomSlider_; }
    QPushButton* debugButton() const { return debugButton_; }
    QCheckBox* focusMarkerCheckbox() const { return focusMarkerCheckbox_; }
    QSlider* zoomCenterXSlider() const { return zoomCenterXSlider_; }
    QSlider* zoomCenterYSlider() const { return zoomCenterYSlider_; }
    QCheckBox* joystickCheckbox() const { return joystickCheckbox_; }
    QToolButton* controlsToggleButton() const { return controlsToggleButton_; }
    QWidget* controlsContainer() const { return controlsContainer_; }
    QCheckBox* blurCheckbox() const { return blurCheckbox_; }
    QSlider* blurSigmaSlider() const { return blurSigmaSlider_; }
    QSlider* blurRadiusSlider() const { return blurRadiusSlider_; }
    QLabel* blurSigmaValueLabel() const { return blurSigmaValueLabel_; }
    QLabel* blurRadiusValueLabel() const { return blurRadiusValueLabel_; }
    QCheckBox* temporalSmoothCheckbox() const { return temporalSmoothCheckbox_; }
    QSlider* temporalSmoothSlider() const { return temporalSmoothSlider_; }
    QLabel* temporalSmoothValueLabel() const { return temporalSmoothValueLabel_; }
    QCheckBox* spatialSharpenCheckbox() const { return spatialSharpenCheckbox_; }
    QComboBox* spatialBackendCombo() const { return spatialBackendCombo_; }
    QSlider* spatialSharpnessSlider() const { return spatialSharpnessSlider_; }
    QLabel* spatialSharpnessValueLabel() const { return spatialSharpnessValueLabel_; }
    QLabel* processingStatusLabel() const { return processingStatusLabel_; }

protected:
    void keyPressEvent(QKeyEvent* event) override {
        if (app_ && app_->HandlePanKey(event->key(), true)) {
            event->accept();
            return;
        }
        QMainWindow::keyPressEvent(event);
    }

    void keyReleaseEvent(QKeyEvent* event) override {
        if (app_ && app_->HandlePanKey(event->key(), false)) {
            event->accept();
            return;
        }
        QMainWindow::keyReleaseEvent(event);
    }

    bool eventFilter(QObject* watched, QEvent* event) override {
        if (watched == renderWidget_) {
            switch (event->type()) {
            case QEvent::Wheel: {
                auto* wheel = static_cast<QWheelEvent*>(event);
                if (wheel->modifiers() & Qt::ControlModifier) {
                    if (app_) {
                        app_->HandleZoomWheel(wheel->angleDelta().y(), wheel->position());
                    }
                    event->accept();
                    return true;
                } else if (app_ && app_->HandlePanScroll(wheel)) {
                    event->accept();
                    return true;
                }
                break;
            }
            case QEvent::MouseButtonPress: {
                auto* mouse = static_cast<QMouseEvent*>(event);
                if (mouse->button() == Qt::MiddleButton) {
                    if (app_) {
                        app_->BeginMousePan(mouse->position(), renderWidget_->size());
                    }
                    event->accept();
                    return true;
                }
                break;
            }
            case QEvent::MouseMove: {
                auto* mouse = static_cast<QMouseEvent*>(event);
                if (app_ && app_->UpdateMousePan(mouse->position())) {
                    event->accept();
                    return true;
                }
                break;
            }
            case QEvent::MouseButtonRelease: {
                auto* mouse = static_cast<QMouseEvent*>(event);
                if (mouse->button() == Qt::MiddleButton) {
                    if (app_) {
                        app_->EndMousePan();
                    }
                    event->accept();
                    return true;
                }
                break;
            }
            default:
                break;
            }
        }
        return QMainWindow::eventFilter(watched, event);
    }

private:
    RenderWidget* renderWidget_{};
    QComboBox* cameraCombo_{};
    QCheckBox* bwCheckbox_{};
    QSlider* bwSlider_{};
    QCheckBox* zoomCheckbox_{};
    QSlider* zoomSlider_{};
    QPushButton* debugButton_{};
    QCheckBox* focusMarkerCheckbox_{};
    QSlider* zoomCenterXSlider_{};
    QSlider* zoomCenterYSlider_{};
    QCheckBox* joystickCheckbox_{};
    QToolButton* controlsToggleButton_{};
    QWidget* controlsContainer_{};
    QCheckBox* blurCheckbox_{};
    QSlider* blurSigmaSlider_{};
    QSlider* blurRadiusSlider_{};
    QLabel* blurSigmaValueLabel_{};
    QLabel* blurRadiusValueLabel_{};
    QCheckBox* temporalSmoothCheckbox_{};
    QSlider* temporalSmoothSlider_{};
    QLabel* temporalSmoothValueLabel_{};
    QCheckBox* spatialSharpenCheckbox_{};
    QComboBox* spatialBackendCombo_{};
    QSlider* spatialSharpnessSlider_{};
    QLabel* spatialSharpnessValueLabel_{};
    QLabel* processingStatusLabel_{};
    OpenZoomApp* app_{};
};


OpenZoomApp::OpenZoomApp(int& argc, char** argv)
    : QObject(nullptr) {
    qtApp_ = new QApplication(argc, argv);
    ResolveCudaBufferFormatFromOptions();
    InitializePlatform();

    presenter_ = std::make_unique<D3D12Presenter>();
    ResetCudaFenceState();

    mainWindow_ = std::make_unique<MainWindow>();
    mainWindow_->setApp(this);
    renderWidget_ = mainWindow_->renderWidget();
    renderWidget_->setPresenter(presenter_.get());
    cameraCombo_ = mainWindow_->cameraCombo();
    bwCheckbox_ = mainWindow_->blackWhiteCheckbox();
    bwSlider_ = mainWindow_->blackWhiteSlider();
    zoomCheckbox_ = mainWindow_->zoomCheckbox();
    zoomSlider_ = mainWindow_->zoomSlider();
    debugButton_ = mainWindow_->debugButton();
    focusMarkerCheckbox_ = mainWindow_->focusMarkerCheckbox();
    zoomCenterXSlider_ = mainWindow_->zoomCenterXSlider();
    zoomCenterYSlider_ = mainWindow_->zoomCenterYSlider();
    joystickCheckbox_ = mainWindow_->joystickCheckbox();
    collapseButton_ = mainWindow_->controlsToggleButton();
    controlsContainer_ = mainWindow_->controlsContainer();
    blurCheckbox_ = mainWindow_->blurCheckbox();
    blurSigmaSlider_ = mainWindow_->blurSigmaSlider();
    blurRadiusSlider_ = mainWindow_->blurRadiusSlider();
    blurSigmaValueLabel_ = mainWindow_->blurSigmaValueLabel();
    blurRadiusValueLabel_ = mainWindow_->blurRadiusValueLabel();
    temporalSmoothCheckbox_ = mainWindow_->temporalSmoothCheckbox();
    temporalSmoothSlider_ = mainWindow_->temporalSmoothSlider();
    temporalSmoothValueLabel_ = mainWindow_->temporalSmoothValueLabel();
    spatialSharpenCheckbox_ = mainWindow_->spatialSharpenCheckbox();
    spatialBackendCombo_ = mainWindow_->spatialBackendCombo();
    spatialSharpnessSlider_ = mainWindow_->spatialSharpnessSlider();
    spatialSharpnessValueLabel_ = mainWindow_->spatialSharpnessValueLabel();
    processingStatusLabel_ = mainWindow_->processingStatusLabel();

    joystickOverlay_ = new JoystickOverlay(renderWidget_);
    connect(joystickOverlay_, &JoystickOverlay::JoystickChanged,
            this, [this](float x, float y) {
                joystickPanX_ = std::clamp(x, -1.0f, 1.0f);
                joystickPanY_ = std::clamp(y, -1.0f, 1.0f);
            });
    UpdateJoystickVisibility();

    if (zoomCenterXSlider_) {
        zoomCenterX_ = std::clamp(static_cast<float>(zoomCenterXSlider_->value()) / static_cast<float>(kZoomFocusSliderScale), 0.0f, 1.0f);
    }
    if (zoomCenterYSlider_) {
        zoomCenterY_ = std::clamp(static_cast<float>(zoomCenterYSlider_->value()) / static_cast<float>(kZoomFocusSliderScale), 0.0f, 1.0f);
    }

    SetZoomCenter(zoomCenterX_, zoomCenterY_, true);

    if (blurSigmaSlider_) {
        blurSigma_ = SliderValueToSigma(blurSigmaSlider_->value());
    }
    if (blurRadiusSlider_) {
        int normalized = NormalizeRadiusValue(blurRadiusSlider_->value());
        blurRadius_ = normalized;
        if (blurRadiusSlider_->value() != normalized) {
            blurRadiusSlider_->setValue(normalized);
        }
    }
    blurEnabled_ = blurCheckbox_ ? blurCheckbox_->isChecked() : false;
    temporalSmoothEnabled_ = temporalSmoothCheckbox_ ? temporalSmoothCheckbox_->isChecked() : false;
    if (temporalSmoothSlider_) {
        temporalSmoothAlpha_ = std::clamp(static_cast<float>(temporalSmoothSlider_->value()) / 100.0f, 0.0f, 1.0f);
    }
    spatialSharpenEnabled_ = spatialSharpenCheckbox_ ? spatialSharpenCheckbox_->isChecked() : false;
    if (spatialBackendCombo_) {
        QSignalBlocker block(spatialBackendCombo_);
        spatialBackendCombo_->setCurrentIndex(static_cast<int>(SpatialUpscaler::kNis));
    }
    spatialUpscaler_ = SpatialUpscaler::kNis;
    if (spatialSharpnessSlider_) {
        spatialSharpness_ = std::clamp(static_cast<float>(spatialSharpnessSlider_->value()) / 100.0f, 0.0f, 1.0f);
    }
    UpdateBlurUiLabels();
    UpdateTemporalSmoothUi();
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
    if (focusMarkerCheckbox_) {
        focusMarkerEnabled_ = focusMarkerCheckbox_->isChecked();
    }

    PopulateCameraCombo();

    connect(cameraCombo_, &QComboBox::currentIndexChanged,
            this, &OpenZoomApp::OnCameraSelectionChanged);
    connect(bwCheckbox_, &QCheckBox::toggled,
            this, &OpenZoomApp::OnBlackWhiteToggled);
    connect(bwSlider_, &QSlider::valueChanged,
            this, &OpenZoomApp::OnBlackWhiteThresholdChanged);
    connect(zoomCheckbox_, &QCheckBox::toggled,
            this, &OpenZoomApp::OnZoomToggled);
    connect(zoomSlider_, &QSlider::valueChanged,
            this, &OpenZoomApp::OnZoomAmountChanged);
    if (debugButton_) {
        connect(debugButton_, &QPushButton::toggled,
                this, &OpenZoomApp::OnDebugViewToggled);
    }
    if (zoomCenterXSlider_) {
        connect(zoomCenterXSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnZoomCenterXChanged);
    }
    if (zoomCenterYSlider_) {
        connect(zoomCenterYSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnZoomCenterYChanged);
    }
    if (collapseButton_) {
        connect(collapseButton_, &QToolButton::toggled,
                this, &OpenZoomApp::OnControlsCollapsedToggled);
        OnControlsCollapsedToggled(collapseButton_->isChecked());
    }
    if (joystickCheckbox_) {
        joystickCheckbox_->setChecked(false);
        connect(joystickCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnVirtualJoystickToggled);
    }
    if (blurCheckbox_) {
        QSignalBlocker block(blurCheckbox_);
        blurCheckbox_->setChecked(blurEnabled_);
        connect(blurCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnBlurToggled);
    }
    if (blurSigmaSlider_) {
        connect(blurSigmaSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnBlurSigmaChanged);
    }
    if (blurRadiusSlider_) {
        connect(blurRadiusSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnBlurRadiusChanged);
    }
    if (temporalSmoothCheckbox_) {
        connect(temporalSmoothCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnTemporalSmoothToggled);
    }
    if (temporalSmoothSlider_) {
        connect(temporalSmoothSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnTemporalSmoothStrengthChanged);
    }
    if (spatialSharpenCheckbox_) {
        connect(spatialSharpenCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnSpatialSharpenToggled);
    }
    if (spatialBackendCombo_) {
        connect(spatialBackendCombo_, &QComboBox::currentIndexChanged,
                this, &OpenZoomApp::OnSpatialUpscalerChanged);
    }
    if (spatialSharpnessSlider_) {
        connect(spatialSharpnessSlider_, &QSlider::valueChanged,
                this, &OpenZoomApp::OnSpatialSharpnessChanged);
    }
    if (focusMarkerCheckbox_) {
        connect(focusMarkerCheckbox_, &QCheckBox::toggled,
                this, &OpenZoomApp::OnFocusMarkerToggled);
    }

    frameTimer_ = new QTimer(this);
    connect(frameTimer_, &QTimer::timeout, this, &OpenZoomApp::OnFrameTick);
    frameTimer_->start(16);

    mainWindow_->show();

    if (!cameras_.empty()) {
        cameraCombo_->setCurrentIndex(0);
        StartCameraCapture(0);
    }
}

OpenZoomApp::~OpenZoomApp() {
    if (frameTimer_) {
        frameTimer_->stop();
    }
    StopCameraCapture();

    if (mfInitialized_) {
        MFShutdown();
        mfInitialized_ = false;
    }

    if (comInitialized_) {
        CoUninitialize();
        comInitialized_ = false;
    }

    delete qtApp_;
    qtApp_ = nullptr;
}

int OpenZoomApp::Run() {
    return qtApp_->exec();
}

void OpenZoomApp::InitializePlatform() {
    const HRESULT coInit = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (SUCCEEDED(coInit) || coInit == RPC_E_CHANGED_MODE) {
        comInitialized_ = SUCCEEDED(coInit);
    } else {
        ThrowIfFailed(coInit, "Failed to initialize COM");
    }

    const HRESULT mfInit = MFStartup(MF_VERSION);
    if (FAILED(mfInit)) {
        ThrowIfFailed(mfInit, "Failed to initialize Media Foundation");
    }
    mfInitialized_ = true;

    EnumerateCameras();
}

void OpenZoomApp::ResolveCudaBufferFormatFromOptions() {
    cudaBufferFormat_ = CudaBufferFormat::kRgba8;

    bool ok = false;
    const QByteArray envValue = qgetenv("OPENZOOM_CUDA_BUFFER_FORMAT");
    if (!envValue.isEmpty()) {
        const auto parsed = ParseCudaBufferFormatToken(QString::fromUtf8(envValue), &ok);
        if (ok) {
            cudaBufferFormat_ = parsed;
        } else {
            qWarning() << "Ignoring unknown OPENZOOM_CUDA_BUFFER_FORMAT value" << envValue;
        }
    }

    const QStringList args = qtApp_->arguments();
    for (const QString& arg : args) {
        if (arg.startsWith(QStringLiteral("--cuda-buffer-format="), Qt::CaseInsensitive)) {
            const QString value = arg.section('=', 1);
            const auto parsed = ParseCudaBufferFormatToken(value, &ok);
            if (ok) {
                cudaBufferFormat_ = parsed;
            } else {
                qWarning() << "Ignoring unknown --cuda-buffer-format option" << value;
            }
        } else if (arg.compare(QStringLiteral("--cuda-buffer-fp16"), Qt::CaseInsensitive) == 0) {
            cudaBufferFormat_ = CudaBufferFormat::kRgba16F;
        } else if (arg.compare(QStringLiteral("--cuda-buffer-rgba8"), Qt::CaseInsensitive) == 0) {
            cudaBufferFormat_ = CudaBufferFormat::kRgba8;
        }
    }

    const char* label = (cudaBufferFormat_ == CudaBufferFormat::kRgba16F) ? "RGBA16F" : "RGBA8";
    qInfo() << "CUDA staging buffer format set to" << label;
}

void OpenZoomApp::EnumerateCameras() {
    cameras_.clear();

    Microsoft::WRL::ComPtr<IMFAttributes> attributes;
    ThrowIfFailed(MFCreateAttributes(attributes.GetAddressOf(), 1),
                  "Failed to create MF attributes for enumeration");
    ThrowIfFailed(attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                      MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID),
                  "Failed to set MF device attribute");

    IMFActivate** devices = nullptr;
    UINT32 count = 0;
    ThrowIfFailed(MFEnumDeviceSources(attributes.Get(), &devices, &count),
                  "Failed to enumerate camera devices");

    for (UINT32 i = 0; i < count; ++i) {
        CameraInfo info{};

        WCHAR* name = nullptr;
        UINT32 nameLen = 0;
        if (SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                                                     &name,
                                                     &nameLen))) {
            info.name.assign(name, nameLen);
            CoTaskMemFree(name);
        }

        WCHAR* symbolic = nullptr;
        UINT32 symbolicLen = 0;
        if (SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                                                     &symbolic,
                                                     &symbolicLen))) {
            info.symbolicLink.assign(symbolic, symbolicLen);
            CoTaskMemFree(symbolic);
        }

        info.activation.Attach(devices[i]);
        cameras_.push_back(std::move(info));
    }

    if (devices) {
        CoTaskMemFree(devices);
    }

    std::sort(cameras_.begin(), cameras_.end(), [](const CameraInfo& a, const CameraInfo& b) {
        return _wcsicmp(a.name.c_str(), b.name.c_str()) < 0;
    });

    selectedCameraIndex_ = cameras_.empty() ? -1 : 0;
}

void OpenZoomApp::PopulateCameraCombo() {
    if (!cameraCombo_) {
        return;
    }

    cameraCombo_->clear();
    for (const auto& camera : cameras_) {
        cameraCombo_->addItem(QString::fromWCharArray(camera.name.c_str()));
    }
}

void OpenZoomApp::OnCameraSelectionChanged(int index) {
    if (index < 0 || static_cast<size_t>(index) >= cameras_.size()) {
        return;
    }

    StartCameraCapture(static_cast<size_t>(index));
}

void OpenZoomApp::OnBlackWhiteToggled(bool checked) {
    blackWhiteEnabled_ = checked;
    if (bwSlider_) {
        bwSlider_->setEnabled(checked);
    }
}

void OpenZoomApp::OnBlackWhiteThresholdChanged(int value) {
    blackWhiteThreshold_ = std::clamp(static_cast<float>(value) / 255.0f, 0.0f, 1.0f);
}

void OpenZoomApp::OnZoomToggled(bool checked) {
    zoomEnabled_ = checked;
    if (zoomSlider_) {
        zoomSlider_->setEnabled(checked);
    }
}

void OpenZoomApp::OnZoomAmountChanged(int value) {
    zoomAmount_ = std::max(1.0f, static_cast<float>(value) / static_cast<float>(kZoomSliderScale));
}

void OpenZoomApp::OnDebugViewToggled(bool checked) {
    debugViewEnabled_ = checked;
    if (focusMarkerCheckbox_) {
        focusMarkerCheckbox_->setEnabled(!checked);
    }
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnZoomCenterXChanged(int value) {
    if (suspendControlSync_) {
        return;
    }
    const float norm = std::clamp(static_cast<float>(value) / static_cast<float>(kZoomFocusSliderScale), 0.0f, 1.0f);
    SetZoomCenter(norm, zoomCenterY_, false);
}

void OpenZoomApp::OnZoomCenterYChanged(int value) {
    if (suspendControlSync_) {
        return;
    }
    const float norm = std::clamp(static_cast<float>(value) / static_cast<float>(kZoomFocusSliderScale), 0.0f, 1.0f);
    SetZoomCenter(zoomCenterX_, norm, false);
}

void OpenZoomApp::OnControlsCollapsedToggled(bool checked) {
    controlsCollapsed_ = !checked;
    if (controlsContainer_) {
        controlsContainer_->setVisible(checked);
    }
    if (collapseButton_) {
        collapseButton_->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        collapseButton_->setText(checked ? "Hide Controls" : "Show Controls");
    }
}

void OpenZoomApp::OnVirtualJoystickToggled(bool checked) {
    virtualJoystickEnabled_ = checked;
    if (!virtualJoystickEnabled_) {
        joystickPanX_ = 0.0f;
        joystickPanY_ = 0.0f;
        if (joystickOverlay_) {
            joystickOverlay_->ResetKnob();
        }
    } else {
        joystickPanX_ = 0.0f;
        joystickPanY_ = 0.0f;
    }
    UpdateJoystickVisibility();
}

void OpenZoomApp::OnBlurToggled(bool checked) {
    blurEnabled_ = checked;
    UpdateBlurUiLabels();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnBlurSigmaChanged(int value) {
    blurSigma_ = SliderValueToSigma(value);
    UpdateBlurUiLabels();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnBlurRadiusChanged(int value) {
    int normalized = NormalizeRadiusValue(value);
    if (blurRadiusSlider_ && normalized != value) {
        QSignalBlocker blocker(blurRadiusSlider_);
        blurRadiusSlider_->setValue(normalized);
    }
    blurRadius_ = normalized;
    UpdateBlurUiLabels();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnFocusMarkerToggled(bool checked) {
    focusMarkerEnabled_ = checked;
}

void OpenZoomApp::OnSpatialSharpenToggled(bool checked) {
    spatialSharpenEnabled_ = checked;
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnSpatialUpscalerChanged(int index) {
    const int clamped = std::clamp(index, 0, 1);
    spatialUpscaler_ = static_cast<SpatialUpscaler>(clamped);
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnSpatialSharpnessChanged(int value) {
    spatialSharpness_ = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
    if (spatialSharpnessValueLabel_) {
        spatialSharpnessValueLabel_->setText(QString::number(spatialSharpness_, 'f', 2));
    }
    UpdateSpatialSharpenUi();
    UpdateProcessingStatusLabel();
}

void OpenZoomApp::OnTemporalSmoothToggled(bool checked) {
    temporalSmoothEnabled_ = checked;
    if (temporalSmoothSlider_) {
        temporalSmoothSlider_->setEnabled(checked);
    }
    temporalHistoryValid_ = false;
    temporalHistoryCpu_.clear();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
    }
    UpdateTemporalSmoothUi();
}

void OpenZoomApp::OnTemporalSmoothStrengthChanged(int value) {
    const int sliderMin = temporalSmoothSlider_ ? temporalSmoothSlider_->minimum() : 1;
    const int sliderMax = temporalSmoothSlider_ ? temporalSmoothSlider_->maximum() : 100;
    const int clamped = std::clamp(value, sliderMin, sliderMax);
    if (temporalSmoothSlider_ && clamped != value) {
        QSignalBlocker block(temporalSmoothSlider_);
        temporalSmoothSlider_->setValue(clamped);
    }
    temporalSmoothAlpha_ = std::clamp(static_cast<float>(clamped) / 100.0f, 0.0f, 1.0f);
    if (temporalSmoothValueLabel_) {
        temporalSmoothValueLabel_->setText(QString::number(temporalSmoothAlpha_, 'f', 2));
    }
    temporalHistoryValid_ = false;
    temporalHistoryCpu_.clear();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
    }
    UpdateTemporalSmoothUi();
}

void OpenZoomApp::SetZoomCenter(float normX, float normY, bool syncUi) {
    const float clampedX = std::clamp(normX, 0.0f, 1.0f);
    const float clampedY = std::clamp(normY, 0.0f, 1.0f);
    zoomCenterX_ = clampedX;
    zoomCenterY_ = clampedY;

    if (syncUi) {
        suspendControlSync_ = true;
        if (zoomCenterXSlider_) {
            QSignalBlocker blockX(zoomCenterXSlider_);
            zoomCenterXSlider_->setValue(static_cast<int>(std::round(clampedX * kZoomFocusSliderScale)));
        }
        if (zoomCenterYSlider_) {
            QSignalBlocker blockY(zoomCenterYSlider_);
            zoomCenterYSlider_->setValue(static_cast<int>(std::round(clampedY * kZoomFocusSliderScale)));
        }
        suspendControlSync_ = false;
    }
}

bool OpenZoomApp::HandlePanKey(int key, bool pressed) {
    switch (key) {
    case Qt::Key_Left:
        panLeftPressed_ = pressed;
        return true;
    case Qt::Key_Right:
        panRightPressed_ = pressed;
        return true;
    case Qt::Key_Up:
        panUpPressed_ = pressed;
        return true;
    case Qt::Key_Down:
        panDownPressed_ = pressed;
        return true;
    default:
        break;
    }
    return false;
}

bool OpenZoomApp::HandlePanScroll(const QWheelEvent* wheelEvent) {
    if (!wheelEvent || !renderWidget_) {
        return false;
    }

    if (debugViewEnabled_) {
        return false;
    }

    if (!zoomEnabled_ || zoomAmount_ <= 1.0f) {
        return false;
    }

    QPointF pixelDelta = wheelEvent->pixelDelta();
    float deltaX = 0.0f;
    float deltaY = 0.0f;
    bool hasPixelPrecision = false;

    if (!pixelDelta.isNull()) {
        deltaX = pixelDelta.x();
        deltaY = pixelDelta.y();
        hasPixelPrecision = true;
    } else {
        QPoint angleDelta = wheelEvent->angleDelta();
        if (angleDelta.isNull()) {
            return false;
        }
        deltaX = static_cast<float>(angleDelta.x()) / 120.0f;
        deltaY = static_cast<float>(angleDelta.y()) / 120.0f;
    }

    const float zoomFactor = std::max(zoomAmount_, 1.0f);
    float moveX = 0.0f;
    float moveY = 0.0f;

    if (hasPixelPrecision) {
        const float widgetWidth = static_cast<float>(std::max(1, renderWidget_->width()));
        const float widgetHeight = static_cast<float>(std::max(1, renderWidget_->height()));
        moveX = -deltaX / widgetWidth / zoomFactor;
        moveY = -deltaY / widgetHeight / zoomFactor;
    } else {
        constexpr float wheelStepScale = 1.2f;
        moveX = -deltaX * kPanKeyboardStep * wheelStepScale / zoomFactor;
        moveY = -deltaY * kPanKeyboardStep * wheelStepScale / zoomFactor;
    }

    if (std::abs(moveX) < 1e-6f && std::abs(moveY) < 1e-6f) {
        return false;
    }

    SetZoomCenter(zoomCenterX_ + moveX,
                  zoomCenterY_ + moveY,
                  true);
    return true;
}

void OpenZoomApp::ApplyInputForces() {
    float moveX = 0.0f;
    if (panLeftPressed_) {
        moveX -= 1.0f;
    }
    if (panRightPressed_) {
        moveX += 1.0f;
    }
    moveX += joystickPanX_;

    float moveY = 0.0f;
    if (panUpPressed_) {
        moveY -= 1.0f;
    }
    if (panDownPressed_) {
        moveY += 1.0f;
    }
    moveY += -joystickPanY_;

    if (std::abs(moveX) < 1e-5f && std::abs(moveY) < 1e-5f) {
        return;
    }

    const float length = std::sqrt(moveX * moveX + moveY * moveY);
    float normalizedX = moveX;
    float normalizedY = moveY;
    if (length > 1e-5f) {
        normalizedX /= length;
        normalizedY /= length;
    }

    const bool keyboardActive = panLeftPressed_ || panRightPressed_ || panUpPressed_ || panDownPressed_;
    const float analogStrength = std::sqrt(joystickPanX_ * joystickPanX_ + joystickPanY_ * joystickPanY_);

    float baseStep = 0.0f;
    if (keyboardActive) {
        baseStep = kPanKeyboardStep;
    }
    if (analogStrength > 0.001f) {
        baseStep = std::max(baseStep, kPanJoystickStep * std::clamp(analogStrength, 0.1f, 1.0f));
    }
    if (baseStep <= 0.0f) {
        baseStep = kPanJoystickStep;
    }

    const float zoomFactor = std::max(1.0f, zoomAmount_);
    const float step = baseStep / zoomFactor;

    SetZoomCenter(zoomCenterX_ + normalizedX * step,
                  zoomCenterY_ + normalizedY * step,
                  true);
}

void OpenZoomApp::UpdateJoystickVisibility() {
    if (!joystickOverlay_) {
        return;
    }
    if (virtualJoystickEnabled_) {
        joystickOverlay_->ResetKnob();
        joystickOverlay_->show();
    } else {
        joystickOverlay_->hide();
    }
}

void OpenZoomApp::UpdateBlurUiLabels() {
    const QString sigmaText = QString::number(blurSigma_, 'f', 1);
    const bool blurActive = blurEnabled_;
    if (blurSigmaValueLabel_) {
        blurSigmaValueLabel_->setText(sigmaText);
        blurSigmaValueLabel_->setEnabled(blurActive);
    }
    if (blurRadiusValueLabel_) {
        blurRadiusValueLabel_->setText(QString::number(blurRadius_));
        blurRadiusValueLabel_->setEnabled(blurActive);
    }
    if (blurSigmaSlider_) {
        blurSigmaSlider_->setEnabled(blurActive);
    }
    if (blurRadiusSlider_) {
        blurRadiusSlider_->setEnabled(blurActive);
    }
    if (blurCheckbox_) {
        QSignalBlocker block(blurCheckbox_);
        blurCheckbox_->setChecked(blurEnabled_);
        blurCheckbox_->setEnabled(true);
    }
}

void OpenZoomApp::UpdateTemporalSmoothUi() {
    if (temporalSmoothCheckbox_) {
        QSignalBlocker block(temporalSmoothCheckbox_);
        temporalSmoothCheckbox_->setChecked(temporalSmoothEnabled_);
    }
    if (temporalSmoothSlider_) {
        temporalSmoothSlider_->setEnabled(temporalSmoothEnabled_);
        const int sliderValue = static_cast<int>(std::round(temporalSmoothAlpha_ * 100.0f));
        QSignalBlocker block(temporalSmoothSlider_);
        temporalSmoothSlider_->setValue(std::clamp(sliderValue,
                                                  temporalSmoothSlider_->minimum(),
                                                  temporalSmoothSlider_->maximum()));
    }
    if (temporalSmoothValueLabel_) {
        temporalSmoothValueLabel_->setEnabled(temporalSmoothEnabled_);
        temporalSmoothValueLabel_->setText(QString::number(temporalSmoothAlpha_, 'f', 2));
    }
}

void OpenZoomApp::ApplyTemporalSmoothCpu(std::vector<uint8_t>& frame, UINT width, UINT height) {
    if (!temporalSmoothEnabled_) {
        temporalHistoryValid_ = false;
        temporalHistoryCpu_.clear();
        return;
    }

    if (frame.empty() || width == 0 || height == 0) {
        return;
    }

    const size_t channelCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    if (frame.size() < channelCount) {
        return;
    }

    if (temporalHistoryCpu_.size() != channelCount) {
        temporalHistoryCpu_.assign(channelCount, 0.0f);
        temporalHistoryValid_ = false;
    }

    const float alpha = std::clamp(temporalSmoothAlpha_, 0.0f, 1.0f);
    const float oneMinusAlpha = 1.0f - alpha;

    if (!temporalHistoryValid_) {
        for (size_t i = 0; i < channelCount; ++i) {
            temporalHistoryCpu_[i] = static_cast<float>(frame[i]);
        }
        temporalHistoryValid_ = true;
        return;
    }

    for (size_t i = 0; i < channelCount; ++i) {
        if ((i & 3u) == 3u) {
            temporalHistoryCpu_[i] = 255.0f;
            frame[i] = 255u;
            continue;
        }

        const float curr = static_cast<float>(frame[i]);
        const float prev = temporalHistoryCpu_[i];
        const float blended = alpha * curr + oneMinusAlpha * prev;
        temporalHistoryCpu_[i] = blended;
        frame[i] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(blended)), 0, 255));
    }
}

void OpenZoomApp::UpdateSpatialSharpenUi() {
    const bool enabled = spatialSharpenEnabled_;

    if (spatialSharpenCheckbox_) {
        QSignalBlocker block(spatialSharpenCheckbox_);
        spatialSharpenCheckbox_->setChecked(enabled);
    }
    if (spatialBackendCombo_) {
        QSignalBlocker block(spatialBackendCombo_);
        spatialBackendCombo_->setCurrentIndex(static_cast<int>(spatialUpscaler_));
        spatialBackendCombo_->setEnabled(enabled);
    }
    if (spatialSharpnessSlider_) {
        spatialSharpnessSlider_->setEnabled(enabled);
        if (enabled) {
            QSignalBlocker block(spatialSharpnessSlider_);
            spatialSharpnessSlider_->setValue(static_cast<int>(std::round(spatialSharpness_ * 100.0f)));
        }
    }
    if (spatialSharpnessValueLabel_) {
        spatialSharpnessValueLabel_->setEnabled(enabled);
        spatialSharpnessValueLabel_->setText(QString::number(spatialSharpness_, 'f', 2));
    }
}

void OpenZoomApp::UpdateProcessingStatusLabel() {
    if (!processingStatusLabel_) {
        return;
    }

    QString text;
    QString color;

    auto backendLabel = [this]() -> QString {
        if (spatialSharpenEnabled_) {
            if (spatialUpscaler_ == SpatialUpscaler::kNis) {
                return QStringLiteral("Spatial Sharpen (NIS)");
            }
            return QStringLiteral("Spatial Sharpen (FSR)");
        }
        if (blurEnabled_) {
            return QStringLiteral("Gaussian Blur");
        }
        return QStringLiteral("None");
    };

    if (!cameraActive_) {
        if (lastCameraError_.isEmpty()) {
            text = QStringLiteral("Processing: Idle (camera offline)");
        } else {
            text = QStringLiteral("Processing: Idle (camera offline — %1)").arg(lastCameraError_);
        }
        color = QStringLiteral("#c0392b");
    } else if (debugViewEnabled_) {
        text = QStringLiteral("Processing: CPU (debug view)");
        color = QStringLiteral("#d17c00");
    } else if (usingCudaLastFrame_ && cudaPipelineAvailable_) {
        const QString backend = backendLabel();
        if (cudaFenceInteropEnabled_) {
            text = QStringLiteral("Processing: GPU (fence interop, %1)").arg(backend);
        } else {
            text = QStringLiteral("Processing: GPU (%1)").arg(backend);
        }
        color = QStringLiteral("#1c9c3e");
    } else if (cudaPipelineAvailable_) {
        text = QStringLiteral("Processing: CPU (fallback, %1)").arg(backendLabel());
        color = QStringLiteral("#c0392b");
    } else {
        text = QStringLiteral("Processing: CPU (%1)").arg(backendLabel());
        color = QStringLiteral("#c0392b");
    }

    processingStatusLabel_->setText(text);
    processingStatusLabel_->setStyleSheet(QStringLiteral("color: %1;").arg(color));
}

void OpenZoomApp::HandleZoomWheel(int delta, const QPointF& localPos) {
    if (!zoomSlider_) {
        return;
    }

    float focusU = 0.0f;
    float focusV = 0.0f;
    bool hasFocus = MapViewToSource(localPos, focusU, focusV);
    if (debugViewEnabled_) {
        hasFocus = false;
    }

    if (!zoomEnabled_) {
        if (zoomCheckbox_) {
            QSignalBlocker blocker(zoomCheckbox_);
            zoomCheckbox_->setChecked(true);
        }
        zoomEnabled_ = true;
        zoomSlider_->setEnabled(true);
    }

    const int stepUnits = (delta / 120); // standard wheel step
    if (stepUnits == 0) {
        return;
    }
    const float prevZoom = zoomAmount_;
    const int stepSize = std::max(zoomSlider_->pageStep() / 2, 1);
    const int deltaValue = stepUnits * stepSize;
    const int newValue = std::clamp(zoomSlider_->value() + deltaValue,
                                    zoomSlider_->minimum(),
                                    zoomSlider_->maximum());

    if (newValue == zoomSlider_->value()) {
        return;
    }

    QSignalBlocker blockSlider(zoomSlider_);
    zoomSlider_->setValue(newValue);
    blockSlider.unblock();
    OnZoomAmountChanged(newValue);
    const float newZoom = zoomAmount_;

    if (hasFocus) {
        const float factor = (prevZoom <= 0.0f || newZoom <= 0.0f) ? 1.0f : (prevZoom / newZoom);
        const float newCenterX = focusU - (focusU - zoomCenterX_) * factor;
        const float newCenterY = focusV - (focusV - zoomCenterY_) * factor;
        SetZoomCenter(newCenterX, newCenterY, true);
    }
}

bool OpenZoomApp::MapViewToSource(const QPointF& pos, float& outX, float& outY) const {
    if (!renderWidget_ || cameraFrameWidth_ == 0 || cameraFrameHeight_ == 0) {
        return false;
    }

    const int targetWidth = std::max(1, renderWidget_->width());
    const int targetHeight = std::max(1, renderWidget_->height());

    const float srcWidth = static_cast<float>(cameraFrameWidth_);
    const float srcHeight = static_cast<float>(cameraFrameHeight_);

    bool cropToFill = !debugViewEnabled_;
    float cropWidth = srcWidth;
    float cropHeight = srcHeight;

    if (cropToFill) {
        const float targetAspect = static_cast<float>(targetWidth) / static_cast<float>(targetHeight);
        const float srcAspect = srcWidth / srcHeight;
        if (targetAspect > srcAspect) {
            cropHeight = srcWidth / targetAspect;
            cropHeight = std::min(cropHeight, srcHeight);
        } else {
            cropWidth = srcHeight * targetAspect;
            cropWidth = std::min(cropWidth, srcWidth);
        }
    }

    cropWidth = std::clamp(cropWidth, 1.0f, srcWidth);
    cropHeight = std::clamp(cropHeight, 1.0f, srcHeight);

    float centerX = zoomCenterX_ * (srcWidth - 1.0f);
    float centerY = zoomCenterY_ * (srcHeight - 1.0f);

    const float halfCropWidth = cropWidth * 0.5f;
    const float halfCropHeight = cropHeight * 0.5f;

    const float minCenterX = std::max(0.0f, halfCropWidth - 0.5f);
    const float maxCenterX = std::max(minCenterX, (srcWidth - 1.0f) - (halfCropWidth - 0.5f));
    const float minCenterY = std::max(0.0f, halfCropHeight - 0.5f);
    const float maxCenterY = std::max(minCenterY, (srcHeight - 1.0f) - (halfCropHeight - 0.5f));

    centerX = std::clamp(centerX, minCenterX, maxCenterX);
    centerY = std::clamp(centerY, minCenterY, maxCenterY);

    float startX = centerX - halfCropWidth + 0.5f;
    float startY = centerY - halfCropHeight + 0.5f;
    startX = std::clamp(startX, 0.0f, srcWidth - cropWidth);
    startY = std::clamp(startY, 0.0f, srcHeight - cropHeight);

    float scaleFactor;
    if (cropToFill) {
        scaleFactor = static_cast<float>(targetWidth) / cropWidth;
    } else {
        const float widthScale = static_cast<float>(targetWidth) / cropWidth;
        const float heightScale = static_cast<float>(targetHeight) / cropHeight;
        scaleFactor = std::min(widthScale, heightScale);
    }
    if (!(scaleFactor > 0.0f) || !std::isfinite(scaleFactor)) {
        scaleFactor = 1.0f;
    }

    UINT activeWidth = static_cast<UINT>(std::round(cropWidth * scaleFactor));
    UINT activeHeight = static_cast<UINT>(std::round(cropHeight * scaleFactor));
    activeWidth = std::clamp(activeWidth, 1u, static_cast<UINT>(targetWidth));
    activeHeight = std::clamp(activeHeight, 1u, static_cast<UINT>(targetHeight));

    const UINT offsetX = (targetWidth > static_cast<int>(activeWidth)) ?
                         static_cast<UINT>((targetWidth - static_cast<int>(activeWidth)) / 2) : 0;
    const UINT offsetY = (targetHeight > static_cast<int>(activeHeight)) ?
                         static_cast<UINT>((targetHeight - static_cast<int>(activeHeight)) / 2) : 0;

    float localX = static_cast<float>(pos.x()) - static_cast<float>(offsetX);
    float localY = static_cast<float>(pos.y()) - static_cast<float>(offsetY);

    localX = std::clamp(localX, 0.0f, static_cast<float>(activeWidth - 1));
    localY = std::clamp(localY, 0.0f, static_cast<float>(activeHeight - 1));

    const float stepX = cropWidth / static_cast<float>(activeWidth);
    const float stepY = cropHeight / static_cast<float>(activeHeight);

    const float sampleX = startX + localX * stepX;
    const float sampleY = startY + localY * stepY;

    const float denomX = std::max(srcWidth - 1.0f, 1.0f);
    const float denomY = std::max(srcHeight - 1.0f, 1.0f);
    outX = std::clamp(sampleX / denomX, 0.0f, 1.0f);
    outY = std::clamp(sampleY / denomY, 0.0f, 1.0f);
    return true;
}

void OpenZoomApp::ResetCudaFenceState() {
    const UINT64 baseValue = presenter_ ? presenter_->GetLastSignaledFenceValue() : 0;
    sharedFenceCounter_ = baseValue + 1;
    lastCudaSignalValue_ = 0;
    lastGraphicsSignalValue_ = baseValue;
    cudaFenceInteropEnabled_ = false;
}

bool OpenZoomApp::EnsureCudaSurface(UINT width, UINT height) {
    if (!presenter_ || !renderWidget_ || !renderWidget_->isPresenterReady()) {
        qWarning() << "CUDA surface unavailable: presenter or render widget not ready";
        return false;
    }

    if (cudaSurface_ && cudaSurface_->IsValid() &&
        cudaSurfaceWidth_ == width && cudaSurfaceHeight_ == height) {
        return true;
    }

    cudaSurface_.reset();
    cudaSharedTexture_.Reset();
    cudaSurfaceWidth_ = 0;
    cudaSurfaceHeight_ = 0;
    ResetCudaFenceState();

    ID3D12Device* device = presenter_->GetDevice();
    if (!device) {
        qWarning() << "CUDA surface unavailable: presenter returned null device";
        return false;
    }

    try {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Alignment = 0;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;

        ThrowIfFailed(device->CreateCommittedResource(&heapProps,
                                                      D3D12_HEAP_FLAG_SHARED,
                                                      &desc,
                                                      D3D12_RESOURCE_STATE_COMMON,
                                                      nullptr,
                                                      IID_PPV_ARGS(&cudaSharedTexture_)),
                      "Failed to create CUDA shared texture");

        auto surface = std::make_unique<CudaInteropSurface>(cudaSharedTexture_.Get(), presenter_->GetFence());
        if (!surface || !surface->IsValid()) {
            if (surface) {
                const std::string& err = surface->LastError();
                if (!err.empty()) {
                    qWarning() << "CUDA surface detail:" << err.c_str();
                }
            }
            cudaSharedTexture_.Reset();
            qWarning() << "CUDA surface initialization failed: surface invalid"
                       << "(requested" << width << "x" << height << ")";
            return false;
        }

        cudaSurface_ = std::move(surface);
        cudaSurfaceWidth_ = width;
        cudaSurfaceHeight_ = height;
        cudaPipelineAvailable_ = true;
        cudaFenceInteropEnabled_ = cudaSurface_->HasExternalSemaphore();
        if (cudaFenceInteropEnabled_) {
            lastGraphicsSignalValue_ = presenter_->GetLastSignaledFenceValue();
            sharedFenceCounter_ = lastGraphicsSignalValue_ + 1;
            lastCudaSignalValue_ = 0;
            qInfo() << "CUDA fence interop enabled; base fence value"
                    << static_cast<unsigned long long>(lastGraphicsSignalValue_);
        } else {
            qInfo() << "CUDA surface ready without fence interop";
        }
        qInfo() << "CUDA surface ready for" << width << "x" << height;
        return true;
    } catch (...) {
        cudaSurface_.reset();
        cudaSharedTexture_.Reset();
        cudaSurfaceWidth_ = 0;
        cudaSurfaceHeight_ = 0;
        cudaPipelineAvailable_ = false;
        ResetCudaFenceState();
        qWarning() << "CUDA surface creation exception triggered fallback";
        return false;
    }
}

bool OpenZoomApp::ProcessFrameWithCuda(UINT width, UINT height) {
    if (stageRaw_.empty()) {
        qWarning() << "CUDA pipeline skipped: stageRaw_ empty";
        return false;
    }

    if (!EnsureCudaSurface(width, height)) {
        qWarning() << "CUDA pipeline disabled: failed to ensure CUDA surface";
        if (cudaSurface_) {
            const std::string& err = cudaSurface_->LastError();
            if (!err.empty()) {
                qWarning() << "CUDA surface detail:" << err.c_str();
            }
        }
        usingCudaLastFrame_ = false;
        return false;
    }

    if (!cudaSurface_) {
        qWarning() << "CUDA pipeline disabled: surface not available";
        usingCudaLastFrame_ = false;
        return false;
    }

    ProcessingInput input{};
    input.hostPixels = stageRaw_.data();
    input.hostStrideBytes = width * 4;
    input.pixelSizeBytes = static_cast<unsigned int>(sizeof(uint32_t));
    input.width = width;
    input.height = height;

    ProcessingSettings settings{};
    settings.enableBlackWhite = blackWhiteEnabled_;
    settings.blackWhiteThreshold = blackWhiteThreshold_;
    settings.enableZoom = zoomEnabled_;
    settings.zoomAmount = zoomAmount_;
    settings.zoomCenterX = zoomCenterX_;
    settings.zoomCenterY = zoomCenterY_;
    settings.enableBlur = blurEnabled_;
    settings.blurRadius = std::clamp(blurRadius_, 1, 15);
    settings.blurSigma = blurSigma_;
    settings.drawFocusMarker = focusMarkerEnabled_ && !debugViewEnabled_;
    settings.enableSpatialSharpen = spatialSharpenEnabled_;
    settings.spatialUpscaler = spatialUpscaler_;
    settings.spatialSharpness = spatialSharpness_;
    settings.stagingFormat = cudaBufferFormat_;
    settings.enableTemporalSmoothing = temporalSmoothEnabled_;
    settings.temporalSmoothingAlpha = temporalSmoothAlpha_;

    FenceSyncParams cudaSyncParams{};
    uint64_t cudaSignalCandidate = 0;
    if (cudaFenceInteropEnabled_) {
        cudaSyncParams.enable = true;
        cudaSyncParams.waitValue = lastGraphicsSignalValue_;
        cudaSignalCandidate = sharedFenceCounter_;
        cudaSyncParams.signalValue = cudaSignalCandidate;
    }

    if (!cudaSurface_->ProcessFrame(input, settings, cudaSyncParams)) {
        cudaPipelineAvailable_ = false;
        qWarning() << "CUDA pipeline processing failed, falling back to CPU";
        ResetCudaFenceState();
        usingCudaLastFrame_ = false;
        return false;
    }

    if (cudaSyncParams.enable) {
        lastCudaSignalValue_ = cudaSignalCandidate;
        sharedFenceCounter_ = cudaSignalCandidate + 1;
    }

    cudaPipelineAvailable_ = true;
    FenceSyncParams presentSync{};
    uint64_t graphicsSignalCandidate = 0;
    if (cudaFenceInteropEnabled_) {
        presentSync.enable = true;
        presentSync.waitValue = lastCudaSignalValue_;
        graphicsSignalCandidate = sharedFenceCounter_;
        presentSync.signalValue = graphicsSignalCandidate;
    }

    presenter_->PresentFromTexture(cudaSharedTexture_.Get(),
                                   width,
                                   height,
                                   presentSync.enable ? &presentSync : nullptr);

    if (presentSync.enable) {
        lastGraphicsSignalValue_ = graphicsSignalCandidate;
        sharedFenceCounter_ = graphicsSignalCandidate + 1;
    }

    usingCudaLastFrame_ = true;
    return true;
}

void OpenZoomApp::BeginMousePan(const QPointF& pos, const QSize& widgetSize) {
    middlePanActive_ = true;
    middlePanLastPos_ = pos;
    middlePanWidgetSize_ = widgetSize;
    if (renderWidget_) {
        renderWidget_->setCursor(Qt::ClosedHandCursor);
        renderWidget_->grabMouse(Qt::ClosedHandCursor);
    }
}

bool OpenZoomApp::UpdateMousePan(const QPointF& pos) {
    if (!middlePanActive_) {
        return false;
    }

    float prevU = 0.0f;
    float prevV = 0.0f;
    float currU = 0.0f;
    float currV = 0.0f;

    bool prevValid = MapViewToSource(middlePanLastPos_, prevU, prevV);
    bool currValid = MapViewToSource(pos, currU, currV);

    middlePanLastPos_ = pos;

    if (!prevValid || !currValid) {
        return false;
    }

    const float deltaU = prevU - currU;
    const float deltaV = prevV - currV;
    if (std::abs(deltaU) < 1e-6f && std::abs(deltaV) < 1e-6f) {
        return false;
    }

    SetZoomCenter(zoomCenterX_ + deltaU,
                  zoomCenterY_ + deltaV,
                  true);
    return true;
}

void OpenZoomApp::EndMousePan() {
    if (!middlePanActive_) {
        return;
    }
    middlePanActive_ = false;
    if (renderWidget_) {
        renderWidget_->releaseMouse();
        renderWidget_->unsetCursor();
    }
}

void OpenZoomApp::StartCameraCapture(size_t index) {
    if (index >= cameras_.size()) {
        return;
    }

    selectedCameraIndex_ = static_cast<int>(index);
    StopCameraCapture();
    temporalHistoryValid_ = false;
    temporalHistoryCpu_.clear();
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
    }

    try {
        CameraInfo& camera = cameras_[index];
        if (!camera.activation) {
            throw std::runtime_error("Camera activation metadata missing");
        }

        Microsoft::WRL::ComPtr<IMFMediaSource> mediaSource;
        ThrowIfFailed(camera.activation->ActivateObject(__uuidof(IMFMediaSource),
                                                    reinterpret_cast<void**>(mediaSource.GetAddressOf())),
                      "Failed to activate camera");

        Microsoft::WRL::ComPtr<IMFAttributes> readerAttributes;
        ThrowIfFailed(MFCreateAttributes(readerAttributes.GetAddressOf(), 4),
                      "Failed to create reader attributes");
        ThrowIfFailed(readerAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE),
                      "Failed to enable video processing");
        ThrowIfFailed(readerAttributes->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, TRUE),
                      "Failed to disable DXVA");
        ThrowIfFailed(readerAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, FALSE),
                      "Failed to allow converters");

        Microsoft::WRL::ComPtr<IMFSourceReader> reader;
        ThrowIfFailed(MFCreateSourceReaderFromMediaSource(mediaSource.Get(),
                                                      readerAttributes.Get(),
                                                      reader.GetAddressOf()),
                      "Failed to create source reader");

        ThrowIfFailed(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE),
                      "Failed to disable default streams");
        ThrowIfFailed(reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE),
                      "Failed to enable video stream");

        Microsoft::WRL::ComPtr<IMFMediaType> desiredType;
        ThrowIfFailed(MFCreateMediaType(desiredType.GetAddressOf()),
                      "Failed to create media type");
        ThrowIfFailed(desiredType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video),
                      "Failed to set major type");
        ThrowIfFailed(desiredType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_ARGB32),
                      "Failed to set subtype");

        HRESULT hr = reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                 nullptr,
                                                 desiredType.Get());

        if (FAILED(hr)) {
            const std::array<GUID, 3> fallbacks{MFVideoFormat_RGB32, MFVideoFormat_NV12, MFVideoFormat_YUY2};
            bool configured = false;
            for (const GUID& format : fallbacks) {
                Microsoft::WRL::ComPtr<IMFMediaType> fallback;
                ThrowIfFailed(MFCreateMediaType(fallback.GetAddressOf()),
                              "Failed to create fallback media type");
                ThrowIfFailed(fallback->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video),
                              "Failed to set fallback major type");
                ThrowIfFailed(fallback->SetGUID(MF_MT_SUBTYPE, format),
                              "Failed to set fallback subtype");

                if (SUCCEEDED(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                      nullptr,
                                                      fallback.Get()))) {
                    desiredType = fallback;
                    configured = true;
                    break;
                }
            }

            if (!configured) {
                ThrowIfFailed(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                      desiredType.GetAddressOf()),
                              "Failed to retrieve native media type");
            }
        }

        Microsoft::WRL::ComPtr<IMFMediaType> currentType;
        ThrowIfFailed(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                              currentType.GetAddressOf()),
                      "Failed to get current media type");

        GUID subtype = GUID_NULL;
        ThrowIfFailed(currentType->GetGUID(MF_MT_SUBTYPE, &subtype),
                      "Failed to get subtype");

        UINT32 frameWidth = 0;
        UINT32 frameHeight = 0;
        ThrowIfFailed(MFGetAttributeSize(currentType.Get(), MF_MT_FRAME_SIZE, &frameWidth, &frameHeight),
                      "Failed to get frame size");

        LONG stride = 0;
        if (FAILED(MFGetStrideForBitmapInfoHeader(subtype.Data1, frameWidth, &stride))) {
            if (IsEqualGUID(subtype, MFVideoFormat_ARGB32) || IsEqualGUID(subtype, MFVideoFormat_RGB32)) {
                stride = static_cast<LONG>(frameWidth * 4);
            } else if (IsEqualGUID(subtype, MFVideoFormat_NV12)) {
                stride = static_cast<LONG>(frameWidth);
            } else if (IsEqualGUID(subtype, MFVideoFormat_YUY2)) {
                stride = static_cast<LONG>(frameWidth * 2);
            } else {
                ThrowIfFailed(E_FAIL, "Unsupported camera format stride");
            }
        }

        {
            std::scoped_lock lock(cameraMutex_);
            cameraFrameBuffer_.clear();
            cameraFrameWidth_ = frameWidth;
            cameraFrameHeight_ = frameHeight;
            cameraFrameStride_ = static_cast<UINT>(std::abs(stride));
            cameraFrameDataSize_ = 0;
            cameraFrameSubtype_ = subtype;
            cameraFrameValid_ = false;
        }

        cameraMediaSource_ = std::move(mediaSource);
        cameraReader_ = std::move(reader);
        cameraActive_ = true;
        cameraCaptureRunning_ = true;
        cameraThread_ = std::thread(&OpenZoomApp::CameraCaptureLoop, this);

        lastCameraError_.clear();
        UpdateProcessingStatusLabel();
    } catch (const std::exception& e) {
        HandleCameraStartFailure(QString::fromUtf8(e.what()));
    } catch (...) {
        HandleCameraStartFailure(QStringLiteral("Unknown camera initialization failure"));
    }
}

void OpenZoomApp::StopCameraCapture() {
    cameraCaptureRunning_ = false;

    if (cameraReader_) {
        cameraReader_->Flush(MF_SOURCE_READER_ALL_STREAMS);
    }

    if (cameraThread_.joinable()) {
        cameraThread_.join();
    }

    if (cameraMediaSource_) {
        cameraMediaSource_->Shutdown();
    }

    cameraReader_.Reset();
    cameraMediaSource_.Reset();
    cameraActive_ = false;

    std::scoped_lock lock(cameraMutex_);
    cameraFrameBuffer_.clear();
    cameraFrameWidth_ = 0;
    cameraFrameHeight_ = 0;
    cameraFrameStride_ = 0;
    cameraFrameDataSize_ = 0;
    cameraFrameSubtype_ = GUID_NULL;
    cameraFrameValid_ = false;
    temporalHistoryCpu_.clear();
    temporalHistoryValid_ = false;
    if (cudaSurface_) {
        cudaSurface_->ResetTemporalHistory();
    }
}

void OpenZoomApp::HandleCameraStartFailure(const QString& message) {
    qWarning() << "Camera start failed:" << message;
    lastCameraError_ = message;
    StopCameraCapture();
    UpdateProcessingStatusLabel();
    if (mainWindow_) {
        QMessageBox::warning(mainWindow_.get(), QStringLiteral("Camera Error"), message);
    }
}


void OpenZoomApp::CameraCaptureLoop() {
    HRESULT coInit = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool threadCom = SUCCEEDED(coInit);
    if (FAILED(coInit) && coInit != RPC_E_CHANGED_MODE) {
        cameraCaptureRunning_ = false;
        return;
    }

    Microsoft::WRL::ComPtr<IMFSourceReader> reader = cameraReader_;
    if (!reader) {
        if (threadCom) {
            CoUninitialize();
        }
        return;
    }

    while (cameraCaptureRunning_) {
        DWORD streamIndex = 0;
        DWORD flags = 0;
        LONGLONG timestamp = 0;
        Microsoft::WRL::ComPtr<IMFSample> sample;

        HRESULT hr = reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                        0,
                                        &streamIndex,
                                        &flags,
                                        &timestamp,
                                        &sample);

        if (!cameraCaptureRunning_) {
            break;
        }

        if (FAILED(hr)) {
            cameraCaptureRunning_ = false;
            break;
        }

        if ((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0) {
            cameraCaptureRunning_ = false;
            break;
        }

        if ((flags & MF_SOURCE_READERF_STREAMTICK) != 0) {
            continue;
        }

        if (!sample) {
            continue;
        }

        Microsoft::WRL::ComPtr<IMFMediaBuffer> buffer;
        hr = sample->ConvertToContiguousBuffer(&buffer);
        if (FAILED(hr)) {
            continue;
        }

        BYTE* data = nullptr;
        DWORD length = 0;
        hr = buffer->Lock(&data, nullptr, &length);
        if (FAILED(hr) || !data) {
            if (SUCCEEDED(hr)) {
                buffer->Unlock();
            }
            continue;
        }

        {
            std::scoped_lock lock(cameraMutex_);
            cameraFrameBuffer_.assign(data, data + length);
            cameraFrameDataSize_ = static_cast<size_t>(length);
            cameraFrameValid_ = true;
        }

        buffer->Unlock();
    }

    if (threadCom) {
        CoUninitialize();
    }
}

void OpenZoomApp::OnFrameTick() {
    if (!cameraActive_) {
        return;
    }

    ApplyInputForces();

    std::vector<uint8_t> localFrame;
    GUID subtype = GUID_NULL;
    UINT width = 0;
    UINT height = 0;
    UINT stride = 0;
    size_t dataSize = 0;

    {
        std::scoped_lock lock(cameraMutex_);
        if (cameraFrameValid_ && !cameraFrameBuffer_.empty()) {
            localFrame = cameraFrameBuffer_;
            subtype = cameraFrameSubtype_;
            width = cameraFrameWidth_;
            height = cameraFrameHeight_;
            stride = cameraFrameStride_;
            dataSize = cameraFrameDataSize_;
            cameraFrameBuffer_.clear();
            cameraFrameDataSize_ = 0;
            cameraFrameValid_ = false;
        }
    }

    if (localFrame.empty() || width == 0 || height == 0) {
        return;
    }

    if (!ConvertFrameToBgra(localFrame, subtype, width, height, stride, dataSize)) {
        return;
    }

    BuildCompositeAndPresent(width, height);
}

bool OpenZoomApp::ConvertFrameToBgra(const std::vector<uint8_t>& frame,
                                     const GUID& subtype,
                                     UINT width,
                                     UINT height,
                                     UINT stride,
                                     UINT dataSize)
{
    if (IsEqualGUID(subtype, MFVideoFormat_ARGB32)) {
        CopyArgbToBgra(frame.data(), stride != 0 ? stride : width * 4, width, height, stageRaw_);
        return true;
    }

    if (IsEqualGUID(subtype, MFVideoFormat_RGB32)) {
        CopyRgbxToBgra(frame.data(), stride != 0 ? stride : width * 4, width, height, stageRaw_);
        return true;
    }

    if (IsEqualGUID(subtype, MFVideoFormat_NV12)) {
        return ConvertNv12ToBgra(frame.data(), dataSize, stride != 0 ? stride : width, width, height, stageRaw_);
    }

    if (IsEqualGUID(subtype, MFVideoFormat_YUY2)) {
        return ConvertYuy2ToBgra(frame.data(), dataSize, stride != 0 ? stride : width * 2, width, height, stageRaw_);
    }

    return false;
}

void OpenZoomApp::BuildCompositeAndPresent(UINT width, UINT height) {
    usingCudaLastFrame_ = false;
    if (!debugViewEnabled_ && ProcessFrameWithCuda(width, height)) {
        usingCudaLastFrame_ = true;
        UpdateProcessingStatusLabel();
        return;
    }

    stageBw_ = stageRaw_;
    if (blackWhiteEnabled_) {
        ApplyBlackWhite(stageRaw_, stageBw_, blackWhiteThreshold_);
    }

    const std::vector<uint8_t>* currentStage = blackWhiteEnabled_ ? &stageBw_ : &stageRaw_;

    const std::vector<uint8_t>& zoomSource = blackWhiteEnabled_ ? stageBw_ : stageRaw_;
    stageZoom_ = zoomSource;
    if (zoomEnabled_) {
        ApplyZoom(zoomSource, stageZoom_, width, height, zoomAmount_, zoomCenterX_, zoomCenterY_);
        currentStage = &stageZoom_;
    }

    if (blurEnabled_) {
        ApplyGaussianBlur(*currentStage, blurScratch_, stageBlur_, width, height, blurRadius_, blurSigma_);
        currentStage = &stageBlur_;
    } else {
        stageBlur_.clear();
    }

    stageFinal_ = *currentStage;
    ApplyTemporalSmoothCpu(stageFinal_, width, height);

    if (stageFinal_.empty()) {
        UpdateProcessingStatusLabel();
        return;
    }

    const uint8_t* displayData = nullptr;
    UINT displayWidth = 0;
    UINT displayHeight = 0;

    if (debugViewEnabled_) {
        std::vector<const std::vector<uint8_t>*> debugStages;
        debugStages.reserve(5);
        debugStages.push_back(&stageRaw_);
        if (blackWhiteEnabled_) {
            debugStages.push_back(&stageBw_);
        }
        if (zoomEnabled_) {
            debugStages.push_back(&stageZoom_);
        }
        if (blurEnabled_ && !stageBlur_.empty()) {
            debugStages.push_back(&stageBlur_);
        }
        debugStages.push_back(&stageFinal_);

        if (!debugStages.empty()) {
            const size_t stageCount = debugStages.size();
            const size_t columns = static_cast<size_t>(std::ceil(std::sqrt(static_cast<float>(stageCount))));
            const size_t rows = (stageCount + columns - 1) / columns;

            const UINT compositeWidth = width * static_cast<UINT>(columns);
            const UINT compositeHeight = height * static_cast<UINT>(rows);
            const UINT compositeStride = compositeWidth * 4;
            compositeBuffer_.assign(static_cast<size_t>(compositeStride) * compositeHeight, 0);

            const int frameWidthInt = static_cast<int>(width);
            const int frameHeightInt = static_cast<int>(height);
            const float dominantExtent = static_cast<float>(std::max(width, height));
            const int paddingCandidate = static_cast<int>(std::roundf(dominantExtent * 0.05f));
            const int maxPaddingWidth = std::max(0, (frameWidthInt - 1) / 2);
            const int maxPaddingHeight = std::max(0, (frameHeightInt - 1) / 2);
            const int padding = std::clamp(paddingCandidate, 0, std::min(maxPaddingWidth, maxPaddingHeight));
            const UINT paddingU = static_cast<UINT>(padding);
            const UINT innerWidth = static_cast<UINT>(std::max(1, frameWidthInt - 2 * padding));
            const UINT innerHeight = static_cast<UINT>(std::max(1, frameHeightInt - 2 * padding));

            auto blitScaled = [&](const std::vector<uint8_t>& src, UINT destX, UINT destY) {
                if (innerWidth == 0 || innerHeight == 0) {
                    return;
                }

                const UINT stride = width * 4;
                const float scaleX = static_cast<float>(width) / static_cast<float>(innerWidth);
                const float scaleY = static_cast<float>(height) / static_cast<float>(innerHeight);

                for (UINT y = 0; y < innerHeight; ++y) {
                    const UINT srcY = std::min(height - 1, static_cast<UINT>(std::lroundf(static_cast<float>(y) * scaleY)));
                    uint8_t* dstRow = compositeBuffer_.data() +
                                      (static_cast<size_t>(destY + paddingU + y) * compositeStride) +
                                      (destX + paddingU) * 4;
                    const uint8_t* srcRow = src.data() + static_cast<size_t>(srcY) * stride;
                    for (UINT x = 0; x < innerWidth; ++x) {
                        const UINT srcX = std::min(width - 1, static_cast<UINT>(std::lroundf(static_cast<float>(x) * scaleX)));
                        const uint8_t* srcPixel = srcRow + srcX * 4;
                        uint8_t* dstPixel = dstRow + x * 4;
                        dstPixel[0] = srcPixel[0];
                        dstPixel[1] = srcPixel[1];
                        dstPixel[2] = srcPixel[2];
                        dstPixel[3] = srcPixel[3];
                    }
                }
            };

            for (size_t index = 0; index < stageCount; ++index) {
                const size_t row = index / columns;
                const size_t column = index % columns;
                const UINT destX = static_cast<UINT>(column) * width;
                const UINT destY = static_cast<UINT>(row) * height;
                const auto* stage = debugStages[index];
                if (stage && !stage->empty()) {
                    blitScaled(*stage, destX, destY);
                }
            }

            displayData = compositeBuffer_.data();
            displayWidth = compositeWidth;
            displayHeight = compositeHeight;
        } else {
            displayData = stageFinal_.data();
            displayWidth = width;
            displayHeight = height;
        }
    } else {
        displayData = stageFinal_.data();
        displayWidth = width;
        displayHeight = height;
    }

    const bool cropToFill = !debugViewEnabled_;
    const float centerX = cropToFill ? zoomCenterX_ : 0.5f;
    const float centerY = cropToFill ? zoomCenterY_ : 0.5f;
    PresentFitted(displayData, displayWidth, displayHeight, cropToFill, centerX, centerY);
}

void OpenZoomApp::PresentFitted(const uint8_t* data,
                                UINT srcWidth,
                                UINT srcHeight,
                                bool cropToFill,
                                float centerXNorm,
                                float centerYNorm) {
    if (!data || srcWidth == 0 || srcHeight == 0) {
        return;
    }

    if (!renderWidget_ || !renderWidget_->isPresenterReady()) {
        return;
    }

    const int targetWidthInt = std::max(1, renderWidget_->width());
    const int targetHeightInt = std::max(1, renderWidget_->height());
    const UINT targetWidth = static_cast<UINT>(targetWidthInt);
    const UINT targetHeight = static_cast<UINT>(targetHeightInt);

    presentationBuffer_.assign(static_cast<size_t>(targetWidth) * targetHeight * 4, 0);

    const float srcWidthF = static_cast<float>(srcWidth);
    const float srcHeightF = static_cast<float>(srcHeight);
    const float targetAspect = static_cast<float>(targetWidth) / static_cast<float>(targetHeight);
    const float srcAspect = srcWidthF / srcHeightF;

    float cropWidth = srcWidthF;
    float cropHeight = srcHeightF;

    if (cropToFill) {
        if (targetAspect > srcAspect) {
            cropHeight = srcWidthF / targetAspect;
            cropHeight = std::min(cropHeight, srcHeightF);
        } else {
            cropWidth = srcHeightF * targetAspect;
            cropWidth = std::min(cropWidth, srcWidthF);
        }
    }

    cropWidth = std::clamp(cropWidth, 1.0f, srcWidthF);
    cropHeight = std::clamp(cropHeight, 1.0f, srcHeightF);

    float centerX = std::clamp(centerXNorm, 0.0f, 1.0f) * (srcWidthF - 1.0f);
    float centerY = std::clamp(centerYNorm, 0.0f, 1.0f) * (srcHeightF - 1.0f);

    const float halfCropWidth = cropWidth * 0.5f;
    const float halfCropHeight = cropHeight * 0.5f;

    const float minCenterX = std::max(0.0f, halfCropWidth - 0.5f);
    const float maxCenterX = std::max(minCenterX, (srcWidthF - 1.0f) - (halfCropWidth - 0.5f));
    const float minCenterY = std::max(0.0f, halfCropHeight - 0.5f);
    const float maxCenterY = std::max(minCenterY, (srcHeightF - 1.0f) - (halfCropHeight - 0.5f));

    if (minCenterX <= maxCenterX) {
        centerX = std::clamp(centerX, minCenterX, maxCenterX);
    } else {
        centerX = (srcWidthF - 1.0f) * 0.5f;
    }

    if (minCenterY <= maxCenterY) {
        centerY = std::clamp(centerY, minCenterY, maxCenterY);
    } else {
        centerY = (srcHeightF - 1.0f) * 0.5f;
    }

    float startX = centerX - halfCropWidth + 0.5f;
    float startY = centerY - halfCropHeight + 0.5f;
    startX = std::clamp(startX, 0.0f, srcWidthF - cropWidth);
    startY = std::clamp(startY, 0.0f, srcHeightF - cropHeight);

    float scaleFactor;
    if (cropToFill) {
        scaleFactor = static_cast<float>(targetWidth) / cropWidth;
    } else {
        const float widthScale = static_cast<float>(targetWidth) / cropWidth;
        const float heightScale = static_cast<float>(targetHeight) / cropHeight;
        scaleFactor = std::min(widthScale, heightScale);
    }

    if (!(scaleFactor > 0.0f) || !std::isfinite(scaleFactor)) {
        scaleFactor = 1.0f;
    }

    UINT activeWidth = static_cast<UINT>(std::roundf(cropWidth * scaleFactor));
    UINT activeHeight = static_cast<UINT>(std::roundf(cropHeight * scaleFactor));
    activeWidth = std::clamp(activeWidth, 1u, targetWidth);
    activeHeight = std::clamp(activeHeight, 1u, targetHeight);

    const UINT offsetX = (targetWidth > activeWidth) ? (targetWidth - activeWidth) / 2 : 0;
    const UINT offsetY = (targetHeight > activeHeight) ? (targetHeight - activeHeight) / 2 : 0;

    const float stepX = cropWidth / static_cast<float>(activeWidth);
    const float stepY = cropHeight / static_cast<float>(activeHeight);
    const UINT srcStride = srcWidth * 4;
    const UINT dstStride = targetWidth * 4;

    for (UINT y = 0; y < activeHeight; ++y) {
        const float sampleY = startY + static_cast<float>(y) * stepY;
        int srcYIndex = static_cast<int>(std::lroundf(sampleY));
        srcYIndex = std::clamp(srcYIndex, 0, static_cast<int>(srcHeight) - 1);
        uint8_t* dstRow = presentationBuffer_.data() +
                          (static_cast<size_t>(offsetY + y) * dstStride) +
                          offsetX * 4;
        const uint8_t* srcRow = data + static_cast<size_t>(srcYIndex) * srcStride;
        for (UINT x = 0; x < activeWidth; ++x) {
            const float sampleX = startX + static_cast<float>(x) * stepX;
            int srcXIndex = static_cast<int>(std::lroundf(sampleX));
            srcXIndex = std::clamp(srcXIndex, 0, static_cast<int>(srcWidth) - 1);
            const uint8_t* srcPixel = srcRow + srcXIndex * 4;
            uint8_t* dstPixel = dstRow + x * 4;
            dstPixel[0] = srcPixel[0];
            dstPixel[1] = srcPixel[1];
            dstPixel[2] = srcPixel[2];
            dstPixel[3] = srcPixel[3];
        }
    }

    if (focusMarkerEnabled_ && cropToFill) {
        const float localX = (centerX - startX) / stepX;
        const float localY = (centerY - startY) / stepY;
        const float markerX = static_cast<float>(offsetX) + localX;
        const float markerY = static_cast<float>(offsetY) + localY;

        auto drawFilledCircle = [&](float cx, float cy, float radius,
                                    uint8_t b, uint8_t g, uint8_t r, uint8_t a) {
            const int minX = std::max(0, static_cast<int>(std::floor(cx - radius)));
            const int maxX = std::min(static_cast<int>(targetWidth) - 1,
                                      static_cast<int>(std::ceil(cx + radius)));
            const int minY = std::max(0, static_cast<int>(std::floor(cy - radius)));
            const int maxY = std::min(static_cast<int>(targetHeight) - 1,
                                      static_cast<int>(std::ceil(cy + radius)));
            const float radiusSq = radius * radius;
            for (int py = minY; py <= maxY; ++py) {
                const float dy = (static_cast<float>(py) + 0.5f) - cy;
                for (int px = minX; px <= maxX; ++px) {
                    const float dx = (static_cast<float>(px) + 0.5f) - cx;
                    if (dx * dx + dy * dy <= radiusSq) {
                        uint8_t* pixel = presentationBuffer_.data() +
                                         (static_cast<size_t>(py) * targetWidth + px) * 4;
                        pixel[0] = b;
                        pixel[1] = g;
                        pixel[2] = r;
                        pixel[3] = a;
                    }
                }
            }
        };

        constexpr float kMarkerRadius = 18.0f;
        drawFilledCircle(markerX, markerY, kMarkerRadius, 0, 0, 255, 255);
        constexpr float kInnerRadius = 6.0f;
        drawFilledCircle(markerX, markerY, kInnerRadius, 255, 255, 255, 255);
    }

    presenter_->Present(presentationBuffer_.data(), targetWidth, targetHeight);
    UpdateProcessingStatusLabel();
}

} // namespace openzoom

#include "app.moc"

#endif // _WIN32
