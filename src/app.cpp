#ifdef _WIN32

#include "openzoom/app.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <QString>
#include <QSizePolicy>
#include <QPaintEngine>
#include <QResizeEvent>
#include <QShowEvent>

#include <windows.h>
#include <combaseapi.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <shlwapi.h>

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
               float zoomAmount)
{
    const float zoom = std::max(1.0f, zoomAmount);
    const UINT stride = width * 4;
    if (dst.size() != src.size()) {
        dst.resize(src.size());
    }

    const float cx = static_cast<float>(width) * 0.5f;
    const float cy = static_cast<float>(height) * 0.5f;

    for (UINT y = 0; y < height; ++y) {
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * stride;
        for (UINT x = 0; x < width; ++x) {
            const float sx = (static_cast<float>(x) - cx) / zoom + cx;
            const float sy = (static_cast<float>(y) - cy) / zoom + cy;

            int sampleX = static_cast<int>(std::roundf(sx));
            int sampleY = static_cast<int>(std::roundf(sy));
            if (sampleX < 0 || sampleX >= static_cast<int>(width) ||
                sampleY < 0 || sampleY >= static_cast<int>(height)) {
                dstRow[x * 4 + 0] = 0;
                dstRow[x * 4 + 1] = 0;
                dstRow[x * 4 + 2] = 0;
                dstRow[x * 4 + 3] = 255;
            } else {
                const uint8_t* srcPixel = src.data() + static_cast<size_t>(sampleY) * stride + sampleX * 4;
                uint8_t* dstPixel = dstRow + x * 4;
                dstPixel[0] = srcPixel[0];
                dstPixel[1] = srcPixel[1];
                dstPixel[2] = srcPixel[2];
                dstPixel[3] = srcPixel[3];
            }
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
        for (UINT adapterIndex = 0;
             factory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND;
             ++adapterIndex) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                continue;
            }
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
                break;
            }
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
        ThrowIfFailed(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
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

class MainWindow : public QMainWindow {
public:
    MainWindow() {
        setWindowTitle("OpenZoom");
        resize(1280, 720);

        auto* central = new QWidget(this);
        auto* rootLayout = new QVBoxLayout(central);
        rootLayout->setContentsMargins(12, 12, 12, 12);
        rootLayout->setSpacing(8);

        auto* cameraLayout = new QHBoxLayout();
        cameraLayout->setSpacing(8);
        auto* cameraLabel = new QLabel("Camera:");
        cameraCombo_ = new QComboBox();
        cameraCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        cameraLayout->addWidget(cameraLabel);
        cameraLayout->addWidget(cameraCombo_, 1);
        rootLayout->addLayout(cameraLayout);

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
        rootLayout->addLayout(bwLayout);

        auto* zoomLayout = new QHBoxLayout();
        zoomLayout->setSpacing(8);
        zoomCheckbox_ = new QCheckBox("Zoom");
        zoomSlider_ = new QSlider(Qt::Horizontal);
        zoomSlider_->setRange(kZoomSliderScale, 4 * kZoomSliderScale);
        zoomSlider_->setPageStep(10);
        zoomSlider_->setValue(kZoomSliderScale);
        zoomSlider_->setEnabled(false);
        zoomLayout->addWidget(zoomCheckbox_);
        zoomLayout->addWidget(zoomSlider_, 1);
        rootLayout->addLayout(zoomLayout);

        renderWidget_ = new RenderWidget();
        rootLayout->addWidget(renderWidget_, 1);

        setCentralWidget(central);
    }

    RenderWidget* renderWidget() const { return renderWidget_; }
    QComboBox* cameraCombo() const { return cameraCombo_; }
    QCheckBox* blackWhiteCheckbox() const { return bwCheckbox_; }
    QSlider* blackWhiteSlider() const { return bwSlider_; }
    QCheckBox* zoomCheckbox() const { return zoomCheckbox_; }
    QSlider* zoomSlider() const { return zoomSlider_; }

private:
    RenderWidget* renderWidget_{};
    QComboBox* cameraCombo_{};
    QCheckBox* bwCheckbox_{};
    QSlider* bwSlider_{};
    QCheckBox* zoomCheckbox_{};
    QSlider* zoomSlider_{};
};


OpenZoomApp::OpenZoomApp(int& argc, char** argv)
    : QObject(nullptr) {
    qtApp_ = new QApplication(argc, argv);
    InitializePlatform();

    presenter_ = std::make_unique<D3D12Presenter>();

    mainWindow_ = std::make_unique<MainWindow>();
    renderWidget_ = mainWindow_->renderWidget();
    renderWidget_->setPresenter(presenter_.get());
    cameraCombo_ = mainWindow_->cameraCombo();
    bwCheckbox_ = mainWindow_->blackWhiteCheckbox();
    bwSlider_ = mainWindow_->blackWhiteSlider();
    zoomCheckbox_ = mainWindow_->zoomCheckbox();
    zoomSlider_ = mainWindow_->zoomSlider();

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

void OpenZoomApp::StartCameraCapture(size_t index) {
    if (index >= cameras_.size()) {
        return;
    }

    selectedCameraIndex_ = static_cast<int>(index);
    StopCameraCapture();

    CameraInfo& camera = cameras_[index];
    if (!camera.activation) {
        return;
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
    stageBw_ = stageRaw_;
    if (blackWhiteEnabled_) {
        ApplyBlackWhite(stageRaw_, stageBw_, blackWhiteThreshold_);
    }

    const std::vector<uint8_t>& zoomSource = blackWhiteEnabled_ ? stageBw_ : stageRaw_;
    stageZoom_ = zoomSource;
    if (zoomEnabled_) {
        ApplyZoom(zoomSource, stageZoom_, width, height, zoomAmount_);
    }

    if (zoomEnabled_) {
        stageFinal_ = stageZoom_;
    } else if (blackWhiteEnabled_) {
        stageFinal_ = stageBw_;
    } else {
        stageFinal_ = stageRaw_;
    }

    const UINT compositeWidth = width * 2;
    const UINT compositeHeight = height * 2;
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

    blitScaled(stageRaw_, 0, 0);
    blitScaled(stageBw_, width, 0);
    blitScaled(stageZoom_, 0, height);
    blitScaled(stageFinal_, width, height);

    if (renderWidget_ && renderWidget_->isPresenterReady()) {
        presenter_->Present(compositeBuffer_.data(), compositeWidth, compositeHeight);
    }
}

} // namespace openzoom

#endif // _WIN32
