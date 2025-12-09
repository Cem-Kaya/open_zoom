#ifdef _WIN32

#include "openzoom/d3d12/presenter.hpp"

#include <QDebug>

#include <dxgi1_6.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>

namespace openzoom {

namespace {

inline void ThrowIfFailed(HRESULT hr, const char* message)
{
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(message) + " (hr=0x" + std::to_string(static_cast<unsigned long>(hr)) + ")");
    }
}

inline D3D12_RESOURCE_BARRIER TransitionBarrier(ID3D12Resource* resource,
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

D3D12Presenter::D3D12Presenter() = default;

D3D12Presenter::~D3D12Presenter()
{
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

void D3D12Presenter::Initialize(HWND hwnd, UINT width, UINT height)
{
    hwnd_ = hwnd;
    CreateDevice();
    CreateCommandObjects();
    CreateFenceObjects();
    CreateSwapChain(width, height);
    EnsureUploadBuffer(width, height);
    initialized_ = true;
}

bool D3D12Presenter::IsInitialized() const
{
    return initialized_;
}

void D3D12Presenter::Resize(UINT width, UINT height)
{
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

void D3D12Presenter::Present(const uint8_t* data, UINT width, UINT height)
{
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

void D3D12Presenter::PresentFromTexture(ID3D12Resource* texture,
                                        UINT width,
                                        UINT height,
                                        const FenceSyncParams* fenceSync)
{
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

    if (useFenceSync) {
        ThrowIfFailed(commandQueue_->Signal(fence_.Get(), fenceSync->signalValue),
                      "Failed to signal shared fence");
        fenceValue_ = fenceSync->signalValue;
        WaitForFenceValue(fenceSync->signalValue);
    } else {
        WaitForGpu();
    }

    ThrowIfFailed(swapChain_->Present(1, 0), "Failed to present swap chain");
}

ID3D12Device* D3D12Presenter::GetDevice() const
{
    return device_.Get();
}

ID3D12Fence* D3D12Presenter::GetFence() const
{
    return fence_.Get();
}

UINT64 D3D12Presenter::GetLastSignaledFenceValue() const
{
    return fenceValue_;
}

void D3D12Presenter::CreateDevice()
{
    UINT dxgiFactoryFlags = 0;
#if defined(_DEBUG)
    Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
        debugController->EnableDebugLayer();
        dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }
#endif
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
    ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)),
                  "Failed to create DXGI factory");
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

void D3D12Presenter::CreateCommandObjects()
{
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

void D3D12Presenter::CreateFenceObjects()
{
    ThrowIfFailed(device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence_)),
                  "Failed to create fence");
    fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent_) {
        ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()), "Failed to create fence event");
    }
}

void D3D12Presenter::CreateSwapChain(UINT width, UINT height)
{
    DXGI_SWAP_CHAIN_DESC1 scDesc{};
    scDesc.Width = width;
    scDesc.Height = height;
    scDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scDesc.SampleDesc.Count = 1;
    scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scDesc.BufferCount = 2;
    scDesc.Scaling = DXGI_SCALING_STRETCH;
    scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
    ThrowIfFailed(factory_->CreateSwapChainForHwnd(commandQueue_.Get(),
                                                   hwnd_,
                                                   &scDesc,
                                                   nullptr,
                                                   nullptr,
                                                   &swapChain1),
                  "Failed to create swap chain");

    ThrowIfFailed(factory_->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER),
                  "Failed to disable Alt-Enter");

    ThrowIfFailed(swapChain1.As(&swapChain_), "Failed to query IDXGISwapChain3");

    backBuffers_.resize(scDesc.BufferCount);
    AcquireBackBuffers();
    width_ = width;
    height_ = height;
}

void D3D12Presenter::AcquireBackBuffers()
{
    for (UINT i = 0; i < backBuffers_.size(); ++i) {
        ThrowIfFailed(swapChain_->GetBuffer(i, IID_PPV_ARGS(&backBuffers_[i])),
                      "Failed to retrieve swap chain buffer");
    }
}

void D3D12Presenter::EnsureUploadBuffer(UINT width, UINT height)
{
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
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

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
    bufferDesc.Width = uploadTotalBytes_;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

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

void D3D12Presenter::CopyToUpload(const uint8_t* data, UINT width, UINT height)
{
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

void D3D12Presenter::WaitForGpu()
{
    const UINT64 fenceValue = ++fenceValue_;
    ThrowIfFailed(commandQueue_->Signal(fence_.Get(), fenceValue), "Failed to signal fence");
    if (fence_->GetCompletedValue() < fenceValue) {
        ThrowIfFailed(fence_->SetEventOnCompletion(fenceValue, fenceEvent_),
                      "Failed to set fence completion event");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
}

void D3D12Presenter::WaitForFenceValue(UINT64 value)
{
    if (!fence_) {
        return;
    }
    if (fence_->GetCompletedValue() < value) {
        ThrowIfFailed(fence_->SetEventOnCompletion(value, fenceEvent_),
                      "Failed to set fence completion event");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
}

} // namespace openzoom

#endif // _WIN32

