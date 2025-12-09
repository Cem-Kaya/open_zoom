#ifdef _WIN32

#include "openzoom/capture/media_capture.hpp"

#include <QDebug>

#include <mferror.h>

#include <algorithm>
#include <array>
#include <cwchar>
#include <stdexcept>
#include <utility>

namespace openzoom {

namespace {

void ThrowIfFailed(HRESULT hr, const char* message)
{
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(message) + " (hr=0x" + std::to_string(static_cast<unsigned long>(hr)) + ")");
    }
}

void SafeShutdown(IMFMediaSource* source)
{
    if (source) {
        source->Shutdown();
    }
}

static const GUID kPreferredSubtypes[] = {
    MFVideoFormat_ARGB32,
    MFVideoFormat_RGB32,
    MFVideoFormat_NV12,
    MFVideoFormat_YUY2,
};

} // namespace

MediaCapture::MediaCapture() = default;

MediaCapture::~MediaCapture()
{
    StopCapture();
    Shutdown();
}

bool MediaCapture::Initialize()
{
    return true;
}

void MediaCapture::Shutdown()
{
    StopCapture();
    SafeShutdown(mediaSource_.Get());
    mediaSource_.Reset();
    sourceReader_.Reset();
}

std::vector<CameraDescriptor> MediaCapture::EnumerateCameras()
{
    std::vector<CameraDescriptor> cameras;

    Microsoft::WRL::ComPtr<IMFAttributes> attributes;
    if (FAILED(MFCreateAttributes(attributes.GetAddressOf(), 1))) {
        return cameras;
    }

    HRESULT hr = attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                      MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    if (FAILED(hr)) {
        return cameras;
    }

    IMFActivate** devices = nullptr;
    UINT32 count = 0;
    hr = MFEnumDeviceSources(attributes.Get(), &devices, &count);
    if (FAILED(hr)) {
        return cameras;
    }

    cameras.reserve(count);
    for (UINT32 i = 0; i < count; ++i) {
        CameraDescriptor descriptor;

        WCHAR* friendlyName = nullptr;
        UINT32 friendlyLen = 0;
        if (SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                                                     &friendlyName,
                                                     &friendlyLen))) {
            descriptor.name.assign(friendlyName, friendlyName + friendlyLen);
            CoTaskMemFree(friendlyName);
        }

        WCHAR* symbolicLink = nullptr;
        UINT32 linkLen = 0;
        if (SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                                                     &symbolicLink,
                                                     &linkLen))) {
            descriptor.symbolicLink.assign(symbolicLink, symbolicLink + linkLen);
            CoTaskMemFree(symbolicLink);
        }

        descriptor.activation = devices[i];
        descriptor.activation->AddRef();
        cameras.push_back(std::move(descriptor));
    }

    for (UINT32 i = 0; i < count; ++i) {
        devices[i]->Release();
    }
    CoTaskMemFree(devices);

    std::sort(cameras.begin(), cameras.end(), [](const CameraDescriptor& a, const CameraDescriptor& b) {
        return _wcsicmp(a.name.c_str(), b.name.c_str()) < 0;
    });

    return cameras;
}

bool MediaCapture::StartCapture(const CameraDescriptor& descriptor,
                                FrameCallback callback,
                                GUID preferredSubtype)
{
    StopCapture();

    if (!descriptor.activation) {
        return false;
    }

    try {
        Microsoft::WRL::ComPtr<IMFMediaSource> mediaSource;
        ThrowIfFailed(descriptor.activation->ActivateObject(__uuidof(IMFMediaSource),
                                                            reinterpret_cast<void**>(mediaSource.GetAddressOf())),
                      "ActivateObject");

        Microsoft::WRL::ComPtr<IMFAttributes> readerAttributes;
        ThrowIfFailed(MFCreateAttributes(readerAttributes.GetAddressOf(), 4),
                      "Create reader attributes");
        ThrowIfFailed(readerAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE),
                      "Enable video processing");
        ThrowIfFailed(readerAttributes->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, TRUE),
                      "Disable DXVA");
        ThrowIfFailed(readerAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, FALSE),
                      "Allow converters");

        Microsoft::WRL::ComPtr<IMFSourceReader> reader;
        ThrowIfFailed(MFCreateSourceReaderFromMediaSource(mediaSource.Get(),
                                                          readerAttributes.Get(),
                                                          reader.GetAddressOf()),
                      "Create source reader");

        FrameFormat format;
        if (!ConfigureReader(reader.Get(), preferredSubtype, format)) {
            return false;
        }

        mediaSource_ = std::move(mediaSource);
        sourceReader_ = std::move(reader);
        currentFormat_ = format;

        running_ = true;
        captureThread_ = std::thread(&MediaCapture::CaptureLoop, this, std::move(callback));
        return true;
    } catch (const std::exception& e) {
        qWarning() << "MediaCapture::StartCapture exception:" << e.what();
    } catch (...) {
        qWarning() << "MediaCapture::StartCapture unknown exception";
    }

    StopCapture();
    return false;
}

void MediaCapture::StopCapture()
{
    running_ = false;

    if (sourceReader_) {
        sourceReader_->Flush(MF_SOURCE_READER_ALL_STREAMS);
    }

    if (captureThread_.joinable()) {
        captureThread_.join();
    }

    SafeShutdown(mediaSource_.Get());
    mediaSource_.Reset();
    sourceReader_.Reset();
    currentFormat_ = FrameFormat{};
}

bool MediaCapture::ConfigureReader(IMFSourceReader* reader,
                                   GUID preferredSubtype,
                                   FrameFormat& outFormat)
{
    ThrowIfFailed(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE),
                  "Disable default streams");
    ThrowIfFailed(reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE),
                  "Enable video stream");

    Microsoft::WRL::ComPtr<IMFMediaType> desiredType;
    ThrowIfFailed(MFCreateMediaType(desiredType.GetAddressOf()), "Create media type");
    ThrowIfFailed(desiredType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video),
                  "Set major type");
    ThrowIfFailed(desiredType->SetGUID(MF_MT_SUBTYPE, preferredSubtype),
                  "Set subtype");

    HRESULT hr = reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                             nullptr,
                                             desiredType.Get());

    GUID configuredSubtype = preferredSubtype;
    if (FAILED(hr)) {
        for (const GUID& fallback : kPreferredSubtypes) {
            Microsoft::WRL::ComPtr<IMFMediaType> fallbackType;
            ThrowIfFailed(MFCreateMediaType(fallbackType.GetAddressOf()), "Create fallback media type");
            ThrowIfFailed(fallbackType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video),
                          "Set fallback major type");
            ThrowIfFailed(fallbackType->SetGUID(MF_MT_SUBTYPE, fallback),
                          "Set fallback subtype");

            if (SUCCEEDED(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                      nullptr,
                                                      fallbackType.Get()))) {
                configuredSubtype = fallback;
                break;
            }
        }
    }

    Microsoft::WRL::ComPtr<IMFMediaType> currentType;
    ThrowIfFailed(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                              currentType.GetAddressOf()),
                  "Get current media type");

    GUID subtype = GUID_NULL;
    ThrowIfFailed(currentType->GetGUID(MF_MT_SUBTYPE, &subtype), "Get subtype");

    UINT32 width = 0;
    UINT32 height = 0;
    ThrowIfFailed(MFGetAttributeSize(currentType.Get(), MF_MT_FRAME_SIZE, &width, &height),
                  "Get frame size");

    LONG rawStride = 0;
    if (FAILED(MFGetStrideForBitmapInfoHeader(subtype.Data1, width, &rawStride))) {
        if (IsEqualGUID(subtype, MFVideoFormat_ARGB32) || IsEqualGUID(subtype, MFVideoFormat_RGB32)) {
            rawStride = static_cast<LONG>(width * 4);
        } else if (IsEqualGUID(subtype, MFVideoFormat_NV12)) {
            rawStride = static_cast<LONG>(width);
        } else if (IsEqualGUID(subtype, MFVideoFormat_YUY2)) {
            rawStride = static_cast<LONG>(width * 2);
        } else {
            return false;
        }
    }

    outFormat.subtype = subtype;
    outFormat.width = width;
    outFormat.height = height;
    outFormat.stride = static_cast<UINT>(std::abs(rawStride));
    return true;
}

void MediaCapture::CaptureLoop(FrameCallback callback)
{
    HRESULT coInit = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool threadCom = SUCCEEDED(coInit) || coInit == RPC_E_CHANGED_MODE;
    if (FAILED(coInit) && coInit != RPC_E_CHANGED_MODE) {
        running_ = false;
        return;
    }

    Microsoft::WRL::ComPtr<IMFSourceReader> reader = sourceReader_;
    if (!reader) {
        if (threadCom && coInit == S_OK) {
            CoUninitialize();
        }
        return;
    }

    FrameFormat format = currentFormat_;

    while (running_) {
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

        if (!running_) {
            break;
        }

        if (FAILED(hr)) {
            qWarning() << "ReadSample failed" << Qt::hex << hr;
            running_ = false;
            break;
        }

        if ((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0) {
            running_ = false;
            break;
        }

        if ((flags & MF_SOURCE_READERF_STREAMTICK) != 0 || !sample) {
            continue;
        }

        Microsoft::WRL::ComPtr<IMFMediaBuffer> buffer;
        hr = sample->ConvertToContiguousBuffer(&buffer);
        if (FAILED(hr) || !buffer) {
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

        MediaFrame frame;
        frame.data.assign(data, data + length);
        frame.subtype = format.subtype;
        frame.width = format.width;
        frame.height = format.height;
        frame.stride = format.stride;
        frame.dataSize = frame.data.size();

        buffer->Unlock();

        if (callback) {
            callback(frame);
        }
    }

    if (threadCom && coInit == S_OK) {
        CoUninitialize();
    }
}

} // namespace openzoom

#endif // _WIN32
