#ifdef _WIN32

#include "openzoom/capture/media_capture.hpp"

#include <QDebug>

#include <mferror.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cwchar>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace openzoom {

namespace {

std::string FormatHResult(HRESULT hr)
{
    char code[16]{};
    std::snprintf(code, sizeof(code), "0x%08lX", static_cast<unsigned long>(hr));

    LPSTR systemMessage = nullptr;
    const DWORD length = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                            FORMAT_MESSAGE_FROM_SYSTEM |
                                            FORMAT_MESSAGE_IGNORE_INSERTS,
                                        nullptr,
                                        static_cast<DWORD>(hr),
                                        0,
                                        reinterpret_cast<LPSTR>(&systemMessage),
                                        0,
                                        nullptr);

    std::string result = std::string("hr=") + code;
    if (length != 0 && systemMessage) {
        std::string detail(systemMessage, length);
        while (!detail.empty() && std::isspace(static_cast<unsigned char>(detail.back()))) {
            detail.pop_back();
        }
        if (!detail.empty()) {
            result += ": " + detail;
        }
    }
    if (systemMessage) {
        LocalFree(systemMessage);
    }
    return result;
}

void ThrowIfFailed(HRESULT hr, const char* message)
{
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(message) + " (" + FormatHResult(hr) + ")");
    }
}

void SafeShutdown(IMFMediaSource* source)
{
    if (source) {
        source->Shutdown();
    }
}

class ActivationShutdownGuard {
public:
    explicit ActivationShutdownGuard(IMFActivate* activation)
        : activation_(activation)
    {
    }

    ~ActivationShutdownGuard()
    {
        if (activation_) {
            const HRESULT hr = activation_->ShutdownObject();
            if (FAILED(hr)) {
                qWarning() << "IMFActivate::ShutdownObject failed:"
                           << QString::fromStdString(FormatHResult(hr));
            }
        }
    }

    void Dismiss() { activation_ = nullptr; }

private:
    IMFActivate* activation_{};
};

static const GUID kPreferredSubtypes[] = {
    MFVideoFormat_NV12,
    MFVideoFormat_YUY2,
    MFVideoFormat_ARGB32,
    MFVideoFormat_RGB32,
};

// Busy/resource-class errors that often clear up within a second (device still
// warming up, another app releasing it, phone camera waking). Worth retrying.
bool IsTransientStartError(HRESULT hr)
{
    return hr == MF_E_HW_MFT_FAILED_START_STREAMING ||
           hr == MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED ||
           hr == MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED ||
           hr == E_ACCESSDENIED ||
           hr == HRESULT_FROM_WIN32(ERROR_BUSY) ||
           hr == HRESULT_FROM_WIN32(ERROR_SHARING_VIOLATION) ||
           hr == HRESULT_FROM_WIN32(ERROR_DEVICE_NOT_CONNECTED);
}

CameraFailureKind ClassifyCameraFailure(HRESULT hr)
{
    if (hr == E_ACCESSDENIED) {
        return CameraFailureKind::AccessDenied;
    }
    if (hr == MF_E_HW_MFT_FAILED_START_STREAMING ||
        hr == MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED ||
        hr == HRESULT_FROM_WIN32(ERROR_BUSY) ||
        hr == HRESULT_FROM_WIN32(ERROR_SHARING_VIOLATION)) {
        return CameraFailureKind::DeviceBusy;
    }
    if (hr == MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED ||
        hr == MF_E_SHUTDOWN ||
        hr == HRESULT_FROM_WIN32(ERROR_DEVICE_NOT_CONNECTED) ||
        hr == HRESULT_FROM_WIN32(ERROR_DEVICE_REMOVED) ||
        hr == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND)) {
        return CameraFailureKind::DeviceMissing;
    }
    return CameraFailureKind::Other;
}

// Plain-language message shown directly to the user; technical detail kept in
// parentheses for support/logging.
std::string DescribeCameraFailure(CameraFailureKind kind, HRESULT hr, const char* stage)
{
    switch (kind) {
    case CameraFailureKind::DeviceBusy:
        return std::string("The camera is in use by another app or not streaming yet. "
                           "If this is a phone camera, open Phone Link and make sure the phone is awake. (") +
               stage + ", " + FormatHResult(hr) + ")";
    case CameraFailureKind::DeviceMissing:
        return std::string("The camera was disconnected or could not be found. "
                           "Check that it is plugged in and connected, then try again. (") +
               stage + ", " + FormatHResult(hr) + ")";
    case CameraFailureKind::AccessDenied:
        return std::string("Windows blocked access to the camera. Allow camera access for "
                           "desktop apps in Settings > Privacy & security > Camera. (") +
               stage + ", " + FormatHResult(hr) + ")";
    default:
        return std::string(stage) + " failed (" + FormatHResult(hr) + ")";
    }
}

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

std::vector<VideoFormat> MediaCapture::EnumerateFormats(const CameraDescriptor& descriptor)
{
    lastError_.clear();
    std::vector<VideoFormat> formats;

    if (!descriptor.activation) {
        lastError_ = "Invalid camera activation";
        return formats;
    }

    try {
        Microsoft::WRL::ComPtr<IMFMediaSource> mediaSource;
        ThrowIfFailed(descriptor.activation->ActivateObject(__uuidof(IMFMediaSource),
                                                            reinterpret_cast<void**>(mediaSource.GetAddressOf())),
                      "ActivateObject");
        ActivationShutdownGuard activationGuard(descriptor.activation.Get());

        Microsoft::WRL::ComPtr<IMFAttributes> readerAttributes;
        ThrowIfFailed(MFCreateAttributes(readerAttributes.GetAddressOf(), 2),
                      "Create reader attributes");
        ThrowIfFailed(readerAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE),
                      "Enable video processing");
        ThrowIfFailed(readerAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, FALSE),
                      "Allow converters");

        Microsoft::WRL::ComPtr<IMFSourceReader> reader;
        ThrowIfFailed(MFCreateSourceReaderFromMediaSource(mediaSource.Get(),
                                                          readerAttributes.Get(),
                                                          reader.GetAddressOf()),
                      "Create source reader");

        formats = ExtractFormats(reader.Get());
    } catch (const std::exception& e) {
        lastError_ = e.what();
    } catch (...) {
        lastError_ = "Unknown exception while enumerating formats";
    }

    return formats;
}

bool MediaCapture::StartCapture(const CameraDescriptor& descriptor,
                                FrameCallback callback,
                                GUID preferredSubtype,
                                CaptureErrorCallback errorCallback)
{
    StopCapture();
    lastError_.clear();
    lastFailureKind_.store(CameraFailureKind::None);
    deviceLost_.store(false);

    if (!descriptor.activation) {
        lastError_ = "Invalid camera activation";
        lastFailureKind_.store(CameraFailureKind::DeviceMissing);
        return false;
    }

    try {
        Microsoft::WRL::ComPtr<IMFMediaSource> mediaSource;
        Microsoft::WRL::ComPtr<IMFSourceReader> reader;

        // Retry transient (busy/resource) failures up to 3 times; fail fast on
        // structural errors such as unsupported formats.
        static constexpr DWORD kRetryDelaysMs[] = {150, 300, 600};
        const char* failedStage = "ActivateObject";
        HRESULT hr = TryOpenDevice(descriptor, mediaSource, reader, failedStage);
        for (const DWORD delayMs : kRetryDelaysMs) {
            if (SUCCEEDED(hr) || !IsTransientStartError(hr)) {
                break;
            }
            qWarning() << "Camera start failed with transient error"
                       << QString::fromStdString(FormatHResult(hr))
                       << "- retrying in" << delayMs << "ms";
            ::Sleep(delayMs);
            hr = TryOpenDevice(descriptor, mediaSource, reader, failedStage);
        }

        if (FAILED(hr)) {
            const CameraFailureKind kind = ClassifyCameraFailure(hr);
            lastFailureKind_.store(kind);
            lastError_ = DescribeCameraFailure(kind, hr, failedStage);
            return false;
        }

        ActivationShutdownGuard activationGuard(descriptor.activation.Get());

        FrameFormat format;
        if (!ConfigureReader(reader.Get(), preferredSubtype, format)) {
            lastError_ = "ConfigureReader failed to select format";
            lastFailureKind_.store(CameraFailureKind::Other);
            return false;
        }

        mediaSource_ = std::move(mediaSource);
        sourceReader_ = std::move(reader);
        activeActivation_ = descriptor.activation;
        currentFormat_ = format;
        frameRateNumerator_.store(format.frameRateNumerator);
        frameRateDenominator_.store(format.frameRateDenominator);
        lastSymbolicLink_ = descriptor.symbolicLink;
        activationGuard.Dismiss();

        running_ = true;
        captureThread_ = std::thread(&MediaCapture::CaptureLoop,
                                     this,
                                     std::move(callback),
                                     std::move(errorCallback));
        return true;
    } catch (const std::exception& e) {
        lastError_ = e.what();
        lastFailureKind_.store(CameraFailureKind::Other);
        qWarning() << "MediaCapture::StartCapture exception:" << e.what();
    } catch (...) {
        lastError_ = "Unknown exception while starting capture";
        lastFailureKind_.store(CameraFailureKind::Other);
        qWarning() << "MediaCapture::StartCapture unknown exception";
    }

    StopCapture();
    return false;
}

HRESULT MediaCapture::TryOpenDevice(const CameraDescriptor& descriptor,
                                    Microsoft::WRL::ComPtr<IMFMediaSource>& outSource,
                                    Microsoft::WRL::ComPtr<IMFSourceReader>& outReader,
                                    const char*& failedStage)
{
    outSource.Reset();
    outReader.Reset();

    Microsoft::WRL::ComPtr<IMFMediaSource> mediaSource;
    HRESULT hr = descriptor.activation->ActivateObject(__uuidof(IMFMediaSource),
                                                       reinterpret_cast<void**>(mediaSource.GetAddressOf()));
    if (FAILED(hr)) {
        failedStage = "ActivateObject";
        return hr;
    }
    ActivationShutdownGuard activationGuard(descriptor.activation.Get());

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
    hr = MFCreateSourceReaderFromMediaSource(mediaSource.Get(),
                                             readerAttributes.Get(),
                                             reader.GetAddressOf());
    if (FAILED(hr)) {
        failedStage = "Create source reader";
        // The guard shuts the activation down so a retry can reopen the device.
        return hr;
    }

    activationGuard.Dismiss();
    outSource = std::move(mediaSource);
    outReader = std::move(reader);
    return S_OK;
}

bool MediaCapture::ConsumeDeviceLost()
{
    return deviceLost_.exchange(false);
}

double MediaCapture::CurrentFrameRate() const
{
    const UINT denominator = frameRateDenominator_.load();
    return denominator == 0
               ? 0.0
               : static_cast<double>(frameRateNumerator_.load()) /
                     static_cast<double>(denominator);
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

    sourceReader_.Reset();
    if (activeActivation_) {
        const HRESULT hr = activeActivation_->ShutdownObject();
        if (FAILED(hr)) {
            qWarning() << "IMFActivate::ShutdownObject failed:"
                       << QString::fromStdString(FormatHResult(hr));
            SafeShutdown(mediaSource_.Get());
        }
    } else {
        SafeShutdown(mediaSource_.Get());
    }
    mediaSource_.Reset();
    activeActivation_.Reset();
    currentFormat_ = FrameFormat{};
    frameRateNumerator_.store(0);
    frameRateDenominator_.store(0);
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
    bool typeSelected = SUCCEEDED(hr);

    if (!typeSelected) {
        for (const GUID& fallback : kPreferredSubtypes) {
            if (IsEqualGUID(fallback, preferredSubtype)) {
                continue;
            }
            Microsoft::WRL::ComPtr<IMFMediaType> fallbackType;
            ThrowIfFailed(MFCreateMediaType(fallbackType.GetAddressOf()), "Create fallback media type");
            ThrowIfFailed(fallbackType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video),
                          "Set fallback major type");
            ThrowIfFailed(fallbackType->SetGUID(MF_MT_SUBTYPE, fallback),
                          "Set fallback subtype");

            if (SUCCEEDED(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                      nullptr,
                                                      fallbackType.Get()))) {
                typeSelected = true;
                break;
            }
        }
    }

    return typeSelected && ReadCurrentFormat(reader, outFormat);
}

bool MediaCapture::ReadCurrentFormat(IMFSourceReader* reader, FrameFormat& outFormat)
{
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
    UINT32 frameRateNumerator = 0;
    UINT32 frameRateDenominator = 0;
    if (SUCCEEDED(MFGetAttributeRatio(currentType.Get(),
                                      MF_MT_FRAME_RATE,
                                      &frameRateNumerator,
                                      &frameRateDenominator))) {
        outFormat.frameRateNumerator = frameRateNumerator;
        outFormat.frameRateDenominator = frameRateDenominator;
    }
    return true;
}

std::vector<VideoFormat> MediaCapture::ExtractFormats(IMFSourceReader* reader)
{
    std::vector<VideoFormat> formats;
    if (!reader) {
        return formats;
    }

    std::set<std::tuple<UINT, UINT, UINT, UINT>> uniqueKeys;
    for (DWORD index = 0;; ++index) {
        Microsoft::WRL::ComPtr<IMFMediaType> mediaType;
        HRESULT hr = reader->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, index, mediaType.GetAddressOf());
        if (hr == MF_E_NO_MORE_TYPES) {
            break;
        }
        if (FAILED(hr) || !mediaType) {
            lastError_ = HrToString(hr);
            break;
        }

        UINT32 width = 0, height = 0;
        if (FAILED(MFGetAttributeSize(mediaType.Get(), MF_MT_FRAME_SIZE, &width, &height))) {
            continue;
        }

        UINT32 num = 0, den = 0;
        if (FAILED(MFGetAttributeRatio(mediaType.Get(), MF_MT_FRAME_RATE, &num, &den))) {
            num = 0;
            den = 0;
        }

        auto key = std::make_tuple(width, height, num, den);
        if (uniqueKeys.insert(key).second) {
            VideoFormat fmt;
            fmt.width = width;
            fmt.height = height;
            fmt.numerator = num;
            fmt.denominator = den;
            formats.push_back(fmt);
        }
    }
    return formats;
}

std::string MediaCapture::HrToString(HRESULT hr)
{
    return FormatHResult(hr);
}

void MediaCapture::CaptureLoop(FrameCallback callback, CaptureErrorCallback errorCallback)
{
    HRESULT coInit = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool shouldUninitializeCom = SUCCEEDED(coInit);
    auto reportFailure = [&errorCallback](const std::string& message) {
        qWarning() << "Camera capture stopped:" << QString::fromStdString(message);
        if (errorCallback) {
            try {
                errorCallback(message);
            } catch (const std::exception& e) {
                qWarning() << "Capture error callback failed:" << e.what();
            } catch (...) {
                qWarning() << "Capture error callback failed with an unknown exception";
            }
        }
    };

    if (FAILED(coInit) && coInit != RPC_E_CHANGED_MODE) {
        running_ = false;
        reportFailure(std::string("Capture thread COM initialization failed (") +
                      FormatHResult(coInit) + ")");
        return;
    }

    Microsoft::WRL::ComPtr<IMFSourceReader> reader = sourceReader_;
    if (!reader) {
        if (shouldUninitializeCom) {
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
            running_ = false;
            // Classify and flag device loss so the app's frame tick can poll
            // ConsumeDeviceLost() and drive reconnection.
            const CameraFailureKind kind = ClassifyCameraFailure(hr);
            lastFailureKind_.store(kind);
            deviceLost_.store(true);
            reportFailure(DescribeCameraFailure(kind, hr, "ReadSample"));
            break;
        }

        if ((flags & MF_SOURCE_READERF_ERROR) != 0) {
            running_ = false;
            lastFailureKind_.store(CameraFailureKind::DeviceMissing);
            deviceLost_.store(true);
            reportFailure("The camera reported a stream error and stopped delivering frames");
            break;
        }

        if ((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0) {
            running_ = false;
            lastFailureKind_.store(CameraFailureKind::DeviceMissing);
            deviceLost_.store(true);
            reportFailure("The camera stream ended unexpectedly");
            break;
        }

        if ((flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) != 0) {
            // Mid-stream format change (S7): re-query the current media type and
            // refresh the cached format so we never copy frames with a stale stride.
            try {
                FrameFormat updatedFormat;
                if (!ReadCurrentFormat(reader.Get(), updatedFormat)) {
                    running_ = false;
                    reportFailure("The camera changed to an unsupported media format");
                    break;
                }
                format = updatedFormat;
                currentFormat_ = updatedFormat;
                frameRateNumerator_.store(updatedFormat.frameRateNumerator);
                frameRateDenominator_.store(updatedFormat.frameRateDenominator);
            } catch (const std::exception& e) {
                running_ = false;
                reportFailure(std::string("Failed to read the camera's changed media format: ") + e.what());
                break;
            }
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
            try {
                callback(std::move(frame));
            } catch (const std::exception& e) {
                running_ = false;
                reportFailure(std::string("Frame callback failed: ") + e.what());
                break;
            } catch (...) {
                running_ = false;
                reportFailure("Frame callback failed with an unknown exception");
                break;
            }
        }
    }

    if (shouldUninitializeCom) {
        CoUninitialize();
    }
}

} // namespace openzoom

#endif // _WIN32
