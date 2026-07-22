#ifdef _WIN32

#include "openzoom/common/media_writer.hpp"

#include <mferror.h>
#include <propvarutil.h>

#include <wrl/client.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace openzoom {

namespace {

std::string HrToString(HRESULT hr)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "hr=0x%08lx", static_cast<unsigned long>(hr));
    return std::string(buffer);
}

void ThrowIfFailed(HRESULT hr, const char* msg)
{
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(msg) + " (" + HrToString(hr) + ")");
    }
}

constexpr ULONGLONG kMinFreeBytesToStart = 500ull * 1024ull * 1024ull;
constexpr ULONGLONG kMinFreeBytesWhileRecording = 200ull * 1024ull * 1024ull;
constexpr ULONGLONG kSpaceCheckIntervalMs = 5000;

const char kDiskFullMessage[] = "Recording stopped: disk almost full. The recording so far was saved.";

bool QueryFreeBytesForPath(const std::wstring& filePath, ULONGLONG& outFreeBytes)
{
    std::wstring directory = filePath;
    const size_t separator = directory.find_last_of(L"\\/");
    if (separator == std::wstring::npos) {
        directory.clear();
    } else {
        directory.erase(separator + 1);
    }

    ULARGE_INTEGER freeBytesAvailable{};
    if (!GetDiskFreeSpaceExW(directory.empty() ? nullptr : directory.c_str(),
                             &freeBytesAvailable,
                             nullptr,
                             nullptr)) {
        return false;
    }
    outFreeBytes = freeBytesAvailable.QuadPart;
    return true;
}

bool IsDiskFullError(HRESULT hr)
{
    return hr == HRESULT_FROM_WIN32(ERROR_DISK_FULL) ||
           hr == HRESULT_FROM_WIN32(ERROR_HANDLE_DISK_FULL) ||
           hr == STG_E_MEDIUMFULL;
}

} // namespace

VideoRecorder::VideoRecorder() = default;

VideoRecorder::~VideoRecorder()
{
    Stop();
}

void VideoRecorder::SetError(const std::string& err)
{
    lastError_ = err;
}

bool VideoRecorder::Start(const std::wstring& filePath, UINT width, UINT height, UINT fps)
{
    Stop();

    ULONGLONG freeBytes = 0;
    if (QueryFreeBytesForPath(filePath, freeBytes) && freeBytes < kMinFreeBytesToStart) {
        SetError("Not enough free disk space to start recording (less than 500 MB available). "
                 "Free up some space and try again.");
        return false;
    }

    try {
        if (!InitializeSink(filePath, width, height, fps)) {
            return false;
        }
        frameWidth_ = width;
        frameHeight_ = height;
        fps_ = std::max(1u, fps);
        rtStart_ = 0;
        rtDuration_ = 10'000'000LL / static_cast<LONGLONG>(fps_);
        targetPath_ = filePath;
        lastSpaceCheckTicks_ = GetTickCount64();
        stopReason_ = StopReason::None;
        recording_ = true;
        lastError_.clear();
        return true;
    } catch (const std::exception& e) {
        SetError(e.what());
    } catch (...) {
        SetError("Unknown exception starting recorder");
    }
    Stop();
    return false;
}

void VideoRecorder::Stop()
{
    FinalizeAndStop(StopReason::Manual);
}

void VideoRecorder::FinalizeAndStop(StopReason reason)
{
    if (sinkWriter_) {
        // Finalize flushes the trailing fragment; already-written fragments are
        // valid on disk regardless.
        sinkWriter_->Finalize();
    }
    sinkWriter_.Reset();
    if (recording_) {
        stopReason_ = reason;
    }
    recording_ = false;
    rtStart_ = 0;
}

bool VideoRecorder::InitializeSink(const std::wstring& filePath, UINT width, UINT height, UINT fps)
{
    Microsoft::WRL::ComPtr<IMFAttributes> attrs;
    ThrowIfFailed(MFCreateAttributes(attrs.GetAddressOf(), 3), "Create sink attributes");
    attrs->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
    attrs->SetUINT32(MF_SINK_WRITER_DISABLE_THROTTLING, TRUE);
    // Fragmented MP4: the file is playable up to the last completed fragment
    // even if the process dies mid-recording (no moov finalize needed).
    ThrowIfFailed(attrs->SetGUID(MF_TRANSCODE_CONTAINERTYPE, MFTranscodeContainerType_FMPEG4),
                  "Set fragmented MP4 container");

    Microsoft::WRL::ComPtr<IMFSinkWriter> writer;
    ThrowIfFailed(MFCreateSinkWriterFromURL(filePath.c_str(), nullptr, attrs.Get(), writer.GetAddressOf()),
                  "Create sink writer");

    // Output type (H.264 in MP4)
    Microsoft::WRL::ComPtr<IMFMediaType> outType;
    ThrowIfFailed(MFCreateMediaType(outType.GetAddressOf()), "Create output type");
    ThrowIfFailed(outType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "Set out major type");
    ThrowIfFailed(outType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264), "Set out subtype");
    ThrowIfFailed(outType->SetUINT32(MF_MT_AVG_BITRATE, width * height * 5), "Set bitrate"); // rough default
    ThrowIfFailed(outType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive), "Set interlace");
    ThrowIfFailed(MFSetAttributeSize(outType.Get(), MF_MT_FRAME_SIZE, width, height), "Set out frame size");
    ThrowIfFailed(MFSetAttributeRatio(outType.Get(), MF_MT_FRAME_RATE, fps, 1), "Set out frame rate");
    ThrowIfFailed(MFSetAttributeRatio(outType.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1), "Set PAR");

    ThrowIfFailed(writer->AddStream(outType.Get(), &streamIndex_), "Add stream");

    // Input type (BGRA)
    Microsoft::WRL::ComPtr<IMFMediaType> inType;
    ThrowIfFailed(MFCreateMediaType(inType.GetAddressOf()), "Create input type");
    ThrowIfFailed(inType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "Set in major type");
    ThrowIfFailed(inType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_ARGB32), "Set in subtype");
    ThrowIfFailed(inType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive), "Set in interlace");
    ThrowIfFailed(MFSetAttributeSize(inType.Get(), MF_MT_FRAME_SIZE, width, height), "Set in frame size");
    ThrowIfFailed(MFSetAttributeRatio(inType.Get(), MF_MT_FRAME_RATE, fps, 1), "Set in frame rate");
    ThrowIfFailed(MFSetAttributeRatio(inType.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1), "Set PAR");

    ThrowIfFailed(writer->SetInputMediaType(streamIndex_, inType.Get(), nullptr), "Set input media type");
    ThrowIfFailed(writer->BeginWriting(), "BeginWriting");

    sinkWriter_ = writer;
    return true;
}

bool VideoRecorder::AddFrame(const uint8_t* bgraData, size_t strideBytes)
{
    if (!recording_ || !sinkWriter_ || !bgraData) {
        return false;
    }

    // Periodic disk-space guard: stop cleanly before the volume runs dry so the
    // fragmented MP4 written so far stays intact.
    const ULONGLONG nowTicks = GetTickCount64();
    if (nowTicks - lastSpaceCheckTicks_ >= kSpaceCheckIntervalMs) {
        lastSpaceCheckTicks_ = nowTicks;
        ULONGLONG freeBytes = 0;
        if (QueryFreeBytesForPath(targetPath_, freeBytes) && freeBytes < kMinFreeBytesWhileRecording) {
            FinalizeAndStop(StopReason::DiskFull);
            SetError(kDiskFullMessage);
            return false;
        }
    }

    try {
        const size_t bufferSize = strideBytes * frameHeight_;
        Microsoft::WRL::ComPtr<IMFMediaBuffer> buffer;
        ThrowIfFailed(MFCreateMemoryBuffer(static_cast<DWORD>(bufferSize), buffer.GetAddressOf()),
                      "CreateMemoryBuffer");

        BYTE* dest = nullptr;
        DWORD maxLen = 0;
        ThrowIfFailed(buffer->Lock(&dest, &maxLen, nullptr), "Lock buffer");
        std::memcpy(dest, bgraData, bufferSize);
        ThrowIfFailed(buffer->Unlock(), "Unlock buffer");
        ThrowIfFailed(buffer->SetCurrentLength(static_cast<DWORD>(bufferSize)), "SetCurrentLength");

        Microsoft::WRL::ComPtr<IMFSample> sample;
        ThrowIfFailed(MFCreateSample(sample.GetAddressOf()), "CreateSample");
        ThrowIfFailed(sample->AddBuffer(buffer.Get()), "AddBuffer");
        ThrowIfFailed(sample->SetSampleTime(rtStart_), "SetSampleTime");
        ThrowIfFailed(sample->SetSampleDuration(rtDuration_), "SetSampleDuration");

        const HRESULT writeHr = sinkWriter_->WriteSample(streamIndex_, sample.Get());
        if (IsDiskFullError(writeHr)) {
            FinalizeAndStop(StopReason::DiskFull);
            SetError(kDiskFullMessage);
            return false;
        }
        ThrowIfFailed(writeHr, "WriteSample");
        rtStart_ += rtDuration_;
        return true;
    } catch (const std::exception& e) {
        SetError(e.what());
    } catch (...) {
        SetError("Unknown exception adding frame");
    }
    FinalizeAndStop(StopReason::WriteFailed);
    return false;
}

double VideoRecorder::DurationSeconds() const
{
    if (rtDuration_ <= 0) {
        return 0.0;
    }
    return static_cast<double>(rtStart_) / 10'000'000.0;
}

} // namespace openzoom

#endif // _WIN32
