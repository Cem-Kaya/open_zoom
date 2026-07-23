#ifdef _WIN32

#include "openzoom/app/recording_manager.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QPushButton>
#include <QSignalBlocker>
#include <QStringList>

#include <array>
#include <utility>

namespace openzoom {

namespace {

const char* StateName(RecordingState state)
{
    switch (state) {
    case RecordingState::Idle:
        return "Idle";
    case RecordingState::Starting:
        return "Starting";
    case RecordingState::Recording:
        return "Recording";
    case RecordingState::Stopping:
        return "Stopping";
    case RecordingState::Error:
        return "Error";
    }
    return "Unknown";
}

} // namespace

RecordingManager::RecordingManager(QPushButton* recordButton,
                                   StatusCallback statusCallback)
    : recordButton_(recordButton),
      statusCallback_(std::move(statusCallback))
{
    UpdateButton();
}

RecordingManager::~RecordingManager()
{
    SetRecordingState(RecordingState::Stopping);
    SetRecordingState(RecordingState::Idle);
}

bool RecordingManager::IsActive() const
{
    return state_ == RecordingState::Starting || state_ == RecordingState::Recording;
}

void RecordingManager::SetRequested(bool requested)
{
    if (requested) {
        sessionTimestamp_ =
            QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"));
        codecName_.clear();
        frameCount_ = 0;
        recordingTimer_.restart();
        SetRecordingState(RecordingState::Starting);
        return;
    }
    Stop();
}

void RecordingManager::Stop(const QString& message)
{
    if (state_ == RecordingState::Idle) {
        return;
    }
    SetRecordingState(RecordingState::Stopping);
    SetRecordingState(RecordingState::Idle);
    if (!message.isEmpty()) {
        ShowStatus(message);
    }
}

bool RecordingManager::SetRecordingState(RecordingState state,
                                         const StartParameters* startParameters)
{
    qInfo() << "Recording state:" << StateName(state);

    if (state == RecordingState::Starting) {
        processedRecorder_.Stop();
        originalRecorder_.Stop();
        pendingOriginalReadbacks_.clear();
        processedWidth_ = 0;
        processedHeight_ = 0;
        originalWidth_ = 0;
        originalHeight_ = 0;
        state_ = state;
        UpdateButton();
        return true;
    }

    if (state == RecordingState::Recording) {
        if (!startParameters) {
            return false;
        }

        const QString dirPath = EnsureOutputDirectory();
        const QString processedPath = QDir(dirPath).filePath(
            QStringLiteral("VID_%1_processed.mp4").arg(sessionTimestamp_));
        const QString originalPath = QDir(dirPath).filePath(
            QStringLiteral("VID_%1_original.mp4").arg(sessionTimestamp_));
        constexpr UINT fps = 30;
        const std::array codecs{VideoRecorder::Codec::Av1, VideoRecorder::Codec::H264};
        QStringList failures;

        for (const VideoRecorder::Codec codec : codecs) {
            processedRecorder_.Stop();
            originalRecorder_.Stop();
            QFile::remove(processedPath);
            QFile::remove(originalPath);

            if (!processedRecorder_.Start(processedPath.toStdWString(),
                                          startParameters->processedWidth,
                                          startParameters->processedHeight,
                                          fps,
                                          codec)) {
                failures.append(QStringLiteral("%1 processed: %2")
                                    .arg(QString::fromLatin1(VideoRecorder::CodecName(codec)),
                                         QString::fromStdString(processedRecorder_.LastError())));
                continue;
            }
            if (!originalRecorder_.Start(originalPath.toStdWString(),
                                         startParameters->originalWidth,
                                         startParameters->originalHeight,
                                         fps,
                                         codec)) {
                failures.append(QStringLiteral("%1 original: %2")
                                    .arg(QString::fromLatin1(VideoRecorder::CodecName(codec)),
                                         QString::fromStdString(originalRecorder_.LastError())));
                processedRecorder_.Stop();
                QFile::remove(processedPath);
                continue;
            }

            processedWidth_ = startParameters->processedWidth;
            processedHeight_ = startParameters->processedHeight;
            originalWidth_ = startParameters->originalWidth;
            originalHeight_ = startParameters->originalHeight;
            codecName_ = QString::fromLatin1(VideoRecorder::CodecName(codec));
            recordingTimer_.restart();
            frameCount_ = 0;
            state_ = state;
            UpdateButton();
            qInfo() << "Paired recording started with" << codecName_
                    << ":" << processedPath << "and" << originalPath;
            ShowStatus(QStringLiteral("Recording original and processed video with %1.")
                           .arg(codecName_),
                       5000);
            return true;
        }

        qWarning() << "Failed to start paired recording:" << failures;
        processedRecorder_.Stop();
        originalRecorder_.Stop();
        QFile::remove(processedPath);
        QFile::remove(originalPath);
        if (!failures.isEmpty()) {
            ShowStatus(QStringLiteral("Recording could not start: %1")
                           .arg(failures.constLast()));
        }
        return false;
    }

    processedRecorder_.Stop();
    originalRecorder_.Stop();
    pendingOriginalReadbacks_.clear();
    codecName_.clear();
    if (state == RecordingState::Idle) {
        sessionTimestamp_.clear();
    }
    state_ = state;
    UpdateButton();
    return true;
}

void RecordingManager::StorePendingOriginal(UINT64 requestId, CapturedFrame frame)
{
    if (!IsActive() || !frame.IsValid()) {
        return;
    }
    pendingOriginalReadbacks_.insert_or_assign(requestId, std::move(frame));
}

void RecordingManager::HandleProcessedReadback(UINT64 requestId,
                                               const uint8_t* processedData,
                                               UINT processedWidth,
                                               UINT processedHeight)
{
    if (IsActive()) {
        auto original = pendingOriginalReadbacks_.find(requestId);
        if (original != pendingOriginalReadbacks_.end()) {
            CapturedFrame matchedOriginal = std::move(original->second);
            pendingOriginalReadbacks_.erase(original);
            AddFrame(processedData, processedWidth, processedHeight, matchedOriginal);
        }
    }

    for (auto pending = pendingOriginalReadbacks_.begin();
         pending != pendingOriginalReadbacks_.end();) {
        if (pending->first <= requestId) {
            pending = pendingOriginalReadbacks_.erase(pending);
        } else {
            ++pending;
        }
    }
}

void RecordingManager::ClearPendingReadbacks()
{
    pendingOriginalReadbacks_.clear();
}

void RecordingManager::AddFrame(const uint8_t* processedData,
                                UINT processedWidth,
                                UINT processedHeight,
                                const CapturedFrame& originalFrame)
{
    if (!IsActive() || !processedData || processedWidth == 0 || processedHeight == 0 ||
        !originalFrame.IsValid()) {
        return;
    }

    if (state_ == RecordingState::Starting) {
        const StartParameters parameters{
            processedWidth,
            processedHeight,
            originalFrame.width,
            originalFrame.height
        };
        if (!SetRecordingState(RecordingState::Recording, &parameters)) {
            SetRecordingState(RecordingState::Error);
            return;
        }
    }

    if (processedWidth != processedWidth_ || processedHeight != processedHeight_ ||
        originalFrame.width != originalWidth_ || originalFrame.height != originalHeight_) {
        qInfo() << "Recording stopped: frame size changed from"
                << processedWidth_ << "x" << processedHeight_ << "and"
                << originalWidth_ << "x" << originalHeight_ << "to"
                << processedWidth << "x" << processedHeight << "and"
                << originalFrame.width << "x" << originalFrame.height;
        Stop(QStringLiteral(
            "Recording stopped: a video size changed. Both recordings so far were saved."));
        return;
    }

    const bool processedWritten = processedRecorder_.AddFrame(
        processedData, static_cast<size_t>(processedWidth) * 4);
    const bool originalWritten = originalRecorder_.AddFrame(
        originalFrame.pixels.data(), static_cast<size_t>(originalFrame.width) * 4);
    if (!processedWritten || !originalWritten) {
        const VideoRecorder& failedRecorder =
            processedWritten ? originalRecorder_ : processedRecorder_;
        const VideoRecorder::StopReason reason = failedRecorder.LastStopReason();
        const QString message = QString::fromStdString(failedRecorder.LastError());
        SetRecordingState(RecordingState::Error);
        if (reason == VideoRecorder::StopReason::DiskFull) {
            qInfo() << "Recording finalized (disk full):" << message;
            ShowStatus(QStringLiteral(
                "Recording stopped: disk almost full. Both recordings so far were saved."));
        } else {
            qWarning() << "Recording error:" << message;
            if (!message.isEmpty()) {
                ShowStatus(message);
            }
        }
        return;
    }

    ++frameCount_;
    constexpr double kMaxSeconds = 12.0 * 3600.0;
    if (processedRecorder_.DurationSeconds() >= kMaxSeconds) {
        qInfo() << "Recording stopped: reached 12-hour limit";
        Stop(QStringLiteral("Recording stopped after reaching the 12-hour limit."));
    }
}

QString RecordingManager::EnsureOutputDirectory() const
{
    QDir base(QCoreApplication::applicationDirPath());
    const QString outputPath = base.filePath(QStringLiteral("output/vid"));
    QDir output(outputPath);
    if (!output.exists()) {
        output.mkpath(QStringLiteral("."));
    }
    return output.absolutePath();
}

void RecordingManager::ShowStatus(const QString& message, int durationMs) const
{
    if (statusCallback_ && !message.isEmpty()) {
        statusCallback_(message, durationMs);
    }
}

void RecordingManager::UpdateButton()
{
    if (!recordButton_) {
        return;
    }
    const bool active = IsActive();
    QSignalBlocker block(recordButton_);
    recordButton_->setChecked(active);
    recordButton_->setText(active ? QStringLiteral("Stop") : QStringLiteral("Record"));
}

} // namespace openzoom

#endif // _WIN32
