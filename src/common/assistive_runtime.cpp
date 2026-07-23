#ifdef _WIN32

#include "openzoom/common/assistive_runtime.hpp"
#include "openzoom/common/codex_app_server_client.hpp"

#include <QBuffer>
#include <QByteArray>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QProcessEnvironment>
#include <QSaveFile>
#include <QStandardPaths>
#include <QStringList>
#include <QTemporaryFile>
#include <QTimer>
#include <QUrl>
#include <QVariant>

#include <algorithm>
#include <limits>

#if OPENZOOM_HAS_TTS
#include <QTextToSpeech>
#endif

namespace openzoom {

namespace {

constexpr int kMinFrameEdge = 64;
constexpr int kMaxVlmFrameEdge = 2048;
constexpr int kOcrWatchdogMs = 10000;
constexpr int kVlmTransferTimeoutMs = 30000;
constexpr auto kNotesInsertionMarker = "<!-- OPENZOOM_NOTES_APPEND -->";

QString SanitizeText(QString text)
{
    text.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    text.replace(QStringLiteral("\r"), QStringLiteral("\n"));
    return text.trimmed();
}

QString TruncateText(const QString& text, int maxChars)
{
    if (text.size() <= maxChars) {
        return text;
    }
    return text.left(maxChars).trimmed() + QStringLiteral("...");
}

QString NotesImageUrl(const QString& notesFilePath, const QString& imagePath)
{
    const QFileInfo imageInfo(imagePath);
    const QDir notesDirectory(QFileInfo(notesFilePath).absolutePath());
    QString relativePath = notesDirectory.relativeFilePath(imageInfo.absoluteFilePath());
    relativePath = QDir::fromNativeSeparators(relativePath);

    const QUrl url = QDir::isAbsolutePath(relativePath)
                         ? QUrl::fromLocalFile(imageInfo.absoluteFilePath())
                         : QUrl(relativePath);
    return QString::fromUtf8(url.toEncoded(QUrl::FullyEncoded));
}

// Non-empty configured values take precedence over the environment variable.
QString ResolvedSetting(const QString& configured, const char* envName)
{
    const QString fromConfig = configured.trimmed();
    if (!fromConfig.isEmpty()) {
        return fromConfig;
    }
    return qEnvironmentVariable(envName).trimmed();
}

QString VlmNotConfiguredMessage()
{
    return QStringLiteral("VLM not configured. Set the server URL and model in AI Settings "
                          "or via OPENZOOM_VLM_API_URL and OPENZOOM_VLM_MODEL. An API key is optional for local servers.");
}

QString CodexNotAvailableMessage()
{
    return QStringLiteral("Codex is not ready. Open Advanced > Assistant to connect a ChatGPT account, "
                          "or choose an OpenAI-compatible provider in AI Settings.");
}

QString ParseVlmResponseText(const QByteArray& payload)
{
    QJsonParseError error{};
    const QJsonDocument doc = QJsonDocument::fromJson(payload, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject()) {
        return {};
    }

    const QJsonObject root = doc.object();

    const QJsonArray choices = root.value(QStringLiteral("choices")).toArray();
    if (!choices.isEmpty()) {
        const QJsonObject firstChoice = choices.first().toObject();
        const QJsonObject message = firstChoice.value(QStringLiteral("message")).toObject();
        const QJsonValue content = message.value(QStringLiteral("content"));
        if (content.isString()) {
            return SanitizeText(content.toString());
        }
        if (content.isArray()) {
            QStringList parts;
            for (const QJsonValue& itemValue : content.toArray()) {
                const QJsonObject item = itemValue.toObject();
                if (item.value(QStringLiteral("type")).toString() == QStringLiteral("text")) {
                    parts.push_back(item.value(QStringLiteral("text")).toString());
                }
            }
            return SanitizeText(parts.join(QStringLiteral("\n")));
        }
    }

    const QJsonArray output = root.value(QStringLiteral("output")).toArray();
    QStringList responseParts;
    for (const QJsonValue& blockValue : output) {
        const QJsonObject block = blockValue.toObject();
        const QJsonArray content = block.value(QStringLiteral("content")).toArray();
        for (const QJsonValue& itemValue : content) {
            const QJsonObject item = itemValue.toObject();
            if (item.value(QStringLiteral("type")).toString() == QStringLiteral("output_text")) {
                responseParts.push_back(item.value(QStringLiteral("text")).toString());
            }
        }
    }
    return SanitizeText(responseParts.join(QStringLiteral("\n")));
}

} // namespace

AssistiveRuntime::AssistiveRuntime(QObject* parent)
    : QObject(parent)
{
    networkManager_ = new QNetworkAccessManager(this);
    ocrProcess_ = std::make_unique<QProcess>(this);
    codexClient_ = std::make_unique<CodexAppServerClient>(this);

    connect(codexClient_.get(), &CodexAppServerClient::ServerStateChanged,
            this, &AssistiveRuntime::CodexServerStateChanged);
    connect(codexClient_.get(), &CodexAppServerClient::AccountChanged,
            this, &AssistiveRuntime::CodexAccountChanged);
    connect(codexClient_.get(), &CodexAppServerClient::ModelsChanged,
            this, &AssistiveRuntime::CodexModelsChanged);
    connect(codexClient_.get(), &CodexAppServerClient::ModelCatalogChanged,
            this, &AssistiveRuntime::CodexModelCatalogChanged);
    connect(codexClient_.get(), &CodexAppServerClient::RateLimitChanged,
            this, &AssistiveRuntime::CodexRateLimitChanged);
    connect(codexClient_.get(), &CodexAppServerClient::LoginUrlReady,
            this, &AssistiveRuntime::CodexLoginUrlReady);
    connect(codexClient_.get(), &CodexAppServerClient::ConversationCreated,
            this, &AssistiveRuntime::AssistantConversationCreated);
    connect(codexClient_.get(), &CodexAppServerClient::ConversationTranscriptLoaded,
            this, &AssistiveRuntime::AssistantTranscriptLoaded);
    connect(codexClient_.get(), &CodexAppServerClient::ConversationRenamed,
            this, &AssistiveRuntime::AssistantConversationRenamed);
    connect(codexClient_.get(), &CodexAppServerClient::ConversationDeleted,
            this, &AssistiveRuntime::AssistantConversationDeleted);
    connect(codexClient_.get(), &CodexAppServerClient::TurnStarted,
            this, &AssistiveRuntime::AssistantTurnStarted);
    connect(codexClient_.get(), &CodexAppServerClient::TurnTextDelta,
            this,
            [this](const QString& threadId, const QString& turnId, const QString& delta) {
                vlmText_ += delta;
                vlmStatus_.clear();
                RefreshOverlay();
                emit AssistantTextDelta(threadId, turnId, delta);
            });
    connect(codexClient_.get(), &CodexAppServerClient::TurnFinished,
            this,
            [this](const QString& threadId,
                   const QString& turnId,
                   const QString& text,
                   const QString& error,
                   bool interrupted,
                   bool persistent) {
                if (interrupted) {
                    vlmText_.clear();
                    vlmStatus_ = QStringLiteral("Assistant stopped.");
                    RefreshOverlay();
                } else if (!error.isEmpty()) {
                    FinishVlmError(error);
                } else {
                    FinishVlmSuccess(text);
                }
                emit AssistantTurnFinished(threadId, turnId, text, error, interrupted, persistent);
            });

    ocrWatchdogTimer_ = new QTimer(this);
    ocrWatchdogTimer_->setSingleShot(true);
    ocrWatchdogTimer_->setInterval(kOcrWatchdogMs);
    connect(ocrWatchdogTimer_, &QTimer::timeout, this, [this]() {
        if (!ocrProcess_ || ocrProcess_->state() == QProcess::NotRunning) {
            return;
        }
        ocrTimedOut_ = true;
        FinishOcrError(QStringLiteral("OCR timed out."));
        // The finished/errorOccurred handlers remove the temp image once the
        // process is gone and the file is no longer locked.
        ocrProcess_->kill();
    });

    connect(ocrProcess_.get(),
            qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this,
            [this](int exitCode, QProcess::ExitStatus exitStatus) {
                ocrWatchdogTimer_->stop();
                const QString stdoutText = SanitizeText(QString::fromUtf8(ocrProcess_->readAllStandardOutput()));
                const QString stderrText = SanitizeText(QString::fromUtf8(ocrProcess_->readAllStandardError()));
                if (!pendingOcrImagePath_.isEmpty()) {
                    QFile::remove(pendingOcrImagePath_);
                    pendingOcrImagePath_.clear();
                }
                if (ocrTimedOut_) {
                    // The watchdog already reported the timeout.
                    ocrTimedOut_ = false;
                    ocrRunForced_ = false;
                    return;
                }
                if (exitStatus == QProcess::NormalExit && exitCode == 0) {
                    FinishOcrSuccess(stdoutText);
                    return;
                }
                ocrRunForced_ = false;
                QString errorText = stderrText;
                if (errorText.isEmpty()) {
                    errorText = QStringLiteral("tesseract exited with code %1").arg(exitCode);
                }
                FinishOcrError(errorText);
            });

    connect(ocrProcess_.get(),
            &QProcess::errorOccurred,
            this,
            [this](QProcess::ProcessError error) {
                ocrWatchdogTimer_->stop();
                if (!pendingOcrImagePath_.isEmpty()) {
                    QFile::remove(pendingOcrImagePath_);
                    pendingOcrImagePath_.clear();
                }
                if (ocrTimedOut_) {
                    // Kill after timeout surfaces as Crashed; already reported.
                    return;
                }
                ocrRunForced_ = false;
                QString errorText;
                switch (error) {
                case QProcess::FailedToStart:
                    errorText = QStringLiteral("tesseract not found. Install it or set its path in the "
                                               "assistive settings or via OPENZOOM_TESSERACT_PATH.");
                    break;
                case QProcess::Crashed:
                    errorText = QStringLiteral("tesseract crashed during OCR.");
                    break;
                default:
                    errorText = QStringLiteral("tesseract OCR process failed.");
                    break;
                }
                FinishOcrError(errorText);
            });

    RefreshOverlay();
}

AssistiveRuntime::~AssistiveRuntime()
{
    if (activeReply_) {
        activeReply_->abort();
    }
    if (ocrProcess_ && ocrProcess_->state() != QProcess::NotRunning) {
        ocrProcess_->kill();
        ocrProcess_->waitForFinished(1000);
    }
    if (!pendingOcrImagePath_.isEmpty()) {
        QFile::remove(pendingOcrImagePath_);
    }
}

void AssistiveRuntime::SetConfig(const AssistiveRuntimeConfig& config)
{
    const bool notesTargetChanged = config.notesDirectory.trimmed() != config_.notesDirectory.trimmed();
    const bool speechConfigChanged =
        config.ttsEngine != config_.ttsEngine ||
        config.ttsVoiceName != config_.ttsVoiceName ||
        config.ttsVoiceLocale != config_.ttsVoiceLocale ||
        config.ttsRate != config_.ttsRate;
    if (speechConfigChanged) {
        StopSpeech();
#if OPENZOOM_HAS_TTS
        delete tts_;
        tts_ = nullptr;
#endif
    }
    config_ = config;
    if (codexClient_) {
        codexClient_->Configure(config_.codexExecutablePath,
                                config_.codexModel,
                                config_.codexReasoningEffort,
                                config_.assistantInstructions,
                                config_.codexInternetEnabled,
                                config_.codexCodingEnabled,
                                config_.codexWorkspaceDirectory);
        if (UsesCodexProvider()) {
            codexClient_->Start();
        }
    }

    // New credentials or a new tesseract path may fix a previous hard failure.
    ocrHardUnavailable_ = false;
    vlmHardUnavailable_ = false;

    if (notesTargetChanged) {
        notesFilePath_.clear();
        lastNotedOcrText_.clear();
    }

    if (vlmEnabled_ && vlmText_.isEmpty()) {
        vlmStatus_ = VlmConfigured() ? QStringLiteral("VLM ready.") : VlmNotConfiguredMessage();
        RefreshOverlay();
    }
}

void AssistiveRuntime::SetModes(bool ocrEnabled, bool vlmEnabled)
{
    const bool ocrTurningOff = ocrEnabled_ && !ocrEnabled;
    const bool vlmTurningOff = vlmEnabled_ && !vlmEnabled;

    if (ocrTurningOff && ocrProcess_ && ocrProcess_->state() != QProcess::NotRunning) {
        ocrWatchdogTimer_->stop();
        ocrProcess_->kill();
        ocrProcess_->waitForFinished(250);
    }
    if (vlmTurningOff && activeReply_) {
        activeReply_->abort();
    }

    ocrEnabled_ = ocrEnabled;
    vlmEnabled_ = vlmEnabled;
    if (ocrEnabled_) {
        ocrHardUnavailable_ = false;
    }
    if (vlmEnabled_) {
        vlmHardUnavailable_ = false;
    }
    if (ocrTurningOff) {
        ocrForcedVisible_ = false;
        ocrText_.clear();
        ocrStatus_.clear();
    } else if (ocrEnabled_ && ocrText_.isEmpty() && ocrStatus_.isEmpty()) {
        ocrStatus_ = QStringLiteral("OCR ready. Install tesseract or set its path in the assistive "
                                    "settings or via OPENZOOM_TESSERACT_PATH if detection fails.");
    }

    if (vlmTurningOff) {
        vlmForcedVisible_ = false;
        vlmText_.clear();
        vlmStatus_.clear();
    } else if (vlmEnabled_ && vlmText_.isEmpty() && vlmStatus_.isEmpty()) {
        if (VlmConfigured()) {
            vlmStatus_ = QStringLiteral("VLM ready.");
        } else {
            vlmStatus_ = VlmNotConfiguredMessage();
        }
    }

    RefreshOverlay();
}

void AssistiveRuntime::ReadAloud(const QString& text)
{
    SpeakText(text);
}

void AssistiveRuntime::DismissOverlay()
{
    overlayDismissed_ = true;
    RefreshOverlay();
}

bool AssistiveRuntime::WantsAnalysis() const
{
    // Subscription-backed Codex explanations are user initiated. This avoids
    // spending a user's Codex allowance every 1.6 seconds in an assistive mode.
    const bool automaticVlm = !UsesCodexProvider() && vlmEnabled_ && !vlmHardUnavailable_;
    return (ocrEnabled_ && !ocrHardUnavailable_) || automaticVlm;
}

bool AssistiveRuntime::IsBusy() const
{
    const bool ocrBusy = ocrProcess_ && ocrProcess_->state() != QProcess::NotRunning;
    return ocrBusy || activeReply_ != nullptr ||
           (codexClient_ && codexClient_->IsTurnActive());
}

bool AssistiveRuntime::IsCodexTurnActive() const
{
    return codexClient_ && codexClient_->IsTurnActive();
}

void AssistiveRuntime::SubmitFrame(const uint8_t* bgraData, int width, int height)
{
    if (!WantsAnalysis() || !ValidateFrame(bgraData, width, height)) {
        return;
    }

    if (ocrEnabled_ && ocrProcess_ && ocrProcess_->state() == QProcess::NotRunning) {
        StartOcr(bgraData, width, height, false);
    }
    if (vlmEnabled_ && !UsesCodexProvider() && activeReply_ == nullptr) {
        StartVlm(bgraData, width, height);
    }
}

void AssistiveRuntime::SubmitFrameForced(const uint8_t* bgraData, int width, int height, bool runOcr, bool runVlm)
{
    if ((!runOcr && !runVlm) || !ValidateFrame(bgraData, width, height)) {
        return;
    }
    overlayDismissed_ = false;

    if (runOcr && !runVlm && !vlmEnabled_) {
        vlmForcedVisible_ = false;
        vlmText_.clear();
        vlmStatus_.clear();
    } else if (runVlm && !runOcr && !ocrEnabled_) {
        ocrForcedVisible_ = false;
        ocrText_.clear();
        ocrStatus_.clear();
    }

    if (runOcr) {
        ocrForcedVisible_ = true;
        if (ocrProcess_ && ocrProcess_->state() != QProcess::NotRunning) {
            FinishOcrError(QStringLiteral("OCR is busy with a previous capture. Try again in a moment."));
        } else {
            StartOcr(bgraData, width, height, true);
        }
    }

    if (runVlm) {
        vlmForcedVisible_ = true;
        if (activeReply_ != nullptr || (codexClient_ && codexClient_->IsTurnActive())) {
            FinishVlmError(QStringLiteral("Scene explanation is busy with a previous request. Try again in a moment."));
        } else if (!VlmConfigured()) {
            FinishVlmError(VlmNotConfiguredMessage());
        } else {
            vlmHardUnavailable_ = false;
            StartVlm(bgraData, width, height);
        }
    }
}

void AssistiveRuntime::NoteCapturedPhoto(const QString& filePath)
{
    if (filePath.trimmed().isEmpty()) {
        return;
    }
    AppendNoteSection(QStringLiteral("Photo captured"), {}, filePath);
}

void AssistiveRuntime::StartCodexLogin()
{
    if (codexClient_) {
        codexClient_->Start();
        codexClient_->StartChatGptLogin();
    }
}

void AssistiveRuntime::StopAssistant()
{
    if (activeReply_) {
        activeReply_->abort();
        return;
    }
    if (codexClient_) {
        codexClient_->InterruptTurn();
    }
}

void AssistiveRuntime::SubmitAssistantPrompt(const QString& prompt,
                                             const QString& threadId,
                                             const uint8_t* bgraData,
                                             int width,
                                             int height,
                                             bool attachFrame)
{
    if (!UsesCodexProvider()) {
        emit AssistantTurnFinished(threadId, {}, {},
                                   QStringLiteral("Persistent Assistant conversations require the Codex subscription provider."),
                                   false, true);
        return;
    }
    if (!codexClient_ || codexClient_->IsTurnActive()) {
        emit AssistantTurnFinished(threadId, {}, {},
                                   QStringLiteral("Another assistant request is already running."),
                                   false, true);
        return;
    }

    QString imagePath;
    if (attachFrame) {
        if (!ValidateFrame(bgraData, width, height)) {
            emit AssistantTurnFinished(threadId, {}, {},
                                       QStringLiteral("No camera frame is available to attach."),
                                       false, true);
            return;
        }
        imagePath = SaveCodexFrame(bgraData, width, height);
        if (imagePath.isEmpty()) {
            emit AssistantTurnFinished(threadId, {}, {},
                                       QStringLiteral("Could not prepare the current camera view."),
                                       false, true);
            return;
        }
    }
    overlayDismissed_ = false;
    vlmForcedVisible_ = true;
    vlmText_.clear();
    vlmStatus_ = QStringLiteral("Thinking...");
    RefreshOverlay();
    codexClient_->RequestVisionTurn(prompt, imagePath, threadId, true);
}

void AssistiveRuntime::LoadAssistantConversation(const QString& threadId)
{
    if (codexClient_) {
        codexClient_->LoadConversation(threadId);
    }
}

void AssistiveRuntime::RenameAssistantConversation(const QString& threadId, const QString& name)
{
    if (codexClient_) {
        codexClient_->RenameConversation(threadId, name);
    }
}

void AssistiveRuntime::DeleteAssistantConversation(const QString& threadId)
{
    if (codexClient_) {
        codexClient_->DeleteConversation(threadId);
    }
}

QString AssistiveRuntime::notesFilePath() const
{
    return notesFilePath_;
}

void AssistiveRuntime::RefreshOverlay()
{
    QStringList sections;
    if (ocrEnabled_ || ocrForcedVisible_) {
        QString text = ocrText_.isEmpty() ? ocrStatus_ : ocrText_;
        if (!text.isEmpty()) {
            sections.push_back(QStringLiteral("OCR\n%1").arg(text));
        }
    }
    if (vlmEnabled_ || vlmForcedVisible_) {
        QString text = vlmText_.isEmpty() ? vlmStatus_ : vlmText_;
        if (!text.isEmpty()) {
            sections.push_back(QStringLiteral("Scene Explain\n%1").arg(text));
        }
    }

    const QString body = sections.join(QStringLiteral("\n\n"));
    emit OverlayUpdated(QStringLiteral("Assistive View"),
                        body,
                        !overlayDismissed_ && !body.isEmpty());
}

QString AssistiveRuntime::TesseractProgram() const
{
    const QString resolved = ResolvedSetting(config_.tesseractPath, "OPENZOOM_TESSERACT_PATH");
    if (!resolved.isEmpty()) {
        const QFileInfo configured(resolved);
        if (configured.isDir()) {
            return QDir(configured.absoluteFilePath()).filePath(QStringLiteral("tesseract.exe"));
        }
        return resolved;
    }

    const QString appDirectory = QCoreApplication::applicationDirPath();
    QStringList candidates{
        QDir(QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation))
            .filePath(QStringLiteral("OpenZoom/tools/tesseract/tesseract.exe")),
        QDir(appDirectory).filePath(QStringLiteral("tools/tesseract/tesseract.exe")),
        QDir(appDirectory).filePath(QStringLiteral("tesseract/tesseract.exe")),
        QDir(appDirectory).filePath(QStringLiteral("tesseract.exe"))};

    const QString pathExecutable = QStandardPaths::findExecutable(QStringLiteral("tesseract.exe"));
    if (!pathExecutable.isEmpty()) {
        candidates.push_back(pathExecutable);
    }

    const QString programFiles = qEnvironmentVariable("ProgramFiles").trimmed();
    if (!programFiles.isEmpty()) {
        candidates.push_back(QDir(programFiles).filePath(QStringLiteral("Tesseract-OCR/tesseract.exe")));
    }
    const QString localAppData = qEnvironmentVariable("LOCALAPPDATA").trimmed();
    if (!localAppData.isEmpty()) {
        candidates.push_back(QDir(localAppData).filePath(
            QStringLiteral("Programs/Tesseract-OCR/tesseract.exe")));
    }

    for (const QString& candidate : candidates) {
        const QFileInfo info(candidate);
        if (info.isFile()) {
            return info.absoluteFilePath();
        }
    }
    return QStringLiteral("tesseract");
}

bool AssistiveRuntime::VlmConfigured() const
{
    if (UsesCodexProvider()) {
        return codexClient_ != nullptr;
    }
    return !ResolvedSetting(config_.vlmApiUrl, "OPENZOOM_VLM_API_URL").isEmpty() &&
           !ResolvedSetting(config_.vlmModel, "OPENZOOM_VLM_MODEL").isEmpty();
}

bool AssistiveRuntime::UsesCodexProvider() const
{
    return config_.aiProvider.trimmed().compare(QStringLiteral("codex"), Qt::CaseInsensitive) == 0;
}

bool AssistiveRuntime::ValidateFrame(const uint8_t* bgraData, int width, int height)
{
    if (!bgraData) {
        return false;
    }
    if (width < kMinFrameEdge || height < kMinFrameEdge) {
        if (!warnedDegenerateFrame_) {
            warnedDegenerateFrame_ = true;
            qWarning("AssistiveRuntime: ignoring degenerate %dx%d frame (minimum edge is %d px).",
                     width, height, kMinFrameEdge);
        }
        return false;
    }
    // BGRA stride math (width * 4 * height) must not overflow int; camera
    // dimensions are untrusted driver input.
    if (width > std::numeric_limits<int>::max() / 4 ||
        height > std::numeric_limits<int>::max() / (width * 4)) {
        qWarning("AssistiveRuntime: rejecting oversized %dx%d frame.", width, height);
        return false;
    }
    return true;
}

void AssistiveRuntime::StartOcr(const uint8_t* bgraData, int width, int height, bool forced)
{
    if (ocrHardUnavailable_ && !forced) {
        return;
    }
    QImage frameImage(bgraData, width, height, width * 4, QImage::Format_ARGB32);
    QImage copy = frameImage.copy();

    QTemporaryFile tempFile(QDir(QStandardPaths::writableLocation(QStandardPaths::TempLocation))
                                .filePath(QStringLiteral("openzoom_ocr_XXXXXX.png")));
    tempFile.setAutoRemove(false);
    if (!tempFile.open()) {
        FinishOcrError(QStringLiteral("Failed to create temporary image for OCR."));
        return;
    }
    const QString imagePath = tempFile.fileName();
    tempFile.close();

    if (!copy.save(imagePath, "PNG")) {
        QFile::remove(imagePath);
        FinishOcrError(QStringLiteral("Failed to save OCR input image."));
        return;
    }

    pendingOcrImagePath_ = imagePath;
    ocrRunForced_ = forced;
    ocrTimedOut_ = false;
    ocrStatus_ = QStringLiteral("Running OCR...");
    RefreshOverlay();

    QStringList arguments{imagePath, QStringLiteral("stdout"), QStringLiteral("--psm"), QStringLiteral("6")};
    const QString language = config_.ocrLanguage.trimmed();
    if (!language.isEmpty()) {
        arguments << QStringLiteral("-l") << language;
    }

    const QString program = TesseractProgram();
    ocrProcess_->setProgram(program);
    ocrProcess_->setArguments(arguments);
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    const QFileInfo programInfo(program);
    if (programInfo.isFile()) {
        const QString tessdataPath = QDir(programInfo.absolutePath()).filePath(QStringLiteral("tessdata"));
        if (QDir(tessdataPath).exists()) {
            environment.insert(QStringLiteral("TESSDATA_PREFIX"), QDir::toNativeSeparators(tessdataPath));
        }
        ocrProcess_->setWorkingDirectory(programInfo.absolutePath());
    } else {
        ocrProcess_->setWorkingDirectory(QString());
    }
    ocrProcess_->setProcessEnvironment(environment);
    ocrProcess_->start();
    ocrWatchdogTimer_->start();
}

void AssistiveRuntime::StartVlm(const uint8_t* bgraData, int width, int height)
{
    if (UsesCodexProvider()) {
        QString prompt = config_.vlmPrompt.trimmed();
        if (prompt.isEmpty()) {
            prompt = QStringLiteral("Describe the visible scene briefly for a low-vision user. "
                                    "Focus on readable text, controls, and major objects.");
        }
        StartCodexVlm(bgraData, width, height, prompt, {}, false);
        return;
    }
    if (vlmHardUnavailable_) {
        return;
    }
    if (!VlmConfigured()) {
        vlmHardUnavailable_ = true;
        FinishVlmError(VlmNotConfiguredMessage());
        return;
    }

    QImage frameImage(bgraData, width, height, width * 4, QImage::Format_ARGB32);
    QImage copy = frameImage.copy();
    if (copy.width() > kMaxVlmFrameEdge || copy.height() > kMaxVlmFrameEdge) {
        copy = copy.scaled(kMaxVlmFrameEdge, kMaxVlmFrameEdge,
                           Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    QByteArray jpegBytes;
    QBuffer buffer(&jpegBytes);
    buffer.open(QIODevice::WriteOnly);
    if (!copy.save(&buffer, "JPG", 82)) {
        FinishVlmError(QStringLiteral("Failed to encode frame for VLM request."));
        return;
    }

    const QString apiUrl = ResolvedSetting(config_.vlmApiUrl, "OPENZOOM_VLM_API_URL");
    const QString apiKey = ResolvedSetting(config_.vlmApiKey, "OPENZOOM_VLM_API_KEY");
    const QString model = ResolvedSetting(config_.vlmModel, "OPENZOOM_VLM_MODEL");
    QString prompt = ResolvedSetting(config_.vlmPrompt, "OPENZOOM_VLM_PROMPT");
    if (prompt.isEmpty()) {
        prompt = QStringLiteral("Describe the visible scene briefly for a low-vision user. Focus on readable text, UI elements, and major objects.");
    }

    const QString dataUri = QStringLiteral("data:image/jpeg;base64,%1")
                                .arg(QString::fromLatin1(jpegBytes.toBase64()));

    QJsonObject textPart;
    textPart.insert(QStringLiteral("type"), QStringLiteral("text"));
    textPart.insert(QStringLiteral("text"), prompt);

    QJsonObject imagePart;
    imagePart.insert(QStringLiteral("type"), QStringLiteral("image_url"));
    imagePart.insert(QStringLiteral("image_url"), QJsonObject{{QStringLiteral("url"), dataUri}});

    QJsonObject message;
    message.insert(QStringLiteral("role"), QStringLiteral("user"));
    message.insert(QStringLiteral("content"), QJsonArray{textPart, imagePart});

    QJsonArray messages;
    const QString assistantInstructions = config_.assistantInstructions.trimmed();
    if (!assistantInstructions.isEmpty()) {
        messages.append(QJsonObject{{QStringLiteral("role"), QStringLiteral("system")},
                                    {QStringLiteral("content"), assistantInstructions}});
    }
    messages.append(message);

    QJsonObject requestBody;
    requestBody.insert(QStringLiteral("model"), model);
    requestBody.insert(QStringLiteral("messages"), messages);
    requestBody.insert(QStringLiteral("max_tokens"), 180);

    QNetworkRequest request{QUrl(apiUrl)};
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    if (!apiKey.isEmpty()) {
        request.setRawHeader("Authorization", QStringLiteral("Bearer %1").arg(apiKey).toUtf8());
    }
    request.setTransferTimeout(kVlmTransferTimeoutMs);

    vlmStatus_ = QStringLiteral("Querying VLM...");
    RefreshOverlay();

    activeReply_ = networkManager_->post(request, QJsonDocument(requestBody).toJson(QJsonDocument::Compact));
    connect(activeReply_, &QNetworkReply::finished, this, [this]() {
        QNetworkReply* reply = activeReply_;
        activeReply_ = nullptr;
        if (!reply) {
            return;
        }

        const QByteArray payload = reply->readAll();
        if (reply->error() != QNetworkReply::NoError) {
            QString detail = reply->errorString();
            const QVariant statusAttr = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute);
            if (statusAttr.isValid()) {
                detail += QStringLiteral(" (HTTP %1)").arg(statusAttr.toInt());
            }
            const QString excerpt = TruncateText(SanitizeText(QString::fromUtf8(payload)), 160);
            if (!excerpt.isEmpty()) {
                detail += QStringLiteral(" - %1").arg(excerpt);
            }
            FinishVlmError(QStringLiteral("VLM request failed: %1").arg(detail));
            reply->deleteLater();
            return;
        }

        const QString text = ParseVlmResponseText(payload);
        if (text.isEmpty()) {
            FinishVlmError(QStringLiteral("VLM response did not contain readable text."));
        } else {
            FinishVlmSuccess(text);
        }
        reply->deleteLater();
    });
}

void AssistiveRuntime::StartCodexVlm(const uint8_t* bgraData,
                                     int width,
                                     int height,
                                     const QString& prompt,
                                     const QString& threadId,
                                     bool persistent)
{
    if (!codexClient_) {
        FinishVlmError(CodexNotAvailableMessage());
        return;
    }
    const QString imagePath = SaveCodexFrame(bgraData, width, height);
    if (imagePath.isEmpty()) {
        FinishVlmError(QStringLiteral("Failed to prepare the current view for Codex."));
        return;
    }
    vlmText_.clear();
    vlmStatus_ = QStringLiteral("Thinking...");
    RefreshOverlay();
    codexClient_->RequestVisionTurn(prompt, imagePath, threadId, persistent);
}

QString AssistiveRuntime::SaveCodexFrame(const uint8_t* bgraData, int width, int height)
{
    QImage frameImage(bgraData, width, height, width * 4, QImage::Format_ARGB32);
    QImage copy = frameImage.copy();
    if (copy.width() > kMaxVlmFrameEdge || copy.height() > kMaxVlmFrameEdge) {
        copy = copy.scaled(kMaxVlmFrameEdge, kMaxVlmFrameEdge,
                           Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    QTemporaryFile tempFile(QDir(QStandardPaths::writableLocation(QStandardPaths::TempLocation))
                                .filePath(QStringLiteral("openzoom_codex_XXXXXX.jpg")));
    tempFile.setAutoRemove(false);
    if (!tempFile.open()) {
        return {};
    }
    const QString path = tempFile.fileName();
    tempFile.close();
    if (!copy.save(path, "JPG", 85)) {
        QFile::remove(path);
        return {};
    }
    return path;
}

void AssistiveRuntime::FinishOcrSuccess(const QString& text)
{
    ocrRunForced_ = false;
    const QString fullText = SanitizeText(text);
    ocrText_ = TruncateText(fullText, 700);
    if (ocrText_.isEmpty()) {
        ocrStatus_ = QStringLiteral("OCR found no readable text.");
    } else {
        ocrStatus_.clear();
        if (fullText != lastNotedOcrText_) {
            lastNotedOcrText_ = fullText;
            AppendNoteSection(QStringLiteral("Text on screen"), fullText);
        }
    }
    RefreshOverlay();
}

void AssistiveRuntime::FinishOcrError(const QString& errorText)
{
    ocrText_.clear();
    ocrStatus_ = SanitizeText(errorText);
    if (ocrStatus_.contains(QStringLiteral("not found"), Qt::CaseInsensitive)) {
        ocrHardUnavailable_ = true;
    }
    RefreshOverlay();
}

void AssistiveRuntime::FinishVlmSuccess(const QString& text)
{
    const QString fullText = SanitizeText(text);
    vlmText_ = TruncateText(fullText, 700);
    if (vlmText_.isEmpty()) {
        vlmStatus_ = QStringLiteral("VLM returned an empty description.");
    } else {
        vlmStatus_.clear();
        AppendNoteSection(QStringLiteral("Scene explanation"), fullText);
    }
    RefreshOverlay();
}

void AssistiveRuntime::FinishVlmError(const QString& errorText)
{
    vlmText_.clear();
    vlmStatus_ = SanitizeText(errorText);
    RefreshOverlay();
}

bool AssistiveRuntime::EnsureNotesFile()
{
    if (!notesFilePath_.isEmpty()) {
        return true;
    }
    const QString directory = config_.notesDirectory.trimmed();
    if (directory.isEmpty()) {
        return false;
    }
    if (!QDir().mkpath(directory)) {
        qWarning("AssistiveRuntime: failed to create notes directory %s", qPrintable(directory));
        return false;
    }

    const QDateTime now = QDateTime::currentDateTime();
    const QString fileName = QStringLiteral("NOTES_%1.html")
                                 .arg(now.toString(QStringLiteral("yyyyMMdd_HHmmss")));
    const QString path = QDir(directory).filePath(fileName);

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning("AssistiveRuntime: failed to create notes file %s", qPrintable(path));
        return false;
    }
    const QString displayTime = now.toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
    const QString machineTime = now.toString(Qt::ISODate);
    const QString document = QStringLiteral(
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "  <title>OpenZoom Lecture Notes - %1</title>\n"
        "  <style>\n"
        "    :root { color-scheme: light dark; font: 18px/1.6 system-ui, sans-serif; }\n"
        "    body { margin: 0; background: Canvas; color: CanvasText; }\n"
        "    main { width: min(72rem, calc(100% - 2rem)); margin: 0 auto; padding: 2rem 0 4rem; }\n"
        "    h1 { font-size: clamp(1.8rem, 5vw, 3rem); line-height: 1.15; margin: 0; }\n"
        "    .created { color: GrayText; margin: .5rem 0 2rem; }\n"
        "    section { border-top: 2px solid GrayText; padding: 1.5rem 0; }\n"
        "    h2 { font-size: 1.25rem; line-height: 1.3; margin: 0 0 1rem; }\n"
        "    time { font-variant-numeric: tabular-nums; }\n"
        "    .note-text { white-space: pre-wrap; overflow-wrap: anywhere; }\n"
        "    figure { margin: 0; }\n"
        "    img { display: block; max-width: 100%; height: auto; border: 2px solid GrayText; }\n"
        "    figcaption { margin-top: .5rem; color: GrayText; }\n"
        "    a:focus-visible { outline: 4px solid Highlight; outline-offset: 4px; }\n"
        "    @media print { main { width: 100%; } section { break-inside: avoid; } }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "<main>\n"
        "  <h1>OpenZoom Lecture Notes</h1>\n"
        "  <p class=\"created\">Started <time datetime=\"%2\">%1</time></p>\n"
        "  %3\n"
        "</main>\n"
        "</body>\n"
        "</html>\n")
                                 .arg(displayTime.toHtmlEscaped(),
                                      machineTime.toHtmlEscaped(),
                                      QString::fromLatin1(kNotesInsertionMarker));
    const QByteArray documentBytes = document.toUtf8();
    if (file.write(documentBytes) != documentBytes.size() || !file.commit()) {
        qWarning("AssistiveRuntime: failed to finalize notes file %s", qPrintable(path));
        return false;
    }

    notesFilePath_ = path;
    return true;
}

void AssistiveRuntime::AppendNoteSection(const QString& heading,
                                         const QString& bodyText,
                                         const QString& imagePath)
{
    if (!config_.lectureNotesEnabled || config_.notesDirectory.trimmed().isEmpty()) {
        return;
    }
    if (bodyText.trimmed().isEmpty() && imagePath.trimmed().isEmpty()) {
        return;
    }
    if (!EnsureNotesFile()) {
        return;
    }

    QFile existingFile(notesFilePath_);
    if (!existingFile.open(QIODevice::ReadOnly)) {
        qWarning("AssistiveRuntime: failed to read notes file %s", qPrintable(notesFilePath_));
        return;
    }
    QByteArray document = existingFile.readAll();
    existingFile.close();

    const QByteArray marker(kNotesInsertionMarker);
    const qsizetype markerOffset = document.indexOf(marker);
    if (markerOffset < 0) {
        qWarning("AssistiveRuntime: notes insertion marker missing from %s", qPrintable(notesFilePath_));
        return;
    }

    const QString timestamp = QTime::currentTime().toString(QStringLiteral("HH:mm:ss"));
    QString content;
    if (!imagePath.trimmed().isEmpty()) {
        const QString imageUrl = NotesImageUrl(notesFilePath_, imagePath).toHtmlEscaped();
        content = QStringLiteral(
                      "    <figure>\n"
                      "      <a href=\"%1\"><img src=\"%1\" alt=\"Captured processed camera view\" loading=\"lazy\"></a>\n"
                      "      <figcaption>Processed camera view</figcaption>\n"
                      "    </figure>\n")
                      .arg(imageUrl);
    } else {
        content = QStringLiteral("    <div class=\"note-text\">%1</div>\n")
                      .arg(bodyText.toHtmlEscaped());
    }
    const QString section = QStringLiteral(
                                "  <section>\n"
                                "    <h2><time>[%1]</time> %2</h2>\n"
                                "%3"
                                "  </section>\n")
                                .arg(timestamp.toHtmlEscaped(), heading.toHtmlEscaped(), content);
    document.insert(markerOffset, section.toUtf8());

    QSaveFile outputFile(notesFilePath_);
    if (!outputFile.open(QIODevice::WriteOnly) ||
        outputFile.write(document) != document.size() ||
        !outputFile.commit()) {
        qWarning("AssistiveRuntime: failed to update notes file %s", qPrintable(notesFilePath_));
    }
}

void AssistiveRuntime::SpeakText(const QString& text)
{
#if OPENZOOM_HAS_TTS
    if (text.trimmed().isEmpty()) {
        return;
    }
    if (!tts_) {
        const QStringList engines = QTextToSpeech::availableEngines();
        auto availableEngine = [&engines](const QString& requested) {
            for (const QString& engine : engines) {
                if (engine.compare(requested, Qt::CaseInsensitive) == 0) {
                    return engine;
                }
            }
            return QString();
        };

        QString engine = availableEngine(config_.ttsEngine.trimmed());
        if (engine.isEmpty()) {
            engine = availableEngine(QStringLiteral("winrt"));
        }
        tts_ = engine.isEmpty() ? new QTextToSpeech(this)
                                : new QTextToSpeech(engine, this);
        if (tts_->state() == QTextToSpeech::Error &&
            tts_->engine().compare(QStringLiteral("winrt"), Qt::CaseInsensitive) == 0) {
            const QString fallbackEngine = availableEngine(QStringLiteral("sapi"));
            if (!fallbackEngine.isEmpty()) {
                tts_->setEngine(fallbackEngine);
            }
        }
    }

    tts_->setRate(std::clamp(config_.ttsRate, -1.0, 1.0));
    if (!config_.ttsVoiceName.trimmed().isEmpty()) {
        const QList<QVoice> voices = tts_->findVoices();
        for (const QVoice& voice : voices) {
            const bool nameMatches = voice.name() == config_.ttsVoiceName;
            const bool localeMatches = config_.ttsVoiceLocale.trimmed().isEmpty() ||
                                       voice.locale().name() == config_.ttsVoiceLocale;
            if (nameMatches && localeMatches) {
                tts_->setVoice(voice);
                break;
            }
        }
    }
    tts_->stop();
    tts_->say(text);
#else
    Q_UNUSED(text);
#endif
}

void AssistiveRuntime::StopSpeech()
{
#if OPENZOOM_HAS_TTS
    if (tts_) {
        tts_->stop();
    }
#endif
}

} // namespace openzoom

#endif // _WIN32
