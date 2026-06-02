#ifdef _WIN32

#include "openzoom/common/assistive_runtime.hpp"

#include <QBuffer>
#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QStandardPaths>
#include <QStringList>
#include <QTemporaryFile>
#include <QUrl>

namespace openzoom {

namespace {

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

    connect(ocrProcess_.get(),
            qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this,
            [this](int exitCode, QProcess::ExitStatus exitStatus) {
                const QString stdoutText = SanitizeText(QString::fromUtf8(ocrProcess_->readAllStandardOutput()));
                const QString stderrText = SanitizeText(QString::fromUtf8(ocrProcess_->readAllStandardError()));
                if (!pendingOcrImagePath_.isEmpty()) {
                    QFile::remove(pendingOcrImagePath_);
                    pendingOcrImagePath_.clear();
                }
                if (exitStatus == QProcess::NormalExit && exitCode == 0) {
                    FinishOcrSuccess(stdoutText);
                    return;
                }
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
                if (!pendingOcrImagePath_.isEmpty()) {
                    QFile::remove(pendingOcrImagePath_);
                    pendingOcrImagePath_.clear();
                }
                QString errorText;
                switch (error) {
                case QProcess::FailedToStart:
                    errorText = QStringLiteral("tesseract not found. Install it or set OPENZOOM_TESSERACT_PATH.");
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

void AssistiveRuntime::SetModes(bool ocrEnabled, bool vlmEnabled)
{
    if (!ocrEnabled && ocrProcess_ && ocrProcess_->state() != QProcess::NotRunning) {
        ocrProcess_->kill();
        ocrProcess_->waitForFinished(250);
    }
    if (!vlmEnabled && activeReply_) {
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
    if (!ocrEnabled_) {
        ocrText_.clear();
        ocrStatus_.clear();
    } else if (ocrText_.isEmpty() && ocrStatus_.isEmpty()) {
        ocrStatus_ = QStringLiteral("OCR ready. Install tesseract or set OPENZOOM_TESSERACT_PATH if detection fails.");
    }

    if (!vlmEnabled_) {
        vlmText_.clear();
        vlmStatus_.clear();
    } else if (vlmText_.isEmpty() && vlmStatus_.isEmpty()) {
        if (VlmConfigured()) {
            vlmStatus_ = QStringLiteral("VLM ready.");
        } else {
            vlmStatus_ = QStringLiteral("VLM not configured. Set OPENZOOM_VLM_API_URL, OPENZOOM_VLM_API_KEY, and OPENZOOM_VLM_MODEL.");
        }
    }

    RefreshOverlay();
}

bool AssistiveRuntime::WantsAnalysis() const
{
    return (ocrEnabled_ && !ocrHardUnavailable_) || (vlmEnabled_ && !vlmHardUnavailable_);
}

bool AssistiveRuntime::IsBusy() const
{
    const bool ocrBusy = ocrProcess_ && ocrProcess_->state() != QProcess::NotRunning;
    return ocrBusy || activeReply_ != nullptr;
}

void AssistiveRuntime::SubmitFrame(const uint8_t* bgraData, int width, int height)
{
    if (!WantsAnalysis() || !bgraData || width <= 0 || height <= 0) {
        return;
    }

    if (ocrEnabled_ && ocrProcess_ && ocrProcess_->state() == QProcess::NotRunning) {
        StartOcr(bgraData, width, height);
    }
    if (vlmEnabled_ && activeReply_ == nullptr) {
        StartVlm(bgraData, width, height);
    }
}

void AssistiveRuntime::RefreshOverlay()
{
    QStringList sections;
    if (ocrEnabled_) {
        QString text = ocrText_.isEmpty() ? ocrStatus_ : ocrText_;
        if (!text.isEmpty()) {
            sections.push_back(QStringLiteral("OCR\n%1").arg(text));
        }
    }
    if (vlmEnabled_) {
        QString text = vlmText_.isEmpty() ? vlmStatus_ : vlmText_;
        if (!text.isEmpty()) {
            sections.push_back(QStringLiteral("Scene Explain\n%1").arg(text));
        }
    }

    const QString body = sections.join(QStringLiteral("\n\n"));
    emit OverlayUpdated(QStringLiteral("Assistive View"), body, !body.isEmpty());
}

QString AssistiveRuntime::TesseractProgram() const
{
    const QString configured = qEnvironmentVariable("OPENZOOM_TESSERACT_PATH");
    if (!configured.trimmed().isEmpty()) {
        return configured.trimmed();
    }
    return QStringLiteral("tesseract");
}

bool AssistiveRuntime::VlmConfigured() const
{
    return !qEnvironmentVariable("OPENZOOM_VLM_API_URL").trimmed().isEmpty() &&
           !qEnvironmentVariable("OPENZOOM_VLM_API_KEY").trimmed().isEmpty() &&
           !qEnvironmentVariable("OPENZOOM_VLM_MODEL").trimmed().isEmpty();
}

void AssistiveRuntime::StartOcr(const uint8_t* bgraData, int width, int height)
{
    if (ocrHardUnavailable_) {
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
    ocrStatus_ = QStringLiteral("Running OCR...");
    RefreshOverlay();

    ocrProcess_->setProgram(TesseractProgram());
    ocrProcess_->setArguments({imagePath, QStringLiteral("stdout"), QStringLiteral("--psm"), QStringLiteral("6")});
    ocrProcess_->start();
}

void AssistiveRuntime::StartVlm(const uint8_t* bgraData, int width, int height)
{
    if (vlmHardUnavailable_) {
        return;
    }
    if (!VlmConfigured()) {
        vlmHardUnavailable_ = true;
        FinishVlmError(QStringLiteral("VLM not configured. Set OPENZOOM_VLM_API_URL, OPENZOOM_VLM_API_KEY, and OPENZOOM_VLM_MODEL."));
        return;
    }

    QImage frameImage(bgraData, width, height, width * 4, QImage::Format_ARGB32);
    QImage copy = frameImage.copy();
    QByteArray jpegBytes;
    QBuffer buffer(&jpegBytes);
    buffer.open(QIODevice::WriteOnly);
    if (!copy.save(&buffer, "JPG", 82)) {
        FinishVlmError(QStringLiteral("Failed to encode frame for VLM request."));
        return;
    }

    const QString apiUrl = qEnvironmentVariable("OPENZOOM_VLM_API_URL").trimmed();
    const QString apiKey = qEnvironmentVariable("OPENZOOM_VLM_API_KEY").trimmed();
    const QString model = qEnvironmentVariable("OPENZOOM_VLM_MODEL").trimmed();
    QString prompt = qEnvironmentVariable("OPENZOOM_VLM_PROMPT").trimmed();
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

    QJsonObject requestBody;
    requestBody.insert(QStringLiteral("model"), model);
    requestBody.insert(QStringLiteral("messages"), QJsonArray{message});
    requestBody.insert(QStringLiteral("max_tokens"), 180);

    QNetworkRequest request{QUrl(apiUrl)};
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    request.setRawHeader("Authorization", QStringLiteral("Bearer %1").arg(apiKey).toUtf8());

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
            FinishVlmError(QStringLiteral("VLM request failed: %1").arg(reply->errorString()));
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

void AssistiveRuntime::FinishOcrSuccess(const QString& text)
{
    ocrText_ = TruncateText(SanitizeText(text), 700);
    if (ocrText_.isEmpty()) {
        ocrStatus_ = QStringLiteral("OCR found no readable text.");
    } else {
        ocrStatus_.clear();
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
    vlmText_ = TruncateText(SanitizeText(text), 700);
    if (vlmText_.isEmpty()) {
        vlmStatus_ = QStringLiteral("VLM returned an empty description.");
    } else {
        vlmStatus_.clear();
    }
    RefreshOverlay();
}

void AssistiveRuntime::FinishVlmError(const QString& errorText)
{
    vlmText_.clear();
    vlmStatus_ = SanitizeText(errorText);
    RefreshOverlay();
}

} // namespace openzoom

#endif // _WIN32
