#ifdef _WIN32

#include "openzoom/app/settings_store.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QSaveFile>
#include <QStandardPaths>
#include <QVariant>

#include <algorithm>
#include <cmath>

namespace openzoom::settings {

namespace {

int SnapRotation(int turns)
{
    int value = turns % 4;
    if (value < 0) {
        value += 4;
    }
    return value;
}

} // namespace

QString ResolveSettingsPath()
{
    QString basePath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    if (basePath.isEmpty()) {
        basePath = QCoreApplication::applicationDirPath();
    }
    QDir dir(basePath);
    return dir.filePath(QStringLiteral("settings.json"));
}

void EnsureSettingsDirectory(const QString& path)
{
    QFileInfo info(path);
    QDir dir = info.dir();
    if (!dir.exists()) {
        dir.mkpath(QStringLiteral("."));
    }
}

std::optional<PersistentSettings> Load(const QString& path)
{
    QFile file(path);
    if (!file.exists()) {
        return std::nullopt;
    }

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return std::nullopt;
    }

    const QByteArray data = file.readAll();
    file.close();

    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        return std::nullopt;
    }

    const QJsonObject root = doc.object();

    PersistentSettings settings;
    settings.cameraIndex = root.value(QStringLiteral("cameraIndex")).toInt(settings.cameraIndex);

    const QJsonObject blackWhite = root.value(QStringLiteral("blackWhite")).toObject();
    settings.blackWhiteEnabled = blackWhite.value(QStringLiteral("enabled")).toBool(settings.blackWhiteEnabled);
    settings.blackWhiteThreshold = static_cast<float>(blackWhite.value(QStringLiteral("threshold")).toDouble(settings.blackWhiteThreshold));
    settings.blackWhiteThreshold = std::clamp(settings.blackWhiteThreshold, 0.0f, 1.0f);

    const QJsonObject zoom = root.value(QStringLiteral("zoom")).toObject();
    settings.zoomEnabled = zoom.value(QStringLiteral("enabled")).toBool(settings.zoomEnabled);
    settings.zoomAmount = static_cast<float>(zoom.value(QStringLiteral("amount")).toDouble(settings.zoomAmount));
    if (!(settings.zoomAmount > 0.0f) || !std::isfinite(settings.zoomAmount)) {
        settings.zoomAmount = 1.0f;
    }
    settings.zoomCenterX = static_cast<float>(zoom.value(QStringLiteral("centerX")).toDouble(settings.zoomCenterX));
    settings.zoomCenterY = static_cast<float>(zoom.value(QStringLiteral("centerY")).toDouble(settings.zoomCenterY));
    settings.zoomCenterX = std::clamp(settings.zoomCenterX, 0.0f, 1.0f);
    settings.zoomCenterY = std::clamp(settings.zoomCenterY, 0.0f, 1.0f);

    const QJsonObject blur = root.value(QStringLiteral("blur")).toObject();
    settings.blurEnabled = blur.value(QStringLiteral("enabled")).toBool(settings.blurEnabled);
    settings.blurSigma = static_cast<float>(blur.value(QStringLiteral("sigma")).toDouble(settings.blurSigma));
    if (!(settings.blurSigma > 0.0f) || !std::isfinite(settings.blurSigma)) {
        settings.blurSigma = 1.0f;
    }
    settings.blurRadius = blur.value(QStringLiteral("radius")).toInt(settings.blurRadius);

    const QJsonObject temporal = root.value(QStringLiteral("temporalSmooth")).toObject();
    settings.temporalSmoothEnabled = temporal.value(QStringLiteral("enabled")).toBool(settings.temporalSmoothEnabled);
    settings.temporalSmoothAlpha = static_cast<float>(temporal.value(QStringLiteral("alpha")).toDouble(settings.temporalSmoothAlpha));
    settings.temporalSmoothAlpha = std::clamp(settings.temporalSmoothAlpha, 0.0f, 1.0f);

    const QJsonObject spatial = root.value(QStringLiteral("spatialSharpen")).toObject();
    settings.spatialSharpenEnabled = spatial.value(QStringLiteral("enabled")).toBool(settings.spatialSharpenEnabled);
    const QString backend = spatial.value(QStringLiteral("backend")).toString();
    if (backend.compare(QStringLiteral("fsr"), Qt::CaseInsensitive) == 0) {
        settings.spatialUpscaler = 0;
    } else {
        settings.spatialUpscaler = 1;
    }
    settings.spatialSharpness = static_cast<float>(spatial.value(QStringLiteral("sharpness")).toDouble(settings.spatialSharpness));
    settings.spatialSharpness = std::clamp(settings.spatialSharpness, 0.0f, 1.0f);

    settings.debugView = root.value(QStringLiteral("debugView")).toBool(settings.debugView);
    settings.focusMarker = root.value(QStringLiteral("focusMarker")).toBool(settings.focusMarker);
    settings.virtualJoystick = root.value(QStringLiteral("virtualJoystick")).toBool(settings.virtualJoystick);
    settings.controlsCollapsed = root.value(QStringLiteral("controlsCollapsed")).toBool(settings.controlsCollapsed);
    settings.rotationQuarterTurns = SnapRotation(root.value(QStringLiteral("rotationQuarterTurns")).toInt(settings.rotationQuarterTurns));

    return settings;
}

bool Save(const QString& path, const PersistentSettings& settings)
{
    EnsureSettingsDirectory(path);

    QJsonObject root;
    root.insert(QStringLiteral("version"), 1);
    root.insert(QStringLiteral("cameraIndex"), settings.cameraIndex);

    QJsonObject blackWhite;
    blackWhite.insert(QStringLiteral("enabled"), settings.blackWhiteEnabled);
    blackWhite.insert(QStringLiteral("threshold"), settings.blackWhiteThreshold);
    root.insert(QStringLiteral("blackWhite"), blackWhite);

    QJsonObject zoom;
    zoom.insert(QStringLiteral("enabled"), settings.zoomEnabled);
    zoom.insert(QStringLiteral("amount"), settings.zoomAmount);
    zoom.insert(QStringLiteral("centerX"), settings.zoomCenterX);
    zoom.insert(QStringLiteral("centerY"), settings.zoomCenterY);
    root.insert(QStringLiteral("zoom"), zoom);

    QJsonObject blur;
    blur.insert(QStringLiteral("enabled"), settings.blurEnabled);
    blur.insert(QStringLiteral("sigma"), settings.blurSigma);
    blur.insert(QStringLiteral("radius"), settings.blurRadius);
    root.insert(QStringLiteral("blur"), blur);

    QJsonObject temporal;
    temporal.insert(QStringLiteral("enabled"), settings.temporalSmoothEnabled);
    temporal.insert(QStringLiteral("alpha"), settings.temporalSmoothAlpha);
    root.insert(QStringLiteral("temporalSmooth"), temporal);

    QJsonObject spatial;
    spatial.insert(QStringLiteral("enabled"), settings.spatialSharpenEnabled);
    spatial.insert(QStringLiteral("backend"), settings.spatialUpscaler == 0 ? QStringLiteral("fsr") : QStringLiteral("nis"));
    spatial.insert(QStringLiteral("sharpness"), settings.spatialSharpness);
    root.insert(QStringLiteral("spatialSharpen"), spatial);

    root.insert(QStringLiteral("debugView"), settings.debugView);
    root.insert(QStringLiteral("focusMarker"), settings.focusMarker);
    root.insert(QStringLiteral("virtualJoystick"), settings.virtualJoystick);
    root.insert(QStringLiteral("controlsCollapsed"), settings.controlsCollapsed);
    root.insert(QStringLiteral("rotationQuarterTurns"), SnapRotation(settings.rotationQuarterTurns));

    const QJsonDocument doc(root);
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    file.write(doc.toJson(QJsonDocument::Indented));
    return file.commit();
}

} // namespace openzoom::settings

#endif // _WIN32

