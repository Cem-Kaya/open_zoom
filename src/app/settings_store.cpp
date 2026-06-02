#ifdef _WIN32

#include "openzoom/app/settings_store.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QStandardPaths>

#include <algorithm>
#include <cmath>
#include <utility>

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

float Clamp01(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

AdvancedConfig MakeConfig(QString id,
                          QString name,
                          QString description,
                          bool blackWhiteEnabled,
                          float blackWhiteThreshold,
                          bool zoomEnabled,
                          float zoomAmount,
                          float zoomCenterX,
                          float zoomCenterY,
                          bool blurEnabled,
                          float blurSigma,
                          int blurRadius,
                          bool temporalSmoothEnabled,
                          float temporalSmoothAlpha,
                          bool spatialSharpenEnabled,
                          int spatialUpscaler,
                          float spatialSharpness,
                          bool debugView,
                          bool focusMarker,
                          int rotationQuarterTurns,
                          bool ocrAssistEnabled,
                          bool vlmAssistEnabled,
                          bool assistiveOverlayEnabled)
{
    AdvancedConfig config;
    config.id = std::move(id);
    config.name = std::move(name);
    config.description = std::move(description);
    config.blackWhiteEnabled = blackWhiteEnabled;
    config.blackWhiteThreshold = Clamp01(blackWhiteThreshold);
    config.zoomEnabled = zoomEnabled;
    config.zoomAmount = std::max(1.0f, zoomAmount);
    config.zoomCenterX = Clamp01(zoomCenterX);
    config.zoomCenterY = Clamp01(zoomCenterY);
    config.blurEnabled = blurEnabled;
    config.blurSigma = std::max(0.1f, blurSigma);
    config.blurRadius = std::max(0, blurRadius);
    config.temporalSmoothEnabled = temporalSmoothEnabled;
    config.temporalSmoothAlpha = Clamp01(temporalSmoothAlpha);
    config.spatialSharpenEnabled = spatialSharpenEnabled;
    config.spatialUpscaler = (spatialUpscaler == 0) ? 0 : 1;
    config.spatialSharpness = Clamp01(spatialSharpness);
    config.debugView = debugView;
    config.focusMarker = focusMarker;
    config.rotationQuarterTurns = SnapRotation(rotationQuarterTurns);
    config.ocrAssistEnabled = ocrAssistEnabled;
    config.vlmAssistEnabled = vlmAssistEnabled;
    config.assistiveOverlayEnabled = assistiveOverlayEnabled;
    return config;
}

const std::vector<AdvancedConfig>& BuiltInConfigsStorage()
{
    static const std::vector<AdvancedConfig> kConfigs = {
        MakeConfig(QStringLiteral("preset-reading-config"),
                   QStringLiteral("Reading"),
                   QStringLiteral("Balanced magnification for reading text comfortably."),
                   false, 0.5f, true, 1.6f, 0.5f, 0.5f,
                   false, 1.0f, 3,
                   true, 0.25f,
                   true, 1, 0.25f,
                   false, false, 0,
                   false, false, true),
        MakeConfig(QStringLiteral("preset-high-contrast-config"),
                   QStringLiteral("High Contrast"),
                   QStringLiteral("Thresholded monochrome for strong text separation."),
                   true, 0.62f, true, 1.5f, 0.5f, 0.5f,
                   false, 1.0f, 3,
                   true, 0.18f,
                   false, 1, 0.25f,
                   false, false, 0,
                   false, false, true),
        MakeConfig(QStringLiteral("preset-steady-text-config"),
                   QStringLiteral("Steady Text"),
                   QStringLiteral("Favors stability and reduced shimmer during movement."),
                   false, 0.5f, true, 1.45f, 0.5f, 0.5f,
                   false, 1.0f, 3,
                   true, 0.45f,
                   false, 1, 0.25f,
                   false, false, 0,
                   false, false, true),
        MakeConfig(QStringLiteral("preset-sharp-text-config"),
                   QStringLiteral("Sharp Text"),
                   QStringLiteral("Stronger sharpening for crisp UI and document edges."),
                   false, 0.5f, true, 1.55f, 0.5f, 0.5f,
                   false, 1.0f, 3,
                   false, 0.25f,
                   true, 1, 0.45f,
                   false, false, 0,
                   false, false, true),
        MakeConfig(QStringLiteral("preset-large-zoom-config"),
                   QStringLiteral("Large Zoom"),
                   QStringLiteral("Prioritizes magnification with a simpler processing stack."),
                   false, 0.5f, true, 2.3f, 0.5f, 0.5f,
                   false, 1.0f, 3,
                   false, 0.25f,
                   false, 1, 0.25f,
                   false, true, 0,
                   false, false, true),
        MakeConfig(QStringLiteral("preset-low-light-config"),
                   QStringLiteral("Low Light"),
                   QStringLiteral("Softer smoothing-oriented preset for noisy scenes."),
                   false, 0.5f, true, 1.35f, 0.5f, 0.5f,
                   true, 1.6f, 3,
                   true, 0.35f,
                   false, 1, 0.25f,
                   false, false, 0,
                   false, false, true),
        MakeConfig(QStringLiteral("preset-ocr-assist-config"),
                   QStringLiteral("OCR Assist"),
                   QStringLiteral("Reading preset with OCR hooks enabled for future overlays."),
                   false, 0.5f, true, 1.7f, 0.5f, 0.5f,
                   false, 1.0f, 3,
                   true, 0.28f,
                   true, 1, 0.28f,
                   false, false, 0,
                   true, false, true),
        MakeConfig(QStringLiteral("preset-scene-explain-config"),
                   QStringLiteral("Scene Explain"),
                   QStringLiteral("Context-oriented preset with VLM hooks enabled for scene summaries."),
                   false, 0.5f, true, 1.25f, 0.5f, 0.5f,
                   false, 1.0f, 3,
                   true, 0.22f,
                   false, 1, 0.25f,
                   false, false, 0,
                   false, true, true),
    };
    return kConfigs;
}

const std::vector<PresetDefinition>& BuiltInPresetsStorage()
{
    static const std::vector<PresetDefinition> kPresets = {
        {QStringLiteral("preset-reading"), QStringLiteral("Reading"),
         QStringLiteral("Balanced magnification for general reading."), QStringLiteral("preset-reading-config"), true},
        {QStringLiteral("preset-high-contrast"), QStringLiteral("High Contrast"),
         QStringLiteral("Monochrome thresholding for stronger contrast."), QStringLiteral("preset-high-contrast-config"), true},
        {QStringLiteral("preset-steady-text"), QStringLiteral("Steady Text"),
         QStringLiteral("More smoothing for less shimmer while moving."), QStringLiteral("preset-steady-text-config"), true},
        {QStringLiteral("preset-sharp-text"), QStringLiteral("Sharp Text"),
         QStringLiteral("Sharper rendering for crisp text edges."), QStringLiteral("preset-sharp-text-config"), true},
        {QStringLiteral("preset-large-zoom"), QStringLiteral("Large Zoom"),
         QStringLiteral("Higher magnification with fewer extra effects."), QStringLiteral("preset-large-zoom-config"), true},
        {QStringLiteral("preset-low-light"), QStringLiteral("Low Light"),
         QStringLiteral("Gentler smoothing for dim or noisy scenes."), QStringLiteral("preset-low-light-config"), true},
        {QStringLiteral("preset-ocr-assist"), QStringLiteral("OCR Assist"),
         QStringLiteral("Preset reserved for text extraction and readable overlays."), QStringLiteral("preset-ocr-assist-config"), true},
        {QStringLiteral("preset-scene-explain"), QStringLiteral("Scene Explain"),
         QStringLiteral("Preset reserved for future VLM-based scene summaries."), QStringLiteral("preset-scene-explain-config"), true},
    };
    return kPresets;
}

QJsonObject ConfigToJson(const AdvancedConfig& config)
{
    QJsonObject object;
    object.insert(QStringLiteral("id"), config.id);
    object.insert(QStringLiteral("name"), config.name);
    object.insert(QStringLiteral("description"), config.description);
    object.insert(QStringLiteral("blackWhiteEnabled"), config.blackWhiteEnabled);
    object.insert(QStringLiteral("blackWhiteThreshold"), config.blackWhiteThreshold);
    object.insert(QStringLiteral("zoomEnabled"), config.zoomEnabled);
    object.insert(QStringLiteral("zoomAmount"), config.zoomAmount);
    object.insert(QStringLiteral("zoomCenterX"), config.zoomCenterX);
    object.insert(QStringLiteral("zoomCenterY"), config.zoomCenterY);
    object.insert(QStringLiteral("blurEnabled"), config.blurEnabled);
    object.insert(QStringLiteral("blurSigma"), config.blurSigma);
    object.insert(QStringLiteral("blurRadius"), config.blurRadius);
    object.insert(QStringLiteral("temporalSmoothEnabled"), config.temporalSmoothEnabled);
    object.insert(QStringLiteral("temporalSmoothAlpha"), config.temporalSmoothAlpha);
    object.insert(QStringLiteral("spatialSharpenEnabled"), config.spatialSharpenEnabled);
    object.insert(QStringLiteral("spatialUpscaler"), config.spatialUpscaler);
    object.insert(QStringLiteral("spatialSharpness"), config.spatialSharpness);
    object.insert(QStringLiteral("debugView"), config.debugView);
    object.insert(QStringLiteral("focusMarker"), config.focusMarker);
    object.insert(QStringLiteral("rotationQuarterTurns"), config.rotationQuarterTurns);
    object.insert(QStringLiteral("ocrAssistEnabled"), config.ocrAssistEnabled);
    object.insert(QStringLiteral("vlmAssistEnabled"), config.vlmAssistEnabled);
    object.insert(QStringLiteral("assistiveOverlayEnabled"), config.assistiveOverlayEnabled);
    return object;
}

AdvancedConfig ConfigFromJson(const QJsonObject& object, const AdvancedConfig& defaults = AdvancedConfig{})
{
    AdvancedConfig config = defaults;
    config.id = object.value(QStringLiteral("id")).toString(config.id);
    config.name = object.value(QStringLiteral("name")).toString(config.name);
    config.description = object.value(QStringLiteral("description")).toString(config.description);
    config.blackWhiteEnabled = object.value(QStringLiteral("blackWhiteEnabled")).toBool(config.blackWhiteEnabled);
    config.blackWhiteThreshold = Clamp01(static_cast<float>(object.value(QStringLiteral("blackWhiteThreshold")).toDouble(config.blackWhiteThreshold)));
    config.zoomEnabled = object.value(QStringLiteral("zoomEnabled")).toBool(config.zoomEnabled);
    config.zoomAmount = std::max(1.0f, static_cast<float>(object.value(QStringLiteral("zoomAmount")).toDouble(config.zoomAmount)));
    config.zoomCenterX = Clamp01(static_cast<float>(object.value(QStringLiteral("zoomCenterX")).toDouble(config.zoomCenterX)));
    config.zoomCenterY = Clamp01(static_cast<float>(object.value(QStringLiteral("zoomCenterY")).toDouble(config.zoomCenterY)));
    config.blurEnabled = object.value(QStringLiteral("blurEnabled")).toBool(config.blurEnabled);
    config.blurSigma = std::max(0.1f, static_cast<float>(object.value(QStringLiteral("blurSigma")).toDouble(config.blurSigma)));
    config.blurRadius = std::max(0, object.value(QStringLiteral("blurRadius")).toInt(config.blurRadius));
    config.temporalSmoothEnabled = object.value(QStringLiteral("temporalSmoothEnabled")).toBool(config.temporalSmoothEnabled);
    config.temporalSmoothAlpha = Clamp01(static_cast<float>(object.value(QStringLiteral("temporalSmoothAlpha")).toDouble(config.temporalSmoothAlpha)));
    config.spatialSharpenEnabled = object.value(QStringLiteral("spatialSharpenEnabled")).toBool(config.spatialSharpenEnabled);
    config.spatialUpscaler = object.value(QStringLiteral("spatialUpscaler")).toInt(config.spatialUpscaler) == 0 ? 0 : 1;
    config.spatialSharpness = Clamp01(static_cast<float>(object.value(QStringLiteral("spatialSharpness")).toDouble(config.spatialSharpness)));
    config.debugView = object.value(QStringLiteral("debugView")).toBool(config.debugView);
    config.focusMarker = object.value(QStringLiteral("focusMarker")).toBool(config.focusMarker);
    config.rotationQuarterTurns = SnapRotation(object.value(QStringLiteral("rotationQuarterTurns")).toInt(config.rotationQuarterTurns));
    config.ocrAssistEnabled = object.value(QStringLiteral("ocrAssistEnabled")).toBool(config.ocrAssistEnabled);
    config.vlmAssistEnabled = object.value(QStringLiteral("vlmAssistEnabled")).toBool(config.vlmAssistEnabled);
    config.assistiveOverlayEnabled = object.value(QStringLiteral("assistiveOverlayEnabled")).toBool(config.assistiveOverlayEnabled);
    return config;
}

QJsonObject PresetToJson(const PresetDefinition& preset)
{
    QJsonObject object;
    object.insert(QStringLiteral("id"), preset.id);
    object.insert(QStringLiteral("name"), preset.name);
    object.insert(QStringLiteral("description"), preset.description);
    object.insert(QStringLiteral("configId"), preset.configId);
    object.insert(QStringLiteral("isBuiltIn"), preset.isBuiltIn);
    return object;
}

PresetDefinition PresetFromJson(const QJsonObject& object)
{
    PresetDefinition preset;
    preset.id = object.value(QStringLiteral("id")).toString();
    preset.name = object.value(QStringLiteral("name")).toString();
    preset.description = object.value(QStringLiteral("description")).toString();
    preset.configId = object.value(QStringLiteral("configId")).toString();
    preset.isBuiltIn = object.value(QStringLiteral("isBuiltIn")).toBool(false);
    return preset;
}

AdvancedConfig LegacyConfigFromRoot(const QJsonObject& root)
{
    AdvancedConfig config = BuiltInConfigsStorage().front();
    config.id = QStringLiteral("legacy-current");
    config.name = QStringLiteral("Current Setup");
    config.description = QStringLiteral("Migrated from the legacy flat settings format.");

    const QJsonObject blackWhite = root.value(QStringLiteral("blackWhite")).toObject();
    config.blackWhiteEnabled = blackWhite.value(QStringLiteral("enabled")).toBool(config.blackWhiteEnabled);
    config.blackWhiteThreshold = Clamp01(static_cast<float>(blackWhite.value(QStringLiteral("threshold")).toDouble(config.blackWhiteThreshold)));

    const QJsonObject zoom = root.value(QStringLiteral("zoom")).toObject();
    config.zoomEnabled = zoom.value(QStringLiteral("enabled")).toBool(config.zoomEnabled);
    config.zoomAmount = std::max(1.0f, static_cast<float>(zoom.value(QStringLiteral("amount")).toDouble(config.zoomAmount)));
    config.zoomCenterX = Clamp01(static_cast<float>(zoom.value(QStringLiteral("centerX")).toDouble(config.zoomCenterX)));
    config.zoomCenterY = Clamp01(static_cast<float>(zoom.value(QStringLiteral("centerY")).toDouble(config.zoomCenterY)));

    const QJsonObject blur = root.value(QStringLiteral("blur")).toObject();
    config.blurEnabled = blur.value(QStringLiteral("enabled")).toBool(config.blurEnabled);
    config.blurSigma = std::max(0.1f, static_cast<float>(blur.value(QStringLiteral("sigma")).toDouble(config.blurSigma)));
    config.blurRadius = std::max(0, blur.value(QStringLiteral("radius")).toInt(config.blurRadius));

    const QJsonObject temporal = root.value(QStringLiteral("temporalSmooth")).toObject();
    config.temporalSmoothEnabled = temporal.value(QStringLiteral("enabled")).toBool(config.temporalSmoothEnabled);
    config.temporalSmoothAlpha = Clamp01(static_cast<float>(temporal.value(QStringLiteral("alpha")).toDouble(config.temporalSmoothAlpha)));

    const QJsonObject spatial = root.value(QStringLiteral("spatialSharpen")).toObject();
    config.spatialSharpenEnabled = spatial.value(QStringLiteral("enabled")).toBool(config.spatialSharpenEnabled);
    const QString backend = spatial.value(QStringLiteral("backend")).toString();
    config.spatialUpscaler = (backend.compare(QStringLiteral("fsr"), Qt::CaseInsensitive) == 0) ? 0 : 1;
    config.spatialSharpness = Clamp01(static_cast<float>(spatial.value(QStringLiteral("sharpness")).toDouble(config.spatialSharpness)));

    config.debugView = root.value(QStringLiteral("debugView")).toBool(config.debugView);
    config.focusMarker = root.value(QStringLiteral("focusMarker")).toBool(config.focusMarker);
    config.rotationQuarterTurns = SnapRotation(root.value(QStringLiteral("rotationQuarterTurns")).toInt(config.rotationQuarterTurns));
    return config;
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

    const int version = root.value(QStringLiteral("version")).toInt(1);
    if (version <= 1 && root.contains(QStringLiteral("zoom"))) {
        settings.virtualJoystick = root.value(QStringLiteral("virtualJoystick")).toBool(settings.virtualJoystick);
        settings.controlsCollapsed = root.value(QStringLiteral("controlsCollapsed")).toBool(settings.controlsCollapsed);
        settings.currentConfig = LegacyConfigFromRoot(root);
        settings.selectedPresetId.clear();
        return settings;
    }

    const QJsonObject ui = root.value(QStringLiteral("ui")).toObject();
    settings.virtualJoystick = ui.value(QStringLiteral("virtualJoystick")).toBool(settings.virtualJoystick);
    settings.controlsCollapsed = ui.value(QStringLiteral("controlsCollapsed")).toBool(settings.controlsCollapsed);
    settings.selectedPresetId = ui.value(QStringLiteral("selectedPresetId")).toString();

    const AdvancedConfig defaultConfig = BuiltInConfigsStorage().front();
    settings.currentConfig = ConfigFromJson(root.value(QStringLiteral("currentConfig")).toObject(), defaultConfig);
    if (settings.currentConfig.id.isEmpty()) {
        settings.currentConfig.id = QStringLiteral("current-live");
    }
    if (settings.currentConfig.name.isEmpty()) {
        settings.currentConfig.name = QStringLiteral("Current Setup");
    }

    const QJsonArray configArray = root.value(QStringLiteral("customConfigs")).toArray();
    settings.customConfigs.reserve(static_cast<std::size_t>(configArray.size()));
    for (const QJsonValue& value : configArray) {
        if (!value.isObject()) {
            continue;
        }
        AdvancedConfig config = ConfigFromJson(value.toObject());
        if (!config.id.isEmpty()) {
            settings.customConfigs.push_back(std::move(config));
        }
    }

    const QJsonArray presetArray = root.value(QStringLiteral("customPresets")).toArray();
    settings.customPresets.reserve(static_cast<std::size_t>(presetArray.size()));
    for (const QJsonValue& value : presetArray) {
        if (!value.isObject()) {
            continue;
        }
        PresetDefinition preset = PresetFromJson(value.toObject());
        if (!preset.id.isEmpty() && !preset.configId.isEmpty()) {
            settings.customPresets.push_back(std::move(preset));
        }
    }

    return settings;
}

bool Save(const QString& path, const PersistentSettings& settings)
{
    EnsureSettingsDirectory(path);

    QJsonObject root;
    root.insert(QStringLiteral("version"), 2);
    root.insert(QStringLiteral("cameraIndex"), settings.cameraIndex);

    QJsonObject ui;
    ui.insert(QStringLiteral("virtualJoystick"), settings.virtualJoystick);
    ui.insert(QStringLiteral("controlsCollapsed"), settings.controlsCollapsed);
    ui.insert(QStringLiteral("selectedPresetId"), settings.selectedPresetId);
    root.insert(QStringLiteral("ui"), ui);

    root.insert(QStringLiteral("currentConfig"), ConfigToJson(settings.currentConfig));

    QJsonArray configArray;
    for (const AdvancedConfig& config : settings.customConfigs) {
        configArray.append(ConfigToJson(config));
    }
    root.insert(QStringLiteral("customConfigs"), configArray);

    QJsonArray presetArray;
    for (const PresetDefinition& preset : settings.customPresets) {
        presetArray.append(PresetToJson(preset));
    }
    root.insert(QStringLiteral("customPresets"), presetArray);

    const QJsonDocument doc(root);
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    file.write(doc.toJson(QJsonDocument::Indented));
    return file.commit();
}

const std::vector<AdvancedConfig>& BuiltInConfigs()
{
    return BuiltInConfigsStorage();
}

const std::vector<PresetDefinition>& BuiltInPresets()
{
    return BuiltInPresetsStorage();
}

QString DefaultPresetId()
{
    return QStringLiteral("preset-reading");
}

const AdvancedConfig* FindAdvancedConfigById(const QString& configId,
                                             const std::vector<AdvancedConfig>& customConfigs)
{
    for (const AdvancedConfig& config : BuiltInConfigsStorage()) {
        if (config.id == configId) {
            return &config;
        }
    }
    for (const AdvancedConfig& config : customConfigs) {
        if (config.id == configId) {
            return &config;
        }
    }
    return nullptr;
}

const PresetDefinition* FindPresetById(const QString& presetId,
                                       const std::vector<PresetDefinition>& customPresets)
{
    for (const PresetDefinition& preset : BuiltInPresetsStorage()) {
        if (preset.id == presetId) {
            return &preset;
        }
    }
    for (const PresetDefinition& preset : customPresets) {
        if (preset.id == presetId) {
            return &preset;
        }
    }
    return nullptr;
}

std::optional<AdvancedConfig> ResolveConfigForPreset(const QString& presetId,
                                                     const std::vector<AdvancedConfig>& customConfigs,
                                                     const std::vector<PresetDefinition>& customPresets)
{
    if (presetId.isEmpty()) {
        return std::nullopt;
    }
    const PresetDefinition* preset = FindPresetById(presetId, customPresets);
    if (!preset) {
        return std::nullopt;
    }
    const AdvancedConfig* config = FindAdvancedConfigById(preset->configId, customConfigs);
    if (!config) {
        return std::nullopt;
    }
    return *config;
}

bool AreConfigsEquivalent(const AdvancedConfig& lhs, const AdvancedConfig& rhs)
{
    auto almostEqual = [](float a, float b) {
        return std::abs(a - b) < 0.0005f;
    };

    return lhs.blackWhiteEnabled == rhs.blackWhiteEnabled &&
           almostEqual(lhs.blackWhiteThreshold, rhs.blackWhiteThreshold) &&
           lhs.zoomEnabled == rhs.zoomEnabled &&
           almostEqual(lhs.zoomAmount, rhs.zoomAmount) &&
           almostEqual(lhs.zoomCenterX, rhs.zoomCenterX) &&
           almostEqual(lhs.zoomCenterY, rhs.zoomCenterY) &&
           lhs.blurEnabled == rhs.blurEnabled &&
           almostEqual(lhs.blurSigma, rhs.blurSigma) &&
           lhs.blurRadius == rhs.blurRadius &&
           lhs.temporalSmoothEnabled == rhs.temporalSmoothEnabled &&
           almostEqual(lhs.temporalSmoothAlpha, rhs.temporalSmoothAlpha) &&
           lhs.spatialSharpenEnabled == rhs.spatialSharpenEnabled &&
           lhs.spatialUpscaler == rhs.spatialUpscaler &&
           almostEqual(lhs.spatialSharpness, rhs.spatialSharpness) &&
           lhs.debugView == rhs.debugView &&
           lhs.focusMarker == rhs.focusMarker &&
           SnapRotation(lhs.rotationQuarterTurns) == SnapRotation(rhs.rotationQuarterTurns) &&
           lhs.ocrAssistEnabled == rhs.ocrAssistEnabled &&
           lhs.vlmAssistEnabled == rhs.vlmAssistEnabled &&
           lhs.assistiveOverlayEnabled == rhs.assistiveOverlayEnabled;
}

} // namespace openzoom::settings

#endif // _WIN32
