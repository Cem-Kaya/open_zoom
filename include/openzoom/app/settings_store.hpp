#pragma once

#include <QString>

#include <optional>
#include <vector>

namespace openzoom::settings {

struct AdvancedConfig {
    QString id;
    QString name;
    QString description;
    bool blackWhiteEnabled{false};
    float blackWhiteThreshold{0.5f};
    bool zoomEnabled{false};
    float zoomAmount{1.0f};
    float zoomCenterX{0.5f};
    float zoomCenterY{0.5f};
    bool blurEnabled{false};
    float blurSigma{1.0f};
    int blurRadius{3};
    bool temporalSmoothEnabled{false};
    float temporalSmoothAlpha{0.25f};
    bool spatialSharpenEnabled{false};
    int spatialUpscaler{1};
    float spatialSharpness{0.25f};
    bool debugView{false};
    bool focusMarker{false};
    int rotationQuarterTurns{0};
    bool ocrAssistEnabled{false};
    bool vlmAssistEnabled{false};
    bool assistiveOverlayEnabled{true};
};

struct PresetDefinition {
    QString id;
    QString name;
    QString description;
    QString configId;
    bool isBuiltIn{false};
};

struct PersistentSettings {
    int cameraIndex{-1};
    bool virtualJoystick{false};
    bool controlsCollapsed{true};
    QString selectedPresetId;
    AdvancedConfig currentConfig{};
    std::vector<AdvancedConfig> customConfigs;
    std::vector<PresetDefinition> customPresets;
};

QString ResolveSettingsPath();
void EnsureSettingsDirectory(const QString& path);
std::optional<PersistentSettings> Load(const QString& path);
bool Save(const QString& path, const PersistentSettings& settings);

const std::vector<AdvancedConfig>& BuiltInConfigs();
const std::vector<PresetDefinition>& BuiltInPresets();
QString DefaultPresetId();

const AdvancedConfig* FindAdvancedConfigById(const QString& configId,
                                             const std::vector<AdvancedConfig>& customConfigs);
const PresetDefinition* FindPresetById(const QString& presetId,
                                       const std::vector<PresetDefinition>& customPresets);
std::optional<AdvancedConfig> ResolveConfigForPreset(const QString& presetId,
                                                     const std::vector<AdvancedConfig>& customConfigs,
                                                     const std::vector<PresetDefinition>& customPresets);
bool AreConfigsEquivalent(const AdvancedConfig& lhs, const AdvancedConfig& rhs);

} // namespace openzoom::settings
