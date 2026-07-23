#include "openzoom/app/settings_controller.hpp"

#include <QDateTime>

namespace openzoom {

namespace {

QString MakeCustomEntityId(const QString& prefix)
{
    return QStringLiteral("%1-%2")
        .arg(prefix)
        .arg(QDateTime::currentMSecsSinceEpoch());
}

} // namespace

SettingsController::SettingsController()
    : settingsPath_(settings::ResolveSettingsPath())
{
    if (auto loaded = settings::Load(settingsPath_)) {
        settings_ = std::move(*loaded);
        return;
    }

    settings_.selectedPresetId = settings::DefaultPresetId();
    if (auto defaultConfig = ResolvePreset(settings_.selectedPresetId)) {
        settings_.currentConfig = std::move(*defaultConfig);
    } else if (!settings::BuiltInConfigs().empty()) {
        settings_.currentConfig = settings::BuiltInConfigs().front();
    }
}

const settings::PersistentSettings& SettingsController::Settings() const noexcept
{
    return settings_;
}

settings::PersistentSettings& SettingsController::MutableSettings() noexcept
{
    return settings_;
}

settings::AdvancedConfig SettingsController::DecorateLiveConfig(
    settings::AdvancedConfig config) const
{
    const settings::AdvancedConfig& stored = settings_.currentConfig;
    config.id = stored.id.isEmpty() ? QStringLiteral("current-live") : stored.id;
    config.name = stored.name.isEmpty() ? QStringLiteral("Current Setup") : stored.name;
    config.description = stored.description.isEmpty()
        ? QStringLiteral("Live configuration derived from quick mode and advanced tuning.")
        : stored.description;

    if (settings_.selectedPresetId.isEmpty()) {
        config.id = QStringLiteral("current-live");
        config.name = QStringLiteral("Current Setup");
        config.description =
            QStringLiteral("Live configuration derived from quick mode and advanced tuning.");
    } else if (auto selected = ResolvePreset(settings_.selectedPresetId)) {
        config.id = selected->id;
        config.name = selected->name;
        config.description = selected->description;
    }
    return config;
}

QString SettingsController::MatchPreset(const settings::AdvancedConfig& current,
                                        bool preserveCurrentSelection)
{
    QString matchedPresetId;
    if (preserveCurrentSelection && ResolvePreset(settings_.selectedPresetId)) {
        matchedPresetId = settings_.selectedPresetId;
    }

    const auto matches = [this, &current](const settings::PresetDefinition& preset) {
        const auto config = ResolvePreset(preset.id);
        return config && settings::AreConfigsEquivalent(current, *config);
    };

    if (matchedPresetId.isEmpty()) {
        for (const settings::PresetDefinition& preset : settings::BuiltInPresets()) {
            if (matches(preset)) {
                matchedPresetId = preset.id;
                break;
            }
        }
    }
    if (matchedPresetId.isEmpty()) {
        for (const settings::PresetDefinition& preset : settings_.customPresets) {
            if (matches(preset)) {
                matchedPresetId = preset.id;
                break;
            }
        }
    }

    settings_.selectedPresetId = matchedPresetId;
    settings_.currentConfig = DecorateLiveConfig(current);
    return matchedPresetId;
}

std::optional<settings::AdvancedConfig> SettingsController::ResolvePreset(
    const QString& presetId) const
{
    return settings::ResolveConfigForPreset(
        presetId, settings_.customConfigs, settings_.customPresets);
}

QString SettingsController::DefaultPromotedPresetName() const
{
    if (const settings::PresetDefinition* preset =
            settings::FindPresetById(settings_.selectedPresetId,
                                     settings_.customPresets)) {
        return preset->name + QStringLiteral(" Copy");
    }
    return QStringLiteral("Custom Quick Option");
}

settings::PresetDefinition SettingsController::PromoteCurrentConfig(
    settings::AdvancedConfig config, const QString& name)
{
    config.id = MakeCustomEntityId(QStringLiteral("custom-config"));
    config.name = name;
    config.description =
        QStringLiteral("Custom quick option created from Advanced Tuning.");

    settings::PresetDefinition preset;
    preset.id = MakeCustomEntityId(QStringLiteral("custom-preset"));
    preset.name = name;
    preset.description = config.description;
    preset.configId = config.id;
    preset.isBuiltIn = false;

    settings_.customConfigs.push_back(config);
    settings_.customPresets.push_back(preset);
    settings_.currentConfig = config;
    settings_.selectedPresetId = preset.id;
    return preset;
}

bool SettingsController::Save(const settings::AdvancedConfig& current)
{
    settings_.currentConfig = DecorateLiveConfig(current);
    return settings::Save(settingsPath_, settings_);
}

} // namespace openzoom
