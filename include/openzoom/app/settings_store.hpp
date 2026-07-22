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
    bool stabilizationEnabled{false};
    float stabilizationStrength{0.85f};
    int displayColorMode{0};
    float contrast{1.0f};
    float brightness{0.0f};
    bool keystoneEnabled{false};
    bool autoContrastEnabled{false};
    float autoContrastStrength{0.7f};
};

// AI / assistive configuration. Values here take precedence; the matching
// OPENZOOM_* environment variables act as fallbacks when a field is empty.
struct AssistiveSettings {
    QString aiProvider{QStringLiteral("codex")};
    QString codexExecutablePath;
    QString codexModel{QStringLiteral("gpt-5.5")};
    QString codexReasoningEffort{QStringLiteral("xhigh")};
    bool codexInternetEnabled{false};
    bool codexCodingEnabled{false};
    QString codexWorkspaceDirectory;
    QString assistantInstructions{
        QStringLiteral("Reply in the same language as the user's request unless they ask for "
                       "another language. Keep answers concise and easy to understand.")};
    QString vlmApiUrl;
    QString vlmApiKey;
    QString vlmModel;
    QString vlmPrompt;
    QString tesseractPath;
    QString ocrLanguage{QStringLiteral("eng")};
    QString ttsEngine;
    QString ttsVoiceName;
    QString ttsVoiceLocale;
    double ttsRate{0.0};
    bool lectureNotesEnabled{true};
};

// OpenZoom keeps a small index of only the persistent Codex conversations it
// created. The transcript itself remains in Codex's own thread store.
struct CodexConversation {
    QString threadId;
    QString title;
    QString preview;
    qint64 createdAt{0};
    qint64 updatedAt{0};
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
    int rotationQuarterTurns{0};
    bool virtualJoystick{false};
    bool controlsCollapsed{true};
    bool simpleUiMode{true};
    QString selectedPresetId;
    AdvancedConfig currentConfig{};
    AssistiveSettings assistive{};
    std::vector<CodexConversation> codexConversations;
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
