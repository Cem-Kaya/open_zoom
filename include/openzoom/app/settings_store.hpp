#pragma once

#include <QString>

#include <optional>

namespace openzoom::settings {

struct PersistentSettings {
    int cameraIndex{-1};
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
    bool virtualJoystick{false};
    bool controlsCollapsed{false};
    int rotationQuarterTurns{0};
};

QString ResolveSettingsPath();
void EnsureSettingsDirectory(const QString& path);
std::optional<PersistentSettings> Load(const QString& path);
bool Save(const QString& path, const PersistentSettings& settings);

} // namespace openzoom::settings

