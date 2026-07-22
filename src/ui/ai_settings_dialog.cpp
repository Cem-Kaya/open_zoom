#ifdef _WIN32

#include "openzoom/ui/ai_settings_dialog.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScopedValueRollback>
#include <QSignalBlocker>
#include <QSlider>
#include <QStandardPaths>
#include <QStyle>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#if OPENZOOM_HAS_TTS
#include <QTextToSpeech>
#include <QVoice>
#endif

namespace openzoom {

namespace {

#if OPENZOOM_HAS_TTS
QString VoiceLabel(const QVoice& voice)
{
    const QLocale locale = voice.locale();
    const QString language = QLocale::languageToString(locale.language());
    const QString territory = QLocale::territoryToString(locale.territory());
    if (territory.isEmpty() || locale.territory() == QLocale::AnyTerritory) {
        return QStringLiteral("%1 - %2").arg(voice.name(), language);
    }
    return QStringLiteral("%1 - %2 (%3)").arg(voice.name(), language, territory);
}
#endif

} // namespace

AiSettingsDialog::AiSettingsDialog(const settings::AssistiveSettings& initial, QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle("AI Settings");
    setMinimumWidth(600);

    // Large-ish fonts and clear focus outlines, matching the main window.
    setStyleSheet(QStringLiteral(R"(
        QWidget { font-size: 12pt; }
        QLineEdit, QPlainTextEdit {
            padding: 6px;
            border: 2px solid palette(mid);
            border-radius: 6px;
            background: palette(base);
        }
        QLineEdit:focus, QPlainTextEdit:focus { border-color: palette(highlight); }
        QPushButton { min-height: 32px; padding: 4px 14px; }
        QCheckBox { spacing: 8px; }
        QCheckBox::indicator { width: 20px; height: 20px; }
    )"));

    auto* layout = new QVBoxLayout(this);
    layout->setSpacing(12);

    auto* introLabel = new QLabel(
        "Use a signed-in Codex CLI with a ChatGPT subscription, or select an "
        "OpenAI-compatible server such as LM Studio or Ollama.");
    introLabel->setWordWrap(true);
    layout->addWidget(introLabel);

    auto* form = new QFormLayout();
    form->setSpacing(10);

    providerCombo_ = new QComboBox();
    providerCombo_->addItem("Codex subscription", QStringLiteral("codex"));
    providerCombo_->addItem("OpenAI-compatible server", QStringLiteral("openai-compatible"));
    const int providerIndex = providerCombo_->findData(initial.aiProvider);
    providerCombo_->setCurrentIndex(providerIndex >= 0 ? providerIndex : 0);
    form->addRow("AI provider:", providerCombo_);

    codexPathEdit_ = new QLineEdit(initial.codexExecutablePath);
    codexPathEdit_->setPlaceholderText("Auto-detect codex.exe");
    auto* codexBrowseButton = new QPushButton("Browse...");
    connect(codexBrowseButton, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(
            this, "Select Codex CLI", codexPathEdit_->text(),
            "Codex CLI (codex.exe);;Executables (*.exe);;All Files (*)");
        if (!path.isEmpty()) {
            codexPathEdit_->setText(path);
        }
    });
    auto* codexPathRow = new QHBoxLayout();
    codexPathRow->setSpacing(8);
    codexPathRow->addWidget(codexPathEdit_, 1);
    codexPathRow->addWidget(codexBrowseButton);
    form->addRow("Codex CLI path:", codexPathRow);

    codexModelEdit_ = new QLineEdit(initial.codexModel.trimmed().isEmpty()
                                        ? QStringLiteral("gpt-5.5")
                                        : initial.codexModel.trimmed());
    codexModelEdit_->setPlaceholderText("gpt-5.5");
    form->addRow("Codex model:", codexModelEdit_);

    codexReasoningCombo_ = new QComboBox();
    codexReasoningCombo_->addItem("Low", QStringLiteral("low"));
    codexReasoningCombo_->addItem("Medium", QStringLiteral("medium"));
    codexReasoningCombo_->addItem("High", QStringLiteral("high"));
    codexReasoningCombo_->addItem("Extra high", QStringLiteral("xhigh"));
    const int reasoningIndex = codexReasoningCombo_->findData(initial.codexReasoningEffort);
    codexReasoningCombo_->setCurrentIndex(reasoningIndex >= 0 ? reasoningIndex : 3);
    form->addRow("Reasoning:", codexReasoningCombo_);

    codexInternetCheckbox_ = new QCheckBox("Allow internet access");
    codexInternetCheckbox_->setChecked(initial.codexInternetEnabled);
    form->addRow("Advanced Assistant:", codexInternetCheckbox_);

    codexCodingCheckbox_ = new QCheckBox("Allow coding commands and file changes");
    codexCodingCheckbox_->setChecked(initial.codexCodingEnabled);
    form->addRow(QString(), codexCodingCheckbox_);

    codexWorkspaceEdit_ = new QLineEdit(initial.codexWorkspaceDirectory);
    codexWorkspaceEdit_->setPlaceholderText("Required when coding is enabled");
    codexWorkspaceBrowseButton_ = new QPushButton("Browse...");
    connect(codexWorkspaceBrowseButton_, &QPushButton::clicked, this, [this]() {
        QString initialDirectory = codexWorkspaceEdit_->text().trimmed();
        if (initialDirectory.isEmpty()) {
            initialDirectory = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        }
        const QString path = QFileDialog::getExistingDirectory(
            this, "Select Assistant Coding Workspace", initialDirectory,
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
        if (!path.isEmpty()) {
            codexWorkspaceEdit_->setText(QDir::toNativeSeparators(QDir::cleanPath(path)));
        }
    });
    auto* codexWorkspaceRow = new QHBoxLayout();
    codexWorkspaceRow->setSpacing(8);
    codexWorkspaceRow->addWidget(codexWorkspaceEdit_, 1);
    codexWorkspaceRow->addWidget(codexWorkspaceBrowseButton_);
    form->addRow("Coding workspace:", codexWorkspaceRow);

    assistantInstructionsEdit_ = new QPlainTextEdit(initial.assistantInstructions);
    assistantInstructionsEdit_->setPlaceholderText(
        "Example: Always answer in Turkish. Use short sentences and explain technical terms.");
    assistantInstructionsEdit_->setMaximumHeight(110);
    auto* assistantBehaviorGroup = new QGroupBox("Assistant behavior");
    auto* assistantBehaviorLayout = new QFormLayout(assistantBehaviorGroup);
    assistantBehaviorLayout->addRow("Instructions:", assistantInstructionsEdit_);
    form->addRow(assistantBehaviorGroup);

    apiUrlEdit_ = new QLineEdit(initial.vlmApiUrl);
    apiUrlEdit_->setPlaceholderText(
        "https://api.openai.com/v1/chat/completions or http://localhost:11434/v1/chat/completions");
    form->addRow("VLM server URL:", apiUrlEdit_);

    apiKeyEdit_ = new QLineEdit(initial.vlmApiKey);
    apiKeyEdit_->setEchoMode(QLineEdit::Password);
    apiKeyEdit_->setPlaceholderText("Optional for local servers");
    form->addRow("API key:", apiKeyEdit_);

    modelEdit_ = new QLineEdit(initial.vlmModel);
    modelEdit_->setPlaceholderText("gpt-4o-mini or llava");
    form->addRow("Model:", modelEdit_);

    promptEdit_ = new QPlainTextEdit(initial.vlmPrompt);
    promptEdit_->setPlaceholderText(
        "Instructions for the vision model, e.g. describe the lecture slide briefly.");
    promptEdit_->setMaximumHeight(100);
    form->addRow("Scene prompt:", promptEdit_);

    connect(providerCombo_, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this]() { UpdateProviderFields(); });
    connect(codexCodingCheckbox_, &QCheckBox::toggled,
            this, [this]() { UpdateProviderFields(); });
    UpdateProviderFields();

    tesseractPathEdit_ = new QLineEdit(initial.tesseractPath);
    tesseractPathEdit_->setPlaceholderText("Path to tesseract.exe");
    auto* browseButton = new QPushButton("Browse…");
    connect(browseButton, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(
            this,
            "Select Tesseract Executable",
            tesseractPathEdit_->text(),
            "Executables (*.exe);;All Files (*)");
        if (!path.isEmpty()) {
            tesseractPathEdit_->setText(path);
        }
    });
    auto* tesseractRow = new QHBoxLayout();
    tesseractRow->setSpacing(8);
    tesseractRow->addWidget(tesseractPathEdit_, 1);
    tesseractRow->addWidget(browseButton);
    form->addRow("Tesseract path:", tesseractRow);

    ocrLanguageEdit_ = new QLineEdit(
        initial.ocrLanguage.isEmpty() ? QStringLiteral("eng") : initial.ocrLanguage);
    ocrLanguageEdit_->setPlaceholderText("eng");
    form->addRow("OCR language:", ocrLanguageEdit_);

    preferredVoiceName_ = initial.ttsVoiceName;
    preferredVoiceLocale_ = initial.ttsVoiceLocale;
    ttsEngine_ = initial.ttsEngine;

    ttsVoiceCombo_ = new QComboBox();
    ttsVoiceCombo_->setMinimumContentsLength(30);
    ttsVoiceCombo_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    form->addRow("Voice:", ttsVoiceCombo_);

    ttsRateSlider_ = new QSlider(Qt::Horizontal);
    ttsRateSlider_->setRange(-100, 100);
    ttsRateSlider_->setSingleStep(10);
    ttsRateSlider_->setPageStep(25);
    ttsRateSlider_->setValue(std::clamp(
        static_cast<int>(std::lround(initial.ttsRate * 100.0)), -100, 100));
    ttsRateValueLabel_ = new QLabel();
    ttsRateValueLabel_->setMinimumWidth(105);
    ttsRateValueLabel_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    ttsPreviewButton_ = new QPushButton("Preview");
    ttsPreviewButton_->setIcon(style()->standardIcon(QStyle::SP_MediaVolume));

    auto* speechRateRow = new QHBoxLayout();
    speechRateRow->setSpacing(8);
    speechRateRow->addWidget(ttsRateSlider_, 1);
    speechRateRow->addWidget(ttsRateValueLabel_);
    speechRateRow->addWidget(ttsPreviewButton_);
    form->addRow("Speed:", speechRateRow);

    connect(ttsRateSlider_, &QSlider::valueChanged, this, [this]() {
        UpdateSpeechRateLabel();
        ApplySpeechPreviewSettings();
    });
    connect(ttsVoiceCombo_, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this]() { ApplySpeechPreviewSettings(); });
    connect(ttsPreviewButton_, &QPushButton::clicked,
            this, &AiSettingsDialog::PreviewSpeech);
    UpdateSpeechRateLabel();

#if OPENZOOM_HAS_TTS
    const QStringList engines = QTextToSpeech::availableEngines();
    auto resolveEngine = [&engines](const QString& requested) {
        for (const QString& engine : engines) {
            if (engine.compare(requested, Qt::CaseInsensitive) == 0) {
                return engine;
            }
        }
        return QString();
    };
    ttsEngine_ = resolveEngine(ttsEngine_);
    if (ttsEngine_.isEmpty()) {
        ttsEngine_ = resolveEngine(QStringLiteral("winrt"));
    }
    if (ttsEngine_.isEmpty()) {
        ttsEngine_ = resolveEngine(QStringLiteral("sapi"));
    }
    speechPreview_ = ttsEngine_.isEmpty() ? new QTextToSpeech(this)
                                          : new QTextToSpeech(ttsEngine_, this);
    ttsEngine_ = speechPreview_->engine();
    PopulateSpeechVoices();
    connect(speechPreview_, &QTextToSpeech::stateChanged, this,
            [this](QTextToSpeech::State state) {
                if (!speechVoicesLoaded_ &&
                    (state == QTextToSpeech::Ready || state == QTextToSpeech::Error)) {
                    PopulateSpeechVoices();
                }
            });
    QTimer::singleShot(0, this, &AiSettingsDialog::PopulateSpeechVoices);
#else
    ttsVoiceCombo_->addItem("Text-to-speech is unavailable in this build");
    ttsVoiceCombo_->setEnabled(false);
    ttsRateSlider_->setEnabled(false);
    ttsPreviewButton_->setEnabled(false);
#endif

    layout->addLayout(form);

    lectureNotesCheckbox_ = new QCheckBox("Write lecture notes file");
    lectureNotesCheckbox_->setChecked(initial.lectureNotesEnabled);
    layout->addWidget(lectureNotesCheckbox_);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, this, [this]() {
        if (providerCombo_->currentData().toString() == QStringLiteral("codex") &&
            codexCodingCheckbox_->isChecked()) {
            const QFileInfo workspace(codexWorkspaceEdit_->text().trimmed());
            if (!workspace.exists() || !workspace.isDir()) {
                QMessageBox::warning(this,
                                     QStringLiteral("Coding Workspace Required"),
                                     QStringLiteral("Choose an existing workspace folder before enabling coding."));
                codexWorkspaceEdit_->setFocus();
                return;
            }
        }
        accept();
    });
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttons);

    auto setA11y = [](QWidget* widget, const QString& name, const QString& description) {
        widget->setAccessibleName(name);
        widget->setAccessibleDescription(description);
    };
    setA11y(providerCombo_, "AI Provider",
            "Choose Codex subscription or an OpenAI-compatible server");
    setA11y(codexPathEdit_, "Codex CLI Path",
            "Optional location of codex.exe; leave blank for automatic detection");
    setA11y(codexBrowseButton, "Browse for Codex CLI",
            "Pick the Codex command line executable from disk");
    setA11y(codexModelEdit_, "Codex Model",
            "Codex image model; GPT 5.5 is the current default for this installation");
    setA11y(codexReasoningCombo_, "Codex Reasoning",
            "Choose how much reasoning Codex uses; extra high is the default");
    setA11y(codexInternetCheckbox_, "Allow Assistant Internet Access",
            "Allow persistent Advanced Assistant conversations to use web search and network access");
    setA11y(codexCodingCheckbox_, "Allow Assistant Coding",
            "Allow persistent Advanced Assistant conversations to run commands and change files in the selected workspace");
    setA11y(codexWorkspaceEdit_, "Assistant Coding Workspace",
            "Folder that Codex may inspect and modify when coding is enabled");
    setA11y(codexWorkspaceBrowseButton_, "Browse for Coding Workspace",
            "Choose the folder available to Codex coding commands and file changes");
    setA11y(assistantInstructionsEdit_, "Assistant Instructions",
            "Set the response language, tone, detail, and other response preferences");
    setA11y(apiUrlEdit_, "VLM Server URL",
            "Address of the OpenAI-compatible chat completions endpoint");
    setA11y(apiKeyEdit_, "API Key",
            "Secret key for the vision server, optional for local servers");
    setA11y(modelEdit_, "Model",
            "Name of the vision model, for example gpt-4o-mini or llava");
    setA11y(promptEdit_, "Scene Prompt",
            "Scene-specific instructions sent with each camera frame");
    setA11y(tesseractPathEdit_, "Tesseract Path",
            "Location of the Tesseract OCR executable");
    setA11y(browseButton, "Browse for Tesseract",
            "Pick the Tesseract OCR executable from disk");
    setA11y(ocrLanguageEdit_, "OCR Language",
            "Tesseract language code, for example eng");
    setA11y(ttsVoiceCombo_, "Read Aloud Voice",
            "Choose an installed Windows voice for the Read Aloud button");
    setA11y(ttsRateSlider_, "Read Aloud Speed",
            "Adjust how slowly or quickly OpenZoom reads a result");
    setA11y(ttsPreviewButton_, "Preview Read Aloud Voice",
            "Speak a short sample with the selected voice and speed");
    setA11y(lectureNotesCheckbox_, "Write Lecture Notes File",
            "Append OCR and scene descriptions to a lecture notes file");
}

settings::AssistiveSettings AiSettingsDialog::result() const
{
    settings::AssistiveSettings out;
    out.aiProvider = providerCombo_->currentData().toString();
    out.codexExecutablePath = codexPathEdit_->text().trimmed();
    out.codexModel = codexModelEdit_->text().trimmed();
    out.codexReasoningEffort = codexReasoningCombo_->currentData().toString();
    out.codexInternetEnabled = codexInternetCheckbox_->isChecked();
    out.codexCodingEnabled = codexCodingCheckbox_->isChecked();
    out.codexWorkspaceDirectory = QDir::toNativeSeparators(
        QDir::cleanPath(codexWorkspaceEdit_->text().trimmed()));
    if (codexWorkspaceEdit_->text().trimmed().isEmpty()) {
        out.codexWorkspaceDirectory.clear();
    }
    out.assistantInstructions = assistantInstructionsEdit_->toPlainText().trimmed();
    out.vlmApiUrl = apiUrlEdit_->text().trimmed();
    out.vlmApiKey = apiKeyEdit_->text();
    out.vlmModel = modelEdit_->text().trimmed();
    out.vlmPrompt = promptEdit_->toPlainText();
    out.tesseractPath = tesseractPathEdit_->text().trimmed();
    out.ocrLanguage = ocrLanguageEdit_->text().trimmed();
    if (out.ocrLanguage.isEmpty()) {
        out.ocrLanguage = QStringLiteral("eng");
    }
    out.ttsEngine = ttsEngine_;
    out.ttsRate = static_cast<double>(ttsRateSlider_->value()) / 100.0;
#if OPENZOOM_HAS_TTS
    const QVoice voice = ttsVoiceCombo_->currentData().value<QVoice>();
    out.ttsVoiceName = voice.name();
    out.ttsVoiceLocale = voice.locale().name();
#else
    out.ttsVoiceName = preferredVoiceName_;
    out.ttsVoiceLocale = preferredVoiceLocale_;
#endif
    out.lectureNotesEnabled = lectureNotesCheckbox_->isChecked();
    return out;
}

void AiSettingsDialog::PopulateSpeechVoices()
{
#if OPENZOOM_HAS_TTS
    if (!speechPreview_ || populatingSpeechVoices_) {
        return;
    }
    QScopedValueRollback<bool> populationGuard(populatingSpeechVoices_, true);

    // availableVoices() is limited to the active locale. The settings picker
    // needs every voice the selected Windows engine exposes.
    QList<QVoice> voices = speechPreview_->findVoices();
    const QTextToSpeech::State state = speechPreview_->state();
    if (voices.isEmpty() && speechPreview_->engine().compare(
                                QStringLiteral("winrt"), Qt::CaseInsensitive) == 0 &&
        state == QTextToSpeech::Error) {
        const QStringList engines = QTextToSpeech::availableEngines();
        for (const QString& engine : engines) {
            if (engine.compare(QStringLiteral("sapi"), Qt::CaseInsensitive) == 0 &&
                speechPreview_->setEngine(engine)) {
                ttsEngine_ = speechPreview_->engine();
                voices = speechPreview_->findVoices();
                break;
            }
        }
    }
    std::sort(voices.begin(), voices.end(), [](const QVoice& lhs, const QVoice& rhs) {
        return VoiceLabel(lhs).localeAwareCompare(VoiceLabel(rhs)) < 0;
    });

    QSignalBlocker blocker(ttsVoiceCombo_);
    ttsVoiceCombo_->clear();
    int selectedIndex = -1;
    for (const QVoice& voice : voices) {
        const int index = ttsVoiceCombo_->count();
        ttsVoiceCombo_->addItem(VoiceLabel(voice), QVariant::fromValue(voice));
        const bool nameMatches = voice.name() == preferredVoiceName_;
        const bool localeMatches = preferredVoiceLocale_.isEmpty() ||
                                   voice.locale().name() == preferredVoiceLocale_;
        if (selectedIndex < 0 && nameMatches && localeMatches) {
            selectedIndex = index;
        }
    }

    speechVoicesLoaded_ = !voices.isEmpty();
    if (!speechVoicesLoaded_) {
        const bool stillLoading = speechPreview_->state() != QTextToSpeech::Ready &&
                                  speechPreview_->state() != QTextToSpeech::Error;
        ttsVoiceCombo_->addItem(stillLoading ? "Loading installed Windows voices..."
                                             : "No installed Windows voices found");
        ttsVoiceCombo_->setEnabled(false);
        ttsPreviewButton_->setEnabled(false);
        return;
    }

    if (selectedIndex < 0) {
        const QVoice currentVoice = speechPreview_->voice();
        for (int i = 0; i < ttsVoiceCombo_->count(); ++i) {
            if (ttsVoiceCombo_->itemData(i).value<QVoice>() == currentVoice) {
                selectedIndex = i;
                break;
            }
        }
    }
    ttsVoiceCombo_->setEnabled(true);
    ttsPreviewButton_->setEnabled(true);
    ttsVoiceCombo_->setCurrentIndex(selectedIndex >= 0 ? selectedIndex : 0);
    const QVoice selectedVoice = ttsVoiceCombo_->currentData().value<QVoice>();
    preferredVoiceName_ = selectedVoice.name();
    preferredVoiceLocale_ = selectedVoice.locale().name();
    ApplySpeechPreviewSettings();
#endif
}

void AiSettingsDialog::UpdateSpeechRateLabel()
{
    const int rate = ttsRateSlider_->value();
    if (rate == 0) {
        ttsRateValueLabel_->setText("Normal");
    } else if (rate < 0) {
        ttsRateValueLabel_->setText(QStringLiteral("%1% slower").arg(-rate));
    } else {
        ttsRateValueLabel_->setText(QStringLiteral("%1% faster").arg(rate));
    }
}

void AiSettingsDialog::ApplySpeechPreviewSettings()
{
#if OPENZOOM_HAS_TTS
    if (!speechPreview_ || !speechVoicesLoaded_) {
        return;
    }
    speechPreview_->setRate(static_cast<double>(ttsRateSlider_->value()) / 100.0);
    const QVoice voice = ttsVoiceCombo_->currentData().value<QVoice>();
    if (!voice.name().isEmpty()) {
        speechPreview_->setVoice(voice);
        preferredVoiceName_ = voice.name();
        preferredVoiceLocale_ = voice.locale().name();
    }
#endif
}

void AiSettingsDialog::PreviewSpeech()
{
#if OPENZOOM_HAS_TTS
    if (!speechPreview_) {
        return;
    }
    ApplySpeechPreviewSettings();
    speechPreview_->stop();
    speechPreview_->say(QStringLiteral("OpenZoom will read this result using the selected voice."));
#endif
}

void AiSettingsDialog::UpdateProviderFields()
{
    const bool codex = providerCombo_->currentData().toString() == QStringLiteral("codex");
    codexPathEdit_->setEnabled(codex);
    codexModelEdit_->setEnabled(codex);
    codexReasoningCombo_->setEnabled(codex);
    codexInternetCheckbox_->setEnabled(codex);
    codexCodingCheckbox_->setEnabled(codex);
    const bool coding = codex && codexCodingCheckbox_->isChecked();
    codexWorkspaceEdit_->setEnabled(coding);
    codexWorkspaceBrowseButton_->setEnabled(coding);
    apiUrlEdit_->setEnabled(!codex);
    apiKeyEdit_->setEnabled(!codex);
    modelEdit_->setEnabled(!codex);
}

} // namespace openzoom

#endif // _WIN32
