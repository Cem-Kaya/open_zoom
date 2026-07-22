#ifdef _WIN32

#include "openzoom/ui/main_window.hpp"

#include "openzoom/app/app.hpp"
#include "openzoom/app/constants.hpp"
#include "openzoom/d3d12/presenter.hpp"

#include <QAbstractItemModel>
#include <QAccessible>
#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QEvent>
#include <QAbstractItemView>
#include <QFrame>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QListWidget>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QParallelAnimationGroup>
#include <QPlainTextEdit>
#include <QPropertyAnimation>
#include <QPushButton>
#include <QRegion>
#include <QResizeEvent>
#include <QScrollArea>
#include <QScrollBar>
#include <QShortcut>
#include <QShowEvent>
#include <QSizePolicy>
#include <QSignalBlocker>
#include <QSlider>
#include <QStyle>
#include <QTabWidget>
#include <QTextBrowser>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWindow>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include <windows.h>

namespace openzoom {

using namespace app_constants;

namespace {

constexpr int kSimpleChromeIdleMs = 5000;
constexpr int kSimpleChromeFadeMs = 260;
constexpr int kModeToastMs = 1200;
constexpr int kSimpleShortcutCount = 9;

QString PlainPresetLabel(const QString& original)
{
    if (original == QStringLiteral("Reading")) return QStringLiteral("Read a Page");
    if (original == QStringLiteral("High Contrast")) return QStringLiteral("High Contrast");
    if (original == QStringLiteral("Steady Text")) return QStringLiteral("Keep It Steady");
    if (original == QStringLiteral("Sharp Text")) return QStringLiteral("Sharpen Text");
    if (original == QStringLiteral("Large Zoom")) return QStringLiteral("Zoom In More");
    if (original == QStringLiteral("Low Light")) return QStringLiteral("See in Low Light");
    if (original == QStringLiteral("OCR Assist")) return QStringLiteral("Read Text Aloud");
    if (original == QStringLiteral("Scene Explain")) return QStringLiteral("Describe the Scene");
    return original;
}

QString SpokenAssistiveText(QString body)
{
    const std::array<QString, 2> sectionLabels{
        QStringLiteral("OCR"),
        QStringLiteral("Scene Explain")};
    for (const QString& label : sectionLabels) {
        const QString leadingHeader = label + QLatin1Char('\n');
        if (body.startsWith(leadingHeader)) {
            body.remove(0, leadingHeader.size());
        }
        body.replace(QStringLiteral("\n\n%1\n").arg(label), QStringLiteral("\n\n"));
    }
    return body.trimmed();
}

}  // namespace

RenderWidget::RenderWidget(QWidget* parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_NativeWindow);
    setAttribute(Qt::WA_PaintOnScreen);
    setAttribute(Qt::WA_NoSystemBackground);
    setFocusPolicy(Qt::StrongFocus);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

QPaintEngine* RenderWidget::paintEngine() const
{
    return nullptr;
}

void RenderWidget::setPresenter(D3D12Presenter* presenter)
{
    presenter_ = presenter;
}

bool RenderWidget::isPresenterReady() const
{
    return presenter_ && presenter_->IsInitialized();
}

void RenderWidget::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    EnsurePresenter();
}

void RenderWidget::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    if (EnsurePresenter()) {
        presenter_->Resize(std::max(1, width()), std::max(1, height()));
    }
}

bool RenderWidget::EnsurePresenter()
{
    if (!presenter_ || presenter_->IsInitialized()) {
        return presenter_ != nullptr;
    }

    HWND hwnd = reinterpret_cast<HWND>(winId());
    presenter_->Initialize(hwnd, std::max(1, width()), std::max(1, height()));
    return true;
}

AssistiveOverlay::AssistiveOverlay(QWidget* parent)
    : QWidget(parent, Qt::Tool | Qt::FramelessWindowHint | Qt::NoDropShadowWindowHint)
{
    setObjectName(QStringLiteral("assistiveOverlay"));
    setAttribute(Qt::WA_StyledBackground, true);
    setFocusPolicy(Qt::NoFocus);
    setMouseTracking(true);
    setMinimumSize(360, 260);
    setAccessibleName(QStringLiteral("Assistive results"));
    setStyleSheet(QStringLiteral(R"(
        QWidget#assistiveOverlay {
            background: #111111;
            border: 3px solid #f2f2f2;
            border-radius: 8px;
        }
        QLabel#assistiveTitle {
            color: #fff7d6;
            font-size: 14pt;
            font-weight: 700;
            border: none;
        }
        QTextBrowser#assistiveBody {
            color: #f5f5f5;
            background: transparent;
            border: none;
            font-size: 13pt;
            selection-background-color: #a84bc1;
            selection-color: #ffffff;
        }
        QPushButton#assistiveReadButton, QPushButton#assistiveAskButton {
            color: #ffffff;
            background: #3d3d3d;
            border: 2px solid #737373;
            border-radius: 6px;
            min-height: 36px;
        }
        QPushButton#assistiveReadButton:focus, QPushButton#assistiveAskButton:focus {
            border: 3px solid #bd52d3;
        }
        QLineEdit#assistiveQuestion {
            color: #ffffff;
            background: #202020;
            border: 2px solid #9a9a9a;
            border-radius: 6px;
            min-height: 38px;
            padding: 2px 8px;
        }
        QLineEdit#assistiveQuestion:focus { border: 3px solid #bd52d3; }
        QToolButton#assistiveCloseButton {
            background: #080808;
            border: 3px solid #ffffff;
            border-radius: 6px;
            padding: 7px;
        }
        QToolButton#assistiveCloseButton:hover,
        QToolButton#assistiveCloseButton:focus {
            background: #a747c5;
            border-color: #ffffff;
        }
        QToolButton#assistiveCloseButton:pressed { background: #733087; }
    )"));

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(14, 10, 14, 12);
    layout->setSpacing(8);

    headerWidget_ = new QWidget(this);
    headerWidget_->setObjectName(QStringLiteral("assistiveHeader"));
    headerWidget_->setCursor(Qt::SizeAllCursor);
    headerWidget_->setAccessibleName(QStringLiteral("Move assistive panel"));
    auto* header = new QHBoxLayout(headerWidget_);
    header->setContentsMargins(0, 0, 0, 0);
    header->setSpacing(8);
    titleLabel_ = new QLabel();
    titleLabel_->setObjectName(QStringLiteral("assistiveTitle"));
    titleLabel_->setAccessibleName(QStringLiteral("Assistive result title"));
    header->addWidget(titleLabel_, 1);

    closeButton_ = new QToolButton();
    closeButton_->setObjectName(QStringLiteral("assistiveCloseButton"));
    closeButton_->setIcon(QIcon(QStringLiteral(":/openzoom/icons/close.svg")));
    closeButton_->setIconSize(QSize(28, 28));
    closeButton_->setToolTip(QStringLiteral("Close result"));
    closeButton_->setAccessibleName(QStringLiteral("Close assistive result"));
    closeButton_->setAccessibleDescription(
        QStringLiteral("Hide the current OCR or scene explanation result"));
    closeButton_->setFixedSize(52, 52);
    header->addWidget(closeButton_);
    layout->addWidget(headerWidget_);

    bodyView_ = new QTextBrowser();
    bodyView_->setObjectName(QStringLiteral("assistiveBody"));
    bodyView_->setReadOnly(true);
    bodyView_->setOpenExternalLinks(false);
    bodyView_->setFocusPolicy(Qt::StrongFocus);
    bodyView_->setTextInteractionFlags(Qt::TextSelectableByKeyboard | Qt::TextSelectableByMouse);
    bodyView_->setAccessibleName(QStringLiteral("Assistive result text"));
    bodyView_->setAccessibleDescription(
        QStringLiteral("Streaming OCR and scene explanation result. Use arrow keys to read the text."));
    layout->addWidget(bodyView_, 1);

    auto* questionRow = new QHBoxLayout();
    questionRow->setSpacing(8);
    questionEdit_ = new QLineEdit();
    questionEdit_->setObjectName(QStringLiteral("assistiveQuestion"));
    questionEdit_->setPlaceholderText(QStringLiteral("Ask about this view..."));
    questionEdit_->setAccessibleName(QStringLiteral("Question about the current view"));
    questionEdit_->setAccessibleDescription(
        QStringLiteral("Ask a follow-up question in the shared OpenZoom Assistant conversation"));
    askButton_ = new QPushButton(QStringLiteral("Ask"));
    askButton_->setObjectName(QStringLiteral("assistiveAskButton"));
    askButton_->setEnabled(false);
    askButton_->setAccessibleName(QStringLiteral("Ask Assistant"));
    askButton_->setAccessibleDescription(
        QStringLiteral("Send the question with the current camera view to OpenZoom Assistant"));
    questionRow->addWidget(questionEdit_, 1);
    questionRow->addWidget(askButton_);
    layout->addLayout(questionRow);

    auto* footer = new QHBoxLayout();
    footer->addStretch(1);
    readAloudButton_ = new QPushButton(QStringLiteral("Read Aloud"));
    readAloudButton_->setObjectName(QStringLiteral("assistiveReadButton"));
    readAloudButton_->setIcon(QIcon(QStringLiteral(":/openzoom/icons/read.svg")));
    readAloudButton_->setIconSize(QSize(24, 24));
    readAloudButton_->setAccessibleName(QStringLiteral("Read assistive result aloud"));
    readAloudButton_->setAccessibleDescription(
        QStringLiteral("Speak the current OCR or scene explanation result"));
    footer->addWidget(readAloudButton_);
    layout->addLayout(footer);

    connect(closeButton_, &QToolButton::clicked, this, [this]() {
        hide();
        emit Dismissed();
    });
    connect(readAloudButton_, &QPushButton::clicked, this, [this]() {
        const QString speechText = SpokenAssistiveText(body_);
        if (!speechText.isEmpty()) {
            emit ReadAloudRequested(speechText);
        }
    });
    connect(questionEdit_, &QLineEdit::textChanged, this, [this](const QString& text) {
        askButton_->setEnabled(questionEdit_->isEnabled() && !text.trimmed().isEmpty());
    });
    connect(questionEdit_, &QLineEdit::returnPressed, this, &AssistiveOverlay::SubmitQuestion);
    connect(askButton_, &QPushButton::clicked, this, &AssistiveOverlay::SubmitQuestion);
    auto* closeShortcut = new QShortcut(QKeySequence::Cancel, this);
    closeShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(closeShortcut, &QShortcut::activated, closeButton_, &QToolButton::click);

    headerWidget_->installEventFilter(this);
    titleLabel_->installEventFilter(this);
    headerWidget_->setMouseTracking(true);
    titleLabel_->setMouseTracking(true);

    setVisible(false);
    if (parent) {
        parent->installEventFilter(this);
        if (parent->window() != parent) {
            parent->window()->installEventFilter(this);
        }
    }
}

void AssistiveOverlay::SetBusy(bool busy)
{
    questionEdit_->setEnabled(!busy);
    askButton_->setEnabled(!busy && !questionEdit_->text().trimmed().isEmpty());
}

void AssistiveOverlay::SetContent(const QString& title, const QString& body, bool visible)
{
    if (title_ != title) {
        title_ = title;
        titleLabel_->setText(title_);
    }
    if (body_ != body) {
        if (!body_.isEmpty() && body.startsWith(body_)) {
            QTextCursor cursor = bodyView_->textCursor();
            cursor.movePosition(QTextCursor::End);
            cursor.insertText(body.mid(body_.size()));
            bodyView_->setTextCursor(cursor);
        } else {
            bodyView_->setPlainText(body);
        }
        body_ = body;
        bodyView_->ensureCursorVisible();
    }
    const bool shouldShow = visible && (!title_.isEmpty() || !body_.isEmpty());
    const bool wasVisible = isVisible();
    if (shouldShow != wasVisible) {
        setVisible(shouldShow);
    }
    if (shouldShow && !wasVisible) {
        raise();
    }
}

bool AssistiveOverlay::eventFilter(QObject* watched, QEvent* event)
{
    const bool parentGeometryChanged = parentWidget() &&
                                       (watched == parentWidget() || watched == parentWidget()->window()) &&
                                       (event->type() == QEvent::Move ||
                                        event->type() == QEvent::Resize ||
                                        event->type() == QEvent::WindowStateChange);
    if (parentGeometryChanged) {
        UpdatePlacement();
    } else if ((watched == headerWidget_ || watched == titleLabel_) &&
               event->type() == QEvent::MouseButtonPress) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        if (mouse->button() == Qt::LeftButton) {
            BeginDrag(mouse->globalPosition().toPoint());
            return true;
        }
    } else if ((watched == headerWidget_ || watched == titleLabel_) &&
               event->type() == QEvent::MouseMove && dragging_) {
        ContinueDrag(static_cast<QMouseEvent*>(event)->globalPosition().toPoint());
        return true;
    } else if ((watched == headerWidget_ || watched == titleLabel_) &&
               event->type() == QEvent::MouseButtonRelease && dragging_) {
        dragging_ = false;
        releaseMouse();
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

void AssistiveOverlay::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    UpdatePlacement();
}

void AssistiveOverlay::UpdatePlacement()
{
    if (!parentWidget()) {
        return;
    }
    const int parentWidth = parentWidget()->width();
    const int parentHeight = parentWidget()->height();
    const QPoint parentOrigin = parentWidget()->mapToGlobal(QPoint(0, 0));
    if (!placementInitialized_) {
        const int margin = std::min(20, std::max(0, std::min(parentWidth, parentHeight) / 12));
        const int overlayWidth = std::clamp(parentWidth * 3 / 5, 360, 760);
        const int overlayHeight = std::clamp(parentHeight * 2 / 5, 260, 520);
        setGeometry(ConstrainedGeometry(
            QRect(parentOrigin + QPoint(margin, margin), QSize(overlayWidth, overlayHeight))));
        parentOrigin_ = parentOrigin;
        placementInitialized_ = true;
        return;
    }
    QRect requested = geometry();
    requested.translate(parentOrigin - parentOrigin_);
    parentOrigin_ = parentOrigin;
    const QRect constrained = ConstrainedGeometry(requested);
    if (constrained != geometry()) {
        setGeometry(constrained);
    }
}

void AssistiveOverlay::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && ResizeEdgesAt(event->position().toPoint()) != Qt::Edges{}) {
        BeginResize(event->position().toPoint(), event->globalPosition().toPoint());
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void AssistiveOverlay::mouseMoveEvent(QMouseEvent* event)
{
    if (dragging_) {
        ContinueDrag(event->globalPosition().toPoint());
        event->accept();
        return;
    }
    if (resizing_) {
        ContinueResize(event->globalPosition().toPoint());
        event->accept();
        return;
    }
    UpdateResizeCursor(event->position().toPoint());
    QWidget::mouseMoveEvent(event);
}

void AssistiveOverlay::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && dragging_) {
        dragging_ = false;
        releaseMouse();
        event->accept();
        return;
    }
    if (event->button() == Qt::LeftButton && resizing_) {
        resizing_ = false;
        resizeEdges_ = {};
        releaseMouse();
        UpdateResizeCursor(event->position().toPoint());
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void AssistiveOverlay::leaveEvent(QEvent* event)
{
    if (!resizing_) {
        unsetCursor();
    }
    QWidget::leaveEvent(event);
}

void AssistiveOverlay::BeginDrag(const QPoint& globalPosition)
{
    resizing_ = false;
    dragging_ = false;
    raise();
    if (QWindow* nativeWindow = windowHandle(); nativeWindow && nativeWindow->startSystemMove()) {
        return;
    }
    dragging_ = true;
    pointerStartGlobal_ = globalPosition;
    pointerStartGeometry_ = geometry();
    grabMouse();
}

void AssistiveOverlay::ContinueDrag(const QPoint& globalPosition)
{
    if (!dragging_) {
        return;
    }
    QRect requested = pointerStartGeometry_;
    requested.moveTopLeft(pointerStartGeometry_.topLeft() + globalPosition - pointerStartGlobal_);
    setGeometry(ConstrainedGeometry(requested));
}

void AssistiveOverlay::BeginResize(const QPoint& localPosition, const QPoint& globalPosition)
{
    resizeEdges_ = ResizeEdgesAt(localPosition);
    if (resizeEdges_ == Qt::Edges{}) {
        return;
    }
    dragging_ = false;
    resizing_ = false;
    raise();
    if (QWindow* nativeWindow = windowHandle();
        nativeWindow && nativeWindow->startSystemResize(resizeEdges_)) {
        return;
    }
    resizing_ = true;
    pointerStartGlobal_ = globalPosition;
    pointerStartGeometry_ = geometry();
    grabMouse();
}

void AssistiveOverlay::ContinueResize(const QPoint& globalPosition)
{
    if (!resizing_) {
        return;
    }
    const QPoint delta = globalPosition - pointerStartGlobal_;
    QRect requested = pointerStartGeometry_;
    if (resizeEdges_.testFlag(Qt::LeftEdge)) requested.setLeft(requested.left() + delta.x());
    if (resizeEdges_.testFlag(Qt::RightEdge)) requested.setRight(requested.right() + delta.x());
    if (resizeEdges_.testFlag(Qt::TopEdge)) requested.setTop(requested.top() + delta.y());
    if (resizeEdges_.testFlag(Qt::BottomEdge)) requested.setBottom(requested.bottom() + delta.y());
    setGeometry(ConstrainedGeometry(requested.normalized()));
}

Qt::Edges AssistiveOverlay::ResizeEdgesAt(const QPoint& localPosition) const
{
    constexpr int kResizeMargin = 9;
    Qt::Edges edges;
    if (localPosition.x() <= kResizeMargin) edges |= Qt::LeftEdge;
    if (localPosition.x() >= width() - kResizeMargin) edges |= Qt::RightEdge;
    if (localPosition.y() <= kResizeMargin) edges |= Qt::TopEdge;
    if (localPosition.y() >= height() - kResizeMargin) edges |= Qt::BottomEdge;
    return edges;
}

void AssistiveOverlay::UpdateResizeCursor(const QPoint& localPosition)
{
    const Qt::Edges edges = ResizeEdgesAt(localPosition);
    const bool horizontal = edges.testFlag(Qt::LeftEdge) || edges.testFlag(Qt::RightEdge);
    const bool vertical = edges.testFlag(Qt::TopEdge) || edges.testFlag(Qt::BottomEdge);
    if (horizontal && vertical) {
        const bool forward = (edges.testFlag(Qt::LeftEdge) && edges.testFlag(Qt::TopEdge)) ||
                             (edges.testFlag(Qt::RightEdge) && edges.testFlag(Qt::BottomEdge));
        setCursor(forward ? Qt::SizeFDiagCursor : Qt::SizeBDiagCursor);
    } else if (horizontal) {
        setCursor(Qt::SizeHorCursor);
    } else if (vertical) {
        setCursor(Qt::SizeVerCursor);
    } else {
        unsetCursor();
    }
}

QRect AssistiveOverlay::ConstrainedGeometry(const QRect& requested) const
{
    if (!parentWidget()) {
        return requested;
    }
    const QRect available(parentWidget()->mapToGlobal(QPoint(0, 0)), parentWidget()->size());
    const int minWidth = std::min(minimumWidth(), available.width());
    const int minHeight = std::min(minimumHeight(), available.height());
    const int width = std::clamp(requested.width(), minWidth, available.width());
    const int height = std::clamp(requested.height(), minHeight, available.height());
    const int x = std::clamp(requested.x(), available.left(),
                             std::max(available.left(), available.right() - width + 1));
    const int y = std::clamp(requested.y(), available.top(),
                             std::max(available.top(), available.bottom() - height + 1));
    return QRect(x, y, width, height);
}

void AssistiveOverlay::SubmitQuestion()
{
    const QString question = questionEdit_->text().trimmed();
    if (question.isEmpty() || !questionEdit_->isEnabled()) {
        questionEdit_->setFocus();
        return;
    }
    questionEdit_->clear();
    emit QuestionSubmitted(question);
}

JoystickOverlay::JoystickOverlay(QWidget* parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_TransparentForMouseEvents, false);
    setAttribute(Qt::WA_NoSystemBackground, true);
    setAttribute(Qt::WA_TranslucentBackground, true);
    setVisible(false);
    if (parent) {
        parent->installEventFilter(this);
    }
    setFixedSize(160, 160);
    UpdateMask();
    ResetKnob();
}

void JoystickOverlay::ResetKnob()
{
    knobPos_ = QPointF(width() / 2.0, height() / 2.0);
    update();
}

bool JoystickOverlay::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == parentWidget()) {
        if (event->type() == QEvent::Resize) {
            UpdatePlacement();
        }
    }
    return QWidget::eventFilter(watched, event);
}

void JoystickOverlay::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    UpdatePlacement();
}

void JoystickOverlay::paintEvent(QPaintEvent*)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.setCompositionMode(QPainter::CompositionMode_Source);
    painter.fillRect(rect(), Qt::transparent);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

    painter.setBrush(QColor(60, 60, 60, 180));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(rect());

    painter.setBrush(QColor(230, 230, 230, 230));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(KnobRect());
}

void JoystickOverlay::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        dragging_ = true;
        UpdateFromPosition(event->position());
    }
}

void JoystickOverlay::mouseMoveEvent(QMouseEvent* event)
{
    if (dragging_) {
        UpdateFromPosition(event->position());
    }
}

void JoystickOverlay::mouseReleaseEvent(QMouseEvent* event)
{
    if (dragging_ && event->button() == Qt::LeftButton) {
        dragging_ = false;
        ResetKnob();
        emit JoystickChanged(0.0f, 0.0f);
        update();
    }
}

void JoystickOverlay::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    UpdateMask();
}

QRectF JoystickOverlay::KnobRect() const
{
    constexpr qreal knobRadius = 24.0;
    return QRectF(knobPos_.x() - knobRadius,
                  knobPos_.y() - knobRadius,
                  knobRadius * 2.0,
                  knobRadius * 2.0);
}

void JoystickOverlay::UpdatePlacement()
{
    if (!parentWidget()) {
        return;
    }
    const int margin = 20;
    const int x = parentWidget()->width() - width() - margin;
    const int y = parentWidget()->height() - height() - margin;
    move(std::max(0, x), std::max(0, y));
}

void JoystickOverlay::UpdateFromPosition(const QPointF& pos)
{
    const QPointF center(width() / 2.0, height() / 2.0);
    QPointF delta = pos - center;
    const qreal maxRadius = width() / 2.0;
    if (delta.manhattanLength() < 0.0001) {
        delta = QPointF(0, 0);
    }
    const qreal distance = std::sqrt(delta.x() * delta.x() + delta.y() * delta.y());
    if (distance > maxRadius) {
        delta *= maxRadius / distance;
    }
    knobPos_ = center + delta;
    update();

    float normX = static_cast<float>(delta.x() / maxRadius);
    float normY = static_cast<float>(delta.y() / maxRadius);
    normX = std::clamp(normX, -1.0f, 1.0f);
    normY = std::clamp(normY, -1.0f, 1.0f);

    emit JoystickChanged(normX, -normY);
}

void JoystickOverlay::UpdateMask()
{
    if (width() <= 0 || height() <= 0) {
        clearMask();
        return;
    }
    QRegion region(rect(), QRegion::Ellipse);
    setMask(region);
}

MainWindow::MainWindow()
{
    setWindowTitle("OpenZoom");
    resize(1280, 720);

    // Whole-app readability: bigger base font, a strong visible focus outline
    // on every interactive control, and chunky slider handles. Palette roles
    // are used throughout so the native light/dark theme is preserved.
    setStyleSheet(QStringLiteral(R"(
        QWidget { font-size: 12pt; }
        QPushButton, QToolButton {
            min-height: 32px;
            padding: 4px 14px;
            border: 2px solid palette(mid);
            border-radius: 6px;
            background: palette(button);
        }
        QPushButton:hover, QToolButton:hover { background: palette(midlight); }
        QPushButton:checked, QToolButton:checked {
            background: palette(highlight);
            color: palette(highlighted-text);
        }
        QPushButton:focus, QToolButton:focus { border: 3px solid palette(highlight); }
        QPushButton:disabled, QToolButton:disabled { color: palette(mid); }
        QComboBox {
            min-height: 32px;
            padding: 2px 10px;
            border: 2px solid palette(mid);
            border-radius: 6px;
            background: palette(button);
        }
        QComboBox:focus { border: 3px solid palette(highlight); }
        QCheckBox { spacing: 8px; padding: 2px; border: 3px solid transparent; border-radius: 6px; }
        QCheckBox:focus { border-color: palette(highlight); }
        QCheckBox::indicator { width: 20px; height: 20px; }
        QSlider { min-height: 30px; border: 3px solid transparent; border-radius: 6px; }
        QSlider:focus { border-color: palette(highlight); }
        QSlider::groove:horizontal { height: 8px; border-radius: 4px; background: palette(mid); }
        QSlider::sub-page:horizontal { background: palette(highlight); border-radius: 4px; }
        QSlider::handle:horizontal {
            width: 22px;
            margin: -8px 0;
            border-radius: 11px;
            background: palette(button);
            border: 2px solid palette(dark);
        }
        QSlider::handle:horizontal:hover { background: palette(midlight); }
        QListWidget { border: 2px solid palette(mid); border-radius: 6px; }
        QListWidget:focus { border: 3px solid palette(highlight); }
        QListWidget::item { padding: 6px; }
        QListWidget::item:selected { background: palette(highlight); color: palette(highlighted-text); }
        QPushButton#simpleModeButton, QPushButton#advancedModeButton {
            font-size: 13pt;
            font-weight: 600;
            min-height: 36px;
            padding: 2px 20px;
        }
        QPushButton#simpleModeButton {
            border-top-right-radius: 0;
            border-bottom-right-radius: 0;
        }
        QPushButton#advancedModeButton {
            border-top-left-radius: 0;
            border-bottom-left-radius: 0;
        }
        QWidget#topLeftPanel, QWidget#bottomLeftPanel, QWidget#bottomRightPanel,
        QWidget#modeGridPopup, QWidget#modeToast {
            background: #111111;
            border: 3px solid #f4f4f4;
        }
        QWidget#topLeftPanel { border-top: 0; border-left: 0; border-bottom-right-radius: 8px; }
        QWidget#bottomLeftPanel { border-bottom: 0; border-left: 0; border-top-right-radius: 8px; }
        QWidget#bottomRightPanel { border-bottom: 0; border-right: 0; border-top-left-radius: 8px; }
        QWidget#modeGridPopup {
            border-left: 0;
            border-top-left-radius: 0;
            border-bottom-left-radius: 0;
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        QWidget#modeToast { border-color: palette(highlight); border-radius: 8px; }
        QLabel#brandLabel { font-size: 15pt; font-weight: 700; color: #ffffff; }
        QLabel#processingStatusLabel { font-size: 11pt; font-weight: 700; }
        QLabel#modeToastTitle { font-size: 34pt; font-weight: 800; color: #ffffff; }
        QLabel#modeToastSubtitle { font-size: 16pt; font-weight: 600; color: #ffffff; }
        QPushButton#currentModeButton { font-size: 15pt; font-weight: 700; min-width: 220px; }
        QPushButton#previousModeButton, QPushButton#nextModeButton,
        QToolButton#modeGridButton { min-width: 52px; min-height: 52px; padding: 2px; }
        QWidget#bottomRightPanel QPushButton { min-width: 88px; min-height: 58px; font-size: 11pt; }
        QToolButton#advancedTabArrow {
            min-width: 40px;
            max-width: 40px;
            min-height: 40px;
            max-height: 40px;
            padding: 5px;
        }
        QPushButton#advancedNavButton {
            min-height: 40px;
            max-height: 40px;
            padding: 5px 10px;
        }
        QListWidget#presetList {
            background: #111111;
            border: 0;
            border-radius: 0;
            font-size: 13pt;
            font-weight: 650;
            outline: 0;
        }
        QListWidget#presetList::item {
            background: #1c1c1c;
            color: #ffffff;
            border: 2px solid #f4f4f4;
            border-radius: 6px;
            padding: 10px;
        }
        QListWidget#presetList::item:selected {
            background: palette(highlight);
            color: palette(highlighted-text);
            border: 3px solid #ffffff;
        }
        QListWidget#presetList::item:focus { border: 3px solid palette(highlight); }
        QTextBrowser, QPlainTextEdit {
            padding: 7px;
            border: 2px solid palette(mid);
            border-radius: 6px;
            background: palette(base);
        }
        QTextBrowser:focus, QPlainTextEdit:focus { border: 3px solid palette(highlight); }
        QTabBar::tab { min-height: 34px; min-width: 92px; padding: 4px 10px; }
        QLabel#sectionLabel {
            font-size: 13pt;
            font-weight: 600;
            padding: 8px 0 4px 0;
            color: palette(highlight);
        }
    )"));

    auto* central = new QWidget(this);
    auto* rootLayout = new QVBoxLayout(central);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    // These controls are parented into solid overlays after the render widget
    // is created. Advanced keeps the same inspector and reuses the mode switch.
    simpleModeButton_ = new QPushButton("Simple");
    simpleModeButton_->setObjectName("simpleModeButton");
    simpleModeButton_->setCheckable(true);
    simpleModeButton_->setChecked(true);
    advancedModeButton_ = new QPushButton("Advanced");
    advancedModeButton_->setObjectName("advancedModeButton");
    advancedModeButton_->setCheckable(true);
    auto* modeGroup = new QButtonGroup(this);
    modeGroup->setExclusive(true);
    modeGroup->addButton(simpleModeButton_);
    modeGroup->addButton(advancedModeButton_);

    processingStatusLabel_ = new QLabel("Processing: CPU");
    processingStatusLabel_->setObjectName("processingStatusLabel");
    processingStatusLabel_->setWordWrap(true);
    processingStatusLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse |
                                                     Qt::TextSelectableByKeyboard);
    processingStatusLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    processingStatusLabel_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

    presetList_ = new QListWidget();
    presetList_->setObjectName("presetList");
    presetList_->setSelectionMode(QAbstractItemView::SingleSelection);
    presetList_->setFocusPolicy(Qt::StrongFocus);
    presetList_->setFlow(QListView::LeftToRight);
    presetList_->setViewMode(QListView::IconMode);
    presetList_->setWrapping(true);
    presetList_->setMovement(QListView::Static);
    presetList_->setResizeMode(QListView::Adjust);
    presetList_->setWordWrap(true);
    presetList_->setTextElideMode(Qt::ElideNone);
    presetList_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    presetList_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    presetList_->setGridSize(QSize(205, 102));
    presetList_->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
    presetList_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    presetList_->setSpacing(4);

    presetDescriptionLabel_ = new QLabel();
    presetDescriptionLabel_->setObjectName("presetDescriptionLabel");
    presetDescriptionLabel_->hide();

    capturePhotoButton_ = new QPushButton("Photo");
    recordButton_ = new QPushButton("Record");
    recordButton_->setCheckable(true);
    explainNowButton_ = new QPushButton("Explain");
    readTextButton_ = new QPushButton("Read");
    capturePhotoButton_->setToolTip("Capture processed photo");
    recordButton_->setToolTip("Start or stop recording");
    explainNowButton_->setToolTip("Explain the current view");
    readTextButton_->setToolTip("Read text in the current view");

    // Advanced mode uses a right-side inspector so detailed controls do not
    // push the live image below the fold.
    auto* advancedPage = new QWidget();
    advancedPage->setObjectName("advancedPage");
    auto* advancedLayout = new QVBoxLayout(advancedPage);
    advancedLayout->setContentsMargins(10, 4, 10, 10);
    advancedLayout->setSpacing(8);

    auto makeSectionLabel = [](const QString& text) {
        auto* label = new QLabel(text);
        label->setObjectName("sectionLabel");
        return label;
    };

    advancedLayout->addWidget(makeSectionLabel("Global device"));
    auto* cameraLabel = new QLabel("Camera");
    cameraCombo_ = new QComboBox();
    cameraCombo_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    cameraCombo_->setMinimumContentsLength(20);
    cameraLabel->setBuddy(cameraCombo_);
    advancedLayout->addWidget(cameraLabel);
    advancedLayout->addWidget(cameraCombo_);

    auto* rotationLabel = new QLabel("Orientation");
    rotationCombo_ = new QComboBox();
    rotationCombo_->addItem("0°");
    rotationCombo_->addItem("90°");
    rotationCombo_->addItem("180°");
    rotationCombo_->addItem("270°");
    rotationCombo_->setCurrentIndex(0);
    rotationCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    rotationCombo_->setEditable(false);
    rotationCombo_->setToolTip("Rotate input/output clockwise");
    rotationLabel->setBuddy(rotationCombo_);
    advancedLayout->addWidget(rotationLabel);
    advancedLayout->addWidget(rotationCombo_);

    controlsToggleButton_ = new QToolButton();
    controlsToggleButton_->setText("Advanced Tuning");
    controlsToggleButton_->setCheckable(true);
    controlsToggleButton_->setChecked(false);
    controlsToggleButton_->setArrowType(Qt::RightArrow);
    controlsToggleButton_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    promotePresetButton_ = new QPushButton("Save As Quick Option");
    joystickCheckbox_ = new QCheckBox("Virtual Joystick");

    auto* modesLabel = new QLabel("Available modes");
    cameraModesList_ = new QListWidget();
    cameraModesList_->setSelectionMode(QAbstractItemView::NoSelection);
    cameraModesList_->setFocusPolicy(Qt::NoFocus);
    cameraModesList_->setMinimumHeight(72);
    cameraModesList_->setMaximumHeight(100);
    cameraModesList_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    cameraModesList_->setSizeAdjustPolicy(QListWidget::AdjustToContents);
    auto* modesLayout = new QVBoxLayout();
    modesLayout->setContentsMargins(0, 0, 0, 0);
    modesLayout->setSpacing(4);
    modesLayout->addWidget(modesLabel);
    modesLayout->addWidget(cameraModesList_);
    advancedLayout->addLayout(modesLayout);

    advancedLayout->addWidget(makeSectionLabel("Current profile"));
    auto* advancedHeaderLayout = new QVBoxLayout();
    advancedHeaderLayout->setSpacing(6);
    controlsToggleButton_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    promotePresetButton_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    advancedHeaderLayout->addWidget(controlsToggleButton_);
    advancedHeaderLayout->addWidget(promotePresetButton_);
    advancedLayout->addLayout(advancedHeaderLayout);

    controlsContainer_ = new QWidget();
    auto* controlsLayout = new QVBoxLayout(controlsContainer_);
    controlsLayout->setContentsMargins(0, 0, 0, 0);
    controlsLayout->setSpacing(8);

    controlsLayout->addWidget(makeSectionLabel("Magnification and image"));

    auto* bwLayout = new QHBoxLayout();
    bwLayout->setSpacing(8);
    bwCheckbox_ = new QCheckBox("Black && White");
    bwSlider_ = new QSlider(Qt::Horizontal);
    bwSlider_->setRange(0, 255);
    bwSlider_->setPageStep(8);
    bwSlider_->setValue(128);
    bwSlider_->setEnabled(false);
    bwLayout->addWidget(bwCheckbox_);
    bwLayout->addWidget(bwSlider_, 1);
    controlsLayout->addLayout(bwLayout);

    auto* zoomLayout = new QHBoxLayout();
    zoomLayout->setSpacing(8);
    zoomCheckbox_ = new QCheckBox("Zoom");
    zoomSlider_ = new QSlider(Qt::Horizontal);
    zoomSlider_->setRange(kZoomSliderScale, kZoomSliderMaxMultiplier * kZoomSliderScale);
    zoomSlider_->setPageStep(10);
    zoomSlider_->setValue(kZoomSliderScale);
    zoomSlider_->setEnabled(false);
    zoomLayout->addWidget(zoomCheckbox_);
    zoomLayout->addWidget(zoomSlider_, 1);
    controlsLayout->addLayout(zoomLayout);

    auto* blurLayout = new QVBoxLayout();
    blurLayout->setSpacing(8);
    blurCheckbox_ = new QCheckBox("Gaussian Blur");
    blurSigmaSlider_ = new QSlider(Qt::Horizontal);
    blurSigmaSlider_->setRange(kBlurSigmaSliderMin, kBlurSigmaSliderMax);
    blurSigmaSlider_->setPageStep(2);
    blurSigmaSlider_->setSingleStep(1);
    blurSigmaSlider_->setValue(10);
    blurSigmaSlider_->setEnabled(false);
    blurSigmaValueLabel_ = new QLabel("1.0");
    blurSigmaValueLabel_->setMinimumWidth(40);

    blurRadiusSlider_ = new QSlider(Qt::Horizontal);
    blurRadiusSlider_->setRange(kSupportedBlurRadii.front(), kSupportedBlurRadii.back());
    blurRadiusSlider_->setPageStep(1);
    blurRadiusSlider_->setSingleStep(1);
    blurRadiusSlider_->setTickInterval(1);
    blurRadiusSlider_->setTickPosition(QSlider::TicksBelow);
    blurRadiusSlider_->setValue(3);
    blurRadiusSlider_->setEnabled(false);
    blurRadiusValueLabel_ = new QLabel("3");
    blurRadiusValueLabel_->setMinimumWidth(40);

    blurLayout->addWidget(blurCheckbox_);
    auto* blurSigmaLayout = new QHBoxLayout();
    blurSigmaLayout->addWidget(new QLabel("Sigma"));
    blurSigmaLayout->addWidget(blurSigmaSlider_, 1);
    blurSigmaLayout->addWidget(blurSigmaValueLabel_);
    blurLayout->addLayout(blurSigmaLayout);
    auto* blurRadiusLayout = new QHBoxLayout();
    blurRadiusLayout->addWidget(new QLabel("Radius"));
    blurRadiusLayout->addWidget(blurRadiusSlider_, 1);
    blurRadiusLayout->addWidget(blurRadiusValueLabel_);
    blurLayout->addLayout(blurRadiusLayout);
    controlsLayout->addLayout(blurLayout);

    controlsLayout->addWidget(makeSectionLabel("Motion and display"));
    auto* temporalLayout = new QHBoxLayout();
    temporalLayout->setSpacing(8);
    temporalSmoothCheckbox_ = new QCheckBox("Temporal Smooth");
    temporalSmoothCheckbox_->setChecked(false);
    temporalSmoothSlider_ = new QSlider(Qt::Horizontal);
    temporalSmoothSlider_->setRange(5, 100);
    temporalSmoothSlider_->setPageStep(5);
    temporalSmoothSlider_->setValue(25);
    temporalSmoothSlider_->setEnabled(false);
    temporalSmoothValueLabel_ = new QLabel("0.25");
    temporalSmoothValueLabel_->setMinimumWidth(40);
    temporalLayout->addWidget(temporalSmoothCheckbox_);
    temporalLayout->addSpacing(12);
    temporalLayout->addWidget(new QLabel("Blend:"));
    temporalLayout->addWidget(temporalSmoothSlider_, 1);
    temporalLayout->addWidget(temporalSmoothValueLabel_);
    controlsLayout->addLayout(temporalLayout);

    auto* stabilizationLayout = new QHBoxLayout();
    stabilizationLayout->setSpacing(8);
    stabilizationCheckbox_ = new QCheckBox("Stabilize Image");
    stabilizationStrengthSlider_ = new QSlider(Qt::Horizontal);
    stabilizationStrengthSlider_->setRange(0, 98);
    stabilizationStrengthSlider_->setPageStep(5);
    stabilizationStrengthSlider_->setValue(85);
    stabilizationStrengthSlider_->setEnabled(false);
    stabilizationLayout->addWidget(stabilizationCheckbox_);
    stabilizationLayout->addSpacing(12);
    stabilizationLayout->addWidget(new QLabel("Strength:"));
    stabilizationLayout->addWidget(stabilizationStrengthSlider_, 1);
    controlsLayout->addLayout(stabilizationLayout);

    controlsLayout->addWidget(makeSectionLabel("Screen fix"));
    auto* keystoneLayout = new QHBoxLayout();
    keystoneLayout->setSpacing(8);
    keystoneCheckbox_ = new QCheckBox("Straighten Screen (Keystone)");
    keystoneLayout->addWidget(keystoneCheckbox_);
    keystoneLayout->addStretch(1);
    controlsLayout->addLayout(keystoneLayout);

    auto* autoContrastLayout = new QHBoxLayout();
    autoContrastLayout->setSpacing(8);
    autoContrastCheckbox_ = new QCheckBox("Auto Contrast");
    autoContrastStrengthSlider_ = new QSlider(Qt::Horizontal);
    autoContrastStrengthSlider_->setRange(0, 100);
    autoContrastStrengthSlider_->setPageStep(5);
    autoContrastStrengthSlider_->setValue(70);
    autoContrastStrengthSlider_->setEnabled(false);
    autoContrastLayout->addWidget(autoContrastCheckbox_);
    autoContrastLayout->addSpacing(12);
    autoContrastLayout->addWidget(new QLabel("Strength:"));
    autoContrastLayout->addWidget(autoContrastStrengthSlider_, 1);
    controlsLayout->addLayout(autoContrastLayout);

    auto* displayColorLayout = new QVBoxLayout();
    displayColorLayout->setSpacing(8);
    displayColorLayout->addWidget(new QLabel("Display colors"));
    displayColorCombo_ = new QComboBox();
    displayColorCombo_->addItem("Normal Colors");
    displayColorCombo_->addItem("Inverted");
    displayColorCombo_->addItem("White on Black");
    displayColorCombo_->addItem("Yellow on Black");
    displayColorCombo_->addItem("Black on Yellow");
    displayColorCombo_->setCurrentIndex(0);
    displayColorLayout->addWidget(displayColorCombo_);
    contrastSlider_ = new QSlider(Qt::Horizontal);
    contrastSlider_->setRange(25, 400);
    contrastSlider_->setPageStep(25);
    contrastSlider_->setValue(100);
    auto* contrastLayout = new QHBoxLayout();
    contrastLayout->addWidget(new QLabel("Contrast"));
    contrastLayout->addWidget(contrastSlider_, 1);
    displayColorLayout->addLayout(contrastLayout);
    brightnessSlider_ = new QSlider(Qt::Horizontal);
    brightnessSlider_->setRange(-100, 100);
    brightnessSlider_->setPageStep(10);
    brightnessSlider_->setValue(0);
    auto* brightnessLayout = new QHBoxLayout();
    brightnessLayout->addWidget(new QLabel("Brightness"));
    brightnessLayout->addWidget(brightnessSlider_, 1);
    displayColorLayout->addLayout(brightnessLayout);
    controlsLayout->addLayout(displayColorLayout);

    controlsLayout->addWidget(makeSectionLabel("Assistive"));
    auto* assistiveLayout = new QVBoxLayout();
    assistiveLayout->setSpacing(6);
    ocrAssistCheckbox_ = new QCheckBox("OCR Assist");
    vlmAssistCheckbox_ = new QCheckBox("Scene Explain");
    assistiveOverlayCheckbox_ = new QCheckBox("Assistive Overlay");
    assistiveOverlayCheckbox_->setChecked(true);
    aiSettingsButton_ = new QPushButton(QStringLiteral("AI Settings"));
    aiSettingsButton_->setObjectName(QStringLiteral("advancedNavButton"));
    aiSettingsButton_->setIcon(QIcon(QStringLiteral(":/openzoom/icons/open-settings.svg")));
    aiSettingsButton_->setIconSize(QSize(26, 26));
    aiSettingsButton_->setToolTip(QStringLiteral("Open AI Settings dialog"));
    openNotesButton_ = new QPushButton("Open Notes");
    assistiveLayout->addWidget(ocrAssistCheckbox_);
    assistiveLayout->addWidget(vlmAssistCheckbox_);
    assistiveLayout->addWidget(assistiveOverlayCheckbox_);
    assistiveLayout->addWidget(openNotesButton_);
    controlsLayout->addLayout(assistiveLayout);

    controlsLayout->addWidget(makeSectionLabel("Sharpen and focus"));
    auto* spatialLayout = new QVBoxLayout();
    spatialLayout->setSpacing(6);
    spatialSharpenCheckbox_ = new QCheckBox("Spatial Sharpen");
    spatialSharpenCheckbox_->setChecked(false);
    spatialBackendCombo_ = new QComboBox();
    spatialBackendCombo_->addItem("AMD FSR 1.0 (EASU + RCAS)");
    spatialBackendCombo_->addItem("NVIDIA Image Scaling (default)");
    spatialBackendCombo_->setEnabled(false);
    spatialSharpnessSlider_ = new QSlider(Qt::Horizontal);
    spatialSharpnessSlider_->setRange(0, 100);
    spatialSharpnessSlider_->setPageStep(5);
    spatialSharpnessSlider_->setValue(25);
    spatialSharpnessSlider_->setEnabled(false);
    spatialSharpnessValueLabel_ = new QLabel("0.25");
    spatialSharpnessValueLabel_->setMinimumWidth(40);

    spatialLayout->addWidget(spatialSharpenCheckbox_);
    spatialLayout->addWidget(new QLabel("Backend"));
    spatialLayout->addWidget(spatialBackendCombo_);
    auto* sharpnessLayout = new QHBoxLayout();
    sharpnessLayout->addWidget(new QLabel("Sharpness"));
    sharpnessLayout->addWidget(spatialSharpnessSlider_, 1);
    sharpnessLayout->addWidget(spatialSharpnessValueLabel_);
    spatialLayout->addLayout(sharpnessLayout);
    controlsLayout->addLayout(spatialLayout);

    auto* focusLayout = new QVBoxLayout();
    focusLayout->setSpacing(8);
    auto* focusXLabel = new QLabel("Focus X:");
    zoomCenterXSlider_ = new QSlider(Qt::Horizontal);
    zoomCenterXSlider_->setRange(0, kZoomFocusSliderScale);
    zoomCenterXSlider_->setPageStep(5);
    zoomCenterXSlider_->setValue(kZoomFocusSliderScale / 2);
    auto* focusXLayout = new QHBoxLayout();
    focusXLayout->addWidget(focusXLabel);
    focusXLayout->addWidget(zoomCenterXSlider_, 1);
    focusLayout->addLayout(focusXLayout);

    auto* focusYLabel = new QLabel("Focus Y:");
    zoomCenterYSlider_ = new QSlider(Qt::Horizontal);
    zoomCenterYSlider_->setRange(0, kZoomFocusSliderScale);
    zoomCenterYSlider_->setPageStep(5);
    zoomCenterYSlider_->setValue(kZoomFocusSliderScale / 2);
    auto* focusYLayout = new QHBoxLayout();
    focusYLayout->addWidget(focusYLabel);
    focusYLayout->addWidget(zoomCenterYSlider_, 1);
    focusLayout->addLayout(focusYLayout);
    controlsLayout->addLayout(focusLayout);

    controlsLayout->addWidget(makeSectionLabel("Interaction and diagnostics"));
    auto* debugLayout = new QHBoxLayout();
    debugLayout->setSpacing(8);
    debugButton_ = new QPushButton("Debug View");
    debugButton_->setCheckable(true);
    debugButton_->setChecked(false);
    debugLayout->addWidget(debugButton_);
    focusMarkerCheckbox_ = new QCheckBox("Show Focus Point");
    focusMarkerCheckbox_->setChecked(false);
    focusMarkerCheckbox_->setToolTip("Overlay a red marker at the current zoom focus");
    debugLayout->addWidget(focusMarkerCheckbox_);
    debugLayout->addWidget(joystickCheckbox_);
    debugLayout->addStretch(1);
    controlsLayout->addLayout(debugLayout);
    auto* processingStatusLayout = new QVBoxLayout();
    processingStatusLayout->setSpacing(3);
    processingStatusLayout->addWidget(new QLabel(QStringLiteral("Pipeline status")));
    processingStatusLayout->addWidget(processingStatusLabel_);
    controlsLayout->addLayout(processingStatusLayout);

    advancedLayout->addWidget(controlsContainer_);

    auto* advancedScroll = new QScrollArea();
    advancedScroll->setWidget(advancedPage);
    advancedScroll->setWidgetResizable(true);
    advancedScroll->setFrameShape(QFrame::NoFrame);
    advancedScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    advancedScroll->setMinimumWidth(380);
    advancedScroll->setMaximumWidth(520);
    advancedScroll->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

    auto* assistantPage = new QWidget();
    auto* assistantLayout = new QVBoxLayout(assistantPage);
    assistantLayout->setContentsMargins(10, 8, 10, 10);
    assistantLayout->setSpacing(8);

    assistantConnectionLabel_ = new QLabel("Starting Codex...");
    assistantConnectionLabel_->setWordWrap(true);
    assistantUsageLabel_ = new QLabel();
    assistantUsageLabel_->setWordWrap(true);
    assistantConnectButton_ = new QPushButton("Connect ChatGPT");
    assistantConnectButton_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    assistantLayout->addWidget(assistantConnectionLabel_);
    assistantLayout->addWidget(assistantUsageLabel_);
    assistantLayout->addWidget(assistantConnectButton_);

    auto* assistantTabs = new QTabWidget();
    auto* chatPage = new QWidget();
    auto* chatLayout = new QVBoxLayout(chatPage);
    chatLayout->setContentsMargins(0, 6, 0, 0);
    chatLayout->setSpacing(8);
    assistantTranscript_ = new QTextBrowser();
    assistantTranscript_->setOpenExternalLinks(false);
    assistantTranscript_->setPlaceholderText("Ask about the current camera view.");
    assistantTranscript_->setMinimumHeight(220);
    assistantPromptEdit_ = new QPlainTextEdit();
    assistantPromptEdit_->setPlaceholderText("Ask about what is visible...");
    assistantPromptEdit_->setMaximumHeight(105);
    assistantAttachFrameCheckbox_ = new QCheckBox("Attach current view");
    assistantAttachFrameCheckbox_->setChecked(true);
    assistantNewButton_ = new QPushButton("New Conversation");
    assistantSendButton_ = new QPushButton("Send");
    assistantStopButton_ = new QPushButton("Stop");
    assistantStopButton_->setEnabled(false);
    auto* chatActions = new QHBoxLayout();
    chatActions->setSpacing(6);
    chatActions->addWidget(assistantNewButton_);
    chatActions->addStretch(1);
    chatActions->addWidget(assistantStopButton_);
    chatActions->addWidget(assistantSendButton_);
    chatLayout->addWidget(assistantTranscript_, 1);
    chatLayout->addWidget(assistantPromptEdit_);
    chatLayout->addWidget(assistantAttachFrameCheckbox_);
    chatLayout->addLayout(chatActions);

    auto* historyPage = new QWidget();
    auto* historyLayout = new QVBoxLayout(historyPage);
    historyLayout->setContentsMargins(0, 6, 0, 0);
    historyLayout->setSpacing(8);
    assistantHistoryList_ = new QListWidget();
    assistantHistoryList_->setSelectionMode(QAbstractItemView::SingleSelection);
    assistantHistoryList_->setWordWrap(true);
    assistantRenameButton_ = new QPushButton("Rename");
    assistantExportButton_ = new QPushButton("Export");
    assistantDeleteButton_ = new QPushButton("Delete");
    auto* historyActions = new QHBoxLayout();
    historyActions->setSpacing(6);
    historyActions->addWidget(assistantRenameButton_);
    historyActions->addWidget(assistantExportButton_);
    historyActions->addWidget(assistantDeleteButton_);
    historyLayout->addWidget(assistantHistoryList_, 1);
    historyLayout->addLayout(historyActions);

    assistantTabs->addTab(chatPage, "Chat");
    assistantTabs->addTab(historyPage, "History");
    assistantLayout->addWidget(assistantTabs, 1);

    auto* advancedTabs = new QTabWidget();
    advancedTabs->addTab(advancedScroll, "Image");
    advancedTabs->addTab(assistantPage, "Assistant");
    auto* previousAdvancedTabButton = new QToolButton();
    previousAdvancedTabButton->setObjectName(QStringLiteral("advancedTabArrow"));
    previousAdvancedTabButton->setIcon(QIcon(QStringLiteral(":/openzoom/icons/previous.svg")));
    previousAdvancedTabButton->setIconSize(QSize(26, 26));
    previousAdvancedTabButton->setToolTip(QStringLiteral("Previous Advanced section"));
    previousAdvancedTabButton->setAccessibleName(QStringLiteral("Previous Advanced section"));
    auto* nextAdvancedTabButton = new QToolButton();
    nextAdvancedTabButton->setObjectName(QStringLiteral("advancedTabArrow"));
    nextAdvancedTabButton->setIcon(QIcon(QStringLiteral(":/openzoom/icons/next.svg")));
    nextAdvancedTabButton->setIconSize(QSize(26, 26));
    nextAdvancedTabButton->setToolTip(QStringLiteral("Next Advanced section"));
    nextAdvancedTabButton->setAccessibleName(QStringLiteral("Next Advanced section"));
    auto* leftTabCorner = new QWidget();
    auto* leftTabCornerLayout = new QHBoxLayout(leftTabCorner);
    leftTabCornerLayout->setContentsMargins(0, 0, 4, 0);
    leftTabCornerLayout->addWidget(previousAdvancedTabButton);
    auto* rightTabCorner = new QWidget();
    auto* rightTabCornerLayout = new QHBoxLayout(rightTabCorner);
    rightTabCornerLayout->setContentsMargins(4, 0, 0, 0);
    rightTabCornerLayout->setSpacing(4);
    rightTabCornerLayout->addWidget(aiSettingsButton_);
    rightTabCornerLayout->addWidget(nextAdvancedTabButton);
    advancedTabs->setCornerWidget(leftTabCorner, Qt::TopLeftCorner);
    advancedTabs->setCornerWidget(rightTabCorner, Qt::TopRightCorner);
    connect(previousAdvancedTabButton, &QToolButton::clicked, this, [advancedTabs]() {
        const int count = advancedTabs->count();
        if (count > 0) advancedTabs->setCurrentIndex((advancedTabs->currentIndex() + count - 1) % count);
    });
    connect(nextAdvancedTabButton, &QToolButton::clicked, this, [advancedTabs]() {
        const int count = advancedTabs->count();
        if (count > 0) advancedTabs->setCurrentIndex((advancedTabs->currentIndex() + 1) % count);
    });
    advancedTabs->setMinimumWidth(420);
    advancedTabs->setMaximumWidth(580);
    advancedTabs->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    advancedPanel_ = advancedTabs;

    renderWidget_ = new RenderWidget();
    renderWidget_->installEventFilter(this);
    renderWidget_->setMouseTracking(true);
    renderWidget_->setMinimumSize(320, 240);
    auto* contentLayout = new QHBoxLayout();
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(0);
    contentLayout->addWidget(renderWidget_, 1);
    contentLayout->addWidget(advancedPanel_, 0);
    rootLayout->addLayout(contentLayout, 1);

    // D3D presents directly into a native window, so frameless owned tool
    // windows provide predictable stacking and opacity above the swap chain.
    const Qt::WindowFlags chromeFlags = Qt::Tool |
                                        Qt::FramelessWindowHint |
                                        Qt::NoDropShadowWindowHint;
    topLeftPanel_ = new QWidget(this, chromeFlags);
    topLeftPanel_->setObjectName("topLeftPanel");
    auto* topLeftLayout = new QHBoxLayout(topLeftPanel_);
    topLeftLayout->setContentsMargins(10, 8, 8, 8);
    topLeftLayout->setSpacing(8);
    auto* brandLabel = new QLabel("OpenZoom");
    brandLabel->setObjectName("brandLabel");
    topLeftLayout->addWidget(brandLabel);
    topLeftLayout->addWidget(simpleModeButton_);
    topLeftLayout->addWidget(advancedModeButton_);

    bottomLeftPanel_ = new QWidget(this, chromeFlags);
    bottomLeftPanel_->setObjectName("bottomLeftPanel");
    auto* bottomLeftLayout = new QHBoxLayout(bottomLeftPanel_);
    bottomLeftLayout->setContentsMargins(8, 8, 10, 10);
    bottomLeftLayout->setSpacing(6);
    modeGridButton_ = new QToolButton();
    modeGridButton_->setObjectName("modeGridButton");
    modeGridButton_->setIcon(style()->standardIcon(QStyle::SP_FileDialogListView));
    modeGridButton_->setIconSize(QSize(30, 30));
    modeGridButton_->setToolTip("Show all quick modes");
    previousModeButton_ = new QPushButton();
    previousModeButton_->setObjectName("previousModeButton");
    previousModeButton_->setIcon(style()->standardIcon(QStyle::SP_ArrowBack));
    previousModeButton_->setIconSize(QSize(30, 30));
    previousModeButton_->setToolTip("Previous quick mode");
    currentModeButton_ = new QPushButton("Read a Page   [1]");
    currentModeButton_->setObjectName("currentModeButton");
    currentModeButton_->setToolTip("Show all quick modes");
    nextModeButton_ = new QPushButton();
    nextModeButton_->setObjectName("nextModeButton");
    nextModeButton_->setIcon(style()->standardIcon(QStyle::SP_ArrowForward));
    nextModeButton_->setIconSize(QSize(30, 30));
    nextModeButton_->setToolTip("Next quick mode");
    bottomLeftLayout->addWidget(modeGridButton_);
    bottomLeftLayout->addWidget(previousModeButton_);
    bottomLeftLayout->addWidget(currentModeButton_);
    bottomLeftLayout->addWidget(nextModeButton_);

    bottomRightPanel_ = new QWidget(this, chromeFlags);
    bottomRightPanel_->setObjectName("bottomRightPanel");
    auto* bottomRightLayout = new QHBoxLayout(bottomRightPanel_);
    bottomRightLayout->setContentsMargins(10, 8, 8, 10);
    bottomRightLayout->setSpacing(6);
    capturePhotoButton_->setIcon(QIcon(QStringLiteral(":/openzoom/icons/camera.svg")));
    recordButton_->setIcon(QIcon(QStringLiteral(":/openzoom/icons/record.svg")));
    explainNowButton_->setIcon(QIcon(QStringLiteral(":/openzoom/icons/explain.svg")));
    readTextButton_->setIcon(QIcon(QStringLiteral(":/openzoom/icons/read.svg")));
    for (QPushButton* button : {capturePhotoButton_, recordButton_, explainNowButton_, readTextButton_}) {
        button->setIconSize(QSize(28, 28));
    }
    bottomRightLayout->addWidget(capturePhotoButton_);
    bottomRightLayout->addWidget(recordButton_);
    bottomRightLayout->addWidget(explainNowButton_);
    bottomRightLayout->addWidget(readTextButton_);

    modeGridPopup_ = new QWidget(this, chromeFlags);
    modeGridPopup_->setObjectName("modeGridPopup");
    auto* modeGridLayout = new QVBoxLayout(modeGridPopup_);
    modeGridLayout->setContentsMargins(8, 8, 8, 8);
    modeGridLayout->addWidget(presetList_);
    modeGridPopup_->hide();

    modeToast_ = new QWidget(this, Qt::ToolTip |
                                   Qt::FramelessWindowHint |
                                   Qt::NoDropShadowWindowHint);
    modeToast_->setObjectName("modeToast");
    auto* modeToastLayout = new QVBoxLayout(modeToast_);
    modeToastLayout->setContentsMargins(28, 18, 28, 18);
    modeToastLayout->setSpacing(4);
    modeToastTitle_ = new QLabel("READ A PAGE");
    modeToastTitle_->setObjectName("modeToastTitle");
    modeToastTitle_->setAlignment(Qt::AlignCenter);
    modeToastSubtitle_ = new QLabel("Reading profile");
    modeToastSubtitle_->setObjectName("modeToastSubtitle");
    modeToastSubtitle_->setAlignment(Qt::AlignCenter);
    modeToastLayout->addWidget(modeToastTitle_);
    modeToastLayout->addWidget(modeToastSubtitle_);
    modeToast_->setAccessibleName("Quick mode changed");
    modeToast_->setAttribute(Qt::WA_ShowWithoutActivating);
    modeToast_->setAttribute(Qt::WA_TransparentForMouseEvents);
    modeToast_->hide();
    for (QWidget* panel : {topLeftPanel_, bottomLeftPanel_, bottomRightPanel_}) {
        panel->setAttribute(Qt::WA_ShowWithoutActivating);
    }

    simpleChromeIdleTimer_ = new QTimer(this);
    simpleChromeIdleTimer_->setSingleShot(true);
    simpleChromeIdleTimer_->setInterval(kSimpleChromeIdleMs);
    connect(simpleChromeIdleTimer_, &QTimer::timeout, this, &MainWindow::FadeSimpleChrome);
    modeToastTimer_ = new QTimer(this);
    modeToastTimer_->setSingleShot(true);
    modeToastTimer_->setInterval(kModeToastMs);
    connect(modeToastTimer_, &QTimer::timeout, this, [this]() {
        modeToast_->hide();
    });

    connect(modeGridButton_, &QToolButton::clicked, this, &MainWindow::ToggleModeGrid);
    connect(currentModeButton_, &QPushButton::clicked, this, &MainWindow::ToggleModeGrid);
    connect(previousModeButton_, &QPushButton::clicked, this, [this]() { ActivateRelativePreset(-1); });
    connect(nextModeButton_, &QPushButton::clicked, this, [this]() { ActivateRelativePreset(1); });
    connect(presetList_, &QListWidget::currentItemChanged,
            this, &MainWindow::UpdateCurrentPresetUi);
    connect(presetList_, &QListWidget::itemClicked, this, [this]() {
        modeGridPopup_->hide();
        RevealSimpleChrome();
    });
    connect(presetList_->model(), &QAbstractItemModel::rowsInserted,
            this, [this](const QModelIndex&, int first, int last) {
                for (int row = first; row <= last; ++row) {
                    QListWidgetItem* item = presetList_->item(row);
                    if (!item) continue;
                    const QString original = item->text();
                    const QString plain = PlainPresetLabel(original);
                    item->setData(Qt::AccessibleTextRole, plain);
                    item->setData(Qt::StatusTipRole, original);
                    item->setText(row < kSimpleShortcutCount
                                      ? QStringLiteral("%1\n%2").arg(row + 1).arg(plain)
                                      : plain);
                    item->setTextAlignment(Qt::AlignCenter);
                }
            });
    connect(recordButton_, &QPushButton::toggled, this, [this](bool checked) {
        recordButton_->setIcon(style()->standardIcon(checked ? QStyle::SP_MediaStop
                                                             : QStyle::SP_MediaPlay));
    });

    qApp->installEventFilter(this);
    qApp->installNativeEventFilter(this);
    QTimer::singleShot(0, this, [this]() {
        UpdateSimpleChromeGeometry();
        RevealSimpleChrome();
    });

    connect(simpleModeButton_, &QPushButton::toggled, this, [this](bool checked) {
        if (checked) {
            setSimpleMode(true);
        }
    });
    connect(advancedModeButton_, &QPushButton::toggled, this, [this](bool checked) {
        if (checked) {
            setSimpleMode(false);
        }
    });
    connect(stabilizationCheckbox_, &QCheckBox::toggled,
            stabilizationStrengthSlider_, &QWidget::setEnabled);
    connect(autoContrastCheckbox_, &QCheckBox::toggled,
            autoContrastStrengthSlider_, &QWidget::setEnabled);

    // Screen-reader metadata for every interactive control. OpenZoom targets
    // low-vision users, many of whom drive the app through Narrator/NVDA.
    auto setA11y = [](QWidget* widget, const QString& name, const QString& description) {
        widget->setAccessibleName(name);
        widget->setAccessibleDescription(description);
    };
    setA11y(controlsToggleButton_, "Advanced Tuning",
            "Show or hide the advanced tuning controls");
    setA11y(promotePresetButton_, "Save As Quick Option",
            "Save the current advanced settings as a reusable quick mode");
    setA11y(capturePhotoButton_, "Capture Photo",
            "Save the current processed frame as an image file");
    setA11y(recordButton_, "Record Video",
            "Start or stop recording the processed video to an MP4 file");
    setA11y(joystickCheckbox_, "Virtual Joystick",
            "Show an on-screen joystick overlay for panning the zoom focus");
    setA11y(rotationCombo_, "Rotation",
            "Rotate the camera image clockwise in 90 degree steps");
    setA11y(cameraCombo_, "Camera",
            "Select the active camera device");
    setA11y(cameraModesList_, "Camera Modes",
            "Capture modes supported by the selected camera");
    setA11y(presetList_, "Quick Modes",
            "Choose a task-oriented preset such as reading or high contrast");
    setA11y(bwCheckbox_, "Black and White",
            "Convert the image to thresholded black and white");
    setA11y(bwSlider_, "Black and White Threshold",
            "Brightness threshold for the black and white conversion");
    setA11y(zoomCheckbox_, "Zoom",
            "Enable the magnifier");
    setA11y(zoomSlider_, "Zoom Amount",
            "Magnification level");
    setA11y(blurCheckbox_, "Gaussian Blur",
            "Smooth the image with a Gaussian blur");
    setA11y(blurSigmaSlider_, "Blur Sigma",
            "Strength of the Gaussian blur");
    setA11y(blurRadiusSlider_, "Blur Radius",
            "Radius of the Gaussian blur in pixels");
    setA11y(temporalSmoothCheckbox_, "Temporal Smooth",
            "Reduce flicker by averaging consecutive frames");
    setA11y(temporalSmoothSlider_, "Temporal Smooth Blend",
            "How strongly new frames blend into the running average");
    setA11y(ocrAssistCheckbox_, "OCR Assist",
            "Read on-screen text aloud using optical character recognition");
    setA11y(vlmAssistCheckbox_, "Scene Explain",
            "Describe the magnified scene using an AI vision model");
    setA11y(assistiveOverlayCheckbox_, "Assistive Overlay",
            "Show OCR and scene descriptions as an on-screen overlay");
    setA11y(spatialSharpenCheckbox_, "Spatial Sharpen",
            "Sharpen and upscale the image on the GPU");
    setA11y(spatialBackendCombo_, "Sharpen Backend",
            "Choose the GPU sharpening algorithm");
    setA11y(spatialSharpnessSlider_, "Sharpness",
            "Strength of the spatial sharpening");
    setA11y(zoomCenterXSlider_, "Focus X",
            "Horizontal position of the zoom focus");
    setA11y(zoomCenterYSlider_, "Focus Y",
            "Vertical position of the zoom focus");
    setA11y(debugButton_, "Debug View",
            "Show the intermediate processing stages in a grid");
    setA11y(focusMarkerCheckbox_, "Show Focus Point",
            "Overlay a marker at the current zoom focus");
    setA11y(processingStatusLabel_, "Processing Status",
            "Whether frames are processed on the CPU or the GPU");
    setA11y(simpleModeButton_, "Simple Mode",
            "Show the simple view with quick modes and large controls");
    setA11y(advancedModeButton_, "Advanced Mode",
            "Show the advanced view with every tuning control");
    setA11y(explainNowButton_, "Explain Now",
            "Describe the current camera view once using the AI vision model");
    setA11y(readTextButton_, "Read Text",
            "Read the text in the current camera view once using OCR");
    setA11y(stabilizationCheckbox_, "Stabilize Image",
            "Reduce hand shake and phone wobble in the camera image");
    setA11y(stabilizationStrengthSlider_, "Stabilization Strength",
            "How strongly the camera image is stabilized");
    setA11y(keystoneCheckbox_, "Straighten Screen (Keystone)",
            "Automatically straighten a projected screen viewed at an angle");
    setA11y(autoContrastCheckbox_, "Auto Contrast",
            "Automatically stretch washed-out colors for better readability");
    setA11y(autoContrastStrengthSlider_, "Auto Contrast Strength",
            "How strongly the automatic contrast correction is applied");
    setA11y(displayColorCombo_, "Display Colors",
            "Choose a high contrast color scheme such as white on black");
    setA11y(contrastSlider_, "Contrast",
            "Contrast of the displayed image");
    setA11y(brightnessSlider_, "Brightness",
            "Brightness of the displayed image");
    setA11y(aiSettingsButton_, "AI Settings",
            "Configure the AI vision server, OCR engine, and speech output");
    setA11y(openNotesButton_, "Open Notes",
            "Open the lecture notes file written by the assistive features");
    setA11y(assistantConnectionLabel_, "Codex Connection Status",
            "Current Codex app-server and ChatGPT account status");
    setA11y(assistantUsageLabel_, "Codex Usage",
            "Percentage remaining in the current Codex subscription usage window");
    setA11y(assistantConnectButton_, "Connect ChatGPT",
            "Sign in to Codex with a ChatGPT subscription");
    setA11y(assistantTranscript_, "Assistant Conversation",
            "Messages in the current OpenZoom assistant conversation");
    setA11y(assistantPromptEdit_, "Assistant Question",
            "Question for the OpenZoom vision assistant");
    setA11y(assistantAttachFrameCheckbox_, "Attach Current View",
            "Include the current processed camera frame with the question");
    setA11y(assistantSendButton_, "Send Question",
            "Send the question to the OpenZoom vision assistant");
    setA11y(assistantStopButton_, "Stop Assistant",
            "Stop the current assistant response");
    setA11y(assistantNewButton_, "New Conversation",
            "Start a new persistent OpenZoom assistant conversation");
    setA11y(assistantHistoryList_, "Assistant History",
            "OpenZoom assistant conversations saved by Codex");
    setA11y(assistantRenameButton_, "Rename Conversation",
            "Rename the selected assistant conversation");
    setA11y(assistantExportButton_, "Export Conversation",
            "Export the current assistant transcript to a text file");
    setA11y(assistantDeleteButton_, "Delete Conversation",
            "Permanently delete the selected assistant conversation");
    setA11y(modeGridButton_, "Show Quick Modes",
            "Open the grid of premade quick modes");
    setA11y(previousModeButton_, "Previous Quick Mode",
            "Apply the previous premade image setting");
    setA11y(currentModeButton_, "Current Quick Mode",
            "Show all premade quick modes");
    setA11y(nextModeButton_, "Next Quick Mode",
            "Apply the next premade image setting");

    // Tab order is local to each frameless overlay window. The application
    // event filter below bridges those groups while Simple mode is active.
    QWidget::setTabOrder(simpleModeButton_, advancedModeButton_);
    QWidget::setTabOrder(modeGridButton_, previousModeButton_);
    QWidget::setTabOrder(previousModeButton_, currentModeButton_);
    QWidget::setTabOrder(currentModeButton_, nextModeButton_);
    QWidget::setTabOrder(capturePhotoButton_, recordButton_);
    QWidget::setTabOrder(recordButton_, explainNowButton_);
    QWidget::setTabOrder(explainNowButton_, readTextButton_);
    QWidget::setTabOrder(cameraCombo_, rotationCombo_);

    setCentralWidget(central);
    setSimpleMode(true);
}

MainWindow::~MainWindow()
{
    if (qApp) {
        qApp->removeNativeEventFilter(this);
        qApp->removeEventFilter(this);
    }
}

std::array<QWidget*, 5> AssistiveOverlay::FocusTargets() const
{
    return {bodyView_, questionEdit_, askButton_, readAloudButton_, closeButton_};
}

void MainWindow::setApp(OpenZoomApp* app)
{
    app_ = app;
}

RenderWidget* MainWindow::renderWidget() const
{
    return renderWidget_;
}

QComboBox* MainWindow::cameraCombo() const { return cameraCombo_; }
QListWidget* MainWindow::presetList() const { return presetList_; }
QLabel* MainWindow::presetDescriptionLabel() const { return presetDescriptionLabel_; }
QPushButton* MainWindow::promotePresetButton() const { return promotePresetButton_; }
QCheckBox* MainWindow::blackWhiteCheckbox() const { return bwCheckbox_; }
QSlider* MainWindow::blackWhiteSlider() const { return bwSlider_; }
QCheckBox* MainWindow::zoomCheckbox() const { return zoomCheckbox_; }
QSlider* MainWindow::zoomSlider() const { return zoomSlider_; }
QPushButton* MainWindow::debugButton() const { return debugButton_; }
QComboBox* MainWindow::rotationCombo() const { return rotationCombo_; }
QCheckBox* MainWindow::focusMarkerCheckbox() const { return focusMarkerCheckbox_; }
QSlider* MainWindow::zoomCenterXSlider() const { return zoomCenterXSlider_; }
QSlider* MainWindow::zoomCenterYSlider() const { return zoomCenterYSlider_; }
QCheckBox* MainWindow::joystickCheckbox() const { return joystickCheckbox_; }
QToolButton* MainWindow::controlsToggleButton() const { return controlsToggleButton_; }
QWidget* MainWindow::controlsContainer() const { return controlsContainer_; }
QCheckBox* MainWindow::blurCheckbox() const { return blurCheckbox_; }
QSlider* MainWindow::blurSigmaSlider() const { return blurSigmaSlider_; }
QSlider* MainWindow::blurRadiusSlider() const { return blurRadiusSlider_; }
QLabel* MainWindow::blurSigmaValueLabel() const { return blurSigmaValueLabel_; }
QLabel* MainWindow::blurRadiusValueLabel() const { return blurRadiusValueLabel_; }
QListWidget* MainWindow::cameraModesList() const { return cameraModesList_; }
QPushButton* MainWindow::capturePhotoButton() const { return capturePhotoButton_; }
QPushButton* MainWindow::recordButton() const { return recordButton_; }
QCheckBox* MainWindow::temporalSmoothCheckbox() const { return temporalSmoothCheckbox_; }
QSlider* MainWindow::temporalSmoothSlider() const { return temporalSmoothSlider_; }
QLabel* MainWindow::temporalSmoothValueLabel() const { return temporalSmoothValueLabel_; }
QCheckBox* MainWindow::ocrAssistCheckbox() const { return ocrAssistCheckbox_; }
QCheckBox* MainWindow::vlmAssistCheckbox() const { return vlmAssistCheckbox_; }
QCheckBox* MainWindow::assistiveOverlayCheckbox() const { return assistiveOverlayCheckbox_; }
QCheckBox* MainWindow::spatialSharpenCheckbox() const { return spatialSharpenCheckbox_; }
QComboBox* MainWindow::spatialBackendCombo() const { return spatialBackendCombo_; }
QSlider* MainWindow::spatialSharpnessSlider() const { return spatialSharpnessSlider_; }
QLabel* MainWindow::spatialSharpnessValueLabel() const { return spatialSharpnessValueLabel_; }
QLabel* MainWindow::processingStatusLabel() const { return processingStatusLabel_; }
QAbstractButton* MainWindow::simpleModeButton() const { return simpleModeButton_; }
QAbstractButton* MainWindow::advancedModeButton() const { return advancedModeButton_; }
QPushButton* MainWindow::explainNowButton() const { return explainNowButton_; }
QPushButton* MainWindow::readTextButton() const { return readTextButton_; }
QCheckBox* MainWindow::stabilizationCheckbox() const { return stabilizationCheckbox_; }
QSlider* MainWindow::stabilizationStrengthSlider() const { return stabilizationStrengthSlider_; }
QCheckBox* MainWindow::keystoneCheckbox() const { return keystoneCheckbox_; }
QCheckBox* MainWindow::autoContrastCheckbox() const { return autoContrastCheckbox_; }
QSlider* MainWindow::autoContrastStrengthSlider() const { return autoContrastStrengthSlider_; }
QComboBox* MainWindow::displayColorCombo() const { return displayColorCombo_; }
QSlider* MainWindow::contrastSlider() const { return contrastSlider_; }
QSlider* MainWindow::brightnessSlider() const { return brightnessSlider_; }
QPushButton* MainWindow::aiSettingsButton() const { return aiSettingsButton_; }
QPushButton* MainWindow::openNotesButton() const { return openNotesButton_; }
QLabel* MainWindow::assistantConnectionLabel() const { return assistantConnectionLabel_; }
QLabel* MainWindow::assistantUsageLabel() const { return assistantUsageLabel_; }
QPushButton* MainWindow::assistantConnectButton() const { return assistantConnectButton_; }
QTextBrowser* MainWindow::assistantTranscript() const { return assistantTranscript_; }
QPlainTextEdit* MainWindow::assistantPromptEdit() const { return assistantPromptEdit_; }
QCheckBox* MainWindow::assistantAttachFrameCheckbox() const { return assistantAttachFrameCheckbox_; }
QPushButton* MainWindow::assistantSendButton() const { return assistantSendButton_; }
QPushButton* MainWindow::assistantStopButton() const { return assistantStopButton_; }
QPushButton* MainWindow::assistantNewButton() const { return assistantNewButton_; }
QListWidget* MainWindow::assistantHistoryList() const { return assistantHistoryList_; }
QPushButton* MainWindow::assistantRenameButton() const { return assistantRenameButton_; }
QPushButton* MainWindow::assistantExportButton() const { return assistantExportButton_; }
QPushButton* MainWindow::assistantDeleteButton() const { return assistantDeleteButton_; }

void MainWindow::ActivatePresetRow(int row)
{
    if (!presetList_ || row < 0 || row >= presetList_->count()) {
        return;
    }

    QListWidgetItem* item = presetList_->item(row);
    if (presetList_->currentRow() == row && item) {
        const QString label = item->data(Qt::AccessibleTextRole).toString();
        ShowModeAnnouncement(label, item->data(Qt::StatusTipRole).toString());
    } else {
        presetList_->setCurrentRow(row);
    }
    presetList_->scrollToItem(item, QAbstractItemView::PositionAtCenter);
    modeGridPopup_->hide();
    RevealSimpleChrome();
}

void MainWindow::ActivateRelativePreset(int offset)
{
    if (!presetList_ || presetList_->count() == 0) {
        return;
    }
    const int count = presetList_->count();
    int row = presetList_->currentRow();
    if (row < 0) {
        row = 0;
    } else {
        row = (row + offset + count) % count;
    }
    ActivatePresetRow(row);
}

void MainWindow::ToggleModeGrid()
{
    if (!modeGridPopup_ || !isSimpleMode()) {
        return;
    }
    const bool show = !modeGridPopup_->isVisible();
    if (show) {
        RevealSimpleChrome();
        simpleChromeIdleTimer_->stop();
        UpdateSimpleChromeGeometry();
        modeGridPopup_->show();
        modeGridPopup_->raise();
        presetList_->setFocus(Qt::ShortcutFocusReason);
        if (QListWidgetItem* current = presetList_->currentItem()) {
            presetList_->scrollToItem(current, QAbstractItemView::PositionAtCenter);
        }
    } else {
        modeGridPopup_->hide();
        RevealSimpleChrome();
        if (currentModeButton_) {
            currentModeButton_->setFocus(Qt::PopupFocusReason);
        }
    }
}

void MainWindow::UpdateCurrentPresetUi(QListWidgetItem* current, QListWidgetItem* previous)
{
    if (!currentModeButton_) {
        return;
    }

    if (!current) {
        currentModeButton_->setText("Custom Setup");
        currentModeButton_->setAccessibleName("Current quick mode: Custom Setup");
        return;
    }

    QString label = current->data(Qt::AccessibleTextRole).toString();
    if (label.isEmpty()) {
        label = PlainPresetLabel(current->data(Qt::StatusTipRole).toString());
    }
    const int row = presetList_->row(current);
    const QString shortcut = row >= 0 && row < kSimpleShortcutCount
                                 ? QStringLiteral("   [%1]").arg(row + 1)
                                 : QString();
    currentModeButton_->setText(label + shortcut);
    currentModeButton_->setAccessibleName(QStringLiteral("Current quick mode: %1").arg(label));

    if (previous && previous != current && isSimpleMode()) {
        ShowModeAnnouncement(label, current->data(Qt::StatusTipRole).toString());
    }
}

void MainWindow::ShowModeAnnouncement(const QString& label, const QString& profileName)
{
    if (label.trimmed().isEmpty() || !modeToast_) {
        return;
    }

    modeToastTitle_->setText(label.toUpper());
    modeToastSubtitle_->setText(profileName.trimmed().isEmpty()
                                    ? QStringLiteral("Quick mode")
                                    : QStringLiteral("%1 profile").arg(profileName));
    modeToast_->setAccessibleName(QStringLiteral("Quick mode changed to %1").arg(label));
    UpdateSimpleChromeGeometry();
    modeToast_->show();
    modeToast_->raise();
    modeToastTimer_->start();

    QAccessibleAnnouncementEvent accessibleAnnouncement(modeToast_, label);
    accessibleAnnouncement.setPoliteness(QAccessible::AnnouncementPoliteness::Assertive);
    QAccessible::updateAccessibility(&accessibleAnnouncement);
}

void MainWindow::UpdateSimpleChromeGeometry()
{
    if (!renderWidget_ || !topLeftPanel_ || renderWidget_->width() <= 0 || renderWidget_->height() <= 0) {
        return;
    }

    const int viewWidth = renderWidget_->width();
    const int viewHeight = renderWidget_->height();
    const QPoint viewOrigin = renderWidget_->mapToGlobal(QPoint(0, 0));
    for (QWidget* panel : {topLeftPanel_, bottomLeftPanel_, bottomRightPanel_}) {
        panel->adjustSize();
    }

    topLeftPanel_->move(viewOrigin);
    bottomLeftPanel_->move(viewOrigin.x(),
                           viewOrigin.y() + std::max(0, viewHeight - bottomLeftPanel_->height()));
    const bool bottomPanelsOverlap = bottomLeftPanel_->width() + bottomRightPanel_->width() > viewWidth;
    const int bottomRightY = bottomPanelsOverlap
                                 ? viewHeight - bottomLeftPanel_->height() - bottomRightPanel_->height()
                                 : viewHeight - bottomRightPanel_->height();
    bottomRightPanel_->move(viewOrigin.x() + std::max(0, viewWidth - bottomRightPanel_->width()),
                            viewOrigin.y() + std::max(0, bottomRightY));

    if (modeGridPopup_) {
        const int popupWidth = std::min(viewWidth, std::max(420, std::min(860, viewWidth * 3 / 4)));
        const int columns = std::max(2, (popupWidth - 28) / 220);
        const int rows = std::max(1, (presetList_->count() + columns - 1) / columns);
        const int popupHeight = std::min(std::max(126, rows * 106 + 16),
                                        std::max(126, viewHeight - bottomLeftPanel_->height()));
        const QSize gridSize(std::max(150, (popupWidth - 32) / columns), 102);
        presetList_->setGridSize(gridSize);
        for (int row = 0; row < presetList_->count(); ++row) {
            if (QListWidgetItem* item = presetList_->item(row)) {
                item->setSizeHint(gridSize - QSize(8, 8));
            }
        }
        modeGridPopup_->setGeometry(viewOrigin.x(),
                                    viewOrigin.y() + std::max(0, viewHeight - bottomLeftPanel_->height() - popupHeight),
                                    popupWidth,
                                    popupHeight);
    }

    if (modeToast_) {
        const int toastWidth = std::min(760, std::max(300, viewWidth - 32));
        const int toastHeight = std::min(170, std::max(112, viewHeight / 4));
        modeToast_->setGeometry(viewOrigin.x() + std::max(0, (viewWidth - toastWidth) / 2),
                                viewOrigin.y() + std::max(0, (viewHeight - toastHeight) / 2),
                                toastWidth,
                                toastHeight);
    }
}

void MainWindow::SetChromeOpacity(qreal opacity, int durationMs, bool hideWhenFinished)
{
    if (chromeAnimation_) {
        chromeAnimation_->stop();
        chromeAnimation_->deleteLater();
        chromeAnimation_ = nullptr;
    }

    chromeAnimation_ = new QParallelAnimationGroup(this);
    for (QWidget* panel : {topLeftPanel_, bottomLeftPanel_, bottomRightPanel_}) {
        if (!panel) {
            continue;
        }
        if (opacity > 0.0) {
            panel->show();
            panel->raise();
        }
        auto* animation = new QPropertyAnimation(panel, "windowOpacity", chromeAnimation_);
        animation->setDuration(durationMs);
        animation->setStartValue(panel->windowOpacity());
        animation->setEndValue(opacity);
        chromeAnimation_->addAnimation(animation);
    }

    QParallelAnimationGroup* animationGroup = chromeAnimation_;
    connect(animationGroup, &QParallelAnimationGroup::finished, this,
            [this, animationGroup, hideWhenFinished]() {
                if (hideWhenFinished && !simpleChromeVisible_ && isSimpleMode()) {
                    for (QWidget* panel : {topLeftPanel_, bottomLeftPanel_, bottomRightPanel_}) {
                        if (panel) {
                            panel->hide();
                        }
                    }
                }
                if (chromeAnimation_ == animationGroup) {
                    chromeAnimation_ = nullptr;
                }
                animationGroup->deleteLater();
            });
    animationGroup->start();
}

bool MainWindow::SimpleChromeHasFocus() const
{
    QWidget* focus = QApplication::focusWidget();
    if (!focus) {
        return false;
    }
    for (QWidget* panel : {topLeftPanel_, bottomLeftPanel_, bottomRightPanel_, modeGridPopup_}) {
        if (panel && (focus == panel || panel->isAncestorOf(focus))) {
            return true;
        }
    }
    return false;
}

void MainWindow::RevealSimpleChrome()
{
    if (!isSimpleMode()) {
        return;
    }
    const bool needsAnimation = !simpleChromeVisible_ ||
                                (topLeftPanel_ && !topLeftPanel_->isVisible());
    simpleChromeVisible_ = true;
    UpdateSimpleChromeGeometry();
    if (needsAnimation) {
        SetChromeOpacity(1.0, 120, false);
    } else {
        for (QWidget* panel : {topLeftPanel_, bottomLeftPanel_, bottomRightPanel_}) {
            if (panel) {
                panel->show();
                panel->raise();
            }
        }
    }
    if (!chromePinned_ && !modeGridPopup_->isVisible()) {
        simpleChromeIdleTimer_->start();
    }
}

void MainWindow::FadeSimpleChrome()
{
    if (!isSimpleMode() || chromePinned_) {
        return;
    }
    if (modeGridPopup_->isVisible() || SimpleChromeHasFocus()) {
        simpleChromeIdleTimer_->start();
        return;
    }
    simpleChromeVisible_ = false;
    SetChromeOpacity(0.0, kSimpleChromeFadeMs, true);
}

void MainWindow::setSimpleMode(bool simple)
{
    if (advancedPanel_) {
        advancedPanel_->setVisible(!simple);
    }
    // Checking one button in the exclusive group unchecks the other; the
    // toggled handlers re-enter here but converge because setChecked() and
    // setCurrentIndex() are no-ops once the state matches.
    if (simple && simpleModeButton_) {
        simpleModeButton_->setChecked(true);
        bottomLeftPanel_->show();
        bottomRightPanel_->show();
        UpdateSimpleChromeGeometry();
        RevealSimpleChrome();
    } else if (!simple && advancedModeButton_) {
        advancedModeButton_->setChecked(true);
        simpleChromeIdleTimer_->stop();
        modeToastTimer_->stop();
        modeGridPopup_->hide();
        modeToast_->hide();
        simpleChromeVisible_ = true;
        SetChromeOpacity(1.0, 0, false);
        topLeftPanel_->show();
        bottomLeftPanel_->hide();
        bottomRightPanel_->hide();
        UpdateSimpleChromeGeometry();
    }
}

bool MainWindow::isSimpleMode() const
{
    return !advancedPanel_ || !advancedPanel_->isVisible();
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    if (app_ && app_->HandlePanKey(event->key(), true)) {
        event->accept();
        return;
    }
    QMainWindow::keyPressEvent(event);
}

void MainWindow::keyReleaseEvent(QKeyEvent* event)
{
    if (app_ && app_->HandlePanKey(event->key(), false)) {
        event->accept();
        return;
    }
    QMainWindow::keyReleaseEvent(event);
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event)
{
    const QEvent::Type type = event->type();
    const bool userActivity = type == QEvent::MouseMove ||
                              type == QEvent::MouseButtonPress ||
                              type == QEvent::Wheel ||
                              type == QEvent::KeyPress ||
                              type == QEvent::FocusIn ||
                              type == QEvent::Enter ||
                              type == QEvent::TouchBegin ||
                              type == QEvent::ApplicationActivate;
    if (isSimpleMode() && userActivity) {
        RevealSimpleChrome();
    }

    if (type == QEvent::ApplicationDeactivate) {
        // The chrome panels, grid popup, and toast are top-level tool windows,
        // which stay above OTHER applications' windows. Hide them when the user
        // switches away (e.g. to take notes); reactivation reveals them again
        // via the ApplicationActivate branch above.
        simpleChromeIdleTimer_->stop();
        if (modeToastTimer_) {
            modeToastTimer_->stop();
        }
        if (modeToast_) {
            modeToast_->hide();
        }
        if (modeGridPopup_) {
            modeGridPopup_->hide();
        }
        for (QWidget* panel : {topLeftPanel_, bottomLeftPanel_, bottomRightPanel_}) {
            if (panel) {
                panel->hide();
            }
        }
        simpleChromeVisible_ = false;
    }

    if (isSimpleMode() && type == QEvent::KeyPress) {
        auto* key = static_cast<QKeyEvent*>(event);
        if (key->key() == Qt::Key_Escape && modeGridPopup_->isVisible()) {
            modeGridPopup_->hide();
            RevealSimpleChrome();
            if (currentModeButton_) {
                currentModeButton_->setFocus(Qt::PopupFocusReason);
            }
            event->accept();
            return true;
        }
        if (key->key() == Qt::Key_H && key->modifiers() == Qt::ControlModifier) {
            chromePinned_ = !chromePinned_;
            RevealSimpleChrome();
            if (chromePinned_) {
                simpleChromeIdleTimer_->stop();
            }
            QAccessibleAnnouncementEvent announcement(
                this,
                chromePinned_ ? QStringLiteral("Controls pinned")
                              : QStringLiteral("Controls will hide automatically"));
            announcement.setPoliteness(QAccessible::AnnouncementPoliteness::Assertive);
            QAccessible::updateAccessibility(&announcement);
            event->accept();
            return true;
        }
        if (key->key() == Qt::Key_Tab &&
            !(key->modifiers() & (Qt::ControlModifier | Qt::AltModifier)) &&
            !modeGridPopup_->isVisible()) {
            std::vector<QWidget*> focusOrder{
                simpleModeButton_, advancedModeButton_, modeGridButton_,
                previousModeButton_, currentModeButton_, nextModeButton_,
                capturePhotoButton_, recordButton_, explainNowButton_, readTextButton_};
            if (auto* overlay = renderWidget_->findChild<AssistiveOverlay*>();
                overlay && overlay->isVisible()) {
                const auto overlayTargets = overlay->FocusTargets();
                focusOrder.insert(focusOrder.end(), overlayTargets.begin(), overlayTargets.end());
            }
            QWidget* current = QApplication::focusWidget();
            int currentIndex = -1;
            for (int i = 0; i < static_cast<int>(focusOrder.size()); ++i) {
                if (focusOrder[i] == current) {
                    currentIndex = i;
                    break;
                }
            }
            const int direction = key->modifiers() & Qt::ShiftModifier ? -1 : 1;
            if (currentIndex < 0) {
                currentIndex = direction > 0 ? -1 : 0;
            }
            for (int step = 1; step <= static_cast<int>(focusOrder.size()); ++step) {
                const int candidate = (currentIndex + direction * step +
                                       static_cast<int>(focusOrder.size())) %
                                      static_cast<int>(focusOrder.size());
                QWidget* target = focusOrder[candidate];
                if (target && target->isVisible() && target->isEnabled()) {
                    target->setFocus(direction > 0 ? Qt::TabFocusReason
                                                   : Qt::BacktabFocusReason);
                    event->accept();
                    return true;
                }
            }
        }
        const bool plainKey = key->modifiers() == Qt::NoModifier || key->modifiers() == Qt::KeypadModifier;
        if (plainKey && key->key() >= Qt::Key_1 && key->key() <= Qt::Key_9) {
            ActivatePresetRow(key->key() - Qt::Key_1);
            event->accept();
            return true;
        }
    }

    if (watched == this && (type == QEvent::Move ||
                            type == QEvent::Resize ||
                            type == QEvent::WindowStateChange)) {
        QTimer::singleShot(0, this, &MainWindow::UpdateSimpleChromeGeometry);
    }

    if (watched == renderWidget_) {
        switch (type) {
        case QEvent::Resize:
            UpdateSimpleChromeGeometry();
            break;
        case QEvent::Wheel: {
            auto* wheel = static_cast<QWheelEvent*>(event);
            if (wheel->modifiers() & Qt::ControlModifier) {
                if (app_) {
                    app_->HandleZoomWheel(wheel->angleDelta().y(), wheel->position());
                }
                event->accept();
                return true;
            } else if (app_ && app_->HandlePanScroll(wheel)) {
                event->accept();
                return true;
            }
            break;
        }
        case QEvent::MouseButtonPress: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (modeGridPopup_ && modeGridPopup_->isVisible()) {
                modeGridPopup_->hide();
                RevealSimpleChrome();
            }
            if (mouse->button() == Qt::MiddleButton) {
                if (app_) {
                    app_->BeginMousePan(mouse->position(), renderWidget_->size());
                }
                event->accept();
                return true;
            }
            break;
        }
        case QEvent::MouseMove: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (app_ && app_->UpdateMousePan(mouse->position())) {
                event->accept();
                return true;
            }
            break;
        }
        case QEvent::MouseButtonRelease: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::MiddleButton) {
                if (app_) {
                    app_->EndMousePan();
                }
                event->accept();
                return true;
            }
            break;
        }
        default:
            break;
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

bool MainWindow::nativeEventFilter(const QByteArray&, void* message, qintptr*)
{
    const auto* nativeMessage = static_cast<const MSG*>(message);
    if (!nativeMessage || !isSimpleMode()) {
        return false;
    }

    switch (nativeMessage->message) {
    case WM_MOUSEMOVE:
    case WM_LBUTTONDOWN:
    case WM_MBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MOUSEWHEEL:
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
    case WM_SETFOCUS:
        RevealSimpleChrome();
        break;
    default:
        break;
    }
    return false;
}

} // namespace openzoom

#endif // _WIN32
