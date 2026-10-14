# -*- coding: utf-8 -*-
"""浅色置顶回复助手：回复建议、会话级自动回复和独立设置页。"""
import os
import sys
import threading
from datetime import datetime
from math import isfinite
from types import SimpleNamespace

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSizeGrip, QSizePolicy,
    QStackedWidget, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel, CardWidget, CheckBox, ComboBox, EditableComboBox, FluentIcon as FIF,
    HyperlinkButton, IndeterminateProgressBar, LineEdit, PasswordLineEdit, PlainTextEdit,
    PrimaryPushButton, PushButton, ScrollArea, SpinBox, SwitchButton, Theme, TransparentToolButton,
    setCustomStyleSheet, setFont, setTheme, setThemeColor,
)

from app import settings
from app.version import VERSION
from core import jev_client, llm, providers, vision
from core.questions import CHOICE_LABELS, JUDGE_QUESTIONS, build_state
from core.llm_judge import validate_answers
from core.diagnostics import log_event, redact_text, register_secret

_LOG_LINES = 300
_MUTED = "#68776f"
_GREEN = "#18794e"
_RELATIONSHIPS = [
    ("恋人", "romantic partners"), ("朋友", "friends"), ("同事", "colleagues"),
    ("家人", "family"), ("自定义", None),
]
_JUDGE_MODES = [
    ("Jev 优先，失败后通用模型兜底", "jev_fallback"),
    ("仅使用 Jev", "jev_only"),
    ("仅使用通用模型", "llm_only"),
]


_MEDIA_MODES = [("仅本地 OCR，不上传图片", "ocr"), ("视觉模型识别，失败回退 OCR", "vision"),
                ("OCR + 视觉模型互补", "hybrid")]
_LOG_LEVEL_LABELS = [("调试（DEBUG）", "DEBUG"), ("信息（INFO）", "INFO"),
                     ("警告（WARNING）", "WARNING"), ("错误（ERROR）", "ERROR"),
                     ("严重（CRITICAL）", "CRITICAL")]


def _choice(answers, name):
    return CHOICE_LABELS[name].get((answers.get(name) or {}).get("choice"), "暂未判断")


def _mp_banner_path() -> str:
    """打包后在 _MEIPASS/docs，源码跑在仓库 docs/。"""
    root = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "docs", "wechat-mp.png")


class _MpBanner(QLabel):
    """公众号长条横幅，宽度跟着设置页走，高度按原图比例。"""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._src = QPixmap(path)
        self._shown = 0
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        if self._src.isNull() or w <= 0 or self._src.width() <= 0:
            return 0
        return max(1, round(w * self._src.height() / self._src.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        if w <= 0 or w == self._shown or self._src.isNull():
            return
        h = self.heightForWidth(w)
        self._shown = w
        self.setPixmap(self._src.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.height() != h:
            self.setFixedHeight(h)


class _FitCombo(ComboBox):
    """长名字不撑开窄布局。按钮上按当前宽度省略；条目仍是全文，findText 靠它。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def setText(self, text):
        self._full = text or ""
        QPushButton.setText(self, self._elide(self._full))
        if self._full and self.text() != self._full:
            self.setToolTip(self._full)

    def minimumSizeHint(self):
        hint = QPushButton.minimumSizeHint(self)
        return QSize(48, hint.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        shown = self._elide(self._full)
        if shown != self.text():
            QPushButton.setText(self, shown)
        if self._full and shown != self._full:
            self.setToolTip(self._full)

    def _elide(self, text):
        # 右侧箭头大约 28px。还没排上版时先按一个窄宽度省略，避免最小宽度被整句名字撑开。
        avail = self.width() - 36 if self.width() > 64 else 120
        return self.fontMetrics().elidedText(text, Qt.ElideRight, max(24, avail))


def _label(text="", size=14, color=None, bold=False, parent=None):
    label = BodyLabel(text, parent)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    setFont(label, size, QFont.DemiBold if bold else QFont.Normal)
    if color:
        qss = f"BodyLabel {{ color: {color}; background: transparent; }}"
        setCustomStyleSheet(label, qss, qss)
    return label


def _tool(icon, title, callback, parent=None):
    button = TransparentToolButton(icon, parent)
    button.setFixedSize(32, 32)
    button.setToolTip(title)
    button.setAccessibleName(title)
    button.clicked.connect(callback)
    return button


class _Surface(CardWidget):
    def __init__(self, parent=None, accent=False):
        self.accent = accent
        super().__init__(parent)
        self.setBorderRadius(12)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def _normalBackgroundColor(self):
        return QColor("#edf7f0" if self.accent else "#ffffff")

    def _hoverBackgroundColor(self):
        return self._normalBackgroundColor()

    def _pressedBackgroundColor(self):
        return self._normalBackgroundColor()


class _Fetched(QObject):
    """取模型列表的后台线程 → 主线程：哪一组（SimpleNamespace）、取回来的模型 id、失败原因（成功是空串）。
    Qt 不让跨线程碰控件，信号是跨线程唯一干净的路。"""
    done = Signal(object, list, str)
    tested = Signal(str)
    jevTested = Signal(object, str)


class _TitleBar(QWidget):
    """只有标题栏可拖动，选择正文或按按钮不会意外移动窗口。"""
    def __init__(self, parent):
        super().__init__(parent)
        self._drag = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.window().pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is not None and event.buttons() & Qt.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag = None
        super().mouseReleaseEvent(event)


class _MainWindow(QWidget):
    """窗口大小变了就叫 Overlay 重新排布；断点没跨过时 _relayout 自己不做事，这里不用防抖。"""
    def __init__(self, relayout):
        super().__init__()
        self._relayout = relayout

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout(event.size().width(), event.size().height())


class _ReplyCard(_Surface):
    def __init__(self, owner, index, recommended=False, number=1, score=None):
        super().__init__(accent=recommended)
        box = QVBoxLayout(self)
        self.box = box
        box.setSpacing(10)
        top = QHBoxLayout()
        label = "推荐回复" if recommended else f"备选 {number}"
        if score is not None:
            label += f" · {round(score * 100)}%"
        top.addWidget(_label(label, 12, _GREEN if recommended else _MUTED, True))
        self.copyButton = _tool(FIF.COPY, "复制这条回复", lambda: owner._copy(index), self)
        self.copyButton.setFixedSize(24, 24)
        top.addWidget(self.copyButton)
        box.addLayout(top)
        self.text = _label(owner.cands[index], 15)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(self.text)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.fillButton = (PrimaryPushButton if recommended else PushButton)("填入", self)
        self.fillButton.setAccessibleName(f"填入{'推荐回复' if recommended else f'备选 {number}'}")
        self.fillButton.clicked.connect(lambda: owner._fill(index))
        bottom.addWidget(self.fillButton)
        box.addLayout(bottom)
        self.set_compact(owner._compact)

    def set_available(self, enabled):
        self.fillButton.setEnabled(enabled)
        self.copyButton.setEnabled(enabled)

    def set_compact(self, compact):
        self.box.setContentsMargins(12, 8, 12, 8) if compact else self.box.setContentsMargins(16, 12, 16, 12)
        self.fillButton.setMinimumWidth(80 if compact else 100)


class Overlay:
    def __init__(self, on_fill, on_toggle_capture=None, on_target_change=None, result_of=None,
                 on_toggle_debug=None, app_key_of=None, on_processing_change=None, on_logging_change=None):
        """result_of(会话名) → 那个会话上次的结果或 None；切着看别的会话时用它把旧结果放回来。
        on_target_change(会话名, 人名) → 用户在群里挑了回复对象。
        on_toggle_debug(开不开) → 开关调试视图那个独立窗口。"""
        self.app = QApplication.instance() or QApplication([])
        setTheme(Theme.LIGHT)
        setThemeColor(_GREEN, save=False)
        self.on_fill = on_fill
        self.on_toggle_capture = on_toggle_capture
        self.on_target_change = on_target_change
        self.on_toggle_debug = on_toggle_debug
        self.app_key_of = app_key_of or (lambda: "wechat")
        self.result_of = result_of
        self.on_processing_change = on_processing_change
        self.on_logging_change = on_logging_change
        self.read_states = {}
        self.cands = []
        self.cards = []
        self._busy = False
        self._current = False
        self._no_reply_reason = ""
        self._compact = None  # 断点模式：None 保证 _relayout 第一次调用必定生效
        self._pageLayouts = []
        self._hintLabels = []
        self.feeds = {}  # {会话名: [排好版的记录]}
        self.counts = {}  # {会话名: 消息条数}
        self.hers = {}  # {会话名: 对方最近一句}
        self.targets = {}  # {会话名: ([发言人], 当前回复对象)}
        self._chat = ""  # 微信当前开着的会话
        self._shown = ""  # 界面上正在看的会话（浏览时和上面不一样）
        self.win = _MainWindow(self._relayout)
        self.win.setObjectName("assistantWindow")
        self.win.setWindowTitle("Local AI Chat")
        self.win.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.win.setStyleSheet(
            "QWidget#assistantWindow { background: #f5f7f6; border: 1px solid #dce3de; border-radius: 14px; }"
        )
        self.win.setMinimumWidth(320)
        self.win.setMaximumWidth(640)
        outer = QVBoxLayout(self.win)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        header = _TitleBar(self.win)
        title = QHBoxLayout(header)
        title.setContentsMargins(18, 12, 10, 10)
        title.setSpacing(8)
        name = _label("AI", 20, "#233c2f", True)
        name.setFixedWidth(40)
        name.setAttribute(Qt.WA_TransparentForMouseEvents)
        title.addWidget(name)
        self.subtitle = _label("Local AI Chat", 12, _MUTED)
        self.subtitle.setAttribute(Qt.WA_TransparentForMouseEvents)
        title.addWidget(self.subtitle, 1)
        self.captureSwitch = SwitchButton(header)
        self.captureSwitch.setOnText("采集中")
        self.captureSwitch.setOffText("已暂停")
        self.captureSwitch.setToolTip("开启或暂停消息读取与本地 SQLite 保存，与生成建议和自动发送独立")
        self.captureSwitch.setAccessibleName("开启或暂停采集")
        self.captureSwitch.setChecked(True)
        self.captureSwitch.checkedChanged.connect(self._capture_toggled)
        title.addWidget(self.captureSwitch)
        self.settingsButton = _tool(FIF.SETTING, "设置", self.open_settings, header)
        title.addWidget(self.settingsButton)
        title.addWidget(_tool(FIF.REMOVE, "最小化", self.win.showMinimized, header))
        title.addWidget(_tool(FIF.CLOSE, "关闭助手", self.win.close, header))
        outer.addWidget(header)
        self.updateBar = QWidget(self.win)
        update_row = QHBoxLayout(self.updateBar)
        update_row.setContentsMargins(18, 4, 8, 4)
        update_row.setSpacing(8)
        self.updateLabel = _label("", 12, _GREEN, True)
        update_row.addWidget(self.updateLabel, 1)
        self.updateLink = HyperlinkButton("", "去下载", self.updateBar)
        self.updateLink.setFixedHeight(24)
        update_row.addWidget(self.updateLink)
        closeUpdate = TransparentToolButton(FIF.CLOSE, self.updateBar)
        closeUpdate.setFixedSize(20, 20)
        closeUpdate.setToolTip("关闭更新提示")
        closeUpdate.setAccessibleName("关闭更新提示")
        closeUpdate.clicked.connect(lambda: self.updateBar.hide())
        update_row.addWidget(closeUpdate)
        self.updateBar.setFixedHeight(32)
        self.updateBar.hide()
        outer.addWidget(self.updateBar)
        self.pages = QStackedWidget(self.win)
        outer.addWidget(self.pages, 1)
        self._build_home()
        self._build_settings()
        footer = QHBoxLayout()
        footer.setContentsMargins(20, 9, 8, 8)
        footer.addWidget(_label(f"手动填入不发送 · 自动回复按会话开启 · v{VERSION}", 11, _MUTED), 1)
        grip = QSizeGrip(self.win)
        grip.setFixedSize(16, 16)
        footer.addWidget(grip, 0, Qt.AlignBottom)
        outer.addLayout(footer)
        screen = self.app.primaryScreen().availableGeometry()
        self.win.setMinimumHeight(min(360, screen.height() - 32))
        self.win.resize(min(440, screen.width() - 32), min(820, screen.height() - 48))
        self.win.move(screen.right() - self.win.width() - 20, screen.top() + 24)
        self._relayout(self.win.width(), self.win.height())  # resizeEvent 补不到构造时这一次
        self.set_status("仅采集模式：消息保存到本地 SQLite" if not settings.analysis_enabled()
                        else "等待新消息" if settings.has_key() else "需要配置模型",
                        "idle" if not settings.analysis_enabled() or settings.has_key() else "warning")
        self.win.show()

    def _scroll_page(self):
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.viewport().setAutoFillBackground(False)
        content = QWidget()
        content.setObjectName("pageContent")
        content.setStyleSheet("QWidget#pageContent { background: transparent; }")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 8, 20, 12)
        layout.setSpacing(14)
        scroll.setWidget(content)
        self.pages.addWidget(scroll)
        self._pageLayouts.append(layout)
        return scroll, layout

    def _relayout(self, w, h):
        """宽度跨过断点才重新摆布局（省事）；高度每次都重算，反正只是设个定高。"""
        compact = w < 400
        if compact != self._compact:
            self._compact = compact
            self._apply_compact(compact)
        self.feed.setFixedHeight(max(100, min(240, int(h * 0.25))))

    def _apply_compact(self, compact):
        """紧凑/常规两套间距和可见性；断点没变时不会被调用。"""
        self.subtitle.setVisible(not compact)
        self.captureSwitch.setOnText("" if compact else "采集中")
        self.captureSwitch.setOffText("" if compact else "已暂停")
        for label in self._hintLabels:
            label.setVisible(not compact)
        self.referenceNote.setVisible(bool(self.cands) and not compact)
        self._sync_model_fields()
        margins = (12, 8, 12, 12) if compact else (20, 8, 20, 12)
        for layout in self._pageLayouts:
            layout.setContentsMargins(*margins)
        for card in self.cards:
            card.set_compact(compact)

    def _build_home(self):
        self.home, body = self._scroll_page()
        heading = QHBoxLayout()
        heading.addWidget(_label("回复建议", 23, "#24382d", True), 1)
        self.updated = _label("", 11, _MUTED)
        self.updated.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        heading.addWidget(self.updated)
        body.addLayout(heading)
        chat_row = QHBoxLayout()
        chat_row.setSpacing(8)
        prefix = _label("当前会话", 12, _MUTED)
        prefix.setFixedWidth(56)
        chat_row.addWidget(prefix)
        self.chatBox = _FitCombo()
        self.chatBox.setPlaceholderText("尚未识别到会话")
        self.chatBox.setAccessibleName("当前会话")
        self.chatBox.setToolTip("聊天窗口切到哪个会话这里就跟到哪个；也可以自己选一个，只看它的记录和建议")
        self.chatBox.currentIndexChanged.connect(self._on_chat_selected)
        chat_row.addWidget(self.chatBox, 1)
        self.chatFollow = _label("", 11, _MUTED)
        self.chatFollow.setFixedWidth(52)
        self.chatFollow.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        chat_row.addWidget(self.chatFollow)
        body.addLayout(chat_row)
        processing_row = QHBoxLayout()
        processing_row.addWidget(_label("生成建议", 12, _MUTED))
        self.analysisSwitch = SwitchButton()
        self.analysisSwitch.setOnText("开")
        self.analysisSwitch.setOffText("关")
        self.analysisSwitch.setAccessibleName("生成回复建议")
        self.analysisSwitch.setToolTip("关闭后只在本地读取和保存消息，不调用模型")
        processing_row.addWidget(self.analysisSwitch)
        processing_row.addStretch(1)
        processing_row.addWidget(_label("允许发送", 12, _MUTED))
        self.sendSwitch = SwitchButton()
        self.sendSwitch.setOnText("开")
        self.sendSwitch.setOffText("关")
        self.sendSwitch.setAccessibleName("允许自动发送")
        self.sendSwitch.setToolTip("全局发送开关；仍需当前会话明确开启自动回复，手动填入不受影响")
        processing_row.addWidget(self.sendSwitch)
        body.addLayout(processing_row)
        self._render_processing()
        self.analysisSwitch.checkedChanged.connect(self._processing_toggled)
        self.sendSwitch.checkedChanged.connect(self._processing_toggled)
        self.readState = _label("消息读取后自动保存到本地 SQLite", 11, _MUTED)
        self.readState.setWordWrap(True)
        self.readState.setToolTip(settings.message_db_path())
        body.addWidget(self.readState)
        auto_row = QHBoxLayout()
        auto_row.setSpacing(8)
        auto_prefix = _label("自动回复", 12, _MUTED)
        auto_prefix.setFixedWidth(56)
        auto_row.addWidget(auto_prefix)
        self.autoKeyBox = ComboBox()
        self.autoKeyBox.addItems(["Enter 发送", "Ctrl+Enter 发送"])
        self.autoKeyBox.setAccessibleName("自动回复发送快捷键")
        self.autoKeyBox.setToolTip("必须和微信中的发送快捷键设置一致；选出最佳回复后会自动发送")
        self.autoKeyBox.currentIndexChanged.connect(self._auto_key_changed)
        auto_row.addWidget(self.autoKeyBox, 1)
        self.autoReplySwitch = SwitchButton()
        self.autoReplySwitch.setOnText("开")
        self.autoReplySwitch.setOffText("关")
        self.autoReplySwitch.setAccessibleName("当前会话自动回复")
        self.autoReplySwitch.setToolTip("只对当前显示的会话生效；开启后最佳回复会覆盖输入框并直接发送")
        self.autoReplySwitch.checkedChanged.connect(self._auto_reply_toggled)
        auto_row.addWidget(self.autoReplySwitch)
        body.addLayout(auto_row)
        self._render_auto_reply()
        self.targetRow = QWidget()  # 只有开了「群聊指定回复对象」且这个会话是群聊才露出来
        target_row = QHBoxLayout(self.targetRow)
        target_row.setContentsMargins(0, 0, 0, 0)
        target_row.setSpacing(8)
        target_prefix = _label("回复对象", 12, _MUTED)
        target_prefix.setFixedWidth(56)
        target_row.addWidget(target_prefix)
        self.targetBox = _FitCombo()
        self.targetBox.setAccessibleName("回复对象")
        self.targetBox.setToolTip("三条候选都按这个人来写；不选就跟着最近说话的那位")
        self.targetBox.currentIndexChanged.connect(self._on_target_selected)
        target_row.addWidget(self.targetBox, 1)
        self.atCheck = CheckBox("填入时带 @")
        self.atCheck.setChecked(True)
        self.atCheck.setToolTip("填入时在开头加「@名字 」。只是普通文字，不会变成真正的 @")
        target_row.addWidget(self.atCheck)
        self.targetRow.hide()
        body.addWidget(self.targetRow)
        self.status = _label("", 12, _MUTED)
        body.addWidget(self.status)
        self.progress = IndeterminateProgressBar()
        self.progress.setFixedHeight(3)
        self.progress.hide()
        body.addWidget(self.progress)
        self.context = QWidget()
        context_box = QVBoxLayout(self.context)
        context_box.setContentsMargins(0, 0, 0, 0)
        context_box.setSpacing(5)
        context_box.addWidget(_label("对方最近说", 11, _MUTED))
        self.latest = _label("", 14, "#42574a")
        self.latest.setTextInteractionFlags(Qt.TextSelectableByMouse)
        context_box.addWidget(self.latest)
        self.context.hide()
        body.addWidget(self.context)

        self.insight = _Surface()
        insight_box = QVBoxLayout(self.insight)
        insight_box.setContentsMargins(14, 12, 14, 12)
        insight_box.setSpacing(7)
        row = QHBoxLayout()
        self.insightTitle = _label("对话参考", 12, _MUTED)
        row.addWidget(self.insightTitle, 1)
        self.tension = _label("", 11)
        self.tension.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.tension)
        insight_box.addLayout(row)
        self.summary = _label("", 14, "#304c3c", True)
        insight_box.addWidget(self.summary)
        self.intent = _label("", 12, _MUTED)
        insight_box.addWidget(self.intent)
        self.insight.setToolTip("根据当前聊天片段推测，可能理解有偏差。紧张度为 0–9 的参考评分。")
        self.insight.hide()
        body.addWidget(self.insight)

        self.empty = _Surface()
        empty_box = QVBoxLayout(self.empty)
        empty_box.setContentsMargins(24, 36, 24, 36)
        empty_box.setSpacing(14)
        symbol = _label("…", 30, _GREEN, True)
        symbol.setAlignment(Qt.AlignCenter)
        empty_box.addWidget(symbol)
        self.emptyTitle = _label("等待对方的新消息", 17, "#304c3c", True)
        self.emptyTitle.setAlignment(Qt.AlignCenter)
        empty_box.addWidget(self.emptyTitle)
        self.emptyHint = _label("保持聊天窗口打开。\n收到新消息后，回复建议会出现在这里。", 13, _MUTED)
        self.emptyHint.setAlignment(Qt.AlignCenter)
        empty_box.addWidget(self.emptyHint)
        self.setupButton = PrimaryPushButton("前往设置")
        self.setupButton.clicked.connect(self.open_settings)
        self.setupButton.setVisible(not settings.has_key())
        empty_box.addWidget(self.setupButton, 0, Qt.AlignHCenter)
        if not settings.has_key():
            self.emptyTitle.setText("先设置，再开始")
            self.emptyHint.setText("配置模型和关系背景，\n让建议更贴近你们的对话。")
        body.addWidget(self.empty)
        self.replyBox = QVBoxLayout()
        self.replyBox.setSpacing(10)
        body.addLayout(self.replyBox)
        self.referenceNote = _label("AI 建议可能有偏差；自动回复只为你明确开启的会话发送。", 11, _MUTED)
        self.referenceNote.hide()
        body.addWidget(self.referenceNote)

        self.historyButton = PushButton(FIF.HISTORY, "聊天记录")
        self.historyButton.clicked.connect(self._toggle_history)
        self.historyButton.setAccessibleName("展开或收起聊天记录")
        body.addWidget(self.historyButton)
        self.feed = PlainTextEdit()
        self.feed.setReadOnly(True)
        self.feed.setPlaceholderText("识别到的聊天内容会显示在这里")
        self.feed.setMaximumBlockCount(_LOG_LINES)
        self.feed.setFixedHeight(160)
        self.feed.hide()
        body.addWidget(self.feed)
        self._history_title()
        body.addStretch(1)

    def _build_settings(self):
        self.settingsPage, body = self._scroll_page()
        heading = QHBoxLayout()
        heading.addWidget(_tool(FIF.RETURN, "返回回复建议", self._back_home))
        heading.addWidget(_label("设置", 23, "#24382d", True), 1)
        body.addLayout(heading)
        body.addWidget(_label("调整关系背景，配置判断和起草用的两个模型。", 13, _MUTED))
        preference = _Surface()
        box = QVBoxLayout(preference)
        box.setContentsMargins(16, 16, 16, 18)
        box.setSpacing(12)
        box.addWidget(_label("回复偏好", 16, "#304c3c", True))
        relation_label = _label("你们的关系", 13)
        box.addWidget(relation_label)
        self.relationshipBox = ComboBox()
        self.relationshipBox.setMinimumWidth(0)
        self.relationshipBox.addItems([name for name, value in _RELATIONSHIPS])
        self.relationshipBox.setAccessibleName("你们的关系")
        relation_label.setBuddy(self.relationshipBox)
        box.addWidget(self.relationshipBox)
        self.relEdit = LineEdit()
        self.relEdit.setPlaceholderText("例如：刚认识的朋友，正在慢慢熟悉")
        self.relEdit.setAccessibleName("自定义关系背景")
        box.addWidget(self.relEdit)
        self.relationshipBox.currentIndexChanged.connect(
            lambda index: self.relEdit.setVisible(_RELATIONSHIPS[index][1] is None)
        )
        box.addWidget(self._hint("帮助助手把握称呼、语气和回应分寸。"))
        style_label = _label("说话风格（可选）", 13)
        box.addWidget(style_label)
        self.styleEdit = LineEdit()
        self.styleEdit.setPlaceholderText("例如：话少、不用标点、偶尔用 doge、不说客套话")
        self.styleEdit.setAccessibleName("说话风格")
        style_label.setBuddy(self.styleEdit)
        box.addWidget(self.styleEdit)
        box.addWidget(self._hint("候选本来就照着你最近发的消息模仿；这里可以再补一句你自己的口吻。"))
        context_label = _label("参考上下文", 13)
        box.addWidget(context_label)
        self.contextBox = SpinBox()
        self.contextBox.setRange(3, 30)
        self.contextBox.setAccessibleName("参考的最近消息条数")
        context_label.setBuddy(self.contextBox)
        box.addWidget(self.contextBox)
        box.addWidget(self._hint(
            "生成和判断时看最近这么多条消息。太少会丢上下文，太多会稀释重点，建议 6–12。"
        ))
        target_row = QHBoxLayout()
        target_row.addWidget(_label("群聊指定回复对象", 13), 1)
        self.targetSwitch = SwitchButton()
        self.targetSwitch.setOnText("开")
        self.targetSwitch.setOffText("关")
        self.targetSwitch.setAccessibleName("群聊指定回复对象")
        target_row.addWidget(self.targetSwitch)
        box.addLayout(target_row)
        box.addWidget(self._hint(
            "开了以后群聊里可以选回复给谁，候选会针对 TA 写，填入时可带 @。关了就正常回复。"
        ))
        auto_switch_row = QHBoxLayout()
        auto_switch_row.addWidget(_label("自动切换新消息会话", 13), 1)
        self.autoSwitch = SwitchButton()
        self.autoSwitch.setOnText("开")
        self.autoSwitch.setOffText("关")
        self.autoSwitch.setAccessibleName("自动切换新消息会话")
        auto_switch_row.addWidget(self.autoSwitch)
        box.addLayout(auto_switch_row)
        box.addWidget(self._hint(
            "检测到运行期间的新未读后串行切换并采集入库；关闭生成建议即可仅采集，"
            "未开启会话自动回复时不会自动调用模型。"
        ))
        ignored_label = _label("忽略微信会话（可选，每行一个完整名称）", 13)
        box.addWidget(ignored_label)
        self.ignoredChatsEdit = PlainTextEdit()
        self.ignoredChatsEdit.setFixedHeight(88)
        self.ignoredChatsEdit.setPlaceholderText("例如：某银行服务号\n营销通知群\n文件传输助手")
        self.ignoredChatsEdit.setAccessibleName("忽略微信会话，每行一个完整名称")
        ignored_label.setBuddy(self.ignoredChatsEdit)
        box.addWidget(self.ignoredChatsEdit)
        box.addWidget(self._hint(
            "公众号、订阅号、服务号入口默认忽略。名单中的会话不自动切换、不生成建议、不自动发送；"
            "只匹配完整名称，不影响你在微信里手动浏览。"
        ))
        update_row = QHBoxLayout()
        update_row.addWidget(_label("启动时检查更新", 13), 1)
        self.updateSwitch = SwitchButton()
        self.updateSwitch.setOnText("开")
        self.updateSwitch.setOffText("关")
        self.updateSwitch.setAccessibleName("启动时检查更新")
        update_row.addWidget(self.updateSwitch)
        box.addLayout(update_row)
        box.addWidget(self._hint(
            "只向 GitHub 查最新版本号，不发送任何数据。国内访问 GitHub 慢的话关掉也行。"
        ))
        debug_row = QHBoxLayout()
        debug_row.addWidget(_label("调试视图", 13), 1)
        self.debugSwitch = SwitchButton()
        self.debugSwitch.setOnText("开")
        self.debugSwitch.setOffText("关")
        self.debugSwitch.setAccessibleName("调试视图")
        self.debugSwitch.checkedChanged.connect(self._debug_toggled)  # 这个开关立刻生效，不等「保存设置」
        debug_row.addWidget(self.debugSwitch)
        box.addLayout(debug_row)
        box.addWidget(self._hint(
            "另开一个窗口实时显示截到的画面和识别框：粉 = 未读候选、绿 = 我、蓝 = 对方、灰 = 过滤掉的灰字、"
            "红 = 当成图片丢掉、黄 = 小字丢掉。只在内存里画，不存图。"
        ))
        logging_row = QHBoxLayout()
        logging_row.addWidget(_label("记录诊断日志", 13), 1)
        self.loggingSwitch = SwitchButton()
        self.loggingSwitch.setOnText("开")
        self.loggingSwitch.setOffText("关")
        self.loggingSwitch.setAccessibleName("记录诊断日志")
        self.loggingSwitch.checkedChanged.connect(self._logging_changed)
        logging_row.addWidget(self.loggingSwitch)
        box.addLayout(logging_row)
        self.loggingLevelBox = ComboBox()
        self.loggingLevelBox.addItems([label for label, _ in _LOG_LEVEL_LABELS])
        self.loggingLevelBox.setAccessibleName("诊断日志级别")
        self.loggingLevelBox.currentIndexChanged.connect(self._logging_changed)
        box.addWidget(self.loggingLevelBox)
        box.addWidget(self._hint(
            "DEBUG 排查细节；INFO 关键运行事件；WARNING 可恢复问题；ERROR 操作失败及异常堆栈；"
            "CRITICAL 未捕获的严重异常。关闭后停止新日志记录，已有日志文件不会删除。"
        ))
        body.addWidget(preference)

        models = _Surface()
        box = QVBoxLayout(models)
        box.setContentsMargins(16, 16, 16, 18)
        box.setSpacing(12)
        box.addWidget(_label("模型", 16, "#304c3c", True))
        self._jev_testing = False
        self._fetched = _Fetched()
        self._fetched.done.connect(self._models_fetched)
        self._fetched.jevTested.connect(self._jev_tested)
        mode_label = _label("判断策略", 13)
        box.addWidget(mode_label)
        self.judgeModeBox = ComboBox()
        self.judgeModeBox.addItems([label for label, _ in _JUDGE_MODES])
        self.judgeModeBox.setAccessibleName("判断策略")
        self.judgeModeBox.currentIndexChanged.connect(self._sync_model_fields)
        mode_label.setBuddy(self.judgeModeBox)
        box.addWidget(self.judgeModeBox)
        box.addWidget(self._hint(
            "推荐 Jev 优先：连接失败、限流或服务过载时由通用模型接管；密钥或地址错误会直接提示。"
        ))
        self.jev = self._model_group(box, "判断 · Jev", "jev", providers.JEV_PROVIDERS)
        self.jevTestButton = PushButton("测试判断接口")
        self.jevTestButton.setAccessibleName("使用内置测试文本验证 Jev 或 Decisions 判断接口")
        self.jevTestButton.clicked.connect(self._test_jev)
        box.addWidget(self.jevTestButton)
        box.addWidget(self._hint(
            "仅发送内置测试文本，不读取聊天；使用当前填写的配置，无需先保存，调用可能产生少量 API 费用。"
        ))
        box.addWidget(self._hint(
            "Jev 或 OpenAI Decisions 负责意图、紧张度和候选排序；仅通用模型模式下不调用。"
            "Decisions 地址填到 /v1 即可，留空用官方地址；目前支持 gpt-6-luna，也可手填后续模型。"
        ))
        reuse_row = QHBoxLayout()
        reuse_row.addWidget(_label("通用判断复用起草配置", 13), 1)
        self.judgeReuseSwitch = SwitchButton()
        self.judgeReuseSwitch.setOnText("复用")
        self.judgeReuseSwitch.setOffText("独立")
        self.judgeReuseSwitch.setAccessibleName("通用判断是否复用起草配置")
        self.judgeReuseSwitch.checkedChanged.connect(self._sync_model_fields)
        reuse_row.addWidget(self.judgeReuseSwitch)
        box.addLayout(reuse_row)
        box.addWidget(self._hint(
            "复用最简单；切到独立后可为判断单独配置来源、模型、地址和 JUDGE_API_KEY。"
        ))
        self.judge = self._model_group(box, "通用判断 · 语言模型", "judge", providers.DRAFT_PROVIDERS)
        box.addWidget(self._hint("按固定结构输出判断与排序，不负责写回复；独立密钥只存系统环境变量。"))
        self.draft = self._model_group(box, "起草 · 语言模型", "draft", providers.DRAFT_PROVIDERS)
        box.addWidget(self._hint(
            "写那三条候选。OpenAI / Anthropic / Gemini 三种接口都走各自官方 SDK。"
            "默认 DeepSeek 官网直连，国内最快。"
        ))
        box.addWidget(_label("图片、视频封面与表情识别", 14))
        self.mediaModeBox = ComboBox()
        self.mediaModeBox.addItems([label for label, _ in _MEDIA_MODES])
        self.mediaModeBox.setAccessibleName("媒体识别方式")
        self.mediaModeBox.setMinimumWidth(0)
        box.addWidget(self.mediaModeBox)
        upload_row = QHBoxLayout()
        upload_row.addWidget(_label("允许目标消息图片上传", 13), 1)
        self.visionUploadSwitch = SwitchButton()
        self.visionUploadSwitch.setOnText("允许")
        self.visionUploadSwitch.setOffText("禁止")
        self.visionUploadSwitch.setAccessibleName("允许目标消息图片发送到视觉模型")
        upload_row.addWidget(self.visionUploadSwitch)
        box.addLayout(upload_row)
        box.addWidget(self._hint(
            "默认仅本地 OCR。开启图片上传后，只发送单条目标消息裁剪图（内存 Base64），不保存原图；"
            "视频只看封面。照片和表情需要支持图像输入的模型，模糊缩略图可能无法完整识别；"
            "仅 OCR 或识别失败时仍可自动回复，信息不确定的部分会用自然的疑问句确认。"
        ))
        self.vision = self._model_group(box, "视觉识别 · 独立模型", "vision", providers.VISION_PROVIDERS)
        self.visionTestButton = PushButton("验证图像输入能力")
        self.visionTestButton.setAccessibleName("使用合成图片验证视觉模型")
        self.visionTestButton.clicked.connect(self._test_vision)
        box.addWidget(self.visionTestButton)
        box.addWidget(self._hint("验证只发送程序生成的测试图，不发送微信截图；获取模型列表不等于模型支持识图。"))
        self._fetched.tested.connect(self._vision_tested)
        self.mediaModeBox.currentIndexChanged.connect(self._sync_model_fields)
        self.visionUploadSwitch.checkedChanged.connect(self._sync_model_fields)
        think_row = QHBoxLayout()
        think_row.addWidget(_label("起草时开启思考模式", 13), 1)
        self.thinkingSwitch = SwitchButton()
        self.thinkingSwitch.setOnText("开")
        self.thinkingSwitch.setOffText("关")
        self.thinkingSwitch.setAccessibleName("起草时开启思考模式")
        think_row.addWidget(self.thinkingSwitch)
        box.addLayout(think_row)
        box.addWidget(self._hint(
            "关：秒回，够用。开：模型先想再写，更斟酌但慢好几倍、贵一些。"
            "只有 " + " / ".join(providers.THINKING) + " 认这个开关。"
        ))
        body.addWidget(models)
        self.settingsFeedback = _label("", 13, _GREEN)
        self.settingsFeedback.hide()
        body.addWidget(self.settingsFeedback)
        actions = QHBoxLayout()
        back = PushButton("返回")
        back.clicked.connect(self._back_home)
        actions.addWidget(back)
        actions.addStretch(1)
        self.saveButton = PrimaryPushButton("保存设置")
        self.saveButton.clicked.connect(self._save)
        actions.addWidget(self.saveButton)
        body.addLayout(actions)
        body.addWidget(self._hint("保存后用于下一次生成的回复。"))
        body.addWidget(self._hint("Local AI Chat 独立维护；问题反馈请使用 LACERBA/local-ai-chat 的 GitHub Issues。"))
        body.addStretch(1)
        self._load_settings()

    def _hint(self, text):
        """设置页字段下面的灰字说明：记下来，紧凑模式一起隐藏。"""
        label = _label(text, 12, _MUTED)
        self._hintLabels.append(label)
        return label

    def _model_group(self, box, title, kind, table):
        """一组「来源 / 密钥 / 模型」控件，判断和起草各一份。table 是 core/providers.py 里那张表。"""
        key_titles = {"jev": "Jev 判断", "judge": "通用判断", "draft": "起草", "vision": "视觉识别"}
        stored_keys = {"jev": settings.jev_key, "judge": settings.judge_key,
                       "draft": settings.llm_key, "vision": settings.vision_key}
        group = SimpleNamespace(kind=kind, table=table, ids=list(table),
                                keyTitle=key_titles[kind], stored_key=stored_keys[kind],
                                widgets=[])
        heading = QHBoxLayout()
        heading.addWidget(_label(title, 14, "#304c3c", True), 1)
        group.keyState = _label("", 12, _GREEN)
        group.keyState.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        heading.addWidget(group.keyState)
        box.addLayout(heading)
        source_label = _label("来源", 13)
        box.addWidget(source_label)
        group.providerBox = ComboBox()
        group.providerBox.setMinimumWidth(0)  # 选项文字长短不一，别让它撑开设置页
        group.providerBox.addItems([table[i].name for i in group.ids])
        group.providerBox.setAccessibleName(f"{title} 来源")
        source_label.setBuddy(group.providerBox)
        box.addWidget(group.providerBox)
        group.baseLabel = _label("Base URL（留空使用 OpenAI 官方）" if kind == "jev" else "Base URL", 13)
        box.addWidget(group.baseLabel)
        group.baseEdit = LineEdit()
        group.baseEdit.setPlaceholderText("https://你的服务/v1")
        group.baseEdit.setAccessibleName(f"{title}自定义来源 Base URL")
        group.baseLabel.setBuddy(group.baseEdit)
        box.addWidget(group.baseEdit)
        key_label = _label("密钥", 13)
        box.addWidget(key_label)
        group.keyEdit = PasswordLineEdit()
        group.keyEdit.setAccessibleName(f"{title} API 密钥")
        key_label.setBuddy(group.keyEdit)
        group.keyEdit.returnPressed.connect(self._save)
        box.addWidget(group.keyEdit)
        box.addWidget(self._hint(
            "按来源填写 OpenRouter、TypeSafe 或 Decisions 服务的 key；密钥只存系统环境变量。" if kind == "jev"
            else "上面选哪家就填哪家的 key；换来源重填一次，只存对应密钥。"))
        model_label = _label("模型", 13)
        box.addWidget(model_label)
        row = QHBoxLayout()
        row.setSpacing(8)
        group.modelBox = EditableComboBox()  # 能选也能手打，接口新出的模型不用等我改代码
        group.modelBox.setMinimumWidth(0)
        group.modelBox.setAccessibleName(f"{title} 模型")
        model_label.setBuddy(group.modelBox)
        row.addWidget(group.modelBox, 1)
        group.fetchButton = PushButton("获取模型")
        group.fetchButton.setAccessibleName(f"获取{title}的可用模型列表")
        group.fetchButton.clicked.connect(lambda: self._fetch_models(group))
        row.addWidget(group.fetchButton)
        box.addLayout(row)
        group.status = _label("", 12, _MUTED)
        box.addWidget(group.status)
        group.providerBox.currentIndexChanged.connect(lambda _: self._provider_changed(group))
        group.widgets = [group.providerBox, group.keyEdit, group.modelBox, group.fetchButton]
        if group.baseEdit:
            group.widgets.append(group.baseEdit)
        return group

    @staticmethod
    def _provider_of(group):
        return group.ids[max(0, group.providerBox.currentIndex())]

    def _provider_changed(self, group):
        """换来源：模型框回到这家该有的值（存的就是这家才用存的，否则用它的默认），状态清掉。"""
        provider = self._provider_of(group)
        saved = {"jev": settings.jev_provider, "judge": settings.judge_provider,
                 "draft": settings.draft_provider, "vision": settings.vision_provider}[group.kind]()
        stored = {"jev": settings.jev_model, "judge": settings.judge_model,
                  "draft": settings.draft_model, "vision": settings.vision_model}[group.kind]()
        group.modelBox.clear()
        group.modelBox.setText(stored if provider == saved else group.table[provider].default)
        if group.kind == "jev" and provider == "openai_decisions" and not group.baseEdit.text().strip():
            group.baseEdit.setText(settings.jev_base_url() or group.table[provider].base)
        group.status.setText("")
        self._sync_model_fields()

    def _sync_model_fields(self, *_):
        """同步三组模型的密钥状态、可用性和自定义地址显隐。"""
        mode = _JUDGE_MODES[max(0, self.judgeModeBox.currentIndex())][1]
        reuse = self.judgeReuseSwitch.isChecked()
        visual = _MEDIA_MODES[max(0, self.mediaModeBox.currentIndex())][1] != "ocr"
        self.visionUploadSwitch.setEnabled(visual)
        active = {"jev": mode != "llm_only", "judge": mode != "jev_only" and not reuse,
                  "draft": True, "vision": visual and self.visionUploadSwitch.isChecked()}
        self.visionTestButton.setEnabled(active["vision"])
        self.jevTestButton.setEnabled(active["jev"] and not self._jev_testing)
        for group in (self.jev, self.judge, self.draft, self.vision):
            provider = self._provider_of(group)
            name = group.table[provider].name
            configured = bool(group.stored_key())
            if group.kind == "judge" and mode != "jev_only" and reuse:
                group.keyState.setText("复用起草配置")
            else:
                group.keyState.setText("已配置" if configured else "未配置")
            group.keyEdit.setPlaceholderText(
                "已配置，留空保留" if configured else f"输入 {name} API 密钥")
            for widget in group.widgets:
                widget.setEnabled(active[group.kind])
            if self._compact:
                name = group.providerBox.fontMetrics().elidedText(name, Qt.ElideRight, 180)
            group.providerBox.setText(name)
            if group.baseLabel:
                custom = active[group.kind] and (provider in providers.CUSTOM
                                                  or group.kind == "jev" and provider == "openai_decisions")
                group.baseLabel.setVisible(custom)
                group.baseEdit.setVisible(custom)

    def _fetch_models(self, group):
        """「获取模型」：拿填的 key（没填就拿存的）去问接口，网络调用丢后台线程。"""
        provider = self._provider_of(group)
        custom = provider in providers.CUSTOM
        base = group.baseEdit.text().strip() if custom or provider == "openai_decisions" else None
        key = group.keyEdit.text().strip() or group.stored_key()
        if not key:
            group.status.setText("先填密钥")
            return
        if custom and not base:
            group.status.setText("先填 Base URL")
            return
        group.status.setText("获取中…")
        group.fetchButton.setEnabled(False)
        threading.Thread(target=lambda: self._list_models(group, provider, key, base),
                         daemon=True).start()

    def _list_models(self, group, provider, key, base):
        """后台线程：判断走 jev_client，起草按协议走 llm；失败把原因一起送回主线程。"""
        register_secret(key)
        log_event("settings.models_requested", group=group.kind, provider=provider)
        try:
            if group.kind == "jev":
                models = jev_client.list_models(provider, key, base_url=base)
            else:
                spec = group.table[provider]
                models = llm.list_models(spec.protocol, base or spec.base, key, headers=spec.headers)
                if spec.keep:  # 目录里混了别的协议时，只留这条路打得通的
                    models = [m for m in models if spec.keep(m)]
            reason = "" if models else "这个来源没返回任何模型"
        except Exception as exc:  # 线程里漏异常会静默吞掉，按钮就永远停在禁用态
            log_event("settings.models_failed", level="error", exc_info=True, group=group.kind, provider=provider)
            models, reason = [], jev_client.redact_secrets(str(exc).replace(key, "[REDACTED]"))[:120]
        log_event("settings.models_completed", group=group.kind, provider=provider,
                  model_count=len(models), reason=reason)
        self._fetched.done.emit(group, models, reason)

    def _jev_test_config(self):
        provider = self._provider_of(self.jev)
        return {"provider": provider, "model": self.jev.modelBox.text().strip(),
                "base_url": self.jev.baseEdit.text().strip() if provider == "openai_decisions" else "",
                "api_key": self.jev.keyEdit.text().strip() or self.jev.stored_key()}

    def _test_jev(self):
        if self._jev_testing or not self.jevTestButton.isEnabled():
            return
        config = self._jev_test_config()
        if not config["api_key"]:
            self.jev.status.setText("先填密钥")
            self.jev.keyEdit.setFocus()
            return
        if not config["model"]:
            self.jev.status.setText("先获取并选择一个模型，或手动填写模型")
            self.jev.modelBox.setFocus()
            return
        self._jev_testing = True
        self._sync_model_fields()
        self.jev.status.setText("正在使用内置测试文本验证…")
        threading.Thread(target=self._test_jev_bg, args=(config,), daemon=True).start()

    def _test_jev_bg(self, config):
        register_secret(config["api_key"])
        log_event("settings.jev_test_requested", provider=config["provider"], model=config["model"])
        # 使用固定文本覆盖概率、分类和评分，不进入真实聊天的起草或发送链路。
        state = build_state([("her", "你好，今天一起吃午饭吗？")], "friends")
        questions = {name: JUDGE_QUESTIONS[name] for name in ("literal_question", "true_intent", "danger_level")}
        try:
            result = jev_client.ask(state, questions, provider=config["provider"], model=config["model"],
                                    base_url=config["base_url"] or None, api_key=config["api_key"],
                                    timeout=20, max_retries=1)
            answers = validate_answers(result.get("answers"), questions,
                                       allow_fractional_score=config["provider"] == "openai_decisions")
            message = (f"判断接口已通过验证：字面概率 {answers['literal_question']['noul']:.0%}，"
                       f"意图 {_choice(answers, 'true_intent')}，紧张度 {answers['danger_level']['score']:g}/9")
            log_event("settings.jev_test_completed", provider=config["provider"], model=config["model"])
        except Exception as exc:
            log_event("settings.jev_test_failed", level="error", exc_info=True,
                      provider=config["provider"], model=config["model"])
            message = "判断接口验证失败：" + redact_text(str(exc))[:160]
        self._fetched.jevTested.emit(config, message)

    def _jev_tested(self, config, message):
        self._jev_testing = False
        self._sync_model_fields()
        # 测试期间配置可能变更，旧请求的成功结果不能用来证明新配置可用。
        self.jev.status.setText(message if config == self._jev_test_config() else "配置已变更，请重新测试当前配置。")

    def _test_vision(self):
        if not self.visionUploadSwitch.isChecked():
            return
        provider = self._provider_of(self.vision)
        config = {"provider": provider, "model": self.vision.modelBox.text().strip(),
                  "base_url": self.vision.baseEdit.text().strip() if provider in providers.CUSTOM else "",
                  "api_key": self.vision.keyEdit.text().strip() or self.vision.stored_key(), "mode": "vision"}
        self.visionTestButton.setEnabled(False)
        self.vision.status.setText("正在使用合成图片验证…")
        threading.Thread(target=self._test_vision_bg, args=(config,), daemon=True).start()

    def _test_vision_bg(self, config):
        from io import BytesIO
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (180, 100), "white")
        ImageDraw.Draw(image).rectangle((40, 20, 140, 80), fill="red")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        try:
            result = vision.recognize(buffer.getvalue(), "image", "", config)
            message = "图像输入已通过验证：" + result["summary"] if result["status"] == "ready" else "模型未能识别测试图"
        except Exception as exc:
            log_event("settings.vision_test_failed", level="error", exc_info=True)
            message = "图像输入验证失败：" + redact_text(str(exc))[:160]
        self._fetched.tested.emit(message)

    def _vision_tested(self, message):
        self._sync_model_fields()
        self.vision.status.setText(message)

    def _models_fetched(self, group, models, reason):
        """回到主线程：填进下拉框，原来选中的还在列表里就留着。"""
        group.fetchButton.setEnabled(True)
        if not models:
            group.status.setText(reason or "获取失败，检查密钥、网络或 Base URL")
            return
        current = group.modelBox.text().strip()
        group.modelBox.clear()
        group.modelBox.addItems(models)
        if current in models:
            group.modelBox.setCurrentIndex(models.index(current))
        else:
            group.modelBox.setText(current)  # 手打的没在列表里也不清掉
        group.status.setText(f"共 {len(models)} 个")

    def _set_group(self, group, provider, model):
        """把存下来的来源和模型放回一组控件里；填充不算用户操作，别触发换来源的重置。"""
        group.providerBox.blockSignals(True)
        group.providerBox.setCurrentIndex(group.ids.index(provider))
        group.providerBox.blockSignals(False)
        group.keyEdit.clear()
        group.modelBox.clear()
        group.modelBox.setText(model)
        group.status.setText("")

    def _load_settings(self):
        relationship = settings.relationship()
        index = next((i for i, (_, value) in enumerate(_RELATIONSHIPS) if value == relationship),
                     len(_RELATIONSHIPS) - 1)
        self.relationshipBox.setCurrentIndex(index)
        self.relEdit.setText(relationship if _RELATIONSHIPS[index][1] is None else "")
        self.relEdit.setVisible(_RELATIONSHIPS[index][1] is None)
        self.styleEdit.setText(settings.style())
        self.contextBox.setValue(settings.context())
        self.targetSwitch.setChecked(settings.reply_target())
        mode = settings.judge_mode()
        self.judgeModeBox.setCurrentIndex(next(i for i, (_, value) in enumerate(_JUDGE_MODES)
                                                   if value == mode))
        self.judgeReuseSwitch.setChecked(settings.judge_reuse_draft())
        self._set_group(self.jev, settings.jev_provider(), settings.jev_model())
        self.jev.baseEdit.setText(settings.jev_base_url())
        self._set_group(self.judge, settings.judge_provider(), settings.judge_model())
        self._set_group(self.draft, settings.draft_provider(), settings.draft_model())
        self.judge.baseEdit.setText(settings.judge_base_url())
        self.draft.baseEdit.setText(settings.draft_base_url())
        self._set_group(self.vision, settings.vision_provider(), settings.vision_model())
        self.vision.baseEdit.setText(settings.vision_base_url())
        self.mediaModeBox.setCurrentIndex(next(i for i, (_, value) in enumerate(_MEDIA_MODES)
                                              if value == settings.media_mode()))
        self.visionUploadSwitch.setChecked(settings.vision_upload())
        self.thinkingSwitch.setChecked(settings.thinking())
        self.autoSwitch.setChecked(settings.auto_switch())
        self.ignoredChatsEdit.setPlainText("\n".join(settings.ignored_chats("wechat")))
        self.updateSwitch.setChecked(settings.check_update())
        self.set_debug_switch(settings.debug_view())  # 屏蔽信号地拨，别在加载时开关一遍窗口
        self._load_logging_settings()
        self._sync_model_fields()  # 上面屏蔽了信号，这里补一次
        self.settingsFeedback.hide()

    def _load_logging_settings(self):
        logging_switch_blocked = self.loggingSwitch.blockSignals(True)
        logging_level_blocked = self.loggingLevelBox.blockSignals(True)
        self.loggingSwitch.setChecked(settings.logging_enabled())
        self.loggingLevelBox.setCurrentIndex(next(
            i for i, (_, value) in enumerate(_LOG_LEVEL_LABELS) if value == settings.logging_level()))
        self.loggingLevelBox.setEnabled(settings.logging_enabled())
        self.loggingSwitch.blockSignals(logging_switch_blocked)
        self.loggingLevelBox.blockSignals(logging_level_blocked)

    def _save(self):
        previous_media = (settings.media_mode(), settings.vision_upload(), settings.vision_provider(),
                          settings.vision_model(), settings.vision_base_url())
        vision_key_changed = bool(self.vision.keyEdit.text().strip())
        relationship = _RELATIONSHIPS[self.relationshipBox.currentIndex()][1]
        relationship = relationship or self.relEdit.text().strip()
        mode = _JUDGE_MODES[max(0, self.judgeModeBox.currentIndex())][1]
        reuse = self.judgeReuseSwitch.isChecked()
        jev_provider = self._provider_of(self.jev)
        judge_provider = self._provider_of(self.judge)
        draft_provider = self._provider_of(self.draft)
        if not relationship:
            self._settings_feedback("请填写关系背景，或选择一个已有选项。", error=True)
            self.relEdit.setFocus()
            return
        active = [(self.draft, draft_provider)]
        if mode != "llm_only":
            active.append((self.jev, jev_provider))
        if mode != "jev_only" and not reuse:
            active.append((self.judge, judge_provider))
        media_mode = _MEDIA_MODES[max(0, self.mediaModeBox.currentIndex())][1]
        if media_mode != "ocr" and self.visionUploadSwitch.isChecked():
            active.append((self.vision, self._provider_of(self.vision)))
        for group, provider in active:
            name = group.table[provider].name
            if group.baseEdit and provider in providers.CUSTOM and not group.baseEdit.text().strip():
                self._settings_feedback(f"{group.keyTitle}的自定义来源要填 Base URL。", error=True)
                group.baseEdit.setFocus()
                return
            if not group.keyEdit.text().strip() and not group.stored_key():
                self._settings_feedback(f"请先填写 {group.keyTitle} 的 API 密钥。", error=True)
                group.keyEdit.setFocus()
                return
            if not group.modelBox.text().strip():
                self._settings_feedback(f"{name} 请先获取并选择一个模型。", error=True)
                group.modelBox.setFocus()
                return
        try:
            settings.save(relationship, self.contextBox.value(),
                          jev_provider_text=jev_provider,
                          jev_key_text=self.jev.keyEdit.text().strip() or None,
                          jev_model_text=self.jev.modelBox.text().strip(),
                          jev_base_url_text=self.jev.baseEdit.text().strip(),
                          judge_mode_text=mode,
                          judge_reuse_draft_on=reuse,
                          judge_provider_text=judge_provider,
                          judge_key_text=self.judge.keyEdit.text().strip() or None,
                          judge_model_text=self.judge.modelBox.text().strip(),
                          judge_base_url_text=self.judge.baseEdit.text().strip(),
                          draft_provider_text=draft_provider,
                          llm_key_text=self.draft.keyEdit.text().strip() or None,
                          draft_model_text=self.draft.modelBox.text().strip(),
                          draft_base_url_text=self.draft.baseEdit.text().strip(),
                          media_mode_text=media_mode,
                          vision_upload_on=self.visionUploadSwitch.isChecked(),
                          vision_provider_text=self._provider_of(self.vision),
                          vision_key_text=self.vision.keyEdit.text().strip() or None,
                          vision_model_text=self.vision.modelBox.text().strip(),
                          vision_base_url_text=self.vision.baseEdit.text().strip(),
                          reply_target_on=self.targetSwitch.isChecked(),
                          style_text=self.styleEdit.text().strip(),
                          thinking_on=self.thinkingSwitch.isChecked(),
                          check_update_on=self.updateSwitch.isChecked(),
                          auto_switch_on=self.autoSwitch.isChecked(),
                          ignored_chats_value=self.ignoredChatsEdit.toPlainText().splitlines())
        except Exception:
            log_event("settings.save_failed", level="error", exc_info=True)
            self._settings_feedback("保存失败，请检查配置文件是否可写后重试。", error=True)
            return
        current_media = (settings.media_mode(), settings.vision_upload(), settings.vision_provider(),
                         settings.vision_model(), settings.vision_base_url())
        if self.on_processing_change and (previous_media != current_media or vision_key_changed):
            self.on_processing_change()
        self._load_settings()
        self._render_auto_reply()
        self._render_targets()  # 开关刚改过，回到首页时这一行该显该藏得重算一次
        ignored = settings.is_ignored_chat(self._shown, self.app_key_of())
        if ignored:
            self.show_cached(None)
            self.set_busy(False)
            self.set_status("当前会话已忽略，不参与自动处理", "idle")
        self._settings_feedback("设置已保存，将用于下一次回复。")
        self.setupButton.hide()
        if not ignored and not self.cands and not self._busy:
            self._empty_text()
            self.set_status("设置已就绪，等待新消息", "idle")

    def _logging_changed(self, *_):
        enabled = self.loggingSwitch.isChecked()
        level = _LOG_LEVEL_LABELS[max(0, self.loggingLevelBox.currentIndex())][1]
        self.loggingLevelBox.setEnabled(enabled)
        try:
            settings.save(logging_enabled_on=enabled, logging_level_text=level)
            if self.on_logging_change:
                self.on_logging_change()
            self.settingsFeedback.hide()
        except Exception:
            log_event("settings.logging_save_failed", level="error", exc_info=True)
            self._settings_feedback("日志设置保存或应用失败，请检查配置文件和 log 目录是否可写。", error=True)
            self._load_logging_settings()

    def _debug_toggled(self, on):
        """调试视图独立于「保存设置」：拨一下就开窗/收窗，顺手落盘，重启还在。"""
        settings.save(debug_view_on=on)
        if self.on_toggle_debug:
            self.on_toggle_debug(on)

    def set_debug_switch(self, on):
        """调试窗被用户直接关掉时把开关拨回去；屏蔽信号，免得又回调一圈。"""
        self.debugSwitch.blockSignals(True)
        self.debugSwitch.setChecked(on)
        self.debugSwitch.blockSignals(False)

    def _settings_feedback(self, text, error=False):
        color = "#b44832" if error else _GREEN
        qss = f"BodyLabel {{ color: {color}; background: transparent; }}"
        setCustomStyleSheet(self.settingsFeedback, qss, qss)
        self.settingsFeedback.setText(text)
        self.settingsFeedback.show()

    def open_settings(self):
        if self.pages.currentWidget() != self.settingsPage:
            self._load_settings()
        self.pages.setCurrentWidget(self.settingsPage)
        self.settingsButton.setEnabled(False)
        (self.relationshipBox if settings.has_key() else self.jev.keyEdit).setFocus()

    def _back_home(self):
        self.jev.keyEdit.clear()
        self.draft.keyEdit.clear()
        self.pages.setCurrentWidget(self.home)
        self.settingsButton.setEnabled(True)

    def _fill(self, index):
        if self._busy or not self._current or index >= len(self.cands):
            return
        try:
            self.on_fill(self.cands[index])
        except Exception as e:
            log_event("reply.manual_fill_failed", level="error", exc_info=True, chat=self._shown, candidate_index=index)
            # 状态栏保持友好文案；真实原因和压缩堆栈进聊天记录，认得出是哪一步炸的
            import traceback
            self.set_status("未能填入，请确认聊天窗口可用后重试，或复制回复。", "error")
            self.log(f"[填入失败] {type(e).__name__}: {e}")
            self.log(f"[填入失败堆栈] {' '.join(traceback.format_exc().split())[:300]}")
            return
        self.set_status("已尝试填入，请确认内容后发送。", "success")

    def _copy(self, index):
        if self._busy or not self._current or index >= len(self.cands):
            return
        self.app.clipboard().setText(self.cands[index])
        self.set_status("回复已复制，可粘贴并修改。", "success")

    def _capture_toggled(self, on):
        """用户自己拨的开关：界面先改，再通知父进程去开/停采集。"""
        self._capture_text(on)
        if self.on_toggle_capture:
            self.on_toggle_capture(on)

    def set_update(self, latest, url):
        """main.py 后台线程查到比当前新的版本才会调这个。只显示版本号和 Release 链接，别的什么都没有。"""
        self.updateLabel.setText(f"有新版本 v{latest}")
        self.updateLink.setUrl(url)
        self.updateBar.show()

    def set_capture(self, on, reason=""):
        """父进程回报的状态：只改界面，不回调（不然和父进程来回打架）。reason 为空用默认说明。"""
        self.captureSwitch.blockSignals(True)
        self.captureSwitch.setChecked(on)
        self.captureSwitch.blockSignals(False)
        self._capture_text(on, reason)

    def _capture_text(self, on, reason=""):
        """开关状态对应的状态行和空态文案。已有的候选不受影响，暂停了照样能填入/复制。"""
        configured = settings.has_key()
        if not on:
            self.set_status(reason or "采集已暂停，聊天内容不再读取", "warning")
        elif not settings.analysis_enabled():
            self.set_status("仅采集模式：正在读取并保存到本地 SQLite", "idle")
        elif configured:
            self.set_status("等待新消息", "idle")
        else:
            self.set_status("请先在设置中配置模型", "warning")
        if self._busy or self.cands:  # 正在生成或已有候选时，空态卡片本来就看不见
            return
        if not on:
            self.emptyTitle.setText("采集已暂停")
            self.emptyHint.setText("聊天内容暂时不再读取。\n打开标题栏的开关，继续接收新消息。")
            self.setupButton.setVisible(not configured)
        else:
            self._empty_text()

    def set_busy(self, busy):
        self._busy = busy
        self.progress.setVisible(busy)
        if busy:
            self.invalidate_replies()
            self.progress.start()
            self.set_status("正在根据新消息整理回复…", "busy")
            if not self.cands:
                self.emptyTitle.setText("正在想一句合适的回复")
                self.emptyHint.setText("正在结合上下文生成建议，稍等一下。")
                self.setupButton.hide()
        else:
            self.progress.stop()
            if not self.cands:
                self._empty_text()
        for card in self.cards:
            card.set_available(self._current and not busy)

    def _empty_text(self):
        """空态卡片的默认文案，配好没配好两套说法。"""
        if settings.is_ignored_chat(self._shown, self.app_key_of()):
            self.emptyTitle.setText("当前会话已忽略")
            self.emptyHint.setText("不自动切换、不生成回复建议、不自动发送。\n你仍可以在微信中手动浏览。")
            self.setupButton.hide()
            return
        if not settings.analysis_enabled():
            self.emptyTitle.setText("仅采集并保存消息")
            self.emptyHint.setText("不调用模型，不自动发送。\n需要回复建议时，开启上方的生成建议。")
            self.setupButton.hide()
            return
        if self._no_reply_reason:
            self.emptyTitle.setText("本轮无需回复")
            self.emptyHint.setText(self._no_reply_reason + "\n继续监听，对方有新消息时会重新判断。")
            self.setupButton.hide()
            return
        configured = settings.has_key()
        self.emptyTitle.setText("等待对方的新消息" if configured else "先设置，再开始")
        self.emptyHint.setText("保持聊天窗口打开。\n收到新消息后，回复建议会出现在这里。"
                               if configured else "配置模型和关系背景，\n让建议更贴近你们的对话。")
        self.setupButton.setVisible(not configured)

    def invalidate_replies(self):
        self._current = False
        self._no_reply_reason = ""
        if self.cands:
            self.updated.setText("上次建议")
        for card in self.cards:
            card.set_available(False)

    def set_status(self, text, kind="idle"):
        text = redact_text(text)
        current = (self._shown, text, kind)
        if current != getattr(self, "_last_logged_status", None):
            log_event("ui.status", level=kind, chat=self._shown, status=kind, reason=text)
            self._last_logged_status = current
        self.status.setToolTip(text)
        colors = {"idle": _MUTED, "busy": _GREEN, "success": _GREEN,
                  "warning": "#93611d", "error": "#b44832"}
        markers = {"idle": "●", "busy": "●", "success": "✓", "warning": "!", "error": "!"}
        qss = f"BodyLabel {{ color: {colors.get(kind, _MUTED)}; background: transparent; }}"
        setCustomStyleSheet(self.status, qss, qss)
        self.status.setText(f"{markers.get(kind, '●')}  {text}")
        if kind == "error" and self._busy:
            self.set_busy(False)
        if kind == "error" and not self.cands:
            self.emptyTitle.setText("暂时没有可用的回复")
            self.emptyHint.setText("请按上方提示处理。收到新的对方消息后会再次尝试。")
            self.setupButton.setVisible(not settings.has_key())

    def _toggle_history(self):
        self.feed.setVisible(self.feed.isHidden())
        self._history_title()

    def _history_title(self):
        action = "展开" if self.feed.isHidden() else "收起"
        count = self.counts.get(self._shown, 0)
        self.historyButton.setText(f"{action}聊天记录" + (f" · {count}" if count else ""))

    def log(self, line):
        """采集状态行：只进正在看的那个会话，不按会话存。"""
        bar = self.feed.verticalScrollBar()
        follow = self.feed.isHidden() or bar.value() >= bar.maximum() - 4
        self.feed.appendPlainText(line)
        if follow:
            bar.setValue(bar.maximum())

    def log_message(self, who, text, name="", timestamp=None, chat=None):
        """按会话存一份；只有正在看的那个会往显示区里写。"""
        chat = chat or self._shown
        speaker = (name or "对方") if who == "her" else "我"
        timestamp = timestamp or datetime.now().strftime("%H:%M")
        self.counts[chat] = self.counts.get(chat, 0) + 1
        lines = self.feeds.setdefault(chat, [])
        lines.append(f"{timestamp}  {speaker}\n{text}\n")
        del lines[:-_LOG_LINES]
        if who == "her":
            self.hers[chat] = text
        self._add_chat(chat)
        if chat != self._shown:
            return
        self.log(lines[-1])
        if who == "her":
            self._show_latest(text)
        self._history_title()

    def _show_latest(self, text):
        self.latest.setText(text if len(text) <= 120 else text[:120] + "…")
        self.latest.setToolTip(text)
        self.context.show()

    def replace_history(self, title, messages):
        self.feeds[title] = []
        self.counts[title] = 0
        self.hers.pop(title, None)
        if title == self._shown:
            self.feed.clear()
        for who, text, name in messages:
            self.log_message(who, text, name, timestamp="已采集", chat=title)
        if title == self._shown:
            if title not in self.hers:
                self.latest.clear()
                self.context.hide()
            self._history_title()

    def set_read_state(self, title, read_at, new_count, status):
        moment = datetime.fromtimestamp(read_at).strftime("%H:%M:%S")
        text = (f"本地消息已修订 · {moment} · 新增 {new_count} 条" if status == "corrected"
                else f"识别文本已隔离待确认 · {moment}" if status == "pending"
                else f"本地已保存 · {moment} · 新增 {new_count} 条" if status == "read"
                else f"OCR 暂未识别到文本消息 · {moment}")
        self.read_states[title] = text
        if title == self._shown:
            self.readState.setText(text)

    def _render_processing(self):
        for widget, value in ((self.analysisSwitch, settings.analysis_enabled()),
                              (self.sendSwitch, settings.send_enabled())):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)

    def _processing_toggled(self, _):
        sending = self.sendSwitch.isChecked()
        if sending and not settings.send_enabled():
            answer = QMessageBox.warning(
                self.win, "允许自动发送",
                "开启后，已明确授权的会话会回复最后一条对方消息，包括进入会话前已有的消息。\n"
                "仅采集和仅生成建议都不需要开启。是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self._render_processing()
                return
        try:
            settings.save(analysis_enabled_on=self.analysisSwitch.isChecked(), send_enabled_on=sending)
        except Exception:
            log_event("settings.processing_save_failed", level="error", exc_info=True)
            self._render_processing()
            self.set_status("处理模式保存失败，请检查配置文件是否可写。", "error")
            return
        if self.on_processing_change:
            self.on_processing_change()
        self._empty_text()
        self._render_auto_reply()
        text = ("仅采集模式：本地读取并保存消息，不调用模型、不自动发送"
                if not self.analysisSwitch.isChecked() else "生成建议模式：不会自动发送"
                if not sending else "生成建议与授权会话自动发送已开启")
        self.set_status(text, "warning" if sending and self.analysisSwitch.isChecked() else "success")

    def _auto_send_key(self):
        return "ctrl_enter" if self.autoKeyBox.currentIndex() == 1 else "enter"

    def _render_auto_reply(self):
        app_key = self.app_key_of()
        config = settings.auto_reply_config(self._shown, app_key)
        ignored = settings.is_ignored_chat(self._shown, app_key)
        enabled = bool(self._shown) and app_key == "wechat" and not ignored
        self.autoReplySwitch.blockSignals(True)
        self.autoReplySwitch.setChecked(config["enabled"] if enabled else False)
        self.autoReplySwitch.blockSignals(False)
        self.autoKeyBox.blockSignals(True)
        self.autoKeyBox.setCurrentIndex(1 if config["send_key"] == "ctrl_enter" else 0)
        self.autoKeyBox.blockSignals(False)
        self.autoReplySwitch.setEnabled(enabled)
        self.autoKeyBox.setEnabled(enabled)
        tooltip = ("当前会话已忽略，不参与自动处理" if ignored
                   else "只对当前显示的会话生效；全局允许发送开启后，最佳回复才会覆盖输入框并发送"
                   if app_key == "wechat" else "当前聊天软件暂未开放自动发送，请先使用手动填入")
        self.autoReplySwitch.setToolTip(tooltip)
        self.autoKeyBox.setToolTip("必须和微信中的发送快捷键设置一致；选出最佳回复后会自动发送"
                                   if app_key == "wechat" and not ignored else tooltip)

    def _auto_reply_toggled(self, on):
        title = self._shown
        app_key = self.app_key_of()
        if not title or app_key != "wechat" or settings.is_ignored_chat(title, app_key):
            self._render_auto_reply()
            if settings.is_ignored_chat(title, app_key):
                self.set_status("当前会话已忽略，不参与自动处理", "idle")
            elif title and app_key != "wechat":
                self.set_status("当前聊天软件暂未开放自动发送，请先使用手动填入", "warning")
            return
        config = settings.auto_reply_config(title, app_key)
        confirmed = None
        if on and not config["confirmed"]:
            answer = QMessageBox.warning(
                self.win, "开启自动回复",
                f"开启后，「{title}」生成的最佳回复会覆盖输入框已有文字并直接发送。\n\n"
                "进入会话前已有的最后一条对方消息也会回复，识别不清楚时会提问确认。\n"
                "请确认发送快捷键与微信设置一致。是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self._render_auto_reply()
                return
            confirmed = True
        try:
            settings.set_auto_reply(title, on, self._auto_send_key(), app_key, confirmed=confirmed)
        except Exception:
            log_event("settings.auto_reply_save_failed", level="error", exc_info=True, chat=title)
            self.set_status("自动回复设置保存失败，请检查配置文件是否可写。", "error")
            self._render_auto_reply()
            return
        if self.on_processing_change:
            self.on_processing_change()
        self.set_status(f"「{title}」自动回复已{'开启' if on else '关闭'}",
                        "warning" if on else "success")

    def _auto_key_changed(self, _):
        app_key = self.app_key_of()
        if not self._shown or app_key != "wechat" or settings.is_ignored_chat(self._shown, app_key):
            return
        try:
            settings.set_auto_reply(self._shown, self.autoReplySwitch.isChecked(),
                                    self._auto_send_key(), app_key)
        except Exception:
            log_event("settings.send_shortcut_save_failed", level="error", exc_info=True, chat=self._shown)
            self.set_status("发送快捷键保存失败，请检查配置文件是否可写。", "error")
            self._render_auto_reply()

    def current_chat(self):
        """界面上正在看的会话（不一定是微信当前开着的那个）。"""
        return self._shown

    def set_chat(self, title):
        """微信切到了哪个会话：登记进下拉框并自动跟过去，不触发用户选择的回调。"""
        if not title:
            return
        browsing = self._shown != self._chat  # 正看着的就是它、但之前是「浏览中」：也得重画，把填入放开
        self._chat = title
        self._add_chat(title)
        if title != self._shown or browsing:
            self.chatBox.blockSignals(True)
            self.chatBox.setCurrentIndex(self.chatBox.findText(title))
            self.chatBox.blockSignals(False)
            self._switch_to(title)
        self._follow_text()

    def _add_chat(self, title):
        """新会话自动进下拉框；addItem 添第一条时会自己选中，别让它触发切换。"""
        if not title or self.chatBox.findText(title) >= 0:
            return
        self.chatBox.blockSignals(True)
        self.chatBox.addItem(title)
        self.chatBox.blockSignals(False)

    def _on_chat_selected(self, index):
        """用户自己挑了一个会话：只换看的内容，微信那边不动。"""
        title = self.chatBox.itemText(index)
        if title and title != self._shown:
            self._switch_to(title)

    def _switch_to(self, title):
        """换正在看的会话：记录、对方最近说、条数、上次的建议一起换过去。"""
        self._shown = title
        self.readState.setText(self.read_states.get(title, "消息读取后自动保存到本地 SQLite"))
        self.feed.clear()
        for line in self.feeds.get(title, []):
            self.feed.appendPlainText(line)
        her = self.hers.get(title)
        if her:
            self._show_latest(her)
        else:
            self.context.hide()
        self._history_title()
        self._follow_text()
        self._render_auto_reply()
        self._render_targets()
        self.show_cached(self.result_of(title) if self.result_of else None)

    def set_targets(self, chat, senders, current):
        """某个会话的发言人名单（最近的在前）和当前回复对象；正看着它才重画。"""
        self.targets[chat] = (list(senders), current)
        if chat == self._shown:
            self._render_targets()

    def _render_targets(self):
        """开关关着、或这个会话没有发言人（单聊），这一行就不出现。
        重填下拉框时屏蔽信号，别把自己的填充当成用户挑的。"""
        senders, current = self.targets.get(self._shown, ([], None))
        visible = bool(senders) and settings.reply_target()
        self.targetRow.setVisible(visible)
        if not visible:
            return
        self.targetBox.blockSignals(True)
        self.targetBox.clear()
        self.targetBox.addItems(senders)
        self.targetBox.setCurrentIndex(senders.index(current) if current in senders else 0)
        self.targetBox.blockSignals(False)

    def _on_target_selected(self, index):
        """用户挑了回复对象。浏览别的会话时改的就是那个会话的对象——记录、候选也都按会话走，口径一致。"""
        name = self.targetBox.itemText(index)
        if not name:
            return
        senders, _ = self.targets.get(self._shown, ([], None))
        self.targets[self._shown] = (senders, name)
        self.set_status(f"按「{name}」重新生成…", "busy")
        if self.on_target_change:
            self.on_target_change(self._shown, name)

    def at_prefix_enabled(self):
        """填入时要不要带「@名字 」前缀（只记在界面上，不落盘）。"""
        return self.atCheck.isChecked()

    def _follow_text(self):
        self.chatFollow.setText(("跟随" if self._shown == self._chat else "浏览中") if self._chat else "")

    def show_cached(self, result):
        """把某个会话上次的结果放回界面；没有就回到空态。浏览别的会话时只给看不给填——
        微信当前开着的不是它，填进去就串会话了。"""
        ignored = settings.is_ignored_chat(self._shown, self.app_key_of())
        if result and not ignored:
            self.show(result)
        else:
            self._no_reply_reason = ""
            self.cands = []
            self._clear_cards()
            self.insight.hide()
            self.referenceNote.hide()
            self.empty.show()
            self.updated.setText("")
            self._empty_text()
        if ignored:
            self.context.hide()
            self.targetRow.hide()
            self.set_status("当前会话已忽略，不参与自动处理", "idle")
        elif self._shown != self._chat:
            self.invalidate_replies()
            self.set_status(f"正在浏览「{self._shown}」，只看不填；切回这个会话才能用。")

    def show(self, result):
        """按推荐顺序展示，按钮始终绑定 candidates 的原始索引。"""
        if settings.is_ignored_chat(self._shown, self.app_key_of()):
            self.show_cached(None)
            self.set_busy(False)
            return
        no_reply = result.get("reply_required") is False
        self._no_reply_reason = (redact_text(result.get("no_reply_reason") or "对方已结束话题，本轮无需回复")
                                 if no_reply else "")
        self.cands = [] if no_reply else result["candidates"]
        self.set_busy(False)
        self._current = bool(self.cands)
        self._clear_cards()
        ranking_valid = bool(result.get("ranking_valid", True))
        best = result.get("best_index", 0)
        if best not in range(len(self.cands)):
            best = 0
            ranking_valid = False
        raw_scores = result.get("scores") or []
        scores = [raw_scores[i] if i < len(raw_scores) else None for i in range(len(self.cands))]
        if not any(scores):
            scores = [None] * len(self.cands)
        order = (sorted(range(len(self.cands)), key=lambda i: (i != best, -(scores[i] or 0), i))
                 if ranking_valid else list(range(len(self.cands))))
        for position, index in enumerate(order):
            card = _ReplyCard(self, index, recommended=ranking_valid and index == best,
                              number=position + 1, score=scores[index])
            self.replyBox.addWidget(card)
            self.cards.append(card)
        reply_to = result.get("reply_to")
        self.insightTitle.setText(f"对话参考 · 回复给 {reply_to}" if reply_to else "对话参考")
        answers = result.get("answers") or {}
        self.summary.setText("建议：本轮无需回复" if no_reply else "建议：" + _choice(answers, "best_action"))
        self.intent.setText("可能意图 · " + _choice(answers, "true_intent") +
                            "\n可能需要 · " + _choice(answers, "she_needs"))
        score = (answers.get("danger_level") or {}).get("score")
        valid_score = isinstance(score, (int, float)) and isfinite(score) and 0 <= score <= 9
        self.tension.setText(f"紧张度 {score:.0f}/9" if valid_score else "紧张度待判断")
        color = "#996819" if valid_score and score >= 3 else _MUTED
        if valid_score and score >= 6:
            color = "#b44832"
        qss = f"BodyLabel {{ color: {color}; background: transparent; }}"
        setCustomStyleSheet(self.tension, qss, qss)
        self.empty.setVisible(not self.cands)
        self.insight.setVisible(bool(self.cands) or no_reply)
        self.referenceNote.setVisible((bool(self.cands) or no_reply) and not self._compact)
        self.updated.setText(datetime.now().strftime("%H:%M") + " 更新")
        warnings = result.get("warnings") or []
        if no_reply:
            self.set_status(self._no_reply_reason, "success")
        elif self.cands and not ranking_valid:
            self.set_status("排序不可用，第一条候选将作为默认选项", "warning")
        elif self.cands and result.get("fallback_used"):
            self.set_status("Jev 暂时不可用，本次已由通用判断模型接管", "warning")
        elif self.cands:
            self.set_status("建议已更新，选一句适合你的回复", "success")
        else:
            self.set_status("未生成可用回复，请等待下一条新消息。", "error")
        if warnings:
            self.log("[模型降级] " + "；".join(warnings))

    def _clear_cards(self):
        for card in self.cards:
            self.replyBox.removeWidget(card)
            card.hide()
            card.deleteLater()
        self.cards = []

    def after(self, ms, fn):
        QTimer.singleShot(ms, fn)

    def run(self):
        self.app.exec()
