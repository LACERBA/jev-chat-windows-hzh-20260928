# -*- coding: utf-8 -*-
"""父进程：只管界面。截图 + OCR 在 app/worker.py 的子进程里跑，队列里收新消息 →
冒出新的对方消息才调 engine → 悬浮窗给 3 条候选；明确开启的会话可自动发送最佳回复。静默期零调用。
上下文、结果、聊天记录都按会话名（子进程 OCR 头部标题得来）分开存，切会话不串味。

    pip install rapidocr-onnxruntime numpy windows-capture PySide6-Fluent-Widgets
两个模型（判断 Jev / 起草语言模型）的来源和 key 在独立设置页填写，不用改代码。IDE 里直接 Run。
"""
import ctypes
import multiprocessing
import os
import queue
import threading
import time
import traceback
from collections import deque
from math import isfinite

from app import settings, update, worker
from app.capture import find_chat_hwnd, switch_debug
from app.fill import click, fill, fill_and_send
from app.ocr import similar
from app.message_store import MessageStore
from app.overlay import Overlay
from app.version import VERSION
from core.engine import analyze

# {会话名: {history, result, rev, target, senders}}：每个会话各自的上下文、上次结果和版本号，互不串味
# history 里是 [(who, text, name)]，engine 只认 her/me，name 是群里的发言人（单聊/自己说的是 None）；
# 只是缓冲区，实际喂模型几条由设置里的「参考上下文」决定
# senders：这个群里发过言的人，去重、最近的排最前；target：用户挑的回复对象（None = 跟着最近那个走）
chats = {}
state = {"area": None, "busy": False, "rerun": None, "hwnd": None, "chat": "", "raw_chat": "",
         "ocr_confirmed": False, "read_at": 0.0}
results = queue.Queue()
update_result = queue.Queue()  # 独立小队列，别跟 results 的 (kind, r, title, revision) 形状搅在一起
_AUTO_REPLY_COOLDOWN = 8.0
_AUTO_REPLY_ECHO_WINDOW = 12.0
_AUTO_REPLY_SEND_DELAY_MS = 1200
read_requested = None


def request_read():
    if read_requested is not None:
        read_requested.set()


def chat_is_ignored(title, raw_title=""):
    app_key = state.get("app", "wechat")
    return (settings.is_ignored_chat(title, app_key)
            or settings.is_ignored_chat(raw_title, app_key))


class ConversationController:
    def __init__(self):
        self.phase = "listening"
        self.active = None
        self.pending = {}
        self.badges = {}
        self.processed = deque(maxlen=200)
        self.processed_set = set()
        self.deadline = 0.0
        self.send_revision = None
        self.expected_text = ""
        self.pending_current = None
        self.early_lines = None
        self.early_snapshot = None
        self.last_read_request = 0.0

    def mode_enabled(self):
        return (settings.auto_switch() and state.get("app") == "wechat"
                and capture_on.is_set())

    def on_unread_state(self, badges):
        self.badges = {item["base"]: item for item in badges}
        now = time.monotonic()
        self.pending = {key: event for key, event in self.pending.items()
                        if now - event["at"] < 600 and event["base"] in self.badges}
        self._confirm_switch()

    def on_unread(self, event):
        switch_debug("event_received", event_id=event["id"], name=event.get("name"),
                     point=(event.get("x"), event.get("y")), phase=self.phase,
                     enabled=self.mode_enabled(), age=round(time.monotonic() - event["at"], 3))
        if not self.mode_enabled() or event["id"] in self.processed_set:
            return
        if not event.get("name") or chat_is_ignored(event["name"]):
            switch_debug("event_ignored", event_id=event["id"], name=event.get("name"))
            self._remember(event["id"])
            return
        if self.active and self.active.get("id") == event["id"]:
            return
        self.pending[event["id"]] = event
        if len(self.pending) > 50:
            oldest = min(self.pending, key=lambda key: self.pending[key]["at"])
            self.pending.pop(oldest, None)

    def on_chat(self, title, raw_title):
        switch_debug("chat_changed", title=title, raw_title=raw_title, phase=self.phase,
                     expected=self.active.get("name") if self.active else None)
        if chat_is_ignored(title, raw_title):
            if self.active:
                self.release("当前会话已忽略，继续监听其他新消息", "idle")
            return
        if self.phase == "switching" and self.active:
            if title == self.active["previous_title"] and raw_title == self.active["previous_raw"]:
                return
            self.active.update(candidate_title=title, candidate_raw=raw_title)
            self._confirm_switch()
            return
        if self.active and self.phase != "listening":
            if title != self.active.get("title") or raw_title != self.active.get("raw"):
                self.release("会话被手动切换，本次自动处理已取消", "warning")

    def on_snapshot(self, snapshot):
        title, raw_title = snapshot["title"], snapshot["raw_title"]
        if not snapshot["title_confirmed"]:
            return
        if self.phase == "switching" and self._matches_switch_target(title, raw_title):
            self.early_snapshot = snapshot
            return
        if self.phase != "waiting_lines" or not self._matches(title, raw_title):
            return
        visible = snapshot["visible"]
        if not visible:
            return
        if not settings.analysis_enabled():
            self.release(f"已读取「{raw_title}」的消息并保存到本地，仅采集模式", "success")
        elif not snapshot["new"]:
            self.release(f"已读取「{raw_title}」的可见消息，与本地历史一致，暂无新增", "idle")
        elif visible[-1][0] == "me":
            self.release("你已回复，等待对方的新消息", "idle")

    def route_lines(self, title, raw_title, revision, initial, latest_who):
        if not settings.analysis_enabled():
            return "skip"
        if chat_is_ignored(title, raw_title):
            return "skip"
        if self.phase == "send_failed":
            return "skip"
        if self.phase == "switching":
            if self._matches_switch_target(title, raw_title):
                self.early_lines = (title, raw_title, revision, initial, latest_who)
                switch_debug("lines_buffered", title=title, revision=revision,
                             initial=initial, latest_who=latest_who)
            return "skip"
        if self.phase == "waiting_lines" and self._matches(title, raw_title):
            if not self.active["authorized"]:
                self.release(f"「{raw_title}」未开启自动回复，已跳过分析", "idle")
                return "skip"
            if latest_who != "her":
                self.release("你已回复，等待对方的新消息", "idle")
                return "skip"
            self.phase = "analyzing"
            self.deadline = 0.0
            self.active["revision"] = revision
            return "analyze_fresh"
        if self.phase == "analyzing" and self._matches(title, raw_title):
            self.active["revision"] = revision
            return "analyze"
        if self.phase == "waiting_send" and self._matches(title, raw_title) and latest_who == "her":
            self.phase = "analyzing"
            self.send_revision = None
            self.active["revision"] = revision
            return "analyze"
        if self.phase == "verifying" and self._matches(title, raw_title):
            if latest_who == "her":
                self.pending_current = (title, raw_title, revision)
            return "verify"
        if self.phase != "listening" or not settings.auto_switch() or state.get("app") != "wechat":
            return "normal"
        if not settings.auto_reply_config(raw_title, "wechat")["enabled"]:
            return "skip"
        if initial:
            return "normal"
        self.active = {"id": f"current:{raw_title}:{revision}", "title": title, "raw": raw_title,
                       "authorized": True, "revision": revision}
        self.phase = "analyzing"
        return "analyze"

    def mark_result_ready(self, title, revision):
        if self.phase != "analyzing" or not self.active or self.active.get("title") != title:
            return False
        if not settings.send_enabled():
            self.release("回复建议已生成，自动发送已关闭", "success")
            return False
        self.phase = "waiting_send"
        self.send_revision = revision
        self.deadline = time.monotonic() + 5.0
        return True

    def result_abandoned(self, title):
        if self.active and self.active.get("title") == title:
            self.release("自动处理未完成，继续监听新消息", "warning")

    def send_is_pending(self, title, revision):
        return (self.phase == "waiting_send" and self.active is not None
                and self.active.get("title") == title
                and self.active.get("raw") == state.get("raw_chat")
                and title == state.get("chat") and self.send_revision == revision
                and chat_of(title)["rev"] == revision)

    def on_send_attempt(self, title, revision, sent):
        if (self.phase != "waiting_send" or not self.active
                or self.active.get("title") != title or self.send_revision != revision):
            return
        if not sent:
            self.phase = "send_failed"
            self.deadline = 0.0
            ov.set_status("自动回复未完成，已暂停切换；请检查当前会话后再切换", "warning")
            return
        self.phase = "verifying"
        self.expected_text = chat_of(title)["last_auto_sent_text"]
        self.deadline = time.monotonic() + 5.0
        ov.set_status("已执行发送，正在确认消息气泡", "busy")

    def observe(self, title, lines):
        if self.phase != "verifying" or not self.active or self.active.get("title") != title:
            return False
        if not any(who == "me" and similar(self.expected_text, text) for who, _, text in lines):
            return False
        raw_title = self.active.get("raw") or title
        self.release(f"「{raw_title}」自动回复已确认发送", "success")
        return True

    def advance(self):
        if self.phase != "listening" or state["busy"] and settings.analysis_enabled():
            return
        if not self.pending_current and not self.pending:
            return
        if not self.mode_enabled():
            return
        if self.pending_current:
            title, raw_title, revision = self.pending_current
            self.pending_current = None
            if (title == state["chat"] and raw_title == state["raw_chat"]
                    and revision == chat_of(title)["rev"]
                    and not chat_is_ignored(title, raw_title)
                    and settings.auto_reply_config(raw_title, "wechat")["enabled"]):
                self.active = {"id": f"current:{raw_title}:{revision}", "title": title,
                               "raw": raw_title, "authorized": True, "revision": revision}
                self.phase = "analyzing"
                if not start_analyze(title, list(chat_of(title)["history"])):
                    self.release("模型尚未配置，已跳过自动分析", "warning")
                return
        candidates = []
        skipped_current = False
        current_names = {settings.normalize_chat_title(value) for value in
                         (state.get("chat", ""), state.get("raw_chat", "")) if value}
        for event_id, event in list(self.pending.items()):
            target = self.badges.get(event["base"])
            name = target.get("name", "") if target else ""
            if name and settings.normalize_chat_title(name) in current_names:
                self.pending.pop(event_id, None)
                self._remember(event_id)
                skipped_current = True
                switch_debug("current_chat_unread_skipped", name=name, event_id=event_id)
                continue
            if (not name or chat_is_ignored(name) or chat_is_ignored(event.get("name", ""))
                    or settings.normalize_chat_title(name) != settings.normalize_chat_title(event.get("name", ""))):
                self.pending.pop(event_id, None)
                self._remember(event_id)
                continue
            candidates.append(event)
        if not candidates:
            if skipped_current:
                ov.set_status("新消息已在当前会话，无需切换", "idle")
            return
        event = max(candidates, key=lambda item: (item["at"], -self.badges[item["base"]]["y"], item["id"]))
        self.pending.pop(event["id"], None)
        target = self.badges[event["base"]]
        switch_debug("switch_selected", event_id=event["id"], name=target.get("name"),
                     point=(target["x"], target["y"]), current=state["raw_chat"],
                     age=round(time.monotonic() - event["at"], 3))
        self.active = {**event, "previous_title": state["chat"], "previous_raw": state["raw_chat"]}
        self.phase = "switching"
        self.deadline = time.monotonic() + 3.5
        try:
            click(state["hwnd"], target["x"], target["y"])
        except Exception as e:
            switch_debug("switch_failed", name=target.get("name"), error=f"{type(e).__name__}: {e}")
            ov.log(f"[自动切换失败] {type(e).__name__}: {e}")
            self.release("自动切换会话失败，继续监听新消息", "warning")
            return
        ov.set_status("检测到新消息，正在切换会话", "busy")

    def tick(self):
        if self.phase == "waiting_lines" and time.monotonic() - self.last_read_request >= 1.0:
            request_read()
            self.last_read_request = time.monotonic()
        if self.phase in ("switching", "waiting_lines", "waiting_send", "verifying"):
            if self.deadline and time.monotonic() >= self.deadline:
                switch_debug("conversation_timeout", phase=self.phase,
                             title=self.active.get("title") if self.active else None,
                             expected=self.active.get("name") if self.active else None)
                if self.phase in ("waiting_send", "verifying"):
                    self.deadline = 0.0
                    text = ("自动回复仍在等待发送，暂不切换其他会话" if self.phase == "waiting_send"
                            else "发送已执行但未确认，暂停切换；请检查当前会话后再切换")
                    ov.set_status(text, "warning")
                else:
                    text = ("微信会话切换确认超时，继续监听新消息" if self.phase == "switching"
                            else "会话已切换，但 OCR 尚未确认可见文本消息，继续监听")
                    self.release(text, "warning")
        self.advance()

    def buffers_fresh_message(self, title, raw_title, revision):
        return (self.phase == "switching" and self.active is not None and self.early_lines is not None
                and self.early_lines[0] == title and self.early_lines[1] == raw_title
                and self.early_lines[2] == revision and self.early_lines[4] == "her"
                and title == state["chat"] and raw_title == state["raw_chat"]
                and revision == chat_of(title)["rev"]
                and self._matches_switch_target(title, raw_title))

    def reset(self):
        self.phase = "listening"
        self.active = None
        self.pending.clear()
        self.badges.clear()
        self.deadline = 0.0
        self.send_revision = None
        self.expected_text = ""
        self.pending_current = None
        self.early_lines = None
        self.early_snapshot = None

    def release(self, text=None, kind="idle"):
        if self.active:
            self._remember(self.active.get("id"))
        self.phase = "listening"
        self.active = None
        self.early_lines = None
        self.early_snapshot = None
        self.deadline = 0.0
        self.send_revision = None
        self.expected_text = ""
        if text:
            ov.set_status(text, kind)

    def _confirm_switch(self):
        if (self.phase != "switching" or not self.active or "candidate_title" not in self.active
                or self.active["base"] in self.badges):
            return
        title, raw_title = self.active.pop("candidate_title"), self.active.pop("candidate_raw")
        if (chat_is_ignored(title, raw_title)
                or settings.normalize_chat_title(raw_title) != settings.normalize_chat_title(self.active["name"])):
            self.release("切换后的会话与目标不符，已跳过自动处理", "warning")
            return
        self.active.update(title=title, raw=raw_title,
                           authorized=settings.auto_reply_config(raw_title, "wechat")["enabled"])
        if not self.active["authorized"] and settings.analysis_enabled():
            self.release(f"已切换到「{raw_title}」，该会话未开启自动回复", "idle")
            return
        self.phase = "waiting_lines"
        self.deadline = time.monotonic() + 4.0
        request_read()
        self.last_read_request = time.monotonic()
        snapshot, self.early_snapshot = self.early_snapshot, None
        early = self.early_lines
        self.early_lines = None
        if early and settings.analysis_enabled():
            early_title, early_raw, revision, _, latest_who = early
            if (early_title != title or early_raw != raw_title
                    or state["chat"] != title or state["raw_chat"] != raw_title
                    or revision != chat_of(title)["rev"]):
                self.release("切换期间会话或消息已变化，本次自动处理已取消", "warning")
                return
            if latest_who != "her":
                self.release("切换期间你已回复，等待对方的新消息", "idle")
                return
            self.phase = "analyzing"
            self.deadline = 0.0
            self.active["revision"] = revision
            switch_debug("replay_buffered_message", title=title, revision=revision)
            if not start_analyze(title, list(chat_of(title)["history"])):
                self.release("模型尚未配置，已跳过自动分析", "warning")
            return
        if snapshot:
            self.on_snapshot(snapshot)
            if self.phase != "waiting_lines":
                return
        ov.set_status(f"已切换到「{raw_title}」，正在读取并核对本地历史", "busy")

    def _matches_switch_target(self, title, raw_title):
        if not self.active or not self.active.get("name"):
            return False
        expected = settings.normalize_chat_title(self.active["name"])
        return (settings.normalize_chat_title(raw_title) == expected
                and not chat_is_ignored(title, raw_title))

    def _matches(self, title, raw_title):
        return (self.active is not None and self.active.get("title") == title
                and self.active.get("raw") == raw_title)

    def _remember(self, event_id):
        if not event_id or event_id in self.processed_set:
            return
        if len(self.processed) == self.processed.maxlen:
            self.processed_set.discard(self.processed.popleft())
        self.processed.append(event_id)
        self.processed_set.add(event_id)


conversation_controller = ConversationController()


def chat_of(title):
    return chats.setdefault(title, {"history": deque(maxlen=60), "result": None, "rev": 0,
                                    "target": None, "senders": [], "auto_sent_rev": -1,
                                    "auto_blocked_rev": -1, "last_auto_sent_at": 0.0,
                                    "last_auto_sent_text": ""})


def sync_history(title, history):
    chat = chat_of(title)
    messages = [(who, text, name) for who, name, text in history]
    if list(chat["history"]) == messages:
        return
    chat["history"] = deque(messages, maxlen=60)
    chat["senders"] = list(dict.fromkeys(name for who, _, name in reversed(messages)
                                        if who == "her" and name))
    ov.replace_history(title, messages)
    ov.set_targets(title, chat["senders"], target_of(title))


def restore_history():
    if not os.path.exists(settings.message_db_path()):
        return
    store = None
    try:
        store = MessageStore(settings.message_db_path(), read_only=True)
        app_key = state.get("app", "wechat")
        for title in store.conversations(app_key):
            if not chat_is_ignored(title):
                sync_history(title, store.history(app_key, title))
                chat = chat_of(title)
                chat["auto_blocked_rev"] = max(chat["auto_blocked_rev"], chat["rev"])
    except Exception as e:
        ov.set_status(f"本地消息历史读取失败：{type(e).__name__}", "warning")
    finally:
        if store is not None:
            store.close()


def on_processing_changed():
    conversation_controller.release()
    conversation_controller.pending_current = None
    state["rerun"] = None
    ov.set_busy(False)
    for chat in chats.values():
        chat["rev"] += 1
    ov.invalidate_replies()


def target_of(title):
    """这个会话现在的回复对象：用户挑过且人还在就用它，否则用最近说话的那个；单聊没有发言人 → None。"""
    chat = chat_of(title)
    if chat["target"] in chat["senders"]:
        return chat["target"]
    return chat["senders"][0] if chat["senders"] else None


def reply_text(title, text):
    if settings.reply_target() and ov.at_prefix_enabled():
        target = target_of(title)
        if target:
            return f"@{target} " + text  # 纯文本，微信不认成真正的 @，只是让群里看得出在跟谁说
    return text


def fill_reply(text):
    if state["hwnd"] is None:  # 子进程重开过，hwnd 可能换了，用最新的
        raise RuntimeError("未找到聊天窗口，请确认已经打开")
    if state["area"] is None:
        raise RuntimeError("输入区域尚不可用，请确认聊天窗口可见（不要最小化）")
    fill(state["hwnd"], state["area"], reply_text(ov.current_chat(), text))


def auto_send_reply(title, result, revision, announce=True):
    if not settings.send_enabled() or not settings.analysis_enabled():
        return False
    if os.environ.get("JEV_DISABLE_AUTO_SEND") == "1":
        switch_debug("send_disabled_for_debug", title=title)
        return False
    if chat_is_ignored(title, state.get("raw_chat", "")):
        return False
    if state.get("app") != "wechat":
        return False
    config = settings.auto_reply_config(title, state.get("app"))
    raw_config = settings.auto_reply_config(state.get("raw_chat", ""), state.get("app"))
    chat = chat_of(title)
    if not config["enabled"] or title != state["chat"] or revision != chat["rev"]:
        return False
    if settings.auto_switch() and not raw_config["enabled"]:
        return False
    if (state["hwnd"] is None or state["area"] is None or not capture_on.is_set()
            or not state.get("ocr_confirmed", True)):
        return False
    if "read_at" in state and time.time() - state["read_at"] > 2.0:
        request_read()
        if title == ov.current_chat():
            ov.set_status("发送前的当前会话读取尚未确认，本次不自动发送", "warning")
        return False
    if chat["auto_sent_rev"] == revision:
        return False
    if revision <= chat["auto_blocked_rev"]:
        if title == ov.current_chat():
            ov.set_status("启动时已有的聊天记录只生成建议，不自动发送", "warning")
        return False
    elapsed = time.monotonic() - chat["last_auto_sent_at"]
    if elapsed < _AUTO_REPLY_COOLDOWN:
        if title == ov.current_chat():
            ov.set_status(f"自动回复冷却中，已跳过本次发送（{_AUTO_REPLY_COOLDOWN:.0f} 秒保护）", "warning")
        return False
    candidates = result.get("candidates") or []
    if (not result.get("ranking_valid", False)
            and not result.get("auto_send_fallback", False)):
        if title == ov.current_chat():
            ov.set_status("候选未完成有效排序，本次不会自动发送", "warning")
        return False
    best = result.get("best_index")
    if not isinstance(best, int) or best not in range(len(candidates)):
        if title == ov.current_chat():
            ov.set_status("自动回复已跳过：本次没有选出最佳回复", "warning")
        return False
    score = None
    if result.get("ranking_valid"):
        scores = result.get("scores") or []
        try:
            score = float(scores[best])
        except (IndexError, TypeError, ValueError):
            pass
        if score is not None and (not isfinite(score) or not 0 <= score <= 1):
            score = None
    text = reply_text(title, str(candidates[best]).strip())
    if not text:
        return False
    try:
        fill_and_send(state["hwnd"], state["area"], text, config["send_key"])
    except Exception as e:
        if title == ov.current_chat():
            ov.set_status("自动回复发送失败，请确认聊天窗口和发送快捷键设置。", "error")
            ov.log(f"[自动回复失败] {type(e).__name__}: {e}")
        return False
    chat["auto_sent_rev"] = revision
    chat["last_auto_sent_at"] = time.monotonic()
    chat["last_auto_sent_text"] = text
    if title == ov.current_chat():
        ov.invalidate_replies()
        if announce:
            if result.get("ranking_valid"):
                suffix = f"（{score:.0%}）" if score is not None else ""
                ov.set_status(f"已自动发送最佳回复{suffix}", "success")
            else:
                ov.set_status("排序不可用，已自动发送第一条候选回复", "success")
    return True


def auto_send_and_track(title, result, revision):
    if not conversation_controller.send_is_pending(title, revision):
        switch_debug("stale_auto_send_cancelled", title=title, revision=revision,
                     phase=conversation_controller.phase)
        return
    sent = auto_send_reply(title, result, revision, announce=False)
    conversation_controller.on_send_attempt(title, revision, sent)


def spawn_worker():
    """开一个采集子进程，它跟着 capture_on 走：置位=采集，清掉=暂停。"""
    p = multiprocessing.Process(target=worker.run,
                                args=(q, state["hwnd"], capture_on, debug_on, state.get("app"),
                                      read_requested),
                                daemon=True)
    p.start()
    return p


def set_debug(on):
    """调试视图开关：开 → 开窗 + 置位（子进程这才开始送帧，一帧 2~3MB）；关 → 清掉 + 收窗。"""
    global dbg
    if not on:
        debug_on.clear()
        if dbg is not None:
            dbg.hide()
        return
    if dbg is None:
        from app.debugwin import DebugWindow

        dbg = DebugWindow(on_close=on_debug_closed)
    dbg.show()
    debug_on.set()


def on_debug_closed():
    """用户直接关了调试窗 = 把开关也关了，否则设置页显示开着但没窗。"""
    debug_on.clear()
    ov.set_debug_switch(False)
    settings.save(debug_view_on=False)


def on_toggle_capture(on):
    """标题栏开关。启动时没找到聊天软件就没有子进程，这会儿再找一次，找到了才真开得起来。"""
    global child
    if not on:
        capture_on.clear()
        conversation_controller.reset()
        return
    if child is None:
        try:
            state["hwnd"], found = find_chat_hwnd()
            state["app"] = found.key
        except RuntimeError:
            ov.set_capture(False, "未找到聊天窗口，打开后再开启采集")
            return
        child = spawn_worker()
    capture_on.set()


def analyze_bg(msgs, title, revision, reply_to=None):
    """后台线程只跑网络调用，结果丢队列；UI 只在主线程的 tick 里动（Qt 不能跨线程碰）。"""
    try:
        judge = settings.general_judge_config()
        results.put(("ok", analyze(msgs, settings.relationship(), context=settings.context(),
                                   model=settings.draft_model() or None,
                                   provider=settings.draft_provider(),
                                   base_url=settings.draft_base_url() or None,
                                   reply_to=reply_to, style=settings.style(),
                                   thinking=settings.thinking(),
                                   jev_provider=settings.jev_provider(),
                                   jev_model=settings.jev_model() or None,
                                   judge_mode=settings.judge_mode(),
                                   llm_judge_provider=judge["provider"],
                                   llm_judge_model=judge["model"] or None,
                                   llm_judge_base_url=judge["base_url"] or None,
                                   llm_judge_api_key=judge["api_key"]),
                     title, revision))
    except Exception as e:
        results.put(("err", f"分析失败: {e}", title, revision))


def check_update_bg():
    """启动时后台查一次新版本，跟 analyze_bg 一个套路：网络调用在线程里，UI 只在 tick() 里动。"""
    r = update.check_latest(VERSION)
    if r:
        update_result.put(r)


def start_analyze(title, msgs):
    if not settings.analysis_enabled():
        return False
    raw_title = state.get("raw_chat", "") if title == state.get("chat") else ""
    if chat_is_ignored(title, raw_title):
        ov.invalidate_replies()
        ov.set_busy(False)
        ov.set_status("当前会话已忽略，不参与自动处理", "idle")
        return False
    if not settings.models_ready():
        ov.set_status("请先在设置中补全当前判断策略和起草模型配置", "warning")
        return False
    state["busy"] = True
    ov.set_busy(True)
    reply_to = target_of(title) if settings.reply_target() else None  # 开关关着就是今天的行为
    threading.Thread(target=analyze_bg, args=(msgs, title, chat_of(title)["rev"], reply_to),
                     daemon=True).start()
    return True


def on_target_change(title, name):
    """用户挑了回复对象：记下来，这个会话里有对方的话就照新对象重跑一次。"""
    chat = chat_of(title)
    chat["target"] = name
    msgs = list(chat["history"])
    if not any(m[0] == "her" for m in msgs):
        return
    if state["busy"]:
        state["rerun"] = (title, msgs)
        ov.set_busy(True)
    else:
        start_analyze(title, msgs)


def drain():
    """把子进程队列里攒的东西全收掉。"""
    global child
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            return
        kind = msg[0]
        if kind == "area":  # 只是窗口挪了位置，坐标跟着更新，别的什么都不用动
            state["area"] = msg[1]
            continue
        if kind == "unread_state":
            conversation_controller.on_unread_state(msg[1])
            continue
        if kind == "unread":
            conversation_controller.on_unread(msg[1])
            continue
        if kind == "chat":  # 微信切了会话，界面跟过去（用户正浏览别的会话时也跟，微信是准的）
            state["chat"] = msg[1]
            state["raw_chat"] = msg[2] if len(msg) > 2 else msg[1]
            state["ocr_confirmed"] = False
            ov.set_chat(msg[1])
            conversation_controller.on_chat(state["chat"], state["raw_chat"])
            if chat_is_ignored(state["chat"], state["raw_chat"]):
                state["rerun"] = None
                ov.show_cached(None)
                ov.set_busy(False)
                ov.set_status("当前会话已忽略，不参与自动处理", "idle")
            continue
        if kind == "debug":  # 调试视图的一帧；窗口不在就直接丢掉
            if dbg is not None:
                dbg.show_packet(msg[1])
            continue
        if kind == "status":  # 单帧识别失败/报错，提示一下就好，别把正在跑的分析和已知坐标清掉
            ov.set_status(msg[1], "warning")
            ov.log(msg[1])
            continue
        if kind == "paused":  # 子进程确认已暂停
            ov.set_capture(False)
            continue
        if kind == "resumed":  # 子进程重新开始采集
            ov.set_capture(True)
            restore_history()
            continue
        if kind == "dead":  # 采集彻底停了（微信关了之类），这才是真的要清状态
            state["area"] = None
            conversation_controller.reset()
            for c in chats.values():  # 在跑的分析作废，回来的结果不再往界面上贴
                c["rev"] += 1
            state["rerun"] = None
            ov.invalidate_replies()
            ov.set_busy(False)
            ov.set_capture(False, msg[1])
            ov.log(msg[1])
            if child is not None:  # 子进程已经不干活了，收掉引用，下次打开开关重开一个
                child.terminate()
                child.join()
                child = None
            continue
        snapshot = msg[1] if kind == "snapshot" else None
        if snapshot is not None:
            title, raw_title = snapshot["title"], snapshot["raw_title"]
            state["area"] = snapshot["area"]
            if title == state["chat"] and raw_title == state["raw_chat"]:
                state["ocr_confirmed"] = snapshot["title_confirmed"] and bool(snapshot["visible"])
                state["read_at"] = snapshot["read_at"]
            if chat_is_ignored(title, raw_title):
                continue
            if not snapshot["title_confirmed"]:
                ov.set_status("会话标题尚未识别清楚，暂不处理消息", "warning")
                continue
            sync_history(title, snapshot["history"])
            read_status = ("corrected" if snapshot.get("corrected") else "pending"
                           if snapshot.get("discontinuous") else snapshot["status"])
            ov.set_read_state(title, snapshot["read_at"], len(snapshot["new"]), read_status)
            if snapshot.get("corrected"):
                chat = chat_of(title)
                chat["rev"] += 1
                chat["auto_blocked_rev"] = chat["rev"]
                chat["result"] = None
                if state["rerun"] and state["rerun"][0] == title:
                    state["rerun"] = None
                if title == ov.current_chat():
                    ov.show_cached(None)
                    ov.invalidate_replies()
                    ov.set_status("本地消息已按连续 OCR 证据修订，旧回复建议已作废", "success")
                if conversation_controller._matches(title, raw_title):
                    conversation_controller.release()
            if snapshot.get("discontinuous"):
                chat = chat_of(title)
                chat["rev"] += 1
                chat["auto_blocked_rev"] = chat["rev"]
                ov.invalidate_replies()
                if conversation_controller._matches(title, raw_title):
                    conversation_controller.release()
                chat["result"] = None
                if title == ov.current_chat():
                    ov.show_cached(None)
                    ov.set_status("当前画面与历史缺少连续锚点，文本已隔离待确认，不进入回复上下文", "warning")
                continue
            if not snapshot["new"]:
                if snapshot["initial"]:
                    chat = chat_of(title)
                    chat["auto_blocked_rev"] = max(chat["auto_blocked_rev"], chat["rev"])
                conversation_controller.on_snapshot(snapshot)
                continue
            msg = ("lines", title, snapshot["new"], snapshot["area"],
                   snapshot["initial"], raw_title)
        _, title, new, area, *meta = msg
        state["area"] = area
        if not new:
            continue
        initial = bool(meta and meta[0])
        raw_title = meta[1] if len(meta) > 1 and meta[1] else state.get("raw_chat") or title
        if chat_is_ignored(title, raw_title):
            conversation_controller.on_chat(title, raw_title)
            if title == ov.current_chat():
                ov.invalidate_replies()
                ov.set_status("当前会话已忽略，不参与自动处理", "idle")
            continue
        chat = chat_of(title)
        chat["rev"] += 1  # 这个会话有新消息了，它在跑的分析作废
        if title == ov.current_chat():  # 看的是别的会话就别把人家的候选划掉
            ov.invalidate_replies()
        normalized = []
        for who, name, text in new:
            auto_echo = (who == "her" and chat["last_auto_sent_text"]
                         and time.monotonic() - chat["last_auto_sent_at"] <= _AUTO_REPLY_ECHO_WINDOW
                         and similar(chat["last_auto_sent_text"], text))
            if auto_echo:
                who, name = "me", None
            normalized.append((who, name, text))
            if snapshot is None:
                chat["history"].append((who, text, name))
                ov.log_message(who, text, name, chat=title)
            if who == "her" and name:  # 群里发过言的人，去重后最近的排最前
                if name in chat["senders"]:
                    chat["senders"].remove(name)
                chat["senders"].insert(0, name)
        if snapshot is not None and normalized != new:
            history = snapshot["history"][:-len(new)] + normalized
            sync_history(title, history[-60:])
        ov.set_targets(title, chat["senders"], target_of(title))  # 显不显示这一行由悬浮窗按开关决定
        action = conversation_controller.route_lines(
            title, raw_title, chat["rev"], initial, normalized[-1][0])
        if (initial and action != "analyze_fresh"
                and not conversation_controller.buffers_fresh_message(title, raw_title, chat["rev"])):
            chat["auto_blocked_rev"] = chat["rev"]  # 首帧是屏幕旧记录，只展示建议，绝不自动发送
        if snapshot is not None:
            conversation_controller.on_snapshot(snapshot)
        if not settings.analysis_enabled():
            if title == ov.current_chat():
                ov.set_status("消息已保存到本地，生成建议与自动发送均未执行", "success")
            continue
        if conversation_controller.observe(title, normalized):
            state["rerun"] = None
            ov.set_busy(False)
            continue
        if normalized[-1][0] == "her":  # 只有对方最新说话才值得分析
            if action in ("skip", "verify"):
                continue
            msgs = list(chat["history"])
            if state["busy"]:
                state["rerun"] = (title, msgs)
                ov.set_busy(True)
            elif not start_analyze(title, msgs) and action in ("analyze", "analyze_fresh"):
                conversation_controller.result_abandoned(title)
        else:
            state["rerun"] = None
            ov.set_busy(False)
            if action != "skip":
                ov.set_status("你已回复，等待对方的新消息")


def tick():
    try:
        drain()
        while not update_result.empty():
            latest, url = update_result.get()
            ov.set_update(latest, url)
        while not results.empty():
            kind, r, title, revision = results.get()
            state["busy"] = False
            if not settings.analysis_enabled():
                state["rerun"] = None
                ov.set_busy(False)
                continue
            if state["rerun"]:  # 分析期间又来了新消息，接着跑最新的
                (t, msgs), state["rerun"] = state["rerun"], None
                if not start_analyze(t, msgs):
                    conversation_controller.result_abandoned(t)
                continue
            raw_title = state.get("raw_chat", "") if title == state.get("chat") else ""
            if chat_is_ignored(title, raw_title):
                ov.set_busy(False)
                conversation_controller.on_chat(title, raw_title)
                continue
            if revision != chat_of(title)["rev"]:  # 这个会话后来又说话了，这份结果过期了
                ov.set_busy(False)
                conversation_controller.result_abandoned(title)
                continue
            if kind == "ok":
                chat_of(title)["result"] = r  # 先存着；正看着这个会话才立刻贴上去
                if title == ov.current_chat():
                    ov.show(r)
                else:
                    ov.set_busy(False)
                if (not r.get("ranking_valid", False)
                        and not r.get("auto_send_fallback", False)):
                    if (conversation_controller.active
                            and conversation_controller.active.get("title") == title):
                        conversation_controller.release(
                            "候选未完成有效排序", "warning")
                elif conversation_controller.mark_result_ready(title, revision):
                    request_read()
                    ov.after(_AUTO_REPLY_SEND_DELAY_MS,
                             lambda t=title, result=r, rev=revision: auto_send_and_track(t, result, rev))
                elif settings.send_enabled():
                    request_read()
                    ov.after(_AUTO_REPLY_SEND_DELAY_MS,
                             lambda t=title, result=r, rev=revision: auto_send_reply(t, result, rev))
            else:
                ov.set_busy(False)
                ov.set_status("生成失败，请检查网络和服务设置；新消息到来后会重试。", "error")
                ov.log(r)
                conversation_controller.result_abandoned(title)
        conversation_controller.tick()
    except Exception:
        traceback.print_exc()  # 一帧出错不退出
    ov.after(50, tick)


if __name__ == "__main__":  # Windows 的 spawn 会让子进程重新执行本文件，没这行就无限套娃开进程
    multiprocessing.freeze_support()  # 打包成 exe 后 spawn 出来的子进程会重跑一遍 exe，没这行就无限弹界面
    ctypes.windll.user32.SetProcessDPIAware()
    q = multiprocessing.Queue()
    read_requested = multiprocessing.Event()
    capture_on = multiprocessing.Event()  # 父子进程共用的开关，置位=采集
    debug_on = multiprocessing.Event()  # 同上，置位=子进程往队列里送整帧给调试窗
    ov = Overlay(on_fill=fill_reply, on_toggle_capture=on_toggle_capture,
                 on_target_change=on_target_change, on_toggle_debug=set_debug,
                 app_key_of=lambda: state.get("app", "wechat"),
                 result_of=lambda t: chats.get(t, {}).get("result"),
                 on_processing_change=on_processing_changed)
    child = dbg = None
    try:
        state["hwnd"], found = find_chat_hwnd()
        state["app"] = found.key
        switch_debug("main_started", hwnd=state["hwnd"], app=found.key,
                     auto_switch=settings.auto_switch(),
                     auto_send_disabled=os.environ.get("JEV_DISABLE_AUTO_SEND") == "1")
    except RuntimeError:
        ov.set_capture(False, "未找到聊天窗口，打开后再开启采集")
    else:
        capture_on.set()
        child = spawn_worker()
    if settings.debug_view():  # 上次开着就直接开回来
        set_debug(True)
    if settings.analysis_enabled() and not settings.models_ready():
        ov.set_status("请先在设置中补全当前判断策略和起草模型配置；消息采集入库不受影响", "warning")
        ov.after(0, ov.open_settings)
    if settings.check_update() and update.parse_version(VERSION):  # 开发版没有版本号，不查也不烦源码用户
        threading.Thread(target=check_update_bg, daemon=True).start()
    ov.after(50, tick)
    try:
        ov.run()
    finally:
        if child is not None:
            child.terminate()
