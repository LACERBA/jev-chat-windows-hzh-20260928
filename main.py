# -*- coding: utf-8 -*-
"""父进程：只管界面。截图 + OCR 在 app/worker.py 的子进程里跑，队列里收新消息 →
冒出新的对方消息才调 engine → 悬浮窗给 3 条候选；明确开启的会话可自动发送最佳回复。静默期零调用。
上下文、结果、聊天记录都按会话名（子进程 OCR 头部标题得来）分开存，切会话不串味。

    pip install rapidocr-onnxruntime numpy windows-capture PySide6-Fluent-Widgets
两个模型（判断 Jev / 起草语言模型）的来源和 key 在独立设置页填写，不用改代码。IDE 里直接 Run。
"""
import ctypes
import multiprocessing
import queue
import threading
import time
import traceback
from collections import deque
from math import isfinite

from app import settings, update, worker
from app.capture import find_chat_hwnd
from app.fill import click, fill, fill_and_send
from app.ocr import similar
from app.overlay import Overlay
from app.version import VERSION
from core.engine import analyze

# {会话名: {history, result, rev, target, senders}}：每个会话各自的上下文、上次结果和版本号，互不串味
# history 里是 [(who, text, name)]，engine 只认 her/me，name 是群里的发言人（单聊/自己说的是 None）；
# 只是缓冲区，实际喂模型几条由设置里的「参考上下文」决定
# senders：这个群里发过言的人，去重、最近的排最前；target：用户挑的回复对象（None = 跟着最近那个走）
chats = {}
state = {"area": None, "busy": False, "rerun": None, "hwnd": None, "chat": "", "raw_chat": ""}
results = queue.Queue()
update_result = queue.Queue()  # 独立小队列，别跟 results 的 (kind, r, title, revision) 形状搅在一起
_AUTO_REPLY_COOLDOWN = 8.0
_AUTO_REPLY_ECHO_WINDOW = 12.0
_AUTO_REPLY_SEND_DELAY_MS = 1200


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
        if not self.mode_enabled() or event["id"] in self.processed_set:
            return
        if self.active and self.active.get("id") == event["id"]:
            return
        self.pending[event["id"]] = event
        if len(self.pending) > 50:
            oldest = min(self.pending, key=lambda key: self.pending[key]["at"])
            self.pending.pop(oldest, None)

    def on_chat(self, title, raw_title):
        if self.phase == "switching" and self.active:
            if title == self.active["previous_title"] and raw_title == self.active["previous_raw"]:
                return
            self.active.update(candidate_title=title, candidate_raw=raw_title)
            self._confirm_switch()
            return
        if self.active and self.phase != "listening":
            if title != self.active.get("title") or raw_title != self.active.get("raw"):
                self.release("会话被手动切换，本次自动处理已取消", "warning")

    def route_lines(self, title, raw_title, revision, initial, latest_who):
        if self.phase == "waiting_lines" and self._matches(title, raw_title):
            if not self.active["authorized"]:
                self.release(f"「{raw_title}」未开启自动回复，已跳过分析", "idle")
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
        if self.phase == "switching":
            return "skip"
        if self.phase != "listening" or not settings.auto_switch() or state.get("app") != "wechat":
            return "normal"
        if initial:
            return "normal"
        if not settings.auto_reply_config(raw_title, "wechat")["enabled"]:
            return "skip"
        self.active = {"id": f"current:{raw_title}:{revision}", "title": title, "raw": raw_title,
                       "authorized": True, "revision": revision}
        self.phase = "analyzing"
        return "analyze"

    def mark_result_ready(self, title, revision):
        if self.phase != "analyzing" or not self.active or self.active.get("title") != title:
            return False
        self.phase = "waiting_send"
        self.send_revision = revision
        self.deadline = time.monotonic() + 5.0
        return True

    def result_abandoned(self, title):
        if self.active and self.active.get("title") == title:
            self.release("自动处理未完成，继续监听新消息", "warning")

    def on_send_attempt(self, title, revision, sent):
        if (self.phase != "waiting_send" or not self.active
                or self.active.get("title") != title or self.send_revision != revision):
            return
        if not sent:
            self.release("自动回复未发送，继续监听新消息", "warning")
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
        if self.phase != "listening" or state["busy"]:
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
                    and settings.auto_reply_config(raw_title, "wechat")["enabled"]):
                self.active = {"id": f"current:{raw_title}:{revision}", "title": title,
                               "raw": raw_title, "authorized": True, "revision": revision}
                self.phase = "analyzing"
                if not start_analyze(title, list(chat_of(title)["history"])):
                    self.release("模型尚未配置，已跳过自动分析", "warning")
                return
        candidates = [event for event in self.pending.values() if event["base"] in self.badges]
        if not candidates:
            return
        event = max(candidates, key=lambda item: (item["at"], item["id"]))
        self.pending.pop(event["id"], None)
        target = self.badges[event["base"]]
        self.active = {**event, "previous_title": state["chat"], "previous_raw": state["raw_chat"]}
        self.phase = "switching"
        self.deadline = time.monotonic() + 3.5
        try:
            click(state["hwnd"], target["x"], target["y"])
        except Exception as e:
            ov.log(f"[自动切换失败] {type(e).__name__}: {e}")
            self.release("自动切换会话失败，继续监听新消息", "warning")
            return
        ov.set_status("检测到新消息，正在切换会话", "busy")

    def tick(self):
        if self.phase in ("switching", "waiting_lines", "waiting_send", "verifying"):
            if self.deadline and time.monotonic() >= self.deadline:
                text = ("发送状态未确认，已停止重试" if self.phase == "verifying"
                        else "自动会话处理超时，继续监听新消息")
                self.release(text, "warning")
        self.advance()

    def reset(self):
        self.phase = "listening"
        self.active = None
        self.pending.clear()
        self.badges.clear()
        self.deadline = 0.0
        self.send_revision = None
        self.expected_text = ""
        self.pending_current = None

    def release(self, text=None, kind="idle"):
        if self.active:
            self._remember(self.active.get("id"))
        self.phase = "listening"
        self.active = None
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
        self.active.update(title=title, raw=raw_title,
                           authorized=settings.auto_reply_config(raw_title, "wechat")["enabled"])
        if not self.active["authorized"]:
            self.release(f"已切换到「{raw_title}」，该会话未开启自动回复", "idle")
            return
        self.phase = "waiting_lines"
        self.deadline = time.monotonic() + 4.0
        ov.set_status(f"已切换到「{raw_title}」，正在读取新消息", "busy")

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
    if state.get("app") != "wechat":
        return False
    config = settings.auto_reply_config(title, state.get("app"))
    raw_config = settings.auto_reply_config(state.get("raw_chat", ""), state.get("app"))
    chat = chat_of(title)
    if not config["enabled"] or title != state["chat"] or revision != chat["rev"]:
        return False
    if settings.auto_switch() and not raw_config["enabled"]:
        return False
    if state["hwnd"] is None or state["area"] is None or not capture_on.is_set():
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
    best = result.get("best_index")
    if not isinstance(best, int) or best not in range(len(candidates)):
        if title == ov.current_chat():
            ov.set_status("自动回复已跳过：本次没有选出最佳回复", "warning")
        return False
    scores = result.get("scores") or []
    try:
        score = float(scores[best])
    except (IndexError, TypeError, ValueError):
        score = None
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
            suffix = f"（{score:.0%}）" if score is not None else ""
            ov.set_status(f"已自动发送最佳回复{suffix}", "success")
    return True


def auto_send_and_track(title, result, revision):
    sent = auto_send_reply(title, result, revision, announce=False)
    conversation_controller.on_send_attempt(title, revision, sent)


def spawn_worker():
    """开一个采集子进程，它跟着 capture_on 走：置位=采集，清掉=暂停。"""
    p = multiprocessing.Process(target=worker.run,
                                args=(q, state["hwnd"], capture_on, debug_on, state.get("app")),
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
        results.put(("ok", analyze(msgs, settings.relationship(), context=settings.context(),
                                   model=settings.draft_model() or None,
                                   provider=settings.draft_provider(),
                                   base_url=settings.draft_base_url() or None,
                                   reply_to=reply_to, style=settings.style(),
                                   thinking=settings.thinking(),
                                   jev_provider=settings.jev_provider(),
                                   jev_model=settings.jev_model() or None),
                     title, revision))
    except Exception as e:
        results.put(("err", f"分析失败: {e}", title, revision))


def check_update_bg():
    """启动时后台查一次新版本，跟 analyze_bg 一个套路：网络调用在线程里，UI 只在 tick() 里动。"""
    r = update.check_latest(VERSION)
    if r:
        update_result.put(r)


def start_analyze(title, msgs):
    if not settings.has_jev_key():
        ov.set_status("请先在设置中配置模型", "warning")
        return False
    if not settings.has_llm_key():
        ov.set_status(f"起草来源 {settings.draft_provider_name()} 没填密钥，去设置里补上", "warning")
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
            ov.set_chat(msg[1])
            conversation_controller.on_chat(state["chat"], state["raw_chat"])
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
        _, title, new, area, *meta = msg
        state["area"] = area
        initial = bool(meta and meta[0])
        raw_title = meta[1] if len(meta) > 1 and meta[1] else state.get("raw_chat") or title
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
            chat["history"].append((who, text, name))
            ov.log_message(who, text, name, chat=title)
            if who == "her" and name:  # 群里发过言的人，去重后最近的排最前
                if name in chat["senders"]:
                    chat["senders"].remove(name)
                chat["senders"].insert(0, name)
        ov.set_targets(title, chat["senders"], target_of(title))  # 显不显示这一行由悬浮窗按开关决定
        action = conversation_controller.route_lines(
            title, raw_title, chat["rev"], initial, normalized[-1][0])
        if initial and action != "analyze_fresh":
            chat["auto_blocked_rev"] = chat["rev"]  # 首帧是屏幕旧记录，只展示建议，绝不自动发送
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
            if state["rerun"]:  # 分析期间又来了新消息，接着跑最新的
                (t, msgs), state["rerun"] = state["rerun"], None
                if not start_analyze(t, msgs):
                    conversation_controller.result_abandoned(t)
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
                if conversation_controller.mark_result_ready(title, revision):
                    ov.after(_AUTO_REPLY_SEND_DELAY_MS,
                             lambda t=title, result=r, rev=revision: auto_send_and_track(t, result, rev))
                else:
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
    capture_on = multiprocessing.Event()  # 父子进程共用的开关，置位=采集
    debug_on = multiprocessing.Event()  # 同上，置位=子进程往队列里送整帧给调试窗
    ov = Overlay(on_fill=fill_reply, on_toggle_capture=on_toggle_capture,
                 on_target_change=on_target_change, on_toggle_debug=set_debug,
                 app_key_of=lambda: state.get("app", "wechat"),
                 result_of=lambda t: chats.get(t, {}).get("result"))
    child = dbg = None
    try:
        state["hwnd"], found = find_chat_hwnd()
        state["app"] = found.key
    except RuntimeError:
        ov.set_capture(False, "未找到聊天窗口，打开后再开启采集")
    else:
        capture_on.set()
        child = spawn_worker()
    if settings.debug_view():  # 上次开着就直接开回来
        set_debug(True)
    if not settings.has_jev_key():
        ov.set_status("请先在设置中配置模型", "warning")
        ov.after(0, ov.open_settings)
    if settings.check_update() and update.parse_version(VERSION):  # 开发版没有版本号，不查也不烦源码用户
        threading.Thread(target=check_update_bg, daemon=True).start()
    ov.after(50, tick)
    try:
        ov.run()
    finally:
        if child is not None:
            child.terminate()
