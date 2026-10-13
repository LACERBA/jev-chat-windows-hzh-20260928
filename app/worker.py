# -*- coding: utf-8 -*-
"""子进程：截图 → 定位消息区 → OCR 头部会话名和消息 → 按会话去重，全在这边跑。
一次 OCR 250~800ms，放父进程的 Qt 主线程界面就僵了。
只往队列里丢纯 tuple/str（底色 bg 是 numpy，留在这边不过队列）。帧全程内存，绝不落盘。"""
import ctypes
import time
import traceback

import numpy as np

from app import chatapps, settings
from app.capture import Capture, UnreadTracker, chat_area, switch_debug, unminimize
from app.ocr import Reader, read_session_name, read_title, similar
from app.message_store import MessageStore
from app.media import encode_crop
from core.diagnostics import configure_logging, log_event, redact_text


def _err(q, **fields):
    """异常压成一行发给父进程，子进程的 stderr 一般没人看得见。"""
    log_event("capture.frame_failed", level="error", exc_info=True, **fields)
    q.put(("status", redact_text(" ".join(traceback.format_exc().split())[-200:])))


def _packet(full, area, title, reader, lines, unread):
    """调试视图的一帧：整帧缩到长边 ≤1100 再走队列（原帧 2560 宽裸传要 20MB），只在内存里传，不落盘。
    整数步长切片够用，不引新依赖；框和坐标照发原始值，画的那边按 scale 折算。"""
    k = max(1, -(-max(full.shape[:2]) // 1100))
    small = np.ascontiguousarray(full[::k, ::k])
    return {"w": small.shape[1], "h": small.shape[0], "rgb": small.tobytes(), "scale": k,
            "area": tuple(int(v) for v in area[:4]) if area else None,
            "pane_top": int(area[5]) if area else 0, "title": title,
            "boxes": reader.last_boxes if reader else [],
            "unread": [item["box"] for item in unread],
            "lines": [(w, n, t) for w, n, t, _ in lines],
            "ocr_ms": reader.last_ms if reader else 0, "ts": time.time()}


def run(q, hwnd, enabled, debug_on, app_key=None, read_requested=None, log_queue=None):
    """enabled 置位=采集，清掉=暂停。暂停时停掉 WGC 会话（Windows 那圈黄色采集边框也跟着没了），
    恢复时重开一个；readers 一直留着，去重状态不丢，恢复后不会把屏幕上的旧消息再报一遍。
    debug_on 置位才往队列里送整帧（一帧 2~3MB），关着一点额外活都不干。"""
    if log_queue is not None:
        configure_logging(log_queue, worker=True)
    ctypes.windll.user32.SetProcessDPIAware()
    app = chatapps.get(app_key)
    switch_debug("worker_started", hwnd=hwnd, app=app.key)
    cap = None
    try:
        store = MessageStore(settings.message_db_path())
    except Exception:
        log_event("storage.open_failed", level="error", exc_info=True, app=app.key)
        q.put(("dead", "本地消息数据库无法打开，请检查程序目录的写入权限"))
        enabled.clear()
        return
    title_confirmed = False
    readers = {}  # {会话名: Reader}，一个会话一套去重状态
    title, raw_title, head = "", "", None  # 当前会话名 / 上一帧的头部像素
    unread_tracker = UnreadTracker(read_name=lambda full, area, badge: read_session_name(full, area, badge, app))
    last_area = None  # 上次发给父进程的 4 元组，变了才再发一次
    last_chat = None
    dispatched_media = set()
    media_config = None
    warned = False  # 消息区识别失败是否已经报过，拖窗口时别每帧刷一条
    while True:
        if not enabled.is_set():
            if cap is not None:
                cap.stop()
                cap = None
                log_event("capture.paused", app=app.key)
                q.put(("paused",))
            enabled.wait()
            continue
        if cap is None:
            try:
                cap = Capture(hwnd)
            except Exception as e:
                log_event("capture.start_failed", level="error", exc_info=True, hwnd=hwnd)
                q.put(("dead", "无法开始采集：" + (redact_text(" ".join(str(e).split())[:120]) or type(e).__name__)))
                enabled.clear()  # 自己清掉，下一圈就去等着，别一秒重试几十次
                continue
            head = last_chat = None
            dispatched_media.clear()
            log_event("capture.resumed", app=app.key, hwnd=hwnd)
            q.put(("resumed",))
        if not cap.alive():
            break
        try:
            unminimize(hwnd)
            requested = read_requested is not None and read_requested.is_set()
            full = cap.snapshot() if requested else cap.settled()
            if full is not None:
                if requested:
                    read_requested.clear()
                reader, lines, unread = None, [], []  # 调试视图要用，消息区没认出来时就是空的
                area = chat_area(full)  # 每次停稳都重算：拖完窗口微信布局会晚一拍才铺好，只按尺寸变化算一次会锁死
                if area is None:
                    if not warned:
                        log_event("capture.area_unrecognized", level="warning", frame_shape=full.shape)
                        q.put(("status", "消息区认不出来（窗口太小？）"))
                        warned = True
                else:
                    warned = False
                    cap.area = area  # 采集线程拿它做 diff
                    x0, y0, x1, y1, bg, y_pane = area
                    rect = (x0, y0, x1, y1)
                    if rect != last_area:
                        q.put(("area", rect))
                        last_area, last_chat = rect, None
                        unread_tracker = UnreadTracker(read_name=lambda full, area, badge: read_session_name(full, area, badge, app))
                    if app.key == "wechat":
                        unread_tracker.scale = (ctypes.windll.user32.GetDpiForWindow(hwnd) or 96) / 96
                        unread, events = unread_tracker.update(full, area)
                        q.put(("unread_state", unread))
                        for event in events:
                            q.put(("unread", event))
                    crop = full[y_pane:y0, x0:x1]  # 头部：会话名在这里
                    if requested or head is None or not np.array_equal(crop, head):  # 名字没动就别白跑一次 OCR
                        head = crop
                        raw_name = read_title(crop, app)
                        title_confirmed = bool(raw_name)
                        if not title_confirmed:
                            head = None
                        switch_debug("header_title", raw_title=raw_name,
                                     area=rect, frame_shape=full.shape)
                        # OCR 抖一下（「小分队」↔「小分认」）不能分裂出一个新会话
                        name = (raw_name if settings.is_ignored_chat(raw_name, app.key)
                                else next((k for k in readers if similar(k, raw_name)), raw_name)) if raw_name else ""
                        # ponytail: 认不出就沿用上次；开头就认不出给个占位名，总比把消息全丢了强
                        name = name or title or "当前会话"
                        next_raw = raw_name or raw_title or name
                        if name != title or next_raw != raw_title:
                            title, raw_title = name, next_raw
                            last_chat = None
                            q.put(("chat", title, raw_title))
                    if (settings.is_ignored_chat(raw_title, app.key)
                            or settings.is_ignored_chat(title, app.key)):
                        if debug_on.is_set():
                            q.put(("debug", _packet(full, area, title, reader, lines, unread)))
                        continue
                    initial = title not in readers
                    reader = readers.setdefault(title, Reader(app))
                    chat_pixels = full[y0:y1, x0:x1]
                    if requested or last_chat is None or not np.array_equal(chat_pixels, last_chat):
                        lines = reader.read(chat_pixels, bg)
                        visible = [(who, name, text) for who, name, text, _ in lines]
                        latest = cap.latest
                        if latest is not None and not np.array_equal(crop, latest[y_pane:y0, x0:x1]):
                            log_event("ocr.discarded", level="warning", chat=title,
                                      reason="识别期间会话标题变化，重新读取")
                            head = last_chat = None
                            if read_requested is not None:
                                read_requested.set()
                            continue
                        media = getattr(reader, "last_media", [])
                        media = media if isinstance(media, list) else []
                        config = (settings.media_mode(), settings.vision_upload(), settings.analysis_enabled())
                        if config != media_config:
                            dispatched_media.clear()
                            media_config = config
                        vision = (config[1] and config[0] != "ocr" and config[2]
                                  and (not settings.auto_switch() or settings.auto_reply_config(raw_title, app.key)["enabled"]))
                        snapshot = store.record_snapshot(app.key, title, raw_title, visible, rect,
                                                         reader.last_ms, title_confirmed, media=media, vision=vision)
                        snapshot["media_jobs"] = [job for job in snapshot["media_jobs"]
                                                  if job["message_id"] not in dispatched_media]
                        for job in snapshot["media_jobs"]:
                            job["image"] = encode_crop(job.pop("pixels"))
                            job.pop("rect", None)
                        for item in media:
                            item.pop("pixels", None)
                        snapshot["initial"] = initial or snapshot["baseline"]
                        log_event("snapshot.saved", app=app.key, chat=title, raw_chat=raw_title,
                                  title_confirmed=title_confirmed, ocr_ms=reader.last_ms,
                                  visible_count=len(visible), new_count=len(snapshot["new"]),
                                  latest_id=snapshot.get("latest_id"), status=snapshot["status"],
                                  corrected=snapshot.get("corrected", False),
                                  discontinuous=snapshot.get("discontinuous", False),
                                  media_job_count=len(snapshot["media_jobs"]), requested=requested)
                        q.put(("snapshot", snapshot))
                        dispatched_media.update(job["message_id"] for job in snapshot["media_jobs"])
                        last_chat = chat_pixels if visible and title_confirmed else None
                if debug_on.is_set():
                    q.put(("debug", _packet(full, area, title, reader, lines, unread)))
        except Exception:
            _err(q, chat=title, raw_chat=raw_title, app=app.key)  # 一帧出错不退出
        time.sleep(0.05)
    store.close()
    log_event("capture.stopped", level="warning", app=app.key, hwnd=hwnd)
    q.put(("dead", "采集停了（聊天窗口关了？）"))
    try:
        cap.wait()  # 采集线程若是报错死的，这里把错抛出来
    except Exception:
        _err(q)
