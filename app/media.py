import hashlib
import re
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


LABELS = {"image": "图片", "video_cover": "视频封面", "emoji": "表情", "mixed": "文字和表情"}
TOKEN = re.compile(r"^\[\[media:([0-9a-f]{32})\]\]$")


def media_token(fingerprint):
    return f"[[media:{fingerprint}]]"


def describe_media(media):
    label = LABELS.get(media["kind"], "媒体")
    parts = [f"[{label}消息，机器识别内容，不是对方的文字原话]"]
    if media.get("ocr_text"):
        parts.append("可见文字：" + media["ocr_text"])
    if media.get("summary"):
        parts.append("画面描述：" + media["summary"])
    if media.get("uncertainty"):
        parts.append("不确定信息：" + media["uncertainty"])
    if media["kind"] == "video_cover":
        parts.append("范围：仅封面，未读取视频正文")
    if media.get("status") == "pending":
        parts.append("状态：视觉识别中")
    elif not media.get("summary"):
        parts.append("限制：仅提取文字，未理解画面；无文字不代表图片没有内容")
    return "\n".join(parts)


def encode_crop(pixels):
    """目标消息在内存中编码为 PNG，不保存原图，也不扩大裁剪范围。"""
    image = Image.fromarray(pixels)
    image.thumbnail((1600, 1600))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _fingerprint(pixels):
    inset = max(1, min(pixels.shape[:2]) // 40)
    inner = pixels[inset:-inset, inset:-inset] if min(pixels.shape[:2]) > 4 else pixels
    gray = cv2.resize(cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = int("".join("1" if bit else "0" for bit in (gray[:, 1:] > gray[:, :-1]).flat), 2)
    colors = cv2.resize(inner, (2, 2), interpolation=cv2.INTER_AREA) // 8
    tail = 0
    for value in colors.flat:
        tail = (tail << 5) | int(value)
    tail = (tail << 4) | min(15, round(pixels.shape[1] / pixels.shape[0] * 4))
    return f"{bits:016x}{tail:016x}"


def same_media(left, right):
    """感知指纹允许缩放造成的少量抖动，但同时约束颜色与长宽比。"""
    if left == right:
        return True
    if (int(left[:16], 16) ^ int(right[:16], 16)).bit_count() > 3:
        return False
    a, b = int(left[16:], 16), int(right[16:], 16)
    if a & 15 != b & 15:
        return False
    a, b = a >> 4, b >> 4
    return all(abs(((a >> shift) & 31) - ((b >> shift) & 31)) <= 1 for shift in range(0, 60, 5))


def _runs(values):
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return list(zip(changes[::2], changes[1::2]))


def _play_icon(pixels):
    h, w = pixels.shape[:2]
    if min(h, w) < 100:
        return False
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min(h, w) // 2,
                              param1=100, param2=24, minRadius=max(10, min(h, w) // 20),
                              maxRadius=min(h, w) // 6)
    if circles is None:
        return False
    yy, xx = np.ogrid[:h, :w]
    white = pixels.min(axis=2) > 215
    for x, y, radius in circles[0]:
        if abs(x - w / 2) >= w * 0.18 or abs(y - h / 2) >= h * 0.25:
            continue
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        ring = np.abs(distance - radius) < max(2, radius * 0.09)
        if white[ring].mean() < 0.25:
            continue
        inner = (white & (distance < radius * 0.65)).astype(np.uint8)
        contours, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            polygon = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            _, _, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if len(polygon) in (3, 4) and area > radius ** 2 * 0.2 and area < width * height * 0.75:
                return True
    return False


class MediaReader:
    """用头像锚点约束消息边界；没有可靠边界时不把邻近区域送给视觉模型。"""

    def __init__(self):
        self.previous = []
        self.previous_context = None

    def read(self, chat, pane_bg, lines, boxes, ocr):
        h, w = chat.shape[:2]
        mask = (np.abs(chat.astype(np.int16) - pane_bg).sum(axis=2) > 12).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        context = chat.copy()
        for old in self.previous:
            x0, y0, x1, y1 = old["rect"]
            context[y0:y1, x0:x1] = 0
        context_key = hashlib.sha256(context.tobytes()).digest()
        scale = max(0.7, min(2.5, w / 950))
        avatars = []
        for x, y, bw, bh, size in stats[1:]:
            if (20 * scale <= bw <= 90 * scale and 20 * scale <= bh <= 90 * scale
                    and 0.7 <= bw / bh <= 1.4 and size > bw * bh * 0.35
                    and (x + bw < w / 2 and x <= bw * 0.85
                         or x > w / 2 and w - x - bw <= bw * 0.85)):
                avatars.append((int(x), int(y), int(bw), int(bh)))
        avatars.sort(key=lambda rect: rect[1])
        media = []
        output = list(lines)
        for index, (ax, ay, aw, ah) in enumerate(avatars):
            who = "her" if ax < w / 2 else "me"
            end = avatars[index + 1][1] - 4 if index + 1 < len(avatars) else h
            left = ax + aw + max(3, aw // 6) if who == "her" else max(2, int(w * 0.09))
            right = w - max(2, int(w * 0.09)) if who == "her" else ax - max(3, aw // 6)
            if end <= ay or right <= left:
                continue
            band = mask[ay:end, left:right]
            active = band.sum(axis=1) >= max(3, aw // 8)
            for top, bottom in _runs(active):
                if bottom - top < max(18, ah // 2):
                    continue
                ys, xs = np.nonzero(band[top:bottom])
                x0, x1 = left + int(xs.min()), left + int(xs.max()) + 1
                y0, y1 = ay + int(top), ay + int(bottom)
                if (who == "her" and x0 - left > aw or who == "me" and right - x1 > aw
                        or y0 == 0 or y1 >= h - 1 or x1 - x0 < aw // 2):
                    continue
                normal = [line for line in output if line[0] == who and y0 - 3 <= line[3] < y1]
                contained = [box for box in boxes if x0 <= (box[0] + box[2]) / 2 < x1
                             and y0 <= (box[1] + box[3]) / 2 < y1]
                text = "".join(box[5] for box in contained if box[4] not in ("name", "gray"))
                if re.search("转账|红包|收款|微信支付", text):
                    continue
                pixels = chat[y0:y1, x0:x1]
                samples = pixels[::2, ::2].reshape(-1, 3)
                colors, amounts = np.unique(samples // 8, axis=0, return_counts=True)
                dominant = colors[amounts.argmax()].astype(int) * 8
                colorful = ((pixels.max(axis=2).astype(int) - pixels.min(axis=2)) > 40)
                colorful &= np.abs(pixels.astype(int) - dominant).sum(axis=2) > 60
                if dominant.max() - dominant.min() > 40:
                    hue = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)[:, :, 0].astype(int)
                    background_hue = int(cv2.cvtColor(dominant.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0, 0])
                    difference = np.abs(hue - background_hue)
                    colorful &= np.minimum(difference, 180 - difference) > 12
                has_color = colorful.sum() >= max(20, pixels.shape[0] * pixels.shape[1] * 0.005)
                filtered = any(box[4] in ("image", "tiny") for box in contained)
                textured = amounts.max() / len(samples) < 0.55
                large = y1 - y0 > ah * 2
                if not (has_color or filtered and large or textured and y1 - y0 > ah * 1.5
                        or not normal and y1 - y0 >= ah and (not text or large)):
                    continue
                names = [box[5] for box in boxes if box[4] == "name" and ax < box[0] < left + aw
                         and ay - ah <= box[1] < y0]
                name = names[-1] if who == "her" and names else None
                if normal and not filtered and y1 - y0 < ah * 2:
                    kind = "mixed"
                elif y1 - y0 < ah * 2 and x1 - x0 < aw * 3:
                    kind = "emoji"
                else:
                    kind = "video_cover" if "视频号" in text or _play_icon(pixels) else "image"
                fingerprint = _fingerprint(pixels)
                rect = (x0, y0, x1, y1)
                for old in self.previous:
                    if old["who"] != who or old["rect"] != rect:
                        continue
                    if context_key == self.previous_context:
                        fingerprint = old["fingerprint"]
                        break
                cached = next((old for old in self.previous if same_media(old["fingerprint"], fingerprint)), None)
                if cached:
                    fingerprint, ocr_text, kind = cached["fingerprint"], cached["ocr_text"], cached["kind"]
                else:
                    result, _ = ocr(pixels, use_cls=False)
                    ocr_text = "\n".join(item[1] for item in sorted(result or [], key=lambda item:
                                         (item[0][0][1], item[0][0][0])) if item[2] >= 0.6)[:6000]
                if re.search("转账|红包|收款|微信支付", ocr_text):
                    continue
                item = {"who": who, "sender": name, "kind": kind, "fingerprint": fingerprint,
                        "rect": rect, "ocr_text": ocr_text, "pixels": pixels.copy()}
                media.append(item)
                output = [line for line in output if line not in normal]
                output.append((who, name, media_token(fingerprint), y0))
        self.previous = [{key: value for key, value in item.items() if key != "pixels"} for item in media]
        context = chat.copy()
        for item in media:
            x0, y0, x1, y1 = item["rect"]
            context[y0:y1, x0:x1] = 0
        self.previous_context = hashlib.sha256(context.tobytes()).digest() if media else None
        return sorted(output, key=lambda line: line[3]), media
