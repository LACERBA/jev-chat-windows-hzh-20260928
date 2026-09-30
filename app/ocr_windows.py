# -*- coding: utf-8 -*-
"""Windows.Media.Ocr 后端，用于 RapidOCR 内置模型未覆盖的系统语言。"""
from __future__ import annotations

import asyncio

import numpy as np
from winsdk.windows.globalization import Language
from winsdk.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
from winsdk.windows.media.ocr import OcrEngine
from winsdk.windows.security.cryptography import CryptographicBuffer

GAP = 0.8
SCALE = 2


def available(lang: str | None = None) -> bool:
    try:
        return _engine_for(lang) is not None
    except Exception:
        return False


def _engine_for(lang: str | None):
    if lang:
        return OcrEngine.try_create_from_language(Language(lang))
    return OcrEngine.try_create_from_user_profile_languages()


class WindowsOcr:
    """每个进程复用一个 Windows OCR 引擎和事件循环。"""

    def __init__(self, lang: str | None = None):
        self._engine = _engine_for(lang)
        if self._engine is None:
            raise RuntimeError(
                f"Windows OCR has no recognizer for {lang or 'the user profile languages'}. "
                "Add the language under Settings > Time & language > Language & region.")
        self._loop = asyncio.new_event_loop()

    @property
    def language(self) -> str:
        return self._engine.recognizer_language.language_tag

    def __call__(self, img: np.ndarray, use_cls: bool = False):
        if SCALE > 1:
            img = np.repeat(np.repeat(img, SCALE, axis=0), SCALE, axis=1)
        h, w = img.shape[:2]
        bgra = np.dstack([img[:, :, ::-1], np.full((h, w, 1), 255, np.uint8)])
        buf = CryptographicBuffer.create_from_byte_array(bgra.tobytes())
        bitmap = SoftwareBitmap.create_copy_from_buffer(buf, BitmapPixelFormat.BGRA8, w, h,
                                                        BitmapAlphaMode.PREMULTIPLIED)
        result = self._loop.run_until_complete(_recognize(self._engine, bitmap))
        return [run for line in result.lines for run in _runs(line, SCALE)], None

    def close(self):
        self._loop.close()


async def _recognize(engine, bitmap):
    return await engine.recognize_async(bitmap)


def _runs(line, scale: int = 1):
    """把 Windows OCR 的单词按水平间距拆成与 RapidOCR 兼容的文本段。"""
    words = [(word.bounding_rect, word.text) for word in line.words]
    out, current = [], []
    for rect, text in words:
        if current and rect.x - (current[-1][0].x + current[-1][0].width) > GAP * rect.height:
            out.append(_box(current, scale))
            current = []
        current.append((rect, text))
    if current:
        out.append(_box(current, scale))
    return out


def _box(run, scale: int = 1):
    x0 = min(rect.x for rect, _ in run) / scale
    y0 = min(rect.y for rect, _ in run) / scale
    x1 = max(rect.x + rect.width for rect, _ in run) / scale
    y1 = max(rect.y + rect.height for rect, _ in run) / scale
    text = " ".join(value for _, value in run)
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, 1.0]


if __name__ == "__main__":
    class _Rect:
        def __init__(self, x, width, height=20, y=0):
            self.x, self.y, self.width, self.height = x, y, width, height

    class _Word:
        def __init__(self, rect, text):
            self.bounding_rect, self.text = rect, text

    class _Line:
        def __init__(self, words):
            self.words = words

    line = _Line([_Word(_Rect(0, 40), "배포"), _Word(_Rect(45, 40), "안"),
                  _Word(_Rect(400, 30), "오전")])
    runs = _runs(line)
    assert [text for _, text, _ in runs] == ["배포 안", "오전"], runs
    (box, _, score), = _runs(_Line([_Word(_Rect(10, 40, y=6), "하이")]), scale=2)
    assert box[0] == [5.0, 3.0] and box[2] == [25.0, 13.0] and score == 1.0, box
    print("ocr_windows ok")
