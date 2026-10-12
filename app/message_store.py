import difflib
import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4096)
def _same_text(left, right):
    return (left == right or difflib.SequenceMatcher(None, left, right).ratio() >= 0.75
            or len(left) == len(right) >= 3 and sum(a != b for a, b in zip(left, right)) <= 1)


def align_messages(history, visible):
    """按顺序对齐气泡，允许 OCR 抖动和发言人名称缺失，不把重复文字当消息 ID。"""
    n, m = len(history), len(visible)
    lengths = [[0] * (m + 1) for _ in range(n + 1)]
    matches = [[old[0] == new[0] and _same_text(old[2], new[2])
                and (not old[1] or not new[1] or old[1] == new[1])
                for new in visible] for old in history]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            lengths[i][j] = (1 + lengths[i + 1][j + 1] if matches[i][j]
                             else max(lengths[i + 1][j], lengths[i][j + 1]))
    aligned = []
    i = j = 0
    while i < n and j < m:
        if lengths[i + 1][j] == lengths[i][j]:
            i += 1
        elif matches[i][j]:
            aligned.append((i, j))
            i += 1
            j += 1
        else:
            j += 1
    return aligned


def new_messages(history, visible):
    aligned = align_messages(history, visible)
    if not aligned:
        return list(visible), False
    if aligned[-1][0] != len(history) - 1:
        return [], True
    return list(visible[aligned[-1][1] + 1:]), True


class MessageStore:
    """采集端写入 SQLite，消费端通过独立连接读取；所有截图仍只保存在内存。"""

    def __init__(self, path, read_only=False):
        source = Path(path).resolve().as_uri() + "?mode=ro" if read_only else path
        self.connection = sqlite3.connect(source, timeout=5, uri=read_only)
        self.connection.row_factory = sqlite3.Row
        if read_only:
            self.connection.execute("PRAGMA query_only=ON")
            return
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                app TEXT NOT NULL,
                title TEXT NOT NULL,
                raw_title TEXT NOT NULL,
                last_read_at REAL NOT NULL,
                UNIQUE(app, title)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                who TEXT NOT NULL CHECK(who IN ('her', 'me')),
                sender TEXT,
                text TEXT NOT NULL,
                observed_at REAL NOT NULL,
                baseline INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_by_conversation
                ON messages(conversation_id, id);
            CREATE TABLE IF NOT EXISTS ocr_snapshots (
                id INTEGER PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                read_at REAL NOT NULL,
                raw_title TEXT NOT NULL,
                area TEXT NOT NULL,
                visible_messages TEXT NOT NULL,
                new_count INTEGER NOT NULL,
                ocr_ms INTEGER NOT NULL,
                title_confirmed INTEGER NOT NULL,
                status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS snapshots_by_conversation
                ON ocr_snapshots(conversation_id, id);
        """)

    def close(self):
        self.connection.close()

    def history(self, app, title, limit=60):
        rows = self.connection.execute("""
            SELECT m.who, m.sender, m.text FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.app = ? AND c.title = ? ORDER BY m.id DESC LIMIT ?
        """, (app, title, limit)).fetchall()
        return [(row["who"], row["sender"], row["text"]) for row in reversed(rows)]

    def conversations(self, app):
        return [row[0] for row in self.connection.execute(
            "SELECT title FROM conversations WHERE app = ? ORDER BY last_read_at", (app,))]

    def record_snapshot(self, app, title, raw_title, visible, area, ocr_ms=0,
                        title_confirmed=True):
        now = time.time()
        previous = self.history(app, title, 500)
        new, anchored = new_messages(previous, visible)
        baseline = not previous or not anchored or not title_confirmed
        if not title_confirmed:
            new = []
        status = "read" if visible and title_confirmed else "empty" if title_confirmed else "unconfirmed"
        encoded = json.dumps(visible, ensure_ascii=False)
        rect = json.dumps(area)
        with self.connection:
            self.connection.execute("""
                INSERT INTO conversations(app, title, raw_title, last_read_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(app, title) DO UPDATE SET raw_title = excluded.raw_title,
                    last_read_at = excluded.last_read_at
            """, (app, title, raw_title, now))
            conversation_id = self.connection.execute(
                "SELECT id FROM conversations WHERE app = ? AND title = ?", (app, title)).fetchone()[0]
            self.connection.executemany("""
                INSERT INTO messages(conversation_id, who, sender, text, observed_at, baseline)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(conversation_id, who, name, text, now, int(baseline)) for who, name, text in new])
            last = self.connection.execute("""
                SELECT id, raw_title, area, visible_messages, title_confirmed, status
                FROM ocr_snapshots WHERE conversation_id = ? ORDER BY id DESC LIMIT 1
            """, (conversation_id,)).fetchone()
            if last and tuple(last)[1:] == (raw_title, rect, encoded, int(title_confirmed), status):
                snapshot_id = last["id"]
                self.connection.execute(
                    "UPDATE ocr_snapshots SET read_at = ?, ocr_ms = ? WHERE id = ?",
                    (now, ocr_ms, snapshot_id))
            else:
                snapshot_id = self.connection.execute("""
                    INSERT INTO ocr_snapshots(conversation_id, read_at, raw_title, area,
                        visible_messages, new_count, ocr_ms, title_confirmed, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (conversation_id, now, raw_title, rect, encoded, len(new), ocr_ms,
                      int(title_confirmed), status)).lastrowid
        return {"id": snapshot_id, "title": title, "raw_title": raw_title,
                "visible": list(visible), "new": new, "history": self.history(app, title),
                "area": tuple(area), "read_at": now, "baseline": baseline,
                "title_confirmed": bool(title_confirmed), "status": status,
                "discontinuous": bool(previous and visible and not anchored)}
