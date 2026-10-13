import difflib
import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

from app.media import TOKEN, describe_media, media_token, same_media


@lru_cache(maxsize=4096)
def _same_text(left, right):
    media_left, media_right = TOKEN.fullmatch(left), TOKEN.fullmatch(right)
    if media_left or media_right:
        return bool(media_left and media_right and same_media(media_left[1], media_right[1]))
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
            self.has_states = bool(self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'message_states'").fetchone())
            self.has_media = bool(self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'message_media'").fetchone())
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
            CREATE TABLE IF NOT EXISTS message_states (
                message_id INTEGER PRIMARY KEY REFERENCES messages(id),
                status TEXT NOT NULL,
                who TEXT NOT NULL,
                sender TEXT,
                text TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS message_revisions (
                id INTEGER PRIMARY KEY,
                message_id INTEGER NOT NULL REFERENCES messages(id),
                snapshot_id INTEGER REFERENCES ocr_snapshots(id),
                previous_value TEXT NOT NULL,
                revised_value TEXT NOT NULL,
                reason TEXT NOT NULL,
                revised_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS correction_candidates (
                message_id INTEGER PRIMARY KEY REFERENCES messages(id),
                value TEXT NOT NULL,
                observations INTEGER NOT NULL,
                seen_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS message_media (
                message_id INTEGER PRIMARY KEY REFERENCES messages(id),
                fingerprint TEXT NOT NULL,
                kind TEXT NOT NULL,
                ocr_text TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                uncertainty TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'ocr',
                status TEXT NOT NULL DEFAULT 'ocr',
                updated_at REAL NOT NULL
            );
            CREATE VIEW IF NOT EXISTS effective_messages AS
                SELECT m.id, m.conversation_id, COALESCE(s.who, m.who) AS who,
                    CASE WHEN s.message_id IS NULL THEN m.sender ELSE s.sender END AS sender,
                    COALESCE(s.text, m.text) AS text, m.observed_at, m.baseline
                FROM messages m LEFT JOIN message_states s ON s.message_id = m.id
                WHERE COALESCE(s.status, 'active') = 'active';
        """)
        self.has_states = True
        self.has_media = True
        self._quarantine_legacy()

    def close(self):
        self.connection.close()

    def _history_rows(self, app, title, limit=500):
        fields = ("COALESCE(s.who, m.who) AS who, CASE WHEN s.message_id IS NULL THEN m.sender "
                  "ELSE s.sender END AS sender, COALESCE(s.text, m.text) AS text"
                  if self.has_states else "m.who, m.sender, m.text")
        join = "LEFT JOIN message_states s ON s.message_id = m.id" if self.has_states else ""
        status = "AND COALESCE(s.status, 'active') = 'active'" if self.has_states else ""
        rows = self.connection.execute(f"""
            SELECT m.id, {fields} FROM messages m
            JOIN conversations c ON c.id = m.conversation_id {join}
            WHERE c.app = ? AND c.title = ? {status} ORDER BY m.id DESC LIMIT ?
        """, (app, title, limit)).fetchall()
        return list(reversed(rows))

    def history(self, app, title, limit=60):
        return self.context(app, title, limit)["history"]

    def context(self, app, title, limit=60):
        """文本和 ID 基于同一次查询，避免并发采集时将媒体描述拼到其他消息上。"""
        rows = self._history_rows(app, title, limit)
        media = self._media_rows(rows)
        history = [(row["who"], row["sender"], describe_media(media[row["id"]])
                    if row["id"] in media else row["text"]) for row in rows]
        return {"history": history, "history_ids": [row["id"] for row in rows], "media_state": media}

    def history_ids(self, app, title, limit=60):
        return [row["id"] for row in self._history_rows(app, title, limit)]

    def media_state(self, app, title, limit=60):
        return self._media_rows(self._history_rows(app, title, limit))

    def _media_rows(self, messages):
        if not self.has_media or not messages:
            return {}
        ids = [row["id"] for row in messages]
        roles = {row["id"]: row["who"] for row in messages}
        marks = ",".join("?" for _ in ids)
        return {row["message_id"]: {**dict(row), "who": roles[row["message_id"]]} for row in self.connection.execute(
            f"SELECT * FROM message_media WHERE message_id IN ({marks})", ids)}

    def update_media(self, app, title, message_id, fingerprint, result):
        """异步识别只更新原消息；身份和状态校验防止过期结果覆盖新的识别。"""
        with self.connection:
            changed = self.connection.execute("""
                UPDATE message_media SET summary = ?, uncertainty = ?, source = ?, status = ?, updated_at = ?
                WHERE message_id = ? AND fingerprint = ? AND status = 'pending'
                AND message_id IN (SELECT m.id FROM messages m JOIN conversations c
                    ON c.id = m.conversation_id WHERE c.app = ? AND c.title = ?)
            """, (result.get("summary", ""), result.get("uncertainty", ""), result.get("source", "ocr"),
                  result.get("status", "failed"), time.time(), message_id, fingerprint, app, title)).rowcount
        return bool(changed)

    def _revise(self, row, value, status, reason, snapshot_id=None):
        now = time.time()
        previous = self.connection.execute(
            "SELECT status, who, sender, text FROM message_states WHERE message_id = ?", (row["id"],)
        ).fetchone()
        old = tuple(previous) if previous else ("active", row["who"], row["sender"], row["text"])
        revised = (status, *value)
        if old == revised:
            return False
        self.connection.execute("""
            INSERT INTO message_revisions(message_id, snapshot_id, previous_value, revised_value, reason, revised_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row["id"], snapshot_id, json.dumps(old, ensure_ascii=False),
              json.dumps(revised, ensure_ascii=False), reason, now))
        self.connection.execute("""
            INSERT INTO message_states(message_id, status, who, sender, text, updated_at) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET status = excluded.status, who = excluded.who,
                sender = excluded.sender, text = excluded.text, updated_at = excluded.updated_at
        """, (row["id"], *revised, now))
        return True

    def _quarantine_legacy(self):
        """把旧版已判定为不连续的基线批次隔离，原始消息及快照不删除。"""
        conversations = self.connection.execute("SELECT id, app, title FROM conversations").fetchall()
        repair = []
        with self.connection:
            for conversation in conversations:
                rows = self.connection.execute("""
                    SELECT m.* FROM messages m WHERE m.conversation_id = ?
                    AND NOT EXISTS (SELECT 1 FROM message_states s WHERE s.message_id = m.id)
                    ORDER BY m.id
                """, (conversation["id"],)).fetchall()
                trusted = []
                batches = {}
                for row in rows:
                    batches.setdefault(row["observed_at"], []).append(row)
                for batch in batches.values():
                    values = [(row["who"], row["sender"], row["text"]) for row in batch]
                    pending = bool(trusted and batch[0]["baseline"] and not align_messages(trusted, values))
                    source_id = None
                    if pending:
                        if conversation not in repair:
                            repair.append(conversation)
                        snapshots = self.connection.execute(
                            "SELECT id, visible_messages FROM ocr_snapshots WHERE conversation_id = ? ORDER BY id",
                            (conversation["id"],)).fetchall()
                        source_id = next((snapshot["id"] for snapshot in snapshots
                                          if all(list(value) in json.loads(snapshot["visible_messages"])
                                                 for value in values)), None)
                    for row, value in zip(batch, values):
                        if pending:
                            self._revise(row, value, "pending", "legacy_unanchored_batch", source_id)
                        else:
                            self.connection.execute("""
                                INSERT INTO message_states(message_id, status, who, sender, text, updated_at)
                                VALUES (?, 'active', ?, ?, ?, ?)
                            """, (row["id"], *value, time.time()))
                    if not pending:
                        trusted.extend(values)
        for conversation in repair:
            latest = self.connection.execute("""
                SELECT * FROM ocr_snapshots WHERE conversation_id = ? AND title_confirmed = 1
                ORDER BY id DESC LIMIT 1
            """, (conversation["id"],)).fetchone()
            if latest:
                self.record_snapshot(conversation["app"], conversation["title"], latest["raw_title"],
                                     [tuple(value) for value in json.loads(latest["visible_messages"])],
                                     json.loads(latest["area"]), latest["ocr_ms"],
                                     observed_at=latest["read_at"])

    def _correct_observations(self, rows, visible, snapshot_id):
        previous = [(row["who"], row["sender"], row["text"]) for row in rows]
        aligned = align_messages(previous, visible)
        exact = {(i, j) for i, j in aligned if previous[i] == tuple(visible[j])}
        corrected = False
        for i, j in aligned:
            value = tuple(visible[j])
            if TOKEN.fullmatch(previous[i][2]):
                continue
            if previous[i] == value:
                self.connection.execute(
                    "UPDATE correction_candidates SET observations = 0 WHERE message_id = ?", (rows[i]["id"],))
                continue
            if (i - 1, j - 1) not in exact or (i + 1, j + 1) not in exact:
                continue
            encoded = json.dumps(value, ensure_ascii=False)
            candidate = self.connection.execute(
                "SELECT value, observations, seen_at FROM correction_candidates WHERE message_id = ?",
                (rows[i]["id"],)).fetchone()
            now = time.time()
            count = (candidate["observations"] + 1 if candidate and candidate["value"] == encoded
                     and now - candidate["seen_at"] <= 15 else 1)
            self.connection.execute("""
                INSERT INTO correction_candidates(message_id, value, observations, seen_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET value = excluded.value,
                    observations = excluded.observations, seen_at = excluded.seen_at
            """, (rows[i]["id"], encoded, count, now))
            if count >= 2:
                corrected |= self._revise(rows[i], value, "active", "stable_ocr_between_anchors", snapshot_id)
        return corrected

    def conversations(self, app):
        return [row[0] for row in self.connection.execute(
            "SELECT title FROM conversations WHERE app = ? ORDER BY last_read_at", (app,))]

    @staticmethod
    def _stable_unanchored(previous, last, visible):
        """上一帧同样对不上已确认的历史，并且与当前帧互相完全对齐，说明画面稳定停在新位置。"""
        earlier = [tuple(value) for value in json.loads(last["visible_messages"])]
        if not earlier or new_messages(previous, earlier)[1]:
            return False
        aligned = align_messages(earlier, visible)
        smaller = min(len(earlier), len(visible))
        if len(aligned) == smaller:  # 一帧是另一帧的子集（OCR 漏识别了某个气泡）
            return True
        # 顶部被裁切的气泡每帧指纹都会变：长度相同、位置不动，只允许一个气泡对不上；滚动会让位置错开，不算稳定
        return (len(earlier) == len(visible) >= 3 and len(aligned) >= smaller - 1
                and all(i == j for i, j in aligned))

    def record_snapshot(self, app, title, raw_title, visible, area, ocr_ms=0,
                        title_confirmed=True, observed_at=None, media=None, vision=False):
        now = time.time() if observed_at is None else float(observed_at)
        rows = self._history_rows(app, title)
        previous = [(row["who"], row["sender"], row["text"]) for row in rows]
        new, anchored = new_messages(previous, visible)
        discontinuous = bool(previous and visible and not anchored)
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
            last = self.connection.execute("""
                SELECT id, raw_title, area, visible_messages, title_confirmed, status
                FROM ocr_snapshots WHERE conversation_id = ? ORDER BY id DESC LIMIT 1
            """, (conversation_id,)).fetchone()
            stable = bool(discontinuous and title_confirmed and last and last["title_confirmed"]
                          and self._stable_unanchored(previous, last, visible))
            if stable:  # 画面连续两帧稳定停在没有锚点的新位置：把它当作新的起点，不再一直隔离
                discontinuous = False
                new = list(visible)
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
            if stable:  # 旧历史标记为已被取代（原文和审计记录都保留），回到底部时才能重新对齐、不吞新消息
                for row, old in zip(rows, previous):
                    self._revise(row, old, "superseded", "rebaseline_stable_unanchored", snapshot_id)
            duplicate_pending = discontinuous and last and last["visible_messages"] == encoded
            if not duplicate_pending:
                for value in new:
                    message_id = self.connection.execute("""
                        INSERT INTO messages(conversation_id, who, sender, text, observed_at, baseline)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (conversation_id, *value, now, int(baseline))).lastrowid
                    original = {"id": message_id, "who": value[0], "sender": value[1], "text": value[2]}
                    if discontinuous:
                        self._revise(original, value, "pending", "unanchored_batch", snapshot_id)
            corrected = (self._correct_observations(rows, visible, snapshot_id)
                         if title_confirmed and observed_at is None else False)
            jobs = []
            effective = self._history_rows(app, title)
            if media and title_confirmed and not discontinuous:
                aligned = dict((j, effective[i]) for i, j in align_messages(
                    [(row["who"], row["sender"], row["text"]) for row in effective], visible))
                for item in media:
                    token = media_token(item["fingerprint"])
                    for j, row in aligned.items():
                        if visible[j][2] != token or visible[j][0] != item["who"]:
                            continue
                        existing = self.connection.execute(
                            "SELECT * FROM message_media WHERE message_id = ?", (row["id"],)).fetchone()
                        fresh = not existing and (bool(rows) and row["id"] > rows[-1]["id"]
                                                  or not rows and j == len(visible) - 1)
                        latest = (existing and existing["status"] == "ocr" and j == len(visible) - 1
                                  and row["id"] == effective[-1]["id"])
                        eligible = vision and item["who"] == "her" and (fresh or latest)
                        state = "pending" if eligible else "ocr"
                        if eligible and existing and existing["status"] == "ocr":
                            self.connection.execute("UPDATE message_media SET status = 'pending', updated_at = ? "
                                                    "WHERE message_id = ?", (now, row["id"]))
                        if not existing:
                            self.connection.execute("""
                                INSERT INTO message_media(message_id, fingerprint, kind, ocr_text, status, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (row["id"], item["fingerprint"], item["kind"], item["ocr_text"], state, now))
                        if eligible or vision and existing and existing["status"] == "pending":
                            jobs.append({**item, "message_id": row["id"],
                                         "fingerprint": existing["fingerprint"] if existing else item["fingerprint"]})
            if not vision:
                self.connection.execute("""
                    UPDATE message_media SET status = 'ocr', updated_at = ? WHERE status = 'pending'
                    AND message_id IN (SELECT id FROM messages WHERE conversation_id = ?)
                """, (now, conversation_id))
            media_values = self.media_state(app, title)
            rendered = self.history(app, title)
            new_ids = {row["id"] for row in effective if row["id"] > (rows[-1]["id"] if rows else 0)}
            rendered_new = [value for row, value in zip(effective[-60:], rendered) if row["id"] in new_ids]
        return {"id": snapshot_id, "title": title, "raw_title": raw_title,
                "visible": list(visible), "new": [] if discontinuous or not title_confirmed else rendered_new,
                "history": rendered, "corrected": corrected,
                "area": tuple(area), "read_at": now, "baseline": baseline,
                "title_confirmed": bool(title_confirmed), "status": status,
                "discontinuous": discontinuous, "rebaselined": stable, "media_jobs": jobs, "media_state": media_values,
                "latest_id": effective[-1]["id"] if effective else None,
                "history_ids": [row["id"] for row in effective[-60:]]}
