# jev-chat-windows — 接续说明（OCR 版）

个人用的聊天回复辅助工具，挂在自己电脑的微信旁边：读到对方最新消息 → 判断意图/情绪 →
给出 3 条候选回复（已排序）→ 默认由人一键**填入**微信输入框；明确授权的会话可以自动发送最佳回复。

## 硬约束（照做，别破）

1. 只读**自己设备上、自己有权查看**的对话。
2. 采集只用**窗口级截图 + 本地离线 OCR**。不 hook、不注入、不读微信数据库、不解密、不碰微信进程。
3. **截图不落盘**：捕获得到的位图始终是内存里的对象（numpy/PIL），全程不写磁盘、不进日志、不上传。
4. **自动发送默认关闭且只能按会话明确授权**；发送前必须校验当前会话、最佳回复、单次 revision 和冷却时间。
5. **不碰钱**：转账、红包、收款相关界面元素一律不碰。
6. 密钥只从环境变量 / 注册表读取，任何文件不出现 key：Jev 用 `JEV_API_KEY`，独立通用判断用 `JUDGE_API_KEY`，起草用 `LLM_API_KEY`；通用判断复用起草配置时不需要第三把。
7. Python 读写文件一律 `encoding='utf-8'`。

## 为什么走 OCR（已实测的结论，别重测）

- 目标窗口界面自绘在一块 GPU 合成画布上（`MMUIRenderSubWindowHW`）。UIA 树只有 2 个节点、**没有控件树**——实测证伪。
- 所以唯一干净的非侵入采集路 = 截自己的微信窗口 + 本地 OCR。离线、零 token。

## 已经建好，直接用（`core/`，平台无关）

| 文件 | 作用 |
|---|---|
| `core/jev_client.py` | Jev 判断 API 客户端（stdlib、脱敏、429/529 退避）。`ask(state, questions)` |
| `core/llm_judge.py` | 普通语言模型判断器，输出与 Jev 相同的 answers 结构并严格校验 |
| `core/questions.py` | 7 道判断题 + `build_state()` + `build_rank_question()` |
| `core/draft.py` | 生成模型起草 3 条候选（OpenRouter，默认 DeepSeek）。解析器已自测 |
| `core/engine.py` | **唯一入口** `analyze(messages, relationship)` → `{candidates, best_index, best_reply, answers, usage}` |
| `tools/demo.py` | 端到端冒烟（需 key + 联网）：`python tools/demo.py` |

`messages` 形如 `[("her","中文"),("me","中文")]`，`from` 只用 `her`/`me`，最新一条在最后。
引擎完全不关心消息怎么来的——OCR 把屏幕上的对话整理成这个 list 喂进 `analyze()` 即可。

## 待建（新会话干这些）

1. **`capture.py` — 窗口级截图到内存，不落盘**
   - 目标窗口按进程名和主窗口标题挑，不按面积。
   - 微信是 GPU 合成窗口，`PrintWindow` 容易黑屏 → **优先用 Windows Graphics Capture**
     （pip `windows-capture`，帧直接是 numpy，可捕获被遮挡/GPU 窗口）。
     兜底：`PrintWindow` 带 `PW_RENDERFULLCONTENT=2`；再兜底：区域抓屏（需窗口可见）。
   - 返回内存位图，**绝不 `.save()`**。
2. **`ocr.py` — 内存内 OCR + 分说话人**
   - RapidOCR（`rapidocr-onnxruntime`）直接吃 numpy 数组，不落文件。
   - 裁到聊天气泡区；按每段文字**框的 x 中心**分左右：右=me、左=her（可加气泡背景色过滤头像/时间戳/系统提示）。
   - **diff**：跟上一帧比，只在冒出**新的 her 消息**时才触发下游（省 token，也避免重复分析）。
   - `probe/probe_ocr.py` 里已有 x 阈值分左右的草稿，可参考。
3. **`overlay.py` — 半透明置顶悬浮窗**：显示判断摘要 + 3 条候选（★ 推荐），每条一个「填入」按钮。
   tkinter（`-topmost`/`-alpha`，stdlib）够用；要更顺再上 PySide6。
4. **`fill.py` — 填入与授权发送**：手动候选只粘贴；会话明确开启自动回复时覆盖输入框并模拟已配置的发送快捷键。
5. **`main.py` — 主循环**：capture → ocr → 检测新 her 消息 → `engine.analyze()` → overlay。
   静默期不调 Jev（10 分钟无新消息 = 0 次调用）。

## 技术坑备忘

- GPU 窗口截图黑屏 → 用 Windows Graphics Capture，别用普通 BitBlt/PrintWindow。
- 图片全程内存对象（numpy/PIL），OCR 引擎吃数组不吃路径，天然不落盘。
- 中文 OCR：RapidOCR 够用；不够准就换 PaddleOCR 或只 OCR 裁剪后的聊天区。
- 群聊按一对一分析会不准，先只做单聊。

## 参考项目

安卓原版 `Finderchangchang/jev-chat-JARVIS`（同一套 Jev 判断内核，采集是安卓无障碍）。
Jev 接口、题目口径都跟 `core/` 里一致。

## 本地消息采集与回归验证

- `app/message_store.py` 使用 Python 标准库 SQLite；采集子进程写入 `messages.sqlite3`，消费端使用只读连接，WAL 支持并行读写。不需要新增依赖。
- `conversations` 按聊天软件和 OCR 会话名隔离；`messages` 存气泡文本、说话人和采集时间；`ocr_snapshots` 存可见文本与识别元数据，不存截图。采集时间不是微信原始发送时间。
- 采集、生成建议、允许自动发送分别控制；`send_enabled` 默认关闭，开启后仍须会话授权。关闭 `analysis_enabled` 时采集入库不依赖模型配置。
- 去重按气泡顺序对齐，不按历史文本是否出现去重；已有历史之后的无锚点批次隔离为 `pending`，原始文本保留，不进入有效上下文、不触发自动回复。只能采集已显示并被 OCR 识别的内容，不代表完整微信历史。
- `message_states` 存有效文本和状态，`message_revisions` 存修订原因、前后值及来源快照；`effective_messages` 视图和 `MessageStore.history()` 提供修订后的有效消息，不能直接用原始 `messages` 表作为回复上下文。
- 自动文字修订须有前后相邻的准确消息锚点，且连续两次 OCR 一致；滚动隐藏消息不代表识别错误。启动时隔离旧版无锚点基线污染并从已存正确快照恢复漏入库消息；恢复记录沿用原采集时间，不算实时新消息。
- Windows 离线回归命令：`.venv\Scripts\python.exe -m unittest app.test_ocr_geometry app.test_worker_storage app.test_message_store app.test_conversation_filter app.test_model_settings core.test_judge_fallback`。真实截图回归可设置 `JEV_OCR_TEST_IMAGE` 指向用户提供的本地截图；不复制截图进仓库。需要无窗口运行时设置 `QT_QPA_PLATFORM=offscreen`；发送相关测试使用 Mock，不实际操作聊天窗口。
- 数据库及其 `-wal`、`-shm` 文件不进 Git；SQLite 文本未加密，依靠本机目录权限保护。开启生成建议后，最近的消息上下文仍会发给用户配置的模型服务。
