# 项目信息

- 这是 Windows Python / PySide6 桌面应用，入口为 `main.py`，本地解释器为 `.venv\Scripts\python.exe`。
- 测试使用标准库 `unittest`，不是 pnpm / Vitest。项目根目录运行：
  - `$env:QT_QPA_PLATFORM = 'offscreen'` 后执行 `.\.venv\Scripts\python.exe -m unittest discover -s app -p 'test_*.py'`。
  - `.\.venv\Scripts\python.exe -m unittest discover -s core -p 'test_*.py'`。
- 部分截图回归测试需要 `JEV_MEDIA_BUG_IMAGE`、`JEV_MEDIA_TEST_DIR` 或 `JEV_OCR_TEST_IMAGE`；未提供样本时会跳过。
- 诊断日志统一使用 `core.diagnostics`，采集子进程通过主进程队列写入日志，避免多个进程同时轮转同一文件。
- 不记录密钥、完整聊天上下文、候选正文或原始截图；会话名、消息 ID、版本号、数量和异常原因用于关联处理链路。
- 验证时使用 mock（模拟对象）和临时数据库，不启动真实采集，不操作真实聊天窗口或发送消息。
