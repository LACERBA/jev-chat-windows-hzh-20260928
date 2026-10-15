import unittest
from unittest.mock import Mock, patch

from app import capture, fill


class ClickDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.user32 = Mock()
        self.kernel32 = Mock()
        self.dwmapi = Mock()
        self.debug = Mock()
        self.cursor = (900, 800)
        self.user32.GetForegroundWindow.return_value = 1
        self.user32.GetDpiForWindow.return_value = 144
        self.user32.GetThreadDpiAwarenessContext.return_value = -4
        self.user32.GetAwarenessFromDpiAwarenessContext.return_value = 2
        self.user32.GetPhysicalCursorPos.side_effect = self._read_cursor
        self.user32.SetPhysicalCursorPos.side_effect = self._move_cursor
        self.dwmapi.DwmGetWindowAttribute.side_effect = self._window_rect
        # 替换全部窗口和鼠标 API，测试不会操作真实桌面。
        patches = [patch.object(fill, "u32", self.user32),
                   patch.object(fill, "k32", self.kernel32),
                   patch.object(fill.ctypes, "windll", Mock(dwmapi=self.dwmapi)),
                   patch.object(capture, "unminimize"),
                   patch.object(capture, "switch_debug", self.debug),
                   patch.object(fill.time, "sleep")]
        for target in patches:
            target.start()
            self.addCleanup(target.stop)

    def _window_rect(self, hwnd, attribute, pointer, size):
        pointer._obj.left, pointer._obj.top = 100, 50
        pointer._obj.right, pointer._obj.bottom = 700, 600
        return 0

    def _read_cursor(self, pointer):
        pointer._obj.x, pointer._obj.y = self.cursor
        return 1

    def _move_cursor(self, x, y):
        self.cursor = (x, y)
        return 1

    def test_click_uses_window_origin_and_restores_cursor(self):
        fill.click(1, 120, 90)
        self.user32.SetPhysicalCursorPos.assert_any_call(220, 140)
        self.user32.SetPhysicalCursorPos.assert_called_with(900, 800)
        self.assertEqual(self.user32.mouse_event.call_count, 2)
        self.debug.assert_any_call("click_physical_position", position=(220, 140))

    def test_activation_failure_is_logged_before_click_is_cancelled(self):
        self.user32.GetForegroundWindow.return_value = 2
        self.user32.SetForegroundWindow.return_value = 0
        with self.assertRaisesRegex(RuntimeError, "无法激活聊天窗口"):
            fill.click(1, 120, 90)
        self.debug.assert_any_call("click_activation", target_hwnd=1, previous_foreground=2,
                                   activated=False)
        self.user32.SetPhysicalCursorPos.assert_not_called()
        self.user32.mouse_event.assert_not_called()

    def test_dpi_rounding_within_two_pixels_still_clicks_and_restores_cursor(self):
        for dx, dy in ((-1, -1), (1, 1), (-2, 2), (2, -2)):
            with self.subTest(offset=(dx, dy)):
                self.user32.mouse_event.reset_mock()

                def read_cursor(pointer):
                    point = (900, 800) if self.cursor == (900, 800) else (220 + dx, 140 + dy)
                    pointer._obj.x, pointer._obj.y = point
                    return 1

                self.user32.GetPhysicalCursorPos.side_effect = read_cursor
                fill.click(1, 120, 90)
                self.assertEqual(self.user32.mouse_event.call_count, 2)
                self.user32.SetPhysicalCursorPos.assert_called_with(900, 800)

    def test_position_mismatch_reports_coordinates_and_does_not_click(self):
        def read_cursor(pointer):
            point = (900, 800) if self.cursor == (900, 800) else (223, 140)
            pointer._obj.x, pointer._obj.y = point
            return 1

        self.user32.GetPhysicalCursorPos.side_effect = read_cursor
        with self.assertRaises(RuntimeError) as error:
            fill.click(1, 120, 90)
        self.assertIn("目标 (220, 140)，实际 (223, 140)", str(error.exception))
        self.user32.mouse_event.assert_not_called()
        self.user32.SetPhysicalCursorPos.assert_called_with(900, 800)
