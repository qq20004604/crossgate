# -*- coding: utf-8 -*-
# magic_combat_win11.py
import ctypes
import dxcam
import numpy as np
import cv2
import pyautogui
import time
import json
from win32 import win32gui
from win32.lib import win32con
from PIL import Image
from collections import deque

# ===================== 系统配置 =====================
CONFIG_FILE = "combat_config.json"
DEBUG_MODE = True  # 开启调试信息输出
STEALTH_MODE = True  # 启用防检测模式


# ===================== 核心捕获模块 =====================
class GameCapture:
    def __init__(self, window_title):
        self._init_directx()
        self.hwnd = self._get_window_handle(window_title)
        self._adjust_dpi()
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        self.frame_cache = deque(maxlen=3)  # 用于运动检测的帧缓存

    def _init_directx(self):
        """初始化DirectX捕获环境"""
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PerMonitorV2模式

    def _get_window_handle(self, title):
        """精确获取游戏窗口句柄"""

        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd) and title in win32gui.GetWindowText(hwnd):
                self.target_hwnd = hwnd
            return True

        self.target_hwnd = None
        win32gui.EnumWindows(callback, None)
        if not self.target_hwnd:
            raise RuntimeError(f"Window '{title}' not found")
        return self.target_hwnd

    def _adjust_dpi(self):
        """处理Windows 11 DPI缩放"""
        self.dpi = ctypes.windll.user32.GetDpiForWindow(self.hwnd)
        self.scale = self.dpi / 96.0

    def get_window_region(self):
        """获取物理像素区域"""
        rect = win32gui.GetWindowRect(self.hwnd)
        return (
            int(rect[0] * self.scale),
            int(rect[1] * self.scale),
            int(rect[2] * self.scale),
            int(rect[3] * self.scale)
        )

    def capture_frame(self):
        """捕获当前游戏画面"""
        region = self.get_window_region()
        frame = self.camera.grab(region=region)
        if frame is not None:
            self.frame_cache.append(frame)
            return frame
        return None


# ===================== 战斗检测模块 =====================
class CombatDetector:
    def __init__(self, config):
        self.positions = config["enemy_positions"]
        self.color_profiles = config["color_profiles"]
        self.motion_threshold = 25

    def _color_match(self, roi):
        """多颜色特征匹配"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color in self.color_profiles:
            lower = np.array(color["lower"])
            upper = np.array(color["upper"])
            mask = cv2.inRange(hsv, lower, upper)
            total_mask = cv2.bitwise_or(total_mask, mask)

        return cv2.countNonZero(total_mask) > 500

    def _motion_detect(self, pos_index):
        """多帧运动检测"""
        if len(self.frame_cache) < 3:
            return False

        x, y, w, h = self.positions[pos_index]
        diff_sum = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, len(self.frame_cache)):
            prev = self.frame_cache[i - 1][y:y + h, x:x + w]
            curr = self.frame_cache[i][y:y + h, x:x + w]
            diff = cv2.absdiff(prev, curr)
            diff_sum = cv2.add(diff_sum, diff)

        return np.mean(diff_sum) > self.motion_threshold

    def scan_enemies(self):
        """智能扫描敌人"""
        targets = []
        for idx, (x, y, w, h) in enumerate(self.positions):
            roi = self.frame_cache[-1][y:y + h, x:x + w]
            if self._color_match(roi) and self._motion_detect(idx):
                targets.append({
                    "position": (x + w // 2, y + h // 2),
                    "priority": 0 if idx % 2 == 0 else 1  # 前排优先
                })
        return sorted(targets, key=lambda x: x["priority"])


# ===================== 操作控制模块 =====================
class CombatController:
    def __init__(self):
        self.click_history = []
        self.last_click = 0

    def _human_move(self, start, end):
        """拟人化移动轨迹"""
        steps = np.random.randint(8, 15)
        points = np.linspace(start, end, steps)
        for point in points:
            pyautogui.moveTo(*point, duration=0.02 + np.random.rand() * 0.05)
            time.sleep(0.02)

    def safe_click(self, position):
        """安全点击（带随机偏移）"""
        if time.time() - self.last_click < 0.8:
            return  # 防止操作过频

        offset = np.random.randint(-3, 3, 2)
        target = (position[0] + offset[0], position[1] + offset[1])

        current_pos = pyautogui.position()
        self._human_move(current_pos, target)

        pyautogui.mouseDown()
        time.sleep(0.1 + np.random.rand() * 0.1)
        pyautogui.mouseUp()

        self.last_click = time.time()
        self.click_history.append(target)


# ===================== 主控模块 =====================
def load_config():
    """加载战斗配置"""
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
            # 转换坐标格式为元组
            config["enemy_positions"] = [tuple(pos) for pos in config["enemy_positions"]]
            return config
    except FileNotFoundError:
        raise RuntimeError("配置文件 combat_config.json 不存在")


def calibrate_positions(capture):
    """坐标校准工具"""
    print("=== 敌人位置校准 ===")
    positions = []
    frame = capture.capture_frame()

    for i in range(10):
        cv2.imshow("Calibration", frame)
        print(f"请框选敌人位置{i + 1} (按空格确认)")
        roi = cv2.selectROI("Calibration", frame, showCrosshair=False)
        positions.append(roi)
        cv2.destroyAllWindows()

    # 保存配置
    config = {
        "enemy_positions": positions,
        "color_profiles": [
            {"lower": [30, 50, 50], "upper": [80, 255, 255]}  # 默认绿色系
        ]
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print("校准完成！配置已保存")


def main():
    # 初始化系统
    if STEALTH_MODE:
        dxcam.enable_stealth()

    try:
        config = load_config()
    except RuntimeError:
        capture = GameCapture("魔力寶貝")
        calibrate_positions(capture)
        config = load_config()

    # 初始化模块
    capture = GameCapture("魔力寶貝")
    detector = CombatDetector(config)
    controller = CombatController()

    # 主循环
    while True:
        frame = capture.capture_frame()
        if frame is None:
            time.sleep(0.5)
            continue

        enemies = detector.scan_enemies()
        if enemies:
            target = enemies[0]["position"]
            controller.safe_click(target)
            if DEBUG_MODE:
                print(f"攻击目标坐标: {target}")

        time.sleep(0.1 + np.random.rand() * 0.2)


if __name__ == "__main__":
    main()
