import time
import pygetwindow as gw
from PIL import ImageGrab, ImageDraw
import numpy as np
import cv2
from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
from comtypes import CLSCTX_ALL
import pyautogui
import random
import ctypes
import dxcam
import cv2
import pyautogui
import time
import json
from win32 import win32gui
from win32.lib import win32con
from PIL import Image
from collections import deque
from CombatDetector import CombatDetector
from CombatController import CombatController
from options import *


def contains_color(image, lower_bound, upper_bound, threshold=10):
    """
    检测图像中是否包含指定范围的颜色，超过 threshold 个像素则返回 True。
    """
    img_np = np.array(image)
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    return cv2.countNonZero(mask) > threshold


# ===================== 核心捕获模块 =====================
class GameCapture:
    def __init__(self, window_title):
        self._init_directx()
        self.hwnd = self._get_window_handle(window_title)
        self._adjust_dpi()
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        self.frame_cache = deque(maxlen=10)  # 用于运动检测的帧缓存

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
        if DEBUG_MODE:
            print("frame is None")
        return None

    def reset_cache(self):
        self.frame_cache.clear()

    def save_img(self, frame):
        if frame is not None:
            # 保存为PNG文件（自动创建目录）
            save_path = "./screenshots/{}.png".format(int(time.time()))

            cv2.imwrite(save_path, frame)
            print(f"截图已保存至：{save_path}")

            if DEBUG_MODE:
                # 显示实时画面（调试用）
                # cv2.imshow('Game Capture', frame)
                cv2.waitKey(1)  # 保持窗口响应
        else:
            print("frame is None，无法保存图片")


def load_config():
    """加载战斗配置"""
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
            # 转换坐标格式为元组
            config["enemy_back_positions"] = [tuple(pos) for pos in config["enemy_back_positions"]]
            config["enemy_front_positions"] = [tuple(pos) for pos in config["enemy_front_positions"]]
            config["area"] = [50, 60]
            return config
    except FileNotFoundError:
        raise RuntimeError("配置文件 combat_config.json 不存在")


def main():
    # 初始化系统
    # if STEALTH_MODE:
    #     dxcam.enable_stealth()

    config = load_config()

    # 初始化模块
    capture = GameCapture("魔力寶貝")
    detector = CombatDetector(config, capture)
    # controller = CombatController()

    count = 0
    # 主循环
    while True:
        count += 1
        # 这里不断捕捉画面，实际上是在做帧缓存，最多缓存3帧，缓存间隔是0.2秒一次
        frame = capture.capture_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        # try:
        enemies = detector.scan_enemy_special()
        if DEBUG_MODE:
            print("no enemies")
        if len(enemies) > 0:
            frontEnemy = []
            endEnemy = []
            for item in enemies:
                if item['side'] == 'front':
                    frontEnemy.append(item['priority'])
                if item['side'] == 'back':
                    endEnemy.append(item['priority'])
            print("enemies")
            print(f"前排有{len(frontEnemy)}个，位置分别是{frontEnemy}")
            print(f"后排有{len(endEnemy)}个，位置分别是{endEnemy}")
            capture.reset_cache()
            time.sleep(3)
    # except BaseException as e:
    #     print(e)
    #     if len(enemies) == 0:
    # capture.save_img(frame)
    # break
    #     print("count: ", count)
    #     break
    # if enemies:
    #     target = enemies[0]["position"]
    #     controller.safe_click(target)
    #     if DEBUG_MODE:
    #         print(f"攻击目标坐标: {target}")

    time.sleep(0.2 + np.random.rand() * 0.2)


if __name__ == "__main__":
    # capture = GameCapture("魔力寶貝")
    # frame = capture.capture_frame()
    # capture.save_img(frame)
    main()
