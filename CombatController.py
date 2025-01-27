import numpy as np
import pyautogui
import time


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

    def click(self):
        pyautogui.mouseDown()
        time.sleep(0.1 + np.random.rand() * 0.1)
        pyautogui.mouseUp()
