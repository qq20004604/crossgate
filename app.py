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
import os
from win32 import win32gui
from win32.lib import win32con
from PIL import Image
from collections import deque
from CombatDetector import CombatDetector
from CombatController import CombatController
from options import DEBUG_MODE
from LoadConfig import load_config
import winsound


# ===================== 核心捕获模块 =====================
class GameCapture:
    def __init__(self, window_title):
        self._init_directx()
        self.hwnd = self._get_window_handle(window_title)
        self._adjust_dpi()
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        self.frame_cache = deque(maxlen=10)  # 用于运动检测的帧缓存\
        self.skill_img = None
        self.pet_img = None
        self.load_img("./source/乱射.png")

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

    # 获得某一片区域，指定颜色的像素点数量
    def get_area_color_count(self, area, color):
        (x1, y1, width, height) = area
        x2 = x1 + width
        y2 = y1 + height
        frame = self.frame_cache[-1]
        # 边界检查，防止裁剪超出范围
        frame_height, frame_width = frame.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            print("检测区域超出帧范围")
            return False

        # 裁剪检测区域
        region = frame[y1:y2, x1:x2]
        # 将 RGB 颜色，转换为 BGR 颜色，才能正确识别
        lowColor = (
            0 if color[2] - 20 < 0 else color[2] - 20,
            0 if color[1] - 20 < 0 else color[1] - 20,
            0 if color[0] - 20 < 0 else color[0] - 20,
        )
        upColor = (
            255 if color[2] + 20 > 255 else color[2] + 20,
            255 if color[1] + 20 > 255 else color[1] + 20,
            255 if color[0] + 20 > 255 else color[0] + 20,
        )

        print(lowColor)
        print(upColor)
        # 统计像素点数量
        pixels = cv2.inRange(region, lowColor, upColor)
        pixel_count = cv2.countNonZero(pixels)
        print(pixel_count)
        # if pixel_count == 0:
        # 保存裁剪后的图片
        # save_path = f"./detected_regions/region_{area}.png"
        # cv2.imwrite(save_path, region)
        # print(f"区域图片已保存: {save_path}")

        return pixel_count

    def load_img(self, skill_image_path):
        # file_path = os.path.abspath(skill_image_path)
        # print(f"绝对路径: {file_path}")
        self.skill_img = cv2.imdecode(np.fromfile(skill_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.pet_img = cv2.imdecode(np.fromfile("./source/pet_action_tag.png", dtype=np.uint8), cv2.IMREAD_COLOR)

    # 检查能否使用乱射，返回数据为 False 或者 (91, 283) 这样
    def can_use_luanshe(self):
        return self._get_img_from_img(self.skill_img)

    # 是否宠物可以行动
    def can_pet_act(self):
        result = self._get_img_from_img(self.pet_img)
        print("can_pet_act", result)
        return result

    # 图中搜图
    def _get_img_from_img(self, targetImage, search_area=None):
        """
        在 sourceImage 的指定范围内搜索 targetImage。
        如果能找到，返回目标区域的中心点范围；如果找不到，返回 False。

        :param targetImage: 要搜索的小图（可以是图像路径或 numpy.ndarray）
        :param search_area: 搜索范围，格式为 (x, y, width, height)，默认搜索整个图片
        :return: 匹配区域的中心点坐标 (center_x, center_y) 或 False
        """
        # 检查 frame_cache 是否有帧
        if len(self.frame_cache) == 0:
            print("frame_cache 中没有可用的帧数据！")
            return False

        # 获取最新帧作为 sourceImage
        sourceImage = self.frame_cache[-1]

        # 检查 targetImage 是否加载成功
        if isinstance(targetImage, str):
            targetImage = cv2.imread(targetImage)  # 加载目标图像
        if targetImage is None or not isinstance(targetImage, np.ndarray):
            print("targetImage 加载失败或格式不正确！")
            return False

        # 如果指定了搜索范围，裁剪源图像
        if search_area:
            x, y, width, height = search_area
            x, y, width, height = int(x), int(y), int(width), int(height)

            # 检查边界
            frame_height, frame_width = sourceImage.shape[:2]
            if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
                print("指定的搜索范围超出源图像范围！")
                return False

            # 裁剪源图像
            sourceImage = sourceImage[y:y + height, x:x + width]

        # 确保目标图和源图都是灰度图，减少匹配干扰
        gray_source = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

        # 使用模板匹配方法搜索目标图像
        result = cv2.matchTemplate(gray_source, gray_target, cv2.TM_CCOEFF_NORMED)

        # 设置匹配的阈值，值越高要求匹配越精确
        threshold = 0.95

        # 找到匹配结果中大于阈值的区域
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # print(f"模板匹配结果：min_val={min_val}, max_val={max_val}, min_loc={min_loc}, max_loc={max_loc}")

        if max_val >= threshold:
            # 获取目标区域的左上角位置
            top_left = max_loc
            target_h, target_w = gray_target.shape[:2]

            # 如果使用了裁剪，需要加上裁剪的偏移量
            if search_area:
                top_left = (top_left[0] + x, top_left[1] + y)

            # 计算目标区域的中心点
            center_x = top_left[0] + target_w // 2
            center_y = top_left[1] + target_h // 2

            # print(f"目标匹配成功，中心点坐标为：({center_x}, {center_y})")
            return (center_x, center_y)
        else:
            # 未找到匹配结果
            print("未找到目标匹配结果")
            save_path = f"./detected_regions/gray_source.png"
            cv2.imwrite(save_path, gray_source)
            save_path = f"./detected_regions/gray_target.png"
            cv2.imwrite(save_path, gray_target)
            print(f"区域图片已保存: {save_path}")
            return False

    # 播放警告声
    def play_notice_voice(self):
        # 播放系统默认的“滴”声
        winsound.MessageBeep()

        # 或者生成一个自定义频率和持续时间的声音
        frequency = 1000  # 频率（Hz）
        duration = 500  # 持续时间（毫秒）
        winsound.Beep(frequency, duration)


# 移动循环
def walk_loop(capture):
    """
    主要解决的是非战斗状态的处理
    当进入战斗时，离开本循环，进入战斗循环
    :return:
    """
    count = 0
    while True:
        count += 1
        # 这里不断捕捉画面，实际上是在做帧缓存，最多缓存3帧，缓存间隔是0.2秒一次
        frame = capture.capture_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        # 这里是检测最上方【要求进行对战按钮】位置的的颜色数量，用于确保当前还在非战斗状态
        # 示例图为：[testimg/监测点1.png]
        if capture.get_area_color_count((455, 31, 16, 14), (247, 220, 170)) > 40:
            # 说明还在非战斗状态
            # 限制为每10次检测到该状态时，才进行一次操作
            if count % 10 == 0:
                # 点击某个位置进行移动
                pass
        else:
            # 如果没找到，说明进入非战斗状态了
            print(f"步行结束，10秒后进入战斗状态。count:{count}")
            time.sleep(10)
            break


def battle_loop(capture, detector, controller):
    """
    主要解决的是战斗状态的处理
    当进入战斗时，离开本循环，进入战斗循环
    :return:
    """

    # 1、先判断当前是否可施放技能，如果可以，选择技能。
    # 2、再判断怪物位置，按顺序依次选择前排、后排。
    # 3、每次选择完，确认是否选中（此时可看到宠物行动阶段），如果没选中，则选择下一个目标。若所有目标选完，还没目标，则切换为特殊寻找敌人方式。
    # 4、此时确认是宠物行动，宠物选择同一个目标进行操作。
    # 5、此时继续判断两种情况：①是否可以释放技能；②判断是否战斗结束。如果前者，重新开始循环，如果是后者，break掉。

    count = 0
    while True:
        count += 1
        # 这里不断捕捉画面，实际上是在做帧缓存，最多缓存3帧，缓存间隔是0.2秒一次
        frame = capture.capture_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        # 先检查是否战斗结束
        if capture.get_area_color_count((455, 31, 16, 14), (247, 220, 170)) > 40:
            print(f"战斗结束，10秒后进入非战斗状态。count:{count}")
            time.sleep(10)
            break

        # 检查能否使用乱射
        is_luanshe = capture.can_use_luanshe()
        if is_luanshe is False:
            time.sleep(1)
            continue

        # 如果不是 False，则是类似坐标点之类的，比如 (91, 283)
        # 此时连续点击2下（第一下是点击乱射技能，第二下是选择技能等级）
        # 再把鼠标移动到右下角，再开始检测敌人位置。检测前，需要连续获得 10 帧
        controller.safe_click(is_luanshe)
        print("第一次点击结束")
        time.sleep(0.2 + np.random.rand() * 0.2)
        controller.click()
        print("第2次点击结束")
        time.sleep(0.2 + np.random.rand() * 0.5)

        current_pos = pyautogui.position()
        controller._human_move(current_pos, (341, 470))

        print("开始缓存10帧")
        tempCount = 0
        while True:
            capture.capture_frame()
            time.sleep(0.05)
            if frame is None:
                continue
            else:
                tempCount += 1
            # 循环10次，也就是拿10帧有效图像
            # 写12是为了保险
            if tempCount > 12:
                print("10帧缓存完毕")
                break

        print("开始查找敌人")
        enemies = detector.scan_enemies()
        if len(enemies) == 0:
            enemies = detector.scan_enemy_special()

        frontEnemy = []
        backEnemy = []
        frontPos = []
        backPos = []
        for item in enemies:
            if item['side'] == 'front':
                frontEnemy.append(item['priority'])
                frontPos.append(item['position'])
            if item['side'] == 'back':
                backEnemy.append(item['priority'])
                backPos.append(item['position'])
        print("enemies")
        print(f"前排有{len(frontEnemy)}个，位置分别是{frontEnemy}")
        print(f"坐标{frontPos}")
        print(f"后排有{len(backEnemy)}个，位置分别是{backEnemy}")
        print(f"坐标{backPos}")
        is_pet_action = False
        action_pos = None
        # 按顺序点击，并检查是否到宠物行动了
        if len(frontPos) > 0:
            # 按顺序尝试点击
            for pos in frontPos:
                # 点一下
                controller.safe_click(pos)
                time.sleep(0.2 + np.random.rand() * 1.5)
                # 查看是否到宠物行动了
                if capture.can_pet_act() is False:
                    continue
                else:
                    is_pet_action = True
                    action_pos = pos
                    print("action_pos", action_pos)
                    break
        # 如果前排没点击，那么再检查后排
        if is_pet_action is False:
            print("前排没有敌人")
            if len(backPos) > 0:
                # 按顺序尝试点击
                for pos in backPos:
                    # 点一下
                    controller.safe_click(pos)
                    time.sleep(0.2 + np.random.rand() * 1.5)
                    # 查看是否到宠物行动了
                    if capture.can_pet_act() is False:
                        continue
                    else:
                        is_pet_action = True
                        action_pos = pos
                        print("action_pos", action_pos)
                        break
        # 如果此时还是 False，说明很可能敌人检测失败。（比如说，enemies = detector.scan_enemies() 获得值了，但前排存在怪物却没检测出来，所以点不了后排怪）
        if is_pet_action is False:
            # 重新点击一遍
            print("采用特殊敌人位置捕捉法")
            enemies = detector.scan_enemy_special()
            frontEnemy = []
            backEnemy = []
            frontPos = []
            backPos = []
            for item in enemies:
                if item['side'] == 'front':
                    frontEnemy.append(item['priority'])
                    frontPos.append(item['position'])
                if item['side'] == 'back':
                    backEnemy.append(item['priority'])
                    backPos.append(item['position'])

            # 按顺序点击，并检查是否到宠物行动了
            if len(frontPos) > 0:
                # 按顺序尝试点击
                for pos in frontPos:
                    # 点一下
                    controller.safe_click(pos)
                    time.sleep(0.2 + np.random.rand() * 1.5)
                    # 查看是否到宠物行动了
                    if capture.can_pet_act() is False:
                        continue
                    else:
                        is_pet_action = True
                        action_pos = pos
                        print("action_pos", action_pos)
                        break
            # 如果前排没点击，那么再检查后排
            if is_pet_action is False:
                print("前排没有敌人")
                if len(backPos) > 0:
                    # 按顺序尝试点击
                    for pos in backPos:
                        # 点一下
                        controller.safe_click(pos)
                        time.sleep(0.2 + np.random.rand() * 1.5)
                        # 查看是否到宠物行动了
                        if capture.can_pet_act() is False:
                            continue
                        else:
                            is_pet_action = True
                            action_pos = pos
                            print("action_pos", action_pos)
                            break
        # 如果还点击不了，那么就要报错一下了。播放个声音
        if is_pet_action is False:
            print("最终没有找到宠物可行动情况")
            while True:
                # 每2秒播放一次
                capture.play_notice_voice()
                time.sleep(2)

        # 此时认为宠物可以行动了，点击一下
        time.sleep(0.5)
        controller.safe_click(action_pos)
        time.sleep(0.2 + np.random.rand() * 1.5)
        # 宠物行动完后，开始下一轮


def main_test():
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
        enemies = detector.scan_enemies()
        if DEBUG_MODE:
            print("no enemies")
        if len(enemies) > 0:
            frontEnemy = []
            backEnemy = []
            frontPos = []
            backPos = []
            for item in enemies:
                if item['side'] == 'front':
                    frontEnemy.append(item['priority'])
                    frontPos.append(item['position'])
                if item['side'] == 'back':
                    backEnemy.append(item['priority'])
                    backPos.append(item['position'])
            print("enemies")
            print(f"前排有{len(frontEnemy)}个，位置分别是{frontEnemy}")
            print(f"坐标{frontPos}")
            print(f"后排有{len(backEnemy)}个，位置分别是{backEnemy}")
            print(f"坐标{backPos}")
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


def main():
    config = load_config()

    # 初始化模块
    capture = GameCapture("魔力寶貝")
    detector = CombatDetector(config, capture)
    controller = CombatController()
    # controller = CombatController()

    count = 0
    # 主循环
    while True:
        count += 1
        # 先进行步行，如果结束，说明进入战斗状态了
        # walk_loop()
        battle_loop(capture, detector, controller)

    time.sleep(0.2 + np.random.rand() * 0.2)


if __name__ == "__main__":
    # capture = GameCapture("魔力寶貝")
    # frame = capture.capture_frame()
    # if frame is None:
    #     print("frame is None")

    # count = 0
    # while True:
    #     count += 1
    #     # 这里不断捕捉画面，实际上是在做帧缓存，最多缓存3帧，缓存间隔是0.2秒一次
    #     frame = capture.capture_frame()
    #     if frame is None:
    #         time.sleep(0.05)
    #         continue
    #
    #     is_luanshe = capture.can_use_luanshe()
    #     if is_luanshe:
    #         print(is_luanshe)
    #     print(f"count:{count}")
    #     time.sleep(1)

    # capture.get_area_color_count((455, 31, 16, 14), (247, 220, 170))
    # capture.save_img(frame)
    main()
    # main_test()
