import numpy as np
import cv2
import os


# ===================== 战斗检测模块 =====================
class CombatDetector:
    def __init__(self, config, GameCapture):
        # pos记录的是中心点的位置
        self.front_pos = config["enemy_front_positions"]
        self.back_pos = config["enemy_back_positions"]
        self.area = config["area"]  # 分别是宽高
        self.GameCapture = GameCapture

        # 这里记录的是左上角的点
        self.motion_threshold = 2

    def _motion_detect(self, x, y, w, h):
        if len(self.GameCapture.frame_cache) < 10:
            return False

        # 转换为整数并检查边界
        x, y, w, h = int(x), int(y), int(w), int(h)
        frame_height, frame_width = self.GameCapture.frame_cache[0].shape[:2]
        if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
            return False

        # 获取通道数
        channels = self.GameCapture.frame_cache[0].shape[2] if len(self.GameCapture.frame_cache[0].shape) > 2 else 1

        # 初始化 diff_sum，确保维度匹配
        diff_sum = np.zeros((h, w, channels), dtype=np.uint8)

        for i in range(1, len(self.GameCapture.frame_cache)):
            prev = self.GameCapture.frame_cache[i - 1][y:y + h, x:x + w]
            curr = self.GameCapture.frame_cache[i][y:y + h, x:x + w]

            # 计算绝对差分
            diff = cv2.absdiff(prev, curr)

            # 累加差分
            diff_sum = cv2.add(diff_sum, diff)

        return np.mean(diff_sum) > self.motion_threshold

    # 常规寻找敌人方式
    def scan_enemies(self):
        """智能扫描敌人"""
        targets = []
        # 先检测前排
        for index, (x, y) in enumerate(self.front_pos):
            x1 = int(x - self.area[0] / 2)
            y1 = int(y - self.area[1] / 2)
            w = self.area[0]
            h = self.area[1]

            if self._motion_detect(x1, y1, w, h):
                targets.append({
                    "position": (x, y),
                    "priority": index + 1,
                    "side": "front"
                })
                # 保存区域范围图片
                # self._save_region_image(x1, y1, w, h, f"front_{index}")

        # 再检测后排
        for index, (x, y) in enumerate(self.back_pos):
            x1 = int(x - self.area[0] / 2)
            y1 = int(y - self.area[1] / 2)
            w = self.area[0]
            h = self.area[1]

            if self._motion_detect(x1, y1, w, h):
                targets.append({
                    "position": (x, y),
                    "priority": index + 1,
                    "side": "back"
                })
                # 保存区域范围图片
                # self._save_region_image(x1, y1, w, h, f"back_{index + 1}")
        return targets


    # 补充寻找敌人方式：
    def scan_enemy_special(self):
        targets = []
        # 先检测前排
        for index, (x, y) in enumerate(self.front_pos):
            if self._name_match(x, y):
                targets.append({
                    "position": (x, y),
                    "priority": index + 1,
                    "side": "front"
                })
                # 保存区域范围图片
                # self._save_region_image(x1, y1, w, h, f"front_{index}")

        # 再检测后排
        for index, (x, y) in enumerate(self.back_pos):
            if self._name_match(x, y):
                targets.append({
                    "position": (x, y),
                    "priority": index + 1,
                    "side": "back"
                })
                # 保存区域范围图片
                # self._save_region_image(x1, y1, w, h, f"back_{index + 1}")
        return targets

    def _name_match(self, center_x, center_y):
        # 常规寻找敌人方式，是通过比对画面，确认敌人是否在该出现的位置
        # 但这种方式，存在一个问题，就是如果敌人在的位置比较偏（比如前排最右边），又或者是敌人的动作并不明显，就无法找到敌人
        # 因此做这种方式来补充，其原理是，判断该位置是否有名字。
        # 具体来说，是以截图名字附近区域，查看白色像素点数量（注意，颜色不一定是纯白，而是接近白色）
        print("scan_enemy_special")
        """
        通过检测接近白色的像素点来判断是否存在敌人名字。
        :param center_x: 中心点 x 坐标
        :param center_y: 中心点 y 坐标
        :return: 是否检测到名字（True/False）
        """
        print("scan_enemy_special")

        # 设置名字检测区域的宽度和高度
        width = 30  # 区域宽度
        height = 14  # 区域高度

        # 计算检测区域左上角坐标
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = x1 + width
        y2 = y1 + height

        # 获取最新帧
        frame = self.GameCapture.frame_cache[-1]

        # 边界检查，防止裁剪超出范围
        frame_height, frame_width = frame.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            print("检测区域超出帧范围")
            return False

        # 裁剪检测区域
        region = frame[y1:y2, x1:x2]

        # 转换为灰度图（可选，提升处理效率）
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # 白色像素点的阈值（定义接近白色的像素）
        white_threshold = 240

        # 统计接近白色的像素点数量
        white_pixels = cv2.inRange(region, (white_threshold, white_threshold, white_threshold), (255, 255, 255))
        white_pixel_count = cv2.countNonZero(white_pixels)

        # 定义像素点数量的阈值，超过此值认为检测到名字
        pixel_threshold = 5

        # 输出检测结果
        # print(f"检测到的接近白色像素点数量: {white_pixel_count}")
        if white_pixel_count > pixel_threshold:
            print("检测到名字区域")
            return True
        else:
            print("未检测到名字区域")
            return False

    def _save_region_image(self, x, y, w, h, index):
        """保存指定区域范围的图片"""
        # 获取最新帧
        frame = self.GameCapture.frame_cache[-1]

        # 确保坐标是整数
        x, y, w, h = int(x), int(y), int(w), int(h)

        # 裁剪区域
        cropped = frame[y:y + h, x:x + w]

        # 创建保存目录
        # os.makedirs("detected_regions", exist_ok=True)

        # 保存裁剪后的图片
        save_path = f"./detected_regions/region_{index}.png"
        cv2.imwrite(save_path, cropped)
        print(f"区域图片已保存: {save_path}")
