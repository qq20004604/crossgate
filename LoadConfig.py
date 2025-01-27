from options import *
import json

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
