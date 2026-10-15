# -*- coding: utf-8 -*-
"""想定数据目录解析 —— 「本地完整采集优先，公开地图兜底」。

背景：想定 7527 的地图与编成有**两份**来源：

* `snapshot/parsed/` —— 本地完整采集（含原始响应、会话产物、手册衍生物），
  **不入库**（属服务端数据与第三方版权）；
* `scenario/7527/` —— **可公开**的那一份（地图 + 编成 + 夺控点，见其 README 与
  `tools/publish_scenario.py`），随仓库分发。

引擎、训练环境、规则校验都应当**开箱即用**：本地采集在就用本地（信息更全），
没有（公开克隆、CI）就用 `scenario/7527`。环境变量 `HEXWAR_SNAP` 优先级最高。
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(REPO_ROOT, "snapshot", "parsed")
PUBLISHED = os.path.join(REPO_ROOT, "scenario", "7527")

FILES = ("map_hexes.json", "units.json", "cities.json")


def _complete(path):
    return all(os.path.isfile(os.path.join(path, f)) for f in FILES)


def snapshot_dir():
    """返回可用的想定目录（绝对路径）。"""
    env = os.environ.get("HEXWAR_SNAP")
    if env:
        return env
    if _complete(LOCAL):
        return LOCAL
    return PUBLISHED


def source_label():
    """给报告用：这份数据是从哪来的。"""
    env = os.environ.get("HEXWAR_SNAP")
    if env:
        return "env:HEXWAR_SNAP"
    if _complete(LOCAL):
        return "snapshot/parsed（本地完整采集，不入库）"
    if _complete(PUBLISHED):
        return "scenario/7527（随仓库公开的地图与编成）"
    return "缺失（既没有 snapshot/parsed 也没有 scenario/7527）"


def file_path(name):
    return os.path.join(snapshot_dir(), name)
