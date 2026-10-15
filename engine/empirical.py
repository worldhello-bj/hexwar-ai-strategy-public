# -*- coding: utf-8 -*-
"""实测裁决表（empirical_tables.json）的只读加载器 —— **默认不介入裁决**。

数据由 `tools/hex_crt_maintain.py` 从线上 `Caijue.asp` 样本聚合而来。

设计原则（1:1 复刻用）：
  * **只在实测值无异议（`dissent` 为空）时**才认为可用；有异议一律视为"无实测值"；
  * 有实测值 → 用实测；无 → 调用方退回手册表（`tables.json`）；
  * `enabled()` 默认 False，**当前未接入 `crt.py`**，以免在样本不足时污染已验证的引擎。

接入时机：`docs/实测裁决表.md` 中 `conflicts` 清空、且目标格子覆盖足够之后。
"""
import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "empirical_tables.json")
_ENABLED = os.environ.get("HEXWAR_EMPIRICAL", "") not in ("", "0", "false", "False")
_CACHE = {"data": None}


def enabled():
    """实测覆盖是否启用（默认关闭）。"""
    return _ENABLED


def load():
    if _CACHE["data"] is None:
        try:
            with open(_PATH, encoding="utf-8") as stream:
                _CACHE["data"] = json.load(stream)
        except (OSError, ValueError):
            _CACHE["data"] = {}
    return _CACHE["data"]


def _cell(group, key, roll):
    node = (load().get(group) or {}).get(key) or {}
    cell = node.get(str(int(roll)))
    if not cell or cell.get("dissent"):
        return None
    return cell.get("value")


def vehicle_result(armor, attack_level, roll1):
    """表4 实测值：返回 kills，或 None（无可用实测值）。"""
    if not _ENABLED:
        return None
    return _cell("vehicle_result", "%s|%s" % (armor, int(attack_level)), roll1)


def person_result(attack_level, roll1):
    """表6 实测值：返回 kills/'P'，或 None。"""
    if not _ENABLED:
        return None
    return _cell("person_result", str(int(attack_level)), roll1)


def state_mod(target_state, roll2):
    """表7/表5 修正实测值：返回 int，或 None。"""
    if not _ENABLED:
        return None
    return _cell("state_mod", target_state, roll2)


def stats():
    """覆盖度概要（调试用）。"""
    d = load()
    out = {}
    for g in ("vehicle_result", "person_result", "state_mod"):
        tab = d.get(g) or {}
        filled = sum(len(v) for v in tab.values())
        out[g] = {"keys": len(tab), "filled_cells": filled}
    out["conflicts"] = len(d.get("conflicts") or [])
    out["samples"] = d.get("sample_count")
    return out
