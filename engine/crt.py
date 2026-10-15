# -*- coding: utf-8 -*-
"""战斗裁决表（CRT）实现。

数据来源：比赛手册第 5 章「裁决表」（第 27-31 页），
按页渲染核对后编码；表4/表5 存在网格歧义，见 tables.json 的 note。
"""
import json
import os
import random

try:
    from .equip import ANTI_AIR_CLASS, CRT_AIR_NAME
except ImportError:  # 允许直接以脚本方式运行
    from equip import ANTI_AIR_CLASS, CRT_AIR_NAME

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "tables.json"), encoding="utf-8") as stream:
    T = json.load(stream)

PUSH = "P"          # 压制
INVALID = ""        # 无效


def d6(rng=None):
    return (rng if rng is not None else random).randint(1, 6)


def roll2d6(rng=None):
    return d6(rng) + d6(rng)


# ---------------- 攻击等级 ----------------

def atk_level_person(weapon_name, units, dist):
    """表1：直瞄武器对人员单位攻击等级。units = 车/班数(1-4)，dist = 距离格。
    超出表列范围（即超出该武器对人员的有效射程）返回 None。"""
    row = T["atk_person"].get(weapon_name)
    if not row:
        return None
    arr = row.get(str(min(max(int(units), 1), 4)))
    if not arr or dist >= len(arr):
        return None
    return arr[dist]


def atk_level_vehicle(weapon_name, dist):
    """表2：直瞄武器对车辆单位攻击等级；超出射程返回 None。"""
    arr = T["atk_vehicle"].get(weapon_name)
    if not arr or dist >= len(arr):
        return None
    return arr[dist]


def atk_level_elev_fix(dh, dist):
    """表3：高度差对攻击等级的修正（dh 为米，dist 为格）。"""
    bands = sorted(int(k) for k in T["atk_elev_fix"].keys())
    band = None
    for b in bands:
        if dh >= b:
            band = b
    if band is None:
        return 0
    row = T["atk_elev_fix"][str(band)]
    if dist < 1:
        dist = 1
    return row.get(str(dist), 0)


def atk_level_air(weapon_name, dist):
    """表9：反无人机武器攻击等级。"""
    arr = T["atk_air"].get(weapon_name)
    if not arr:
        return None
    if dist >= len(arr):
        return None
    return arr[dist]


# ---------------- 状态/地形修正（作用于随机数） ----------------

def situational_mod(shooter_stunned=False, shooter_moving=False,
                    terrain=None, target_hidden=False, target_moving=False,
                    target_stacked=False):
    """手册 4.6.2 / 表5 / 表7 共同的射击/目标状态修正。"""
    m = T["state_mods"]
    v = 0
    if shooter_stunned:
        v += m["shooter_stunned"]
    if shooter_moving:
        v += m["shooter_moving"]
    if terrain == "居民地":
        v += m["terrain_town"]
    elif terrain == "丛林地":
        v += m["terrain_jungle"]
    elif terrain in ("工事", "掩体", "建筑物"):
        v += m["terrain_fort"]
    if target_hidden:
        v += m["target_hidden"]
    if target_moving:
        v += m["target_moving"]
    if target_stacked:
        v += m["target_stacked"]
    return v


# ---- 手册只说「作不利修正」、但**未给数值**的目标状态 ----
# 官方软件（`Caijue.asp` 的「目标状态」字段）认得它们，
# 手册第 30 页表7 只列了 机动/掩蔽/堆叠 三项，其余只写"作不利修正"。
# 这些状态**不能静默按 0 处理**，否则引擎会"看起来实现了"——
# 必须显式报出，交由实测标定（见 `docs/未确证裁决表.md` T7-miss）。
UNNUMBERED_STATES = ("卧倒", "行军", "水上", "车内")


def unnumbered_states(states):
    """从目标状态集合里挑出「手册未给修正数值」的那些（返回列表，保序）。"""
    return [s for s in (states or ()) if s in UNNUMBERED_STATES]


def fix_person_result(corrected_roll):
    """表7：人员伤亡结果修正。"""
    for lo, hi, v in T["person_result_fix"]:
        if lo <= corrected_roll <= hi:
            return v
    return 0


# ---------------- 战斗结果 ----------------

def result_person(attack_level, roll):
    """表6：直瞄武器对人员单位结果。返回 'P'(压制) / 0 / 1 / 2。"""
    a = min(max(int(attack_level), 1), 10)
    row = T["person_result"].get(str(min(max(int(roll), 2), 12)))
    if not row:
        return INVALID
    v = row[a - 1]
    return v


def result_grenade(roll):
    """表8：手雷结果。返回 0 或 1。"""
    return T["grenade_result"].get(str(min(max(int(roll), 2), 12)), 0)


def armor_known(armor):
    """该装甲级别是否在手册表5 中有定义。

    官方软件（`Caijue.asp` 模板）还会出现「反应装甲」，但**手册表5 没有它**，
    因此其修正值不可知。这种情况必须显式报出，不能静默当成 0。
    """
    return armor in T["table5"]["by_armor"]


def armor_mod(corrected_roll, armor):
    """表5：按防护力对（车辆）战损随机数再做修正。"""
    tbl = T["table5"]["by_armor"].get(armor)
    if not tbl:
        return 0
    key = str(min(max(int(corrected_roll), -3), 11))
    return tbl.get(key, 0)


def result_vehicle(attack_level, units, corrected_roll):
    """表4：直瞄武器对车辆单位战斗结果 → 消灭车数（0-4）。"""
    n = min(max(int(units), 1), 4)
    cols = T["table4_columns"].get(str(n))
    if not cols:
        return 0
    a = min(max(int(attack_level), 1), 10)
    col = cols.get(str(a))
    if not col:
        return 0
    row = T["table4"].get(str(min(max(int(corrected_roll), 2), 12)))
    if not row:
        return 0
    if col - 1 >= len(row):
        return 0
    return row[col - 1]


def result_air(attack_level, roll):
    """表10：反无人机战斗结果。返回 1=歼灭，0=无效。"""
    a = min(max(int(attack_level), 1), 11)
    row = T["air_result"].get(str(min(max(int(roll), 2), 12)))
    if not row:
        return 0
    return row[a - 1] if a - 1 < len(row) else 0


# ---------------- 战果阶梯（手册 4.6.1 / 4.6.2(4)(5)(6)） ----------------
# 手册把"无效 / 压制 / 消灭 n"写成一条数轴，4.6.2(5)(6) 就是一条加法：
#     最终战果 = 战斗结果 + 修正结果
#     > 0 → 消灭 n（车辆同时压制，人员互斥） ; = 0 → 压制 ; < 0 → 无效
# 表4 的空白格与表6 的空串都表示"无效"，它在数轴上是 -1（比"压制 0"低一档）。
INVALID_LEVEL = -1


def ladder_of(value):
    """把表4/表6 的原始格值转成战果阶梯值（无效 = -1 / 压制 = 0 / 消灭 n = n）。"""
    if value == PUSH:
        return 0
    if value in ("", None):
        return INVALID_LEVEL
    v = int(value)
    return v if v > 0 else INVALID_LEVEL


def combine_result(table_value, fix):
    """最终战果 = 战斗结果 + 修正结果（手册 4.6.2(5)(6)）。

    ⚠️ 但**加法是在「消灭数」上做的**，不是在有符号阶梯值上：
    表4/表6 的空白格（页面显示「无效」）等于「消灭 0 个」，
    一个正的修正结果会直接把它抬成「消灭 n」。

    实测证据（练习服 7527，裁决 #129365，人员 攻级8 / 随机数1=5）：
    页面写「战斗结果 = 无效、修正结果 = +1、最终战果 = 1」。
    若按“无效 = −1”相加会得到 0（压制），与页面不符；
    按“无效 = 消灭 0”相加得 1 ✓。
    同批还观察到：无效 + 0 → 无效（#129366）、压制 + 0 → 压制（#129373）、
    消灭1 + （−1）→ 压制、消灭1 + （−3）→ 无效 ⇒ 其余情形与阶梯加法等价。
    """
    if table_value in (INVALID, None, ""):
        return fix if fix > 0 else INVALID_LEVEL
    return ladder_of(table_value) + fix


# ---------------- 高层裁决入口 ----------------

def direct_fire(shooter, target, weapon_name, dist, dh_m=0, roll=None,
                terrain=None, log=None, rng=None, target_states=None,
                fix_roll=None):
    """一次直瞄射击裁决。

    **两颗骰（官方口径）**：
      - `roll`（随机数1）→ 查表4/表6 得**战斗结果**（**不受状态修正**）；
      - `fix_roll`（随机数2）→ `修正后随机数 = fix_roll + 状态修正`，
        再查表5（车辆防护力）或表7（人员）得**修正结果**；
      - `最终战果 = 战斗结果 + 修正结果`（>0 消灭 / =0 压制 / <0 无效）。

    该口径由 27 条真实裁决明细反查确认（`tools/verify_unfavorable.py` C/D 段），
    也与手册表5 表头「随机数（修正后）」+ 4.6.2(5)(6) 的措辞一致。

    返回 dict(result=..., kills=..., suppressed=..., roll=..., fix_roll=..., level=..., detail=...)
    shooter/target 需要有 typeid / armor / units / moving / stunned / hidden / stacked
    target_states：目标的**文字状态**（如「卧倒」「车内」）。手册对其中若干只说
        「作不利修正」而未给数值，此类状态会被显式写进 detail 告警，不静默当 0。
    """
    if isinstance(dist, bool) or not isinstance(dist, int) or dist < 0:
        raise ValueError("dist must be a nonnegative integer")
    detail = []
    is_air_target = target.get("typeid") == 3
    is_person_target = target.get("typeid") in (1, 4) and target.get("typeid") != 3

    # 1) 攻击等级
    if is_air_target:
        # 手册 4.10：巡飞弹速度极快，不可被任何武器打击
        if target.get("name") == "巡飞弹":
            return dict(result="IMMUNE", kills=0, suppressed=False,
                        detail=["巡飞弹不可被任何武器打击"])
        air_name = CRT_AIR_NAME.get(weapon_name, weapon_name)
        lvl = atk_level_air(air_name, dist)
        if lvl is None:
            return dict(result="NO_WEAPON", kills=0, suppressed=False,
                        detail=["该武器不能打击空中单位"])
        table_used = "表9+表10"
    elif target.get("typeid") == 2:
        lvl = atk_level_vehicle(weapon_name, dist)
        if weapon_name == "步兵轻武器":
            lvl = atk_level_person("步兵轻武器", shooter.get("units", 1), dist)
        if lvl is None:
            return dict(result="NO_WEAPON", kills=0, suppressed=False,
                        detail=["该武器不能打击车辆"])
        table_used = "表2+表4"
    else:
        lvl = atk_level_person(weapon_name, shooter.get("units", 1), dist)
        if lvl is None:
            return dict(result="NO_WEAPON", kills=0, suppressed=False,
                        detail=["超出该武器对人员的有效射程或不能打击人员"])
        table_used = "表1+表6"

    if lvl is not None and lvl <= 0 and not is_air_target:
        return dict(result=INVALID, kills=0, suppressed=False, roll=roll,
                    level=0, detail=["攻击等级为 0，无法造成战果"], table=table_used)

    detail.append("攻击等级(表)=%s" % lvl)

    # 2) 高度差修正
    if not is_air_target:
        fix = atk_level_elev_fix(abs(dh_m), dist)
        if fix:
            lvl = max(0, lvl + fix)
            detail.append("高度差修正 %+d → %d" % (fix, lvl))

    # 3) 掷两颗骰（随机数1 查战斗结果；随机数2 供表5/表7 的「修正后随机数」）
    if roll is None:
        roll = roll2d6(rng)
    if fix_roll is None:
        fix_roll = roll2d6(rng)
    smod = situational_mod(
        shooter_stunned=shooter.get("stunned", False),
        shooter_moving=shooter.get("moving", False),
        terrain=terrain,
        target_hidden=target.get("hidden", False),
        target_moving=target.get("moving", False),
        target_stacked=target.get("stacked", False),
    )
    fixed = fix_roll + smod          # 手册表5 表头：「随机数（修正后）」
    detail.append("随机数1 %d（查战斗结果）；随机数2 %d + 状态修正 %+d → 修正后随机数 %d"
                  % (roll, fix_roll, smod, fixed))
    gaps = unnumbered_states(target_states)
    if gaps:
        detail.append("⚠ 目标状态「%s」手册只说『作不利修正』、未给数值，未作修正"
                      "（待实测标定，见 docs/未确证裁决表.md T7-miss）" % "／".join(gaps))

    # 4) 结果查询
    if is_air_target:
        # 空中目标：手册没有对空的"修正表"（表5/表7 只覆盖车辆/人员），
        # 故此处的状态修正只能折进表10 的行号 —— **未确证，待实测**。
        if lvl is None or lvl <= 0:
            return dict(result=INVALID, kills=0, suppressed=False, roll=roll,
                        fix_roll=fix_roll, level=lvl,
                        detail=detail + ["攻击等级为 0，无法造成战果"],
                        table=table_used)
        r = result_air(lvl, roll + smod)
        return dict(result=("歼灭" if r else INVALID), kills=1 if r else 0,
                    suppressed=False, roll=roll, fix_roll=fix_roll, level=lvl,
                    detail=detail, table=table_used)

    if target.get("typeid") == 2:
        # 步兵轻武器打车辆：表2 没有它，攻级已在上面退到表1（见第 1 步）。
        # 但**打人员**必须走表6（表4 是"对车辆单位战斗结果"），故这里只按目标类型分流。
        raw = result_vehicle(lvl, shooter.get("units", 1), roll)
        base = ladder_of(raw)
        detail.append("表4[攻击等级 %d][随机数1 %d] → 战斗结果 %d" % (lvl, roll, base))
        # 表5：修正后随机数 → 防护力修正
        armor = target.get("armor", "无装甲")
        if not armor_known(armor):
            detail.append("⚠ 装甲级别「%s」不在手册表5 中，未作修正（需人工确认）" % armor)
        am = armor_mod(fixed, armor)
        detail.append("表5[%s][修正后随机数 %d] → 修正结果 %+d" % (armor, fixed, am))
        total = combine_result(raw, am)                    # 手册 4.6.2(5)(6)
        detail.append("最终战果 %d = 战斗结果 %s + 修正结果 %+d"
                      % (total, base if raw else "无效(消灭0)", am))
        if base > 0 and total == 0:
            detail.append("战果被防护力修正为 0 → 仅压制（手册 4.6.2(5)）")
        elif total < 0:
            detail.append("最终战果为负 → 无效（手册 4.6.2(6)）")
        kills = max(0, total)
        # 手册 4.6.2(4)：对车辆只要产生有效战果（含被防护力修正为 0）就**同时**压制。
        suppressed = total >= 0 and (base > 0 or total > 0)
        return dict(result=(kills or INVALID), kills=kills, suppressed=suppressed,
                    kills_raw=max(0, base), final=total, roll=roll,
                    fix_roll=fix_roll, level=lvl, detail=detail, table=table_used)

    # 人员：表6（战斗结果）+ 表7（伤亡结果修正）
    rv = result_person(lvl, roll)
    base = ladder_of(rv)
    rf = fix_person_result(fixed)
    total = combine_result(rv, rf)
    detail.append("表6[攻击等级 %d][随机数1 %d] → 战斗结果 %s"
                  % (lvl, roll, "无效" if rv == INVALID else
                     ("压制" if rv == PUSH else rv)))
    detail.append("表7[修正后随机数 %d] → 结果修正 %+d" % (fixed, rf))
    detail.append("最终战果 %d = 战斗结果 %s + 结果修正 %+d"
                  % (total, base if rv != INVALID else "无效(消灭0)", rf))
    # 手册 4.6.1：人员战果为"消灭 n 个班 / 无效 / 压制"三者之一，消灭与压制**互斥**。
    # （车辆才规定"产生有效战果同时造成压制"，见 4.6.2(4)。）
    if total > 0:
        detail.append("表6 → 消灭 %d 班" % total)
        return dict(result=total, kills=total, suppressed=False, kills_raw=max(0, base),
                    final=total, roll=roll, fix_roll=fix_roll, level=lvl,
                    detail=detail, table=table_used)
    if total == 0:
        detail.append("表6 → 压制")
        return dict(result=PUSH, kills=0, suppressed=True, kills_raw=max(0, base),
                    final=total, roll=roll, fix_roll=fix_roll, level=lvl,
                    detail=detail, table=table_used)
    detail.append("最终战果为负 → 无效（手册 4.6.2(6)）")
    return dict(result=INVALID, kills=0, suppressed=False, kills_raw=max(0, base),
                final=total, roll=roll, fix_roll=fix_roll, level=lvl,
                detail=detail, table=table_used)
