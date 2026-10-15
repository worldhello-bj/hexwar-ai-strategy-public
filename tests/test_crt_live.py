# -*- coding: utf-8 -*-
"""线上实测裁决样本的**冻结回归**（表4 列映射 / 表6 人员表）。

这些数字来自练习服 7527 的真实裁决明细（`7527_transport_capture/**/caijue_*.html`
与同批 `report.md`），原始数据不入库；这里只冻结**结论性样本**，
用来守住两个曾经错过的地方：

1. **表4 的列映射**（`tables.json: table4_columns`）：
   27 条车辆样本（全部攻级 10）反解出真实列号是 **17**，而 PDF 网格斜出来的映射
   写的是 units=1 → 13、units=2 → 12 ⇒ 10/27 残差。改后：全部吻合。
   证据样本：裁决 #129267..#129350（`experiment_20261015-225856` 等）。
2. **表6（对人员）**：10 条攻级 8 的样本逐格吻合（`result_person` 用 `row[攻级-1]`）。

样本格式：(攻击等级, 随机数1, 页面「战斗结果」的阶梯值, 说明)
页面把「消灭 n」写成数字、把「无效」写成空或“无效”，此处统一成阶梯值
（无效 = −1 / 压制 = 0 / 消灭 n = n），与 `crt.ladder_of` 同口径。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import crt          # noqa: E402

# ---- 车辆（typeid=2）：表4。攻级全部为 10（1 格内 重型坦克炮 / 便携导弹）----
VEHICLE_SAMPLES = (
    # (攻级, 随机数1, 战斗结果阶梯值, 证据裁决号)
    (10, 2, 3, "129274/129321/129347"),
    (10, 3, 3, "129281"),
    (10, 4, 1, "129268/129270/129272/129308/129320/129350"),
    (10, 5, 1, "129310/129312/129322"),
    (10, 6, 1, "129280/129285/129287"),
    (10, 7, 3, "129269/129279/129282/129314/129349"),
    (10, 8, 2, "129323"),
    (10, 9, 2, "129267/129324"),
    (10, 11, 3, "129277"),
)

# ---- 人员（typeid=1）：表6。攻级 8（重型坦克炮 打 人员，1 格）----
PERSON_SAMPLES = (
    (8, 5, 0, "129365 一族：页面写「压制」"),
    (8, 6, 0, "129366"),
    (8, 7, 0, "129367"),
    (8, 11, 1, "129316 一族"),
    (8, 12, 2, "129317 一族"),
)


def test_vehicle_table4_columns_reproduce_live():
    """表4：实测样本必须逐条复现（四种「目标单位数」子表都要一致）。"""
    bad = []
    for atk, roll, expect, who in VEHICLE_SAMPLES:
        for units in (1, 2, 3, 4):
            got = crt.ladder_of(crt.result_vehicle(atk, units, roll))
            if got != expect:
                bad.append((atk, roll, units, expect, got, who))
    assert not bad, "表4 与线上不符：%s" % bad


def test_person_table6_reproduce_live():
    """表6：人员样本必须逐条复现。"""
    bad = []
    for atk, roll, expect, who in PERSON_SAMPLES:
        got = crt.ladder_of(crt.result_person(atk, roll))
        if got != expect:
            bad.append((atk, roll, expect, got, who))
    assert not bad, "表6 与线上不符：%s" % bad


def test_table4_column_for_max_attack_level_is_measured():
    """攻级 10 的列号必须落在实测区间 {16, 17}（PDF 原值 13/12 是错的）。"""
    for units in (1, 2, 3, 4):
        col = crt.T["table4_columns"][str(units)]["10"]
        assert col in (16, 17), "units=%s 攻级10 列号=%s 不在实测区间" % (units, col)


def test_state_mods_match_measured_values():
    """表7 状态修正：射手机动 −1、目标机动 −2、目标掩蔽 −2、目标堆叠 +2（已线上确认）。"""
    assert crt.situational_mod(shooter_moving=True) == -1
    assert crt.situational_mod(target_moving=True) == -2
    assert crt.situational_mod(target_hidden=True) == -2
    assert crt.situational_mod(target_stacked=True) == 2
    # 「机动 堆叠」= 两个词同时成立 ⇒ −2 + 2 = 0
    assert crt.situational_mod(target_moving=True, target_stacked=True) == 0
    # 8 条同格样本的实测组合：射手机动 + 目标机动 + 堆叠 = −1
    assert crt.situational_mod(shooter_moving=True, target_moving=True,
                               target_stacked=True) == -1


def test_final_verdict_is_battle_plus_fix():
    """最终战果 = 阶梯(战斗结果) + 修正结果（手册 4.6.2(5)(6)）。"""
    assert crt.ladder_of("P") + (-2) == -2          # 压制 + (−2) → 无效
    assert crt.ladder_of("1") + 2 == 3              # 消灭1 + 2 → 消灭3
    assert crt.ladder_of("") + 0 == crt.INVALID_LEVEL
