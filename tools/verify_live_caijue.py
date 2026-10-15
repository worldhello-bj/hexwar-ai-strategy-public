# -*- coding: utf-8 -*-
"""把**全部实测裁决明细**（`7527_transport_capture/**/caijue*.html`）与引擎逐格对账。

对的是手册 4.6 的三条链路（与 `engine/crt.py` 一一对应）：

  A) 战斗结果 = 表4（对车辆）/ 表6（对人员）[攻级][随机数1]
  B) 修正结果 = 表5（装甲）/ 表7（人员）[装甲][随机数2 + 射手状态修正 + 目标状态修正]
  C) 最终战果 = 阶梯(战斗结果) + 修正结果（>0 歼灭 n / =0 压制 / <0 无效）

用途：回答「是否完全复刻」——把"已实现"换成"线上 N 条样本逐条吻合"。

退出码 0 = 三条链路全部吻合；1 = 有残差（细节写 `analysis/_live_caijue_check.txt`）。
"""
import glob
import io
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hex_combat_round as CR          # noqa: E402
from engine import crt                 # noqa: E402

# 采集数据已封存到 archive/exploration/（见该目录 README）；
# 未封存的检出或 train/ 自包含副本里，老路径仍然存在 —— 两者都支持。
CAP = os.path.join(ROOT, "archive", "exploration", "capture", "7527_transport_capture")
if not os.path.isdir(CAP):
    CAP = os.path.join(ROOT, "7527_transport_capture")
OUT = os.path.join(ROOT, "analysis", "_live_caijue_check.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)   # 公开克隆里可能没有 analysis/

# 裁决页「射击方状态」→ engine 参数
SHOOTER = {"机动中": "shooter_moving", "被压制": "shooter_stunned", "静止": None}
# 裁决页「目标状态」→ engine 参数（同一格可能并列多个词，如「机动 堆叠」）
TARGET = {"机动": "target_moving", "掩蔽": "target_hidden", "堆叠": "target_stacked"}
TERRAIN = {"居民地": "居民地", "丛林地": "丛林地", "工事": "工事",
           "建筑物": "建筑物", "掩体": "工事"}


def to_int(s):
    """裁决页里的整数（'不修正' → 0）。无法解析返回 None。"""
    s = (s or "").strip().replace("＋", "+")
    if s in ("不修正", "", "-", "－"):
        return 0
    try:
        return int(s)
    except ValueError:
        return None


def display(value):
    """按 4.6.1/4.6.2 把阶梯值渲染成裁决页的样子。"""
    if value > 0:
        return str(value)
    if value == 0:
        return "压制"
    return "无效"


def main():
    files = sorted(glob.glob(os.path.join(CAP, "**", "caijue*.html"), recursive=True))
    rows = []
    for path in files:
        with io.open(path, "rb") as fh:
            raw = fh.read()
        try:
            info = CR.parse_caijue(raw)
        except Exception:                                   # noqa: BLE001
            continue
        info["file"] = os.path.basename(path)
        rows.append(info)

    problems = defaultdict(list)
    checked = Counter()
    dist = defaultdict(Counter)
    skipped = []
    for r in rows:
        atk = to_int(r["攻击等级"])
        roll1 = to_int(r["随机数"])
        roll2 = to_int(r["修正前"])
        armor = (r["装甲级别"] or "").strip()
        obs_battle = (r["战斗结果"] or "").strip()
        obs_fix = to_int(r["修正结果"])
        obs_final = (r["最终战果"] or "").strip()
        if None in (atk, roll1, roll2, obs_fix):
            skipped.append((r["file"], "字段缺失"))
            continue

        # 目标类别：裁决页里人员 = 无装甲；车辆 = 各装甲级别
        if armor in ("", "空"):
            skipped.append((r["file"], "装甲级别=%r（对空/未知）" % armor))
            continue
        is_person = armor == "无装甲"

        # 状态修正（与 engine.crt.situational_mod 同口径）
        kwargs = {}
        for tok in (r["射击方状态"] or "").split():
            if tok in SHOOTER and SHOOTER[tok]:
                kwargs[SHOOTER[tok]] = True
        for tok in (r["目标状态"] or "").split():
            if tok in TARGET:
                kwargs[TARGET[tok]] = True
            if tok in TERRAIN:
                kwargs["terrain"] = TERRAIN[tok]
        smod = crt.situational_mod(**kwargs)

        # A) 战斗结果
        if is_person:
            exp_battle = crt.ladder_of(crt.result_person(atk, roll1))
        else:
            exp_battle = crt.ladder_of(crt.result_vehicle(atk, 1, roll1))
        checked["A"] += 1
        if obs_battle != "" or exp_battle != crt.INVALID_LEVEL:
            # 页面上「无效」可能显示为空串，故只在有字面值时严格比
            if obs_battle and to_int(obs_battle) is not None \
                    and to_int(obs_battle) != exp_battle:
                problems["A 战斗结果"].append(
                    (r["file"], "攻级%s 随机数1 %s 装甲%s 观测%s vs 引擎%s"
                     % (atk, roll1, armor, obs_battle, exp_battle)))
            elif obs_battle == "压制" and exp_battle != 0:
                problems["A 战斗结果"].append(
                    (r["file"], "攻级%s 随机数1 %s 装甲%s 观测压制 vs 引擎%s"
                     % (atk, roll1, armor, exp_battle)))
            elif obs_battle == "无效" and exp_battle != crt.INVALID_LEVEL:
                problems["A 战斗结果"].append(
                    (r["file"], "攻级%s 随机数1 %s 装甲%s 观测无效 vs 引擎%s"
                     % (atk, roll1, armor, exp_battle)))

        # B) 修正结果
        corrected = roll2 + smod
        if is_person:
            exp_fix = crt.fix_person_result(corrected)
        else:
            exp_fix = crt.armor_mod(corrected, armor)
        checked["B"] += 1
        if exp_fix != obs_fix:
            problems["B 修正结果"].append(
                (r["file"], "装甲%s 随机数2 %s + 状态修正 %+d = %d → 观测%s vs 引擎%s"
                 % (armor, roll2, smod, corrected, obs_fix, exp_fix)))
        dist["修正结果"][obs_fix] += 1
        dist["随机数2+修正"][corrected] += 1
        dist["装甲"][armor] += 1
        dist["目标状态"][(r["目标状态"] or "").strip()] += 1

        # C) 最终战果 = 阶梯(战斗结果) + 修正结果
        exp_final = display(exp_battle + obs_fix)
        checked["C"] += 1
        if obs_final and obs_final != exp_final:
            problems["C 最终战果"].append(
                (r["file"], "战斗结果%d + 修正%+d = %d → 观测%s vs 引擎%s"
                 % (exp_battle, obs_fix, exp_battle + obs_fix, obs_final, exp_final)))
        dist["最终战果"][obs_final] += 1

    out = ["# 实测裁决 × 引擎 逐格对账", "",
           "样本 %d 条（解析成功）；对账 %d 条；跳过 %d 条"
           % (len(rows), sum(checked.values()) // 3, len(skipped)), ""]
    out.append("| 链路 | 对账条数 | 吻合 | 残差 |")
    out.append("|---|---|---|---|")
    for k in ("A", "B", "C"):
        n = checked[k]
        bad = len([p for p in problems if p.startswith(k)])
        out.append("| %s | %d | %d | %d |" % (k, n, n - bad, bad))
    out.append("")
    if skipped:
        out.append("## 跳过（不可对账）")
        out += ["  %s  %s" % s for s in skipped[:20]]
        out.append("")
    for key in ("装甲", "目标状态", "随机数2+修正", "修正结果", "最终战果"):
        out.append("- **%s**: %s" % (key, dict(sorted(dist[key].items(),
                                                     key=lambda kv: str(kv[0])))))
    out.append("")
    if problems:
        out.append("## 残差明细")
        for key in sorted(problems):
            out.append("### %s（%d 条）" % (key, len(problems[key])))
            out += ["  %s  %s" % p for p in problems[key][:40]]
            out.append("")
    else:
        out.append("**零残差**：A/B/C 三条链路全部与线上裁决一致。")

    io.open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("wrote", OUT)
    print("样本 %d 条，对账 %d 条，残差 %d 条"
          % (len(rows), sum(checked.values()) // 3, sum(len(v) for v in problems.values())))
    for key in sorted(problems):
        print("  %s: %d" % (key, len(problems[key])))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
