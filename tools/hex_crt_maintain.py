# -*- coding: utf-8 -*-
"""裁决表维护器：把实测样本（Caijue 明细页）聚合、校验、落盘，目标 1:1 复刻。

做什么：
  1. 扫描 7527_transport_capture/**/caijue*.html（含 caijue/ 子目录），逐条解析；
  2. 聚合成三张"实测表"：
     - vehicle[(装甲, 攻击等级)][随机数1] → 战斗结果（表4）
     - person[(攻击等级)][随机数1]        → 战斗结果（表6）
     - state[(目标状态)][随机数2]          → 修正结果（表7/表5）
  3. 校验：**冲突**（同键多值）、**覆盖度**（缺哪些格）、**单调性**（战斗结果是否随骰值单调）；
  4. 落盘 engine/empirical_tables.json + docs/实测裁决表.md。

用法：
    py -3.13 tools/hex_crt_maintain.py
"""
import glob
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hex_combat_round as CR                    # noqa: E402  (复用 parse_caijue)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 采集数据已封存到 archive/exploration/（见该目录 README）；
# 未封存的检出或 train/ 自包含副本里，老路径仍然存在 —— 两者都支持。
CAP = os.path.join(ROOT, "archive", "exploration", "capture", "7527_transport_capture")
if not os.path.isdir(CAP):
    CAP = os.path.join(ROOT, "7527_transport_capture")
OUT_JSON = os.path.join(ROOT, "engine", "empirical_tables.json")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)   # 公开克隆里可能没有 analysis/
OUT_MD = os.path.join(ROOT, "docs", "实测裁决表.md")
os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)   # 公开克隆里可能没有 analysis/

ARMORS = ("复合装甲", "重型装甲", "中型装甲", "轻型装甲", "无装甲")
ROLLS = list(range(2, 13))


def norm_mod(v):
    """修正结果：'不修正' → 0，其余转 int。"""
    v = (v or "").strip()
    if v in ("", "不修正", "无"):
        return 0
    try:
        return int(v)
    except ValueError:
        return None


def norm_kills(v):
    """战斗结果：数字即消灭数；'压制'→'P'；空→''。"""
    v = (v or "").strip()
    if v in ("", "无效"):
        return ""
    if v == "压制":
        return "P"
    try:
        return int(v)
    except ValueError:
        return v


def load_samples():
    files = sorted(set(
        glob.glob(os.path.join(CAP, "**", "caijue*.html"), recursive=True)
        + glob.glob(os.path.join(CAP, "**", "caijue", "*.html"), recursive=True)))
    out = []
    for f in files:
        raw = open(f, "rb").read()
        if not raw.strip():
            continue
        d = CR.parse_caijue(raw)
        d["id"] = re.sub(r"\D", "", os.path.basename(f))
        d["file"] = os.path.relpath(f, CAP)
        out.append(d)
    return out


def main():
    samples = load_samples()
    veh = defaultdict(lambda: defaultdict(Counter))     # (armor, lvl) -> roll1 -> kills
    per = defaultdict(lambda: defaultdict(Counter))     # lvl -> roll1 -> kills
    st = defaultdict(lambda: defaultdict(Counter))      # state -> roll2 -> mod

    skipped = 0
    for d in samples:
        lvl = (d["攻击等级"] or "").strip()
        r1 = (d["随机数"] or "").strip()
        r2 = (d.get("修正前") or "").strip()          # g(21)：修正表的骰（随机数2）
        kills = norm_kills(d["战斗结果"])
        mod = norm_mod(d["修正结果"])
        arm = (d["装甲级别"] or "").strip()
        tstate = (d["目标状态"] or "").strip()
        if not lvl.isdigit() or not r1.isdigit():
            skipped += 1
            continue
        if arm in ARMORS or arm == "":
            if arm:
                veh[(arm, lvl)][r1][kills] += 1
            else:
                per[lvl][r1][kills] += 1
        else:
            # 认不出的装甲级别（如"反应装甲"）单独记
            veh[(arm, lvl)][r1][kills] += 1
        if r2.isdigit() and mod is not None and tstate:
            st[tstate][r2][mod] += 1

    # ---- 冲突与覆盖 ----
    conflicts = []
    for tag, tab in (("表4车辆", veh), ("表6人员", per)):
        for key, rolls in tab.items():
            for r, c in rolls.items():
                if len(c) > 1:
                    conflicts.append({"table": tag, "key": list(key) if isinstance(key, tuple)
                                      else key, "roll": r, "values": dict(c)})
    for state, rolls in st.items():
        for r, c in rolls.items():
            if len(c) > 1:
                conflicts.append({"table": "表7状态修正", "key": state, "roll": r,
                                  "values": dict(c)})

    # ---- 整理成可读结构（冲突取最多数）----
    def finalize(tab):
        res = {}
        for key, rolls in tab.items():
            kk = "|".join(key) if isinstance(key, tuple) else str(key)
            res[kk] = {}
            for r, c in sorted(rolls.items(), key=lambda x: int(x[0])):
                val, n = c.most_common(1)[0]
                res[kk][r] = {"value": val, "n": n,
                              "dissent": {k: v for k, v in c.items() if k != val}}
        return res

    empirical = {
        "generated_from": "%d 条裁决样本" % len(samples),
        "note": "由实测样本聚合；value 为众数，n 为样本数，dissent 为异见值。"
                "**在 conflicts 清空前不要直接覆盖 tables.json**。",
        "vehicle_result": finalize(veh),
        "person_result": finalize(per),
        "state_mod": finalize(st),
        "conflicts": conflicts,
        "sample_count": len(samples),
        "skipped": skipped,
    }
    with io.open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(empirical, fh, ensure_ascii=False, indent=1)

    # ---- 控制台 + Markdown 报告 ----
    lines = ["# 实测裁决表（自动维护）", "",
             "> 由 `tools/hex_crt_maintain.py` 从 %d 条 `Caijue.asp` 样本聚合。" % len(samples),
             "> **目标：1:1 复刻官方裁决**。`n`=样本数；`dissent`=异见值（应为空）。", ""]
    print("样本 %d 条（跳过 %d）  冲突 %d 处" % (len(samples), skipped, len(conflicts)))

    def dump(tag, tab, fmt):
        print("\n== %s ==" % tag)
        lines.append("## %s" % tag)
        lines.append("")
        for kk in sorted(tab):
            rolls = tab[kk]
            vec = {int(r): v["value"] for r, v in rolls.items()}
            seq = [vec.get(r, "·") for r in ROLLS]
            flag = ""
            if tag.startswith("表4") or tag.startswith("表6"):
                nums = [v for v in seq if isinstance(v, int)]
                if len(nums) >= 3:
                    mono = all(a <= b for a, b in zip(nums, nums[1:]))
                    flag = "" if mono else "  ⚠非单调"
            print("  %-14s %s%s" % (kk, seq, flag))
            lines.append("- `%s`：%s%s" % (kk, seq, flag))
        lines.append("")

    dump("表4 直瞄对车辆（装甲|攻击等级 → 随机数1 → 战斗结果）", empirical["vehicle_result"], None)
    dump("表6 直瞄对人员（攻击等级 → 随机数1 → 战斗结果）", empirical["person_result"], None)
    dump("表7/表5 修正（目标状态 → 随机数2 → 修正结果）", empirical["state_mod"], None)

    if conflicts:
        print("\n!! 冲突 %d 处（同键多值，需更多样本）" % len(conflicts))
        lines.append("## ⚠ 冲突（同键多值）")
        lines.append("")
        for c in conflicts[:40]:
            lines.append("- %s %s 骰%s → %s" % (c["table"], c["key"], c["roll"], c["values"]))
            print("   %s %s 骰%s → %s" % (c["table"], c["key"], c["roll"], c["values"]))
        lines.append("")

    lines.append("## 骰值序列（2..12 顺序）")
    lines.append("")
    lines.append("`·` = 该骰值尚无样本。碰撞到空位即可继续采样补齐。")
    with io.open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n写出 -> %s\n写出 -> %s" % (OUT_JSON, OUT_MD))


if __name__ == "__main__":
    main()
