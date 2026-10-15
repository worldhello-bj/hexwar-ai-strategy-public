# -*- coding: utf-8 -*-
"""「不利修正」专项验证（手册 4.6.2(1)(2)(3) / 表7 / 表5 的联动）。

手册里「不利修正」有两类写法，本脚本把两类都变成可执行断言：

  ① **明载数值**（表7 / 4.6.2）：
     射击算子 被压制 −1、机动中 −1；目标 机动 −2、掩蔽 −2、堆叠 +2；
     目标地形 居民地 −1、丛林地 −2、工事 −4。
  ② **只说"作不利修正"、未给数值**：目标状态 卧倒 / 行军 / 水上 / 车内
     （官方软件认得，见 `docs/未确证裁决表.md` T7-miss）。
     这类**必须显式告警**，不能静默按 0 —— 静默 0 会让引擎"看起来实现了"。

三段验证：
  A. ① 的数值逐项对照手册，并用**差分实验**证明每一项都真的会削减战果（不是死数据）。
  B. ② 被显式报出（`crt.UNNUMBERED_STATES` / `unnumbered_states()` / `direct_fire` 告警）。
  C. 用已采的 **27 条真实裁决明细**（`Caijue.asp?ID=`）反查官方查表模型：
         修正后随机数 = 随机数2 + 射击算子状态修正 + 目标状态修正
         修正结果     = 表5[装甲][修正后随机数]
         最终战果     = 战斗结果 + 修正结果      （>0 → 歼灭 n；=0 → 压制；<0 → 无效）
     断言逐条命中 ⇒ 表7 的 −1 / −2 数值与 表5 的 band 被线上数据确认。

运行：``py -3.13 tools/verify_unfavorable.py``
"""
from __future__ import annotations

import glob
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import crt                                    # noqa: E402

T = crt.T
# 采集数据已封存到 archive/exploration/（见该目录 README）；
# 未封存的检出或 train/ 自包含副本里，老路径仍然存在 —— 两者都支持。
CAP = os.path.join(ROOT, "archive", "exploration", "capture", "7527_transport_capture")
if not os.path.isdir(CAP):
    CAP = os.path.join(ROOT, "7527_transport_capture")

PASS: list = []
FAIL: list = []
SKIP: list = []


def check(name, ok, note=""):
    (PASS if ok else FAIL).append(name)
    print(("  [通过] " if ok else "  [失败] ") + name + (("　— " + note) if note else ""))


def skip(name, note=""):
    SKIP.append(name)
    print("  [跳过] " + name + (("　— " + note) if note else ""))


# ---- 手册第 30 页表7 / 4.6.2 逐字誊出的修正值（改这里必须同步改表7） ----
MANUAL_MODS = {
    "shooter_stunned": -1,      # 射击算子被压制 −1
    "shooter_moving": -1,       # 射击算子机动中 −1
    "target_moving": -2,        # 目标 机动 −2
    "target_hidden": -2,        # 目标 掩蔽 −2
    "target_stacked": 2,        # 目标 堆叠 +2（有利）
    "terrain_town": -1,         # 目标地形 居民地 −1
    "terrain_jungle": -2,       # 丛林地 −2
    "terrain_fort": -4,         # 工事 −4
}

# ---- 官方软件认得、手册未给数值的目标状态（不得臆造） ----
UNNUMBERED = ("卧倒", "行军", "水上", "车内")

SM = T.get("state_mods") or {}

print("\n== A. 手册明载数值的不利修正（表7 / 4.6.2）==")
for key, val in MANUAL_MODS.items():
    check("state_mods[%s] == %+d" % (key, val), SM.get(key) == val,
          "实际=%s" % SM.get(key))
extra = sorted(set(SM) - set(MANUAL_MODS))
check("没有手册以外的修正项（防臆造）", not extra,
      ("多出=%s" % extra) if extra else "")


def veh_fire(weapon, dist, roll, armor="复合装甲", **flags):
    """走完整 direct_fire 链路（含表4/表5），用于差分实验。

    两颗骰都固定为同一个值，使差分可重现（真实对局里两颗骰独立）。
    """
    shooter = dict(typeid=2, units=1,
                   moving=flags.get("shooter_moving", False),
                   stunned=flags.get("shooter_stunned", False))
    target = dict(typeid=2, armor=armor,
                  hidden=flags.get("target_hidden", False),
                  moving=flags.get("target_moving", False),
                  stacked=flags.get("target_stacked", False))
    return crt.direct_fire(shooter, target, weapon, dist, dh_m=0, roll=roll,
                           fix_roll=roll,
                           terrain=flags.get("terrain"),
                           target_states=flags.get("target_states"))


def scenarios(limit=10, per_weapon=2):
    """挑出 表4 在骰值区间上有跨度的 (武器, 距离)，让差分有意义。

    每个武器最多取 `per_weapon` 个距离，避免所有场景挤在同一个武器上。
    """
    out = []
    for weapon, arr in T["atk_vehicle"].items():
        took = 0
        for dist, lvl in enumerate(arr):
            if not lvl:
                continue
            col = [crt.result_vehicle(lvl, 1, r) for r in range(2, 13)]
            if max(col) > min(col):
                out.append((weapon, dist, lvl))
                took += 1
            if took >= per_weapon or len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out


SC = scenarios()
print("\n  （差分场景 %d 个：%s）" % (len(SC), ", ".join("%s@%d格" % (w, d) for w, d, _ in SC)))
if not SC:
    check("找到可做差分的场景", False, "表4 无跨度")

# 每一项「不利修正」都必须真的能削减战果；「堆叠 +2」则必须能提高战果
DIFFS = [
    ("射击算子 被压制 −1", dict(shooter_stunned=True), -1, "down"),
    ("射击算子 机动中 −1", dict(shooter_moving=True), -1, "down"),
    ("目标 机动 −2", dict(target_moving=True), -1, "down"),
    ("目标 掩蔽 −2", dict(target_hidden=True), -1, "down"),
    ("目标地形 居民地 −1", dict(terrain="居民地"), -1, "down"),
    ("目标地形 丛林地 −2", dict(terrain="丛林地"), -1, "down"),
    ("目标地形 工事 −4", dict(terrain="工事"), -1, "down"),
    ("目标 堆叠 +2（有利）", dict(target_stacked=True), 1, "up"),
]
for label, flags, sign, direction in DIFFS:
    hits = []
    for weapon, dist, _lvl in SC:
        for roll in range(2, 13):
            base = veh_fire(weapon, dist, roll)["kills"]
            got = veh_fire(weapon, dist, roll, **flags)["kills"]
            if (direction == "down" and got < base) or (direction == "up" and got > base):
                hits.append((weapon, dist, roll, base, got))
    check("「%s」真的改变战果（差分 %d 例）" % (label, len(hits)), bool(hits),
          ("例：%s@%d格 骰%d：%d→%d" % hits[0]) if hits else "该修正在所选场景里未产生任何差异")

print("\n== B. 手册只说『作不利修正』、未给数值的状态 ==")
for st in UNNUMBERED:
    check("「%s」不被当成有数值的修正项" % st, st not in SM and st not in MANUAL_MODS)
check("crt.UNNUMBERED_STATES 覆盖四项",
      set(crt.UNNUMBERED_STATES) == set(UNNUMBERED),
      "实际=%s" % (list(crt.UNNUMBERED_STATES),))
check("unnumbered_states() 能从中挑出卧倒",
      crt.unnumbered_states(["卧倒", "掩蔽", "车内"]) == ["卧倒", "车内"])
check("unnumbered_states(None) == []", crt.unnumbered_states(None) == [])

_w, _d, _lvl = SC[0] if SC else ("重型坦克炮", 1, 6)
r = veh_fire(_w, _d, 8, target_states=["卧倒"])
joined = "；".join(r["detail"])
check("direct_fire 显式告警「卧倒」未标定",
      "卧倒" in joined and "未给数值" in joined and "未作修正" in joined, joined[-80:])
check("「卧倒」未被静默折算成数值（状态修正仍为 +0）",
      "状态修正 +0" in joined, joined[:80])

print("\n== C. 真实裁决明细反查官方查表模型 ==")


def load_samples():
    """读封存区里的实测裁决明细（`archive/exploration/capture/7527_transport_capture/**`）。

    ⚠ 解析器 `hex_caijue_tabulate.py` 属**探索工具**，已随封存搬到
    `archive/exploration/tools/`；这里两处都找，都找不到就返回 None（C 段自动跳过）。
    """
    mod = None
    for cand in (os.path.join(ROOT, "tools", "hex_caijue_tabulate.py"),
                 os.path.join(ROOT, "archive", "exploration", "tools",
                              "hex_caijue_tabulate.py")):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("_caijue_tab", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            break
    if mod is None:
        return None
    rows = []
    files = sorted(set(glob.glob(os.path.join(CAP, "**", "caijue*.html"), recursive=True)))
    for f in files:
        raw = open(f, "rb").read()
        if not raw.strip():
            continue
        d = mod.parse(raw)
        d["id"] = re.sub(r"\D", "", os.path.basename(f))
        rows.append(d)
    rows.sort(key=lambda x: int(x["id"] or 0))
    return rows


def num(s, default=None):
    m = re.search(r"-?\d+", (s or "").replace("，", ","))
    return int(m.group()) if m else default


def fix_of(s):
    s = (s or "").strip()
    return 0 if s in ("", "不修正") else num(s, 0)


def final_ok(text, total):
    """最终战果字段与「战斗结果 + 修正结果」的一致性判据。"""
    text = (text or "").strip()
    if total > 0:
        return text.startswith(str(total))
    if total == 0:
        return text == "压制"
    return text == "无效"


rows = load_samples()
if not rows:
    skip("实测样本（封存于 `archive/exploration/capture/`，属服务端数据、不入库）",
         "无样本时无法反查；对账结论见 D 段的委托")
else:
    check("实测样本数不少于历史基线 17 条（当前 %d）" % len(rows), len(rows) >= 17)

    hit_fix = hit_sum = hit_eng = 0
    bad_fix, bad_sum = [], []
    used = 0
    unmodeled = {}
    for d in rows:
        lvl = num(d["攻击等级"])
        r1 = num(d["随机数"])
        r2 = num(d.get("x21"))
        armor = (d["装甲级别"] or "").strip()
        obs_fix = fix_of(d["修正结果"])
        res = num(d["战斗结果"], 0)
        s_txt = (d["射手状态"] or "").strip()
        t_txt = (d["目标状态"] or "").strip()
        # 状态文字 → 引擎里对应的修正项（数值一律取自 state_mods，不另抄一份）
        if s_txt in ("机动中", "被压制", ""):
            sf = SM["shooter_moving"] if s_txt == "机动中" else \
                 (SM["shooter_stunned"] if s_txt == "被压制" else 0)
        else:
            unmodeled.setdefault("射手状态=" + s_txt, 0)
            unmodeled["射手状态=" + s_txt] += 1
            continue
        if t_txt in ("机动", "掩蔽", ""):
            tf = SM["target_moving"] if t_txt == "机动" else \
                 (SM["target_hidden"] if t_txt == "掩蔽" else 0)
        elif t_txt in crt.UNNUMBERED_STATES:
            unmodeled.setdefault("目标状态=" + t_txt, 0)
            unmodeled["目标状态=" + t_txt] += 1
            continue                      # 手册无数值 → 不能拿它做拟合
        else:
            unmodeled.setdefault("目标状态=" + t_txt, 0)
            unmodeled["目标状态=" + t_txt] += 1
            continue
        if None in (lvl, r1, r2):
            continue
        used += 1
        # 官方模型：修正结果 = 表5[装甲][随机数2 + 状态修正合计]
        want_fix = crt.armor_mod(r2 + sf + tf, armor)
        if want_fix == obs_fix:
            hit_fix += 1
        else:
            bad_fix.append((d["id"], r2, sf, tf, armor, obs_fix, want_fix))
        if final_ok(d["最终战果"], res + obs_fix):
            hit_sum += 1
        else:
            bad_sum.append((d["id"], res, obs_fix, d["最终战果"]))
        # 引擎口径对比：它用 随机数1 而不是 随机数2
        if crt.armor_mod(r1 + sf + tf, armor) == obs_fix:
            hit_eng += 1

    print("  ＊ 本段（按「装甲级别」分表的旧口径）已废弃：无人察打车 typeid=2 但装甲级别"
          "就是「无装甲」，会被误当人员而用错表（表4 vs 表6）；结论以 D 段的委托为准。"
          "（旧口径可建模样本 %d 条，未建模 %s）" % (used, unmodeled or "无"))

    # D. 表4 对账：证明「随机数1 不带状态修正」（否则 表4 命中率会崩）
    print("\n== D. 表4 对账（★ 已委托给 `tools/verify_live_crttables.py`）==")
    # 早前这里用「装甲级别」区分人员/车辆，但**无人察打车（UGV，typeid=2）的装甲级别
    # 就是「无装甲」** ⇒ 会把 UGV 当人员、用错表（表4 vs 表6），产生假残差。
    # 现在统一由那个工具做：它读实验报告里的真实 `typeid`，并对三链路（战斗结果 /
    # 修正结果 / 最终战果）逐条对账，有残差则 exit 1。
    import subprocess
    _here = os.path.dirname(os.path.abspath(__file__))
    _checker = os.path.join(_here, "verify_live_crttables.py")
    _proc = subprocess.run([sys.executable, _checker], capture_output=True,
                           cwd=os.path.dirname(_here))
    _out = _proc.stdout.decode("utf-8", "replace") + _proc.stderr.decode("utf-8", "replace")
    check("实测裁决 × 引擎 三条链路零残差（verify_live_crttables.py）",
          _proc.returncode == 0,
          "；".join(l.strip() for l in _out.splitlines() if l.strip())[:300])
    # 实测 #129365（人员 攻级8/随机数1=5：页面写「战斗结果=无效、修正结果=+1、最终战果=1」）
    # ⇒ 平台的加法是**按消灭数**做的：空白格 = 消灭 0，正修正直接抬成消灭 n。
    check("无效（空白格）+ 正修正 = 消灭 n（实测 #129365）",
          crt.combine_result("", 1) == 1 and crt.combine_result("", 0) == crt.INVALID_LEVEL,
          "combine_result('', 1)=%s，combine_result('', 0)=%s"
          % (crt.combine_result("", 1), crt.combine_result("", 0)))
    check("表4 列映射的实测结论：攻级10 列号 ∈ {16,17}",
          all(crt.T["table4_columns"][u]["10"] in (16, 17) for u in ("1", "2", "3", "4")),
          "units=1..4 现在分别是 %s"
          % [crt.T["table4_columns"][u]["10"] for u in ("1", "2", "3", "4")])
    check("引擎采用两颗骰（随机数1 定战斗结果、随机数2 定修正结果）",
          crt.direct_fire(dict(typeid=2, units=1), dict(typeid=2, armor="复合装甲"),
                          "重型坦克炮", 1, roll=7, fix_roll=4)["roll"] == 7,
          "⇒ 随机数1 查表4/表6，随机数2 + 状态修正 查表5/表7，两者相加")

print("\n" + "=" * 60)
print("不利修正验证：%d/%d 通过%s" % (len(PASS), len(PASS) + len(FAIL),
                                    ("，跳过 %d 项" % len(SKIP)) if SKIP else ""))
if FAIL:
    print("未通过：")
    for n in FAIL:
        print("  - " + n)
sys.exit(0 if not FAIL else 1)
