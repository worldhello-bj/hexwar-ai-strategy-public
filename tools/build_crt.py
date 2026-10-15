# -*- coding: utf-8 -*-
"""从手册重建裁决表为 engine/tables.json。

策略：
- 表1/表2/表6/表7/表8/表9/表10 结构简单，直接按提取到的行誊录。
- 表3 为高度差修正矩阵。
- 表4 用网格线提取（edges）得到列槽，再去掉“所有随机数行都为空”的伪列，
  用 车/班数 行中标出的 攻击等级→列 映射查表。
- 表5 为防护力修正矩阵。
产出附带自检：表4 对每个 (车/班数) 应随攻击等级单调不减。
"""
import os
import re
import json
import pdfplumber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "docs", "WARGAMEAI.pdf")
OUT = os.path.join(ROOT, "engine", "tables.json")
REP = os.path.join(ROOT, "analysis", "crt_build.md")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
rep = []
def w(x=""):
    rep.append(str(x))

# ---------------- 基础表（按行誊录） ----------------

# 表1 直瞄武器对人员单位攻击等级：武器 -> {车/班数: [按距离0..的等级]}
ATK_PERSON = {
    "重型坦克炮": {1: [2]*11, 2: [3]*11, 3: [5]*11, 4: [7]*11},
    "中型坦克炮": {1: [2]*11, 2: [3]*11, 3: [5]*11, 4: [7]*11},
    "速射炮": {1: [4,4,4,4,4,4,3,3,3,2,2], 2: [6,6,6,6,6,6,5,5,4,4,3],
              3: [8,8,8,8,8,8,7,7,6,6,5], 4: [9,9,9,9,9,9,8,8,7,7,6]},
    "车载机枪": {1: [2,2,2,2,2,1,1,1,1,1,1], 2: [3,3,3,3,3,2,2,2,2,1,1],
               3: [4,4,4,4,4,3,3,3,3,2,2], 4: [5,5,5,5,5,4,4,4,4,3,3]},
    "高射机枪": {1: [2,2,2,2,2,1,1,1,1,1,1], 2: [3,3,3,3,3,2,2,2,2,1,1],
               3: [4,4,4,4,4,3,3,3,3,2,2], 4: [5,5,5,5,5,4,4,4,4,3,3]},
    "重机枪": {1: [4,4,4,4,4], 2: [6,6,6,6,6], 3: [8,8,8,8,8], 4: [9,9,9,9,9]},
    "步兵轻武器": {1: [8,6,4,2], 2: [5,4,2,1], 3: [3,2,1,0], 4: [1,1,0,0]},
    "巡飞弹": {1: [4,5,5], 2: [5,6,6], 3: [7,8,8], 4: [8,9,9]},
    "拦截弹": {1: [4,5], 2: [5,6], 3: [7,8], 4: [8,9]},
}

# 表2 直瞄武器对车辆单位攻击等级：武器 -> [按距离0..的等级]
ATK_VEHICLE = {
    "车载导弹":   [5,5,5,5,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
    "重型坦克炮": [10,10,10,10,10,9,9,9,9,8,8,8,7,7,6,5,4,3,2],
    "中型坦克炮": [10,10,10,9,9,9,8,8,7,7,7,6,5,5,2,2],
    "便携导弹":   [7,7,7,8,8,8,8,8,8,8,8],
    "速射炮":     [5,5,4,4,3,3,3,2,2,2,2],
    "高射机枪":   [2]*11,
    "车载机枪":   [0]*11,
    "重机枪":     [5,5,4,4,3],
    "步枪":       [1,1,1,1],
    "巡飞弹":     [5,5,5],
    "拦截弹":     [3,3],
}

# 表3 直瞄武器攻击等级高度差修正：高度差档 -> {距离格: 修正}
ATK_ELEV_FIX = {
    20:  {1: -2, 2: -2, 3: -1, 4: -1, 5: -1},
    40:  {1: -2, 2: -2, 3: -2, 4: -1, 5: -1, 6: -1, 7: -1},
    60:  {1: -3, 2: -2, 3: -2, 4: -2, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1},
    80:  {1: -3, 2: -3, 3: -3, 4: -2, 5: -2, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1},
    100: {1: -4, 2: -3, 3: -3, 4: -3, 5: -2, 6: -2, 7: -2, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1},
    120: {1: -4, 2: -4, 3: -4, 4: -3, 5: -3, 6: -2, 7: -2, 8: -2, 9: -1, 10: -1, 11: -1, 12: -1},
    140: {1: -5, 2: -4, 3: -4, 4: -4, 5: -3, 6: -2, 7: -2, 8: -2, 9: -2, 10: -1, 11: -1, 12: -1},
    160: {1: -5, 2: -5, 3: -5, 4: -4, 5: -3, 6: -3, 7: -2, 8: -2, 9: -2, 10: -2, 11: -2, 12: -1},
}

# 表6 直瞄武器对人员单位战斗结果：随机数 -> 攻击等级1..10
PERSON_RESULT = {
    2:  ["P","P","P","P",1,2,2,2,2,2],
    3:  ["", "P","P","P","P",1,1,1,1,1],
    4:  ["", "", "P","P","P","P","P",1,1,1],
    5:  ["", "", "", "", "P","P","P","P",1,1],
    6:  ["", "", "", "", "", "P","P","P","P","P"],
    7:  ["", "", "", "", "", "", "P","P","P","P"],
    8:  ["", "", "", "", "", "", "P","P","P","P"],
    9:  ["", "", "", "", "", "P","P","P","P",1],
    10: ["", "", "", "P","P","P","P","P",1,1],
    11: ["", "P","P","P","P","P","P",1,1,2],
    12: ["P","P","P","P","P",1,2,2,2,2],
}

# 表7 人员伤亡结果修正：修正后随机数 -> 结果修正
PERSON_RESULT_FIX = [(-10**9, 0, -1), (1, 7, 0), (8, 10**9, 1)]

# 表8 手雷结果：随机数 -> 结果
GRENADE_RESULT = {2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1, 10:0, 11:0, 12:0}

# 表9 反无人机武器攻击等级：武器 -> [按格0..的等级]
ATK_AIR = {
    "高射机枪":     [3,3,2,2,1,1],
    "速射炮":       [3,3,2,2,1,1],
    "防空高炮":     [9,9,8,7,6,5,4,4,3,3,3,2,2,2,2,2,2,2,2,2,2],
    "便携防空导弹": [5,5,5,7,7,7,6,6,4,4,4,4,4,4,4,4,4,4,4,4,4],
    "激光":         [0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
}

# 表10 反无人机战斗结果：随机数 -> 攻击等级1..11（歼灭 或 空）
AIR_RESULT = {
    2:  [1,1,1,1,1,1,1,1,1,1,1],
    3:  [1,1,1,1,1,1,1,1,1,0,0],
    4:  [1,1,1,1,1,1,1,0,0,0,0],
    5:  [1,1,1,1,1,0,0,0,0,0,0],
    6:  [1,1,1,0,0,0,0,0,0,0,0],
    7:  [1,0,0,0,0,0,0,0,0,0,0],
    8:  [1,1,0,0,0,0,0,0,0,0,0],
    9:  [1,1,1,1,0,0,0,0,0,0,0],
    10: [1,1,1,1,1,1,0,0,0,0,0],
    11: [1,1,1,1,1,1,1,1,0,0,0],
    12: [1,1,1,1,1,1,1,1,1,1,0],
}

w("# 裁决表构建")
w()

# ---------------- 表4 / 表5：从 PDF 网格提取 ----------------
with pdfplumber.open(PDF) as pdf:
    page = pdf.pages[28]
    vs, hs = [], []
    for e in page.edges:
        if e.get("orientation") == "v" and (e["y1"] - e["y0"]) > 3:
            vs.append(e["x0"])
        elif e.get("orientation") == "h" and (e["x1"] - e["x0"]) > 3:
            hs.append(e["y0"])

def snap(vals, tol=4):
    vals = sorted(vals)
    out = []
    for v in vals:
        if not out or v - out[-1] > tol:
            out.append(v)
        else:
            out[-1] = (out[-1] + v) / 2
    return out

xs = snap(vs)
words = page.extract_words()

# 表4：随机数 2..12 的数据行，y 区间约 199..392
y_rows = {}
for wd in words:
    t = wd["text"]
    if re.fullmatch(r"\d{1,2}", t) and 199 <= wd["top"] <= 392:
        y = round(wd["top"])
        y_rows.setdefault(y, []).append(wd)

row_ys = sorted(y_rows)
w("- 表4 数据行 y: %s" % row_ys)

# 用数据行的取值 x 中心聚类出真实列槽（比网格线可靠）
centers = []
for y in row_ys:
    for wd in y_rows[y]:
        if wd["x0"] >= 169:
            centers.append((wd["x0"] + wd["x1"]) / 2)
centers.sort()
slots = []
for c in centers:
    if not slots or c - slots[-1][-1] > 7:
        slots.append([c])
    else:
        slots[-1].append(c)
slot_x = [sum(s) / len(s) for s in slots]
ncols = len(slot_x)
w("- 数据列数: %d，列中心: %s" % (ncols, [round(v, 1) for v in slot_x]))

def nearest_slot(cx):
    best, bd = None, 1e9
    for i, sx in enumerate(slot_x):
        d = abs(sx - cx)
        if d < bd:
            best, bd = i + 1, d
    return best, bd

raw_grid = {}
for y in row_ys:
    roll = min(int(wd["text"]) for wd in y_rows[y] if wd["x0"] < 169)
    vals = {}
    for wd in y_rows[y]:
        if wd["x0"] < 169:
            continue
        cx = (wd["x0"] + wd["x1"]) / 2
        ci, d = nearest_slot(cx)
        if d < 10:
            vals[ci] = int(wd["text"])
    raw_grid[roll] = vals

# 表4 的列映射（车/班数 -> {攻击等级: 列槽}），来自手册页 29 表头行
# 用坐标法读表头行（y 约 131/147/163/180）
hdr = {}
for wd in words:
    if re.fullmatch(r"\d{1,2}", wd["text"]) and 125 <= wd["top"] <= 190:
        hdr.setdefault(round(wd["top"]), []).append(wd)
hdr_rows = sorted(hdr)
COL4_RAW = {}
for i, y in enumerate(hdr_rows):
    n = i + 1
    mp = {}
    for wd in sorted(hdr[y], key=lambda w: w["x0"]):
        if wd["x0"] < 169:
            continue
        cx = (wd["x0"] + wd["x1"]) / 2
        ci, d = nearest_slot(cx)
        if d < 10:
            mp[int(wd["text"])] = ci
    COL4_RAW[n] = mp

# 表头数字落在“列左界”上 → 其标注的是右侧相邻列；用单调性反解每档的统一偏移
def violations(colmap, shift):
    v = 0
    for roll in sorted(raw_grid):
        seq = []
        for a in sorted(colmap):
            c = colmap[a] + shift
            seq.append(raw_grid[roll].get(c, 0) if 1 <= c <= ncols else 0)
        v += sum(1 for i in range(1, len(seq)) if seq[i] < seq[i - 1])
    return v

COL4 = {}
w()
w("## 表4 列偏移反解（按单调性择优）")
w()
for n in sorted(COL4_RAW):
    best = None
    for s in range(-2, 4):
        v = violations(COL4_RAW[n], s)
        w("- N=%d shift=%+d 违反=%d" % (n, s, v))
        if best is None or v < best[1]:
            best = (s, v)
    w("  -> 采用 shift=%+d（违反 %d）" % best)
    COL4[n] = {a: c + best[0] for a, c in COL4_RAW[n].items()}
    w()

w()
w("## 表4 攻击等级→列槽映射（车/班数 = 1..4）")
w()
for n in sorted(COL4):
    w("- N=%d: %s" % (n, {k: COL4[n][k] for k in sorted(COL4[n])}))

TABLE4 = {}
for roll in sorted(raw_grid):
    TABLE4[str(roll)] = [raw_grid[roll].get(c, 0) for c in range(1, ncols + 1)]

# 自检：单调性
w()
w("## 表4 单调性自检（同一 车/班数 下，攻击等级升高结果不应下降）")
w()
issues = 0
for n in sorted(COL4):
    for roll in sorted(raw_grid):
        seq = []
        for a in sorted(COL4[n]):
            c = COL4[n][a]
            seq.append(TABLE4[str(roll)][c - 1] if c <= ncols else 0)
        bad = [i for i in range(1, len(seq)) if seq[i] < seq[i - 1]]
        if bad:
            issues += 1
            w("- N=%d roll=%d 序列 %s 在位置 %s 下降" % (n, roll, seq, bad))
w()
w("- 违反单调的 (N,随机数) 组合数: %d" % issues)

# ---------------- 人工复核后的更正（★ 重跑本脚本必须保留） ----------------
# 1) 表10（手册第 31 页）：PDF 提取到的是**左右镜像**。用「歼灭」的词坐标反推手册原表后
#    逐行反转；`tools/verify_tables.py` 有全 11×11 格断言守住它。
#    见 docs/手册逐页核对.md「表10 左右存反 —— 已确认并修复」。
for _r in AIR_RESULT:
    AIR_RESULT[_r] = list(reversed(AIR_RESULT[_r]))
# 2) 表4（手册第 29 页）：该页网格是斜的，个别格无法从 PDF 唯一定位。
#    `table4["7"][16]`（攻级10 / 随机数1=7）由 3 条独立实测样本
#    （128246 / 128249 / 128264）校正为 3（原提取值 4）。
#    `tools/verify_unfavorable.py` D/E 段有零残差断言守住它。
TABLE4["7"][16] = 3

# 3) 表4 **列映射**（`table4_columns`）在右端也会偏：PDF 第 29 页网格斜，
#    用单调性择优反解出的列号在「攻级 10」处偏了 4 列（units=1 得 13、units=2 得 12）。
#    线上 35 条实测样本（`tools/hex_table4_sweep.py` + `hex_experiment_suite.py --modes matrix`）
#    反解出攻级 10 的真实列号 ∈ {16, 17}（列 16 与列 17 在全部随机数行上取值相同 ⇒ 取 17，
#    与 units=3/4 一致）。反解工具 `tools/derive_table4_columns.py`，
#    对账 `tools/verify_live_crttables.py`（改后 0 残差），回归 `tests/test_crt_live.py`。
#    攻级 7 的实测样本允许 {8,9,10} 与现映射 9 相容 ⇒ 未动；攻级 ≤6、8、9 尚无实测样本。
for _n, _a in (("1", "10"), ("2", "10")):
    COL4[int(_n)][int(_a)] = 17

NOTE = ("由 docs/WARGAMEAI.pdf 第 27-31 页重建；表4/表5 存在网格歧义，可人工校正；"
        "表10(air_result) 已按手册从左到右(攻级1..11)重排（2026-10-15 逐页核对修正）；"
        "表4 的 table4[\"7\"][16] 已由 3 条实测样本（128246/128249/128264，"
        "攻级10/随机数1=7）校正为 3（原为 4）；"
        "表4 列映射 table4_columns[1/2][10] 已由 35 条线上样本反解校正为 17（原 13/12）")

data = {
    "note": NOTE,
    "atk_person": {k: {str(n): v for n, v in d.items()} for k, d in ATK_PERSON.items()},
    "atk_vehicle": ATK_VEHICLE,
    "atk_elev_fix": {str(k): {str(d): v for d, v in dd.items()} for k, dd in ATK_ELEV_FIX.items()},
    "person_result": {str(k): v for k, v in PERSON_RESULT.items()},
    "person_result_fix": PERSON_RESULT_FIX,
    "grenade_result": {str(k): v for k, v in GRENADE_RESULT.items()},
    "atk_air": ATK_AIR,
    "air_result": {str(k): v for k, v in AIR_RESULT.items()},
    "table4_columns": {str(k): {str(a): c for a, c in v.items()} for k, v in COL4.items()},
    "table4": TABLE4,
    "table4_ncols": ncols,
    "table5": {
        "cols": "roll,复合装甲,重型装甲,中型装甲,轻型装甲,无装甲",
        "by_armor": {
            "复合装甲": {"-3": -3, "-2": -3, "-1": -3, "0": -3, "1": -2, "2": -2, "3": -1, "4": -1, "5": -1, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 1},
            "重型装甲": {"-3": -3, "-2": -3, "-1": -2, "0": -1, "1": -1, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 1, "11": 2},
            "中型装甲": {"-3": -2, "-2": -2, "-1": -1, "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 1, "10": 1, "11": 2},
            "轻型装甲": {"-3": -1, "-2": -1, "-1": 0, "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 1, "7": 1, "8": 1, "9": 1, "10": 2, "11": 2},
            "无装甲":   {"-3": 0, "-2": 0, "-1": 0, "0": 0, "1": 0, "2": 0, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1, "10": 2, "11": 2},
        },
    },
    "state_mods": {
        "shooter_stunned": -1, "shooter_moving": -1,
        "terrain_town": -1, "terrain_jungle": -2, "terrain_fort": -4,
        "target_hidden": -2, "target_moving": -2, "target_stacked": 2,
    },
}

json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(OUT, "a", encoding="utf-8") as _fh:     # 保留文件尾换行（重建幂等）
    _fh.write("\n")
open(REP, "w", encoding="utf-8").write("\n".join(rep))
print("tables written:", OUT, "cols:", ncols, "monotonic issues:", issues)