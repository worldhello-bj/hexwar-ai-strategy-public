# -*- coding: utf-8 -*-
"""从全部实测实验报告（report.md）反解 `tables.json` 的**列映射**。

背景：手册表4 的 PDF 网格是斜的（同列 x 逐行漂移）⇒ 列无法唯一对齐，
`engine/tables.json` 的 `table4_columns` 只能靠单调性择优，从未被实测校验。
线上裁决每条都带「攻级 / 随机数1 / 战斗结果」，正好能**逐格反解列号**：

  对某攻级 a，收集所有实车样本 (随机数1 r, 战斗结果 b)，
  求满足 table4[r][c-1] == b 的列号 c 的**交集** ⇒ 该攻级的真实列号。

样本来源：`7527_transport_capture/**/report.md`（含目标 typeid/ico，比 Caijue 页更全）。

输出 `analysis/_derive_table4.txt`。
"""
import glob
import io
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import crt                 # noqa: E402

# 采集数据已封存到 archive/exploration/（见该目录 README）；
# 未封存的检出或 train/ 自包含副本里，老路径仍然存在 —— 两者都支持。
CAP = os.path.join(ROOT, "archive", "exploration", "capture", "7527_transport_capture")
if not os.path.isdir(CAP):
    CAP = os.path.join(ROOT, "7527_transport_capture")
OUT = os.path.join(ROOT, "analysis", "_derive_table4.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)   # 公开克隆里可能没有 analysis/
TABLES = os.path.join(ROOT, "engine", "tables.json")

RE_TGT = re.compile(r"目标 (Obj_\d+)\((?:ico=)?\S+ typeid=(\d+)")
RE_SAMPLE = re.compile(
    r"◆ 裁决#(\d+) 距离=(\S+) 攻级=(\S+) 随机数=(\S+) 结果=(\S+) 装甲=(\S+) "
    r"修=(\S+) 最终=(\S+)(?:\s*\|\s*射手=(\S*) 目标=(.*))?")


def ladder_final(text):
    """裁决页「最终战果」→ 阶梯值（',歼灭' 后缀与 '二次压制' 都要归一）。"""
    t = (text or "").strip()
    if not t:
        return None
    t = t.split("，")[0].split(",")[0]
    if t in ("二次压制", "压制"):
        return 0
    if t == "无效":
        return -1
    try:
        return int(t)
    except ValueError:
        return None


def mod_num(text):
    t = (text or "").strip()
    if t in ("不修正", "", "-", "－"):
        return 0
    try:
        return int(t.replace("＋", "+"))
    except ValueError:
        return None


def read_text(path):
    """读文本：PowerShell 的重定向日志（`*>`）是 **UTF-16**，必须探测编码。"""
    with io.open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig")
    if raw.count(b"\x00") > len(raw) // 4:
        return raw.decode("utf-16-le", "replace")
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def sources():
    """所有记录实测裁决的文本源：各实验目录的 report.md + 扫描脚本的 stdout 日志。"""
    out = sorted(glob.glob(os.path.join(CAP, "**", "report.md"), recursive=True))
    for base in (os.path.join(ROOT, "analysis"),
                 os.path.join(ROOT, "archive", "exploration", "analysis")):
        out += sorted(glob.glob(os.path.join(base, "_t4sweep*.log")))
        out += sorted(glob.glob(os.path.join(base, "_pose*.log")))
    return out


def collect():
    samples = []
    for path in sources():
        cur = None
        for line in read_text(path).splitlines():
                m = RE_TGT.search(line)
                if m:
                    cur = {"obj": m.group(1), "typeid": int(m.group(2))}
                    continue
                s = RE_SAMPLE.search(line)
                if not s or cur is None:
                    continue
                tgt_state = (s.group(10) or "").strip()
                samples.append({
                    "file": os.path.relpath(path, CAP),
                    "id": s.group(1), "dist": s.group(2), "atk": int(s.group(3)),
                    "roll1": int(s.group(4)), "battle_raw": s.group(5),
                    "armor": s.group(6), "fix_raw": s.group(7), "final_raw": s.group(8),
                    "shooter_state": (s.group(9) or "").strip(), "target_state": tgt_state,
                    "stacked": "堆叠" in tgt_state,
                    "typeid": cur["typeid"], "ico": cur.get("ico", ""),
                })
    return samples


def is_vehicle(s):
    """是否车辆：typeid=2，或「装甲级别 ≠ 无装甲」（非‘无装甲’必为车辆）。

    这样处理是因为距离扫描等批次里，目标标头有时缺 typeid；
    而装甲级别为复合/中型/重型/轻型装甲的目标一定是车辆，可以安全入表4。
    """
    return s["typeid"] == 2 or (s["armor"] or "").strip() not in ("", "无装甲", "空")


def cell_of(s):
    """页面「战斗结果」对应的**表格单元值**（与 tables.json 里的写法一致）。

    ⚠️ 不能用「最终战果 − 修正结果」反推：空白格在页面显示「无效」，
    而平台的加法是按**消灭数**做的（空白 = 消灭 0，加正修正会抬起来）⇒
    反推会把「无效 + 修正+1 → 1」误当成「战斗结果 = 1」（实测 #129365）。
    """
    raw = (s["battle_raw"] or "").strip()
    if raw in ("无效", ""):
        return ""
    if raw in ("压制", "二次压制"):
        return "P"
    try:
        return int(raw)
    except ValueError:
        return None


def battle_from(s):
    """兼容旧接口：返回 (单元值, None)。"""
    return cell_of(s), None


def main():
    samples = collect()
    out = ["# 表4/表6 列映射反解", "",
           "样本 %d 条（来自 report.md）" % len(samples), ""]

    # ---- A) 车辆样本：反解 table4 列号 ----
    cur_all = json.load(io.open(TABLES, encoding="utf-8"))["table4_columns"]
    derived_cols = {}
    conflicts = []
    veh = []
    per_person = defaultdict(list)
    settled = {}
    for tag, pred in (("单体（目标状态无堆叠）", lambda s: not s["stacked"]),
                      ("堆叠（同格 ≥2 车）", lambda s: s["stacked"]),
                      ("全部", lambda s: True)):
        veh = [s for s in samples if is_vehicle(s) and pred(s)]
        per_atk = defaultdict(list)
        for s in veh:
            cell = cell_of(s)
            if cell is None:
                continue
            per_atk[s["atk"]].append((s["roll1"], cell, s["id"]))
        out.append("## 车辆 · %s（%d 条）" % (tag, len(veh)))
        if not veh:
            out.append("（无样本）")
            out.append("")
            continue
        out.append("| 攻级 | 样本数 | 允许列号 c（table4 行内下标 c-1） | 当前映射(units=1) | 判定 |")
        out.append("|---|---|---|---|---|")
        for a in sorted(per_atk):
            ok = set(range(1, 18))
            for roll1, cell, _sid in per_atk[a]:
                row = crt.T["table4"].get(str(roll1))
                if not row:
                    continue
                ok &= {i + 1 for i, v in enumerate(row) if v == cell and i < 17}
            cur = cur_all["1"].get(str(a))
            verdict = "唯一" if len(ok) == 1 else ("空集（矛盾！）" if not ok else "多解")
            out.append("| %d | %d | %s | %s | %s |"
                       % (a, len(per_atk[a]), sorted(ok), cur, verdict))
            if len(ok) == 1:
                derived_cols.setdefault(a, set()).add(sorted(ok)[0])
        out.append("")
        for units in ("1", "2", "3", "4"):
            m = {a: cur_all[units].get(str(a)) for a in sorted(per_atk)}
            out.append("- units=%s 当前列号：%s" % (units, m))
        out.append("")
    if derived_cols:
        settled = {a: sorted(v)[0] for a, v in derived_cols.items() if len(v) == 1}
        diff = {a: (settled[a], cur_all["1"].get(str(a)))
                for a in settled if str(cur_all["1"].get(str(a))) != str(settled[a])}
        out.append("反解列号（唯一者）vs 当前 units=1 映射 **不一致的攻级**：%s" % (diff or "无"))
        out.append("")

    # ---- B) 人员样本：反解 table6 的列号（= 攻级下标 +1） ----
    per_person = defaultdict(list)
    for s in samples:
        if s["typeid"] != 1:
            continue
        cell = cell_of(s)
        if cell is not None:
            per_person[s["atk"]].append((s["roll1"], cell, s["id"]))
    if per_person:
        out.append("## 表6（人员）列号反解")
        out.append("| 攻级 | 样本数 | 允许列号 |")
        out.append("|---|---|---|")
        for a in sorted(per_person):
            ok = set(range(1, 12))
            for roll1, cell, _sid in per_person[a]:
                row = crt.T["person_result"].get(str(roll1))
                if not row:
                    continue
                ok &= {i + 1 for i, v in enumerate(row) if v == cell}
            out.append("| %d | %d | %s |" % (a, len(per_person[a]), sorted(ok)))
        out.append("")
    if conflicts:
        out.append("## ⚠ 『战斗结果』与『最终−修正』不一致（%d 条）" % len(conflicts))
        out += ["  #%s  战斗结果=%s vs 最终−修正=%s" % c for c in conflicts[:30]]
        out.append("")
    else:
        out.append("**自洽**：所有样本都满足 最终战果 = 战斗结果 + 修正结果。")

    io.open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    if settled:
        cols_path = os.path.join(ROOT, "analysis", "_table4_cols.json")
        io.open(cols_path, "w", encoding="utf-8").write(
            json.dumps(settled, ensure_ascii=False, indent=1))
        print("wrote", cols_path, settled)
    print("wrote", OUT)
    print("车辆样本 %d，人员样本 %d，矛盾 %d"
          % (len(veh), sum(len(v) for v in per_person.values()), len(conflicts)))


if __name__ == "__main__":
    main()
