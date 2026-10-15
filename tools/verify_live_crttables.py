# -*- coding: utf-8 -*-
"""实测裁决 × 引擎 对账（**按真实 typeid 分类**版）。

与 `tools/verify_live_caijue.py` 的区别：后者只从 Caijue 页读「装甲级别」，
把 UGV（无人察打车，typeid=2，装甲级别恰好是「无装甲」）误判成人员 ⇒ 用错表。
这里改用实验报告 `report.md`：它逐条记了目标 `typeid`（2=车辆 / 1,4=人员 / 3=空中）。

三条链路：
  A 战斗结果 = 表4（车辆）/ 表6（人员）[攻级][随机数1]
  B 修正结果 = 表5（装甲）/ 表7（人员）[装甲][随机数2 + 射手状态 + 目标状态修正]
  C 最终战果 = 阶梯(战斗结果) + 修正结果
若给了 `--cols file.json`（`{攻级: 列号}`）则 A 用实测列号，否则用引擎现映射（用于对比）。

输出 `analysis/_live_crt_check.txt`；有残差 exit 1。
"""
import argparse
import glob
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import crt                 # noqa: E402

# 采集数据已封存到 archive/exploration/（见该目录 README）；
# 未封存的检出或 train/ 自包含副本里，老路径仍然存在 —— 两者都支持。
CAP = os.path.join(ROOT, "archive", "exploration", "capture", "7527_transport_capture")
if not os.path.isdir(CAP):
    CAP = os.path.join(ROOT, "7527_transport_capture")
OUT = os.path.join(ROOT, "analysis", "_live_crt_check.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)   # 公开克隆里可能没有 analysis/
RE_TGT = re.compile(r"目标 (Obj_\d+)\((?:ico=)?\S+ typeid=(\d+)")
RE_SAMPLE = re.compile(
    r"◆ 裁决#(\d+) 距离=(\S+) 攻级=(\S+) 随机数=(\S+) 结果=(\S+) 装甲=(\S+) "
    r"修=(\S+) 最终=(\S+)(?:\s*\|\s*射手=(\S*) 目标=(.*))?")
SHOOTER = {"机动中": "shooter_moving", "被压制": "shooter_stunned"}
TARGET = {"机动": "target_moving", "掩蔽": "target_hidden", "堆叠": "target_stacked"}
TERRAIN = {"居民地": "居民地", "丛林地": "丛林地", "工事": "工事", "建筑物": "建筑物"}


def cell_value(text):
    """把页面「战斗结果」的字面值还原成**表格值**（`combine_result` 的入参）。

    页面把空白格写成「无效」、把压制写成「压制」；数字原样保留。
    """
    t = (text or "").strip()
    if t in ("无效", ""):
        return crt.INVALID
    if t in ("压制", "二次压制"):
        return crt.PUSH
    try:
        return int(t)
    except ValueError:
        return t


def parse_final(text):
    t = (text or "").strip()
    if not t:
        return None, ""
    head = t.split("，")[0].split(",")[0]
    if head in ("二次压制", "压制"):
        return 0, t
    if head == "无效":
        return -1, t
    try:
        return int(head), t
    except ValueError:
        return None, t


def show_final(value):
    """把阶梯值还原成裁决页的显示形态。

    注意：页面的「无效」对应所有 <0 的值（被 clamp 了），
    所以数值比较必须走这个显示口径，否则 −2 与 −1 会被当成不一致。
    """
    if value > 0:
        return str(value)
    if value == 0:
        return "压制"
    return "无效"


def match_final(shown, expected_value):
    """页面显示是否与期望阶梯值一致（容许「n，歼灭」「1+压制，歼灭」等后缀）。"""
    if not shown:
        return True
    head = shown.split("，")[0].split(",")[0]
    head = head.split("+")[0]
    want = show_final(expected_value)
    if head in ("二次压制",):
        return want == "压制"
    return head == want


def parse_mod(text):
    t = (text or "").strip()
    if t in ("不修正", "", "-", "－"):
        return 0
    try:
        return int(t.replace("＋", "+"))
    except ValueError:
        return None


def parse_battle(text):
    t = (text or "").strip()
    if t in ("压制", "二次压制"):
        return 0
    if t in ("无效", ""):
        return -1
    try:
        return int(t)
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
    if raw.count(b"\x00") > len(raw) // 4:          # 无 BOM 的 UTF-16
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
    rows = []
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
                rows.append({
                    "file": os.path.relpath(path, CAP), "id": s.group(1),
                    "dist": s.group(2), "atk": int(s.group(3)), "roll1": int(s.group(4)),
                    "battle_raw": s.group(5), "armor": s.group(6),
                    "fix_raw": s.group(7), "final_raw": s.group(8),
                    "shooter_state": (s.group(9) or "").strip(),
                    "target_state": (s.group(10) or "").strip(),
                    "typeid": cur["typeid"], "ico": cur.get("ico", "")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", help="实测列号 json：{攻级: 列号}")
    args = ap.parse_args()
    cols = None
    if args.cols:
        cols = json.load(io.open(args.cols, encoding="utf-8-sig"))

    rows = collect()
    problems = defaultdict(list)
    dist = defaultdict(Counter)
    checked = Counter()
    for r in rows:
        smod_kwargs = {}
        for tok in r["shooter_state"].split():
            if tok in SHOOTER:
                smod_kwargs[SHOOTER[tok]] = True
        for tok in r["target_state"].split():
            if tok in TARGET:
                smod_kwargs[TARGET[tok]] = True
            if tok in TERRAIN:
                smod_kwargs["terrain"] = TERRAIN[tok]
        smod = crt.situational_mod(**smod_kwargs)
        roll2, observed_fix = parse_mod(r["fix_raw"]), parse_mod(r["fix_raw"])
        fix_roll = None
        observed_final, final_raw = parse_final(r["final_raw"])
        observed_battle = parse_battle(r["battle_raw"])
        dist["射手状态"][r["shooter_state"]] += 1
        dist["目标状态"][r["target_state"]] += 1
        dist["装甲"][r["armor"]] += 1
        dist["最终战果"][final_raw] += 1
        dist["攻级"][r["atk"]] += 1

        # 修正结果：直接与线上「修」列比（引擎用它算最终战果）
        exp_fix = observed_fix
        # 用 最终战果 − 修正结果 反推第二颗骰之前的原始值不可得（页面只给修正前随机数），
        # 故此处的 B 链路对账放在 verify_live_caijue.py（那里读得到「修正前」）。
        # 这里只做 A/C 两链路 + 修正口径自洽性检查。
        if r["typeid"] == 3:
            continue
        if r["typeid"] == 2:
            if cols and str(r["atk"]) in cols:
                c = int(cols[str(r["atk"])])
                row = crt.T["table4"].get(str(r["roll1"])) or []
                exp_battle = crt.ladder_of(row[c - 1]) if 0 <= c - 1 < len(row) else None
                tag = "A 战斗结果(实测列号)"
            else:
                exp_battle = crt.ladder_of(crt.result_vehicle(r["atk"], 1, r["roll1"]))
                tag = "A 战斗结果(引擎现映射)"
        else:
            exp_battle = crt.ladder_of(crt.result_person(r["atk"], r["roll1"]))
            tag = "A 战斗结果(表6)"
        if exp_battle is None:
            continue
        checked[tag] += 1
        if observed_battle is not None and observed_battle != exp_battle:
            problems[tag].append(
                "#%s %s 攻级%d 随机数1=%d 装甲%s → 观测%d vs 引擎%d"
                % (r["id"], r["dist"], r["atk"], r["roll1"], r["armor"],
                   observed_battle, exp_battle))
        if observed_final is not None:
            exp_final = None
            if exp_battle is not None and exp_fix is not None:
                # 与引擎同口径：空白格（页面「无效」）按「消灭 0」参与加法
                exp_final = crt.combine_result(cell_value(r["battle_raw"]), exp_fix)
            if exp_final is not None and not match_final(final_raw, exp_final):
                problems["C 最终战果 = 战斗结果+修正"].append(
                    "#%s 战斗结果%d + 修正%d = %d（应显示 %s）→ 观测%s"
                    % (r["id"], exp_battle, exp_fix, exp_final,
                       show_final(exp_final), final_raw))

    out = ["# 实测裁决 × 引擎（按 typeid 分类）", "",
           "样本 %d 条" % len(rows), ""]
    out.append("| 链路 | 条数 | 残差 |")
    out.append("|---|---|---|")
    for k in sorted(checked):
        out.append("| %s | %d | %d |" % (k, checked[k], len(problems[k])))
    out.append("")
    for key in ("攻级", "装甲", "目标状态", "射手状态", "最终战果"):
        out.append("- **%s**: %s" % (key, dict(sorted(dist[key].items(),
                                                   key=lambda kv: str(kv[0])))))
    out.append("")
    for k in sorted(problems):
        out.append("## %s（%d 条）" % (k, len(problems[k])))
        out += ["  " + p for p in problems[k][:40]]
        out.append("")
    io.open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("wrote", OUT)
    total_bad = sum(len(v) for v in problems.values())
    for k in sorted(checked):
        print("  %s: %d 条，残差 %d" % (k, checked[k], len(problems[k])))
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
