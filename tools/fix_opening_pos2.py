# -*- coding: utf-8 -*-
"""把「比赛初设」里残留的 `pos2=0` 改回实测真值（初设下 `pos2 == pos`）。

## 背景（实测结论，2026-10-16 练习服复核）
`pos2` = **目标格/上一格**。平台加载想定时给每个单位写 `pos2 = pos`（原地待命），
所以**初设下 36 个单位的 `pos2` 全等于自己的 `pos`**：

- 练习服推演室2（状态 0 空闲）进房后、`StartTime` 为空时的 `Obj_Get.asp`：
  `车内(pos=0 或 blood=0) = 0`、`载具 sons>0 = 0`、**36 单位 pos2 全 == pos**；
- 同一房间 `RoomStart.asp` 开局后（第 0 秒）再读：仍 36 单位全在图上、`pos2 == pos`；
- 中局对照（推演室7，开局 11 小时后）：38 单位也全部 `pos2 == pos`。

本仓库的 `scenario/7527/units.json` 来自**中局**抓包，归档的
`hex_sync_snapshot_opening.py` 只在 `pos2 in ("0","",None) or pos2 == pos` 时才改写，
但**发布的那份没吃到这次同步**，于是 6 辆载具（3×M113TW / 3×ZBD100 里的 6 条）残留了 `pos2=0`。

引擎只在**载员记录**（`pos=0`、`blood=0`、无阵营）上用 `pos2` 找载具（`engine/game.py`
`link_declared_carriers`、`engine/scoring.py`）；初设里没有载员记录，
所以本修正**不改任何裁决结果**，只让公开的初设数据与平台一致（修完实时比对 0 差异）。

## 判定来源
1. `--evidence <json>`（或自动扫描 `analysis/opening_check*.json`）：读实测开局态，
   按**内容签名**取 `pos2`；
2. 无证据时退回规则：**初设下 `pos2 = pos`**。

## 目标
- `scenario/7527/units.json`（发布的初设）
- `offline/mock/Obj_Get.asp`（离线桩，GB18030 写回）
- `snapshot/parsed/units.json`（若本地存在）

用法：
  py -3.13 tools\\fix_opening_pos2.py            # 只看会改什么
  py -3.13 tools\\fix_opening_pos2.py --apply    # 真改
"""
import argparse
import glob
import io
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TARGET_JSON = [os.path.join(ROOT, "scenario", "7527", "units.json"),
               os.path.join(ROOT, "snapshot", "parsed", "units.json")]
MOCK = os.path.join(ROOT, "offline", "mock", "Obj_Get.asp")
EVIDENCE_GLOB = os.path.join(ROOT, "analysis", "opening_check*.json")
# 签名用哪些字段（**不含 pos2**，因为就是要修它）
SIG_KEYS = ("army", "name", "group", "data", "typeid", "color", "blood", "pos",
            "ico", "score", "wz")


def norm(attrs):
    """把 JSON 记录或 HTML 属性统一成小写 key 的 dict（mock 里武器字段是 `Wz`）。"""
    out = dict(attrs)
    if "wz" not in out and "Wz" in out:
        out["wz"] = out["Wz"]
    return out


def sig(d):
    return tuple(str(d.get(k, "")) for k in SIG_KEYS)


def load_json_units(path):
    if not os.path.exists(path):
        return None, None
    arr = json.load(io.open(path, encoding="utf-8"))
    if isinstance(arr, dict):
        return arr, [arr[k] for k in arr]
    return arr, arr


def load_mock(path):
    html = io.open(path, "rb").read().decode("gb18030")
    divs = list(re.finditer(r'<div id="(Obj_\d+)"([^>]*)>', html))
    recs = [dict(re.findall(r'([A-Za-z_]\w*)="([^"]*)"', m.group(2)), id=m.group(1))
            for m in divs]
    return html, divs, recs


def evidence_pos2(explicit=None):
    """从实测开局抓取（本工具同目录的 `hex_opening_check.py` 产物）得到 {签名: pos2}。"""
    paths = [explicit] if explicit else sorted(glob.glob(EVIDENCE_GLOB))
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        data = json.load(io.open(p, encoding="utf-8"))
        for tag in ("before", "after"):
            snap = data.get(tag) or {}
            rows = snap.get("static_units") or {}
            rows = list(rows.values()) if isinstance(rows, dict) else list(rows)
            aboard = [u for u in rows
                      if str(u.get("blood")) in ("0", "")
                      or str(u.get("pos")) in ("0", "")]
            if not rows or aboard:
                continue                      # 不是「全在图上」的初设态，换下一份
            table = defaultdict(set)
            for u in rows:
                table[sig(norm(u))].add(str(u.get("pos2")))
            bad = {k: v for k, v in table.items() if len(v) > 1}
            if bad:
                raise SystemExit("证据自相矛盾：同一签名出现多个 pos2 %s" % list(bad)[:3])
            return ({k: sorted(v)[0] for k, v in table.items()},
                    "%s[%s]（%d 单位全在图上）" % (os.path.basename(p), tag, len(rows)))
    return None, "找不到实测开局抓取（analysis/opening_check*.json）"


def rule_pos2():
    return "初设下平台给每个单位写 pos2 = pos ⇒ pos2 取自己的 pos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真写文件（默认只预览）")
    ap.add_argument("--evidence", default=None, help="实测开局抓取 JSON（默认自动扫描）")
    ap.add_argument("--no-evidence", action="store_true", help="强制用规则，不读实测抓取")
    args = ap.parse_args()

    table, src = (None, None) if args.no_evidence else evidence_pos2(args.evidence)
    if table:
        print("pos2 取值来源：实测 %s" % src)
    else:
        print("pos2 取值来源：规则 —— %s（%s）" % (rule_pos2(), src))
    print("模式：%s\n" % ("apply（写文件）" if args.apply else "dry-run（只预览）"))

    total, unknown = 0, []

    def want_for(rec):
        """返回该记录应有的 pos2；载员记录（pos=0）不动，返回 None 表示跳过。"""
        if str(rec.get("pos") or "0") in ("0", ""):
            return None                       # 车内记录：pos2 是载具格，必须保留
        if table:
            return table.get(sig(norm(rec)))  # None ⇒ 证据里没有 ⇒ 记入 unknown
        return str(rec.get("pos"))

    # ---- 1) JSON 初设 ----
    for path in TARGET_JSON:
        obj, rows = load_json_units(path)
        if rows is None:
            print("-- 跳过（不存在）：%s" % os.path.relpath(path, ROOT))
            continue
        changes = []
        for u in rows:
            want = want_for(u)
            if want is None and str(u.get("pos") or "0") != "0":
                unknown.append((os.path.relpath(path, ROOT), u.get("army"), sig(norm(u))))
            if want is None:
                continue
            if str(u.get("pos2", "")) != want:
                changes.append((u.get("army"), u.get("ico"), u.get("pos2"), want))
                u["pos2"] = want
        print("-- %s：%d 处 %s" % (os.path.relpath(path, ROOT), len(changes),
                                   "已改" if args.apply else "待改"))
        for n, ico, a, b in changes[:12]:
            print("     %-18s %-7s pos2 %s -> %s" % (n, ico, a, b))
        if args.apply and changes:
            io.open(path, "w", encoding="utf-8", newline="\n").write(
                json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
        total += len(changes)

    # ---- 2) 离线桩（GB18030） ----
    if os.path.exists(MOCK):
        html, divs, recs = load_mock(MOCK)
        out, pos, changes = [], 0, []
        for m, rec in zip(divs, recs):
            want = want_for(rec)
            if want is None and str(rec.get("pos") or "0") != "0":
                unknown.append(("offline/mock/Obj_Get.asp", rec.get("army"), sig(norm(rec))))
            out.append(html[pos:m.start()])   # ★ 必须到 m.start()：前缀不含本 div 头
            pos = m.end()
            head = m.group(0)
            if want is not None and str(rec.get("pos2", "")) != want:
                changes.append((rec.get("army"), rec.get("ico"), rec.get("pos2"), want))
                head = re.sub(r'pos2="[^"]*"', 'pos2="%s"' % want, head, count=1)
            out.append(head)
        out.append(html[pos:])
        print("-- offline/mock/Obj_Get.asp：%d 处 %s" % (len(changes),
                                                        "已改" if args.apply else "待改"))
        for n, ico, a, b in changes[:12]:
            print("     %-18s %-7s pos2 %s -> %s" % (n, ico, a, b))
        if args.apply and changes:
            io.open(MOCK, "wb").write("".join(out).encode("gb18030"))
        total += len(changes)
    else:
        print("-- 跳过（不存在）：offline/mock/Obj_Get.asp")

    if unknown:
        print("\n!! 有 %d 个单位在证据里找不到同签名项（需人工确认）：" % len(unknown))
        for x in unknown[:5]:
            print("   ", x)
        return 2
    print("\n合计 %d 处" % total)
    if not args.apply and total:
        print("加 --apply 生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
