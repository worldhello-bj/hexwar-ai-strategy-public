# -*- coding: utf-8 -*-
"""把「比赛初设」里被写错的 `sons` 改回开局真值。

## 背景（实测结论）
`sons` = **车内单位数**（0 = 谁也没上车）。
- 开局（`RoomStart` 后）：所有单位 `sons=0`，乘员以自己的实体停在同一格
  —— 见 `2p_before_start.json` / `2p_after_start.json`（两份都是 36 单位 / 0 人在车内 / sons 全 0）。
- 中局（有人点过"全员上车"）：载具 `sons` = 车内人数
  —— 见 `initial_snapshot.json`（15 人在车内，装甲车 sons=1、火力支援车 sons=4）。

本仓库最初的数据来自**中局**抓包，而归档的 `hex_sync_*_opening.py` 只同步了
`pos/pos2/blood/color/step`，**漏了 `sons`** ⇒ 发布文件与离线桩里残留了中局值。
引擎并不读 `sons`（`grep` 无引用），所以不影响裁决；但作为**要公开的初设数据**必须准。

## 判定来源
1. `--from-evidence`（默认）：读归档的开局抓包，按**内容签名**（忽略 `Obj_xxx` 编号，
   因为复位会换号）取每类的 `sons`；
2. 抓包不在（公开仓库就是这种情况）时退回规则：**开局 36 单位 `sons` 一律为 0**。

## 目标
- `scenario/7527/units.json`（发布的初设）
- `offline/mock/Obj_Get.asp`（离线桩，GB18030 写回）
- `snapshot/parsed/units.json`（若本地存在）

用法：
  py -3.13 tools\\fix_opening_sons.py            # 只看会改什么
  py -3.13 tools\\fix_opening_sons.py --apply    # 真改
"""
import argparse
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CAP_DIRS = [os.path.join(ROOT, "archive", "exploration", "capture",
                         "7527_transport_capture"),
            os.path.join(ROOT, "7527_transport_capture")]
OPENING_EVIDENCE = ["2p_before_start.json", "2p_after_start.json"]
TARGET_JSON = [os.path.join(ROOT, "scenario", "7527", "units.json"),
               os.path.join(ROOT, "snapshot", "parsed", "units.json")]
MOCK = os.path.join(ROOT, "offline", "mock", "Obj_Get.asp")
# 签名用哪些字段（**不含 sons**，因为就是要修它）
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
    raw = io.open(path, "rb").read()
    html = raw.decode("gb18030")
    divs = list(re.finditer(r'<div id="(Obj_\d+)"([^>]*)>', html))
    recs = []
    for m in divs:
        attrs = dict(re.findall(r'([A-Za-z_]\w*)="([^"]*)"', m.group(2)))
        attrs["id"] = m.group(1)
        recs.append(attrs)
    return html, divs, recs


def evidence_sons():
    """从开局抓包得到 {签名: sons}；不可用时返回 (None, 原因)。"""
    for name in OPENING_EVIDENCE:
        for base in CAP_DIRS:
            p = os.path.join(base, name)
            if not os.path.exists(p):
                continue
            data = json.load(io.open(p, encoding="utf-8"))
            rows = data.get("static_units") or data
            rows = list(rows.values()) if isinstance(rows, dict) else list(rows)
            aboard = [u for u in rows
                      if str(u.get("blood")) in ("0", "") or str(u.get("pos")) in ("0", "")]
            if aboard:
                continue                      # 不是开局态，换下一份
            m = defaultdict(set)
            for u in rows:
                m[sig(norm(u))].add(str(u.get("sons")))
            bad = {k: v for k, v in m.items() if len(v) > 1}
            if bad:
                raise SystemExit("证据自相矛盾：同一签名出现多个 sons %s" % list(bad)[:3])
            return {k: sorted(v)[0] for k, v in m.items()}, "%s（%d 单位、0 人在车内）" % (name, len(rows))
    return None, "归档里没有开局抓包"


def rule_sons():
    return "开局无人在车内 ⇒ sons 一律 0（引擎不读该字段，改的只是数据保真度）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真写文件（默认只预览）")
    ap.add_argument("--no-evidence", action="store_true", help="强制用规则，不读抓包")
    args = ap.parse_args()

    table, src = (None, None) if args.no_evidence else evidence_sons()
    if table:
        print("sons 取值来源：证据 %s" % src)
    else:
        print("sons 取值来源：规则 —— %s（%s）" % (rule_sons(), src))
    print("模式：%s\n" % ("apply（写文件）" if args.apply else "dry-run（只预览）"))

    total_changes = 0
    unknown = []

    # ---- 1) JSON 初设 ----
    for path in TARGET_JSON:
        obj, rows = load_json_units(path)
        if rows is None:
            print("-- 跳过（不存在）：%s" % os.path.relpath(path, ROOT))
            continue
        changes = []
        for u in rows:
            want = table.get(sig(norm(u))) if table else "0"
            if want is None:
                unknown.append((os.path.relpath(path, ROOT), u.get("army"), sig(norm(u))))
                continue
            if str(u.get("sons", "")) != want:
                changes.append((u.get("army"), u.get("sons"), want))
                u["sons"] = want
        print("-- %s：%d 处 %s" % (os.path.relpath(path, ROOT), len(changes),
                                   "已改" if args.apply else "待改"))
        for n, a, b in changes[:12]:
            print("     %-18s sons %s -> %s" % (n, a, b))
        if args.apply and changes:
            io.open(path, "w", encoding="utf-8", newline="\n").write(
                json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
        total_changes += len(changes)

    # ---- 2) 离线桩（GB18030） ----
    if os.path.exists(MOCK):
        html, divs, recs = load_mock(MOCK)
        out, pos, changes = [], 0, []
        for m, rec in zip(divs, recs):
            want = table.get(sig(norm(rec))) if table else "0"
            out.append(html[pos:m.start()])   # ★ 必须到 m.start()：前缀不含本 div 头
            pos = m.end()
            head = m.group(0)
            if want is None:
                unknown.append(("offline/mock/Obj_Get.asp", rec.get("army"), sig(norm(rec))))
                out.append(head)
                continue
            if str(rec.get("sons", "")) != want:
                changes.append((rec.get("army"), rec.get("sons"), want))
                head = re.sub(r'sons="[^"]*"', 'sons="%s"' % want, head, count=1)
            out.append(head)
        out.append(html[pos:])
        print("-- offline/mock/Obj_Get.asp：%d 处 %s" % (len(changes),
                                                        "已改" if args.apply else "待改"))
        for n, a, b in changes[:12]:
            print("     %-18s sons %s -> %s" % (n, a, b))
        if args.apply and changes:
            io.open(MOCK, "wb").write("".join(out).encode("gb18030"))
        total_changes += len(changes)
    else:
        print("-- 跳过（不存在）：offline/mock/Obj_Get.asp")

    if unknown:
        print("\n!! 有 %d 个单位在证据里找不到同签名项（需人工确认）：" % len(unknown))
        for x in unknown[:5]:
            print("   ", x)
        return 2
    print("\n合计 %d 处" % total_changes)
    if not args.apply and total_changes:
        print("加 --apply 生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
