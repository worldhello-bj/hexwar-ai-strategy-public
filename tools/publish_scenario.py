# -*- coding: utf-8 -*-
"""把「实际比赛地图」整理成可公开发布的 `scenario/7527/`。

发布什么 / 不发布什么（**刻意白名单**，不是"排除法"）
-----------------------------------------------------
发布（想定 7527「城市攻防作战（2V2）」的**地图与编成**，全部是 `snapshot/parsed/` 里
可由公开练习服接口解析得到的事实数据）：
  * `map_hexes.json` — 2721 个六角格（高程 / 地形 cond / 通行 step / 道路 note）
  * `units.json`     — 36 个单位的编制与开局位置（对齐平台真实开局）
  * `cities.json`    — 3 个夺控点（主要/次要）
不发布（**同目录但属会话或第三方作品的衍生**）：
  * `session.pkl`、`*_logged_in.html`（本地会话/账号身份）
  * `raw/`（原始 HTTP 响应，含 `Set-Cookie` 等响应头）
  * `manual_tables*.json`、`manual_grid.json`（比赛手册的表格衍生物 —— 手册版权属原出版方）
  * `units.har.json`（HAR 采到的中局版本，已被 `units.json` 取代）

产物：
  scenario/7527/{map_hexes.json,units.json,cities.json,README.md,manifest.json}
其中 manifest 记 sha1 + 规模 + 来源端点 + **敏感扫描结论**（逐条断言无 Cookie/账号命中）。

用法：
  py -3.13 tools\\publish_scenario.py            # 生成/更新
  py -3.13 tools\\publish_scenario.py --check    # 只校验已发布内容是否与本地一致
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "snapshot", "parsed")
DST = os.path.join(ROOT, "scenario", "7527")
WHITELIST = ("map_hexes.json", "units.json", "cities.json")

# 发布前必须**零命中**的模式（命中就 abort，不生成）
FORBIDDEN = [
    ("Cookie 头", r"(?i)\b(Set-Cookie|Cookie)\s*:"),
    ("会话 ID", r"(?i)ASPSESSIONID[A-Z]*="),
    ("账号字段", r"(?i)(UserName|UserPassword|PassWord)\s*="),
    # 账号名用通用模式（不在公开文件里写真实账号名）
    ("账号名", r"UserName=[^\s&\"']*[\u4e00-\u9fa5]"),
    ("签名", r"(?i)\b(sign|appId)\s*[=:]\s*[\w\-]{6,}"),
]

README = """# 想定 7527 · 城市攻防作战（2V2）—— 地图与编成

本目录是**实际比赛地图**（比赛服与练习服为同一份想定 7527），
离线复现与规则强化训练都用它：`engine/`（裁决）与 `train/`（训练环境）直接读这里的 JSON。

## 内容

| 文件 | 规模 | 字段 |
|---|---|---|
| `map_hexes.json` | {hexes} 格 | `elevation` 高程(m) / `cond` 地形 / `step` 通行等级 / `note` 道路 / `hex` 贴图索引 |
| `units.json` | {units} 个单位 | `name`/`army` 编制、`typeid`（1 人员 2 车辆 3 空中 4 工事）、`data` 装备、`pos` 开局格、`blood` 兵力、`score` 权重分值 |
| `cities.json` | {cities} 个夺控点 | `name` 主要/次要、`pos` 所在格 |

行列数 {cols}×{rows}；红蓝各 {half} 个单位。`units.json` 已对齐**平台真实开局**
（`RoomStart.asp` 之后的 `Obj_Get.asp`），不是抓包时的中局状态。

## 数据来源与获取方式

来自官方**公开**练习服 `http://www.hexwar.cn/WARGAMEAI/`（同一想定在比赛服
`43.143.209.228/WARGAMEAI` 上逐字节一致）：

| 字段 | 接口 |
|---|---|
| 格高程/地形/通行/道路 | `Map_Get.asp`（客户端 `LoadCurrentCity()` 用的同一份数据） |
| 单位编制与位置 | `Obj_Get.asp` / `ObjAjax.asp` |
| 夺控点 | `City_Get.asp` |
| 列×行/格宽 | `User_Get.asp` 字段 6/7/32 |

采集与解析脚本见 `tools/`（`hex_transport_capture.py` 等）；解析产物
`snapshot/parsed/` 是本地完整采集目录（含原始响应），**不入库**。

## 权利与使用范围

* 地图与编成数据由原厂商（济南棋战 / hexwar.cn）生成，权利归其所有。
* 本目录**只包含互操作所需的事实数据**（格高程、单位位置等），
  **不含**官方前端代码/美术资源，**不含**比赛手册原文或其表格衍生物，
  **不含**任何会话 Cookie、账号或签名信息。
* 用途：离线复现、部队/院校教学、算法训练与研究。如权利人提出异议，删除本目录即可
  （`snapshot/` 里的本地采集不受影响）。
* 若你要在公开场合分发，请自行确认你与权利人之间的许可范围。

## 零依赖校验

```bash
py -3.13 tools/selftest_engine.py          # 读本目录跑引擎自检
HEXWAR_SNAP=scenario/7527 py -3.13 tools/hex_score_audit.py   # 计分对账
py -3.13 tools/publish_scenario.py --check # 本目录与本地采集是否一致
```

`manifest.json` 里有每个文件的 sha1、规模与**敏感扫描结论**（逐条断言无 Cookie/账号命中）。
"""


def sha1(path):
    h = hashlib.sha1()
    with io.open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def scan(text, rel):
    bad = []
    for label, pat in FORBIDDEN:
        m = re.search(pat, text)
        if m:
            bad.append((rel, label, m.group(0)[:40]))
    return bad


def build():
    if not os.path.isdir(SRC):
        print("找不到 %s（本地采集目录）" % SRC)
        return 1
    os.makedirs(DST, exist_ok=True)
    files = {}
    problems = []
    for name in WHITELIST:
        src = os.path.join(SRC, name)
        if not os.path.exists(src):
            print("缺少来源文件 %s" % name)
            return 1
        text = io.open(src, encoding="utf-8").read()
        problems += scan(text, name)
        dst = os.path.join(DST, name)
        io.open(dst, "w", encoding="utf-8").write(text)
        files[name] = {"sha1": sha1(src), "bytes": os.path.getsize(src)}
    if problems:
        print("!! 发布前扫描命中，已中止（不生成 scenario/）：")
        for p in problems:
            print("   %s  %s  %s" % p)
        return 1

    hexes = json.load(io.open(os.path.join(DST, "map_hexes.json"), encoding="utf-8"))
    units = json.load(io.open(os.path.join(DST, "units.json"), encoding="utf-8"))
    cities = json.load(io.open(os.path.join(DST, "cities.json"), encoding="utf-8"))
    us = b""
    for cand in ("User_Get.asp.txt", "User_Get.asp"):
        p = os.path.join(ROOT, "snapshot", "raw", cand)
        if os.path.exists(p):
            us = io.open(p, "rb").read()
            break
    cols = rows = "?"
    if us:
        for enc in ("gb18030", "utf-8"):
            try:
                f = us.decode(enc).split("$")
                cols, rows = f[6], f[7]
                break
            except (UnicodeDecodeError, IndexError):
                continue
    colors = {}
    for u in units:
        colors[u.get("color")] = colors.get(u.get("color"), 0) + 1
    io.open(os.path.join(DST, "README.md"), "w", encoding="utf-8").write(
        README.format(hexes=len(hexes), units=len(units), cities=len(cities),
                      cols=cols, rows=rows, half=max(colors.values()) if colors else "?"))
    man = {
        "scenario_id": 7527,
        "name": "城市攻防作战（2V2）",
        "published_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "practice": "http://www.hexwar.cn/WARGAMEAI/",
            "compete": "http://43.143.209.228/WARGAMEAI",
            "endpoints": ["User_Get.asp", "Map_Get.asp", "Obj_Get.asp", "ObjAjax.asp",
                          "City_Get.asp"],
            "note": "公开练习服抓取后解析；units.json 已对齐 RoomStart 后的真实开局",
        },
        "counts": {"hexes": len(hexes), "units": len(units), "cities": len(cities),
                   "grid": "%sx%s" % (cols, rows), "colors": colors},
        "files": dict(sorted(files.items())),
        "sensitive_scan": {
            "checked": [p[0] for p in FORBIDDEN],
            "hits": [],
            "verdict": "无 Cookie / 会话 ID / 账号 / 签名命中",
        },
        "excluded_on_purpose": [
            "session.pkl 与 *_logged_in.html（本地会话/账号身份）",
            "raw/（原始响应，含响应头）",
            "manual_tables*.json / manual_grid.json（比赛手册的表格衍生物）",
            "units.har.json（HAR 中局版本）",
        ],
    }
    io.open(os.path.join(DST, "manifest.json"), "w", encoding="utf-8").write(
        json.dumps(man, ensure_ascii=False, indent=1) + "\n")
    print("已发布 scenario/7527：%d 格 / %d 单位 / %d 夺控点（%s×%s）"
          % (len(hexes), len(units), len(cities), cols, rows))
    print("敏感扫描：无 Cookie/账号/签名命中 ✅")
    return 0


def check():
    man_p = os.path.join(DST, "manifest.json")
    if not os.path.exists(man_p):
        print("还没生成过：跑 py -3.13 tools\\publish_scenario.py")
        return 1
    man = json.load(io.open(man_p, encoding="utf-8"))
    bad = []
    for name, meta in man["files"].items():
        src = os.path.join(SRC, name)
        dst = os.path.join(DST, name)
        if not os.path.exists(dst):
            bad.append((name, "缺"))
        elif os.path.exists(src) and sha1(src) != sha1(dst):
            bad.append((name, "与本地采集不一致"))
        elif sha1(dst) != meta["sha1"]:
            bad.append((name, "与 manifest 不一致"))
    for name in WHITELIST:
        if os.path.exists(os.path.join(DST, name)):
            hits = scan(io.open(os.path.join(DST, name), encoding="utf-8").read(), name)
            bad += [(h[0], "敏感：" + h[1]) for h in hits]
    if bad:
        print("校验失败：%s" % bad)
        return 1
    print("scenario/7527 与本地采集一致，且无敏感命中 ✅")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    return check() if args.check else build()


if __name__ == "__main__":
    sys.exit(main())
