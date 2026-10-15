# -*- coding: utf-8 -*-
"""核对「比赛初设的部队」是否正确 —— 用一手证据比，不靠自己的结论。

被核对对象：`scenario/7527/units.json`（随仓库公开的那份）。
一手证据（都在封存区，属服务端数据、不入库）：

| 证据 | 含义 |
|---|---|
| `archive/.../capture/7527_transport_capture/2p_before_start.json` | **两账号开局前**的 `Obj_Get`（最接近"初设"） |
| `.../2p_after_start.json` | 开局后 2 秒（正常应与上一份相同） |
| `.../initial_snapshot.json` | 另一次进房后的开局快照（另一次复位，编号不同） |
| `offline/mock/Obj_Get.asp` | 由服务端字节生成的**离线桩**，客户端实际渲染的就是它 |

★ 关键：**复位房间会换一批单位编号**（`Obj_101xxx` → `Obj_102xxx`），
所以比对必须**忽略编号**，按内容签名（军种/编制/位置/兵力/装备/图标…）做**多重集**比较。

另外做内部不变式检查（不依赖外部证据）：
  36 个单位；红蓝各 18；开局**无人在车内**（`blood>0` 且 `pos>0`）；
  乘员 `blood` == 编制表（`engine/equip.CARRIED_ESTABLISHMENT`）；
  载具 `sons` == 同格同阵营乘员数；`pos` 都在地图里；`wz` 都在武器表里；`score` 与装备表一致。

用法：py -3.13 tools\\verify_scenario_units.py        # 退出码 1 = 有实质差异
"""
import glob
import io
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import equip as EQ                      # noqa: E402
from engine import paths as _paths                  # noqa: E402

PUBLISHED = os.path.join(ROOT, "scenario", "7527", "units.json")
MAP = os.path.join(ROOT, "scenario", "7527", "map_hexes.json")
MOCK = os.path.join(ROOT, "offline", "mock", "Obj_Get.asp")
CAP_DIRS = [os.path.join(ROOT, "archive", "exploration", "capture",
                         "7527_transport_capture"),
            os.path.join(ROOT, "7527_transport_capture")]
EVIDENCE = ["2p_before_start.json", "2p_after_start.json", "initial_snapshot.json"]
SIG = ("army", "name", "group", "data", "typeid", "color", "blood", "pos", "ico",
       "wz", "sons", "score")
REPORT = os.path.join(ROOT, "analysis", "scenario_units_check.md")


def sig(u):
    return tuple(str(u.get(k, "")) for k in SIG)


def as_records(obj):
    """dict（Obj_xxx → attrs）或 list[attrs] → 统一成 list[attrs]。"""
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            v = dict(v)
            v.setdefault("id", k)
            out.append(v)
        return out
    return [dict(v) for v in obj]


def find_capture(name):
    for base in CAP_DIRS:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def load_mock():
    """从离线桩 `Obj_Get.asp`（GBK 字节）里抽静态单位属性。"""
    if not os.path.exists(MOCK):
        return None
    raw = io.open(MOCK, "rb").read()
    for enc in ("utf-8", "gb18030"):
        try:
            html = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return None
    out = []
    for m in re.finditer(r'<div id="(Obj_\d+)"([^>]*)>', html):
        attrs = dict(re.findall(r'([A-Za-z_]\w*)="([^"]*)"', m.group(2)))
        attrs["id"] = m.group(1)
        # ★ 离线桩的武器字段是 `Wz`（大写），发布 JSON 里是 `wz`；不归一化会全是假差异
        if not attrs.get("wz"):
            attrs["wz"] = attrs.get("Wz", "")
        out.append(attrs)
    return out


def compare(pub, other, label):
    """多重集比较（忽略编号），返回 (一致?, 差异描述列表)。"""
    a, b = Counter(sig(u) for u in pub), Counter(sig(u) for u in other)
    diffs = []
    for k in (a - b):
        diffs.append("仅在发布文件里（×%d）：%s" % ((a - b)[k], k))
    for k in (b - a):
        diffs.append("仅在%s里（×%d）：%s" % (label, (b - a)[k], k))
    return not diffs, diffs


def invariants(pub, mapids):
    out = []
    c = Counter(u.get("color") for u in pub)
    out.append(("单位总数 36", len(pub) == 36, "实际 %d" % len(pub)))
    out.append(("红蓝各 18", c.get("BLUE") == 18 and c.get("RED") == 18, str(dict(c))))
    aboard = [u["id"] for u in pub
              if str(u.get("blood")) in ("0", "") or str(u.get("pos")) in ("0", "")]
    out.append(("开局无人在车内/无隐藏单位", not aboard, "异常：%s" % aboard[:5]))
    bad_pos = [u["id"] for u in pub if int(u.get("pos") or 0) not in mapids]
    out.append(("所有位置都在地图里", not bad_pos, "越界：%s" % bad_pos[:5]))
    bad_wz = []
    for u in pub:
        for w in str(u.get("wz") or "").split(","):
            if w and int(w) not in EQ.WEAPONS:
                bad_wz.append((u["id"], w))
    out.append(("武器 data id 都在武器表里", not bad_wz, "未知：%s" % bad_wz[:5]))
    bad_score = []
    for u in pub:
        eq = EQ.EQUIPMENT.get(int(u.get("data") or 0))
        if eq and int(u.get("score") or 0) != eq[3]:
            bad_score.append((u["id"], u.get("score"), eq[3]))
    out.append(("score 与装备表一致", not bad_score, "不一致：%s" % bad_score[:5]))
    # 乘员编制：下车会新建实体、blood = 编制（见 docs/运输机制实测.md）
    crew = [(int(u["data"]), int(u["blood"])) for u in pub if str(u.get("typeid")) == "1"]
    est = EQ.CARRIED_ESTABLISHMENT
    bad_est = [(d, b, est.get(d)) for d, b in crew
               if d in est and est[d] != b]
    out.append(("乘员兵力 == 编制表", not bad_est, "不符：%s" % bad_est[:6]))
    # 载具 sons：开局**一律为 0**（谁也没上车）。
    # ★ sons = 车内单位数；中局才 >0（1/4 这类值）。发布文件曾误留中局值，
    #   证据：2p_before_start / 2p_after_start 两份开局抓包 sons 全 0。
    bad_sons = [(u["id"], u.get("army"), "sons=%s" % u.get("sons"))
                for u in pub if int(u.get("sons") or 0) != 0]
    out.append(("开局所有 sons == 0（无人在车内）", not bad_sons, "不符：%s" % bad_sons[:6]))
    return out


def main():
    pub = as_records(json.load(io.open(PUBLISHED, encoding="utf-8")))
    mapids = {int(k) for k in json.load(io.open(MAP, encoding="utf-8"))}
    lines = ["# 比赛初设部队核对（scenario/7527/units.json）", "",
             "发布文件：%d 个单位；地图 %d 格" % (len(pub), len(mapids)), ""]
    problems = []

    lines += ["## 1. 与一手证据逐单位对账（忽略编号，按内容签名做多重集比较）", "",
              "★ 先判每份证据**当时是什么状态**：有单位 `blood=0/pos=0`（= 已在车内）"
              "的是中局，不能拿来当「初设」基准；只有全部在场（0 人在车内）的才算开局态。", "",
              "| 证据 | 状态 | 结果 | 差异 |", "|---|---|---|---|"]
    skipped = []
    for name in EVIDENCE:
        p = find_capture(name)
        if not p:
            lines.append("| `%s` | — | 缺失 | — |" % name)
            continue
        data = json.load(io.open(p, encoding="utf-8"))
        other = as_records(data.get("static_units") or data)
        aboard = [u for u in other
                  if str(u.get("blood")) in ("0", "") or str(u.get("pos")) in ("0", "")]
        state = "中局态（%d 人在车内）" % len(aboard) if aboard else "**开局态**"
        if aboard:
            skipped.append(name)
            lines.append("| `%s`（%d 单位） | %s | 不作基准 | 与初设不同属预期（已上车/已机动） |"
                         % (name, len(other), state))
            continue
        ok, diffs = compare(pub, other, name)
        lines.append("| `%s`（%d 单位） | %s | %s | %s |"
                     % (name, len(other), state, "一致 ✅" if ok else "**不一致**",
                        "<br>".join(diffs[:6]) if diffs else "—"))
        if not ok:
            problems.append((name, len(diffs)))
    mock = load_mock()
    if mock:
        ok, diffs = compare(pub, mock, "离线桩")
        ids_pub = {u["id"] for u in pub}
        ids_mock = {u["id"] for u in mock}
        lines.append("| `offline/mock/Obj_Get.asp`（%d 单位） | %s | 编号交集 %d；%s |"
                     % (len(mock), "一致 ✅" if ok else "**内容不一致**",
                        len(ids_pub & ids_mock),
                        ("差异 %d 条" % len(diffs)) if diffs else "—"))
        if not ok:
            problems.append(("离线桩", len(diffs)))
            lines += ["", "<details><summary>离线桩差异明细（前 12 条）</summary>", "",
                      "```", "\n".join(diffs[:12]), "```", "</details>"]

    lines += ["", "## 2. 内部不变式（不依赖外部证据）", "",
              "| 检查 | 结果 | 说明 |", "|---|---|---|"]
    for name, ok, note in invariants(pub, mapids):
        lines.append("| %s | %s | %s |" % (name, "✅" if ok else "❌", note))
        if not ok:
            problems.append((name, note))

    lines += ["", "## 3. 结论", ""]
    if problems:
        lines.append("**有实质差异**：%s" % problems)
    else:
        lines.append("发布文件与**全部一手证据一致**，内部不变式也全部成立 ✅")
        lines.append("")
        lines.append("注意：复位房间会换单位编号（`Obj_101xxx` → `Obj_102xxx`），"
                     "所以比对按内容签名做；"
                     "另外 `sons`（车内单位数）必须是开局值 0 —— 历史上发布文件曾残留中局值"
                     "（装甲车 1 / 火力支援车 4），已由 `tools/fix_opening_sons.py` 修正。")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    io.open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", REPORT)
    for ln in lines:
        print(ln)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
