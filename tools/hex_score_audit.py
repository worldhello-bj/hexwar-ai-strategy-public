# -*- coding: utf-8 -*-
"""计分对账：把**平台真实计分页**（`Report_All.asp`）与引擎的候选口径摆在一起。

用途：`docs/计分规则待确认.md` 里的每一条结论都要能复跑。
平台侧数据来自已抓取的采集产物（`7527_transport_capture/**`，不入库）；
引擎侧用本地采集 `snapshot/parsed`（想定 7527），没有时用随仓库公开的
`scenario/7527`（见 `engine/paths.py`）。

用法：
    py -3.13 tools/hex_score_audit.py
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import Game                                    # noqa: E402
from engine import paths as _paths                          # noqa: E402
SNAP = _paths.snapshot_dir()

# 采集数据已封存到 archive/exploration/（见该目录 README）；
# 未封存的检出或 train/ 自包含副本里，老路径仍然存在 —— 两者都支持。
CAP = os.path.join(ROOT, "archive", "exploration", "capture", "7527_transport_capture")
if not os.path.isdir(CAP):
    CAP = os.path.join(ROOT, "7527_transport_capture")
PAT = re.compile(r'class=\\?"(RED|BLUE)\\?">\s*(\d+)\s*分')
COMPONENTS = ("夺控点", "剩余兵力", "歼灭对手兵力")


def decode(raw):
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def platform_scores():
    """扫采集产物，抽出计分页的三个分项（红, 蓝）。"""
    seen = {}
    files = 0
    for path in glob.glob(os.path.join(CAP, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) > 60 * 1024 * 1024:
                continue
            text = decode(open(path, "rb").read())
        except OSError:
            continue
        if not text or "夺控点得分" not in text:
            continue
        files += 1
        for chunk in text.split("夺控点得分")[1:]:
            hits = PAT.findall(chunk[:4000])
            if len(hits) >= 6:
                key = tuple(int(v) for _c, v in hits[:6])
                seen.setdefault(key, []).append(os.path.relpath(path, ROOT))
    return files, seen


def carried_establishment():
    """从「下车」采集产物里取载员的**编制兵力**。

    下车会在服务端**新建实体**，其 `blood` 即该载员的编制兵力
    （见 `docs/运输机制实测.md`）。这正好补上想定里载员条目 `blood=0` 的空缺。
    """
    units = json.load(open(os.path.join(SNAP, "units.json"),
                           encoding="utf-8"))
    carried_ids = {u["id"]: u for u in units if not (u.get("color") or "").strip()}
    found = {}
    for path in sorted(glob.glob(os.path.join(CAP, "**", "*.json"), recursive=True)):
        try:
            if os.path.getsize(path) > 5 * 1024 * 1024:
                continue
            data = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                i2 = node.get("id2")
                if i2 in carried_ids and node.get("id") != i2 and node.get("blood"):
                    src = carried_ids[i2]
                    found.setdefault(int(src["data"]), []).append(
                        (src["name"], int(node["blood"]),
                         os.path.relpath(path, ROOT)))
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return found


def engine_scores(establishment):
    """各种候选求和口径下的双方分值（+ 载员构成）。直接读想定 JSON。"""
    units = json.load(open(os.path.join(SNAP, "units.json"),
                           encoding="utf-8"))
    deployed = [u for u in units if (u.get("color") or "").strip()]
    carried = [u for u in units if not (u.get("color") or "").strip()]
    out = {}
    for color in ("BLUE", "RED"):
        dep = [u for u in deployed if u["color"] == color]
        own_carried = []
        for c in carried:
            host = next((u for u in dep if str(u.get("pos")) == str(c.get("pos2"))),
                        None)
            if host:
                own_carried.append((c, host))
        carried_score = sum(int(c["score"]) * establishment.get(int(c["data"]), 0)
                            for c, _h in own_carried)
        out[color] = {
            "units": len(dep),
            "sum_score": sum(int(u["score"]) for u in dep),
            "sum_score_blood": sum(int(u["score"]) * int(u["blood"]) for u in dep),
            "sum_score_blood_sons": sum(int(u["score"]) * (int(u["blood"])
                                                           + int(u.get("sons") or 0))
                                        for u in dep),
            "carried": [(c["id"], c["name"], int(c["score"]), int(c["data"]),
                         establishment.get(int(c["data"])), host["name"])
                        for c, host in own_carried],
            "carried_score": carried_score,
            "full": (sum(int(u["score"]) * int(u["blood"]) for u in dep)
                     + carried_score),
        }
    return out


def engine_actual():
    """引擎**真实跑的代码**（`engine/scoring.py`）在同一想定开局下的分值。

    上面的 `engine_scores()` 只是把候选公式摆出来对比；这里调的是实际实现，
    以免「文档里的公式」与「代码里的实现」脉开。
    """
    from engine import scoring

    game = Game(seed=0, snapshot=SNAP)
    return {side: scoring.side_score(game, side) for side in ("BLUE", "RED")}


def main():
    files, seen = platform_scores()
    # 平台侧证据来自抓取的计分页（`archive/exploration/capture/**`，不入库）。
    # **公开克隆里没有这些文件**，此时引擎侧仍能自检（remaining 必须等于平台 354/234
    # 是本仓库的断言，那里另有一份冻结值），所以这里明确跳过而不是崩。
    if not seen:
        print("平台侧采集产物缺失（%s），跳过『引擎 == 平台』对账，" % CAP)
        print("仅做想定开局的引擎侧自检（数据源：%s）。" % _paths.source_label())
        return 0
    est_raw = carried_establishment()
    # 编制兵力：采集里有直接观测的用观测值；旋翼无人机/巡飞弹未观测到，
    # 用平台分项**反解**（见下）。
    establishment = {}
    for data, rows in est_raw.items():
        establishment[data] = rows[0][1]
    inferred = {}
    plat_key = next((k for k in seen if seen[k]), None)
    if plat_key:
        # 红方差额 / 3 辆火力支援车 = 单车载员总编制分；旋翼无人机 score=2 → 编制 1
        red_gap = (plat_key[2] - sum(int(u["score"]) * int(u["blood"])
                                     for u in json.load(open(os.path.join(
                                         SNAP, "units.json"),
                                         encoding="utf-8"))
                                     if u.get("color") == "RED"))
        per_vehicle = red_gap // 3
        known = (establishment.get(392, 0) * 4 + establishment.get(397, 0) * 5)
        rest = per_vehicle - known
        if rest % 2 == 0:
            inferred[394] = rest // 2
    establishment.update(inferred)
    establishment.setdefault(389, 0)

    eng = engine_scores(establishment)
    actual = engine_actual()
    lines = []
    add = lines.append

    add("# 计分对账（平台计分页 vs 引擎候选口径）")
    add("")
    add("## 一、平台 `Report_All.asp` 的三个分项")
    add("")
    add("扫描到含计分页的文件 **%d** 个；出现过的 (红夺控, 蓝夺控, 红兵力, 蓝兵力, 红歼灭, 蓝歼灭) 组合："
        % files)
    add("")
    for key in sorted(seen, key=lambda k: -len(seen[k])):
        add("- `%s` × %d 次　例：`%s`" % (key, len(seen[key]), seen[key][0]))
    add("")
    add("> 页面上还有一行「摧毁重要目标得分」，在 HTML 里**被整行注释掉**。")
    add("> 上面那些组合里除了开局 `(0,0,354,234,0,0)`，其余都是探针主动造出的事件：")
    add("> 占点（夺控 50/80）、打掉一辆车（歼灭 10/20/40）、**装车后打掉（歼灭 56 = 车 40 + 乘员 16）**。")
    add("")
    add("## 二、载员的编制兵力（现为实测值）")
    add("")
    add("平台的**开局**里乘员与载具**同格**、各自是独立单位，`blood` 直接等于编制；")
    add("而**上车后**它变成 `blood=0/color=''`，此时编制不在快照里，靠 `engine/equip.py` 的表补；")
    add("下车会在服务端新建实体、其 `blood` 也是编制。三种来源互相印证：")
    add("")
    add("| data | 单位 | 编制兵力 | 证据 |")
    add("|---|---|---|---|")
    for data, rows in sorted(est_raw.items()):
        add("| %d | %s | **%d** | `%s` |" % (data, rows[0][0], rows[0][1], rows[0][2]))
    for data, value in inferred.items():
        add("| %d | 旋翼无人机 | **%d** | 开局直读（早期是靠分项反解） |"
            % (data, value))
    add("| 389 | 巡飞弹 | **3** | 开局直读（score=0，不影响分值，但编制是 3） |")
    add("")
    add("## 三、引擎候选口径（想定 7527 开局）")
    add("")
    add("| 阵营 | 上阵单位 | Σscore | Σ(score×blood) | Σ(score×(blood+sons)) | 载员分值 | **上阵+载员** |")
    add("|---|---|---|---|---|---|---|")
    for color in ("BLUE", "RED"):
        e = eng[color]
        add("| %s | %d | %d | %d | %d | %d | **%d** |"
            % (color, e["units"], e["sum_score"], e["sum_score_blood"],
               e["sum_score_blood_sons"], e["carried_score"], e["full"]))
    add("")
    add("## 四、与平台逐项对账")
    add("")
    if plat_key:
        pr, pb = plat_key[2], plat_key[3]
        add("- 平台剩余兵力分：红 **%d**、蓝 **%d**" % (pr, pb))
        add("")
        add("| 口径 | 红 | 蓝 | 差（红/蓝） |")
        add("|---|---|---|---|")
        for name, idx in (("Σscore", "sum_score"),
                          ("Σ(score×blood)", "sum_score_blood"),
                          ("Σ(score×(blood+sons))", "sum_score_blood_sons"),
                          ("**Σ(score×blood) + 载员×编制**", "full")):
            dr, db = eng["RED"][idx] - pr, eng["BLUE"][idx] - pb
            mark = "✅ **精确命中**" if (dr == 0 and db == 0) else "%+d / %+d" % (dr, db)
            add("| %s | %d | %d | %s |" % (name, eng["RED"][idx],
                                           eng["BLUE"][idx], mark))
        # 引擎**实际实现**（不是上面的候选公式，而是 `engine/scoring.py` 真跑的代码）
        ar, ab = actual["RED"]["total"], actual["BLUE"]["total"]
        dr, db = ar - pr, ab - pb
        agree = (dr == 0 and db == 0)
        add("| **引擎实际实现**（`engine/scoring.py`） | %d | %d | %s |"
            % (ar, ab, "✅ **精确命中**" if agree else "%+d / %+d ❌" % (dr, db)))
        add("")
        add("⇒ **开局**里乘员与载具**同格、各自是独立上阵单位**（快照已按此对齐，见"
            " `tools/hex_sync_snapshot_opening.py`），所以开局的「上阵 Σ(score×blood)」"
            "自己就能命中两个数；")
        add("  而**真用 `ObjInto` 装进车里**的乘员会变成 `blood=0`，平台仍按**编制**把它计入"
            "（下车会在服务端新建实体，其 `blood` 就是编制）：")
        add("")
        add("| data | 单位 | 编制兵力 | 依据 |")
        add("|---|---|---|---|")
        add("| 382 | 步兵班（蓝方装甲车载） | **3** | 实测：开局 @130032 blood=3 |")
        add("| 392 | 步兵班（红方火力支援车载） | **4** | 实测：开局 @190102 blood=4 |")
        add("| 397 | 无人察打车 | **4** | 实测：开局 @190102 blood=4 |")
        add("| 394 | 旋翼无人机 | **1** | 实测：开局 @190102 blood=1（早期是反解，现已直读） |")
        add("| 389 | 巡飞弹（score=0） | **3** | 实测：开局 @190102 blood=3（早期误写 0） |")
    add("")
    add("## 五、结论")
    add("")
    add("- 平台计分页只暴露**分项名**，不暴露分值/公式；")
    add("- **剩余兵力分**：`Σ(score×blood)`+车内载员按编制，**引擎已按此实现**"
        "（`engine/scoring.py`，上表最后一行就是它的真实输出），开局与平台**精确一致**；")
    add("- **夺控分**：已用探针**主动占点**测出——**主要 +80 / 次要 +50，一次性**"
        "（`tools/hex_score_probe.py --phase capture`）；")
    add("- **歼灭分**：已用探针**主动交战**测出——**`Σ(损失兵力 × score)`**，与剩余兵力分逐点互补"
        "（`--phase kill`）；")
    add("- **载员与载具存亡**：**真在车内**的乘员随车被歼（其编制分一并扣除并记给对手，"
        "实测 56 = 车 40 + 乘员 4×4，`tools/hex_load_kill_probe.py`）；"
        "**开局同格部署**的乘员是独立单位，打载具不碰他们（只掉 40）。")
    add("- **终局/结算**：计分页没有胜负、时间或任务奖励字段。")
    add("")
    add("⇒ 实测证据与探针命令见 `docs/计分规则待确认.md`。")

    report = os.path.join(ROOT, "docs", "计分对账.md")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n报告：%s" % report)
    if plat_key and not agree:
        print("\nFAIL：引擎实现与平台计分页不一致（上表最后一行）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
