# -*- coding: utf-8 -*-
"""7527 计分口径（**唯一的定义处**）。

## 三项分值均已平台实测（2026-10-15，练习服 7527）

| 分项 | 口径 | 证据 |
|---|---|---|
| **剩余兵力分** | `Σ上阵(score × blood) + Σ车内载员(score × 编制)` | 对 7527 开局**精确命中平台红 354 / 蓝 234**（`tools/hex_score_audit.py` 的断言） |
| **夺控分** | **主要 +80 / 次要 +50，一次性**（占住 60 s 不增长） | 占下主要点→蓝 0→**80**；占下次要点→蓝 0→**50**（`7527_transport_capture/score_probe_capture_*`） |
| **歼灭分** | `Σ(损失兵力 × 单位 score)`，与剩余兵力分**逐点互补** | 坦克（score 10）掉 1 血 → 红剩余 −10 且蓝歼敌 +10；掉满 4 血 → 双方各 ±40（`score_probe_kill_20261015-220112`） |
| **夺控边界**（失守是否扣分 / 重夺是否重复计分） | **失守把该点的分从原主那里扣除，重夺方加上**（不衰减、不封顶） | 蓝方占下次要点 → 蓝 **50**；红方夺回后 → **红 50 / 蓝 0**（`score_flip_20261015-221433`） |
| **载员与载具存亡** | **真在车内**的乘员随车被歼（分随之扣除并记给对手）；**开局同格部署**的乘员是独立单位，不受影响 | ① 开局实况：打掉同格 ZBD100 → 只变化 **40**（354→314，乘员是独立单位）；② `ObjInto` 装车后打掉 → **−56 = 车 40 + 乘员 4×4**，且之后 `Obj2_Out` 返回 `0`（`score_probe_kill_20261015-222414` / `loadkill_20261015-223027`） |

⇒ 所以 **`PENDING_RULES` 现在是空的**：三项分值与夺控边界均已实测，
`verified` 与 `total` 相等。下面保留两套口径是为了将来一旦出现新的未实测项能**自动挡在奖励外**。

⚠️ **已废弃的旧值**：`CAPTURE_POINT_SCORE_PLACEHOLDER = 10` 是**无依据占位**，与实测差 5~8 倍；
`killed` 曾标为“假设”、夺控边界曾标为未确证，现均已被实测证实。

所以每个阵营同时给出两个总分，调用方必须自己选：

| 键 | 含义 | 用途 |
|---|---|---|
| `total` | 三个分项**全算**（平台式） | 只用于对账 / 展示 |
| `verified` | **只算已实测分项**（现与 `total` 相等） | **默认奖励口径**：未实测分值不进训练 |
| `unverified` | 本次实际涉及的未实测项清单（含“编制取自载入时 blood”这类前提） | 显式告警 |

## 载员归属：必须用稳定 ID，不能用格号现场匹配

快照里的载员记录只有 `pos2`（当时的载具格），而 `pos2` 是"目标格/旧格"：
**载具一机动就不再指向它**。若现场按格匹配，载具离开原格后载员分值会凭空消失，
等价于"惩罚机动"。所以归属由 `Game.link_carried()` 在载入时固化到
`carried.detail["carrier"]`（载具 ID），装卸动作写的是同一字段。

无法唯一确定归属的载员**不静默处理**：记入 `issues`
（`carried_unlinked` / `carried_ambiguous`），并进入 `unverified` 清单。
未知型号的载员编制（不在 `CARRIED_ESTABLISHMENT` 里）同样返回 **None 而不是 0**。
"""
from . import equip

# 口径版本。任何改变分值语义的改动都必须 +1：检查点/续跑靠它绑定，
# 以免旧模型与新公式混用（同一快照下两套奖励会污染训练）。
# v6：上车/下车语义也按实测对齐——上车 `blood=0`（编制留在 establishment，计分不变）、
#     下车按**编制**回满；于是车毁时乘员随车被歼的扣除与平台一致。
# v7: stable transport identity, verified filtering, and damage-conserving transport.
# v8: immobile fortifications cannot move or capture; invalidate older training contracts.
SCORING_VERSION = 8

# ---------------- 载员编制兵力 ----------------
# 表本身在**装备层**（`engine/equip.py`）：引擎自己（下车、发射巡飞弹）也要用它，
# 这里只是 re-export，保持 `scoring.carried_establishment` 这个稳定入口。
CARRIED_ESTABLISHMENT = equip.CARRIED_ESTABLISHMENT
carried_establishment = equip.carried_establishment

# ---------------- 夺控分（★ 已实测，2026-10-15）----------------
# 实测（练习服 7527，`tools/hex_score_probe.py --phase capture`）：
#    占下「主要」→ 夺控点得分 **+80**；占下「次要」→ **+50**；
#    且**一次性**：占住 60 秒不再增长（不是按时间累计）。
# 证据产物：`7527_transport_capture/score_probe_capture_20261015-215720`（蓝 0→80）、
#           `...-215820`（蓝 0→50）。
# ⚠️ 之前的 `CAPTURE_POINT_SCORE_PLACEHOLDER = 10` 是**无依据占位**，与实测差 5~8 倍。
CAPTURE_POINT_SCORE = {"主要": 80, "次要": 50}

# 已实测分项（只有这些进 `verified`）。
# `remaining` / `killed` / `captured` 的分值均已由平台实测对账，
# 且夺控边界（失守扣分 / 重夺加分）也已实测。
VERIFIED_COMPONENTS = ("remaining", "killed", "captured")
SIDES = ("BLUE", "RED")


def capture_point_value(name):
    """夺控点分值；**未测到的名称返回 None**（调用方必须显式报出，不得猜）。"""
    return CAPTURE_POINT_SCORE.get(str(name or "").strip())


# 仍未确证的**规则**。**现在是空的**：夺控的最后一条边界（失守是否扣分 /
# 重夺是否重复计分）已于 2026-10-15 用 `tools/hex_flip_probe.py` 实测：
#     蓝方占下 `次要` → 蓝 50；蓝方撤到 2 格外（flag 仍为蓝）；红方走上格并 CityTake
#     → 页面变成「红 50 / 蓝 0」⇒ **失守把分从原主扣除、重夺方加上**（不衰减、不封顶）。
# 同一个实验还交叉验证了手册 3.13：蓝方单位留在**邻格**时红方夺控全被拒（返回空串），
# 撤到 2 格外后才成功 ⇒ 平台确实要求“本格与相邻 6 格无敌方地面单位”。
# 保留这个常量是为了以后一旦出现新的未实测规则，`score()` 能自动把它报出来并挡在奖励外。
PENDING_RULES = ()


def breakdown(game):
    """把棋子拆成「上阵单位」与「车内载员+其载具」，并回报无法归类的问题。

    返回 `(parts, issues)`：

    - `parts[side] = {"deployed": [...], "carried": [(载员, 载具), ...]}`
    - `issues = {"carried_unlinked": [id...], "carried_ambiguous": [(id, [载具id...])],
      "carried_establishment_unknown": [(id, data)...]}`

    归属优先用 `detail["carrier"]`（载入时由 `Game.link_carried` 固化，
    装卸动作写同一字段，**载具移动后依然有效**）；只有缺少该字段的老数据才回退到
    "该格唯一载具"，并且回退事实仍会被记进 `issues`。
    """
    parts = {s: {"deployed": [], "carried": []} for s in SIDES}
    issues = {"carried_unlinked": [], "carried_ambiguous": [],
              "carried_establishment_unknown": []}
    for u in game.units.values():
        if u.color in SIDES:
            parts[u.color]["deployed"].append(u)
    for u in sorted(game.units.values(), key=lambda x: str(x.id)):
        if u.color or not u.pos2 and not u.detail.get("carrier"):
            continue
        host = None
        carrier_id = u.detail.get("carrier")
        if carrier_id:
            candidate = game.units.get(carrier_id)
            if candidate is not None and candidate.color in SIDES:
                host = candidate
            else:
                issues["carried_unlinked"].append(u.id)      # 关联指向已消失/无阵营的载具
                continue  # An explicit stable ID must never be rebound by stale position.
        if host is None:
            # 回退：老数据没有稳定关联，只在该格能唯一确定归属时使用。
            group = [m for m in game.units.values()
                     if m.color in SIDES and m.pos and m.pos == u.pos2]
            vehicles = [m for m in group if m.typeid in (2, 3)]
            pool = vehicles or group
            if len(pool) != 1:
                if pool:
                    issues["carried_ambiguous"].append((u.id, sorted(str(m.id) for m in pool)))
                elif u.id not in issues["carried_unlinked"]:
                    issues["carried_unlinked"].append(u.id)
                continue
            host = sorted(pool, key=lambda x: str(x.id))[0]
        if carried_establishment(u.data) is None:
            issues["carried_establishment_unknown"].append((u.id, u.data))
        parts[host.color]["carried"].append((u, host))
    return parts, issues


def _from_parts(parts, issues, side, cities=()):
    mine = parts[side]
    other = parts["RED" if side == "BLUE" else "BLUE"]

    # ---- 剩余兵力分（★ 已反推并精确验证）----
    # ★★ 两种情形都已实测（2026-10-15）：
    #  ① **开局**（实时 RoomStart 的 Obj_Get，以及 `2p_before_start.json`）：乘员与载具**同格**、
    #     各自是**独立上阵单位**（36 个单位全部在场，没有一条 pos=0 记录）⇒ 打掉载具
    #     只掉车体那 40 分（`score_probe_kill_20261015-222414`：354→314）。
    #  ② **真用 ObjInto 装进车里**的乘员：车被歼时它**连带被歼**，其 `score × 编制` 从剩余兵力分里
    #     扣除、并记给对手（`loadkill_20261015-223027`：354 → **298** = 车 40 + 乘员 4×4=16，
    #     且之后 `Obj2_Out` 返回 `0` —— 放不出来了）。
    #     ⇒ 载具存活才计入车内兵力；v7 中途装车保留实际血量，快照缺失时取编制。
    remaining = sum(u.score * u.blood for u in mine["deployed"])
    for carried, host in mine["carried"]:
        strength = carried.detail.get("aboard_blood", carried_establishment(carried.data))
        if strength is None or not host.alive():
            continue                       # 未知型号 / 载具已毁（乘员随之被歼）：不计入
        remaining += carried.score * strength

    # ---- 歼灭分（★ 已实测：Σ(损失兵力 × score)，与剩余兵力分逐点互补）----
    # 车里的人随车一起算：载具被歼 ⇒ 车体的 `score × 编制` **加上**其载员的 `score × 编制`
    # （实测 56 = 40 + 16）。
    killed = sum(u.score * max(0, u.establishment - u.blood)
                 for u in other["deployed"])
    for carried, host in other["carried"]:
        establishment = (carried.establishment if "aboard_blood" in carried.detail
                         else carried_establishment(carried.data))
        strength = carried.detail.get("aboard_blood", establishment)
        if establishment is not None and strength is not None:
            lost = establishment - strength if host.alive() else establishment
            killed += carried.score * max(0, lost)

    # ---- 夺控分（★ 已实测：主要 +80 / 次要 +50，一次性）----
    captured = 0
    for city in cities:
        if city.get("flag") != side:
            continue
        value = capture_point_value(city.get("name"))
        if value is None:
            continue                       # 名称没测过：不计入，并在下面报出
        captured += value

    # ---- 未实测项（本次计算里实际涉及的）----
    # 三个分项的分值与夺控边界**都已实测**（见模块文档头），这里只剩两类“前提”告警：
    #   establishment_from_load_blood：编制取自载入时 blood（只对开局快照成立）
    #   各类 carried_* / capture_point_name_unknown：数据本身没说清楚
    unverified = ["establishment_from_load_blood"]
    if any(u.detail.get("establishment_unverified") for u in mine["deployed"]):
        unverified.append("establishment_unverified")
    own_ids = {u.id for u in mine["deployed"]} | {c.id for c, _h in mine["carried"]}
    if any(cid in own_ids for cid, _d in issues["carried_establishment_unknown"]):
        unverified.append("carried_establishment_unknown")
    if any(cid in own_ids for cid in issues["carried_unlinked"]):
        unverified.append("carried_unlinked")
    if any(cid in own_ids for cid, _h in issues["carried_ambiguous"]):
        unverified.append("carried_ambiguous")
    if any(u.detail.get("aboard_damage_forgotten") for u in mine["deployed"] + [c for c, _h in mine["carried"]]):
        # 带伤上车：平台模型是“车内不记伤、下车按编制回满”，但这一条**没有直接实测**。
        unverified.append("aboard_damage_forgotten")
    if any(u.detail.get("damaged_transport_unverified")
           for u in mine["deployed"] + [c for c, _h in mine["carried"]]):
        unverified.append("damaged_transport_unverified")
    if any(u.detail.get("carrier") for u in mine["deployed"]):
        # 理论上不该发生：上车后 blood 会被置 0，于是它应当落在 `carried` 而不是 `deployed`。
        # 真出现了（外部构造的状态/旧测试）就报出来，避免惄惄地按“上阵单位”算。
        unverified.append("aboard_counted_as_deployed")
    for city in cities:
        if city.get("flag") == side and capture_point_value(city.get("name")) is None:
            unverified.append("capture_point_name_unknown")
    components = {"remaining": remaining, "captured": captured, "killed": killed}
    return {**components,
            "total": sum(components.values()),
            "verified": sum(components[name] for name in VERIFIED_COMPONENTS),
            "unverified": sorted(set(unverified))}


def _global_unverified(issues):
    """无法归属到某一方的未确证项（例如"载员根本没关联上"）—— 两边都要报出。"""
    out = []
    if issues["carried_unlinked"]:
        out.append("carried_unlinked")
    if issues["carried_ambiguous"]:
        out.append("carried_ambiguous")
    if issues["carried_establishment_unknown"]:
        out.append("carried_establishment_unknown")
    return out


def side_score(game, side):
    """一方三个分项 + `total`/`verified`/`unverified`。"""
    scores, _issues = both_scores(game)
    return scores[side]


def both_scores(game):
    """一次算出两方（避免重复遍历）。返回 `({"BLUE":..,"RED":..}, issues)`。"""
    parts, issues = breakdown(game)
    cities = getattr(game, "cities", ())
    global_reasons = _global_unverified(issues)
    scores = {}
    for side in SIDES:
        party = _from_parts(parts, issues, side, cities)
        party["unverified"] = sorted(set(party["unverified"]) | set(global_reasons))
        scores[side] = party
    return scores, issues


def reward_value(party, mode="verified"):
    """按奖励口径取总分：`verified`（默认，只用已实测分项）或 `platform`（平台式三项和）。"""
    if mode == "verified":
        return party["verified"]
    if mode == "platform":
        return party["total"]
    raise ValueError("scoring mode must be 'verified' or 'platform'")
