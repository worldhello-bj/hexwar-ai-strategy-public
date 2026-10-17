# -*- coding: utf-8 -*-
"""Hexwar 兵推策略搜索环境。

面向对象：AI 智能体 / 强化学习 / LLM 规划器 / 手工脚本。

设计原则
--------
1. **观察是纯 JSON**：可序列化、字段自解释、坐标用「行列号 + mapid」双写，
   便于 LLM 直接读、也便于程序处理。
2. **动作是纯 JSON**：`{"type": "...", ...}`，可被 LLM 直接生成、可被校验。
3. **一次 step = 一个决策 + 时间前推**：把机动耗时算进时钟，
   避免实时调度，天然适合回合制的策略搜索。
4. **迷雾可选**：`fog=True` 时只能看到己方与「当前可观察到的」敌方，
   这才是真正的兵推——不然就是上帝视角。

用法
----
    from env import HexwarEnv
    env = HexwarEnv(side="BLUE")
    obs = env.reset(seed=1)
    while not env.done:
        acts = env.legal_actions()
        obs, r, done, info = env.step(acts[0])
    print(env.score())
"""
import os
import sys
import json
import copy
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Game, geometry as G
from engine.equip import WEAPONS, EQUIPMENT, AMMO_FIELD
from engine.game import DEPLOY_WINDOWS
from engine import crt
from engine import scoring

# ---------------- 时间与代价常量（秒） ----------------
ATTACK_SEC = 10        # 一次直瞄射击占用时间
CAPTURE_SEC = 30       # 夺控判定耗时
HIDE_SEC = 10          # 隐蔽/解除隐蔽
WAIT_SEC = 10          # 主动等待的默认时长
LOAD_SEC = 30          # 上/下车
GRENADE_SEC = 10       # 投掷手雷
AREA_FIRE_SEC = 10     # 一次集火射击（手册 3.16）
# 发射一枚巡飞弹的占用耗时（手册 4.10）。**平台未实测、手册也没给** —— 明确标成
# 占位值，不冒充实测值；规则效果（消耗 / 在飞占用 / 释放 / 计时）都不依赖这个数。
LAUNCH_SEC = 10
# 手册未给出单车载员数，此处按堆叠上限（手册 4.2）假定为 4
VEHICLE_SEATS = 4


def own_units_for_projection(game, side):
    """投影枚举时要保留的「本方单位」：己方在编 **加上己方车内单位**。

    车内记录（`color` 为空、`pos=0`、`blood=0`）不是独立行动单位，但
    `load` / `launch` 这类动作的**主体或宾语**可能就是它（弹在载车的载员表里），
    所以投影必须带上它们：否则 `legal_actions` 里 `g.units.get(missile_id)` 会落空，
    「发射巡飞弹」这个动作永远生成不出来。

    平台侧同样不下发车内记录（`ObjAjax` 里恒为 0 条），玩家靠载具的 `carrying`
    才看得到它们的 id —— 与这里的口径一致。
    """
    out = {}
    for uid, unit in game.units.items():
        if unit.color == side:
            out[uid] = unit
            continue
        host = game.units.get(unit.detail.get("carrier"))
        if host is not None and host.color == side:
            out[uid] = unit
    return out


class HexwarEnv:
    """单方视角的兵推环境。"""

    def __init__(self, snapshot=None, side="BLUE", max_seconds=1800,
                 fog=True, dt=WAIT_SEC, victory="score", move_radius=6,
                 deep_validate=False, scoring_mode="verified",
                 deploy_window="opening"):
        if side not in ("BLUE", "RED"):
            raise ValueError("side must be BLUE or RED")
        if scoring_mode not in ("verified", "platform"):
            raise ValueError("scoring_mode must be 'verified' or 'platform'")
        if deploy_window not in DEPLOY_WINDOWS:
            raise ValueError("deploy_window must be one of %s" % (DEPLOY_WINDOWS,))
        if not isinstance(max_seconds, int) or isinstance(max_seconds, bool) or max_seconds < 0:
            raise ValueError("max_seconds must be a non-negative integer")
        if not isinstance(dt, int) or isinstance(dt, bool) or dt <= 0:
            raise ValueError("dt must be a positive integer")
        if victory not in ("score", "capture"):
            raise ValueError("victory must be score or capture")
        if (not isinstance(move_radius, int) or isinstance(move_radius, bool)
                or move_radius < 0):
            raise ValueError("move_radius must be a non-negative integer")
        if not isinstance(deep_validate, bool):
            raise ValueError("deep_validate must be a boolean")
        self.snapshot = snapshot
        self.side = side
        self.max_seconds = max_seconds
        self.fog = fog
        self.dt = dt
        self.victory = victory
        self.move_radius = move_radius
        self.deep_validate = deep_validate
        # 奖励口径：`verified`（默认）只用已确证分项（剩余兵力分）；
        # `platform` 与 `verified` 现在数值相同（三项分值都已实测），保留两个口径是为了
        # 将来一旦出现新的未实测项时能自动把它挡在奖励之外（仅用于对账/展示时才显式传 platform）。
        self.scoring_mode = scoring_mode
        # 巡飞弹部署（上车）窗口：`"opening"`（默认，手册 4.10(3) 的“对抗开始前”）/ `"any"`
        # （未发射前任意时刻）—— 训练要用弹时显式传 `any`，不静默偏离。
        self.deploy_window = deploy_window
        self.game = None
        self.done = False
        self._history = []

    # ---------------- 生命周期 ----------------
    def reset(self, seed=None):
        self.game = Game(seed=seed, snapshot=self.snapshot,
                         deploy_window=self.deploy_window)
        self.game.start()
        self.game.second = 0
        self.done = False
        self._history = []
        self._remember()
        self._check_done()
        return self.observe()

    def _remember(self):
        """记住当前可见的敌方（形成「已知但可能过期」的情报）。"""
        for u in self.game.units.values():
            if u.color == self.side or not u.alive():
                continue
            if self.fog and not any(
                    o.alive() and o.color == self.side and
                    self.game.can_observe(o.id, u.id)
                    for o in self.game.units.values()):
                continue
            self._history.append({"t": self.game.second, "id": u.id,
                                  "pos": u.pos, "blood": u.blood})

    # ---------------- 观察 ----------------
    def _unit_view(self, u, known=False):
        d = {
            "id": u.id,
            "army": u.army or u.name,
            "name": u.name,
            "typeid": u.typeid,
            "pos": u.pos,
            "hexid": G.hexid(u.pos) if u.pos else "-",
            "blood": u.blood,
            "suppressed": 1 if u.stunned(self.game.second) else 0,
        }
        if not known:
            d.update({
                "color": u.color,
                "score": u.score,
                "armor": u.armor,
                "hidden": 1 if u.ste == 2 else 0,
                "ste": u.ste,
                "loaded_into": u.detail.get("carrier"),
                "carrying": list(u.detail.get("carried") or []),
                "engaging": u.detail.get("target"),
                "weapons": [self._weapon_view(u, wid) for wid in u.wz],
                "can_observe_range": self._observe_range(u),
            })
            # 巡飞弹生命周期显式暴露（issue #1）：状态 + 载车/发射车 + 剩余巡飞秒。
            # 车内记录 `blood=0`、无阵营 ⇒ 不会出现在 `own` 里；但载具的 `carrying`
            # 列出它的 id，所以通过 `load`/`launch` 仍是可发现的。
            if u.is_loitering_missile():
                d["missile_state"] = self.game.missile_state(u)
                if u.detail.get("loiter_until"):
                    d["loiter_seconds_left"] = max(
                        0, int(u.detail["loiter_until"] - self.game.second))
            if u.typeid == 2:
                in_flight = self.game.in_flight_missile(u.id)
                if in_flight is not None:
                    d["missile_in_flight"] = in_flight.id
        return d

    @staticmethod
    def _weapon_view(u, wid):
        w = WEAPONS.get(wid)
        if not w:
            return {"id": wid}
        fld = AMMO_FIELD.get(w[1])
        n = u.ammo.get(fld) if fld else None
        return {"id": wid, "name": w[0], "ammo_type": w[1], "range": w[2],
                "ammo": n if n is not None else -1}

    def _observe_range(self, u):
        eq = EQUIPMENT.get(u.data)
        return {"person": eq[4][0], "vehicle": eq[4][1], "air": eq[4][2]} if eq else {}

    def observe(self):
        g = self.game
        own, seen = [], []
        for u in g.units.values():
            if not u.alive():
                continue
            if u.color == self.side:
                own.append(self._unit_view(u))
            else:
                visible = (not self.fog) or any(
                    self.game.can_observe(o.id, u.id)
                    for o in g.units.values()
                    if o.alive() and o.color == self.side)
                if visible:
                    seen.append(self._unit_view(u))
        known = []
        if self.fog:
            latest = {}
            for h in self._history:
                if h["id"] in {v["id"] for v in seen}:
                    continue
                latest[h["id"]] = h
            for h in latest.values():
                known.append({"id": h["id"], "pos": h["pos"],
                              "hexid": G.hexid(h["pos"]), "blood": h["blood"],
                              "last_seen": h["t"], "stale": True})
        return {
            "time": g.second,
            "time_left": max(0, self.max_seconds - g.second),
            "side": self.side,
            "score": self.score(),
            "objectives": [{"id": c["id"], "name": c.get("name"), "pos": c["pos"],
                            "hexid": G.hexid(c["pos"]), "held_by": c.get("flag")}
                           for c in g.cities],
            "own": own,
            "enemy_visible": seen,
            "enemy_known": known,
            "done": self.done,
        }

    # ---------------- 计分 ----------------
    # 口径全部在 `engine/scoring.py`（唯一处），**三项分值都已平台实测**：
    #   剩余兵力 `Σ上阵(score×blood) + Σ车内载员(score×编制)`（对账平台 红 354 / 蓝 234）
    #   夺控      `主要 80 / 次要 50`，**一次性**（实测）
    #   歼煍      `Σ(损失兵力 × score)`，与剩余兵力分逐点互补（实测）
    # 夺控边界（失守扣分 / 重夺加分）也已实测 ⇒ `PENDING_RULES` 为空。
    # 实测证据与探针：`docs/计分规则待确认.md`；开局对账：`docs/计分对账.md`。
    # 实测的夺控点分值表（主要 80 / 次要 50，一次性）与“未知名称不加分”的入口
    CAPTURE_POINT_SCORE = scoring.CAPTURE_POINT_SCORE
    capture_point_value = staticmethod(scoring.capture_point_value)
    SCORE_RULES_PENDING = scoring.PENDING_RULES

    def score(self):
        """己方 / 敌方的分项分（口径见 `engine/scoring.py`）。

        每个阵营同时给出：

        - `total`：三个分项**全算**（平台式）—— 对账/展示
        - `verified`：**只算已实测分项**（现与 `total` 相等）—— 默认奖励口径
        - `unverified`：本次算分依赖了哪条**未实测前提 / 没说清的数据**

        `self.scoring_mode` 决定 `diff` 用哪一个（`"verified"` 默认 / `"platform"`）。
        两套口径保留的意义：一旦将来出现新的未实测项（新规则、未知夺控点名称、
        未知型号载员编制），它会自动**不进 `verified`**，而不是惄惄地进奖励。
        """
        both, issues = scoring.both_scores(self.game)
        own, opp = both[self.side], both["RED" if self.side == "BLUE" else "BLUE"]
        return {"own": own, "enemy": opp,
                "diff": (scoring.reward_value(own, self.scoring_mode)
                         - scoring.reward_value(opp, self.scoring_mode)),
                "mode": self.scoring_mode,
                "scoring_version": scoring.SCORING_VERSION,
                "pending_rules": list(scoring.PENDING_RULES),
                "unverified": sorted(set(own["unverified"]) | set(opp["unverified"])),
                "issues": dict(issues)}

    # ---------------- 动作空间 ----------------
    # 不需要引擎裁决、可以纯只读判定的动作类型（它们不会改变任何状态或掷骰）
    READONLY_TYPES = ("move", "wait", "stop", "stop_attack")
    # 会改变状态、但“此刻能不能做”只由**当场可读**的条件决定（无需试一次、也无需掷骰）。
    # 为什么单列：事务校验会为**每一个候选项**复制整个 `Game`，真实地图上实测
    # ~3.4 ms/次；实时接口（1 Hz、每秒 2 次枚举）下 hide/load 就有 21 个候选，
    # 占满整次枚举 78% 的耗时。这里的判定逐条照抄 `_apply` 的条件，
    # `_apply` 仍是权威（非法项会被它拒绝）。
    CHEAP_TYPES = ("hide", "load", "grenade")

    def legal_actions(self, unit_id=None, move_radius=None, deep_validate=None):
        """列出当前合法动作。

        `move_radius=None` 时用 `self.move_radius`（默认 6）。半径直接决定机动候选
        数量：半径 r 的六角盘有 1+3r(r+1) 格，真实地图（2721 格、6 个己方单位）
        半径 6 会产出 1500+ 个候选。

        `deep_validate`（None 时用构造参数，默认 False）：
        - `True`：每个候选都过一遍 `_try_actions(commit=False)` 的事务校验（对候选做
          一次**整状态深拷贝**）。语义最保险，但真实地图实测 **1.8 秒/次枚举**（半径 6）。
        - `False`：`READONLY_TYPES` 走只读判定（`_cheap_ok`），其余需要引擎裁决的动作
          （射击/集火/手雷/夺控/装卸/掩蔽）**仍走事务校验**。实测 **0.042 秒/次**（半径 6，×43）。
          机动只需补一条手册规则：冲锋单位（`ste==4`）一个动作只能走 1 格 ——
          手册 4.1.1(4) 每 200 米转疲劳，而 1 格正好是 200 米，故第二格必被拒绝。
        两种口径的等价性由 `tests/test_enumeration_equivalence.py`
        与 `tools/hex_env_diff.py`（可在真实地图上跑）守住。
        """
        g = self.game
        if self.done:
            return []
        if deep_validate is None:
            deep_validate = self.deep_validate
        radius = self.move_radius if move_radius is None else move_radius
        acts = []
        units = [u for u in g.units.values()
                 if u.alive() and u.color == self.side
                 and not u.detail.get("carrier")      # 已上车的不再独立行动
                 and (unit_id is None or u.id == unit_id)]
        # ★ 堆叠计数快照：枚举期间只读、不会位移，所以一次成型交给 `step_seconds`
        #   （原来每次 step_seconds 都现场扫 36 个单位，而一次列举要调 600+ 次）。
        stack = g.stack_index()
        for u in units:
            if u.stunned(g.second):
                continue
            # 机动：邻格逐步扩展（BFS，受通行与堆叠约束）
            fronts = [(u.pos, [u.pos], 0)]
            seenpos = {u.pos}
            for _ in range(radius):
                nxt = []
                for cur, path, elapsed in fronts:
                    for nb in G.neighbors(cur):
                        if nb in seenpos:
                            continue
                        sec, _why = g.step_seconds(u, cur, nb, stack=stack)
                        if sec <= 0:
                            continue
                        seenpos.add(nb)
                        p2 = path + [nb]
                        nxt.append((nb, p2, elapsed + sec))
                        acts.append({"type": "move", "unit": u.id,
                                     "path": p2, "seconds": math.ceil(elapsed + sec)})
                fronts = nxt
                if not fronts:
                    break
            # 射击：射程内 + 通视 + 可观察
            for t in g.units.values():
                if not t.alive() or t.color == u.color:
                    continue
                d = G.distance(u.pos, t.pos)
                picked = None
                # ★ 通视/可观察只取决于 **(u, t) 这一对**，与武器无关 —— 原来这句放在
                #   `for wid in u.wz` 里，等价于同一对单位重复算 2–4 遍。
                #   改成“第一次真的需要时才算、之后复用”，结果完全一致（`can_observe` 无副作用）。
                visible = None
                for wid in u.wz:
                    w = WEAPONS.get(wid)
                    if not w or d > w[2]:
                        continue
                    if visible is None:
                        visible = g.can_observe(u.id, t.id)
                    if not visible:
                        continue
                    acts.append({"type": "attack", "unit": u.id, "target": t.id,
                                 "weapon": wid, "dist": d, "weapon_name": w[0],
                                 "track": 0})
                    if picked is None:
                        picked = (wid, w, d)
                # 手册 3.14：设定目标后持续打击至目标被消灭（官方默认语义）
                if picked:
                    wid, w, d = picked
                    acts.append({"type": "attack", "unit": u.id, "target": t.id,
                                 "weapon": wid, "dist": d, "weapon_name": w[0],
                                 "track": 1})
            # 解除持续打击
            if u.detail.get("target"):
                acts.append({"type": "stop_attack", "unit": u.id})
            # 夺控
            for c in g.cities:
                if c["pos"] == u.pos and c.get("flag") != self.side:
                    acts.append({"type": "capture", "unit": u.id})
            # 隐蔽切换（手册 4.5(2)：打击间隔期间不允许掩蔽）
            if not self._in_attack_gap(u):
                acts.append({"type": "hide", "unit": u.id,
                             "on": 0 if u.ste == 2 else 1})
            # 上车（手册 3.3）：人员与车辆同格；巡飞弹同一条路径（平台走同一个
            # `com/ObjInto.asp`，手册 4.10(3) 的“对抗开始前上车”由 `_load_gate` 把关）。
            if u.typeid == 1 or u.is_loitering_missile():
                for c in g.units.values():
                    if (c.alive() and c.typeid == 2 and c.color == u.color
                            and c.pos == u.pos and not c.stunned(g.second)
                            and not c.detail.get("carrier")
                            and self._seat_left(c) > 0):
                        acts.append({"type": "load", "unit": u.id, "carrier": c.id})
            # 发射巡飞弹（手册 4.10(3)(4)）：由**载车**下令，弹必须在该车车内，
            # 且该车当前没有在飞弹。真正的合法性由 `_try_actions` 事务校验裁决。
            if u.typeid == 2:
                for mid in sorted(u.detail.get("carried") or [], key=str):
                    m = g.units.get(mid)
                    if m is not None and m.is_loitering_missile():
                        acts.append({"type": "launch", "unit": u.id,
                                     "missile": m.id})
            # 下车（手册 3.3）：由车辆下达。★ **巡飞弹不参与下车** —— 客户端「全员下车」
            # 只发 `off=1/2`（人员/无人车），`off=3` 走的是发射命令（见 `Unit.can_disembark`）。
            if u.typeid == 2 and self._disembarkable(g, u):
                acts.append({"type": "unload", "unit": u.id})
            # 手雷（手册 3.15 / 4.7）：投掷范围通常为相邻格
            if u.typeid == 1 and 144 in u.wz and u.ammo.get("a5", 0) > 0:
                for nb in [u.pos] + list(G.neighbors(u.pos)):
                    if g.hex(nb):
                        acts.append({"type": "grenade", "unit": u.id,
                                     "target_hex": nb})
            # 停止机动（清空 pos2）
            if u.pos2 and u.pos2 != u.pos:
                acts.append({"type": "stop", "unit": u.id})
        acts.append({"type": "wait", "seconds": self.dt})

        # 手册 3.16 集火射击：对**六角格坐标**集火，不跟踪目标。
        # 每个“有可见敌人的格子”一个动作，参与单位 = 能打到该格的己方单位。
        own_alive = [u for u in g.units.values()
                     if u.alive() and u.color == self.side
                     and not u.detail.get("carrier") and not u.stunned(g.second)
                     and (unit_id is None or u.id == unit_id)]
        hexes = {t.pos for t in g.units.values()
                 if t.alive() and t.color != self.side and t.pos}
        # ★ 每个格子只取**第一个**敌方单位做代表（原实现是在 (格 × 己方单位) 的
        #   二重循环里跑一遍生成器，等价于每对重复扫描全部单位）。
        #   取第一个 = 与原来 `next(...)` 完全同一个对象（按 `units` 插入顺序）。
        representative = {}
        for t in g.units.values():
            if t.alive() and t.color != self.side and t.pos in hexes:
                representative.setdefault(t.pos, t.id)
        for hx in sorted(hexes):
            shooters = []
            probe = representative.get(hx)
            for u in own_alive:
                d = G.distance(u.pos, hx)
                if any((WEAPONS.get(w) or (None, None, 0))[2] >= d for w in u.wz):
                    if probe is None or g.can_observe(u.id, probe):
                        shooters.append(u.id)
            if shooters:
                acts.append({"type": "area_fire", "unit": shooters[0],
                             "units": shooters, "target_hex": hx})
        if deep_validate:
            # 参考口径：每个候选都过一遍事务校验（慢，仅供差分/回归用）
            return [a for a in acts if self._try_actions([a], commit=False)[0]]
        return self._fast_actions(acts)

    def _cheap_ok(self, act):
        """只读 / 廉价判定：只允许用于 `READONLY_TYPES` 与 `CHEAP_TYPES`。

        - `wait` / `stop` / `stop_attack`：事务校验不会拒。注意 `wait` **故意豁免**
          剩余时间上限——`_try_actions` 的时间检查只在非 wait 分支里，超时的等待
          会被截断到剩余秒（这样时钟才能正好走到 `max_seconds`）。
        - `move`：BFS 已经按 `step_seconds > 0` 逐跳筛过（通行/坡度/压制/堆叠），
          唯一漏掉的是**冲锋疲劳**（见 `legal_actions` 的说明）。
        - `hide` / `load`：会改变状态，但合法性只依赖当场可读的条件，见 `_cheap_ok_stateful`。
        """
        try:
            self._validate(act)
        except (ValueError, TypeError, KeyError, OverflowError):
            return False
        g = self.game
        kind = act["type"]
        if kind in self.CHEAP_TYPES:
            return self._cheap_ok_stateful(act)
        if kind not in self.READONLY_TYPES:
            raise ValueError("cheap check is only valid for readonly action types")
        if kind in ("wait", "stop", "stop_attack"):
            return True
        u = g.units.get(act["unit"])
        if (u is not None and u.ste == 4 and not u.fatigued(g.second)
                and len(act.get("path") or []) > 2):
            return False
        return math.ceil(act.get("seconds", self.dt)) <= self.max_seconds - g.second

    def _cheap_ok_stateful(self, act):
        """`hide` / `load` / `grenade` 的廉价判定（逐条对应 `_apply` 里的条件）。"""
        g = self.game
        kind = act["type"]
        u = g.units.get(act["unit"])
        if u is None:
            return False
        cost = {"hide": HIDE_SEC, "load": LOAD_SEC, "grenade": GRENADE_SEC}[kind]
        if math.ceil(cost) > self.max_seconds - g.second:
            return False
        if kind == "hide":
            return not self._in_attack_gap(u)
        if kind == "grenade":
            # `_validate` 已保证 u 为己方存活且未上车/未被压制，target_hex 是存在的格
            return (144 in u.wz and u.ammo.get("a5", 0) > 0
                    and G.distance(u.pos, int(act["target_hex"])) <= 1)
        car = g.units.get(act.get("carrier"))
        return self._load_gate(u, car)[0]

    def _fast_actions(self, acts):
        """快速枚举过滤：只读类型与 hide/load 走 `_cheap_ok`，其余仍走事务校验。"""
        cheap = set(self.READONLY_TYPES) | set(self.CHEAP_TYPES)
        out = []
        for a in acts:
            if a["type"] in cheap:
                if self._cheap_ok(a):
                    out.append(a)
            elif self._try_actions([a], commit=False)[0]:
                out.append(a)
        return out

    # ---------------- 执行 ----------------
    def _validate(self, act):
        """Reject malformed or unauthorized inputs before engine mutation."""
        if not isinstance(act, dict):
            raise ValueError("action must be an object")
        kind = act.get("type")
        if kind == "wait":
            seconds = act.get("seconds", self.dt)
            if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds <= 0:
                raise ValueError("wait seconds must be a positive integer")
            return
        if kind not in {"move", "attack", "stop_attack", "area_fire", "capture",
                        "hide", "load", "unload", "grenade", "stop", "launch"}:
            raise ValueError("unknown action type")
        g = self.game
        u = g.units.get(act.get("unit"))
        if not u or not u.alive() or u.color != self.side:
            raise ValueError("unit must be a living own unit")
        if u.detail.get("carrier") or u.stunned(g.second):
            raise ValueError("unit cannot act in its current state")
        if kind == "move":
            path = act.get("path")
            if not isinstance(path, (list, tuple)) or not path:
                raise ValueError("move path must be nonempty")
            cur = u.pos
            for dest in path[1:] if path[0] == cur else path:
                if not isinstance(dest, int) or isinstance(dest, bool) or dest not in G.neighbors(cur):
                    raise ValueError("move path must contain adjacent integer hexes")
                cur = dest
            if cur == u.pos and len(path) == 1:
                raise ValueError("move path must change position")
        if kind == "attack":
            target = g.units.get(act.get("target"))
            if not target or not target.alive() or target.color == self.side or target.detail.get("carrier"):
                raise ValueError("target must be a living, unloaded enemy")
            if act.get("weapon") not in (None, 0) and act.get("weapon") not in u.wz:
                raise ValueError("weapon is not equipped")
        if kind == "hide" and act.get("on") not in (0, 1, False, True):
            raise ValueError("hide on must be 0 or 1")
        if kind == "load":
            car = g.units.get(act.get("carrier"))
            if not car or car.id == u.id:
                raise ValueError("load requires another unit as carrier")
            if u.is_loitering_missile():
                # 手册 4.10(3)：巡飞弹上车 = 部署，载体必须是己方**车辆**。
                if car.typeid != 2 or car.color != self.side:
                    raise ValueError("a loitering missile must deploy onto an own vehicle")
            elif (u.typeid != 1 or u.detail.get("carrier") or car.color != self.side
                  or car.typeid != 2 or car.detail.get("carrier")):
                raise ValueError("load requires own infantry and own vehicle")
        if kind == "launch":
            # 手册 4.10：发射由**载车**下令（`unit` = 载车，`missile` = 弹）。
            m = g.units.get(act.get("missile"))
            if u.typeid != 2 or m is None or m.id == u.id:
                raise ValueError("launch requires a vehicle and a missile to launch")
        if kind == "unload" and (u.typeid != 2 or not self._disembarkable(g, u)):
            raise ValueError("unload requires a loaded vehicle with disembarkable units")
        if kind in ("area_fire", "grenade"):
            hx = act.get("target_hex")
            if not isinstance(hx, int) or isinstance(hx, bool) or not g.hex(hx):
                raise ValueError("target_hex must identify an existing hex")
        if kind == "area_fire":
            ids = act.get("units", [u.id])
            if not isinstance(ids, (list, tuple)) or not ids or len(set(ids)) != len(ids):
                raise ValueError("area_fire units must be nonempty and unique")
            for uid in ids:
                shooter = g.units.get(uid)
                if (not shooter or not shooter.alive() or shooter.color != self.side or
                        shooter.stunned(g.second) or shooter.detail.get("carrier")):
                    raise ValueError("area_fire requires available own units")

    def _try_actions(self, actions, commit):
        """Execute on an isolated state; invalid batches are atomic no-ops.

        The immutable map is shared to avoid copying it for each candidate. All
        mutable game state, including the game's random stream, is copied.
        """
        original = self.game
        self.game = copy.deepcopy(original, {id(original.hexes): original.hexes})
        messages, costs = [], []
        commanded = set()
        accepted = False
        try:
            for act in actions:
                self._validate(act)
                if act["type"] != "wait":
                    participants = set(act.get("units", [act["unit"]])) if act["type"] == "area_fire" else {act["unit"]}
                    if commanded & participants:
                        raise ValueError("a batch cannot command the same unit twice")
                    commanded.update(participants)
                if act["type"] == "wait":
                    cost = act.get("seconds", self.dt)
                    msg = "等待"
                else:
                    _, _, _, info = self._apply(act)
                    cost, msg = info.get("seconds", 0), info.get("msg", "")
                    if cost <= 0 and act["type"] not in ("stop", "stop_attack"):
                        raise ValueError(msg or "action failed")
                    if math.ceil(cost) > self.max_seconds - original.second:
                        raise ValueError("action exceeds remaining episode time")
                costs.append(cost)
                messages.append(msg)
            accepted = True
            return True, {"msg": "; ".join(messages), "seconds": min(
                math.ceil(max(costs, default=self.dt)), self.max_seconds - original.second)}
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return False, {"msg": str(exc), "seconds": 0, "invalid_action": True}
        finally:
            if accepted and commit:
                # Preserve public Game/Unit identities used by callers and bridges.
                for uid, updated in self.game.units.items():
                    current = original.units.get(uid)
                    if current is not None:
                        for slot in current.__slots__:
                            setattr(current, slot, getattr(updated, slot))
                        self.game.units[uid] = current
                original.__dict__.clear()
                original.__dict__.update(self.game.__dict__)
            self.game = original

    def step(self, action):
        """Apply one action or an atomic batch and return the score-difference delta.

        Invalid inputs return zero reward without advancing time or changing state.
        Batches execute in order and advance time by their maximum action duration.
        """
        if self.done:
            return self.observe(), 0.0, True, {"msg": "已结束", "seconds": 0}
        before = self.score()["diff"]
        actions = list(action) if isinstance(action, (list, tuple)) else [action]
        if not actions:
            actions = [{"type": "wait"}]
        accepted, info = self._try_actions(actions, commit=True)
        if not accepted:
            return self.observe(), 0.0, self.done, info
        self.game.advance(info["seconds"])
        self._remember()
        self._check_done()
        return self.observe(), float(self.score()["diff"] - before), self.done, info

    @staticmethod
    def _disembarkable(g, carrier):
        """该载车上**可以下车**的载员（人员 / 无人车）。

        巡飞弹与其它空中弹不在此列：它们只能通过 `launch` 离开载车
        （客户端「全员下车」也只发 `off=1/2`，`off=3` 是发射命令）。
        """
        out = []
        for pid in carrier.detail.get("carried") or []:
            p = g.units.get(pid)
            if p is not None and p.can_disembark():
                out.append(p)
        return out

    def _load_gate(self, u, car):
        """`load` 的合法性判定（`_apply` 与 `_cheap_ok_stateful` **共用同一条口径**）。

        返回 `(ok, why)`。**纯读取**，不改状态 —— `_cheap_ok_stateful`（枚举用）依赖这一点。
        巡飞弹的部署规则**不在环境里重复实现**，而是调 `engine` 的纯判定
        （`Game.missile_deploy_block`），避免两份判定漂移。
        """
        g = self.game
        if car is None or car.id == u.id:
            return False, "载具不存在"
        if u.is_loitering_missile():
            # 手册 4.10(3)：巡飞弹只能上车（部署）；具体状态机在引擎。
            why = g.missile_deploy_block(car.id, u.id)
            if why:
                return False, why
        else:
            if u.typeid != 1:
                return False, "只有人员与巡飞弹可以上车"
            if car.typeid != 2:
                return False, "载具必须是车辆"
        if car.color != u.color or car.color != self.side:
            return False, "载具必须是己方单位"
        if not car.alive():
            return False, "载具已被消灭"
        if car.pos != u.pos:
            return False, "人与车必须同格（手册 3.3）"
        if u.stunned(g.second) or car.stunned(g.second):
            return False, "被压制不能上/下车（手册 4.6.3(1)）"
        if self._seat_left(car) <= 0:
            return False, "载员已满"
        return True, ""

    def _apply(self, act):
        g = self.game
        t = act.get("type")
        u = g.units.get(act.get("unit"))
        if u is None and t != "wait":
            return None, 0.0, False, {"msg": "单位不存在", "seconds": 0}
        if t == "move":
            if u.detail.get("carrier"):
                return None, 0.0, False, {"msg": "已上车，随载具机动（手册 3.3）",
                                          "seconds": 0}
            path = list(act.get("path") or [])
            # 允许 path 带上起点（legal_actions 给出的就是含起点的完整路径），
            # 起点自身不产生机动耗时。
            if path and path[0] == u.pos:
                path = path[1:]
            total = 0
            cur = u.pos
            for step_to in path:
                sec, why = g.commit_step(u, cur, step_to)
                if sec <= 0:
                    return None, 0.0, False, {"msg": "机动受阻：%s" % why, "seconds": 0}
                u.pos2 = step_to
                u.pos = step_to
                u.rot = self._rot(cur, step_to)
                total += sec
                cur = step_to
            # 手册 3.3：已完成上车的步兵随车辆机动
            # ★ 不能用 `alive()` 过滤 —— 车内乘员 `blood=0`（平台实测就是这样），
            #   那只表示“车内读不到兵力”，不是阵亡。
            for pid in list(u.detail.get("carried") or []):
                p = g.units.get(pid)
                if p is not None:
                    p.pos = u.pos
                    p.pos2 = u.pos
            g.apply_city_capture(u)
            g.dirty = True
            return None, 0.0, False, {"msg": "机动 %d 格，耗时 %ds" % (len(path), total),
                                      "seconds": total}
        if t == "attack":
            if act.get("track"):
                tid, err2 = g.set_engagement(u.id, act.get("target"),
                                             weapon_id=act.get("weapon"))
                if err2:
                    return None, 0.0, False, {"msg": "设定目标失败：%s" % err2,
                                              "seconds": 0}
                r, err = g.attack(u.id, act.get("target"),
                                  weapon_id=int(act.get("weapon") or 0))
                if err:
                    return None, 0.0, False, {"msg": "射击失败：%s" % err, "seconds": 0}
                u.detail["attacked_at"] = g.second
                note = "" if err else "，首发 %s" % (r["result"],)
                return None, 0.0, False, {
                    "msg": "设定持续打击目标（手册 3.14）%s" % note,
                    "seconds": ATTACK_SEC}
            r, err = g.attack(u.id, act.get("target"), weapon_id=int(act.get("weapon") or 0))
            if err:
                return None, 0.0, False, {"msg": "射击失败：%s" % err, "seconds": 0}
            u.detail["attacked_at"] = g.second
            return None, 0.0, False, {"msg": "射击 %s" % (r["result"],), "seconds": ATTACK_SEC}
        if t == "stop_attack":
            old, _err = g.clear_engagement(u.id)
            return None, 0.0, False, {
                "msg": "已解除持续打击" if old else "本来没有打击目标",
                "seconds": 0}
        if t == "area_fire":
            unit_ids = list(act.get("units") or [u.id])
            res, err = g.area_fire(unit_ids, int(act.get("target_hex") or 0),
                                   weapon_id=act.get("weapon"))
            if err:
                return None, 0.0, False, {"msg": "集火失败：%s" % err, "seconds": 0}
            if res["result"] != "MISS" and not any(not shot.get("error") for shot in res["shots"]):
                return None, 0.0, False, {"msg": "集火没有可执行的射击", "seconds": 0}
            for sid in unit_ids:
                su = g.units.get(sid)
                if su:
                    su.detail["attacked_at"] = g.second
            if res["result"] == "MISS":
                return None, 0.0, False, {
                    "msg": "集火射击失败：目标已离开该格（手册 3.16）",
                    "seconds": AREA_FIRE_SEC}
            return None, 0.0, False, {
                "msg": "集火射击 %d 次" % len(res["shots"]), "seconds": AREA_FIRE_SEC}
        if t == "capture":
            c = g.apply_city_capture(u)
            return None, 0.0, False, {
                "msg": "夺控成功" if c else "夺控条件不满足",
                "seconds": CAPTURE_SEC if c else 0}
        if t == "hide":
            if self._in_attack_gap(u):
                return None, 0.0, False, {
                    "msg": "打击间隔期间不允许掩蔽（手册 4.5(2)）", "seconds": 0}
            u.ste = 2 if act.get("on") else 0
            return None, 0.0, False, {"msg": "掩蔽切换", "seconds": HIDE_SEC}
        if t == "load":
            car = g.units.get(act.get("carrier"))
            ok, why = self._load_gate(u, car)
            if not ok:
                return None, 0.0, False, {"msg": why, "seconds": 0}
            if u.is_loitering_missile():
                # 上车 = **部署**（手册 4.10(3)），状态转换由引擎独占实现。
                ok, why = g.deploy_missile(car.id, u.id)
                if not ok:
                    return None, 0.0, False, {"msg": why, "seconds": 0}
                return None, 0.0, False, {"msg": "巡飞弹部署上车", "seconds": LOAD_SEC}
            # ★ 平台实测（`docs/运输机制实测.md` + `tools/hex_load_kill_probe.py`）：
            #   上车后该单位变成 `blood=0 / color=''`（车内读不到兵力与阵营），**编制留在 establishment**，
            #   平台按编制把它计入剩余兵力分；下车时**新建实体、blood = 编制、color = 载具那方**。
            #   所以这里也要清 color：否则 `scoring.breakdown` 会把它当成“上阵单位”按 0 血计分。
            establishment = (scoring.carried_establishment(u.data) or u.establishment or u.blood)
            if scoring.carried_establishment(u.data) is None:
                u.detail["establishment_unverified"] = True
            if u.blood != establishment:
                # 带伤上车：平台模型是“车内不记伤”，下车按编制回满。
                # 这一条**没有直接实测**，所以显式报出（`aboard_damage_forgotten`）。
                u.detail["aboard_damage_forgotten"] = True
            u.establishment = establishment
            u.blood = 0
            u.color = ""
            u.detail["carrier"] = car.id
            car.detail.setdefault("carried", []).append(u.id)
            return None, 0.0, False, {"msg": "上车完成", "seconds": LOAD_SEC}
        if t == "unload":
            if u.stunned(g.second):
                return None, 0.0, False, {
                    "msg": "车辆被压制时人员不能下车（手册 4.6.3(1)(4)）", "seconds": 0}
            passengers = self._disembarkable(g, u)
            if not passengers:
                # 车上只有弹：**没有**“把弹取回”这个动作（弹只能发射）。
                return None, 0.0, False, {
                    "msg": "车上没有可下车的人员（巡飞弹只能发射，不能下车）",
                    "seconds": 0}
            out = []
            for p in passengers:
                # 注意：**不能**用 `alive()` 过滤 —— 快照里的载员 blood=0 表示
                # "在车内读不到兵力"，不是"已阵亡"。
                # 下车后**恢复独立身份**：平台下车会新建实体并带上阵营色与编制兵力
                # （见 `docs/运输机制实测.md`）；不恢复的话，这个单位会一直“像载员”
                # ——没有阵营就不进剩余兵力分，分值直接凭空消失。
                p.color = u.color
                establishment = scoring.carried_establishment(p.data) or p.establishment or p.blood
                if scoring.carried_establishment(p.data) is None:
                    # 编制未知：不臆造，显式标记并让 `scoring` 报出（不静默算分）。
                    p.detail["establishment_unverified"] = True
                # ★ 平台实测：下车 = 新建实体，其 `blood` 就是**编制**（跟车内期间/之前的伤无关）。
                p.blood = establishment
                p.establishment = establishment
                p.units = max(1, p.blood) if p.typeid in (1, 2) else 1
                p.pos = u.pos
                p.pos2 = u.pos
                p.detail.pop("carrier", None)
                out.append(p.id)
            # 弹留在车上（它们只能通过 `launch` 离开载车）—— 不要把 carried 整个清空。
            u.detail["carried"] = [x for x in (u.detail.get("carried") or [])
                                   if x not in out]
            return None, 0.0, False, {"msg": "下车 %d 个" % len(out), "seconds": LOAD_SEC}
        if t == "launch":
            # 手册 4.10(3)(4)：由载车发射；状态机（在飞占用/一车一枚/超时）在引擎里。
            ok, why = g.launch_missile(u.id, act.get("missile"))
            if not ok:
                return None, 0.0, False, {"msg": "发射失败：%s" % why, "seconds": 0}
            return None, 0.0, False, {
                "msg": "巡飞弹已发射，巡飞上限 %s 秒" % g.loiter_sec,
                "seconds": LAUNCH_SEC}
        if t == "grenade":
            res, err = g.grenade(u.id, int(act.get("target_hex") or u.pos))
            if err:
                return None, 0.0, False, {"msg": "手雷失败：%s" % err, "seconds": 0}
            return None, 0.0, False, {
                "msg": "手雷裁决 %d 个目标" % len(res), "seconds": GRENADE_SEC}
        if t == "stop":
            u.pos2 = u.pos
            return None, 0.0, False, {"msg": "停止", "seconds": 0}
        return None, 0.0, False, {"msg": "未知动作", "seconds": 0}

    # ---------------- 规则辅助 ----------------
    def _in_attack_gap(self, u):
        """手册 4.5(2)：棋子在打击间隔期间不允许掩蔽。"""
        return (self.game.second - u.detail.get("attacked_at", -10 ** 9)) < ATTACK_SEC

    @staticmethod
    def _seat_left(car):
        """车辆剩余载员位（手册未给载员数，按 VEHICLE_SEATS 假定）。"""
        return VEHICLE_SEATS - len(car.detail.get("carried") or [])

    @staticmethod
    def _rot(a, b):
        ax, ay = G.pos_of(a)
        bx, by = G.pos_of(b)
        import math
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return 0
        return int(round(math.degrees(math.atan2(dy, dx)) / 60.0) * 60) % 360

    def _check_done(self):
        g = self.game
        if g.second >= self.max_seconds:
            self.done = True
        if not any(u.alive() and u.color == self.side for u in g.units.values()):
            self.done = True
        other = "RED" if self.side == "BLUE" else "BLUE"
        if not any(u.alive() and u.color == other for u in g.units.values()):
            self.done = True
        if self.victory == "capture" and g.cities and \
                all(c.get("flag") == self.side for c in g.cities):
            self.done = True

    # ---------------- 文本渲染（给 LLM / 调试用） ----------------
    def render(self, width=48, height=16):
        g = self.game
        if not g.hexes:
            return "(空地图)"
        rows = sorted({G.split_mapid(m)[0] for m in g.hexes})
        cols = sorted({G.split_mapid(m)[1] for m in g.hexes})
        r0, r1 = rows[0], min(rows[0] + height - 1, rows[-1])
        c0, c1 = cols[0], min(cols[0] + width - 1, cols[-1])
        grid = {}
        for u in g.units.values():
            if u.alive() and u.pos:
                grid[u.pos] = ("A" if u.color == self.side else "E") + str(u.blood)
        for c in g.cities:
            grid.setdefault(c["pos"], "**")
        out = ["    " + "".join("%2d" % (c % 100) for c in range(c0, c1 + 1))]
        for r in range(r0, r1 + 1):
            line = "%3d " % r
            for c in range(c0, c1 + 1):
                mid = G.make_mapid(r, c)
                h = g.hex(mid)
                if mid in grid:
                    line += "%2s" % grid[mid]
                elif h is None:
                    line += " ."
                else:
                    cond = int(h.get("cond") or 0)
                    line += {0: "  ", 6: "~~", 7: "##", 8: "TT", 9: "TT"}.get(cond, "  ")
            out.append(line)
        out.append("")
        out.append("图例: A#=己方算子 E#=敌方算子 **=夺控点 ~~=过河 ##=居民地 TT=丛林地 .=界外")
        out.append("时间 %ds / 剩余 %ds" % (g.second, self.max_seconds - g.second))
        return "\n".join(out)

    def state_json(self):
        return json.dumps(self.observe(), ensure_ascii=False, indent=1)
